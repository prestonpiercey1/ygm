// Copyright 2019-2026 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <mpi.h>

namespace ygm::detail {

/**
 * @brief Shared memory data structure for YGM performance monitoring.
 *
 * @details Shared with YGM-top reader. Uses fixed-width types only 
 * (uint64_t, uint32_t, double) to avoid 32/64-bit ABI issues between
 *  YGM programs and ygm-top reader.
 */
struct monitor_data {
  // Identity
  uint32_t rank;
  uint32_t comm_size;

  // Communication counters (mirrors comm_stats)
  uint64_t async_count;
  uint64_t barrier_count;
  uint64_t rpc_count;
  uint64_t route_count;
  uint64_t isend_count;
  uint64_t isend_bytes;
  uint64_t isend_test_count;
  uint64_t irecv_count;
  uint64_t irecv_bytes;
  uint64_t irecv_test_count;
  uint64_t iallreduce_count;
  uint64_t waitsome_isend_irecv_count;
  uint64_t waitsome_iallreduce_count;

  // Timing
  double waitsome_isend_irecv_time;
  double waitsome_iallreduce_time;
  double time_start;
  double last_barrier_duration;

  // Memory (from /proc/self/status)
  uint64_t rss_bytes;
  uint64_t vm_size_bytes;

  // Buffer utilization
  uint64_t pending_isend_bytes;
  uint64_t send_local_buffer_bytes;
  uint64_t send_remote_buffer_bytes;
  uint64_t send_queue_depth;
  uint64_t recv_queue_depth;

  // Heartbeat / versioning
  uint64_t update_sequence;
};

// Forward declaration
class comm_stats;

/**
 * @brief YGM performance metric monitor writing to POSIX shared memory.
 *
 * @details Creates per-rank shared memory segments that can be read by an
 * external monitoring tool (ygm-top). The monitor is enabled via the
 * YGM_MONITOR environment variable and uses Boost UUID for job isolation
 * on shared HPC nodes.
 *
 * Lifecycle:
 * - setup() is called from comm_setup() after UUID broadcast
 * - sync() is called periodically (e.g., at barrier, counter rollover)
 * - Destructor handles cleanup (munmap, shm_unlink)
 */
class comm_monitor {
 public:
  comm_monitor() = default;

  ~comm_monitor() { cleanup(); }

  // Non-copyable
  comm_monitor(const comm_monitor&)            = delete;
  comm_monitor& operator=(const comm_monitor&) = delete;

  /**
   * @brief Initialize shared memory segments for monitoring.
   *
   * @param rank Global MPI rank
   * @param comm_size Total number of MPI ranks
   * @param uuid Unique identifier for this YGM session
   * @param local_id Local rank ID on this node (0 = lowest local rank)
   * @param local_ranks Vector of global ranks on this node
   */
  void setup(int rank, int comm_size, const std::string& uuid, int local_id,
             const std::vector<int>& local_ranks) {
#ifdef __linux__
    m_rank     = rank;
    m_local_id = local_id;
    m_uuid     = uuid;
    m_shm_name = "/ygm_" + uuid + "_rank" + std::to_string(rank);

    // Create per-rank shared memory segment
    m_shm_fd = shm_open(m_shm_name.c_str(), O_CREAT | O_RDWR, 0600);
    if (m_shm_fd == -1) {
      std::cerr << "Warning: shm_open failed for " << m_shm_name << ": "
                << strerror(errno) << std::endl;
      return;
    }

    if (ftruncate(m_shm_fd, sizeof(monitor_data)) == -1) {
      std::cerr << "Warning: ftruncate failed for " << m_shm_name << ": "
                << strerror(errno) << std::endl;
      close(m_shm_fd);
      shm_unlink(m_shm_name.c_str());
      m_shm_fd = -1;
      return;
    }

    m_data = static_cast<monitor_data*>(mmap(nullptr, sizeof(monitor_data),
                                             PROT_READ | PROT_WRITE, MAP_SHARED,
                                             m_shm_fd, 0));

    if (m_data == MAP_FAILED) {
      std::cerr << "Warning: mmap failed for " << m_shm_name << ": "
                << strerror(errno) << std::endl;
      close(m_shm_fd);
      shm_unlink(m_shm_name.c_str());
      m_shm_fd = -1;
      m_data   = nullptr;
      return;
    }

    // Initialize monitor_data
    std::memset(m_data, 0, sizeof(monitor_data));
    m_data->rank       = static_cast<uint32_t>(rank);
    m_data->comm_size  = static_cast<uint32_t>(comm_size);
    m_data->time_start = MPI_Wtime();

    // Lowest local rank writes manifest file for discovery
    if (local_id == 0) {
      m_owns_manifest              = true;
      std::string manifest_path    = "/dev/shm/ygm_" + uuid + "_manifest";
      std::ofstream manifest_file(manifest_path);
      if (manifest_file.is_open()) {
        manifest_file << uuid << "\n";
        manifest_file << local_ranks.size() << "\n";
        for (int r : local_ranks) {
          manifest_file << r << "\n";
        }
        manifest_file.close();
      } else {
        std::cerr << "Warning: Could not create manifest file " << manifest_path
                  << std::endl;
      }
    }

    m_enabled = true;
#else
    (void)rank;
    (void)comm_size;
    (void)uuid;
    (void)local_id;
    (void)local_ranks;
    std::cerr << "Warning: YGM monitor only supported on Linux" << std::endl;
#endif
  }

  /**
   * @brief Sync current statistics to shared memory.
   *
   * @param stats Reference to comm_stats object
   * @param pending_isend_bytes Bytes in pending MPI_Isend requests
   * @param send_local_buffer_bytes Bytes in local send buffers
   * @param send_remote_buffer_bytes Bytes in remote send buffers
   * @param send_queue_size Number of outstanding MPI_Isend requests
   * @param recv_queue_size Number of posted MPI_Irecv requests
   */
  void sync(const comm_stats& stats, uint64_t pending_isend_bytes,
            uint64_t send_local_buffer_bytes, uint64_t send_remote_buffer_bytes,
            size_t send_queue_size, size_t recv_queue_size) {
#ifdef __linux__
    if (!m_enabled || m_data == nullptr) return;

    // Copy communication counters
    m_data->async_count     = stats.get_async_count();
    m_data->barrier_count   = stats.get_barrier_count();
    m_data->rpc_count       = stats.get_rpc_count();
    m_data->route_count     = stats.get_route_count();
    m_data->isend_count     = stats.get_isend_count();
    m_data->isend_bytes     = stats.get_isend_bytes();
    m_data->isend_test_count = stats.get_isend_test_count();
    m_data->irecv_count     = stats.get_irecv_count();
    m_data->irecv_bytes     = stats.get_irecv_bytes();
    m_data->irecv_test_count = stats.get_irecv_test_count();
    m_data->iallreduce_count = stats.get_iallreduce_count();
    m_data->waitsome_isend_irecv_count = stats.get_waitsome_isend_irecv_count();
    m_data->waitsome_iallreduce_count  = stats.get_waitsome_iallreduce_count();

    // Copy timing metrics
    m_data->waitsome_isend_irecv_time = stats.get_waitsome_isend_irecv_time();
    m_data->waitsome_iallreduce_time  = stats.get_waitsome_iallreduce_time();

    // Buffer utilization
    m_data->pending_isend_bytes      = pending_isend_bytes;
    m_data->send_local_buffer_bytes  = send_local_buffer_bytes;
    m_data->send_remote_buffer_bytes = send_remote_buffer_bytes;
    m_data->send_queue_depth         = static_cast<uint64_t>(send_queue_size);
    m_data->recv_queue_depth         = static_cast<uint64_t>(recv_queue_size);

    // Read memory metrics from /proc/self/status
    read_proc_memory();

    // Increment sequence number so reader knows data is fresh
    m_data->update_sequence = ++m_update_sequence;
#else
    (void)stats;
    (void)pending_isend_bytes;
    (void)send_local_buffer_bytes;
    (void)send_remote_buffer_bytes;
    (void)send_queue_size;
    (void)recv_queue_size;
#endif
  }

  /**
   * @brief Check if monitoring is enabled and active.
   */
  bool is_enabled() const { return m_enabled; }

 private:
  /**
   * @brief Read VmRSS and VmSize from /proc/self/status.
   */
  void read_proc_memory() {
#ifdef __linux__
    if (m_data == nullptr) return;

    std::ifstream status("/proc/self/status");
    if (!status.is_open()) return;

    std::string line;
    while (std::getline(status, line)) {
      if (line.compare(0, 6, "VmRSS:") == 0) {
        unsigned long value = 0;
        if (std::sscanf(line.c_str(), "VmRSS: %lu kB", &value) == 1) {
          m_data->rss_bytes = static_cast<uint64_t>(value) * 1024;
        }
      } else if (line.compare(0, 7, "VmSize:") == 0) {
        unsigned long value = 0;
        if (std::sscanf(line.c_str(), "VmSize: %lu kB", &value) == 1) {
          m_data->vm_size_bytes = static_cast<uint64_t>(value) * 1024;
        }
      }
    }
#endif
  }

  /**
   * @brief Clean up shared memory resources.
   */
  void cleanup() {
#ifdef __linux__
    if (m_data != nullptr && m_data != MAP_FAILED) {
      munmap(m_data, sizeof(monitor_data));
      m_data = nullptr;
    }
    if (m_shm_fd != -1) {
      close(m_shm_fd);
      m_shm_fd = -1;
    }
    if (m_enabled && !m_shm_name.empty()) {
      shm_unlink(m_shm_name.c_str());
    }
    if (m_owns_manifest && !m_uuid.empty()) {
      std::string manifest_path = "/dev/shm/ygm_" + m_uuid + "_manifest";
      std::remove(manifest_path.c_str());
    }
    m_enabled = false;
#endif
  }

  bool        m_enabled       = false;
  int         m_rank          = -1;
  int         m_local_id      = -1;
  std::string m_uuid;
  std::string m_shm_name;
  bool        m_owns_manifest = false;

#ifdef __linux__
  int           m_shm_fd = -1;
  monitor_data* m_data   = nullptr;
#endif

  uint64_t m_update_sequence = 0;
};

}  // namespace ygm::detail
