// Copyright 2019-2025 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <ygm/container/detail/base_iteration.hpp>

namespace ygm::io {

namespace fs = std::filesystem;

/**
 * @brief Distributed text file parsing.
 */
class line_parser : public ygm::container::detail::base_iteration_value<
                        line_parser, std::tuple<std::string>> {
 public:
  using for_all_args = std::tuple<std::string>;

 private:
  // enum for tracking storage accessiblity
  enum class accessibility_tag { distributed, local };

 public:
  /**
   * @brief Construct a new line parser object
   *
   * @param comm Communicator to use for communication
   * @param stringpaths Vector of paths to files to read
   * @param node_local_filesystem True if paths are to a node-local filesystem
   * @param recursive True if directory traversal should be recursive
   */
  line_parser(ygm::comm& comm, const std::vector<std::string>& stringpaths,
              [[maybe_unused]] bool node_local_filesystem = false,
              bool                  recursive             = false)
      : m_comm(comm), m_skip_first_line(false) {
    check_paths(stringpaths, recursive);
    // if (node_local_filesystem) {
    // YGM_ASSERT_RELEASE(false);
    // check_paths(stringpaths, recursive);
    //} else {
    // if (m_comm.rank0()) {
    // check_paths(stringpaths, recursive);
    //}
    //}
  }

  class iterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type        = std::string;
    using difference_type   = std::ptrdiff_t;
    using pointer           = const std::string*;
    using reference         = const std::string&;

    iterator() = default;  // sentinel/end iterator

    reference operator*() const { return m_impl->current_line; }
    pointer   operator->() const { return &m_impl->current_line; }

    iterator& operator++() {
      m_impl->advance();
      return *this;
    }

    // Post-increment still yields a single-pass iterator: the returned
    // copy shares state with *this, so only use the *this object going
    // forward (standard input_iterator caveat).
    iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const iterator& a, const iterator& b) {
      bool a_end = !a.m_impl || !a.m_impl->valid;
      bool b_end = !b.m_impl || !b.m_impl->valid;
      if (a_end || b_end) return a_end == b_end;
      return a.m_impl == b.m_impl;
    }
    friend bool operator!=(const iterator& a, const iterator& b) {
      return !(a == b);
    }

    bool valid() { return m_impl->valid; }

   private:
    friend class line_parser;

    struct impl {
      const std::vector<std::tuple<fs::path, size_t, size_t>>* files{nullptr};
      size_t        file_index{static_cast<size_t>(-1)};
      std::ifstream ifs;
      size_t        bytes_end{0};
      bool          first_line_of_file{false};
      bool          skip_first_line{false};
      std::string   current_line;
      bool          valid{false};

      void open_file_at(size_t idx) {
        file_index                            = idx;
        const auto& [path, bytes_begin, bend] = (*files)[idx];
        bytes_end                             = bend;
        if (ifs.is_open()) ifs.close();
        ifs.clear();
        ifs.open(path);
        YGM_ASSERT_RELEASE(ifs.good());
        ifs.imbue(std::locale::classic());
        // Throw away line containing bytes_begin, as it was read by the
        // previous rank (unless it corresponds to the start of the file).
        if (bytes_begin > 0) {
          ifs.seekg(bytes_begin);
          std::string throwaway;
          std::getline(ifs, throwaway);
          first_line_of_file = false;
        } else {
          first_line_of_file = true;
        }
      }

      // Advances to the next line this iterator should yield, moving on
      // to subsequent files as each is exhausted. Sets valid=false once
      // every assigned file has been fully consumed.
      void advance() {
        while (true) {
          if (ifs.is_open() && ifs.tellg() <= std::streamoff(bytes_end) &&
              std::getline(ifs, current_line)) {
            if (not current_line.empty() && current_line.back() == 0x0D) {
              current_line.resize(current_line.size() - 1);
            }
            bool is_first      = first_line_of_file;
            first_line_of_file = false;
            if (is_first && skip_first_line) {
              continue;  // discard, keep reading
            }
            valid = true;
            return;
          }
          // Current file (if any) is exhausted; move to the next one.
          if (ifs.is_open()) ifs.close();
          size_t next_index =
              (file_index == static_cast<size_t>(-1)) ? 0 : file_index + 1;
          if (!files || next_index >= files->size()) {
            valid = false;
            return;
          }
          open_file_at(next_index);
        }
      }
    };

    explicit iterator(std::shared_ptr<impl> impl) : m_impl(std::move(impl)) {}

    std::shared_ptr<impl> m_impl;
  };

  using const_iterator = iterator;

  /**
   * @brief Executes a user function for every line in a set of files.
   *
   * @tparam Function functor type
   * @param fn User function to execute
   */
  template <typename Function>
  void for_all(Function fn) {
    for (const auto& line : *this) {
      fn(line);
    }
  }

  std::string read_first_line() {
    std::string line;
    if (m_comm.rank0()) {
      std::ifstream ifs(m_paths[0].first);
      std::getline(ifs, line);
    }

    line = m_comm.mpi_bcast(line, 0, m_comm.get_mpi_comm());

    return line;
  }

  void set_skip_first_line(bool skip_first) { m_skip_first_line = skip_first; }

  ygm::comm& comm() { return m_comm; }

  const ygm::comm& comm() const { return m_comm; }

  /**
   * @brief Returns a lazy, single-pass iterator to the first line assigned
   * to this rank.
   */
  iterator begin() const {
    partition_files();
    auto impl             = std::make_shared<iterator::impl>();
    impl->files           = &m_local_files;
    impl->skip_first_line = m_skip_first_line;
    impl->advance();
    return iterator(impl);
  }

  /**
   * @brief Returns the past-the-end sentinel iterator.
   */
  iterator end() const { return iterator(); }

 private:
  /**
   * @brief Check readability of paths and iterates through directories
   *
   * @param stringpaths Vector of paths to check
   * @param recursive True if directory traversal should be recursive
   */
  void check_paths(const std::vector<std::string>& stringpaths,
                   bool                            recursive) {
    for (const std::string& strp : stringpaths) {
      if (strp.starts_with("local://")) {
        if (m_comm.layout().local_id() == 0) {
          fs::path p(strp.substr(8));  // Remove prefix
          if (fs::exists(p)) {
            if (fs::is_regular_file(p)) {
              if (is_file_good(p)) {
                m_paths.push_back(std::make_pair(p, accessibility_tag::local));
              }
            } else if (fs::is_directory(p)) {
              if (recursive) {
                //
                // If a directory & user requested recursive
                const std::filesystem::recursive_directory_iterator end;
                for (std::filesystem::recursive_directory_iterator itr{p};
                     itr != end; itr++) {
                  if (fs::is_regular_file(itr->path())) {
                    if (is_file_good(itr->path())) {
                      m_paths.push_back(std::make_pair(
                          itr->path(), accessibility_tag::local));
                    }
                  }
                }
              } else {
                //
                // If a directory & user did not request recursive
                const std::filesystem::directory_iterator end;
                for (std::filesystem::directory_iterator itr{p}; itr != end;
                     itr++) {
                  if (fs::is_regular_file(itr->path())) {
                    if (is_file_good(itr->path())) {
                      m_paths.push_back(std::make_pair(
                          itr->path(), accessibility_tag::local));
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        // Assign distributed files to rank 0 for splitting
        if (m_comm.rank0()) {
          fs::path p(strp);
          if (fs::exists(p)) {
            if (fs::is_regular_file(p)) {
              if (is_file_good(p)) {
                m_paths.push_back(
                    std::make_pair(p, accessibility_tag::distributed));
              }
            } else if (fs::is_directory(p)) {
              if (recursive) {
                //
                // If a directory & user requested recursive
                const std::filesystem::recursive_directory_iterator end;
                for (std::filesystem::recursive_directory_iterator itr{p};
                     itr != end; itr++) {
                  if (fs::is_regular_file(itr->path())) {
                    if (is_file_good(itr->path())) {
                      m_paths.push_back(std::make_pair(
                          itr->path(), accessibility_tag::distributed));
                    }
                  }
                }
              } else {
                //
                // If a directory & user did not request recursive
                const std::filesystem::directory_iterator end;
                for (std::filesystem::directory_iterator itr{p}; itr != end;
                     itr++) {
                  if (fs::is_regular_file(itr->path())) {
                    if (is_file_good(itr->path())) {
                      m_paths.push_back(std::make_pair(
                          itr->path(), accessibility_tag::distributed));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    //
    // Remove duplicate paths
    std::sort(m_paths.begin(), m_paths.end());
    m_paths.erase(std::unique(m_paths.begin(), m_paths.end()), m_paths.end());
  }

  void partition_files() const {
    m_local_files.clear();

    ygm::ygm_ptr<std::vector<std::tuple<fs::path, size_t, size_t>>>
        p_my_file_paths = m_comm.make_ygm_ptr(m_local_files);

    //
    //  Splits files over ranks by file size.   8MB is smallest granularity.
    //  This approach could be improved by having rank_layout information.
    // Starts with distributed files from rank 0
    m_comm.barrier();
    if (m_comm.rank0()) {
      std::vector<std::tuple<fs::path, size_t, size_t>> remaining_files;
      size_t                                            total_size{0};
      for (size_t i = 0; i < m_paths.size(); ++i) {
        if (m_paths[i].second == accessibility_tag::distributed) {
          size_t fsize = fs::file_size(m_paths[i].first);
          total_size += fsize;
          remaining_files.push_back(
              std::make_tuple(m_paths[i].first, size_t(0), fsize));
        }
      }

      if (total_size > 0) {
        size_t bytes_per_rank =
            std::max((total_size / m_comm.size()) + 1, size_t(8 * 1024 * 1024));
        for (int rank = 0; rank < m_comm.size(); ++rank) {
          size_t remaining_budget = bytes_per_rank;
          while (remaining_budget > 0 && !remaining_files.empty()) {
            size_t file_remaining = std::get<2>(remaining_files.back()) -
                                    std::get<1>(remaining_files.back());
            size_t& cur_position = std::get<1>(remaining_files.back());
            if (file_remaining > remaining_budget) {
              m_comm.async(
                  rank,
                  [](ygm::ygm_ptr<
                         std::vector<std::tuple<fs::path, size_t, size_t>>>
                                        p_local_file_paths,
                     const std::string& fname, size_t bytes_begin,
                     size_t bytes_end) {
                    p_local_file_paths->push_back(
                        {fs::path(fname), bytes_begin, bytes_end});
                  },
                  p_my_file_paths,
                  (std::string)std::get<0>(remaining_files.back()),
                  cur_position, cur_position + remaining_budget);
              cur_position += remaining_budget;
              remaining_budget = 0;
            } else if (file_remaining <= remaining_budget) {
              m_comm.async(
                  rank,
                  [](ygm::ygm_ptr<
                         std::vector<std::tuple<fs::path, size_t, size_t>>>
                                        p_local_file_paths,
                     const std::string& fname, size_t bytes_begin,
                     size_t bytes_end) {
                    p_local_file_paths->push_back(
                        {fs::path(fname), bytes_begin, bytes_end});
                  },
                  p_my_file_paths,
                  (std::string)std::get<0>(remaining_files.back()),
                  cur_position, std::get<2>(remaining_files.back()));
              remaining_budget -= file_remaining;
              remaining_files.pop_back();
            }
          }
        }
      }
    }

    // First rank on every node checks its local files
    if (m_comm.layout().local_id() == 0) {
      std::vector<std::tuple<fs::path, size_t, size_t>> remaining_files;
      size_t                                            total_size{0};
      for (size_t i = 0; i < m_paths.size(); ++i) {
        if (m_paths[i].second == accessibility_tag::local) {
          size_t fsize = fs::file_size(m_paths[i].first);
          total_size += fsize;
          remaining_files.push_back(
              std::make_tuple(m_paths[i].first, size_t(0), fsize));
        }
      }

      if (total_size > 0) {
        size_t bytes_per_rank =
            std::max((total_size / m_comm.layout().local_size()) + 1,
                     size_t(8 * 1024 * 1024));
        for (int rank : m_comm.layout().local_ranks()) {
          size_t remaining_budget = bytes_per_rank;
          while (remaining_budget > 0 && !remaining_files.empty()) {
            size_t file_remaining = std::get<2>(remaining_files.back()) -
                                    std::get<1>(remaining_files.back());
            size_t& cur_position = std::get<1>(remaining_files.back());
            if (file_remaining > remaining_budget) {
              m_comm.async(
                  rank,
                  [](ygm::ygm_ptr<
                         std::vector<std::tuple<fs::path, size_t, size_t>>>
                                        p_local_file_paths,
                     const std::string& fname, size_t bytes_begin,
                     size_t bytes_end) {
                    p_local_file_paths->push_back(
                        {fs::path(fname), bytes_begin, bytes_end});
                  },
                  p_my_file_paths,
                  (std::string)std::get<0>(remaining_files.back()),
                  cur_position, cur_position + remaining_budget);
              cur_position += remaining_budget;
              remaining_budget = 0;
            } else if (file_remaining <= remaining_budget) {
              m_comm.async(
                  rank,
                  [](ygm::ygm_ptr<
                         std::vector<std::tuple<fs::path, size_t, size_t>>>
                                        p_local_file_paths,
                     const std::string& fname, size_t bytes_begin,
                     size_t bytes_end) {
                    p_local_file_paths->push_back(
                        {fs::path(fname), bytes_begin, bytes_end});
                  },
                  p_my_file_paths,
                  (std::string)std::get<0>(remaining_files.back()),
                  cur_position, std::get<2>(remaining_files.back()));
              remaining_budget -= file_remaining;
              remaining_files.pop_back();
            }
          }
        }
      }
    }
    m_comm.barrier();
  }

  /**
   * @brief Checks if file is readable
   *
   * @param p Path to file
   * @return true
   * @return false
   */
  bool is_file_good(const fs::path& p) {
    std::ifstream ifs(p);
    bool          good = ifs.good();
    if (!good) {
      m_comm.cout("WARNING: unable to open: ", p);
    }
    return good;
  }

  mutable std::vector<std::tuple<fs::path, size_t, size_t>> m_local_files;
  ygm::comm&                                                m_comm;
  std::vector<std::pair<fs::path, accessibility_tag>>       m_paths;
  bool                                                      m_skip_first_line;
};

}  // namespace ygm::io
