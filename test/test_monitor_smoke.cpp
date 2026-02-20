// Copyright 2019-2026 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#undef NDEBUG

#include <ygm/comm.hpp>
#include <chrono>
#include <iostream>
#include <thread>

// Simple smoke test for the YGM monitor — runs barriers in a loop with sleeps
// so ygm-top has time to attach and display data.

int main(int argc, char** argv) {
  ygm::comm world(&argc, &argv);

  for (int i = 0; i < 10; ++i) {
    // Send some async messages to generate stats
    for (int dest = 0; dest < world.size(); ++dest) {
      world.async(dest, [](int sender){}, world.rank());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));

    world.barrier();
    if (world.rank0()) {
      std::cout << "Barrier " << i + 1 << "/10\n" << std::flush;
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }
  return 0;
}
