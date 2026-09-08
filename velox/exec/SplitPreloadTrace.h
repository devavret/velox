/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <thread>

#include <fcntl.h>
#include <unistd.h>

namespace facebook::velox::exec {

/// Lightweight, opt-in process-wide telemetry for the split-preload producer.
///
/// Set VELOX_SPLIT_PRELOAD_TRACE_DIR to a directory writable by the worker.
/// One CSV file is produced per worker process. Sampling occurs on a dedicated
/// thread so a saturated connector I/O executor cannot hide its own backlog.
/// The singleton is intentionally leaked to avoid process-teardown races with
/// the detached sampler.
class SplitPreloadTrace {
 public:
  static SplitPreloadTrace* get() {
    static auto* trace = []() -> SplitPreloadTrace* {
      const auto* directory = std::getenv("VELOX_SPLIT_PRELOAD_TRACE_DIR");
      if (directory == nullptr || directory[0] == '\0') {
        return nullptr;
      }
      return new SplitPreloadTrace(directory);
    }();
    return trace;
  }

  void descriptorQueued() noexcept {
    queuedDescriptors_.fetch_add(1, std::memory_order_relaxed);
    descriptorsReceived_.fetch_add(1, std::memory_order_relaxed);
  }

  void splitSelected(bool hadPreload, bool preloadReady) noexcept {
    queuedDescriptors_.fetch_sub(1, std::memory_order_relaxed);
    splitsSelected_.fetch_add(1, std::memory_order_relaxed);
    if (hadPreload) {
      preloadedSplitsSelected_.fetch_add(1, std::memory_order_relaxed);
    }
    if (preloadReady) {
      readyPreloadedSplitsSelected_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  void preloadTaskQueued() noexcept {
    queuedPrepareTasks_.fetch_add(1, std::memory_order_relaxed);
    preloadRequests_.fetch_add(1, std::memory_order_relaxed);
  }

  void preloadTaskStarted() noexcept {
    queuedPrepareTasks_.fetch_sub(1, std::memory_order_relaxed);
    runningPrepareTasks_.fetch_add(1, std::memory_order_relaxed);
    prepareStarts_.fetch_add(1, std::memory_order_relaxed);
  }

  void preloadTaskFinished() noexcept {
    runningPrepareTasks_.fetch_sub(1, std::memory_order_relaxed);
    prepareCompletions_.fetch_add(1, std::memory_order_relaxed);
  }

  class PrepareTaskExecution {
   public:
    explicit PrepareTaskExecution(SplitPreloadTrace* trace) noexcept
        : trace_{trace} {
      if (trace_ != nullptr) {
        trace_->preloadTaskStarted();
      }
    }

    ~PrepareTaskExecution() {
      if (trace_ != nullptr) {
        trace_->preloadTaskFinished();
      }
    }

   private:
    SplitPreloadTrace* trace_;
  };

 private:
  explicit SplitPreloadTrace(std::string_view directory) {
    try {
      std::filesystem::create_directories(directory);
      const auto path = std::filesystem::path{directory} /
          ("split-preload-" + std::to_string(getpid()) + ".csv");
      outputFd_ = open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
      if (outputFd_ < 0) {
        return;
      }
      constexpr std::string_view header =
          "wall_ns,queued_split_descriptors,prepare_tasks_queued,"
          "prepare_tasks_running,descriptors_received,preload_requests,"
          "prepare_starts,prepare_completions,splits_selected,"
          "preloaded_splits_selected,ready_preloaded_splits_selected\n";
      write(outputFd_, header.data(), header.size());
      std::thread{[this] { sampleLoop(); }}.detach();
    } catch (...) {
      outputFd_ = -1;
    }
  }

  void sampleLoop() const noexcept {
    using namespace std::chrono_literals;
    while (true) {
      std::this_thread::sleep_for(5ms);
      if (outputFd_ < 0) {
        continue;
      }

      const auto wallNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();
      char line[512];
      const auto length = std::snprintf(
          line,
          sizeof(line),
          "%lld,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
          static_cast<long long>(wallNs),
          static_cast<unsigned long long>(
              queuedDescriptors_.load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(
              queuedPrepareTasks_.load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(
              runningPrepareTasks_.load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(
              descriptorsReceived_.load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(
              preloadRequests_.load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(
              prepareStarts_.load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(
              prepareCompletions_.load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(
              splitsSelected_.load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(
              preloadedSplitsSelected_.load(std::memory_order_relaxed)),
          static_cast<unsigned long long>(
              readyPreloadedSplitsSelected_.load(std::memory_order_relaxed)));
      if (length > 0 && static_cast<size_t>(length) < sizeof(line)) {
        write(outputFd_, line, static_cast<size_t>(length));
      }
    }
  }

  int outputFd_{-1};
  std::atomic<uint64_t> queuedDescriptors_{0};
  std::atomic<uint64_t> queuedPrepareTasks_{0};
  std::atomic<uint64_t> runningPrepareTasks_{0};
  std::atomic<uint64_t> descriptorsReceived_{0};
  std::atomic<uint64_t> preloadRequests_{0};
  std::atomic<uint64_t> prepareStarts_{0};
  std::atomic<uint64_t> prepareCompletions_{0};
  std::atomic<uint64_t> splitsSelected_{0};
  std::atomic<uint64_t> preloadedSplitsSelected_{0};
  std::atomic<uint64_t> readyPreloadedSplitsSelected_{0};
};

} // namespace facebook::velox::exec
