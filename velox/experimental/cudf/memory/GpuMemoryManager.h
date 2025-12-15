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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/RmmMemoryAllocator.h"

#include "velox/common/memory/Memory.h"
#include "velox/common/memory/SharedArbitrator.h"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <atomic>
#include <mutex>

namespace facebook::velox::cudf_velox::memory {

/// Returns the process-wide GPU MemoryManager instance used for cudf-based
/// execution. The instance is created on first use using the current
/// CudfConfig settings and an RMM-backed MemoryAllocator.
inline ::facebook::velox::memory::MemoryManager* gpuMemoryManager() {
  using ::facebook::velox::memory::MemoryManager;
  using ::facebook::velox::memory::SharedArbitrator;

  static std::shared_ptr<MemoryManager> manager;
  static std::once_flag once;

  std::call_once(once, []() {
    const auto& cfg = CudfConfig::getInstance();

    // Decide GPU capacity based on total device memory and configured percent.
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    const auto status = cudaMemGetInfo(&freeBytes, &totalBytes);
    VELOX_CHECK_EQ(
        status,
        cudaSuccess,
        "cudaMemGetInfo failed: {}",
        cudaGetErrorString(status));

    const double pct = std::max(0, std::min(100, cfg.memoryPercent));
    const auto capacityBytes = static_cast<std::size_t>(
        static_cast<double>(totalBytes) * (pct / 100.0));

    // TODO (dm): Get the actual allocator from what we set in Util
    auto mr = std::make_shared<rmm::mr::cuda_memory_resource>();
    auto allocator =
        std::make_shared<cudf_velox::RmmMemoryAllocator>(mr, capacityBytes);

    // SharedArbitrator factory might already be registered by other code
    // paths (tests, CPU-side setup). Ignore duplicate registration errors.
    try {
      SharedArbitrator::registerFactory();
    } catch (const ::facebook::velox::VeloxRuntimeError&) {
    }

    MemoryManager::Options opts;
    opts.allocatorCapacity = capacityBytes;
    opts.arbitratorCapacity = capacityBytes;
    opts.arbitratorKind = "SHARED";
    // Disable global arbitration for GPU for now; rely on local arbitration.
    opts.extraArbitratorConfigs = {
        {std::string(SharedArbitrator::ExtraConfig::kGlobalArbitrationEnabled),
         "false"}};

    manager = std::make_shared<MemoryManager>(opts, std::move(allocator));
  });

  return manager.get();
}

} // namespace facebook::velox::cudf_velox::memory
