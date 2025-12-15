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

#include "velox/experimental/cudf/memory/GpuMemoryManager.h"

#include "velox/core/QueryCtx.h"
#include "velox/exec/Task.h"

#include <fmt/format.h>

namespace facebook::velox::cudf_velox::memory {

/// Returns a GPU query root pool for this QueryCtx, creating one on first use.
/// The created pool is attached to QueryCtx via setGpuPool() and reuses
/// QueryCtx::MemoryReclaimer for arbitration.
// TODO (dm): See if this can return raw pointer
inline std::shared_ptr<::facebook::velox::memory::MemoryPool>
getOrCreateGpuQueryPool(const std::shared_ptr<core::QueryCtx>& queryCtx) {
  auto* existing = queryCtx->gpuPool();
  if (existing != nullptr) {
    // Wrap the raw pointer without taking ownership; QueryCtx keeps the owner.
    return std::shared_ptr<::facebook::velox::memory::MemoryPool>(
        existing, [](::facebook::velox::memory::MemoryPool*) {});
  }

  auto* mgr = gpuMemoryManager();
  auto root = mgr->addRootPool(
      core::QueryCtx::generatePoolName(queryCtx->queryId()) + ".gpu",
      ::facebook::velox::memory::kMaxMemory);
  queryCtx->setGpuPool(root);
  return root;
}

/// Creates or returns the GPU task pool for this Task and attaches
/// Task::MemoryReclaimer to it via Task::setGpuPool().
inline std::shared_ptr<::facebook::velox::memory::MemoryPool>
getOrCreateGpuTaskPool(const std::shared_ptr<exec::Task>& task) {
  auto* existing = task->gpuPool();
  if (existing != nullptr) {
    return std::shared_ptr<::facebook::velox::memory::MemoryPool>(
        existing, [](::facebook::velox::memory::MemoryPool*) {});
  }

  auto queryCtx = task->queryCtx();
  auto gpuQueryRoot = getOrCreateGpuQueryPool(queryCtx);
  auto gpuTaskPool = gpuQueryRoot->addAggregateChild(
      fmt::format("task_gpu.{}", task->taskId()));
  task->setGpuPool(gpuTaskPool);
  return gpuTaskPool;
}

/// Creates a leaf GPU operator pool under the Task's GPU task pool for cudf
/// based operators. The returned pointer is a raw MemoryPool; the caller is
/// responsible for holding a shared_ptr reference if it needs to extend the
/// lifetime beyond that of the Task.
inline ::facebook::velox::memory::MemoryPool* addGpuOperatorPool(
    const std::shared_ptr<exec::Task>& task,
    const core::PlanNodeId& planNodeId,
    uint32_t splitGroupId,
    int pipelineId,
    uint32_t driverId,
    const std::string& operatorType) {
  auto gpuTaskPool = getOrCreateGpuTaskPool(task);
  const std::string nodeSuffix =
      splitGroupId == ::facebook::velox::exec::kUngroupedGroupId
      ? ""
      : fmt::format("[{}]", splitGroupId);
  auto opPool = gpuTaskPool->addLeafChild(
      fmt::format(
          "op_gpu.{}{}.{}.{}.{}",
          planNodeId,
          nodeSuffix,
          pipelineId,
          driverId,
          operatorType));
  return opPool.get();
}

} // namespace facebook::velox::cudf_velox::memory
