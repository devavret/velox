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

#include "velox/experimental/cudf/benchmarks/CudfPartitionedHashJoinAdapter.h"
#include "velox/experimental/cudf/exec/CudfPartitionedHashJoin.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"

#include "velox/exec/HashBuild.h"
#include "velox/exec/HashProbe.h"

#include <memory>

namespace facebook::velox::cudf_velox {

namespace {

class PartitionedHashJoinBaseAdapter : public OperatorAdapter {
 public:
  using OperatorAdapter::OperatorAdapter;

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    if (!canHandle(op)) {
      return false;
    }
    auto joinPlanNode =
        std::dynamic_pointer_cast<const core::HashJoinNode>(planNode);
    if (!joinPlanNode) {
      return false;
    }
    return joinPlanNode->isInnerJoin() && joinPlanNode->filter() == nullptr;
  }
};

class PartitionedHashJoinBuildAdapter : public PartitionedHashJoinBaseAdapter {
 public:
  PartitionedHashJoinBuildAdapter(
      int32_t numPartitions,
      bool usePinnedHostMemory)
      : PartitionedHashJoinBaseAdapter("HashJoinBuild"),
        numPartitions_(numPartitions),
        usePinnedHostMemory_(usePinnedHostMemory) {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::HashBuild*>(op) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto joinPlanNode =
        std::dynamic_pointer_cast<const core::HashJoinNode>(planNode);
    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfPartitionedHashJoinBuild>(
            operatorId,
            ctx,
            joinPlanNode,
            numPartitions_,
            usePinnedHostMemory_));
    return result;
  }

 private:
  int32_t const numPartitions_;
  bool const usePinnedHostMemory_;
};

class PartitionedHashJoinProbeAdapter : public PartitionedHashJoinBaseAdapter {
 public:
  explicit PartitionedHashJoinProbeAdapter(int32_t numPartitions)
      : PartitionedHashJoinBaseAdapter("HashJoinProbe"),
        numPartitions_(numPartitions) {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::HashProbe*>(op) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto joinPlanNode =
        std::dynamic_pointer_cast<const core::HashJoinNode>(planNode);
    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfPartitionedHashJoinProbe>(
            operatorId, ctx, joinPlanNode, numPartitions_));
    return result;
  }

 private:
  int32_t const numPartitions_;
};

} // namespace

void registerPartitionedHashJoinAdapter(
    int32_t numPartitions,
    bool usePinnedHostMemory) {
  auto& registry = OperatorAdapterRegistry::getInstance();
  registry.registerAdapter(
      std::make_unique<PartitionedHashJoinBuildAdapter>(
          numPartitions, usePinnedHostMemory),
      /*overwrite=*/true);
  registry.registerAdapter(
      std::make_unique<PartitionedHashJoinProbeAdapter>(numPartitions),
      /*overwrite=*/true);
}

} // namespace facebook::velox::cudf_velox
