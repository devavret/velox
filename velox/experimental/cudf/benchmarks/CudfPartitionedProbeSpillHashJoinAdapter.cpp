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

#include "velox/experimental/cudf/benchmarks/CudfPartitionedProbeSpillHashJoinAdapter.h"
#include "velox/experimental/cudf/exec/CudfPartitionedProbeSpillHashJoin.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"

#include "velox/exec/HashBuild.h"
#include "velox/exec/HashProbe.h"
#include "velox/exec/Operator.h"

#include <memory>

namespace facebook::velox::cudf_velox {

namespace {

class PartitionedProbeSpillHashJoinBaseAdapter : public OperatorAdapter {
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

class PartitionedProbeSpillHashJoinBuildAdapter
    : public PartitionedProbeSpillHashJoinBaseAdapter {
 public:
  PartitionedProbeSpillHashJoinBuildAdapter(
      int32_t numPartitions,
      SpillHostMemoryKind spillHostMemoryKind)
      : PartitionedProbeSpillHashJoinBaseAdapter("HashJoinBuild"),
        numPartitions_(numPartitions),
        spillHostMemoryKind_(spillHostMemoryKind) {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::HashBuild*>(op) != nullptr ||
        dynamic_cast<const CudfHashJoinBuild*>(op) != nullptr ||
        dynamic_cast<const CudfPartitionedProbeSpillHashJoinBuild*>(op) !=
        nullptr;
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
        std::make_unique<CudfPartitionedProbeSpillHashJoinBuild>(
            operatorId,
            ctx,
            joinPlanNode,
            numPartitions_,
            spillHostMemoryKind_));
    return result;
  }

 private:
  int32_t const numPartitions_;
  SpillHostMemoryKind const spillHostMemoryKind_;
};

class PartitionedProbeSpillHashJoinProbeAdapter
    : public PartitionedProbeSpillHashJoinBaseAdapter {
 public:
  PartitionedProbeSpillHashJoinProbeAdapter(
      int32_t numPartitions,
      SpillHostMemoryKind spillHostMemoryKind)
      : PartitionedProbeSpillHashJoinBaseAdapter("HashJoinProbe"),
        numPartitions_(numPartitions),
        spillHostMemoryKind_(spillHostMemoryKind) {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::HashProbe*>(op) != nullptr ||
        dynamic_cast<const CudfHashJoinProbe*>(op) != nullptr ||
        dynamic_cast<const CudfPartitionedProbeSpillHashJoinProbe*>(op) !=
        nullptr;
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
        std::make_unique<CudfPartitionedProbeSpillHashJoinProbe>(
            operatorId,
            ctx,
            joinPlanNode,
            numPartitions_,
            spillHostMemoryKind_));
    return result;
  }

 private:
  int32_t const numPartitions_;
  SpillHostMemoryKind const spillHostMemoryKind_;
};

class PartitionedProbeSpillHashJoinBridgeTranslator
    : public exec::Operator::PlanNodeTranslator {
 public:
  PartitionedProbeSpillHashJoinBridgeTranslator(
      int32_t numPartitions,
      SpillHostMemoryKind spillHostMemoryKind)
      : numPartitions_(numPartitions),
        spillHostMemoryKind_(spillHostMemoryKind) {}

  std::unique_ptr<exec::Operator> toOperator(
      exec::DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override {
    auto joinNode = std::dynamic_pointer_cast<const core::HashJoinNode>(node);
    if (!joinNode) {
      return nullptr;
    }
    return std::make_unique<CudfPartitionedProbeSpillHashJoinProbe>(
        id, ctx, joinNode, numPartitions_, spillHostMemoryKind_);
  }

  std::unique_ptr<exec::JoinBridge> toJoinBridge(
      const core::PlanNodePtr& node) override {
    auto joinNode = std::dynamic_pointer_cast<const core::HashJoinNode>(node);
    if (!joinNode) {
      return nullptr;
    }
    return std::make_unique<CudfHashJoinBridge>();
  }

  exec::OperatorSupplier toOperatorSupplier(
      const core::PlanNodePtr& node) override {
    auto joinNode = std::dynamic_pointer_cast<const core::HashJoinNode>(node);
    if (!joinNode) {
      return nullptr;
    }
    return [joinNode,
            numPartitions = numPartitions_,
            spillHostMemoryKind = spillHostMemoryKind_](
               int32_t operatorId, exec::DriverCtx* ctx) {
      return std::make_unique<CudfPartitionedProbeSpillHashJoinBuild>(
          operatorId, ctx, joinNode, numPartitions, spillHostMemoryKind);
    };
  }

  std::optional<uint32_t> maxDrivers(const core::PlanNodePtr& node) override {
    auto joinNode = std::dynamic_pointer_cast<const core::HashJoinNode>(node);
    if (!joinNode) {
      return std::nullopt;
    }
    return 1;
  }

 private:
  int32_t const numPartitions_;
  SpillHostMemoryKind const spillHostMemoryKind_;
};

} // namespace

void registerPartitionedProbeSpillHashJoinAdapter(
    int32_t numPartitions,
    SpillHostMemoryKind spillHostMemoryKind) {
  exec::Operator::unregisterAllOperators();
  exec::Operator::registerOperator(
      std::make_unique<PartitionedProbeSpillHashJoinBridgeTranslator>(
          numPartitions, spillHostMemoryKind));

  auto& registry = OperatorAdapterRegistry::getInstance();
  registry.registerAdapter(
      std::make_unique<PartitionedProbeSpillHashJoinBuildAdapter>(
          numPartitions, spillHostMemoryKind),
      /*overwrite=*/true);
  registry.registerAdapter(
      std::make_unique<PartitionedProbeSpillHashJoinProbeAdapter>(
          numPartitions, spillHostMemoryKind),
      /*overwrite=*/true);
}

} // namespace facebook::velox::cudf_velox
