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

#include "velox/experimental/cudf/benchmarks/CudfPiecewiseSpillHashJoinAdapter.h"

#include "velox/experimental/cudf/exec/CudfPiecewiseSpillHashJoin.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"

#include "velox/exec/HashBuild.h"
#include "velox/exec/HashProbe.h"

#include <memory>

namespace facebook::velox::cudf_velox {

namespace {

class PiecewiseHashJoinBaseAdapter : public OperatorAdapter {
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
    // Piecewise variant only supports inner equi-join with no filter.
    return joinPlanNode->isInnerJoin() && joinPlanNode->filter() == nullptr;
  }
};

class PiecewiseHashJoinBuildAdapter : public PiecewiseHashJoinBaseAdapter {
 public:
  explicit PiecewiseHashJoinBuildAdapter(cudf::size_type pieceTargetRows)
      : PiecewiseHashJoinBaseAdapter("HashJoinBuild"),
        pieceTargetRows_(pieceTargetRows) {}

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
    result.push_back(std::make_unique<CudfPiecewiseSpillHashJoinBuild>(
        operatorId, ctx, joinPlanNode, pieceTargetRows_));
    return result;
  }

 private:
  cudf::size_type const pieceTargetRows_;
};

class PiecewiseHashJoinProbeAdapter : public PiecewiseHashJoinBaseAdapter {
 public:
  PiecewiseHashJoinProbeAdapter()
      : PiecewiseHashJoinBaseAdapter("HashJoinProbe") {}

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
    result.push_back(std::make_unique<CudfPiecewiseSpillHashJoinProbe>(
        operatorId, ctx, joinPlanNode));
    return result;
  }
};

}  // namespace

void registerPiecewiseSpillHashJoinAdapter(cudf::size_type pieceTargetRows) {
  auto& registry = OperatorAdapterRegistry::getInstance();
  registry.registerAdapter(
      std::make_unique<PiecewiseHashJoinBuildAdapter>(pieceTargetRows),
      /*overwrite=*/true);
  registry.registerAdapter(
      std::make_unique<PiecewiseHashJoinProbeAdapter>(),
      /*overwrite=*/true);
}

}  // namespace facebook::velox::cudf_velox
