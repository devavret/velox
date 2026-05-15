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

#include "velox/experimental/cudf/exec/CudfPartitionedHashJoin.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include "velox/exec/Task.h"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/partitioning.hpp>

#include <nvtx3/nvtx3.hpp>

#include <folly/ScopeGuard.h>

#include <algorithm>
#include <utility>

namespace facebook::velox::cudf_velox {

namespace {

constexpr auto kOobPolicy = cudf::out_of_bounds_policy::NULLIFY;
constexpr auto kPartitionHash = cudf::hash_id::HASH_MURMUR3;
constexpr auto kPartitionSeed = cudf::DEFAULT_HASH_SEED;

bool isSimpleInnerEquiJoin(const core::HashJoinNode& joinNode) {
  return joinNode.isInnerJoin() && joinNode.filter() == nullptr;
}

std::vector<cudf::size_type> splitPointsFromOffsets(
    std::vector<cudf::size_type> offsets,
    int32_t numPartitions) {
  VELOX_CHECK_EQ(
      offsets.size(),
      static_cast<size_t>(numPartitions) + 1,
      "cuDF hash_partition returned unexpected offset count");
  VELOX_CHECK_EQ(offsets.front(), 0);
  if (offsets.size() <= 2) {
    return {};
  }
  offsets.erase(offsets.begin());
  offsets.pop_back();
  return offsets;
}

std::vector<cudf::size_type> keyIndices(
    const RowTypePtr& type,
    const std::vector<core::FieldAccessTypedExprPtr>& keys) {
  std::vector<cudf::size_type> indices(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    indices[i] =
        static_cast<cudf::size_type>(type->getChildIdx(keys[i]->name()));
  }
  return indices;
}

} // namespace

// =============================================================================
// Build
// =============================================================================

CudfPartitionedHashJoinBuild::CudfPartitionedHashJoinBuild(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode,
    int32_t numPartitions)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          nullptr,
          joinNode->id(),
          "CudfPartitionedHashJoinBuild",
          nvtx3::rgb{65, 105, 225}, // Royal Blue
          NvtxMethodFlag::kAll,
          std::nullopt,
          joinNode),
      joinNode_(std::move(joinNode)),
      numPartitions_(numPartitions) {
  VELOX_CHECK(
      isSimpleInnerEquiJoin(*joinNode_),
      "CudfPartitionedHashJoinBuild only supports inner equi-join with no "
      "filter");
  VELOX_CHECK_GT(numPartitions_, 0);
}

bool CudfPartitionedHashJoinBuild::needsInput() const {
  return !noMoreInput_;
}

void CudfPartitionedHashJoinBuild::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  inputs_.push_back(std::move(cudfInput));
}

RowVectorPtr CudfPartitionedHashJoinBuild::doGetOutput() {
  return nullptr;
}

void CudfPartitionedHashJoinBuild::doNoMoreInput() {
  Operator::noMoreInput();

  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    return;
  }

  for (auto& peer : peers) {
    auto* op = peer->findOperator(planNodeId());
    auto* build = dynamic_cast<CudfPartitionedHashJoinBuild*>(op);
    VELOX_CHECK_NOT_NULL(build);
    inputs_.insert(inputs_.end(), build->inputs_.begin(), build->inputs_.end());
  }

  SCOPE_EXIT {
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }
  };

  auto buildType = joinNode_->sources()[1]->outputType();
  auto buildKeyIndices = keyIndices(buildType, joinNode_->rightKeys());
  auto stream = cudfGlobalStreamPool().get_stream();

  std::vector<std::shared_ptr<cudf::table>> buildPartitions;
  std::vector<std::shared_ptr<cudf::hash_join>> hashObjects;
  buildPartitions.reserve(numPartitions_);
  hashObjects.reserve(numPartitions_);

  auto addPartition = [&](std::unique_ptr<cudf::table> partition) {
    VELOX_CHECK_NOT_NULL(partition);
    std::shared_ptr<cudf::table> sharedPartition(std::move(partition));
    std::shared_ptr<cudf::hash_join> hashJoin;
    if (sharedPartition->num_rows() > 0) {
      hashJoin = std::make_shared<cudf::hash_join>(
          sharedPartition->view().select(buildKeyIndices),
          cudf::null_equality::UNEQUAL,
          stream);
    }
    buildPartitions.push_back(std::move(sharedPartition));
    hashObjects.push_back(std::move(hashJoin));
  };

  if (numPartitions_ == 1) {
    addPartition(getConcatenatedTable(
        std::exchange(inputs_, {}), buildType, stream, get_output_mr()));
  } else {
    std::vector<rmm::cuda_stream_view> inputStreams;
    inputStreams.reserve(inputs_.size());
    for (const auto& input : inputs_) {
      inputStreams.push_back(input->stream());
    }
    if (!inputStreams.empty()) {
      cudf::detail::join_streams(inputStreams, stream);
    }

    std::vector<std::vector<std::unique_ptr<cudf::table>>> partitionChunks(
        numPartitions_);
    for (const auto& input : inputs_) {
      auto [partitionedTable, partitionOffsets] = cudf::hash_partition(
          input->getTableView(),
          buildKeyIndices,
          numPartitions_,
          kPartitionHash,
          kPartitionSeed,
          stream,
          get_output_mr());
      auto splitPoints =
          splitPointsFromOffsets(std::move(partitionOffsets), numPartitions_);
      auto partitionViews =
          cudf::split(partitionedTable->view(), splitPoints, stream);

      for (int32_t i = 0; i < numPartitions_; ++i) {
        auto partitionView = partitionViews[i];
        if (partitionView.num_rows() == 0) {
          continue;
        }
        partitionChunks[i].push_back(
            std::make_unique<cudf::table>(
                partitionView, stream, get_output_mr()));
      }
    }

    for (int32_t i = 0; i < numPartitions_; ++i) {
      if (partitionChunks[i].empty()) {
        addPartition(getConcatenatedTable(
            std::vector<CudfVectorPtr>{}, buildType, stream, get_output_mr()));
      } else if (partitionChunks[i].size() == 1) {
        addPartition(std::move(partitionChunks[i][0]));
      } else {
        addPartition(concatenateTables(
            std::move(partitionChunks[i]), stream, get_output_mr()));
      }
    }
  }

  stream.synchronize();
  inputs_.clear();

  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto cudfHashJoinBridge =
      std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
  VELOX_CHECK_NOT_NULL(cudfHashJoinBridge);
  cudfHashJoinBridge->setBuildStream(stream);

  CudfHashJoinBridge::hash_type hashObject{
      std::move(buildPartitions), std::move(hashObjects)};
  cudfHashJoinBridge->setHashTable(std::make_optional(std::move(hashObject)));
}

exec::BlockingReason CudfPartitionedHashJoinBuild::isBlocked(
    ContinueFuture* future) {
  if (!future_.valid()) {
    return exec::BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return exec::BlockingReason::kWaitForJoinBuild;
}

bool CudfPartitionedHashJoinBuild::isFinished() {
  return !future_.valid() && noMoreInput_;
}

// =============================================================================
// Probe
// =============================================================================

CudfPartitionedHashJoinProbe::CudfPartitionedHashJoinProbe(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode,
    int32_t numPartitions)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          joinNode->outputType(),
          joinNode->id(),
          "CudfPartitionedHashJoinProbe",
          nvtx3::rgb{0, 128, 128}, // Teal
          NvtxMethodFlag::kAll,
          std::nullopt,
          joinNode),
      joinNode_(std::move(joinNode)),
      probeType_(joinNode_->sources()[0]->outputType()),
      buildType_(joinNode_->sources()[1]->outputType()),
      numPartitions_(numPartitions) {
  VELOX_CHECK(
      isSimpleInnerEquiJoin(*joinNode_),
      "CudfPartitionedHashJoinProbe only supports inner equi-join with no "
      "filter");
  VELOX_CHECK_GT(numPartitions_, 0);

  leftKeyIndices_ = keyIndices(probeType_, joinNode_->leftKeys());
  rightKeyIndices_ = keyIndices(buildType_, joinNode_->rightKeys());

  auto outputType = joinNode_->outputType();
  outputColumns_.reserve(outputType->size());
  for (size_t i = 0; i < outputType->size(); ++i) {
    auto const& name = outputType->nameOf(i);
    if (auto idx = probeType_->getChildIdxIfExists(name)) {
      outputColumns_.push_back(
          {true, static_cast<cudf::size_type>(idx.value())});
      continue;
    }
    if (auto idx = buildType_->getChildIdxIfExists(name)) {
      outputColumns_.push_back(
          {false, static_cast<cudf::size_type>(idx.value())});
      continue;
    }
    VELOX_FAIL("Output column not found in probe or build: {}", name);
  }
}

bool CudfPartitionedHashJoinProbe::needsInput() const {
  return !noMoreInput_ && probe_ == nullptr;
}

void CudfPartitionedHashJoinProbe::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  probe_ = std::move(cudfInput);
}

exec::BlockingReason CudfPartitionedHashJoinProbe::isBlocked(
    ContinueFuture* future) {
  if (hashObject_.has_value()) {
    return exec::BlockingReason::kNotBlocked;
  }

  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto cudfHashJoinBridge =
      std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
  VELOX_CHECK_NOT_NULL(cudfHashJoinBridge);
  VELOX_CHECK_NOT_NULL(future);

  auto hashObject = cudfHashJoinBridge->hashOrFuture(future);
  if (!hashObject.has_value()) {
    return exec::BlockingReason::kWaitForJoinBuild;
  }

  VELOX_CHECK_EQ(hashObject->first.size(), static_cast<size_t>(numPartitions_));
  VELOX_CHECK_EQ(
      hashObject->second.size(), static_cast<size_t>(numPartitions_));
  hashObject_ = std::move(hashObject);
  return exec::BlockingReason::kNotBlocked;
}

std::unique_ptr<cudf::table> CudfPartitionedHashJoinProbe::joinPartition(
    cudf::table_view probePartition,
    const std::shared_ptr<cudf::table>& buildPartition,
    const std::shared_ptr<cudf::hash_join>& hashJoin,
    rmm::cuda_stream_view stream) {
  if (probePartition.num_rows() == 0 || buildPartition->num_rows() == 0) {
    return nullptr;
  }
  VELOX_CHECK_NOT_NULL(hashJoin);

  auto [leftIdx, rightIdx] = hashJoin->inner_join(
      probePartition.select(leftKeyIndices_),
      std::nullopt,
      stream,
      get_temp_mr());

  cudf::column_view leftIdxCol{
      cudf::device_span<cudf::size_type const>{*leftIdx}};
  cudf::column_view rightIdxCol{
      cudf::device_span<cudf::size_type const>{*rightIdx}};

  std::vector<cudf::size_type> leftGather;
  std::vector<cudf::size_type> rightGather;
  std::vector<size_t> leftOutputAt;
  std::vector<size_t> rightOutputAt;
  for (size_t i = 0; i < outputColumns_.size(); ++i) {
    if (outputColumns_[i].fromLeft) {
      leftGather.push_back(outputColumns_[i].sourceIndex);
      leftOutputAt.push_back(i);
    } else {
      rightGather.push_back(outputColumns_[i].sourceIndex);
      rightOutputAt.push_back(i);
    }
  }

  std::vector<std::unique_ptr<cudf::column>> joined(outputColumns_.size());
  if (!leftGather.empty()) {
    auto gathered = cudf::gather(
        probePartition.select(leftGather),
        leftIdxCol,
        kOobPolicy,
        stream,
        get_output_mr());
    auto cols = gathered->release();
    for (size_t i = 0; i < cols.size(); ++i) {
      joined[leftOutputAt[i]] = std::move(cols[i]);
    }
  }
  if (!rightGather.empty()) {
    auto gathered = cudf::gather(
        buildPartition->view().select(rightGather),
        rightIdxCol,
        kOobPolicy,
        stream,
        get_output_mr());
    auto cols = gathered->release();
    for (size_t i = 0; i < cols.size(); ++i) {
      joined[rightOutputAt[i]] = std::move(cols[i]);
    }
  }

  auto outputTable = std::make_unique<cudf::table>(std::move(joined));
  if (outputTable->num_rows() == 0) {
    return nullptr;
  }
  return outputTable;
}

RowVectorPtr CudfPartitionedHashJoinProbe::doGetOutput() {
  if (finished_ || !hashObject_.has_value()) {
    return nullptr;
  }

  if (probe_ == nullptr) {
    if (noMoreInput_) {
      finished_ = true;
    }
    return nullptr;
  }

  auto stream = probe_->stream();
  auto probeView = probe_->getTableView();
  auto& buildPartitions = hashObject_->first;
  auto& hashObjects = hashObject_->second;

  std::vector<std::unique_ptr<cudf::table>> outputs;
  if (numPartitions_ == 1) {
    if (auto output = joinPartition(
            probeView, buildPartitions[0], hashObjects[0], stream)) {
      outputs.push_back(std::move(output));
    }
  } else {
    auto [partitionedProbe, partitionOffsets] = cudf::hash_partition(
        probeView,
        leftKeyIndices_,
        numPartitions_,
        kPartitionHash,
        kPartitionSeed,
        stream,
        get_temp_mr());
    auto splitPoints =
        splitPointsFromOffsets(std::move(partitionOffsets), numPartitions_);
    auto probePartitions =
        cudf::split(partitionedProbe->view(), splitPoints, stream);

    for (int32_t i = 0; i < numPartitions_; ++i) {
      if (auto output = joinPartition(
              probePartitions[i], buildPartitions[i], hashObjects[i], stream)) {
        outputs.push_back(std::move(output));
      }
    }
  }

  RowVectorPtr result;
  if (!outputs.empty()) {
    auto outputTable =
        concatenateTables(std::move(outputs), stream, get_output_mr());
    auto const numRows = static_cast<vector_size_t>(outputTable->num_rows());
    if (outputTable->num_columns() > 0 && numRows > 0) {
      stream.synchronize();
      result = std::make_shared<CudfVector>(
          pool(), outputType_, numRows, std::move(outputTable), stream);
    } else {
      stream.synchronize();
    }
  } else {
    stream.synchronize();
  }

  probe_.reset();
  finished_ = noMoreInput_;
  return result;
}

void CudfPartitionedHashJoinProbe::doNoMoreInput() {
  Operator::noMoreInput();
}

void CudfPartitionedHashJoinProbe::doClose() {
  Operator::close();
  hashObject_.reset();
  probe_.reset();
}

bool CudfPartitionedHashJoinProbe::isFinished() {
  auto const isFinished = finished_ || (noMoreInput_ && probe_ == nullptr);
  if (isFinished) {
    hashObject_.reset();
  }
  return isFinished;
}

} // namespace facebook::velox::cudf_velox
