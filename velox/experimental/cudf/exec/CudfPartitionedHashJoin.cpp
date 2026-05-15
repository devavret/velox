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
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/join/hash_join_storage.hpp>
#include <cudf/partitioning.hpp>

#include <cuda_runtime_api.h>
#include <nvtx3/nvtx3.hpp>

#include <folly/ScopeGuard.h>
#include <folly/Unit.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/thread_factory/NamedThreadFactory.h>
#include <folly/futures/Future.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

namespace facebook::velox::cudf_velox {

namespace {

constexpr auto kOobPolicy = cudf::out_of_bounds_policy::NULLIFY;
constexpr auto kPartitionHash = cudf::hash_id::HASH_MURMUR3;
constexpr auto kPartitionSeed = cudf::DEFAULT_HASH_SEED;

folly::CPUThreadPoolExecutor* partitionedUnspillExecutor() {
  static auto pool = std::make_unique<folly::CPUThreadPoolExecutor>(
      1, std::make_shared<folly::NamedThreadFactory>("CudfPartUnspill"));
  return pool.get();
}

void checkCuda(cudaError_t err, const char* op) {
  VELOX_CHECK(err == cudaSuccess, "{} failed: {}", op, cudaGetErrorString(err));
}

int currentCudaDevice() {
  int device = 0;
  checkCuda(cudaGetDevice(&device), "cudaGetDevice");
  return device;
}

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

class PartitionedSpillStore {
 public:
  static PartitionedSpillStore& getInstance() {
    static PartitionedSpillStore instance;
    return instance;
  }

  void put(
      const core::PlanNodeId& planNodeId,
      std::shared_ptr<std::vector<HostBuildPiece>> pieces,
      RowTypePtr buildRowType) {
    std::lock_guard<std::mutex> l(mutex_);
    auto& entry = entries_[planNodeId];
    VELOX_CHECK(
        entry.pieces == nullptr,
        "PartitionedSpillStore already has pieces for plan node: {}",
        planNodeId);
    entry.pieces = std::move(pieces);
    entry.buildRowType = std::move(buildRowType);
  }

  std::pair<std::shared_ptr<std::vector<HostBuildPiece>>, RowTypePtr> acquire(
      const core::PlanNodeId& planNodeId,
      int32_t expectedProbeDrivers) {
    std::lock_guard<std::mutex> l(mutex_);
    auto it = entries_.find(planNodeId);
    VELOX_CHECK(
        it != entries_.end(),
        "PartitionedSpillStore has no entry for {}",
        planNodeId);
    auto& entry = it->second;
    VELOX_CHECK_NOT_NULL(entry.pieces);
    if (entry.expectedProbeDrivers == 0) {
      entry.expectedProbeDrivers = expectedProbeDrivers;
    } else {
      VELOX_CHECK_EQ(entry.expectedProbeDrivers, expectedProbeDrivers);
    }
    return {entry.pieces, entry.buildRowType};
  }

  void release(const core::PlanNodeId& planNodeId) {
    std::lock_guard<std::mutex> l(mutex_);
    auto it = entries_.find(planNodeId);
    if (it == entries_.end()) {
      return;
    }
    auto& entry = it->second;
    ++entry.releasedProbeDrivers;
    if (entry.expectedProbeDrivers > 0 &&
        entry.releasedProbeDrivers >= entry.expectedProbeDrivers) {
      entries_.erase(it);
    }
  }

 private:
  struct Entry {
    std::shared_ptr<std::vector<HostBuildPiece>> pieces;
    RowTypePtr buildRowType;
    int32_t expectedProbeDrivers{0};
    int32_t releasedProbeDrivers{0};
  };

  std::mutex mutex_;
  std::unordered_map<std::string, Entry> entries_;
};

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

  auto pieces = std::make_shared<std::vector<HostBuildPiece>>();
  pieces->reserve(numPartitions_);

  auto spillPartition = [&](std::unique_ptr<cudf::table> partition) {
    VELOX_CHECK_NOT_NULL(partition);
    HostBuildPiece piece;
    piece.numRows = partition->num_rows();
    if (piece.numRows == 0) {
      pieces->push_back(std::move(piece));
      return;
    }

    auto packed = cudf::pack(partition->view(), stream, get_output_mr());
    auto metadataCopy = *packed.metadata;
    auto const payloadSize = packed.gpu_data->size();
    auto packedView = cudf::unpack(
        metadataCopy.data(),
        reinterpret_cast<std::uint8_t const*>(packed.gpu_data->data()));

    auto hashJoin = std::make_unique<cudf::hash_join>(
        packedView.select(buildKeyIndices),
        cudf::null_equality::UNEQUAL,
        stream);
    auto storage = hashJoin->release_storage(stream);

    piece.packedMetadata = std::move(metadataCopy);
    piece.payloadBytes = PinnedHostBuffer{payloadSize};
    piece.hashTableBytes = PinnedHostBuffer{storage.slots.size()};
    piece.hashSlotCount = storage.slot_count;
    piece.hashSlotBytes = storage.slot_bytes;
    piece.hashCompareNulls = storage.compare_nulls;
    piece.hashHasNulls = storage.has_nulls;
    piece.hashLoadFactor = storage.load_factor;

    checkCuda(
        cudaMemcpyAsync(
            piece.payloadBytes.data(),
            packed.gpu_data->data(),
            payloadSize,
            cudaMemcpyDeviceToHost,
            stream.value()),
        "cudaMemcpyAsync D2H of packed partition payload");
    checkCuda(
        cudaMemcpyAsync(
            piece.hashTableBytes.data(),
            storage.slots.data(),
            storage.slots.size(),
            cudaMemcpyDeviceToHost,
            stream.value()),
        "cudaMemcpyAsync D2H of partition hash slots");

    pieces->push_back(std::move(piece));
  };

  if (numPartitions_ == 1) {
    spillPartition(getConcatenatedTable(
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
        spillPartition(getConcatenatedTable(
            std::vector<CudfVectorPtr>{}, buildType, stream, get_output_mr()));
      } else if (partitionChunks[i].size() == 1) {
        spillPartition(std::move(partitionChunks[i][0]));
      } else {
        spillPartition(concatenateTables(
            std::move(partitionChunks[i]), stream, get_output_mr()));
      }
    }
  }

  stream.synchronize();
  inputs_.clear();
  VELOX_CHECK_EQ(pieces->size(), static_cast<size_t>(numPartitions_));

  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto cudfHashJoinBridge =
      std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
  VELOX_CHECK_NOT_NULL(cudfHashJoinBridge);
  PartitionedSpillStore::getInstance().put(
      planNodeId(), std::move(pieces), std::move(buildType));
  cudfHashJoinBridge->setHashTable(
      std::make_optional(CudfHashJoinBridge::hash_type{{}, {}}));
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
  if (pieces_ != nullptr) {
    return exec::BlockingReason::kNotBlocked;
  }

  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto cudfHashJoinBridge =
      std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
  VELOX_CHECK_NOT_NULL(cudfHashJoinBridge);
  VELOX_CHECK_NOT_NULL(future);

  auto signal = cudfHashJoinBridge->hashOrFuture(future);
  if (!signal.has_value()) {
    return exec::BlockingReason::kWaitForJoinBuild;
  }

  auto [pieces, buildType] = PartitionedSpillStore::getInstance().acquire(
      planNodeId(), operatorCtx_->task()->numDrivers(operatorCtx_->driver()));
  VELOX_CHECK_NOT_NULL(pieces);
  VELOX_CHECK_EQ(pieces->size(), static_cast<size_t>(numPartitions_));
  pieces_ = std::move(pieces);
  buildType_ = std::move(buildType);
  spillStoreAcquired_ = true;

  if (!slotsInitialized_) {
    slotA_.stream = cudfGlobalStreamPool().get_stream();
    slotB_.stream = cudfGlobalStreamPool().get_stream();
    slotsInitialized_ = true;
  }
  return exec::BlockingReason::kNotBlocked;
}

void CudfPartitionedHashJoinProbe::loadPartitionInto(
    PartitionSlot& target,
    std::size_t partitionIdx) {
  VELOX_CHECK_NOT_NULL(pieces_);
  VELOX_CHECK_LT(partitionIdx, pieces_->size());
  auto& piece = (*pieces_)[partitionIdx];
  VELOX_CHECK_GT(piece.numRows, 0);
  auto stream = target.stream;

  target.hj.reset();
  target.payloadDevice = rmm::device_buffer{piece.payloadBytes.size(), stream};
  checkCuda(
      cudaMemcpyAsync(
          target.payloadDevice.data(),
          piece.payloadBytes.data(),
          piece.payloadBytes.size(),
          cudaMemcpyHostToDevice,
          stream.value()),
      "cudaMemcpyAsync H2D of packed partition payload");

  cudf::hash_join_storage storage;
  storage.slots = rmm::device_buffer{piece.hashTableBytes.size(), stream};
  storage.slot_count = piece.hashSlotCount;
  storage.slot_bytes = piece.hashSlotBytes;
  storage.compare_nulls = piece.hashCompareNulls;
  storage.has_nulls = piece.hashHasNulls;
  storage.load_factor = piece.hashLoadFactor;
  checkCuda(
      cudaMemcpyAsync(
          storage.slots.data(),
          piece.hashTableBytes.data(),
          piece.hashTableBytes.size(),
          cudaMemcpyHostToDevice,
          stream.value()),
      "cudaMemcpyAsync H2D of partition hash slots");

  target.buildTableView = cudf::unpack(
      piece.packedMetadata.data(),
      reinterpret_cast<std::uint8_t const*>(target.payloadDevice.data()));
  target.hj = cudf::hash_join::from_storage(
      std::move(storage),
      target.buildTableView.select(rightKeyIndices_),
      stream);
  target.partitionIdx = partitionIdx;
  target.valid = true;
}

std::unique_ptr<cudf::table> CudfPartitionedHashJoinProbe::joinPartition(
    cudf::table_view probePartition,
    const PartitionSlot& buildSlot,
    rmm::cuda_stream_view probeStream) {
  if (probePartition.num_rows() == 0) {
    return nullptr;
  }
  VELOX_CHECK(buildSlot.valid, "Build partition slot is not populated");
  VELOX_CHECK_NOT_NULL(buildSlot.hj);

  auto stream = buildSlot.stream;
  CudaEvent event(cudaEventDisableTiming);
  event.recordFrom(probeStream).waitOn(stream);

  auto [leftIdx, rightIdx] = buildSlot.hj->inner_join(
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
        buildSlot.buildTableView.select(rightGather),
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
  if (finished_ || pieces_ == nullptr) {
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

  auto nextBuildPartition =
      [&](std::size_t start) -> std::optional<std::size_t> {
    for (auto i = start; i < pieces_->size(); ++i) {
      if ((*pieces_)[i].numRows > 0) {
        return i;
      }
    }
    return std::nullopt;
  };

  auto makeLoadFuture = [&](PartitionSlot& slot, std::size_t partitionIdx) {
    auto* slotPtr = &slot;
    auto const device = currentCudaDevice();
    return folly::via(
        partitionedUnspillExecutor(),
        [this, slotPtr, partitionIdx, device]() -> folly::Unit {
          checkCuda(cudaSetDevice(device), "cudaSetDevice");
          loadPartitionInto(*slotPtr, partitionIdx);
          return folly::Unit{};
        });
  };

  auto currentPartition = nextBuildPartition(0);
  if (!currentPartition.has_value()) {
    probe_.reset();
    finished_ = noMoreInput_;
    return nullptr;
  }

  VELOX_CHECK(slotsInitialized_);
  bool residentIsA = true;
  auto loadFuture = makeLoadFuture(slotA_, *currentPartition);

  std::vector<std::unique_ptr<cudf::table>> outputs;
  std::unique_ptr<cudf::table> partitionedProbe;
  std::vector<cudf::table_view> probePartitions;
  if (numPartitions_ == 1) {
    probePartitions.push_back(probeView);
  } else {
    auto [partitioned, partitionOffsets] = cudf::hash_partition(
        probeView,
        leftKeyIndices_,
        numPartitions_,
        kPartitionHash,
        kPartitionSeed,
        stream,
        get_temp_mr());
    partitionedProbe = std::move(partitioned);
    auto splitPoints =
        splitPointsFromOffsets(std::move(partitionOffsets), numPartitions_);
    probePartitions =
        cudf::split(partitionedProbe->view(), splitPoints, stream);
  }

  while (currentPartition.has_value()) {
    std::move(loadFuture).get();
    auto& resident = residentIsA ? slotA_ : slotB_;
    auto& next = residentIsA ? slotB_ : slotA_;
    VELOX_CHECK_EQ(resident.partitionIdx, *currentPartition);

    auto nextPartition = nextBuildPartition(*currentPartition + 1);
    std::optional<folly::Future<folly::Unit>> prefetchFuture;
    if (nextPartition.has_value()) {
      prefetchFuture = makeLoadFuture(next, *nextPartition);
    }

    try {
      if (auto output = joinPartition(
              probePartitions[*currentPartition], resident, stream)) {
        resident.stream.synchronize();
        outputs.push_back(std::move(output));
      } else {
        resident.stream.synchronize();
      }
    } catch (...) {
      if (prefetchFuture.has_value()) {
        try {
          std::move(*prefetchFuture).get();
        } catch (...) {
        }
      }
      throw;
    }

    if (prefetchFuture.has_value()) {
      loadFuture = std::move(*prefetchFuture);
      currentPartition = nextPartition;
      residentIsA = !residentIsA;
    } else {
      currentPartition = std::nullopt;
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

void CudfPartitionedHashJoinProbe::releaseSpillStore() {
  if (spillStoreAcquired_) {
    PartitionedSpillStore::getInstance().release(planNodeId());
    spillStoreAcquired_ = false;
  }
  pieces_.reset();
}

void CudfPartitionedHashJoinProbe::doClose() {
  Operator::close();
  if (slotsInitialized_) {
    slotA_.stream.synchronize();
    slotB_.stream.synchronize();
  }
  slotA_.hj.reset();
  slotB_.hj.reset();
  slotA_.payloadDevice = rmm::device_buffer{};
  slotB_.payloadDevice = rmm::device_buffer{};
  slotA_.valid = false;
  slotB_.valid = false;
  releaseSpillStore();
  probe_.reset();
}

bool CudfPartitionedHashJoinProbe::isFinished() {
  auto const isFinished = finished_ || (noMoreInput_ && probe_ == nullptr);
  if (isFinished) {
    releaseSpillStore();
  }
  return isFinished;
}

} // namespace facebook::velox::cudf_velox
