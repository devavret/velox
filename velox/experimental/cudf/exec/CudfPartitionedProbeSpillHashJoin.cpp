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

#include "velox/experimental/cudf/exec/CudfPartitionedProbeSpillHashJoin.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include "velox/exec/Task.h"

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
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

folly::CPUThreadPoolExecutor* probeSpillUnspillExecutor() {
  static auto pool = std::make_unique<folly::CPUThreadPoolExecutor>(
      1, std::make_shared<folly::NamedThreadFactory>("CudfProbePartUnspill"));
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

class PartitionedProbeSpillStore {
 public:
  static PartitionedProbeSpillStore& getInstance() {
    static PartitionedProbeSpillStore instance;
    return instance;
  }

  void put(
      const core::PlanNodeId& planNodeId,
      std::shared_ptr<CudfProbeSpillTablePartitions> partitions,
      RowTypePtr buildRowType) {
    std::lock_guard<std::mutex> l(mutex_);
    auto& entry = entries_[planNodeId];
    VELOX_CHECK(
        entry.partitions == nullptr,
        "PartitionedProbeSpillStore already has pieces for plan node: {}",
        planNodeId);
    entry.partitions = std::move(partitions);
    entry.buildRowType = std::move(buildRowType);
  }

  std::pair<std::shared_ptr<CudfProbeSpillTablePartitions>, RowTypePtr> acquire(
      const core::PlanNodeId& planNodeId,
      int32_t expectedProbeDrivers) {
    std::lock_guard<std::mutex> l(mutex_);
    auto it = entries_.find(planNodeId);
    VELOX_CHECK(
        it != entries_.end(),
        "PartitionedProbeSpillStore has no entry for {}",
        planNodeId);
    auto& entry = it->second;
    VELOX_CHECK_NOT_NULL(entry.partitions);
    if (entry.expectedProbeDrivers == 0) {
      entry.expectedProbeDrivers = expectedProbeDrivers;
    } else {
      VELOX_CHECK_EQ(entry.expectedProbeDrivers, expectedProbeDrivers);
    }
    return {entry.partitions, entry.buildRowType};
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
    std::shared_ptr<CudfProbeSpillTablePartitions> partitions;
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

CudfPartitionedProbeSpillHashJoinBuild::CudfPartitionedProbeSpillHashJoinBuild(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode,
    int32_t numPartitions,
    SpillHostMemoryKind spillHostMemoryKind)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          nullptr,
          joinNode->id(),
          "CudfPartitionedProbeSpillHashJoinBuild",
          nvtx3::rgb{65, 105, 225}, // Royal Blue
          NvtxMethodFlag::kAll,
          std::nullopt,
          joinNode),
      joinNode_(std::move(joinNode)),
      numPartitions_(numPartitions),
      spillHostMemoryKind_(spillHostMemoryKind) {
  VELOX_CHECK(
      isSimpleInnerEquiJoin(*joinNode_),
      "CudfPartitionedProbeSpillHashJoinBuild only supports inner equi-join "
      "with no filter");
  VELOX_CHECK_GT(numPartitions_, 0);
}

bool CudfPartitionedProbeSpillHashJoinBuild::needsInput() const {
  return !noMoreInput_;
}

void CudfPartitionedProbeSpillHashJoinBuild::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  inputs_.push_back(std::move(cudfInput));
}

RowVectorPtr CudfPartitionedProbeSpillHashJoinBuild::doGetOutput() {
  return nullptr;
}

CudfProbeSpillTablePiece CudfPartitionedProbeSpillHashJoinBuild::spillTableView(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) const {
  CudfProbeSpillTablePiece piece;
  piece.numRows = tableView.num_rows();
  if (piece.numRows == 0) {
    return piece;
  }

  auto packed = cudf::pack(tableView, stream, get_output_mr());
  piece.packedMetadata = *packed.metadata;
  auto const payloadSize = packed.gpu_data->size();
  piece.payloadBytes = SpillHostBuffer{payloadSize, spillHostMemoryKind_};

  copyDeviceToSpillHost(
      piece.payloadBytes,
      packed.gpu_data->data(),
      payloadSize,
      stream,
      "D2H copy of probe-spilled build partition payload");
  return piece;
}

void CudfPartitionedProbeSpillHashJoinBuild::doNoMoreInput() {
  Operator::noMoreInput();

  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    return;
  }

  for (auto& peer : peers) {
    auto* op = peer->findOperator(planNodeId());
    auto* build = dynamic_cast<CudfPartitionedProbeSpillHashJoinBuild*>(op);
    VELOX_CHECK_NOT_NULL(build);
    for (auto& input : build->inputs_) {
      inputs_.push_back(std::move(input));
    }
    build->inputs_.clear();
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

  auto partitions =
      std::make_shared<CudfProbeSpillTablePartitions>(numPartitions_);

  if (inputs_.empty()) {
    stream.synchronize();
  } else if (numPartitions_ == 1) {
    auto buildInputs = std::exchange(inputs_, {});
    std::vector<rmm::cuda_stream_view> inputStreams;
    inputStreams.reserve(buildInputs.size());
    std::vector<cudf::table_view> inputViews;
    inputViews.reserve(buildInputs.size());
    for (const auto& input : buildInputs) {
      inputStreams.push_back(input->stream());
      inputViews.push_back(input->getTableView());
    }
    if (!inputStreams.empty()) {
      cudf::detail::join_streams(inputStreams, stream);
    }
    for (auto view : inputViews) {
      auto piece = spillTableView(view, stream);
      if (piece.numRows > 0) {
        (*partitions)[0].push_back(std::move(piece));
      }
    }
    stream.synchronize();
    buildInputs.clear();
  } else {
    auto buildInputs = std::exchange(inputs_, {});
    for (auto& input : buildInputs) {
      auto const inputStream = input->stream();
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

      CudaEvent inputReleaseEvent(cudaEventDisableTiming);
      inputReleaseEvent.recordFrom(stream).waitOn(inputStream);
      input.reset();

      for (int32_t i = 0; i < numPartitions_; ++i) {
        auto piece = spillTableView(partitionViews[i], stream);
        if (piece.numRows > 0) {
          (*partitions)[i].push_back(std::move(piece));
        }
      }

      // Keep the partitioned table alive until all pack + D2H copies that read
      // from its views complete, then drop it before processing the next input.
      stream.synchronize();
      partitionViews.clear();
      partitionedTable.reset();
    }
    buildInputs.clear();
  }

  inputs_.clear();

  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto cudfHashJoinBridge =
      std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
  VELOX_CHECK_NOT_NULL(cudfHashJoinBridge);
  PartitionedProbeSpillStore::getInstance().put(
      planNodeId(), std::move(partitions), std::move(buildType));
  cudfHashJoinBridge->setHashTable(
      std::make_optional(CudfHashJoinBridge::hash_type{{}, {}}));
}

exec::BlockingReason CudfPartitionedProbeSpillHashJoinBuild::isBlocked(
    ContinueFuture* future) {
  if (!future_.valid()) {
    return exec::BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return exec::BlockingReason::kWaitForJoinBuild;
}

bool CudfPartitionedProbeSpillHashJoinBuild::isFinished() {
  return !future_.valid() && noMoreInput_;
}

// =============================================================================
// Probe
// =============================================================================

CudfPartitionedProbeSpillHashJoinProbe::CudfPartitionedProbeSpillHashJoinProbe(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode,
    int32_t numPartitions,
    SpillHostMemoryKind spillHostMemoryKind)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          joinNode->outputType(),
          joinNode->id(),
          "CudfPartitionedProbeSpillHashJoinProbe",
          nvtx3::rgb{0, 128, 128}, // Teal
          NvtxMethodFlag::kAll,
          std::nullopt,
          joinNode),
      joinNode_(std::move(joinNode)),
      probeType_(joinNode_->sources()[0]->outputType()),
      buildType_(joinNode_->sources()[1]->outputType()),
      numPartitions_(numPartitions),
      spillHostMemoryKind_(spillHostMemoryKind),
      probePartitions_(numPartitions_) {
  VELOX_CHECK(
      isSimpleInnerEquiJoin(*joinNode_),
      "CudfPartitionedProbeSpillHashJoinProbe only supports inner equi-join "
      "with no filter");
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

bool CudfPartitionedProbeSpillHashJoinProbe::needsInput() const {
  return !noMoreInput_ && !finished_;
}

CudfProbeSpillTablePiece CudfPartitionedProbeSpillHashJoinProbe::spillTableView(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) const {
  CudfProbeSpillTablePiece piece;
  piece.numRows = tableView.num_rows();
  if (piece.numRows == 0) {
    return piece;
  }

  auto packed = cudf::pack(tableView, stream, get_output_mr());
  piece.packedMetadata = *packed.metadata;
  auto const payloadSize = packed.gpu_data->size();
  piece.payloadBytes = SpillHostBuffer{payloadSize, spillHostMemoryKind_};

  copyDeviceToSpillHost(
      piece.payloadBytes,
      packed.gpu_data->data(),
      payloadSize,
      stream,
      "D2H copy of probe-spilled probe partition payload");
  return piece;
}

void CudfPartitionedProbeSpillHashJoinProbe::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }
  VELOX_CHECK_NOT_NULL(
      buildPartitions_,
      "CudfPartitionedProbeSpillHashJoinProbe received input before build "
      "publication");

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  auto stream = cudfInput->stream();

  if (numPartitions_ == 1) {
    if (!(*buildPartitions_)[0].empty()) {
      auto piece = spillTableView(cudfInput->getTableView(), stream);
      if (piece.numRows > 0) {
        probePartitions_[0].push_back(std::move(piece));
      }
    }
    stream.synchronize();
    return;
  }

  auto [partitionedTable, partitionOffsets] = cudf::hash_partition(
      cudfInput->getTableView(),
      leftKeyIndices_,
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
    if ((*buildPartitions_)[i].empty()) {
      continue;
    }
    auto piece = spillTableView(partitionViews[i], stream);
    if (piece.numRows > 0) {
      probePartitions_[i].push_back(std::move(piece));
    }
  }

  stream.synchronize();
  partitionViews.clear();
  partitionedTable.reset();
  cudfInput.reset();
}

exec::BlockingReason CudfPartitionedProbeSpillHashJoinProbe::isBlocked(
    ContinueFuture* future) {
  if (finished_) {
    return exec::BlockingReason::kNotBlocked;
  }
  if (buildPartitions_ != nullptr) {
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

  auto [partitions, buildType] =
      PartitionedProbeSpillStore::getInstance().acquire(
          planNodeId(),
          operatorCtx_->task()->numDrivers(operatorCtx_->driver()));
  VELOX_CHECK_NOT_NULL(partitions);
  VELOX_CHECK_EQ(partitions->size(), static_cast<size_t>(numPartitions_));
  buildPartitions_ = std::move(partitions);
  buildType_ = std::move(buildType);
  spillStoreAcquired_ = true;

  if (!slotsInitialized_) {
    slotA_.stream = cudfGlobalStreamPool().get_stream();
    slotB_.stream = cudfGlobalStreamPool().get_stream();
    slotsInitialized_ = true;
  }
  return exec::BlockingReason::kNotBlocked;
}

CudfPartitionedProbeSpillHashJoinProbe::DeviceTableView
CudfPartitionedProbeSpillHashJoinProbe::loadTablePiece(
    const CudfProbeSpillTablePiece& piece,
    rmm::cuda_stream_view stream) const {
  DeviceTableView table;
  VELOX_CHECK_GT(piece.numRows, 0);
  table.payloadDevice = rmm::device_buffer{piece.payloadBytes.size(), stream};
  copySpillHostToDevice(
      table.payloadDevice.data(),
      piece.payloadBytes,
      stream,
      "H2D copy of probe-spilled partition payload");
  table.tableView = cudf::unpack(
      piece.packedMetadata.data(),
      reinterpret_cast<std::uint8_t const*>(table.payloadDevice.data()));
  return table;
}

void CudfPartitionedProbeSpillHashJoinProbe::resetSlot(
    BuildPartitionSlot& slot) {
  if (slot.valid) {
    slot.stream.synchronize();
  }
  slot.hashJoin.reset();
  slot.table.reset();
  slot.payloadDevices.clear();
  slot.tableView = cudf::table_view{};
  slot.partitionIdx = 0;
  slot.valid = false;
}

void CudfPartitionedProbeSpillHashJoinProbe::loadBuildPartitionInto(
    BuildPartitionSlot& slot,
    std::size_t partitionIdx) {
  VELOX_CHECK_NOT_NULL(buildPartitions_);
  VELOX_CHECK_LT(partitionIdx, buildPartitions_->size());
  auto& pieces = (*buildPartitions_)[partitionIdx];
  VELOX_CHECK(!pieces.empty());
  auto stream = slot.stream;
  resetSlot(slot);

  if (pieces.size() == 1) {
    auto loaded = loadTablePiece(pieces[0], stream);
    slot.payloadDevices.push_back(std::move(loaded.payloadDevice));
    slot.tableView = loaded.tableView;
  } else {
    std::vector<DeviceTableView> loadedPieces;
    loadedPieces.reserve(pieces.size());
    std::vector<cudf::table_view> views;
    views.reserve(pieces.size());
    for (const auto& piece : pieces) {
      auto loaded = loadTablePiece(piece, stream);
      views.push_back(loaded.tableView);
      loadedPieces.push_back(std::move(loaded));
    }
    slot.table = cudf::concatenate(views, stream, get_output_mr());
    stream.synchronize();
    loadedPieces.clear();
    slot.tableView = slot.table->view();
  }

  slot.hashJoin = std::make_unique<cudf::hash_join>(
      slot.tableView.select(rightKeyIndices_),
      cudf::null_equality::UNEQUAL,
      stream);
  slot.partitionIdx = partitionIdx;
  slot.valid = true;
}

folly::Future<folly::Unit>
CudfPartitionedProbeSpillHashJoinProbe::makeLoadFuture(
    BuildPartitionSlot& slot,
    std::size_t partitionIdx) {
  auto* slotPtr = &slot;
  auto const device = currentCudaDevice();
  return folly::via(
      probeSpillUnspillExecutor(),
      [this, slotPtr, partitionIdx, device]() -> folly::Unit {
        checkCuda(cudaSetDevice(device), "cudaSetDevice");
        loadBuildPartitionInto(*slotPtr, partitionIdx);
        return folly::Unit{};
      });
}

std::unique_ptr<cudf::table>
CudfPartitionedProbeSpillHashJoinProbe::joinProbePiece(
    const CudfProbeSpillTablePiece& probePiece,
    const BuildPartitionSlot& buildSlot) {
  VELOX_CHECK(buildSlot.valid, "Build partition slot is not populated");
  VELOX_CHECK_NOT_NULL(buildSlot.hashJoin);
  auto stream = buildSlot.stream;
  auto probe = loadTablePiece(probePiece, stream);

  auto [leftIdx, rightIdx] = buildSlot.hashJoin->inner_join(
      probe.tableView.select(leftKeyIndices_),
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
        probe.tableView.select(leftGather),
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
        buildSlot.tableView.select(rightGather),
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
  stream.synchronize();
  if (outputTable->num_rows() == 0) {
    return nullptr;
  }
  return outputTable;
}

std::optional<std::size_t>
CudfPartitionedProbeSpillHashJoinProbe::nextJoinablePartition(
    std::size_t start) const {
  VELOX_CHECK_NOT_NULL(buildPartitions_);
  for (auto i = start; i < buildPartitions_->size(); ++i) {
    if (!(*buildPartitions_)[i].empty() && !probePartitions_[i].empty()) {
      return i;
    }
  }
  return std::nullopt;
}

void CudfPartitionedProbeSpillHashJoinProbe::maybeStartCurrentPartition() {
  if (currentPartition_.has_value() || finished_) {
    return;
  }
  if (!slotsInitialized_) {
    slotA_.stream = cudfGlobalStreamPool().get_stream();
    slotB_.stream = cudfGlobalStreamPool().get_stream();
    slotsInitialized_ = true;
  }

  auto nextPartition = nextJoinablePartition(0);
  if (!nextPartition.has_value()) {
    finished_ = true;
    releaseSpillStore();
    return;
  }

  residentIsA_ = true;
  currentPartition_ = nextPartition;
  currentProbePiece_ = 0;
  currentLoadFuture_ = makeLoadFuture(slotA_, *currentPartition_);
}

void CudfPartitionedProbeSpillHashJoinProbe::maybeStartNextPartitionPrefetch() {
  if (prefetchFuture_.has_value() || !currentPartition_.has_value()) {
    return;
  }
  auto nextPartition = nextJoinablePartition(*currentPartition_ + 1);
  if (!nextPartition.has_value()) {
    return;
  }
  auto& nextSlot = residentIsA_ ? slotB_ : slotA_;
  prefetchedPartition_ = nextPartition;
  prefetchFuture_ = makeLoadFuture(nextSlot, *nextPartition);
}

void CudfPartitionedProbeSpillHashJoinProbe::advanceToPrefetchedPartition() {
  VELOX_CHECK(prefetchFuture_.has_value());
  VELOX_CHECK(prefetchedPartition_.has_value());
  currentLoadFuture_ = std::move(prefetchFuture_);
  prefetchFuture_.reset();
  currentPartition_ = prefetchedPartition_;
  prefetchedPartition_.reset();
  currentProbePiece_ = 0;
  residentIsA_ = !residentIsA_;
}

void CudfPartitionedProbeSpillHashJoinProbe::finishCurrentPartition() {
  auto& resident = residentIsA_ ? slotA_ : slotB_;
  resetSlot(resident);

  if (prefetchFuture_.has_value()) {
    advanceToPrefetchedPartition();
    return;
  }

  auto nextPartition = nextJoinablePartition(*currentPartition_ + 1);
  currentPartition_.reset();
  currentProbePiece_ = 0;
  if (!nextPartition.has_value()) {
    finished_ = true;
    releaseSpillStore();
    return;
  }

  residentIsA_ = !residentIsA_;
  auto& nextResident = residentIsA_ ? slotA_ : slotB_;
  currentPartition_ = nextPartition;
  currentLoadFuture_ = makeLoadFuture(nextResident, *currentPartition_);
}

RowVectorPtr CudfPartitionedProbeSpillHashJoinProbe::doGetOutput() {
  if (finished_ || buildPartitions_ == nullptr || !noMoreInput_) {
    return nullptr;
  }

  maybeStartCurrentPartition();

  while (!finished_ && currentPartition_.has_value()) {
    if (currentLoadFuture_.has_value()) {
      std::move(*currentLoadFuture_).get();
      currentLoadFuture_.reset();
    }

    auto& resident = residentIsA_ ? slotA_ : slotB_;
    VELOX_CHECK(resident.valid, "Resident build partition slot is empty");
    VELOX_CHECK_EQ(resident.partitionIdx, *currentPartition_);

    maybeStartNextPartitionPrefetch();

    auto& probePieces = probePartitions_[*currentPartition_];
    while (currentProbePiece_ < probePieces.size()) {
      auto output = joinProbePiece(probePieces[currentProbePiece_], resident);
      ++currentProbePiece_;
      if (output != nullptr) {
        auto const numRows = static_cast<vector_size_t>(output->num_rows());
        return std::make_shared<CudfVector>(
            pool(), outputType_, numRows, std::move(output), resident.stream);
      }
    }

    finishCurrentPartition();
  }

  return nullptr;
}

void CudfPartitionedProbeSpillHashJoinProbe::doNoMoreInput() {
  Operator::noMoreInput();
}

void CudfPartitionedProbeSpillHashJoinProbe::releaseSpillStore() {
  if (spillStoreAcquired_) {
    PartitionedProbeSpillStore::getInstance().release(planNodeId());
    spillStoreAcquired_ = false;
  }
  buildPartitions_.reset();
}

void CudfPartitionedProbeSpillHashJoinProbe::doClose() {
  Operator::close();
  if (currentLoadFuture_.has_value()) {
    std::move(*currentLoadFuture_).get();
    currentLoadFuture_.reset();
  }
  if (prefetchFuture_.has_value()) {
    std::move(*prefetchFuture_).get();
    prefetchFuture_.reset();
  }
  if (slotsInitialized_) {
    slotA_.stream.synchronize();
    slotB_.stream.synchronize();
  }
  resetSlot(slotA_);
  resetSlot(slotB_);
  probePartitions_.clear();
  currentPartition_.reset();
  prefetchedPartition_.reset();
  releaseSpillStore();
  finished_ = true;
}

bool CudfPartitionedProbeSpillHashJoinProbe::isFinished() {
  if (finished_) {
    releaseSpillStore();
  }
  return finished_;
}

} // namespace facebook::velox::cudf_velox
