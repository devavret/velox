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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

int getNumCudaDevices() {
  int numDevices{};
  CUDF_CUDA_TRY(cudaGetDeviceCount(&numDevices));
  return numDevices;
}

int getCurrentCudaDevice() {
  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  return device;
}

CudaEvent& eventForThread() {
  // Intentionally leak per-thread, per-device events to avoid CUDA calls from
  // thread-local destructors after CUDA context teardown.
  thread_local static std::vector<CudaEvent*> events(getNumCudaDevices());
  auto const device = getCurrentCudaDevice();
  VELOX_CHECK_GE(device, 0);
  auto const deviceIndex = static_cast<size_t>(device);
  VELOX_CHECK_LT(deviceIndex, events.size());

  if (events[deviceIndex] == nullptr) {
    events[deviceIndex] = new CudaEvent(cudaEventDisableTiming);
  }
  return *events[deviceIndex];
}

vector_size_t checkedVectorSize(size_t rowCount) {
  VELOX_CHECK_LE(
      rowCount,
      static_cast<size_t>(std::numeric_limits<vector_size_t>::max()),
      "cuDF vector row count exceeds Velox vector size limit");
  return static_cast<vector_size_t>(rowCount);
}

void appendStringColumnBytes(
    const cudf::column_view& column,
    rmm::cuda_stream_view stream,
    std::vector<int64_t>& bytes) {
  if (column.type().id() == cudf::type_id::STRING) {
    bytes.push_back(cudf::strings_column_view(column).chars_size(stream));
    return;
  }

  for (cudf::size_type i = 0; i < column.num_children(); ++i) {
    appendStringColumnBytes(column.child(i), stream, bytes);
  }
}

std::vector<int64_t> stringColumnBytes(
    const cudf::table_view& table,
    rmm::cuda_stream_view stream) {
  std::vector<int64_t> bytes;
  for (const auto& column : table) {
    appendStringColumnBytes(column, stream, bytes);
  }
  return bytes;
}

bool reachesOffset64Threshold(
    const std::vector<int64_t>& runningBytes,
    const std::vector<int64_t>& nextBytes,
    int64_t threshold) {
  VELOX_CHECK_EQ(runningBytes.size(), nextBytes.size());
  for (size_t i = 0; i < runningBytes.size(); ++i) {
    VELOX_CHECK_GE(runningBytes[i], 0);
    VELOX_CHECK_GE(nextBytes[i], 0);
    if (nextBytes[i] >= threshold ||
        runningBytes[i] >= threshold - nextBytes[i]) {
      return true;
    }
  }
  return false;
}
} // namespace

size_t maxBatchRows() {
  const auto configured = CudfConfig::getInstance().batchSizeMaxThreshold;
  if (configured.has_value()) {
    VELOX_CHECK_GT(
        configured.value(), 0, "cuDF max batch size must be positive");
    return static_cast<size_t>(configured.value());
  }
  return static_cast<size_t>(std::numeric_limits<cudf::size_type>::max());
}

std::unique_ptr<cudf::table> concatenateTables(
    std::vector<std::unique_ptr<cudf::table>> tables,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Check for empty vector
  VELOX_CHECK_GT(tables.size(), 0);

  if (tables.size() == 1) {
    return std::move(tables[0]);
  }
  std::vector<cudf::table_view> tableViews;
  tableViews.reserve(tables.size());
  std::transform(
      tables.begin(),
      tables.end(),
      std::back_inserter(tableViews),
      [&](const auto& tbl) { return tbl->view(); });
  return cudf::concatenate(tableViews, stream, mr);
}

std::unique_ptr<cudf::table> makeEmptyTable(TypePtr const& inputType) {
  std::vector<std::unique_ptr<cudf::column>> emptyColumns;
  for (size_t i = 0; i < inputType->size(); ++i) {
    if (auto const& childType = inputType->childAt(i);
        childType->kind() == TypeKind::ROW) {
      auto tbl = makeEmptyTable(childType);
      auto structColumn = std::make_unique<cudf::column>(
          cudf::data_type(cudf::type_id::STRUCT),
          0,
          rmm::device_buffer(),
          rmm::device_buffer(),
          0,
          tbl->release());
      emptyColumns.push_back(std::move(structColumn));
    } else {
      auto emptyColumn = cudf::make_empty_column(
          cudf_velox::veloxToCudfDataType(inputType->childAt(i)));
      emptyColumns.push_back(std::move(emptyColumn));
    }
  }
  return std::make_unique<cudf::table>(std::move(emptyColumns));
}

std::unique_ptr<cudf::table> getConcatenatedTable(
    std::vector<CudfVectorPtr>&& tables,
    const TypePtr& tableType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Check for empty vector
  if (tables.size() == 0) {
    return makeEmptyTable(tableType);
  }

  auto inputStreams = std::vector<rmm::cuda_stream_view>();
  auto tableViews = std::vector<cudf::table_view>();

  inputStreams.reserve(tables.size());
  tableViews.reserve(tables.size());

  for (const auto& table : tables) {
    VELOX_CHECK_NOT_NULL(table);
    tableViews.push_back(table->getTableView());
    inputStreams.push_back(table->stream());
  }

  cudf::detail::join_streams(inputStreams, stream);

  // Even for a single input table we must concatenate (copy) rather than
  // release in-place: the output is owned by `stream` but the input buffer was
  // allocated on a different stream, so releasing it would bind deallocation to
  // the wrong stream.
  auto output = cudf::concatenate(tableViews, stream, mr);

  orderCudfVectorDeallocationsAfterStream(tables, inputStreams, stream);
  // Input tables are deallocated here when 'tables' goes out of scope.
  return output;
}

std::vector<std::unique_ptr<cudf::table>> getConcatenatedTableBatched(
    std::vector<CudfVectorPtr>&& tables,
    const TypePtr& tableType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::vector<std::unique_ptr<cudf::table>> concatTables;
  // Check for empty vector
  if (tables.size() == 0) {
    concatTables.push_back(makeEmptyTable(tableType));
    return concatTables;
  }

  auto inputStreams = std::vector<rmm::cuda_stream_view>();
  auto tableViews = std::vector<cudf::table_view>();

  inputStreams.reserve(tables.size());
  tableViews.reserve(tables.size());

  for (const auto& table : tables) {
    VELOX_CHECK_NOT_NULL(table);
    tableViews.push_back(table->getTableView());
    inputStreams.push_back(table->stream());
  }

  cudf::detail::join_streams(inputStreams, stream);

  std::vector<std::unique_ptr<cudf::table>> outputTables;
  const auto maxRows = maxBatchRows();
  const auto offset64Threshold = cudf::strings::get_offset64_threshold();
  VELOX_CHECK_GT(offset64Threshold, 0);

  size_t startpos = 0;
  size_t runningRows = 0;
  auto runningStringBytes = stringColumnBytes(tableViews.front(), stream);
  std::fill(runningStringBytes.begin(), runningStringBytes.end(), 0);

  for (size_t i = 0; i < tableViews.size(); ++i) {
    const auto numRows = static_cast<size_t>(tableViews[i].num_rows());
    const auto nextStringBytes = stringColumnBytes(tableViews[i], stream);
    const bool exceedsRows =
        numRows > maxRows || runningRows > maxRows - numRows;
    const bool requires64BitStringOffsets = reachesOffset64Threshold(
        runningStringBytes, nextStringBytes, offset64Threshold);

    // Besides the configured row limit, keep BatchConcat from introducing a
    // large-strings column. Passing such a materialized result across the
    // operator boundary can produce incorrect grouping keys. A single input
    // that is already over the threshold is emitted alone.
    if (runningRows > 0 && (exceedsRows || requires64BitStringOffsets)) {
      outputTables.push_back(
          cudf::concatenate(
              std::vector<cudf::table_view>(
                  tableViews.begin() + startpos, tableViews.begin() + i),
              stream,
              mr));
      startpos = i;
      runningRows = 0;
      std::fill(runningStringBytes.begin(), runningStringBytes.end(), 0);
    }

    runningRows += numRows;
    for (size_t column = 0; column < runningStringBytes.size(); ++column) {
      VELOX_CHECK_LE(
          nextStringBytes[column],
          std::numeric_limits<int64_t>::max() - runningStringBytes[column]);
      runningStringBytes[column] += nextStringBytes[column];
    }
  }

  // Flush the final batch [startpos, end).
  if (startpos < tableViews.size()) {
    outputTables.push_back(
        cudf::concatenate(
            std::vector<cudf::table_view>(
                tableViews.begin() + startpos, tableViews.end()),
            stream,
            mr));
  }
  orderCudfVectorDeallocationsAfterStream(tables, inputStreams, stream);

  // Input tables are deallocated here when 'tables' goes out of scope.
  return outputTables;
}

std::vector<CudfVectorPtr> getConcatenatedCudfVectorsBatched(
    memory::MemoryPool* pool,
    std::vector<CudfVectorPtr>&& vectors,
    const TypePtr& tableType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_NOT_NULL(pool);

  std::vector<CudfVectorPtr> outputVectors;
  if (tableType->size() > 0) {
    auto tables =
        getConcatenatedTableBatched(std::move(vectors), tableType, stream, mr);
    outputVectors.reserve(tables.size());
    for (auto& table : tables) {
      VELOX_CHECK_NOT_NULL(table);
      const auto rowCount =
          checkedVectorSize(static_cast<size_t>(table->num_rows()));
      outputVectors.push_back(
          std::make_shared<CudfVector>(
              pool, tableType, rowCount, std::move(table), stream));
    }
    return outputVectors;
  }

  size_t remainingRows = 0;
  for (const auto& vector : vectors) {
    VELOX_CHECK_NOT_NULL(vector);
    VELOX_CHECK_EQ(vector->getTableView().num_columns(), 0);
    const auto rowCount = static_cast<size_t>(vector->size());
    VELOX_CHECK_LE(
        rowCount,
        std::numeric_limits<size_t>::max() - remainingRows,
        "zero-column cuDF vector row count overflow");
    remainingRows += rowCount;
  }

  const auto maxRows = maxBatchRows();
  do {
    const auto chunkRows = std::min(remainingRows, maxRows);
    outputVectors.push_back(
        std::make_shared<CudfVector>(
            pool,
            tableType,
            checkedVectorSize(chunkRows),
            makeEmptyTable(tableType),
            stream));
    remainingRows -= chunkRows;
  } while (remainingRows > 0);

  return outputVectors;
}

void streamsWaitForStream(
    CudaEvent& event,
    std::span<const rmm::cuda_stream_view> streams,
    rmm::cuda_stream_view stream) {
  event.recordFrom(stream);
  for (const auto& strm : streams) {
    event.waitOn(strm);
  }
}

CudaEvent::CudaEvent(unsigned int flags) {
  cudaEvent_t ev{};
  CUDF_CUDA_TRY(cudaEventCreateWithFlags(&ev, flags));
  event_ = ev;
}

CudaEvent::~CudaEvent() {
  if (event_ != nullptr) {
    cudaEventDestroy(event_);
    event_ = nullptr;
  }
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
  other.event_ = nullptr;
}

const CudaEvent& CudaEvent::recordFrom(rmm::cuda_stream_view stream) const {
  CUDF_CUDA_TRY(cudaEventRecord(event_, stream.value()));
  return *this;
}

const CudaEvent& CudaEvent::waitOn(rmm::cuda_stream_view stream) const {
  CUDF_CUDA_TRY(cudaStreamWaitEvent(stream.value(), event_, 0));
  return *this;
}

std::string getBaseFunctionName(const std::string& fullName) {
  auto pos = fullName.rfind('.');
  return pos == std::string::npos ? fullName : fullName.substr(pos + 1);
}

std::string stripFunctionPrefix(
    const std::string& name,
    const std::string& prefix) {
  auto base = getBaseFunctionName(name);
  if (!prefix.empty() && base.find(prefix) == 0) {
    return base.substr(prefix.size());
  }
  return base;
}

void orderCudfVectorDeallocationsAfterStream(
    std::span<const CudfVectorPtr> vectors,
    std::span<const rmm::cuda_stream_view> inputStreams,
    rmm::cuda_stream_view stream) {
  bool allRebound = true;
  for (const auto& vector : vectors) {
    VELOX_CHECK_NOT_NULL(vector);
    allRebound &= vector->rebindStream(stream);
  }

  if (!allRebound) {
    streamsWaitForStream(eventForThread(), inputStreams, stream);
  }
}

} // namespace facebook::velox::cudf_velox
