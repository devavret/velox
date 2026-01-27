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

#include "velox/experimental/cudf/exec/CudfBatchConcat.h"

#include "velox/experimental/cudf/exec/Utilities.h"

#include "velox/common/base/Exceptions.h"

#include <limits>

namespace facebook::velox::cudf_velox {

namespace {
constexpr vector_size_t kDefaultGpuBatchSizeRows = 100'000;
constexpr const char* kGpuBatchSizeRows = "velox.cudf.gpu_batch_size_rows";
}

vector_size_t CudfBatchConcat::preferredGpuBatchSizeRows(
    const core::QueryConfig& queryConfig) {
  const auto batchSize = queryConfig.get<int32_t>(
      kGpuBatchSizeRows, kDefaultGpuBatchSizeRows);
  VELOX_CHECK_GT(batchSize, 0, "{} must be > 0", kGpuBatchSizeRows);
  VELOX_CHECK_LE(
      batchSize,
      std::numeric_limits<vector_size_t>::max(),
      "{} must be <= max(vector_size_t)",
      kGpuBatchSizeRows);
  return static_cast<vector_size_t>(batchSize);
}

CudfBatchConcat::CudfBatchConcat(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    RowTypePtr outputType,
    std::string planNodeId)
    : exec::Operator(
          driverCtx,
          std::move(outputType),
          operatorId,
          planNodeId,
          "CudfBatchConcat"),
      NvtxHelper(
          nvtx3::rgb{255, 255, 255}, // White
          operatorId,
          fmt::format("[{}]", planNodeId)),
      targetBatchSizeRows_{preferredGpuBatchSizeRows(driverCtx->queryConfig())} {}

void CudfBatchConcat::addInput(RowVectorPtr input) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (!input || input->size() == 0) {
    return;
  }
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK(cudfInput, "Input must be a CudfVector");
  bufferedRows_ += cudfInput->size();
  inputs_.push_back(std::move(cudfInput));
}

RowVectorPtr CudfBatchConcat::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (inputs_.empty()) {
    return nullptr;
  }

  // Only emit when we have enough rows, unless we're flushing at end-of-input.
  if (!noMoreInput_ && bufferedRows_ < targetBatchSizeRows_) {
    return nullptr;
  }

  // Take a prefix of inputs until we reach targetBatchSizeRows_ (inclusive),
  // or until we exhaust inputs (flush). We do NOT split batches to fit exactly.
  std::vector<CudfVectorPtr> selectedInputs;
  selectedInputs.reserve(inputs_.size());

  vector_size_t totalSize = 0;
  while (!inputs_.empty()) {
    auto front = std::move(inputs_.front());
    inputs_.pop_front();
    VELOX_CHECK_NOT_NULL(front);
    totalSize += front->size();
    selectedInputs.push_back(std::move(front));

    if (totalSize >= targetBatchSizeRows_) {
      break;
    }
    if (noMoreInput_ && inputs_.empty()) {
      break;
    }
  }
  bufferedRows_ -= totalSize;

  if (selectedInputs.empty()) {
    return nullptr;
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  auto resultTable = getConcatenatedTable(selectedInputs, outputType_, stream);
  VELOX_CHECK_NOT_NULL(resultTable);

  const auto size = static_cast<vector_size_t>(resultTable->num_rows());
  if (size == 0) {
    return nullptr;
  }

  return std::make_shared<CudfVector>(
      pool(), outputType_, size, std::move(resultTable), stream);
}

void CudfBatchConcat::noMoreInput() {
  exec::Operator::noMoreInput();
}

} // namespace facebook::velox::cudf_velox

