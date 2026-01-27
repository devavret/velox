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

#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/QueryConfig.h"
#include "velox/exec/Operator.h"

#include <deque>

namespace facebook::velox::cudf_velox {

/// Operator that coalesces small GPU-resident CudfVector batches into larger
/// batches, to improve downstream cuDF kernel efficiency. Intended to be
/// inserted by the cuDF driver adapter (e.g., after LocalExchange).
class CudfBatchConcat : public exec::Operator, public NvtxHelper {
 public:
  CudfBatchConcat(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      RowTypePtr outputType,
      std::string planNodeId);

  bool needsInput() const override {
    return !noMoreInput_ && !outputReady();
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  void noMoreInput() override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return noMoreInput_ && inputs_.empty();
  }

  std::string toString() const override {
    return "CudfBatchConcat";
  }

 private:
  static vector_size_t preferredGpuBatchSizeRows(
      const core::QueryConfig& queryConfig);

  bool outputReady() const {
    return !inputs_.empty() &&
        (noMoreInput_ || bufferedRows_ >= targetBatchSizeRows_);
  }

  const vector_size_t targetBatchSizeRows_;
  std::deque<CudfVectorPtr> inputs_;
  vector_size_t bufferedRows_{0};
};

} // namespace facebook::velox::cudf_velox
