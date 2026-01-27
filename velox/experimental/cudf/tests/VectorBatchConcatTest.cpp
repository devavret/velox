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
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

#include <cuda_runtime_api.h>

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

class VectorBatchConcatTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();

    int deviceCount = 0;
    auto cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "CUDA device not available in this environment";
    }

    // Some test environments don't support cudaMallocAsync.
    cudf_velox::CudfConfig::getInstance().memoryResource = "cuda";
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    HiveConnectorTestBase::TearDown();
  }
};

TEST_F(VectorBatchConcatTest, coalesceOvershootNoSplit) {
  std::unordered_map<std::string, std::string> config{
      {"velox.cudf.gpu_batch_size_rows", "10"}};
  core::QueryConfig queryConfig(std::move(config));

  auto makeCudf = [&](vector_size_t rows) -> RowVectorPtr {
    auto c0 = makeFlatVector<int32_t>(rows, folly::identity);
    auto input = makeRowVector({c0});
    auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
    auto tbl = cudf_velox::with_arrow::toCudfTable(input, pool_.get(), stream);
    stream.synchronize();
    return std::make_shared<cudf_velox::CudfVector>(
        pool_.get(), input->type(), rows, std::move(tbl), stream);
  };

  auto const tableType = makeRowVector({makeFlatVector<int32_t>(1)})->type();

  // Create a minimal task + driver context to construct the operator.
  auto planFragment = exec::test::PlanBuilder()
                          .values({makeRowVector({makeFlatVector<int32_t>(1)})})
                          .planFragment();
  auto queryCtx = core::QueryCtx::create(nullptr, std::move(queryConfig));
  auto task = exec::Task::create(
      "cudf-batch-concat-test",
      std::move(planFragment),
      0,
      queryCtx,
      exec::Task::ExecutionMode::kSerial,
      exec::Consumer{});
  exec::DriverCtx driverCtx(task, 0, 0, exec::kUngroupedGroupId, 0);

  cudf_velox::CudfBatchConcat op(
      /*operatorId*/ 0,
      &driverCtx,
      std::dynamic_pointer_cast<const RowType>(tableType),
      "batch-concat");

  op.addInput(makeCudf(3));
  EXPECT_EQ(op.getOutput(), nullptr);

  op.addInput(makeCudf(7));
  auto out1 = op.getOutput();
  ASSERT_NE(out1, nullptr);
  EXPECT_EQ(out1->size(), 10);

  op.addInput(makeCudf(20));
  auto out2 = op.getOutput();
  ASSERT_NE(out2, nullptr);
  EXPECT_EQ(out2->size(), 20);
}

