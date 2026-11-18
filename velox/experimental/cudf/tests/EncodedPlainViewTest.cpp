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
#include "velox/experimental/cudf/connectors/encoded/EncodedColumnCache.h"
#include "velox/experimental/cudf/connectors/encoded/EncodedConnector.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <fmt/format.h>

#include <atomic>
#include <exception>
#include <string>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox::exec::test;

namespace encoded = facebook::velox::cudf_velox::connector::encoded;

namespace {

constexpr const char* kConnectorId = "test-encoded-plain-views";
constexpr const char* kTable = "t";
constexpr int32_t kRows = 10'000;
constexpr int32_t kSegmentRows = 1'024;
constexpr int32_t kSegmentsPerSplit = 3;

// Where the table is held: device memory, pinned host memory, or pinned host
// memory with LZ4-compressed plain string characters.
enum class Storage { kDevice, kHost, kHostLz4 };

/// Scans with plain filter views enabled. Filters on the plain column 'c'
/// borrow the values a staged batch owns; device-resident batches, which are
/// not staged, fall back to decoding.
class EncodedPlainViewTest : public CudfHiveConnectorTestBase,
                             public testing::WithParamInterface<Storage> {
 protected:
  void SetUp() override {
    CudfHiveConnectorTestBase::SetUp();
    facebook::velox::connector::ConnectorRegistry::global().insert(
        kConnectorId,
        std::make_shared<encoded::EncodedConnector>(kConnectorId));
    // registerCudf() in the base SetUp() reset the adapters.
    encoded::registerEncodedTableScanAdapter(
        /*dictionaryFilters=*/false, /*plainFilterViews=*/true);
    // Small batches, so each scan decodes and filters several.
    auto& config = cudf_velox::CudfConfig::getInstance();
    savedBatchSizeMinThreshold_ = config.batchSizeMinThreshold;
    config.batchSizeMinThreshold = 2'048;

    data_ = makeData();
    createDuckDbTable({data_});
    file_ = TempFilePath::create();
    writeToFile(file_->getPath(), data_);
    loadTable();
  }

  void TearDown() override {
    encoded::EncodedColumnCache::instance().clear();
    cudf_velox::CudfConfig::getInstance().batchSizeMinThreshold =
        savedBatchSizeMinThreshold_;
    facebook::velox::connector::ConnectorRegistry::global().erase(kConnectorId);
    data_.reset();
    CudfHiveConnectorTestBase::TearDown();
  }

  // k: unique keys. c: one distinct string per row, so it is stored plain;
  // every fifth contains 'special ... requests'. f: three values, stored as a
  // dictionary. v: 3 * k. Encoded tables hold no nulls.
  RowVectorPtr makeData() {
    std::vector<std::string> comments;
    std::vector<std::string> flags;
    comments.reserve(kRows);
    flags.reserve(kRows);
    for (int32_t row = 0; row < kRows; ++row) {
      comments.push_back(
          row % 5 == 0
              ? fmt::format("x{} special y requests about the deposits", row)
              : fmt::format("comment {} about the final deposits", row));
      flags.push_back(std::string(1, static_cast<char>('a' + row % 3)));
    }
    return makeRowVector(
        {"k", "c", "f", "v"},
        {makeFlatVector<int32_t>(kRows, [](auto row) { return row; }),
         makeFlatVector<std::string>(comments),
         makeFlatVector<std::string>(flags),
         makeFlatVector<int64_t>(
             kRows, [](auto row) { return int64_t{3} * row; })});
  }

  bool hostResident() const {
    return GetParam() != Storage::kDevice;
  }

  void loadTable() {
    cudf_velox::ensureCudaContextForThread();
    auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
    const auto mr = cudf_velox::get_output_mr();
    static rmm::mr::pinned_host_memory_resource pinnedHost;
    encoded::EncodedColumnCache::instance().loadTable(
        kTable,
        {file_->getPath()},
        kSegmentRows,
        stream,
        mr,
        hostResident() ? rmm::device_async_resource_ref(pinnedHost) : mr,
        hostResident(),
        /*deltaEncodeSortedSegments=*/false,
        /*compressPlainStrings=*/GetParam() == Storage::kHostLz4);
    stream.sync();
    if (GetParam() == Storage::kHostLz4) {
      ASSERT_TRUE(table().column("c").compressedChars != nullptr);
    }
  }

  const encoded::EncodedTable& table() const {
    return encoded::EncodedColumnCache::instance().table(kTable);
  }

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
  splits() const {
    return encoded::makeEncodedSplits(kConnectorId, kTable, kSegmentsPerSplit);
  }

  core::PlanNodePtr scan(
      const RowTypePtr& outputType,
      const std::string& remainingFilter) {
    return PlanBuilder(pool())
        .startTableScan(kConnectorId)
        .tableName(kTable)
        .outputType(outputType)
        .dataColumns(asRowType(data_->type()))
        .remainingFilter(remainingFilter)
        .endTableScan()
        .capturePlanNodeId(scanId_)
        .planNode();
  }

  // Each batch borrowed 'c' when staged, and decoded it otherwise.
  void expectBorrowed(
      const std::shared_ptr<facebook::velox::exec::Task>& task) {
    const auto stats = facebook::velox::exec::toPlanStats(task->taskStats());
    const auto& customStats = stats.at(scanId_).customStats;
    ASSERT_EQ(customStats.count("encodedBorrowedFilterColumns"), 1);
    const auto& metric = customStats.at("encodedBorrowedFilterColumns");
    const int64_t expected = hostResident() ? 1 : 0;
    EXPECT_GE(metric.count, 2);
    EXPECT_EQ(metric.min, expected);
    EXPECT_EQ(metric.max, expected);
  }

  RowVectorPtr data_;
  std::shared_ptr<TempFilePath> file_;
  core::PlanNodeId scanId_;
  int32_t savedBatchSizeMinThreshold_{0};
};

// The plain high-cardinality column only filters.
TEST_P(EncodedPlainViewTest, filterOnlyPlainStrings) {
  auto task = AssertQueryBuilder(
                  scan(
                      ROW({"k", "v"}, {INTEGER(), BIGINT()}),
                      "c like '%special%requests%'"),
                  duckDbQueryRunner_)
                  .splits(splits())
                  .maxDrivers(2)
                  .assertResults(
                      "SELECT k, v FROM tmp WHERE c LIKE '%special%requests%'");
  expectBorrowed(task);
}

// The borrowed filter column is also returned: the retained rows are gathered
// into an owned output column.
TEST_P(EncodedPlainViewTest, borrowedFilterColumnAlsoReturned) {
  auto task = AssertQueryBuilder(
                  scan(
                      ROW({"k", "c"}, {INTEGER(), VARCHAR()}),
                      "c like '%special%requests%'"),
                  duckDbQueryRunner_)
                  .splits(splits())
                  .maxDrivers(2)
                  .assertResults(
                      "SELECT k, c FROM tmp WHERE c LIKE '%special%requests%'");
  expectBorrowed(task);
}

// No row passes.
TEST_P(EncodedPlainViewTest, emptyResult) {
  auto task =
      AssertQueryBuilder(
          scan(
              ROW({"k", "c"}, {INTEGER(), VARCHAR()}),
              "c like '%no such text%'"),
          duckDbQueryRunner_)
          .splits(splits())
          .maxDrivers(2)
          .assertResults("SELECT k, c FROM tmp WHERE c LIKE '%no such text%'");
  expectBorrowed(task);
}

// The filter is NULL for half the rows, which drop out like false ones.
TEST_P(EncodedPlainViewTest, nullFilterResults) {
  auto task =
      AssertQueryBuilder(
          scan(
              ROW({"k", "c"}, {INTEGER(), VARCHAR()}),
              "if(k < 5000, c, cast(null as varchar)) like '%special%'"),
          duckDbQueryRunner_)
          .splits(splits())
          .maxDrivers(2)
          .assertResults(
              "SELECT k, c FROM tmp WHERE "
              "(CASE WHEN k < 5000 THEN c ELSE NULL END) LIKE '%special%'");
  expectBorrowed(task);
}

// A compressed column is borrowed only after its decompression is verified: a
// block whose metadata expects one more byte fails the scan.
TEST_P(EncodedPlainViewTest, verifiesCompressedStagingBeforeBorrowing) {
  if (GetParam() != Storage::kHostLz4) {
    GTEST_SKIP() << "Needs compressed plain strings";
  }
  auto& column = const_cast<encoded::EncodedColumn&>(table().column("c"));
  auto& compressed = *column.compressedChars;
  ASSERT_NE(compressed.numChars % compressed.blockBytes, 0);
  compressed.numChars += 1;
  try {
    AssertQueryBuilder(
        scan(ROW({"k"}, {INTEGER()}), "c like '%special%requests%'"))
        .splits(splits())
        .copyResults(pool());
    ADD_FAILURE() << "Expected the size mismatch to fail the scan";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find("decompressed to"), std::string::npos)
        << e.what();
  }
  compressed.numChars -= 1;
}

// An error after the filter and output work is queued unwinds decodeBatch(),
// which then waits for that work before the staged batch it reads is
// destroyed. Run under compute-sanitizer to check the unwind for races. The
// injection point is compiled out of release builds.
TEST_P(EncodedPlainViewTest, errorAfterQueuedWork) {
#ifdef NDEBUG
  GTEST_SKIP() << "TestValue injection points need a debug build";
#else
  common::testutil::TestValue::enable();
  std::atomic<int32_t> injections{0};
  SCOPED_TESTVALUE_SET(
      "CudfEncodedScan::decodeBatch::beforeSync",
      std::function<void(int64_t*)>([&](int64_t*) {
        if (injections.fetch_add(1, std::memory_order_relaxed) == 0) {
          VELOX_FAIL("Injected decode failure");
        }
      }));
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(scan(
                             ROW({"k", "c"}, {INTEGER(), VARCHAR()}),
                             "c like '%special%requests%'"))
          .splits(splits())
          .copyResults(pool()),
      "Injected decode failure");
  // Later scans are unaffected.
  AssertQueryBuilder(
      scan(
          ROW({"k", "c"}, {INTEGER(), VARCHAR()}),
          "c like '%special%requests%'"),
      duckDbQueryRunner_)
      .splits(splits())
      .assertResults("SELECT k, c FROM tmp WHERE c LIKE '%special%requests%'");
#endif
}

INSTANTIATE_TEST_SUITE_P(
    EncodedPlainViewTest,
    EncodedPlainViewTest,
    testing::Values(Storage::kDevice, Storage::kHost, Storage::kHostLz4),
    [](const testing::TestParamInfo<Storage>& info) -> std::string {
      switch (info.param) {
        case Storage::kDevice:
          return "device";
        case Storage::kHost:
          return "pinnedHost";
        case Storage::kHostLz4:
          return "pinnedHostLz4";
      }
      return "unknown";
    });

} // namespace
