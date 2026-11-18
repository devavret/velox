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
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <fmt/format.h>

#include <string>
#include <tuple>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox::exec::test;

namespace encoded = facebook::velox::cudf_velox::connector::encoded;

namespace {

constexpr const char* kConnectorId = "test-encoded";
constexpr const char* kTable = "t";
constexpr int32_t kRows = 10'000;
constexpr int32_t kSegmentRows = 1'024;
constexpr int32_t kSegmentsPerSplit = 3;

/// Scans of an encoded table whose filters read columns the scan does not
/// return. The parameter keeps the table in pinned host memory, staged to the
/// device batch by batch, instead of device memory.
class EncodedScanTest
    : public CudfHiveConnectorTestBase,
      public testing::WithParamInterface<std::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    CudfHiveConnectorTestBase::SetUp();
    facebook::velox::connector::ConnectorRegistry::global().insert(
        kConnectorId,
        std::make_shared<encoded::EncodedConnector>(kConnectorId));
    // registerCudf() in the base SetUp() reset the adapters.
    encoded::registerEncodedTableScanAdapter(std::get<1>(GetParam()));
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
    file_.reset();
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
          row % 5 == 0 ? fmt::format("x{} special y requests", row)
                       : fmt::format("comment {}", row));
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

  void loadTable() {
    cudf_velox::ensureCudaContextForThread();
    auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
    const auto mr = cudf_velox::get_output_mr();
    static rmm::mr::pinned_host_memory_resource pinnedHost;
    const bool hostResident = std::get<0>(GetParam());
    encoded::EncodedColumnCache::instance().loadTable(
        kTable,
        {file_->getPath()},
        kSegmentRows,
        stream,
        mr,
        hostResident ? rmm::device_async_resource_ref(pinnedHost) : mr,
        hostResident);
    stream.sync();
  }

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
  splits() const {
    return encoded::makeEncodedSplits(kConnectorId, kTable, kSegmentsPerSplit);
  }

  // A scan returning 'outputType' whose filters may read any column of
  // 'dataColumns', by default the full schema of the table.
  core::PlanNodePtr scan(
      const RowTypePtr& outputType,
      const std::vector<std::string>& subfieldFilters,
      const std::string& remainingFilter,
      const RowTypePtr& dataColumns = nullptr) {
    return PlanBuilder(pool())
        .startTableScan(kConnectorId)
        .tableName(kTable)
        .outputType(outputType)
        .dataColumns(dataColumns ? dataColumns : asRowType(data_->type()))
        .subfieldFilters(subfieldFilters)
        .remainingFilter(remainingFilter)
        .endTableScan()
        .capturePlanNodeId(scanId_)
        .planNode();
  }

  // Checks that the scan decoded several batches and that each dropped
  // 'filterOnlyColumns' filter columns instead of gathering them.
  void expectFilterOnlyColumns(
      const std::shared_ptr<facebook::velox::exec::Task>& task,
      int64_t filterOnlyColumns) {
    const auto stats = facebook::velox::exec::toPlanStats(task->taskStats());
    const auto& customStats = stats.at(scanId_).customStats;
    ASSERT_EQ(customStats.count("encodedFilterOnlyColumns"), 1);
    const auto& metric = customStats.at("encodedFilterOnlyColumns");
    EXPECT_GE(metric.count, 2);
    EXPECT_EQ(metric.min, filterOnlyColumns);
    EXPECT_EQ(metric.max, filterOnlyColumns);
  }

  void expectDictionaryFilters(
      const std::shared_ptr<facebook::velox::exec::Task>& task,
      int64_t eligibleColumns) {
    const auto stats = facebook::velox::exec::toPlanStats(task->taskStats());
    const auto& metric =
        stats.at(scanId_).customStats.at("encodedDictionaryFilterColumns");
    const auto expected = std::get<1>(GetParam()) ? eligibleColumns : 0;
    EXPECT_GE(metric.count, 2);
    EXPECT_EQ(metric.min, expected);
    EXPECT_EQ(metric.max, expected);
  }

  RowVectorPtr data_;
  std::shared_ptr<TempFilePath> file_;
  core::PlanNodeId scanId_;
  int32_t savedBatchSizeMinThreshold_{0};
};

// The plain string column c and the dictionary column f filter but are not
// returned.
TEST_P(EncodedScanTest, filterOnlyColumns) {
  auto plan = scan(
      ROW({"k", "v"}, {INTEGER(), BIGINT()}),
      {"f = 'b'"},
      "c like '%special%requests%'");
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .splits(splits())
                  .maxDrivers(2)
                  .assertResults(
                      "SELECT k, v FROM tmp "
                      "WHERE f = 'b' AND c LIKE '%special%requests%'");
  expectFilterOnlyColumns(task, 2);
  expectDictionaryFilters(task, 1);
}

// c filters and is returned, so it is gathered to the passing rows; f only
// filters.
TEST_P(EncodedScanTest, filterColumnAlsoReturned) {
  auto plan = scan(
      ROW({"k", "c"}, {INTEGER(), VARCHAR()}),
      {"f = 'b'"},
      "c like '%special%requests%'");
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .splits(splits())
                  .maxDrivers(2)
                  .assertResults(
                      "SELECT k, c FROM tmp "
                      "WHERE f = 'b' AND c LIKE '%special%requests%'");
  expectFilterOnlyColumns(task, 1);
  expectDictionaryFilters(task, 1);
}

// A remaining filter over three columns, none of them returned.
TEST_P(EncodedScanTest, remainingFilterOnlyColumns) {
  auto plan = scan(
      ROW({"f"}, {VARCHAR()}),
      {},
      "v > cast(k as bigint) * 2 and c like '%special%'");
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .splits(splits())
                  .maxDrivers(2)
                  .assertResults(
                      "SELECT f FROM tmp "
                      "WHERE v > CAST(k AS BIGINT) * 2 AND c LIKE '%special%'");
  expectFilterOnlyColumns(task, 3);
}

// A scan returning no columns: only the passing row count leaves it.
TEST_P(EncodedScanTest, noOutputColumns) {
  auto plan = PlanBuilder(pool())
                  .startTableScan(kConnectorId)
                  .tableName(kTable)
                  .outputType(ROW({}, {}))
                  .dataColumns(asRowType(data_->type()))
                  .remainingFilter("c like '%special%requests%'")
                  .endTableScan()
                  .capturePlanNodeId(scanId_)
                  .singleAggregation({}, {"count(1)"})
                  .planNode();
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .splits(splits())
                  .assertResults(
                      "SELECT count(*) FROM tmp "
                      "WHERE c LIKE '%special%requests%'");
  expectFilterOnlyColumns(task, 1);
}

TEST_P(EncodedScanTest, dictionaryOnlyOutput) {
  auto plan = scan(ROW({"k", "f"}, {INTEGER(), VARCHAR()}), {"f = 'b'"}, "");
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .splits(splits())
                  .maxDrivers(2)
                  .assertResults("SELECT k, f FROM tmp WHERE f = 'b'");
  expectFilterOnlyColumns(task, 0);
  expectDictionaryFilters(task, 1);
}

TEST_P(EncodedScanTest, dictionaryOnlyZeroOutput) {
  auto plan = PlanBuilder(pool())
                  .startTableScan(kConnectorId)
                  .tableName(kTable)
                  .outputType(ROW({}, {}))
                  .dataColumns(asRowType(data_->type()))
                  .subfieldFilters({"f = 'b'"})
                  .endTableScan()
                  .capturePlanNodeId(scanId_)
                  .singleAggregation({}, {"count(1)"})
                  .planNode();
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .splits(splits())
                  .assertResults("SELECT count(*) FROM tmp WHERE f = 'b'");
  expectFilterOnlyColumns(task, 1);
  expectDictionaryFilters(task, 1);
}

TEST_P(EncodedScanTest, dictionaryAlsoUsedByRemaining) {
  auto plan = scan(ROW({"k"}, {INTEGER()}), {"f = 'b'"}, "f like '%b%'");
  auto task =
      AssertQueryBuilder(plan, duckDbQueryRunner_)
          .splits(splits())
          .maxDrivers(2)
          .assertResults("SELECT k FROM tmp WHERE f = 'b' AND f LIKE '%b%'");
  expectFilterOnlyColumns(task, 1);
  expectDictionaryFilters(task, 0);
}

TEST_P(EncodedScanTest, constantFalseRemaining) {
  auto plan = scan(ROW({"k"}, {INTEGER()}), {"f = 'b'"}, "false");
  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .splits(splits())
      .maxDrivers(2)
      .assertResults("SELECT k FROM tmp WHERE false");
}

// A filter column the encoded table lacks fails when the scan is created.
TEST_P(EncodedScanTest, missingFilterColumn) {
  auto plan = scan(
      ROW({"k"}, {INTEGER()}),
      {"z = 'q'"},
      "",
      ROW({"k", "c", "f", "v", "z"},
          {INTEGER(), VARCHAR(), VARCHAR(), BIGINT(), VARCHAR()}));
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).splits(splits()).copyResults(pool()),
      "Filter column z not found in encoded table t");
}

INSTANTIATE_TEST_SUITE_P(
    EncodedScanTest,
    EncodedScanTest,
    testing::Combine(testing::Bool(), testing::Bool()),
    [](const testing::TestParamInfo<std::tuple<bool, bool>>& info)
        -> std::string {
      return std::string(std::get<0>(info.param) ? "pinnedHost" : "device") +
          (std::get<1>(info.param) ? "Dictionary" : "Normal");
    });

} // namespace
