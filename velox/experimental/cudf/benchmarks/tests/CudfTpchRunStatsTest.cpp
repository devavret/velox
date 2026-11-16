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

#include "velox/experimental/cudf/benchmarks/CudfTpchRunStats.h"

#include <folly/json.h>
#include <gtest/gtest.h>

namespace facebook::velox::cudf_velox {
namespace {

// A task of two pipelines: a scan, and a local exchange feeding a group-by
// that spilled. The group-by's spilled bytes do not fit a double exactly.
constexpr int64_t kSpilledBytes{(int64_t{1} << 53) + 1};

velox::exec::TaskStats makeTaskStats() {
  velox::exec::TaskStats stats;
  stats.numTotalSplits = 16;
  stats.numFinishedSplits = 16;
  stats.memoryReclaimCount = 3;
  stats.memoryReclaimMs = 1'234;
  stats.pipelineStats.emplace_back(
      /*_inputPipeline=*/true, /*_outputPipeline=*/false);
  stats.pipelineStats.emplace_back(
      /*_inputPipeline=*/false, /*_outputPipeline=*/true);

  velox::exec::OperatorStats scan(0, 0, "0", "CudfEncodedScan");
  scan.numDrivers = 4;
  scan.outputPositions = 5'999'989'709;
  stats.pipelineStats[0].operatorStats.push_back(scan);

  velox::exec::OperatorStats exchange(0, 1, "6", "CudfLocalExchange");
  stats.pipelineStats[1].operatorStats.push_back(exchange);

  velox::exec::OperatorStats groupby(1, 1, "7", "CudfGroupby");
  groupby.numDrivers = 4;
  groupby.inputPositions = 5'999'989'709;
  groupby.addInputTiming.count = 358;
  groupby.addInputTiming.wallNanos = 31'000'000'123;
  groupby.addInputTiming.cpuNanos = 30'000'000'456;
  groupby.runtimeStats["cudfGroupbySpilledBytes"] = RuntimeMetric(
      kSpilledBytes, 12, 1, kSpilledBytes - 11, RuntimeCounter::Unit::kBytes);
  groupby.runtimeStats["cudfGroupbySpilledLeaves"] =
      RuntimeMetric(12, 12, 1, 1);
  groupby.runtimeStats["memoryArbitrationWallNanos"] = RuntimeMetric(
      7'000'000'001, 2, 1, 7'000'000'000, RuntimeCounter::Unit::kNanos);
  stats.pipelineStats[1].operatorStats.push_back(groupby);

  velox::exec::DriverStats driver;
  driver.runtimeStats[std::string(velox::exec::DriverStats::kTotalPauseTime)] =
      RuntimeMetric(2'000'000, RuntimeCounter::Unit::kNanos);
  stats.pipelineStats[1].driverStats.push_back(driver);
  stats.pipelineStats[1].driverStats.push_back(driver);
  return stats;
}

TEST(CudfTpchRunStatsTest, operatorsKeepIdsAndExactCounters) {
  const auto json = CudfTpchRunStats::toJson(makeTaskStats());
  EXPECT_EQ(json["memory_reclaim_count"].asInt(), 3);
  EXPECT_EQ(json["memory_reclaim_ms"].asInt(), 1'234);
  EXPECT_EQ(json["num_finished_splits"].asInt(), 16);

  // Copies, not references into 'json': some compilers warn about a
  // reference bound to the result of a lookup with a temporary key.
  const auto pipelines = json["pipelines"];
  ASSERT_EQ(pipelines.size(), 2);
  EXPECT_TRUE(pipelines[0]["input_pipeline"].asBool());
  EXPECT_FALSE(pipelines[0]["output_pipeline"].asBool());
  ASSERT_EQ(pipelines[0]["operators"].size(), 1);
  EXPECT_EQ(
      pipelines[0]["operators"][0]["output_positions"].asInt(), 5'999'989'709);

  const auto operators = pipelines[1]["operators"];
  ASSERT_EQ(operators.size(), 2);
  const auto groupby = operators[1];
  EXPECT_EQ(groupby["operator_id"].asInt(), 1);
  EXPECT_EQ(groupby["pipeline_id"].asInt(), 1);
  EXPECT_EQ(groupby["plan_node_id"].asString(), "7");
  EXPECT_EQ(groupby["operator_type"].asString(), "CudfGroupby");
  EXPECT_EQ(groupby["num_drivers"].asInt(), 4);
  EXPECT_EQ(groupby["input_positions"].asInt(), 5'999'989'709);
  EXPECT_EQ(groupby["add_input"]["count"].asInt(), 358);
  EXPECT_EQ(groupby["add_input"]["wall_nanos"].asInt(), 31'000'000'123);
  EXPECT_EQ(groupby["add_input"]["cpu_nanos"].asInt(), 30'000'000'456);
  EXPECT_EQ(groupby["disk_spill"]["spilled_files"].asInt(), 0);

  const auto spilledBytes = groupby["runtime_stats"]["cudfGroupbySpilledBytes"];
  EXPECT_TRUE(spilledBytes["sum"].isInt());
  EXPECT_EQ(spilledBytes["sum"].asInt(), kSpilledBytes);
  EXPECT_EQ(spilledBytes["count"].asInt(), 12);
  EXPECT_EQ(spilledBytes["min"].asInt(), 1);
  EXPECT_EQ(spilledBytes["max"].asInt(), kSpilledBytes - 11);
  EXPECT_EQ(spilledBytes["unit"].asString(), "bytes");
  EXPECT_EQ(
      groupby["runtime_stats"]["cudfGroupbySpilledLeaves"]["unit"].asString(),
      "none");
  EXPECT_EQ(
      groupby["runtime_stats"]["memoryArbitrationWallNanos"]["unit"].asString(),
      "nanos");

  const auto drivers = pipelines[1]["drivers"];
  ASSERT_EQ(drivers.size(), 2);
  EXPECT_EQ(
      drivers[0]["runtime_stats"]["totalDriverPauseWallNanos"]["sum"].asInt(),
      2'000'000);
}

TEST(CudfTpchRunStatsTest, serializedTextKeepsEveryDigit) {
  const auto json = CudfTpchRunStats::toJson(makeTaskStats());
  const auto text = CudfTpchRunStats::serialize(json);
  // 2^53 + 1, which a reader going through doubles would turn into 2^53.
  EXPECT_NE(text.find("9007199254740993"), std::string::npos);
  EXPECT_TRUE(folly::parseJson(text) == json);
}

TEST(CudfTpchRunStatsTest, runtimeStatSums) {
  const auto stats = makeTaskStats();
  EXPECT_EQ(
      CudfTpchRunStats::operatorRuntimeStatSum(
          stats, "cudfGroupbySpilledBytes"),
      kSpilledBytes);
  EXPECT_EQ(
      CudfTpchRunStats::operatorRuntimeStatSum(stats, "cudfJoinSpilledBytes"),
      0);
  EXPECT_EQ(
      CudfTpchRunStats::driverRuntimeStatSum(
          stats, velox::exec::DriverStats::kTotalPauseTime),
      4'000'000);
  EXPECT_EQ(
      CudfTpchRunStats::driverRuntimeStatSum(stats, "cudfGroupbySpilledBytes"),
      0);
}

} // namespace
} // namespace facebook::velox::cudf_velox
