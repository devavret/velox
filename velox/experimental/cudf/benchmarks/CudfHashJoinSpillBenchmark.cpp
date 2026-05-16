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

/// Benchmark for cuDF hash join spilling analysis.
///
/// The plan is always TableScan → HashJoin → Aggregation. The data source is
/// controlled by --preload (inherited from CudfTpchBenchmark):
///
///   --preload=off (default)
///     Reads parquet files at query time. Measures scan + join.
///
///   --preload=gpu
///     Pre-loads parquet data into GPU-resident CudfVectors at startup
///     (outside the timed path). The PreloadedTableScanAdapter substitutes
///     TableScan with PreloadedScanOperator transparently. Measures join only
///     — no scan, no CPU→GPU transfer in the hot path.
///
///   --preload=cpu
///     Like --preload=gpu, but data is staged as CPU RowVectors and converted
///     to GPU on demand.
///
/// Usage:
///   # Full pipeline (scan + join):
///   ./velox_cudf_hashjoin_spill_benchmark \
///       --data_path=/data/tpch/pq_sf100_f64 --include_results
///
///   # Preloaded GPU (join only):
///   ./velox_cudf_hashjoin_spill_benchmark \
///       --data_path=/data/tpch/pq_sf10_f64 --include_results \
///       --preload=gpu
///
///   # Partitioned join experiment:
///   ./velox_cudf_hashjoin_spill_benchmark \
///       --data_path=/data/tpch/pq_sf10_f64 --include_results \
///       --spill_join_strategy=partitioned \
///       --partitioned_join_num_partitions=16 \
///       --spill_join_host_memory=regular

#include "velox/experimental/cudf/benchmarks/CudfBenchmarkHelpers.h"
#include "velox/experimental/cudf/benchmarks/CudfPartitionedHashJoinAdapter.h"
#include "velox/experimental/cudf/benchmarks/CudfPiecewiseSpillHashJoinAdapter.h"
#include "velox/experimental/cudf/benchmarks/CudfTpchBenchmark.h"

#include "velox/common/base/SuccinctPrinter.h"
#include "velox/dwio/common/ColumnSelector.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/tpch/gen/TpchGen.h"

DECLARE_string(data_path);
DECLARE_string(data_format);
DECLARE_string(preload);

DEFINE_string(
    spill_join_strategy,
    "baseline",
    "Hash join strategy for this benchmark: baseline, piecewise, or "
    "partitioned. Only the custom strategies support inner equi-join with no "
    "filter.");

DEFINE_int64(
    piecewise_spill_piece_rows,
    5'000'000,
    "Target rows per spilled build piece for --spill_join_strategy=piecewise.");

DEFINE_int32(
    partitioned_join_num_partitions,
    16,
    "Number of hash partitions for --spill_join_strategy=partitioned.");

DEFINE_string(
    spill_join_host_memory,
    "pinned",
    "Host memory used for spilled build payloads and hash maps. "
    "Allowed values: pinned, regular. regular is pageable host memory and "
    "disables cudaHostAlloc.");

DECLARE_int32(num_drivers);

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace {

bool usePinnedSpillJoinHostMemory() {
  if (FLAGS_spill_join_host_memory == "pinned") {
    return true;
  }
  if (FLAGS_spill_join_host_memory == "regular" ||
      FLAGS_spill_join_host_memory == "pageable") {
    return false;
  }
  VELOX_FAIL(
      "Unsupported --spill_join_host_memory: {}. Expected pinned or regular.",
      FLAGS_spill_join_host_memory);
}

} // namespace

/// Extends CudfTpchBenchmark with a purpose-built join-only plan.
///
/// The plan always uses TableScan nodes. When --preload != off, the base
/// class's PreloadedTableScanAdapter replaces TableScan with
/// PreloadedScanOperator at runtime, so the scan reads from memory instead
/// of disk.
class CudfHashJoinSpillBenchmark : public CudfTpchBenchmark {
 public:
  void runMain(std::ostream& out, RunStats& runStats) override {
    if (FLAGS_spill_join_strategy == "piecewise") {
      VELOX_CHECK_GT(
          FLAGS_piecewise_spill_piece_rows,
          0,
          "--piecewise_spill_piece_rows must be positive");
      // The piecewise host-spill study deliberately runs with a single
      // probe driver so multiple drivers don't compete for the build
      // pieces. Override num_drivers here; users can still set
      // --num_drivers explicitly for the baseline runs.
      if (FLAGS_num_drivers != 1) {
        out << "[piecewise] forcing --num_drivers=1 (was " << FLAGS_num_drivers
            << ")" << std::endl;
        FLAGS_num_drivers = 1;
      }
      cudf_velox::registerPiecewiseSpillHashJoinAdapter(
          static_cast<cudf::size_type>(FLAGS_piecewise_spill_piece_rows),
          usePinnedSpillJoinHostMemory());
    } else if (FLAGS_spill_join_strategy == "partitioned") {
      VELOX_CHECK_GT(
          FLAGS_partitioned_join_num_partitions,
          0,
          "--partitioned_join_num_partitions must be positive");
      cudf_velox::registerPartitionedHashJoinAdapter(
          FLAGS_partitioned_join_num_partitions,
          usePinnedSpillJoinHostMemory());
    } else {
      VELOX_CHECK_EQ(
          FLAGS_spill_join_strategy,
          "baseline",
          "Unsupported --spill_join_strategy: {}. Expected baseline, "
          "piecewise, or partitioned.",
          FLAGS_spill_join_strategy);
    }
    auto pool = memory::memoryManager()->addLeafPool();
    auto format = dwio::common::toFileFormat(FLAGS_data_format);

    auto lineitemStdCols =
        tpch::getTableSchema(tpch::Table::TBL_LINEITEM)->names();
    auto ordersStdCols = tpch::getTableSchema(tpch::Table::TBL_ORDERS)->names();

    auto lineitemInfo = cudf_velox::readTableInfo(
        "lineitem", FLAGS_data_path, lineitemStdCols, format, pool.get());
    auto ordersInfo = cudf_velox::readTableInfo(
        "orders", FLAGS_data_path, ordersStdCols, format, pool.get());

    VELOX_CHECK(
        !lineitemInfo.dataFiles.empty(), "No lineitem data files found");
    VELOX_CHECK(!ordersInfo.dataFiles.empty(), "No orders data files found");

    auto lineitemRowType = dwio::common::ColumnSelector(
                               lineitemInfo.type,
                               std::vector<std::string>{
                                   "l_orderkey",
                                   "l_extendedprice",
                                   "l_discount",
                                   "l_quantity",
                                   "l_partkey"})
                               .buildSelectedReordered();
    auto ordersRowType =
        dwio::common::ColumnSelector(
            ordersInfo.type,
            std::vector<std::string>{"o_orderkey", "o_custkey", "o_totalprice"})
            .buildSelectedReordered();

    out << "=== cuDF Hash Join Spill Benchmark ===" << std::endl;
    out << "Data path: " << FLAGS_data_path << std::endl;
    out << "Preload mode: " << FLAGS_preload << std::endl;
    out << "Join strategy: " << FLAGS_spill_join_strategy << std::endl;
    if (FLAGS_spill_join_strategy == "partitioned") {
      out << "Partitioned join partitions: "
          << FLAGS_partitioned_join_num_partitions << std::endl;
    }
    if (FLAGS_spill_join_strategy == "piecewise" ||
        FLAGS_spill_join_strategy == "partitioned") {
      out << "Spill join host memory: " << FLAGS_spill_join_host_memory
          << std::endl;
    }
    out << "Lineitem files: " << lineitemInfo.dataFiles.size() << std::endl;
    out << "Orders files: " << ordersInfo.dataFiles.size() << std::endl;

    auto tpchPlan = buildScanJoinPlan(
        lineitemInfo, ordersInfo, lineitemRowType, ordersRowType, pool.get());

    out << "Executing..." << std::endl;
    auto [cursor, results] = run(tpchPlan, queryConfigs_);
    if (!cursor) {
      LOG(ERROR) << "Query terminated with error. Exiting";
      exit(1);
    }

    auto task = cursor->task();
    ensureTaskCompletion(task.get());

    if (FLAGS_include_results) {
      printResults(results, out);
      out << std::endl;
    }

    const auto stats = task->taskStats();
    int64_t rawInputBytes = 0;
    for (auto& pipeline : stats.pipelineStats) {
      auto& first = pipeline.operatorStats[0];
      if (first.operatorType == exec::OperatorType::kTableScan) {
        rawInputBytes += first.rawInputBytes;
      }
    }
    runStats.rawInputBytes = rawInputBytes;

    out << fmt::format(
               "Execution time: {}",
               succinctMillis(
                   stats.executionEndTimeMs - stats.executionStartTimeMs))
        << std::endl;
    out << fmt::format(
               "Splits total: {}, finished: {}",
               stats.numTotalSplits,
               stats.numFinishedSplits)
        << std::endl;
    out << exec::printPlanWithStats(
               *tpchPlan.plan, stats, FLAGS_include_custom_stats)
        << std::endl;
  }

 private:
  TpchPlan buildScanJoinPlan(
      const cudf_velox::TableInfo& lineitemInfo,
      const cudf_velox::TableInfo& ordersInfo,
      const RowTypePtr& lineitemRowType,
      const RowTypePtr& ordersRowType,
      memory::MemoryPool* pool) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    core::PlanNodeId ordersScanId;
    core::PlanNodeId lineitemScanId;

    auto ordersScan =
        PlanBuilder(planNodeIdGenerator, pool)
            .tableScan("orders", ordersRowType, ordersInfo.fileColumnNames, {})
            .captureScanNodeId(ordersScanId)
            .planNode();

    auto plan =
        PlanBuilder(planNodeIdGenerator, pool)
            .tableScan(
                "lineitem", lineitemRowType, lineitemInfo.fileColumnNames, {})
            .captureScanNodeId(lineitemScanId)
            .hashJoin(
                {"l_orderkey"},
                {"o_orderkey"},
                ordersScan,
                "",
                {"o_custkey",
                 "o_totalprice",
                 "l_extendedprice",
                 "l_discount",
                 "l_quantity"})
            .partialAggregation(
                {}, {"count(1) as cnt", "sum(l_extendedprice) as total_price"})
            .localPartition(std::vector<std::string>{})
            .finalAggregation()
            .planNode();

    TpchPlan tpchPlan;
    tpchPlan.plan = std::move(plan);
    tpchPlan.dataFiles[ordersScanId] = ordersInfo.dataFiles;
    tpchPlan.dataFiles[lineitemScanId] = lineitemInfo.dataFiles;
    tpchPlan.dataFileFormat = dwio::common::toFileFormat(FLAGS_data_format);
    return tpchPlan;
  }
};

int main(int argc, char** argv) {
  std::string kUsage("Benchmarks cuDF hash join for spilling analysis");
  gflags::SetUsageMessage(kUsage);
  folly::Init init{&argc, &argv, false};
  benchmark = std::make_unique<CudfHashJoinSpillBenchmark>();
  tpchBenchmarkMain();
}
