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
#include "velox/experimental/cudf/benchmarks/CudfTpchBenchmark.h"
#include "velox/experimental/cudf/benchmarks/CudfTpchInputColumns.h"
#include "velox/experimental/cudf/benchmarks/CudfTpchRunStats.h"
#include "velox/experimental/cudf/connectors/encoded/EncodedConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/CudfAggregationNode.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfMemoryArbitration.h"
#include "velox/experimental/cudf/exec/CudfSpillContext.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/SuccinctPrinter.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/common/Options.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/Split.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

#include <experimental/cudf/connectors/hive/CudfHiveConnector.h>
#include <fmt/format.h>
#include <folly/ScopeGuard.h>
#include <folly/String.h>
#include <folly/json.h>
#include <folly/system/HardwareConcurrency.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <unordered_set>

DECLARE_int64(max_coalesced_bytes);
DECLARE_string(max_coalesced_distance_bytes);
DECLARE_int32(parquet_prefetch_rowgroups);
DECLARE_string(data_path);
DECLARE_int32(num_drivers);
DECLARE_int32(num_repeats);
DECLARE_int32(num_splits_per_file);
DECLARE_int32(run_query_verbose);
DECLARE_int32(io_meter_column_pct);

// The process environment, for recording the VELOX_CUDF_* variables a run was
// made with.
extern char** environ;

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::dwio::common;
namespace encoded = facebook::velox::cudf_velox::connector::encoded;

DEFINE_uint64(
    cudf_chunk_read_limit,
    0,
    "Output table chunk read limit for cudf::parquet_chunked_reader.");

DEFINE_uint64(
    cudf_pass_read_limit,
    0,
    "Pass read limit for cudf::parquet_chunked_reader.");

DEFINE_int32(
    cudf_gpu_batch_size_rows,
    100000,
    "Preferred output batch size in rows for cudf operators.");

DEFINE_uint64(
    cudf_local_exchange_buffer_size,
    1UL << 30,
    "Maximum buffered bytes per local exchange before applying backpressure.");

DEFINE_bool(velox_cudf_table_scan, true, "Enable cuDF table scan");
DEFINE_bool(
    cudf_hive_use_buffered_input,
    true,
    "Use BufferedInputDataSource instead of KvikIO for cuDF Hive reads");

DEFINE_bool(
    cudf_hive_preload_column_chunks,
    false,
    "Load column chunks while preloading cuDF Hive splits");

DEFINE_bool(
    cudf_hive_serialize_io_requests,
    false,
    "Serialize I/O submissions to prevent interleaving between concurrent splits");

DEFINE_string(
    cudf_properties,
    "",
    "Path to a properties file for CudfConfig. Each line should be key=value "
    "(e.g. cudf.memory_resource=async). See CudfConfig for available keys.");

DEFINE_bool(
    cudf_compact_decimal_sum_state,
    true,
    "Exchange a 16-byte DECIMAL(38, scale) state for each grouped SUM over a "
    "short decimal column between its partial and final aggregation, in place "
    "of the 32-byte VARBINARY state and its offsets. The plans keep their "
    "nodes, node ids and result types. Only cuDF operators read and write the "
    "state, so this applies only with cudf.allow_cpu_fallback=false.");

DEFINE_bool(
    encoded,
    false,
    "Scan the tables from the encoded in-memory column cache instead of "
    "Parquet. The tables are loaded from --data_path a few row groups at a "
    "time, in file order, and encoded on the GPU before the queries that read "
    "them are timed; sort the files beforehand to cluster them.");

DEFINE_string(
    encoded_memory,
    "device",
    "With --encoded, where the encoded tables are kept: device, or "
    "pinned_host to have scans copy each batch to the device before decoding.");

DEFINE_string(
    encoded_load,
    "query",
    "With --encoded, what is loaded and when: 'query' loads, before each "
    "query, the columns its scans read and drops the columns it does not "
    "read; 'all' loads every column of every table before the first query.");

DEFINE_int32(
    encoded_segment_rows,
    1 << 20,
    "Rows per encoded segment, the unit of min/max pruning.");

DEFINE_int32(
    encoded_segments_per_split,
    4,
    "Segments per split, the unit of work handed to scan drivers. GPU scans "
    "batch segments across splits up to the cuDF batch-size targets.");

DEFINE_uint64(
    host_budget_bytes,
    0,
    "Pinned host memory budget shared by the encoded cache "
    "(--encoded_memory=pinned_host) and the GPU spill pool "
    "(cudf.spill_host_memory_bytes). The cache may hold the budget minus the "
    "spill pool. Zero leaves the cache unlimited.");

DEFINE_uint64(
    gpu_budget_bytes,
    0,
    "GPU memory budget queries are arbitrated against. Overrides "
    "cudf.memory_limit_bytes of --cudf_properties when not zero.");

DEFINE_uint64(
    spill_host_bytes,
    0,
    "Pinned host memory GPU operators may spill to. Overrides "
    "cudf.spill_host_memory_bytes of --cudf_properties when not zero. Without "
    "either, operators do not spill.");

DEFINE_string(
    input_overflow,
    "fail",
    "With --encoded and --host_budget_bytes, what happens to a query whose "
    "input columns do not fit the host cache: 'fail' fails the query; "
    "'parquet' runs that query with Parquet scans of --data_path instead and "
    "says so in its output.");

DEFINE_string(
    run_queries,
    "",
    "Comma-separated query numbers to run verbosely one after another in this "
    "process, initializing once. Each query's output starts with a 'Running "
    "query N' line. Overrides --run_query_verbose.");

DEFINE_double(
    scale_factor,
    1,
    "TPC-H scale factor of the data in --data_path. Sets Q11's FRACTION, "
    "0.0001 / scale factor.");

namespace {
bool validateScaleFactor(const char* /*flagName*/, double value) {
  return std::isfinite(value) && value > 0;
}
} // namespace

DEFINE_validator(scale_factor, &validateScaleFactor);

DEFINE_string(
    results_json_dir,
    "",
    "If set, after timing each run of a query, write that run's result rows "
    "to <dir>/repeat<i>/q<N>.json, DOUBLE values with 17 significant digits, "
    "and its counters to <dir>/repeat<i>/q<N>.stats.json; a failed run writes "
    "only the counters. When every run of a query succeeded, the last run's "
    "rows are also written to <dir>/q<N>.json. With --run_queries, "
    "<dir>/run_summary.json lists the queries run and the ones that failed. "
    "Nothing is written while a run is being timed.");

namespace {

const std::vector<std::string> kTables = {
    "lineitem",
    "orders",
    "customer",
    "part",
    "partsupp",
    "supplier",
    "nation",
    "region",
};

// Orders file names so that part.2.parquet comes before part.10.parquet.
bool naturalLess(const std::string& left, const std::string& right) {
  size_t i = 0;
  size_t j = 0;
  while (i < left.size() && j < right.size()) {
    if (std::isdigit(left[i]) && std::isdigit(right[j])) {
      size_t iEnd = i;
      while (iEnd < left.size() && std::isdigit(left[iEnd])) {
        ++iEnd;
      }
      size_t jEnd = j;
      while (jEnd < right.size() && std::isdigit(right[jEnd])) {
        ++jEnd;
      }
      const auto leftNumber = std::stoull(left.substr(i, iEnd - i));
      const auto rightNumber = std::stoull(right.substr(j, jEnd - j));
      if (leftNumber != rightNumber) {
        return leftNumber < rightNumber;
      }
      i = iEnd;
      j = jEnd;
      continue;
    }
    if (left[i] != right[j]) {
      return left[i] < right[j];
    }
    ++i;
    ++j;
  }
  return left.size() - i < right.size() - j;
}

// The data files of a table directory, in the order the encoded cache loads
// them. The order fixes the segment boundaries, so every load of a table
// must use it.
std::vector<std::string> parquetFiles(const std::string& tableDir) {
  std::vector<std::string> files;
  for (const auto& entry : std::filesystem::directory_iterator(tableDir)) {
    if (entry.is_regular_file() &&
        entry.path().filename().string().front() != '.') {
      files.push_back(entry.path().string());
    }
  }
  std::sort(files.begin(), files.end(), naturalLess);
  VELOX_CHECK(!files.empty(), "No data files in {}", tableDir);
  return files;
}

// The host memory the encoded cache keeps host-resident tables in.
encoded::BoundedPinnedHostResource& encodedHostResource() {
  static encoded::BoundedPinnedHostResource resource;
  return resource;
}

// Unlike QueryBenchmarkBase::printResults(), prints every column; that
// truncates rows to 5 columns.
void printRows(const std::vector<RowVectorPtr>& results, std::ostream& out) {
  out << "Results:" << std::endl;
  if (!results.empty()) {
    out << results.front()->type()->asRow().toString() << std::endl;
  }
  for (const auto& vector : results) {
    for (vector_size_t i = 0; i < vector->size(); ++i) {
      std::vector<std::string> fields;
      for (const auto& child : vector->children()) {
        fields.push_back(child->toString(i));
      }
      out << "{" << folly::join(", ", fields) << "}" << std::endl;
    }
  }
  out << std::endl;
}

// Writes {"types": [...], "rows": [[...], ...]} to 'path'. A value is null, a
// DOUBLE with 17 significant digits, or what BaseVector::toString() prints for
// it.
void writeResultsJson(
    const std::string& path,
    const std::vector<RowVectorPtr>& results) {
  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path());
  folly::dynamic doc = folly::dynamic::object;
  doc["types"] = folly::dynamic::array;
  doc["rows"] = folly::dynamic::array;
  if (!results.empty()) {
    for (const auto& child : results.front()->children()) {
      doc["types"].push_back(child->type()->toString());
    }
  }
  for (const auto& batch : results) {
    for (vector_size_t row = 0; row < batch->size(); ++row) {
      folly::dynamic values = folly::dynamic::array;
      for (const auto& child : batch->children()) {
        if (child->isNullAt(row)) {
          values.push_back(nullptr);
        } else if (child->typeKind() == TypeKind::DOUBLE) {
          const auto* doubles = child->as<SimpleVector<double>>();
          VELOX_CHECK_NOT_NULL(doubles);
          std::ostringstream value;
          value << std::setprecision(17) << doubles->valueAt(row);
          values.push_back(value.str());
        } else {
          values.push_back(child->toString(row));
        }
      }
      doc["rows"].push_back(std::move(values));
    }
  }
  std::ofstream file(path);
  file << folly::toJson(doc) << std::endl;
  VELOX_CHECK(file.good(), "Failed to write {}", path);
}

// The directory under --results_json_dir of the files of run 'repeat'.
std::string repeatDir(int32_t repeat) {
  return fmt::format("{}/repeat{}", FLAGS_results_json_dir, repeat);
}

// The cumulative counters of the process GPU arbitrator, zeros when GPU memory
// is not arbitrated. The difference of two calls is what the runs between
// them added.
memory::MemoryArbitrator::Stats arbitratorStats() {
  if (auto resource = cudf_velox::cudfCustomMemoryResource()) {
    return resource->arbitrator()->stats();
  }
  return {};
}

// folly::dynamic holds integers as int64_t. No counter reported here reaches
// 2^63; one that did would be reported as 2^63 - 1.
template <typename T>
int64_t jsonInt(T value) {
  static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>);
  if constexpr (std::is_unsigned_v<T>) {
    return saturateCast(static_cast<uint64_t>(value));
  } else {
    return static_cast<int64_t>(value);
  }
}

folly::dynamic arbitratorJson(const memory::MemoryArbitrator::Stats& stats) {
  folly::dynamic json = folly::dynamic::object;
  json["num_requests"] = jsonInt(stats.numRequests);
  json["num_succeeded"] = jsonInt(stats.numSucceeded);
  json["num_aborted"] = jsonInt(stats.numAborted);
  json["num_failures"] = jsonInt(stats.numFailures);
  json["reclaimed_free_bytes"] = jsonInt(stats.reclaimedFreeBytes);
  json["reclaimed_used_bytes"] = jsonInt(stats.reclaimedUsedBytes);
  json["num_non_reclaimable_attempts"] =
      jsonInt(stats.numNonReclaimableAttempts);
  return json;
}

// The GPU memory counters of one run: the process arbitrator before and after
// it, and the query's GPU root pool.
folly::dynamic gpuMemoryJson(
    const std::shared_ptr<core::QueryCtx>& queryCtx,
    const memory::MemoryArbitrator::Stats& before,
    const memory::MemoryArbitrator::Stats& after) {
  folly::dynamic json = folly::dynamic::object;
  json["arbitrator_before_run"] = arbitratorJson(before);
  json["arbitrator_after_run"] = arbitratorJson(after);
  // The difference of the two, counter by counter.
  json["arbitrator_during_run"] = arbitratorJson(after - before);
  json["arbitrator_max_capacity_bytes"] = jsonInt(after.maxCapacityBytes);
  json["arbitrator_free_capacity_bytes_after_run"] =
      jsonInt(after.freeCapacityBytes);
  if (queryCtx == nullptr) {
    return json;
  }
  if (auto gpuPool = queryCtx->customPool(
          std::string{cudf_velox::kCudfMemoryResourceTag})) {
    const auto poolStats = gpuPool->stats();
    folly::dynamic pool = folly::dynamic::object;
    pool["peak_bytes"] = jsonInt(gpuPool->peakBytes());
    pool["max_capacity_bytes"] = jsonInt(gpuPool->maxCapacity());
    pool["capacity_bytes_after_run"] = jsonInt(gpuPool->capacity());
    pool["used_bytes_after_run"] = jsonInt(gpuPool->usedBytes());
    pool["reserved_bytes_after_run"] = jsonInt(gpuPool->reservedBytes());
    pool["cumulative_bytes"] = jsonInt(poolStats.cumulativeBytes);
    pool["num_reclaims"] = jsonInt(poolStats.numReclaims);
    pool["num_shrinks"] = jsonInt(poolStats.numShrinks);
    pool["num_capacity_growths"] = jsonInt(poolStats.numCapacityGrowths);
    json["query_pool"] = std::move(pool);
  }
  return json;
}

// Why a run failed, without the stack trace what() may carry.
std::string failureReason(const std::exception& e) {
  if (const auto* veloxError = dynamic_cast<const VeloxException*>(&e)) {
    return fmt::format(
        "{}: {}", veloxError->errorCode(), veloxError->message());
  }
  const std::string reason{e.what()};
  return reason.empty() ? std::string("unknown error") : reason;
}

// The query's spill context, null if the query does not spill. Null with
// 'error' set if the context cannot be made, which is itself a way for a
// query to fail and must not hide that failure when its run is reported.
std::shared_ptr<cudf_velox::CudfSpillContext> spillContext(
    const std::shared_ptr<core::QueryCtx>& queryCtx,
    std::string& error) {
  if (queryCtx == nullptr) {
    return nullptr;
  }
  try {
    return cudf_velox::cudfSpillContext(*queryCtx);
  } catch (const std::exception& e) {
    error = failureReason(e);
    return nullptr;
  }
}

// The pinned host spill pool as one run left it. The "reserved" counters are
// admissions against the spill limit: bytes of spilled data live in the pool
// plus bytes admitted for spills in progress. They are not the volume
// spilled or restored, which each operator reports in its runtime_stats
// (cudfGroupbySpilledBytes, cudfJoinSpilledBytes, ...).
folly::dynamic hostSpillJson(const std::shared_ptr<core::QueryCtx>& queryCtx) {
  folly::dynamic json = folly::dynamic::object;
  std::string contextError;
  const auto spill = spillContext(queryCtx, contextError);
  json["enabled"] = spill != nullptr;
  if (!contextError.empty()) {
    json["context_error"] = contextError;
  }
  if (spill == nullptr) {
    return json;
  }
  json["query_limit_bytes"] = jsonInt(spill->budget->limit());
  json["query_reserved_bytes_after_run"] = jsonInt(spill->budget->used());
  json["query_peak_reserved_bytes"] = jsonInt(spill->budget->peak());
  // The process pool outlives the run: its peak is over every run so far.
  const auto& processBudget = spill->storage->budget();
  json["process_limit_bytes"] = jsonInt(processBudget->limit());
  json["process_reserved_bytes_after_run"] = jsonInt(processBudget->used());
  json["process_peak_reserved_bytes_so_far"] = jsonInt(processBudget->peak());
  json["process_pool_physical_bytes"] =
      jsonInt(spill->storage->reservedBytes());
  return json;
}

// Returns true if the plans get the compact decimal SUM state. Only cuDF
// operators read and write that state, and an aggregation that has it fails
// its query if it cannot run in cuDF. Without CPU fallback every operator must
// run in cuDF anyway, so the state cannot fail a query that would have run.
bool compactDecimalSumStateEnabled() {
  const auto& config = cudf_velox::CudfConfig::getInstance();
  return FLAGS_cudf_compact_decimal_sum_state && config.enabled &&
      !config.allowCpuFallback;
}

// Returns the number of aggregations of 'plan' that exchange the compact
// decimal SUM state.
int32_t countCompactAggregations(const core::PlanNodePtr& plan) {
  int32_t numAggregations{0};
  if (dynamic_cast<const cudf_velox::CudfAggregationNode*>(plan.get()) !=
      nullptr) {
    ++numAggregations;
  }
  for (const auto& source : plan->sources()) {
    numAggregations += countCompactAggregations(source);
  }
  return numAggregations;
}

// The settings a run was made with.
folly::dynamic configJson() {
  const auto& config = cudf_velox::CudfConfig::getInstance();
  folly::dynamic json = folly::dynamic::object;
  json["memory_resource"] = config.memoryResource;
  json["memory_limit_bytes"] = jsonInt(config.memoryLimitBytes);
  json["memory_percent"] = jsonInt(config.memoryPercent);
  json["spill_host_memory_bytes"] = jsonInt(config.spillHostBytes);
  json["spill_partition_fanout"] = jsonInt(config.spillPartitionFanout);
  json["spill_groupby_max_rows_per_leaf"] =
      jsonInt(config.spillGroupbyMaxRowsPerLeaf);
  json["spill_join_partition_bytes"] = jsonInt(config.spillJoinPartitionBytes);
  json["groupby_self_spill_percent"] = jsonInt(config.groupbySelfSpillPercent);
  json["groupby_self_spill_target_percent"] =
      jsonInt(config.groupbySelfSpillTargetPercent);
  json["batch_size_min_threshold"] = jsonInt(config.batchSizeMinThreshold);
  if (config.batchSizeMaxThreshold.has_value()) {
    json["batch_size_max_threshold"] = jsonInt(*config.batchSizeMaxThreshold);
  } else {
    json["batch_size_max_threshold"] = nullptr;
  }
  json["concat_optimization_enabled"] = config.concatOptimizationEnabled;
  json["streaming_groupby_enabled"] = config.streamingGroupbyEnabled;
  json["allow_cpu_fallback"] = config.allowCpuFallback;
  json["compact_decimal_sum_state"] = compactDecimalSumStateEnabled();
  json["gpu_batch_size_rows"] = jsonInt(FLAGS_cudf_gpu_batch_size_rows);
  json["local_exchange_buffer_size_bytes"] =
      jsonInt(FLAGS_cudf_local_exchange_buffer_size);
  json["host_budget_bytes"] = jsonInt(FLAGS_host_budget_bytes);
  json["num_drivers"] = jsonInt(FLAGS_num_drivers);
  json["scale_factor"] = FLAGS_scale_factor;
  return json;
}

// The VELOX_CUDF_* environment variables of the process. They select the
// experimental paths of the cuDF operators, so a run is not described without
// them.
folly::dynamic cudfEnvironmentJson() {
  folly::dynamic json = folly::dynamic::object;
  for (char** entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
    const std::string_view variable{*entry};
    const auto separator = variable.find('=');
    if (variable.rfind("VELOX_CUDF_", 0) != 0 ||
        separator == std::string_view::npos) {
      continue;
    }
    json[std::string(variable.substr(0, separator))] =
        std::string(variable.substr(separator + 1));
  }
  return json;
}

// The pinned host memory the encoded cache holds.
folly::dynamic encodedCacheJson() {
  const auto& cache = encoded::EncodedColumnCache::instance();
  const auto& host = encodedHostResource();
  folly::dynamic json = folly::dynamic::object;
  json["host_bytes"] = jsonInt(cache.hostBytes());
  json["device_bytes"] = jsonInt(cache.deviceBytes());
  json["pinned_used_bytes"] = jsonInt(host.usedBytes());
  json["pinned_peak_bytes"] = jsonInt(host.peakBytes());
  json["pinned_limit_bytes"] = jsonInt(host.limitBytes());
  return json;
}

// Everything recorded about one run of a query, successful or not: what its
// task, operators and drivers counted, and the memory and spill counters
// around them. 'task' is null if the run failed before it had one.
// 'numCompactAggregations' is the number of aggregations of the plan that
// exchange the compact decimal SUM state.
folly::dynamic runStatsJson(
    int32_t queryId,
    int32_t repeat,
    int64_t wallMicros,
    const std::string& error,
    const std::string& input,
    int32_t numCompactAggregations,
    exec::Task* task,
    const memory::MemoryArbitrator::Stats& arbitratorBefore,
    const memory::MemoryArbitrator::Stats& arbitratorAfter) {
  folly::dynamic json = folly::dynamic::object;
  json["schema"] = "velox_cudf_tpch_benchmark.run_stats.v1";
  json["query"] = jsonInt(queryId);
  json["repeat"] = jsonInt(repeat);
  json["status"] = error.empty() ? "ok" : "failed";
  if (!error.empty()) {
    json["error"] = error;
  }
  // For a failed run, the time until the failure was seen.
  json["wall_micros"] = wallMicros;
  json["input"] = input;
  json["compact_decimal_sum_aggregations"] = jsonInt(numCompactAggregations);
  json["config"] = configJson();
  json["environment"] = cudfEnvironmentJson();
  json["encoded_cache"] = encodedCacheJson();
  std::shared_ptr<core::QueryCtx> queryCtx;
  if (task != nullptr) {
    queryCtx = task->queryCtx();
  }
  json["gpu_memory"] =
      gpuMemoryJson(queryCtx, arbitratorBefore, arbitratorAfter);
  json["host_spill"] = hostSpillJson(queryCtx);
  if (task != nullptr) {
    json["task_state"] = exec::taskStateString(task->state());
    json["task"] = cudf_velox::CudfTpchRunStats::toJson(task->taskStats());
  }
  return json;
}

void writeRunStatsJson(const std::string& path, const folly::dynamic& stats) {
  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path());
  std::ofstream file(path);
  file << cudf_velox::CudfTpchRunStats::serialize(stats) << std::endl;
  VELOX_CHECK(file.good(), "Failed to write {}", path);
}

// The spill and reclaim counters of one run for the log, each the sum over
// the run's operators, or drivers, of the runtime statistic of that name as
// recorded. The per-operator values are in the run's stats file.
std::string spillLine(int32_t repeat, const exec::TaskStats& stats) {
  const auto sum = [&](std::string_view name) {
    return cudf_velox::CudfTpchRunStats::operatorRuntimeStatSum(stats, name);
  };
  return fmt::format(
      "Repeat {} spill counters: groupby_spilled_bytes={} "
      "groupby_spilled_host_bytes={} groupby_spilled_rows={} "
      "groupby_spilled_leaves={} groupby_spilled_input_chunks={} "
      "groupby_restored_bytes={} groupby_restored_rows={} "
      "groupby_restored_leaves={} groupby_restored_input_chunks={} "
      "join_spilled_bytes={} join_restored_bytes={} "
      "join_host_bytes_written={} join_probe_spilled_bytes={} "
      "join_probe_restored_bytes={} join_probe_host_bytes_written={} "
      "memory_arbitration_wall_nanos={} task_memory_reclaim_count={} "
      "task_memory_reclaim_ms={} driver_pause_wall_nanos={}",
      repeat,
      sum("cudfGroupbySpilledBytes"),
      sum("cudfGroupbySpilledHostBytes"),
      sum("cudfGroupbySpilledRows"),
      sum("cudfGroupbySpilledLeaves"),
      sum("cudfGroupbySpilledInputChunks"),
      sum("cudfGroupbyRestoredBytes"),
      sum("cudfGroupbyRestoredRows"),
      sum("cudfGroupbyRestoredLeaves"),
      sum("cudfGroupbyRestoredInputChunks"),
      sum("cudfJoinSpilledBytes"),
      sum("cudfJoinRestoredBytes"),
      sum("cudfJoinHostBytesWritten"),
      sum("cudfJoinProbeSpilledBytes"),
      sum("cudfJoinProbeRestoredBytes"),
      sum("cudfJoinProbeHostBytesWritten"),
      sum("memoryArbitrationWallNanos"),
      stats.memoryReclaimCount,
      stats.memoryReclaimMs,
      cudf_velox::CudfTpchRunStats::driverRuntimeStatSum(
          stats, exec::DriverStats::kTotalPauseTime));
}

// The memory and spill counters of one run of a query.
std::string memoryLine(
    int32_t repeat,
    const std::shared_ptr<core::QueryCtx>& queryCtx,
    const memory::MemoryArbitrator::Stats& before,
    const memory::MemoryArbitrator::Stats& after) {
  int64_t gpuPeak = 0;
  int64_t gpuCapacity = 0;
  if (auto gpuPool = queryCtx->customPool(
          std::string{cudf_velox::kCudfMemoryResourceTag})) {
    gpuPeak = gpuPool->peakBytes();
    gpuCapacity = gpuPool->maxCapacity();
  }
  uint64_t spillPeak = 0;
  uint64_t spillLimit = 0;
  uint64_t spillPoolReserved = 0;
  std::string contextError;
  if (auto spill = spillContext(queryCtx, contextError)) {
    spillPeak = spill->budget->peak();
    spillLimit = spill->budget->limit();
    spillPoolReserved = spill->storage->reservedBytes();
  }
  return fmt::format(
      "Repeat {} memory: gpu_query_peak_bytes={} gpu_budget_bytes={} "
      "gpu_arbitration_requests={} gpu_arbitration_failures={} "
      "gpu_reclaimed_bytes={} gpu_non_reclaimable_attempts={} "
      "host_spill_peak_bytes={} host_spill_limit_bytes={} "
      "host_spill_pool_reserved_bytes={} encoded_cache_host_bytes={} "
      "encoded_cache_device_bytes={}",
      repeat,
      gpuPeak,
      gpuCapacity,
      after.numRequests - before.numRequests,
      after.numFailures - before.numFailures,
      after.reclaimedUsedBytes - before.reclaimedUsedBytes,
      after.numNonReclaimableAttempts - before.numNonReclaimableAttempts,
      spillPeak,
      spillLimit,
      spillPoolReserved,
      encoded::EncodedColumnCache::instance().hostBytes(),
      encoded::EncodedColumnCache::instance().deviceBytes());
}

} // namespace

void CudfTpchBenchmark::initialize() {
  auto& config = cudf_velox::CudfConfig::getInstance();
  if (!FLAGS_cudf_properties.empty()) {
    config.initialize(cudf_velox::loadPropertiesFile(FLAGS_cudf_properties));
  }
  if (FLAGS_gpu_budget_bytes != 0) {
    config.memoryLimitBytes = FLAGS_gpu_budget_bytes;
  }
  if (FLAGS_spill_host_bytes != 0) {
    config.spillHostBytes = FLAGS_spill_host_bytes;
  }
  VELOX_USER_CHECK(
      FLAGS_encoded_memory == "device" || FLAGS_encoded_memory == "pinned_host",
      "--encoded_memory must be device or pinned_host: {}",
      FLAGS_encoded_memory);
  VELOX_USER_CHECK(
      FLAGS_encoded_load == "query" || FLAGS_encoded_load == "all",
      "--encoded_load must be query or all: {}",
      FLAGS_encoded_load);
  VELOX_USER_CHECK(
      FLAGS_input_overflow == "fail" || FLAGS_input_overflow == "parquet",
      "--input_overflow must be fail or parquet: {}",
      FLAGS_input_overflow);
  if (FLAGS_host_budget_bytes != 0) {
    VELOX_USER_CHECK_GT(
        FLAGS_host_budget_bytes,
        config.spillHostBytes,
        "--host_budget_bytes must exceed the host spill pool");
    hostCacheLimitBytes_ = FLAGS_host_budget_bytes - config.spillHostBytes;
  }
  encodedHostResource().setLimit(hostCacheLimitBytes_);

  // Not TpchBenchmark::initialize(): the query builder that it makes serves
  // only TpchBenchmark::runQuery() and the folly benchmarks, which run the
  // plans of exec::test::TpchQueryBuilder and are not run here.
  QueryBenchmarkBase::initialize();
  driverExecutor_ = std::make_shared<folly::CPUThreadPoolExecutor>(
      folly::available_concurrency());
  queryBuilder_ = std::make_unique<cudf_velox::CudfTpchQueryBuilder>(
      toFileFormat(FLAGS_data_format), FLAGS_scale_factor);
  queryBuilder_->initialize(FLAGS_data_path);

  parquetConnector_ =
      connector::ConnectorRegistry::tryGet(exec::test::kHiveConnectorId);
  if (FLAGS_velox_cudf_table_scan) {
    auto cudfHiveConfigurationValues =
        std::unordered_map<std::string, std::string>();
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kMaxChunkReadLimit] =
            std::to_string(FLAGS_cudf_chunk_read_limit);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kMaxPassReadLimit] =
            std::to_string(FLAGS_cudf_pass_read_limit);
    cudfHiveConfigurationValues[cudf_velox::connector::hive::CudfHiveConfig::
                                    kAllowMismatchedCudfHiveSchemas] =
        std::to_string(true);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kUseBufferedInput] =
            std::to_string(FLAGS_cudf_hive_use_buffered_input);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kPreloadColumnChunks] =
            std::to_string(FLAGS_cudf_hive_preload_column_chunks);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kSerializeIoRequests] =
            std::to_string(FLAGS_cudf_hive_serialize_io_requests);
    auto cudfHiveProperties = std::make_shared<const config::ConfigBase>(
        std::move(cudfHiveConfigurationValues));

    cudf_velox::connector::hive::CudfHiveConnectorFactory cudfHiveFactory;
    parquetConnector_ = cudfHiveFactory.newConnector(
        exec::test::kHiveConnectorId, cudfHiveProperties, ioExecutor_.get());
  }
  if (FLAGS_encoded) {
    encodedConnector_ = std::make_shared<encoded::EncodedConnector>(
        exec::test::kHiveConnectorId);
  }
  setInputMode(FLAGS_encoded ? InputMode::kEncoded : InputMode::kParquet);

  cudf_velox::registerCudf();
  if (FLAGS_encoded) {
    // After registerCudf(), which resets the operator adapters.
    encoded::registerEncodedTableScanAdapter();
  }
  cudf_velox::registerPrestoFunctions(config.functionNamePrefix);

  queryConfigs_[cudf_velox::CudfFromVelox::kGpuBatchSizeRows] =
      std::to_string(FLAGS_cudf_gpu_batch_size_rows);
  queryConfigs_[core::QueryConfig::kMaxLocalExchangeBufferSize] =
      std::to_string(FLAGS_cudf_local_exchange_buffer_size);

  printBudgets(std::cout);

  if (FLAGS_encoded && FLAGS_encoded_load == "all") {
    for (const auto& table : kTables) {
      loadEncodedColumns(
          table, parquetFiles(FLAGS_data_path + "/" + table), {}, std::cout);
    }
  }
}

void CudfTpchBenchmark::printBudgets(std::ostream& out) const {
  const auto& config = cudf_velox::CudfConfig::getInstance();
  int64_t gpuBudget = 0;
  if (auto resource = cudf_velox::cudfCustomMemoryResource()) {
    gpuBudget = resource->maxCapacity();
  }
  out << fmt::format(
             "Budgets: gpu_budget_bytes={} host_budget_bytes={} "
             "host_spill_bytes={} host_cache_limit_bytes={} "
             "input={} "
             "encoded_memory={} encoded_load={} input_overflow={}",
             gpuBudget,
             FLAGS_host_budget_bytes,
             config.spillHostBytes,
             hostCacheLimitBytes_,
             FLAGS_encoded ? "encoded" : "parquet",
             FLAGS_encoded ? FLAGS_encoded_memory : "-",
             FLAGS_encoded ? FLAGS_encoded_load : "-",
             FLAGS_encoded ? FLAGS_input_overflow : "-")
      << std::endl;
}

void CudfTpchBenchmark::setInputMode(InputMode mode) {
  const auto& connector =
      mode == InputMode::kEncoded ? encodedConnector_ : parquetConnector_;
  VELOX_CHECK_NOT_NULL(connector);
  connector::ConnectorRegistry::global().erase(exec::test::kHiveConnectorId);
  connector::ConnectorRegistry::global().insert(
      exec::test::kHiveConnectorId, connector);
  inputMode_ = mode;
}

void CudfTpchBenchmark::loadEncodedColumns(
    const std::string& table,
    const std::vector<std::string>& files,
    const std::vector<std::string>& columns,
    std::ostream& out) {
  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  const auto mr = cudf_velox::get_output_mr();
  auto& cache = encoded::EncodedColumnCache::instance();
  const bool hostStorage = FLAGS_encoded_memory == "pinned_host";
  size_t numLoadedBefore = 0;
  if (const auto* loaded = cache.findTable(table)) {
    numLoadedBefore = loaded->columns.size();
  }
  const auto start = std::chrono::steady_clock::now();
  cache.loadColumns(
      table,
      files,
      columns,
      FLAGS_encoded_segment_rows,
      stream,
      mr,
      hostStorage ? rmm::device_async_resource_ref(encodedHostResource()) : mr,
      hostStorage);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - start)
                      .count();
  const auto& loaded = cache.table(table);
  out << fmt::format(
             "Encoded {}: {} rows, {} segments, {} columns ({} loaded now), "
             "{} host bytes, {} device bytes, {} ms",
             table,
             loaded.numRows,
             loaded.numSegments,
             loaded.columns.size(),
             loaded.columns.size() - numLoadedBefore,
             loaded.hostBytes(),
             loaded.deviceBytes(),
             ms)
      << std::endl;
}

void CudfTpchBenchmark::prepareEncodedInput(
    int32_t queryId,
    const TpchPlan& plan,
    std::ostream& out) {
  // The union of the columns the query's scans of each table read.
  std::map<std::string, std::set<std::string>> required;
  std::map<std::string, std::string> tableDirs;
  for (const auto& scan : cudf_velox::tpchScanColumns(plan.plan)) {
    required[scan.table].insert(scan.columns.begin(), scan.columns.end());
    const auto it = plan.dataFiles.find(scan.scanId);
    VELOX_CHECK(
        it != plan.dataFiles.end() && !it->second.empty(),
        "No data files for scan {} of table {}",
        scan.scanId,
        scan.table);
    tableDirs[scan.table] =
        std::filesystem::path(it->second.front()).parent_path().string();
  }
  auto& cache = encoded::EncodedColumnCache::instance();
  if (FLAGS_encoded_load == "all") {
    for (const auto& [table, columns] : required) {
      for (const auto& column : columns) {
        VELOX_CHECK(
            cache.table(table).hasColumn(column),
            "Column {} of {} is not in the encoded cache",
            column,
            table);
      }
    }
    return;
  }

  for (const auto& table : cache.tableNames()) {
    const auto it = required.find(table);
    cache.retainColumns(
        table,
        it == required.end() ? std::unordered_set<std::string>{}
                             : std::unordered_set<std::string>(
                                   it->second.begin(), it->second.end()));
  }
  auto& host = encodedHostResource();
  host.resetPeak();
  const auto start = std::chrono::steady_clock::now();
  for (const auto& [table, columns] : required) {
    loadEncodedColumns(
        table,
        parquetFiles(tableDirs.at(table)),
        std::vector<std::string>(columns.begin(), columns.end()),
        out);
  }
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - start)
                      .count();
  out << fmt::format(
             "Encoded input of query {}: {} tables, cache_host_bytes={} "
             "cache_device_bytes={} pinned_used_bytes={} "
             "pinned_peak_bytes={} host_cache_limit_bytes={} load_ms={}",
             queryId,
             required.size(),
             cache.hostBytes(),
             cache.deviceBytes(),
             host.usedBytes(),
             host.peakBytes(),
             host.limitBytes(),
             ms)
      << std::endl;
}

std::shared_ptr<config::ConfigBase>
CudfTpchBenchmark::makeConnectorProperties() {
  auto cfg = TpchBenchmark::makeConnectorProperties();
  using CudfHiveCfg = cudf_velox::connector::hive::CudfHiveConfig;

  // CuDF-specific properties.
  cfg->set(
      CudfHiveCfg::kMaxChunkReadLimit,
      std::to_string(FLAGS_cudf_chunk_read_limit));
  cfg->set(
      CudfHiveCfg::kMaxPassReadLimit,
      std::to_string(FLAGS_cudf_pass_read_limit));
  cfg->set(
      CudfHiveCfg::kUseBufferedInput,
      std::to_string(FLAGS_cudf_hive_use_buffered_input));
  cfg->set(
      CudfHiveCfg::kPreloadColumnChunks,
      std::to_string(FLAGS_cudf_hive_preload_column_chunks));
  cfg->set(
      CudfHiveCfg::kSerializeIoRequests,
      std::to_string(FLAGS_cudf_hive_serialize_io_requests));
  cfg->set(CudfHiveCfg::kAllowMismatchedCudfHiveSchemas, "true");

  return cfg;
}

std::vector<std::shared_ptr<connector::ConnectorSplit>>
CudfTpchBenchmark::listSplits(
    const std::string& path,
    int32_t numSplitsPerFile,
    const exec::test::TpchPlan& plan) {
  if (inputMode_ == InputMode::kEncoded) {
    const auto tableDir = std::filesystem::path(path).parent_path();
    // A table with several files gets its splits once, for its first file.
    if (parquetFiles(tableDir.string()).front() != path) {
      return {};
    }
    return encoded::makeEncodedSplits(
        exec::test::kHiveConnectorId,
        tableDir.filename().string(),
        FLAGS_encoded_segments_per_split);
  }
  // TODO (dm): Figure out a way to enforce 1 split per file in
  // CudfHiveDataSource outside of this benchmark
  if (FLAGS_velox_cudf_table_scan) {
    // TODO (dm): Instead of this, we can maybe use
    // makeHiveConnectorSplits(vector<shared_ptr<TempFilePath>>&
    // filePaths)
    std::vector<std::shared_ptr<connector::ConnectorSplit>> result;
    auto temp = HiveConnectorTestBase::makeHiveConnectorSplits(
        path, 1, plan.dataFileFormat);
    for (auto& i : temp) {
      result.push_back(i);
    }
    return result;
  }

  return TpchBenchmark::listSplits(path, numSplitsPerFile, plan);
}

void CudfTpchBenchmark::runMain(std::ostream& out, RunStats& /*runStats*/) {
  if (FLAGS_io_meter_column_pct != 0) {
    LOG(ERROR)
        << "--io_meter_column_pct is not supported by velox_cudf_tpch_benchmark";
    exit(1);
  }
  if (FLAGS_run_query_verbose != -1) {
    executeQuery(FLAGS_run_query_verbose, /*verbose=*/true, out);
    return;
  }
  // Every query, in place of TpchBenchmark's folly benchmarks.
  for (int32_t queryId = 1; queryId <= 22; ++queryId) {
    out << "Running query " << queryId << std::endl;
    try {
      executeQuery(queryId, /*verbose=*/false, out);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Query " << queryId << " failed: " << e.what()
                 << ". Exiting";
      exit(1);
    }
  }
}

CudfTpchBenchmark::QueryRun CudfTpchBenchmark::runOnce(const TpchPlan& plan) {
  // Long enough for a task that has returned its last batch to free what it
  // spilled. A task that completes sooner is not kept waiting.
  constexpr uint64_t kCompletionWaitMicros{60'000'000};
  QueryRun queryRun;
  const auto start = std::chrono::steady_clock::now();
  const auto elapsedMicros = [&]() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now() - start)
        .count();
  };
  try {
    // What QueryBenchmarkBase::run() does for one repeat, keeping the cursor
    // of a run that fails so that its counters can still be read.
    CursorParameters params;
    params.maxDrivers = FLAGS_num_drivers;
    params.planNode = plan.plan;
    params.queryConfigs = queryConfigs_;
    configureCursorParameters(params);

    queryRun.cursor = TaskCursor::create(params);
    auto* task = queryRun.cursor->task().get();
    queryRun.cursor->start();
    for (const auto& [scanId, paths] : plan.dataFiles) {
      for (const auto& path : paths) {
        for (auto& split : listSplits(path, FLAGS_num_splits_per_file, plan)) {
          task->addSplit(scanId, exec::Split(std::move(split)));
        }
      }
      task->noMoreSplits(scanId);
    }
    queryRun.cursor->setNoMoreSplits();
    while (queryRun.cursor->moveNext()) {
      auto vector = queryRun.cursor->current();
      vector->loadedVector();
      queryRun.results.push_back(std::move(vector));
    }
    if (!waitForTaskCompletion(task, kCompletionWaitMicros)) {
      if (auto error = task->error()) {
        std::rethrow_exception(error);
      }
      VELOX_FAIL("Task did not finish: {}", task->toString());
    }
    queryRun.wallMicros = elapsedMicros();
  } catch (const std::exception& e) {
    queryRun.wallMicros = elapsedMicros();
    queryRun.failed = true;
    queryRun.error = failureReason(e);
    LOG(ERROR) << "Query terminated with: " << e.what();
    if (queryRun.cursor != nullptr) {
      // Operators add their counters to the task as their drivers close. A
      // driver stuck on its thread must not hang the benchmark, so both
      // waits are bounded.
      auto* task = queryRun.cursor->task().get();
      if (task->isRunning()) {
        task->requestAbort()
            .within(std::chrono::microseconds(kCompletionWaitMicros))
            .wait();
      }
      waitForTaskDriversToFinish(task, kCompletionWaitMicros);
    }
  }
  return queryRun;
}

void CudfTpchBenchmark::executeQuery(
    int32_t queryId,
    bool verbose,
    std::ostream& out) {
  auto plan = queryBuilder_->getQueryPlan(queryId);
  // A physical choice of this run, made on the plan the query builder
  // returns: which state the eligible decimal SUMs exchange. The scans, the
  // node ids and the result types stay as the query builder made them.
  if (compactDecimalSumStateEnabled()) {
    plan.plan = cudf_velox::CudfAggregationNode::rewritePlan(plan.plan);
  }
  const auto numCompactAggregations = countCompactAggregations(plan.plan);
  out << fmt::format(
             "Compact decimal SUM state of query {}: {} aggregations",
             queryId,
             numCompactAggregations)
      << std::endl;
  if (FLAGS_encoded) {
    setInputMode(InputMode::kEncoded);
    try {
      prepareEncodedInput(queryId, plan, out);
    } catch (const encoded::EncodedHostBudgetExceeded& e) {
      out << fmt::format(
                 "Encoded input of query {} does not fit the host cache: {}",
                 queryId,
                 e.what())
          << std::endl;
      if (FLAGS_input_overflow != "parquet") {
        VELOX_USER_FAIL(
            "Input of query {} does not fit the host cache limit of {} bytes",
            queryId,
            hostCacheLimitBytes_);
      }
      // The cached columns are of no use to a Parquet scan.
      encoded::EncodedColumnCache::instance().clear();
      setInputMode(InputMode::kParquet);
    }
  }
  const std::string input = inputMode_ == InputMode::kEncoded
      ? fmt::format("encoded cache ({})", FLAGS_encoded_memory)
      : std::string("parquet files");
  out << fmt::format("Input of query {}: {}", queryId, input) << std::endl;

  const bool writeFiles = !FLAGS_results_json_dir.empty();
  QueryRun queryRun;
  for (int32_t repeat = 0; repeat < FLAGS_num_repeats; ++repeat) {
    // The previous run's vectors must go before the cursor owning their pool.
    queryRun.results.clear();
    queryRun.cursor.reset();
    const auto arbitratorBefore = arbitratorStats();
    queryRun = runOnce(plan);

    // The run's wall time ended inside runOnce(): reading counters and
    // writing files from here on is not part of it.
    const auto arbitratorAfter = arbitratorStats();
    const bool failed = queryRun.failed;
    exec::Task* task =
        queryRun.cursor != nullptr ? queryRun.cursor->task().get() : nullptr;
    if (failed) {
      out << fmt::format(
                 "Repeat {} failed after us: {}", repeat, queryRun.wallMicros)
          << std::endl;
    } else {
      out << fmt::format(
                 "Repeat {} wall time us: {}", repeat, queryRun.wallMicros)
          << std::endl;
    }
    try {
      if (task != nullptr) {
        out << memoryLine(
                   repeat, task->queryCtx(), arbitratorBefore, arbitratorAfter)
            << std::endl;
        out << spillLine(repeat, task->taskStats()) << std::endl;
      }
      if (writeFiles) {
        const auto statsPath =
            fmt::format("{}/q{}.stats.json", repeatDir(repeat), queryId);
        writeRunStatsJson(
            statsPath,
            runStatsJson(
                queryId,
                repeat,
                queryRun.wallMicros,
                queryRun.error,
                input,
                numCompactAggregations,
                task,
                arbitratorBefore,
                arbitratorAfter));
        out << fmt::format("Repeat {} stats file: {}", repeat, statsPath)
            << std::endl;
        const auto repeatResultsPath =
            fmt::format("{}/q{}.json", repeatDir(repeat), queryId);
        if (failed) {
          // No result file may stand in for a failed run, including one an
          // earlier invocation left in this directory.
          std::error_code ignored;
          std::filesystem::remove(repeatResultsPath, ignored);
          std::filesystem::remove(
              fmt::format("{}/q{}.json", FLAGS_results_json_dir, queryId),
              ignored);
        } else {
          writeResultsJson(repeatResultsPath, queryRun.results);
        }
      }
    } catch (const std::exception& e) {
      // Reporting a failed run must not replace the reason it failed.
      if (!failed) {
        throw;
      }
      out << fmt::format(
                 "Repeat {} could not be fully reported: {}", repeat, e.what())
          << std::endl;
    }
    // A failed run fails the query: the runs after it are not attempted and
    // <dir>/q<N>.json is not written.
    VELOX_CHECK(
        !failed,
        "Query failed: query {}, repeat {}: {}",
        queryId,
        repeat,
        queryRun.error);
  }
  VELOX_CHECK_NOT_NULL(queryRun.cursor, "No run of the query: {}", queryId);

  if (writeFiles) {
    writeResultsJson(
        fmt::format("{}/q{}.json", FLAGS_results_json_dir, queryId),
        queryRun.results);
  }
  if (!verbose) {
    return;
  }
  if (FLAGS_include_results) {
    printRows(queryRun.results, out);
  }
  const auto stats = queryRun.cursor->task()->taskStats();
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
  out << exec::printPlanWithStats(*plan.plan, stats, FLAGS_include_custom_stats)
      << std::endl;
}

void CudfTpchBenchmark::shutdown() {
  if (auto resource = cudf_velox::cudfCustomMemoryResource()) {
    std::cout << "cuDF GPU memory arbitration stats: "
              << resource->arbitrator()->stats().toString() << std::endl;
  }
  encoded::EncodedColumnCache::instance().clear();
  cudf_velox::unregisterCudf();
  queryBuilder_.reset();
  encodedConnector_.reset();
  parquetConnector_.reset();
  TpchBenchmark::shutdown();
}

void CudfTpchBenchmark::configureCursorParameters(
    CursorParameters& params) const {
  params.queryCtx = core::QueryCtx::Builder()
                        .executor(driverExecutor_.get())
                        .asyncDataCache(cache::AsyncDataCache::getInstance())
                        .queryId("cudf-tpch")
                        .build();
  cudf_velox::attachCudfMemoryPool(*params.queryCtx);
}

int main(int argc, char** argv) {
  std::string kUsage(
      "This program benchmarks TPC-H queries. Run 'velox_cudf_tpch_benchmark -helpon=TpchBenchmark' for available options.\n");
  gflags::SetUsageMessage(kUsage);
  folly::Init init{&argc, &argv, false};
  benchmark = std::make_unique<CudfTpchBenchmark>();
  if (FLAGS_run_queries.empty()) {
    tpchBenchmarkMain();
    return 0;
  }
  std::vector<int32_t> queries;
  std::stringstream list(FLAGS_run_queries);
  for (std::string query; std::getline(list, query, ',');) {
    queries.push_back(std::stoi(query));
  }
  benchmark->initialize();
  // A query that fails does not stop the ones after it, but it fails the
  // process.
  std::vector<int32_t> failedQueries;
  folly::dynamic failures = folly::dynamic::array;
  for (const auto query : queries) {
    FLAGS_run_query_verbose = query;
    std::cout << "Running query " << query << std::endl;
    try {
      RunStats ignore;
      benchmark->runMain(std::cout, ignore);
    } catch (const std::exception& e) {
      failedQueries.push_back(query);
      folly::dynamic failure = folly::dynamic::object;
      failure["query"] = static_cast<int64_t>(query);
      failure["error"] = std::string(e.what());
      failures.push_back(std::move(failure));
      std::cout << "Query " << query << " failed: " << e.what() << std::endl;
    }
    std::cout << "Finished query " << query << std::endl;
  }
  benchmark->shutdown();
  if (!FLAGS_results_json_dir.empty()) {
    // Which queries ran and which failed, including a query that failed
    // before its first run and so has no stats file.
    folly::dynamic summary = folly::dynamic::object;
    summary["schema"] = "velox_cudf_tpch_benchmark.run_summary.v1";
    folly::dynamic ran = folly::dynamic::array;
    for (const auto query : queries) {
      ran.push_back(static_cast<int64_t>(query));
    }
    summary["queries"] = std::move(ran);
    summary["failed"] = std::move(failures);
    summary["num_repeats"] = static_cast<int64_t>(FLAGS_num_repeats);
    try {
      writeRunStatsJson(
          fmt::format("{}/run_summary.json", FLAGS_results_json_dir), summary);
    } catch (const std::exception& e) {
      std::cout << "Failed to write the run summary: " << e.what() << std::endl;
      return 1;
    }
  }
  if (!failedQueries.empty()) {
    std::cout << "Failed queries: " << folly::join(",", failedQueries)
              << std::endl;
    return 1;
  }
  std::cout << "All queries succeeded: " << queries.size() << std::endl;
  return 0;
}
