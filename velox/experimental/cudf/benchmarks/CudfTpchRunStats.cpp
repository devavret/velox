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

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/RuntimeMetrics.h"

#include <folly/json.h>

#include <unordered_map>

namespace facebook::velox::cudf_velox {
namespace {

const char* unitName(RuntimeCounter::Unit unit) {
  switch (unit) {
    case RuntimeCounter::Unit::kNone:
      return "none";
    case RuntimeCounter::Unit::kNanos:
      return "nanos";
    case RuntimeCounter::Unit::kBytes:
      return "bytes";
  }
  VELOX_UNREACHABLE();
}

folly::dynamic timingJson(const CpuWallTiming& timing) {
  folly::dynamic json = folly::dynamic::object;
  json["count"] = saturateCast(timing.count);
  json["wall_nanos"] = saturateCast(timing.wallNanos);
  json["cpu_nanos"] = saturateCast(timing.cpuNanos);
  return json;
}

folly::dynamic runtimeStatsJson(
    const std::unordered_map<std::string, RuntimeMetric>& runtimeStats) {
  folly::dynamic json = folly::dynamic::object;
  for (const auto& [name, metric] : runtimeStats) {
    json[name] = CudfTpchRunStats::toJson(metric);
  }
  return json;
}

} // namespace

// static
folly::dynamic CudfTpchRunStats::toJson(const RuntimeMetric& metric) {
  folly::dynamic json = folly::dynamic::object;
  json["unit"] = unitName(metric.unit);
  json["sum"] = metric.sum;
  json["count"] = saturateCast(metric.count);
  json["min"] = metric.min;
  json["max"] = metric.max;
  return json;
}

// static
folly::dynamic CudfTpchRunStats::toJson(
    const velox::exec::OperatorStats& stats) {
  folly::dynamic json = folly::dynamic::object;
  json["operator_id"] = static_cast<int64_t>(stats.operatorId);
  json["pipeline_id"] = static_cast<int64_t>(stats.pipelineId);
  json["plan_node_id"] = stats.planNodeId;
  json["operator_type"] = stats.operatorType;
  json["num_drivers"] = static_cast<int64_t>(stats.numDrivers);
  json["num_splits"] = stats.numSplits;

  json["raw_input_bytes"] = saturateCast(stats.rawInputBytes);
  json["raw_input_positions"] = saturateCast(stats.rawInputPositions);
  json["input_bytes"] = saturateCast(stats.inputBytes);
  json["input_positions"] = saturateCast(stats.inputPositions);
  json["input_vectors"] = saturateCast(stats.inputVectors);
  json["output_bytes"] = saturateCast(stats.outputBytes);
  json["output_positions"] = saturateCast(stats.outputPositions);
  json["output_vectors"] = saturateCast(stats.outputVectors);
  json["physical_written_bytes"] = saturateCast(stats.physicalWrittenBytes);

  json["add_input"] = timingJson(stats.addInputTiming);
  json["get_output"] = timingJson(stats.getOutputTiming);
  json["finish"] = timingJson(stats.finishTiming);
  json["is_blocked"] = timingJson(stats.isBlockedTiming);
  json["background"] = timingJson(stats.backgroundTiming);
  json["blocked_wall_nanos"] = saturateCast(stats.blockedWallNanos);

  // The operator's pool of the query's CPU memory root. GPU memory is in
  // "runtime_stats".
  folly::dynamic memory = folly::dynamic::object;
  memory["user_reservation_bytes"] =
      saturateCast(stats.memoryStats.userMemoryReservation);
  memory["revocable_reservation_bytes"] =
      saturateCast(stats.memoryStats.revocableMemoryReservation);
  memory["system_reservation_bytes"] =
      saturateCast(stats.memoryStats.systemMemoryReservation);
  memory["peak_user_reservation_bytes"] =
      saturateCast(stats.memoryStats.peakUserMemoryReservation);
  memory["peak_system_reservation_bytes"] =
      saturateCast(stats.memoryStats.peakSystemMemoryReservation);
  memory["peak_total_reservation_bytes"] =
      saturateCast(stats.memoryStats.peakTotalMemoryReservation);
  memory["num_allocations"] =
      saturateCast(stats.memoryStats.numMemoryAllocations);
  json["cpu_memory"] = std::move(memory);

  // What Velox's own operators spilled to files. GPU operators spill to pinned
  // host memory and report it in "runtime_stats", so these stay zero for them.
  folly::dynamic diskSpill = folly::dynamic::object;
  diskSpill["spilled_input_bytes"] = saturateCast(stats.spilledInputBytes);
  diskSpill["spilled_bytes"] = saturateCast(stats.spilledBytes);
  diskSpill["spilled_rows"] = saturateCast(stats.spilledRows);
  diskSpill["spilled_partitions"] =
      static_cast<int64_t>(stats.spilledPartitions);
  diskSpill["spilled_files"] = static_cast<int64_t>(stats.spilledFiles);
  json["disk_spill"] = std::move(diskSpill);

  json["runtime_stats"] = runtimeStatsJson(stats.runtimeStats);
  return json;
}

// static
folly::dynamic CudfTpchRunStats::toJson(const velox::exec::TaskStats& stats) {
  folly::dynamic json = folly::dynamic::object;
  json["num_total_splits"] = static_cast<int64_t>(stats.numTotalSplits);
  json["num_finished_splits"] = static_cast<int64_t>(stats.numFinishedSplits);
  json["num_total_drivers"] = saturateCast(stats.numTotalDrivers);
  json["num_completed_drivers"] = saturateCast(stats.numCompletedDrivers);
  json["execution_start_time_ms"] = saturateCast(stats.executionStartTimeMs);
  json["execution_end_time_ms"] = saturateCast(stats.executionEndTimeMs);
  json["end_time_ms"] = saturateCast(stats.endTimeMs);
  json["termination_time_ms"] = saturateCast(stats.terminationTimeMs);
  json["memory_reclaim_count"] = static_cast<int64_t>(stats.memoryReclaimCount);
  json["memory_reclaim_ms"] = saturateCast(stats.memoryReclaimMs);

  folly::dynamic pipelines = folly::dynamic::array;
  for (size_t pipelineId = 0; pipelineId < stats.pipelineStats.size();
       ++pipelineId) {
    const auto& pipeline = stats.pipelineStats[pipelineId];
    folly::dynamic pipelineJson = folly::dynamic::object;
    pipelineJson["pipeline_id"] = static_cast<int64_t>(pipelineId);
    pipelineJson["input_pipeline"] = pipeline.inputPipeline;
    pipelineJson["output_pipeline"] = pipeline.outputPipeline;

    folly::dynamic operators = folly::dynamic::array;
    for (const auto& operatorStats : pipeline.operatorStats) {
      operators.push_back(toJson(operatorStats));
    }
    pipelineJson["operators"] = std::move(operators);

    folly::dynamic drivers = folly::dynamic::array;
    for (const auto& driverStats : pipeline.driverStats) {
      folly::dynamic driverJson = folly::dynamic::object;
      driverJson["runtime_stats"] = runtimeStatsJson(driverStats.runtimeStats);
      drivers.push_back(std::move(driverJson));
    }
    pipelineJson["drivers"] = std::move(drivers);

    pipelines.push_back(std::move(pipelineJson));
  }
  json["pipelines"] = std::move(pipelines);
  return json;
}

// static
int64_t CudfTpchRunStats::operatorRuntimeStatSum(
    const velox::exec::TaskStats& stats,
    std::string_view name) {
  const std::string key{name};
  int64_t sum{0};
  for (const auto& pipeline : stats.pipelineStats) {
    for (const auto& operatorStats : pipeline.operatorStats) {
      const auto it = operatorStats.runtimeStats.find(key);
      if (it != operatorStats.runtimeStats.end()) {
        sum += it->second.sum;
      }
    }
  }
  return sum;
}

// static
int64_t CudfTpchRunStats::driverRuntimeStatSum(
    const velox::exec::TaskStats& stats,
    std::string_view name) {
  const std::string key{name};
  int64_t sum{0};
  for (const auto& pipeline : stats.pipelineStats) {
    for (const auto& driverStats : pipeline.driverStats) {
      const auto it = driverStats.runtimeStats.find(key);
      if (it != driverStats.runtimeStats.end()) {
        sum += it->second.sum;
      }
    }
  }
  return sum;
}

// static
std::string CudfTpchRunStats::serialize(const folly::dynamic& document) {
  folly::json::serialization_opts options;
  options.pretty_formatting = true;
  options.sort_keys = true;
  return folly::json::serialize(document, options);
}

} // namespace facebook::velox::cudf_velox
