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
#include "velox/experimental/cudf/benchmarks/CudfTpchBenchmark.h"
#include "velox/experimental/cudf/connectors/encoded/EncodedConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/SuccinctPrinter.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/common/Options.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/sparksql/UnscaledValueFunction.h"

#include <cudf/unary.hpp>

#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <nvtx3/nvToolsExt.h>

#include <cuda_profiler_api.h>
#include <experimental/cudf/connectors/hive/CudfHiveConnector.h>
#include <fmt/format.h>
#include <folly/String.h>
#include <folly/json.h>

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>

DECLARE_int64(max_coalesced_bytes);
DECLARE_string(max_coalesced_distance_bytes);
DECLARE_int32(parquet_prefetch_rowgroups);
DECLARE_string(data_path);
DECLARE_int32(num_repeats);
DECLARE_int32(run_query_verbose);

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
    encoded,
    false,
    "Scan the tables from the encoded in-memory column cache instead of "
    "Parquet. The tables are loaded from --data_path a few row groups at a "
    "time, in file order, and encoded on the GPU before any query runs; sort "
    "the files beforehand to cluster them.");

DEFINE_bool(
    encoded_lz4_strings,
    false,
    "Compress host-resident plain string characters with lossless GPU LZ4 blocks");

DEFINE_bool(
    encoded_dictionary_filters,
    false,
    "Evaluate eligible predicates over dictionary keys before decoding row strings");

DEFINE_bool(
    encoded_plain_filter_views,
    false,
    "Filter staged host-resident batches on the plain columns they own instead of gathered copies");

DEFINE_bool(
    encoded_delta,
    false,
    "Use lossless delta coding when it shrinks a sorted integer segment");

DEFINE_bool(
    encoded_scale100_decimals,
    false,
    "Pack scale-2 decimal segments proven to hold multiples of 100 above their minimum as (value - minimum) / 100 when narrower");
DEFINE_bool(
    encoded_codec_stats,
    false,
    "Print per-column encoded sizes and segment codec choices");

DEFINE_string(
    encoded_memory,
    "device",
    "With --encoded, where the encoded tables are kept: device, or "
    "pinned_host to have scans copy each batch to the device before decoding.");

DEFINE_int32(
    encoded_segment_rows,
    1 << 20,
    "Rows per encoded segment, the unit of min/max pruning.");

DEFINE_int32(
    encoded_segments_per_split,
    4,
    "Segments per split, the unit of work handed to scan drivers. GPU scans "
    "batch segments across splits up to the cuDF batch-size targets.");

DEFINE_bool(
    sirius_q22_window,
    false,
    "Reuse Q22 filtered customers through a global SUM/COUNT window");

DEFINE_bool(
    sirius_q5_numeric_group,
    false,
    "Aggregate Q5 by nation key and materialize names afterward");

DEFINE_bool(
    sirius_q5_late_revenue,
    false,
    "Compute Q5 revenue after selective joins");

DEFINE_bool(
    sirius_q9_numeric_group,
    false,
    "Aggregate Q9 by nation key before materializing nation names");
DEFINE_bool(
    sirius_q9_partsupp_first,
    false,
    "Build Q9 partsupp restricted by part before probing lineitem");

DEFINE_bool(
    sirius_prune_scan_outputs,
    false,
    "Prune unused and filter-only columns from aligned TPC-H scan outputs");

DEFINE_bool(
    sirius_filter_only_scans,
    false,
    "Omit filter-only columns from optimized encoded scan outputs");

DEFINE_bool(
    sirius_q4_inner_distinct,
    false,
    "Apply Q4 orders restriction before distinct lineitem order keys");

DEFINE_bool(
    sirius_q15_window_max,
    false,
    "Reuse Q15 supplier revenues through a global MAX window");

DEFINE_bool(
    sirius_q21_pairs,
    false,
    "Evaluate Q21 with candidate late rows and one other-supplier scan");

DEFINE_bool(
    sirius_q18_unscaled_sum,
    false,
    "Use an exact unscaled Q18 sum after checking encoded value bounds");

DEFINE_bool(
    sirius_q17_window,
    false,
    "Compute the Q17 average over selected lineitems with a one-scan window plan");

DEFINE_bool(
    sirius_q21_order_summary,
    false,
    "Evaluate Q21 supplier-existence predicates from per-order summaries");

DEFINE_bool(
    sirius_q13_preaggregate,
    false,
    "Aggregate orders before the Q13 customer outer join");

DEFINE_bool(
    sirius_q10_preaggregate,
    false,
    "Sum Q10 revenue per order customer key before joining customer and nation payload");

DEFINE_bool(
    sirius_q10_topn,
    false,
    "Replace the full sort and limit of the existing Q10 plan with TopN");

DEFINE_bool(
    sirius_q17_filtered_aggregate,
    false,
    "Restrict Q17 average to the parts selected by its outer query");

DEFINE_int32(
    profile_iteration,
    -1,
    "Capture this zero-based repeat with CUDA profiler API");

DEFINE_string(
    results_json_dir,
    "",
    "Write full precision result rows to this directory after timing");

DEFINE_bool(
    sirius_plans,
    false,
    "Use query parameters matching the pinned Sirius SF100 run");

DEFINE_string(
    run_queries,
    "",
    "Comma-separated query numbers to run verbosely one after another in this "
    "process, initializing (and with --encoded, loading the tables) once. Each "
    "query's output starts with a 'Running query N' line. Overrides "
    "--run_query_verbose.");

namespace {

class ParityUnscaledValueFunction final : public cudf_velox::CudfFunction {
 public:
  cudf_velox::ColumnOrView eval(
      std::vector<cudf_velox::ColumnOrView>& inputs,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const override {
    const auto input = cudf_velox::asView(inputs.at(0));
    VELOX_CHECK(
        input.type().id() == cudf::type_id::DECIMAL64 ||
        input.type().id() == cudf::type_id::DECIMAL32);
    // Reinterpret the exact fixed-point representation. The owning result
    // copies it on the caller's stream; no temporary input view escapes.
    const auto repType = input.type().id() == cudf::type_id::DECIMAL32
        ? cudf::type_id::INT32
        : cudf::type_id::INT64;
    cudf::column_view rep(
        cudf::data_type{repType},
        input.size(),
        input.head<void>(),
        input.null_mask(),
        input.null_count(),
        input.offset());
    return cudf::cast(rep, cudf::data_type{cudf::type_id::INT64}, stream, mr);
  }
};

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

std::vector<std::string> parquetFiles(const std::string& tableDir) {
  std::vector<std::string> files;
  for (const auto& entry : std::filesystem::directory_iterator(tableDir)) {
    if (entry.is_regular_file()) {
      files.push_back(entry.path().string());
    }
  }
  std::sort(files.begin(), files.end());
  VELOX_CHECK(!files.empty(), "No data files in {}", tableDir);
  return files;
}

void loadEncodedTables() {
  VELOX_USER_CHECK(
      FLAGS_encoded_memory == "device" || FLAGS_encoded_memory == "pinned_host",
      "--encoded_memory must be device or pinned_host");
  cudf_velox::ensureCudaContextForThread();
  auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  const auto mr = cudf_velox::get_output_mr();
  auto& cache = encoded::EncodedColumnCache::instance();
  static rmm::mr::pinned_host_memory_resource pinnedHost;
  const bool hostStorage = FLAGS_encoded_memory == "pinned_host";
  for (const auto& table : kTables) {
    const auto start = std::chrono::steady_clock::now();
    cache.loadTable(
        table,
        parquetFiles(FLAGS_data_path + "/" + table),
        FLAGS_encoded_segment_rows,
        stream,
        mr,
        hostStorage ? rmm::device_async_resource_ref(pinnedHost) : mr,
        hostStorage,
        FLAGS_encoded_delta,
        FLAGS_encoded_lz4_strings,
        FLAGS_encoded_scale100_decimals);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();
    const auto& loaded = cache.table(table);
    if (FLAGS_encoded_codec_stats) {
      for (const auto& col : loaded.columns) {
        uint64_t bits = 0;
        size_t deltaSegments = 0;
        size_t scaledSegments = 0;
        for (size_t segment = 0; segment < col->segmentBitWidth.size();
             ++segment) {
          bits += static_cast<uint64_t>(col->segmentBitWidth[segment]) *
              loaded.segmentSize(segment);
          if (segment < col->segmentDelta.size() &&
              col->segmentDelta[segment] != 0) {
            ++deltaSegments;
          }
          if (segment < col->segmentScaled.size() &&
              col->segmentScaled[segment] != 0) {
            ++scaledSegments;
          }
        }
        std::cout
            << fmt::format(
                   "Encoded column {}.{}: codec={} bytes={} packed_bits_per_row={:.4f} delta_segments={}/{} lz4_chars={} scaled_segments={}/{}",
                   table,
                   col->name,
                   static_cast<int>(col->codec),
                   col->encodedBytes(),
                   loaded.numRows > 0
                       ? static_cast<double>(bits) / loaded.numRows
                       : 0.0,
                   deltaSegments,
                   col->segmentBitWidth.size(),
                   col->compressedChars != nullptr,
                   scaledSegments,
                   col->segmentBitWidth.size())
            << std::endl;
      }
    }
    int64_t bytes = 0;
    for (const auto& column : loaded.columns) {
      bytes += column->encodedBytes();
    }
    std::cout << fmt::format(
                     "Encoded {}: {} rows, {} segments, {} bytes, {} ms",
                     table,
                     loaded.numRows,
                     loaded.numSegments,
                     bytes,
                     ms)
              << std::endl;
  }
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

} // namespace

void CudfTpchBenchmark::initialize() {
  if (!FLAGS_cudf_properties.empty()) {
    cudf_velox::CudfConfig::getInstance().initialize(
        cudf_velox::loadPropertiesFile(FLAGS_cudf_properties));
  }

  TpchBenchmark::initialize();
  queryBuilder_ =
      std::make_shared<TpchQueryBuilder>(toFileFormat(FLAGS_data_format));
  queryBuilder_->initialize(FLAGS_data_path);
  if (FLAGS_sirius_plans) {
    siriusQueryBuilder_ = std::make_shared<SiriusTpchQueryBuilder>(
        toFileFormat(FLAGS_data_format));
    siriusQueryBuilder_->initialize(FLAGS_data_path);
  }

  if (FLAGS_encoded) {
    connector::ConnectorRegistry::global().erase(exec::test::kHiveConnectorId);
    connector::ConnectorRegistry::global().insert(
        exec::test::kHiveConnectorId,
        std::make_shared<encoded::EncodedConnector>(
            exec::test::kHiveConnectorId));
  } else if (FLAGS_velox_cudf_table_scan) {
    connector::ConnectorRegistry::global().erase(exec::test::kHiveConnectorId);

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
    auto cudfHiveConnector = cudfHiveFactory.newConnector(
        exec::test::kHiveConnectorId, cudfHiveProperties, ioExecutor_.get());
    connector::ConnectorRegistry::global().insert(
        cudfHiveConnector->connectorId(), cudfHiveConnector);
  }

  cudf_velox::registerCudf();
  if (FLAGS_encoded) {
    encoded::registerEncodedTableScanAdapter(
        FLAGS_encoded_dictionary_filters, FLAGS_encoded_plain_filter_views);
  }
  cudf_velox::registerPrestoFunctions(
      cudf_velox::CudfConfig::getInstance().functionNamePrefix);

  if (FLAGS_sirius_q18_unscaled_sum || FLAGS_sirius_q22_window) {
    const auto signatures = functions::sparksql::unscaledValueSignatures();
    exec::registerVectorFunction(
        "parity_unscaled_value",
        signatures,
        functions::sparksql::makeUnscaledValue());
    cudf_velox::registerCudfFunction(
        "parity_unscaled_value",
        [](const std::string&, const core::TypedExprPtr&, memory::MemoryPool*) {
          return std::make_shared<ParityUnscaledValueFunction>();
        },
        signatures);
  }

  queryConfigs_[cudf_velox::CudfFromVelox::kGpuBatchSizeRows] =
      std::to_string(FLAGS_cudf_gpu_batch_size_rows);
  queryConfigs_[core::QueryConfig::kMaxLocalExchangeBufferSize] =
      std::to_string(FLAGS_cudf_local_exchange_buffer_size);

  if (FLAGS_encoded) {
    loadEncodedTables();
  }
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
  if (FLAGS_encoded) {
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

void CudfTpchBenchmark::runMain(std::ostream& out, RunStats& runStats) {
  if (FLAGS_run_query_verbose == -1) {
    TpchBenchmark::runMain(out, runStats);
    return;
  }
  const auto plan = FLAGS_sirius_plans
      ? siriusQueryBuilder_->getQueryPlan(FLAGS_run_query_verbose)
      : queryBuilder_->getQueryPlan(FLAGS_run_query_verbose);
  const auto repeats = FLAGS_num_repeats;
  FLAGS_num_repeats = 1;
  std::pair<std::unique_ptr<exec::TaskCursor>, std::vector<RowVectorPtr>>
      result;
  for (int32_t repeat = 0; repeat < repeats; ++repeat) {
    // The previous run's vectors must go before the cursor owning their pool.
    result.second.clear();
    result.first.reset();
    const bool profile = repeat == FLAGS_profile_iteration;
    if (profile) {
      VELOX_CHECK(cudaProfilerStart() == cudaSuccess);
      const auto label = fmt::format(
          "TPC-H Q{} iteration {}", FLAGS_run_query_verbose, repeat);
      nvtxRangePushA(label.c_str());
    }
    const auto start = std::chrono::steady_clock::now();
    result = run(plan, queryConfigs_);
    if (result.first) {
      ensureTaskCompletion(result.first->task().get());
    }
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();
    if (profile) {
      nvtxRangePop();
      VELOX_CHECK(cudaProfilerStop() == cudaSuccess);
    }
    if (!result.first) {
      break;
    }
    out << fmt::format("Repeat {} wall time us: {}", repeat, us) << std::endl;
  }
  FLAGS_num_repeats = repeats;
  VELOX_CHECK_NOT_NULL(
      result.first, "Query {} failed", FLAGS_run_query_verbose);

  if (FLAGS_include_results) {
    printRows(result.second, out);
  }
  if (!FLAGS_results_json_dir.empty()) {
    std::filesystem::create_directories(FLAGS_results_json_dir);
    folly::dynamic doc = folly::dynamic::object;
    doc["rows"] = folly::dynamic::array;
    doc["types"] = folly::dynamic::array;
    doc["operators"] = folly::dynamic::array;
    const auto capturedStats = result.first->task()->taskStats();
    for (const auto& pipeline : capturedStats.pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        folly::dynamic info = folly::dynamic::object;
        info["type"] = op.operatorType;
        info["plan_node"] = op.planNodeId;
        info["input_rows"] = op.inputPositions;
        info["output_rows"] = op.outputPositions;
        doc["operators"].push_back(std::move(info));
      }
    }
    if (!result.second.empty()) {
      for (const auto& child : result.second.front()->children()) {
        doc["types"].push_back(child->type()->toString());
      }
    }
    for (const auto& batch : result.second) {
      for (vector_size_t row = 0; row < batch->size(); ++row) {
        folly::dynamic values = folly::dynamic::array;
        for (const auto& child : batch->children()) {
          if (child->isNullAt(row)) {
            values.push_back(nullptr);
          } else if (child->typeKind() == TypeKind::DOUBLE) {
            std::ostringstream value;
            value << std::setprecision(17)
                  << child->as<SimpleVector<double>>()->valueAt(row);
            values.push_back(value.str());
          } else {
            values.push_back(child->toString(row));
          }
        }
        doc["rows"].push_back(std::move(values));
      }
    }
    std::ofstream file(
        FLAGS_results_json_dir + "/q" +
        std::to_string(FLAGS_run_query_verbose) + ".json");
    file << folly::toJson(doc) << std::endl;
  }

  const auto stats = result.first->task()->taskStats();
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
  encoded::EncodedColumnCache::instance().clear();
  cudf_velox::unregisterCudf();
  siriusQueryBuilder_.reset();
  queryBuilder_.reset();
  TpchBenchmark::shutdown();
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
  for (const auto query : queries) {
    FLAGS_run_query_verbose = query;
    std::cout << "Running query " << query << std::endl;
    try {
      RunStats ignore;
      benchmark->runMain(std::cout, ignore);
    } catch (const std::exception& e) {
      std::cout << "Query " << query << " failed: " << e.what() << std::endl;
    }
    std::cout << "Finished query " << query << std::endl;
  }
  benchmark->shutdown();
  return 0;
}
