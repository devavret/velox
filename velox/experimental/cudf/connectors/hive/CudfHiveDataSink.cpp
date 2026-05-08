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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSink.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/Counters.h"
#include "velox/common/base/Fs.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/dwio/common/Options.h"
#include "velox/exec/OperatorUtils.h"

#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <string_view>
#include <system_error>

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

std::unordered_map<LocationHandle::TableType, std::string> tableTypeNames() {
  return {
      {LocationHandle::TableType::kNew, "kNew"},
      {LocationHandle::TableType::kExisting, "kExisting"},
  };
}

template <typename K, typename V>
std::unordered_map<V, K> invertMap(const std::unordered_map<K, V>& mapping) {
  std::unordered_map<V, K> inverted;
  for (const auto& [key, value] : mapping) {
    inverted.emplace(value, key);
  }
  return inverted;
}

uint64_t getFinishTimeSliceLimitMsFromCudfHiveConfig(
    const std::shared_ptr<const CudfHiveConfig>& config,
    const config::ConfigBase* sessions) {
  const uint64_t flushTimeSliceLimitMsFromConfig =
      config->sortWriterFinishTimeSliceLimitMs(sessions);
  // NOTE: if the flush time slice limit is set to 0, then we treat it as no
  // limit.
  return flushTimeSliceLimitMsFromConfig == 0
      ? std::numeric_limits<uint64_t>::max()
      : flushTimeSliceLimitMsFromConfig;
}

std::string makeUuid() {
  return boost::lexical_cast<std::string>(boost::uuids::random_generator()());
}

std::string toCudfIoPath(std::string path) {
  constexpr std::string_view kFilePrefix = "file:";
  constexpr std::string_view kS3APrefix = "s3a:";
  if (path.compare(0, kFilePrefix.size(), kFilePrefix) == 0) {
    std::string_view localPath(path);
    localPath.remove_prefix(kFilePrefix.size());
    if (localPath.compare(0, 2, "//") == 0) {
      localPath.remove_prefix(2);
      if (localPath.empty() || localPath.front() == '/') {
        return std::string(localPath);
      }
      const auto slash = localPath.find('/');
      VELOX_CHECK(
          slash != std::string_view::npos,
          "Unsupported cuDF Hive file URI without a path: {}",
          path);
      const auto authority = localPath.substr(0, slash);
      VELOX_CHECK(
          authority == "localhost",
          "Unsupported cuDF Hive file URI authority: {}",
          authority);
      return std::string(localPath.substr(slash));
    }
    return std::string(localPath);
  }
  if (path.compare(0, kS3APrefix.size(), kS3APrefix) == 0) {
    path.erase(kS3APrefix.size() - 2, 1);
  }
  return path;
}

bool isLocalIoPath(std::string_view path) {
  return path.find(':') == std::string_view::npos;
}

cudf::io::compression_type getCompressionType(
    facebook::velox::common::CompressionKind name) {
  using CompressionType = cudf::io::compression_type;

  static std::unordered_map<
      facebook::velox::common::CompressionKind,
      CompressionType> const kMap = {
      {facebook::velox::common::CompressionKind::CompressionKind_NONE,
       CompressionType::NONE},
      {facebook::velox::common::CompressionKind::CompressionKind_SNAPPY,
       CompressionType::SNAPPY},
      {facebook::velox::common::CompressionKind::CompressionKind_LZ4,
       CompressionType::LZ4},
      {facebook::velox::common::CompressionKind::CompressionKind_ZSTD,
       CompressionType::ZSTD}};

  VELOX_CHECK(
      kMap.find(name) != kMap.end(),
      "Unsupported compression type requested. Supported compression types are: "
      "NONE, SNAPPY, LZ4, ZSTD");

  return kMap.at(name);
}

std::shared_ptr<memory::MemoryPool> createSinkPool(
    const std::shared_ptr<memory::MemoryPool>& writerPool) {
  return writerPool->addLeafChild(fmt::format("{}.sink", writerPool->name()));
}

std::shared_ptr<memory::MemoryPool> createSortPool(
    const std::shared_ptr<memory::MemoryPool>& writerPool) {
  return writerPool->addLeafChild(fmt::format("{}.sort", writerPool->name()));
}

CudfHiveWriterParameters::UpdateMode toUpdateMode(
    LocationHandle::TableType tableType) {
  switch (tableType) {
    case LocationHandle::TableType::kNew:
      return CudfHiveWriterParameters::UpdateMode::kNew;
    case LocationHandle::TableType::kExisting:
      return CudfHiveWriterParameters::UpdateMode::kAppend;
    default:
      VELOX_UNSUPPORTED("Unsupported table type.");
  }
}

} // namespace

const std::string LocationHandle::tableTypeName(
    LocationHandle::TableType type) {
  static const auto kTableTypes = tableTypeNames();
  return kTableTypes.at(type);
}

LocationHandle::TableType LocationHandle::tableTypeFromName(
    const std::string& name) {
  static const auto kNameTableTypes = invertMap(tableTypeNames());
  return kNameTableTypes.at(name);
}

CudfHiveDataSink::CudfHiveDataSink(
    RowTypePtr inputType,
    std::shared_ptr<const CudfHiveInsertTableHandle> insertTableHandle,
    const ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy,
    const std::shared_ptr<const CudfHiveConfig>& parquetConfig)
    : inputType_(std::move(inputType)),
      insertTableHandle_(std::move(insertTableHandle)),
      connectorQueryCtx_(connectorQueryCtx),
      commitStrategy_(commitStrategy),
      parquetConfig_(parquetConfig),
      spillConfig_(connectorQueryCtx->spillConfig()),
      sortWriterFinishTimeSliceLimitMs_(
          getFinishTimeSliceLimitMsFromCudfHiveConfig(
              parquetConfig_,
              connectorQueryCtx->sessionProperties())) {
  VELOX_USER_CHECK(
      (commitStrategy_ == CommitStrategy::kNoCommit) ||
          (commitStrategy_ == CommitStrategy::kTaskCommit),
      "Unsupported commit strategy: {}",
      CommitStrategyName::toName(commitStrategy_));

  const auto& writerOptions = dynamic_cast<CudfHiveWriterOptions*>(
      insertTableHandle_->writerOptions().get());

  if (writerOptions != nullptr) {
    sortingColumns_ = std::move(writerOptions->sortingColumns);
  }
}

void CudfHiveDataSink::appendData(RowVectorPtr input) {
  checkRunning();

  // Convert the input RowVectorPtr to cudf::table
  auto stream = cudfGlobalStreamPool().get_stream();
  auto cudfInput =
      with_arrow::toCudfTable(input, input->pool(), stream, get_temp_mr());
  stream.synchronize();
  VELOX_CHECK_NOT_NULL(
      cudfInput, "Failed to convert input RowVectorPtr to cudf::table");

  // Check if the writer doesn't already exist
  if (writer_ == nullptr) {
    writer_ = createCudfWriter(cudfInput->view(), stream);
  }

  // Write the table to the sink
  writer_->write(cudfInput->view());
  writerInfo_->inputSizeInBytes += input->estimateFlatSize();
  writerInfo_->numWrittenRows += input->size();
}

std::unique_ptr<cudf::io::chunked_parquet_writer>
CudfHiveDataSink::createCudfWriter(
    cudf::table_view cudfTable,
    rmm::cuda_stream_view stream) {
  // Create a table_input_metadata from the input
  auto tableInputMetadata = createCudfTableInputMetadata(cudfTable);

  auto compressionKind =
      getCompressionType(insertTableHandle_->compressionKind().value_or(
          facebook::velox::common::CompressionKind::CompressionKind_NONE));

  // Create a sink and writer
  const auto& locationHandle = insertTableHandle_->locationHandle();
  const auto targetFileName = locationHandle->targetFileName().empty()
      ? fmt::format("{}{}", makeUuid(), ".parquet")
      : locationHandle->targetFileName();

  auto writerParameters = CudfHiveWriterParameters(
      toUpdateMode(locationHandle->tableType()),
      targetFileName,
      locationHandle->targetPath(),
      std::nullopt,
      locationHandle->writePath());

  makeWriterOptions(writerParameters);

  const auto writeDirectory =
      toCudfIoPath(writerInfo_->writerParameters.writeDirectory());
  if (isLocalIoPath(writeDirectory)) {
    std::error_code error;
    fs::create_directories(writeDirectory, error);
    VELOX_CHECK(
        !error || fs::exists(writeDirectory),
        "Failed to create cuDF Hive write directory {}: {}",
        writeDirectory,
        error.message());
  }
  const auto writePath =
      (fs::path(writeDirectory) / writerInfo_->writerParameters.writeFileName())
          .string();

  // Create writer options for the given sink
  const auto sinkInfo = cudf::io::sink_info(writePath);
  auto cudfWriterOptions =
      cudf::io::chunked_parquet_writer_options::builder(sinkInfo)
          .metadata(tableInputMetadata)
          .utc_timestamps(parquetConfig_->writeTimestampsAsUTC())
          .write_arrow_schema(parquetConfig_->writeArrowSchema())
          .write_v2_headers(parquetConfig_->writev2PageHeaders())
          .compression(compressionKind)
          .build();

  const auto& writerOptions = dynamic_cast<CudfHiveWriterOptions*>(
      insertTableHandle_->writerOptions().get());

  // If non-null writerOptions were passed, pass them to the chunked parquet
  // writer options
  if (writerOptions != nullptr) {
    // Set encoding for all columns
    std::for_each(
        tableInputMetadata.column_metadata.begin(),
        tableInputMetadata.column_metadata.end(),
        [=](auto& colMeta) { colMeta.set_encoding(writerOptions->encoding); });

    cudfWriterOptions.set_row_group_size_bytes(
        writerOptions->rowGroupSizeBytes);
    cudfWriterOptions.set_row_group_size_rows(writerOptions->rowGroupSizeRows);
    cudfWriterOptions.set_max_page_size_bytes(writerOptions->maxPageSizeBytes);
    cudfWriterOptions.set_max_page_size_rows(writerOptions->maxPageSizeRows);
    cudfWriterOptions.set_dictionary_policy(writerOptions->dictionaryPolicy);
    cudfWriterOptions.set_max_dictionary_size(writerOptions->maxDictionarySize);
    cudfWriterOptions.enable_int96_timestamps(
        writerOptions->writeTimestampsAsInt96);

    // Enable if enabled in the session or the writerOptions
    cudfWriterOptions.enable_utc_timestamps(
        parquetConfig_->writeTimestampsAsUTC() or
        writerOptions->writeTimestampsAsUTC);
    cudfWriterOptions.enable_write_arrow_schema(
        parquetConfig_->writeArrowSchema() or writerOptions->writeArrowSchema);
    cudfWriterOptions.enable_write_v2_headers(
        parquetConfig_->writev2PageHeaders() or writerOptions->v2PageHeaders);
    cudfWriterOptions.set_stats_level(writerOptions->statsLevel);

    if (writerOptions->maxPageFragmentSize.has_value()) {
      cudfWriterOptions.set_max_page_fragment_size(
          writerOptions->maxPageFragmentSize.value());
    }
    // Get compression stats if needed
    if (writerOptions->compressionStats != nullptr) {
      cudfWriterOptions.set_compression_statistics(
          writerOptions->compressionStats);
    }
    // Write sorting columns if available
    if (!sortingColumns_.empty()) {
      cudfWriterOptions.set_sorting_columns(sortingColumns_);
    }
  }

  return std::make_unique<cudf::io::chunked_parquet_writer>(
      cudfWriterOptions, stream);
}

cudf::io::table_input_metadata CudfHiveDataSink::createCudfTableInputMetadata(
    cudf::table_view cudfTable) {
  auto tableInputMetadata = cudf::io::table_input_metadata(cudfTable);
  auto inputColumns = insertTableHandle_->inputColumns();

  // Check if equal number of columns in the input and
  // CudfHiveInsertTableHandle
  VELOX_CHECK_EQ(
      tableInputMetadata.column_metadata.size(),
      inputColumns.size(),
      "Unequal number of columns in the input and CudfHiveInsertTableHandle");

  std::function<void(
      cudf::io::column_in_metadata&, const CudfHiveColumnHandle&)>
      setColumnName = [&](cudf::io::column_in_metadata& colMeta,
                          const CudfHiveColumnHandle& columnHandle) {
        // Check if equal number of children
        const auto& childrenHandles = columnHandle.children();

        // Warn if the mismatch in the number of child cols in CudfHive
        // table_metadata and columnHandles
        if (colMeta.num_children() != childrenHandles.size()) {
          LOG(WARNING) << fmt::format(
              "({} vs {}): Unequal number of child columns in CudfHive table_metadata and ColumnHandles",
              colMeta.num_children(),
              childrenHandles.size());
        }

        // Set children's names
        for (int32_t i = 0; i <
             std::min<int32_t>(colMeta.num_children(), childrenHandles.size());
             ++i) {
          setColumnName(colMeta.child(i), childrenHandles[i]);
        }
        // Set this column's name
        colMeta.set_name(columnHandle.name());
      };

  // Set names for all columns and their children
  for (int32_t i = 0; i < tableInputMetadata.column_metadata.size(); ++i) {
    setColumnName(tableInputMetadata.column_metadata[i], *inputColumns[i]);
  }

  return tableInputMetadata;
}

std::string CudfHiveDataSink::stateString(State state) {
  switch (state) {
    case State::kRunning:
      return "RUNNING";
    case State::kFinishing:
      return "FLUSHING";
    case State::kClosed:
      return "CLOSED";
    case State::kAborted:
      return "ABORTED";
    default:
      VELOX_UNREACHABLE("BAD STATE: {}", static_cast<int>(state));
  }
}

DataSink::Stats CudfHiveDataSink::stats() const {
  Stats stats;
  if (state_ == State::kAborted) {
    return stats;
  }

  if (ioStatistics_ == nullptr) {
    return stats;
  }

  int64_t numWrittenBytes{0};
  int64_t writeIOTimeUs{0};

  numWrittenBytes += ioStatistics_->rawBytesWritten();
  writeIOTimeUs += ioStatistics_->writeIOTimeUs();

  stats.numWrittenBytes = numWrittenBytes;
  stats.writeIOTimeUs = writeIOTimeUs;

  if (state_ != State::kClosed) {
    return stats;
  }

  if (writerInfo_ == nullptr) {
    return stats;
  }

  stats.numWrittenFiles = 1;
  if (!writerInfo_->spillStats->empty()) {
    stats.spillStats += *writerInfo_->spillStats;
  }

  return stats;
}

void CudfHiveDataSink::setState(State newState) {
  checkStateTransition(state_, newState);
  state_ = newState;
}

/// Validates the state transition from 'oldState' to 'newState'.
void CudfHiveDataSink::checkStateTransition(State oldState, State newState) {
  switch (oldState) {
    case State::kRunning:
      if (newState == State::kAborted || newState == State::kFinishing) {
        return;
      }
      break;
    case State::kFinishing:
      if (newState == State::kAborted || newState == State::kClosed ||
          // The finishing state is reentry state if we yield in the
          // middle of finish processing if a single run takes too long.
          newState == State::kFinishing) {
        return;
      }
      [[fallthrough]];
    case State::kAborted:
    case State::kClosed:
    default:
      break;
  }
  VELOX_FAIL("Unexpected state transition from {} to {}", oldState, newState);
}

bool CudfHiveDataSink::finish() {
  setState(State::kFinishing);
  return true;
}

std::vector<std::string> CudfHiveDataSink::close() {
  setState(State::kClosed);
  closeInternal();

  std::vector<std::string> partitionUpdates{};
  if (writerInfo_ == nullptr) {
    return partitionUpdates;
  }

  partitionUpdates.reserve(1);
  // clang-format off
    auto partitionUpdateJson = folly::toJson(
     folly::dynamic::object
        ("name", "")
        ("updateMode", CudfHiveWriterParameters::updateModeToString(
          writerInfo_->writerParameters.updateMode()))
        ("writePath", writerInfo_->writerParameters.writeDirectory())
        ("targetPath", writerInfo_->writerParameters.targetDirectory())
        ("fileWriteInfos", folly::dynamic::array(
          folly::dynamic::object
            ("writeFileName", writerInfo_->writerParameters.writeFileName())
            ("targetFileName", writerInfo_->writerParameters.targetFileName())
            ("fileSize", ioStatistics_->rawBytesWritten())))
        ("rowCount", writerInfo_->numWrittenRows)
        ("inMemoryDataSizeInBytes", writerInfo_->inputSizeInBytes)
        ("onDiskDataSizeInBytes", ioStatistics_->rawBytesWritten())
        ("containsNumberedFileNames", true));
  // clang-format on
  partitionUpdates.emplace_back(partitionUpdateJson);

  return partitionUpdates;
}

void CudfHiveDataSink::abort() {
  setState(State::kAborted);
  closeInternal();
}

void CudfHiveDataSink::closeInternal() {
  VELOX_CHECK_NE(state_, State::kRunning);
  VELOX_CHECK_NE(state_, State::kFinishing);

  TestValue::adjust(
      "facebook::velox::connector::hive::CudfHiveDataSink::closeInternal",
      this);

  if (writer_ == nullptr) {
    return;
  }

  writer_->close();
  if (ioStatistics_ != nullptr && writerInfo_ != nullptr) {
    const auto writeDirectory =
        toCudfIoPath(writerInfo_->writerParameters.writeDirectory());
    if (isLocalIoPath(writeDirectory)) {
      std::error_code error;
      const auto writePath = fs::path(writeDirectory) /
          writerInfo_->writerParameters.writeFileName();
      const auto fileSize = fs::file_size(writePath, error);
      if (!error) {
        const auto rawBytesWritten = ioStatistics_->rawBytesWritten();
        if (fileSize > rawBytesWritten) {
          ioStatistics_->incRawBytesWritten(
              static_cast<int64_t>(fileSize - rawBytesWritten));
        }
      }
    }
  }

  // Reset the unique pointers to Cudf writer and options
  writer_.reset();
}

std::shared_ptr<memory::MemoryPool> CudfHiveDataSink::createWriterPool() {
  auto* connectorPool = connectorQueryCtx_->connectorMemoryPool();
  return connectorPool->addAggregateChild(
      fmt::format("{}.{}", connectorPool->name(), "parquet-writer"));
}

void CudfHiveDataSink::makeWriterOptions(
    CudfHiveWriterParameters writerParameters) {
  auto writerPool = createWriterPool();
  auto sinkPool = createSinkPool(writerPool);
  std::shared_ptr<memory::MemoryPool> sortPool{nullptr};
  if (sortWrite()) {
    sortPool = createSortPool(writerPool);
  }

  writerInfo_ = std::make_shared<CudfHiveWriterInfo>(
      std::move(writerParameters),
      std::move(writerPool),
      std::move(sinkPool),
      std::move(sortPool));

  ioStatistics_ = std::make_shared<io::IoStatistics>();

  // Take the writer options provided by the user as a starting point,
  // or allocate a new one.
  auto options = insertTableHandle_->writerOptions();
  if (!options) {
    options = std::make_unique<CudfHiveWriterOptions>();
  }

  const auto* connectorSessionProperties =
      connectorQueryCtx_->sessionProperties();

  if (options->memoryPool == nullptr) {
    options->memoryPool = writerInfo_->writerPool.get();
  }

  if (!options->compressionKind) {
    options->compressionKind = insertTableHandle_->compressionKind();
  }

  const auto& sessionTimeZoneName = connectorQueryCtx_->sessionTimezone();
  if (!sessionTimeZoneName.empty()) {
    options->sessionTimezoneName = sessionTimeZoneName;
  }
  options->adjustTimestampToTimezone =
      connectorQueryCtx_->adjustTimestampToTimezone();
}

folly::dynamic CudfHiveInsertTableHandle::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "CudfHiveInsertTableHandle";
  folly::dynamic arr = folly::dynamic::array;
  for (const auto& ic : inputColumns_) {
    arr.push_back(ic->serialize());
  }

  obj["inputColumns"] = arr;
  obj["locationHandle"] = locationHandle_->serialize();
  obj["tableStorageFormat"] = dwio::common::toString(storageFormat_);

  if (compressionKind_.has_value()) {
    obj["compressionKind"] = common::compressionKindToString(*compressionKind_);
  }

  folly::dynamic params = folly::dynamic::object;
  for (const auto& [key, value] : serdeParameters_) {
    params[key] = value;
  }
  obj["serdeParameters"] = params;

  return obj;
}

CudfHiveInsertTableHandlePtr CudfHiveInsertTableHandle::create(
    const folly::dynamic& obj) {
  auto inputColumns =
      ISerializable::deserialize<std::vector<CudfHiveColumnHandle>>(
          obj["inputColumns"]);
  auto locationHandle =
      ISerializable::deserialize<LocationHandle>(obj["locationHandle"]);
  std::optional<common::CompressionKind> compressionKind = std::nullopt;
  if (obj.count("compressionKind") > 0) {
    compressionKind =
        common::stringToCompressionKind(obj["compressionKind"].asString());
  }
  std::unordered_map<std::string, std::string> serdeParameters;
  if (obj.count("serdeParameters") > 0) {
    for (const auto& pair : obj["serdeParameters"].items()) {
      serdeParameters.emplace(pair.first.asString(), pair.second.asString());
    }
  }
  return std::make_shared<CudfHiveInsertTableHandle>(
      inputColumns, locationHandle, compressionKind, serdeParameters);
}

std::string CudfHiveInsertTableHandle::toString() const {
  std::ostringstream out;
  out << "CudfHiveInsertTableHandle ["
      << dwio::common::toString(storageFormat_);
  if (compressionKind_.has_value()) {
    out << " " << common::compressionKindToString(compressionKind_.value());
  } else {
    out << " none";
  }
  out << "], [inputColumns: [";
  for (const auto& i : inputColumns_) {
    out << " " << i->toString();
  }
  out << " ], locationHandle: " << locationHandle_->toString();

  out << "]";
  return out.str();
}

void CudfHiveInsertTableHandle::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("HiveInsertTableHandle", CudfHiveInsertTableHandle::create);
  registry.Register(
      "CudfHiveInsertTableHandle", CudfHiveInsertTableHandle::create);
}

std::string LocationHandle::toString() const {
  return fmt::format(
      "LocationHandle [targetPath: {}, writePath: {}, tableType: {},",
      targetPath_,
      writePath_,
      tableTypeName(tableType_));
}

folly::dynamic LocationHandle::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "LocationHandle";
  obj["targetPath"] = targetPath_;
  obj["writePath"] = writePath_;
  obj["tableType"] = tableTypeName(tableType_);
  obj["targetFileName"] = targetFileName_;
  return obj;
}

LocationHandlePtr LocationHandle::create(const folly::dynamic& obj) {
  auto targetPath = obj["targetPath"].asString();
  auto writePath =
      obj.count("writePath") > 0 ? obj["writePath"].asString() : targetPath;
  auto tableType = tableTypeFromName(obj["tableType"].asString());
  auto targetFileName =
      obj.count("targetFileName") > 0 ? obj["targetFileName"].asString() : "";
  return std::make_shared<LocationHandle>(
      targetPath, writePath, tableType, targetFileName);
}

} // namespace facebook::velox::cudf_velox::connector::hive
