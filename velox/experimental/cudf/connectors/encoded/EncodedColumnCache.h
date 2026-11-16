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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace facebook::velox::cudf_velox::connector::encoded {

/// How a column's values are stored. Follows DBCompiler's codec set, with the
/// frame of reference chosen per segment rather than per column.
enum class Codec : uint8_t {
  /// The cuDF column as read, used for high-cardinality strings.
  kPlain,
  /// Unsigned offsets from the segment minimum, bit-packed at the segment's
  /// width. Used for integers, dates and short decimals.
  kForBitpack,
  /// Distinct values plus one code per row, the codes stored as kForBitpack.
  /// Used for low-cardinality strings. The values are in order of first
  /// appearance.
  kDictionary,
};

/// Decode parameters of one segment of a bit-packed stream.
struct SegmentParams {
  int64_t base;
  uint64_t bitOffset;
  uint32_t bitWidth;
  uint32_t pad;
};

/// Thrown when an allocation would take the encoded cache past its host
/// memory limit. The input of the query does not fit; nothing is evicted or
/// redirected to make room.
class EncodedHostBudgetExceeded : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Page-locked host memory for host-resident encoded tables, with a hard limit
/// on the bytes outstanding. An allocation past the limit throws
/// EncodedHostBudgetExceeded. Device kernels can read and write the memory.
/// Copies share one limit and one set of counters.
class BoundedPinnedHostResource {
 public:
  /// 'limitBytes' of zero means no limit.
  explicit BoundedPinnedHostResource(uint64_t limitBytes = 0)
      : state_(std::make_shared<State>()) {
    state_->limitBytes = limitBytes;
  }

  void setLimit(uint64_t limitBytes) {
    state_->limitBytes = limitBytes;
  }

  uint64_t limitBytes() const {
    return state_->limitBytes.load();
  }

  uint64_t usedBytes() const {
    return state_->usedBytes.load();
  }

  /// The most bytes outstanding at once since the last resetPeak().
  uint64_t peakBytes() const {
    return state_->peakBytes.load();
  }

  void resetPeak() {
    state_->peakBytes = state_->usedBytes.load();
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment);

  void deallocate_sync(
      void* pointer,
      std::size_t bytes,
      std::size_t alignment) noexcept;

  void*
  allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment) {
    return allocate_sync(bytes, alignment);
  }

  void deallocate(
      cuda::stream_ref stream,
      void* pointer,
      std::size_t bytes,
      std::size_t alignment) noexcept {
    deallocate_sync(pointer, bytes, alignment);
  }

  bool operator==(const BoundedPinnedHostResource& other) const noexcept {
    return state_ == other.state_;
  }

  bool operator!=(const BoundedPinnedHostResource& other) const noexcept {
    return !(*this == other);
  }

  friend void get_property(
      const BoundedPinnedHostResource&,
      cuda::mr::device_accessible) noexcept {}

 private:
  struct State {
    std::atomic<uint64_t> limitBytes{0};
    std::atomic<uint64_t> usedBytes{0};
    std::atomic<uint64_t> peakBytes{0};
  };

  std::shared_ptr<State> state_;
};

/// One column of an encoded table. Every column is split at the same segment
/// boundaries, every 'segmentRows' rows.
struct EncodedColumn {
  std::string name;
  /// The type the column decodes to.
  cudf::data_type type;
  Codec codec;

  /// Device-resident kForBitpack values or kDictionary codes: packed 64-bit
  /// words, segments concatenated, each starting on a word boundary, plus one
  /// padding word. Empty in a host-resident table.
  rmm::device_uvector<uint64_t> packed;
  /// Device copy of the segment params. Empty in a host-resident table.
  rmm::device_uvector<SegmentParams> segmentParams;
  /// Per-segment minimum and maximum of the decoded integer value, for
  /// pruning. Empty for strings.
  std::vector<int64_t> segmentMin;
  std::vector<int64_t> segmentMax;
  std::vector<uint32_t> segmentBitWidth;

  /// Device-resident kPlain: the whole column. kDictionary: the distinct
  /// values, always in device memory.
  std::unique_ptr<cudf::column> values;

  /// Host-resident storage is kept in the pieces it was loaded in, so loading
  /// never holds a column twice. Every piece but the last holds whole
  /// segments. 'hostPieceBeginRow' has the first row of each piece.
  std::vector<int64_t> hostPieceBeginRow;
  /// Host-resident kForBitpack values or kDictionary codes: the packed words
  /// of each piece, without a padding word.
  std::vector<rmm::device_uvector<uint64_t>> hostPacked;
  /// First word of each piece of 'hostPacked' in the column's stream.
  std::vector<uint64_t> hostPackedBeginWord;
  /// Segment params of a host-resident column, 'bitOffset' counted across
  /// the pieces of 'hostPacked'.
  std::vector<SegmentParams> hostSegmentParams;
  /// Host-resident kPlain: the values of each piece.
  std::vector<std::unique_ptr<cudf::column>> hostValues;

  EncodedColumn(cuda::stream_ref stream, rmm::device_async_resource_ref mr);

  /// Bytes held by the column, in host and device memory.
  int64_t encodedBytes() const {
    return hostBytes() + deviceBytes();
  }

  /// Bytes held in the host storage resource.
  int64_t hostBytes() const;

  /// Bytes held in device memory.
  int64_t deviceBytes() const;
};

/// A table held in encoded form. It has the columns loaded so far, which can
/// be a subset of the columns of its files.
struct EncodedTable {
  std::string name;
  int64_t numRows{0};
  int32_t segmentRows{0};
  int32_t numSegments{0};
  /// The packed streams and kPlain value columns are in host memory, and
  /// scans stage them with stageSegments(). Dictionaries stay in device
  /// memory.
  bool hostResident{false};
  std::vector<std::unique_ptr<EncodedColumn>> columns;
  std::unordered_map<std::string, int32_t> columnIndex;

  bool hasColumn(const std::string& name) const {
    return columnIndex.count(name) != 0;
  }

  const EncodedColumn& column(const std::string& name) const;

  /// Number of rows in 'segment'; only the last segment is short.
  int32_t segmentSize(int32_t segment) const;

  int64_t hostBytes() const;

  int64_t deviceBytes() const;
};

/// Decodes 'column' for the rows of 'segments', in ascending segment order.
/// With 'rows', decodes only those rows, given as offsets into the
/// concatenation of 'segments'. 'segments' is a device copy of segment ids.
/// 'table' must be device-resident, or a table from stageSegments().
std::unique_ptr<cudf::column> decodeColumn(
    const EncodedTable& table,
    const EncodedColumn& column,
    std::span<const int32_t> segments,
    const rmm::device_uvector<int32_t>& deviceSegments,
    std::optional<cudf::column_view> rows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Copies 'columnNames' of a host-resident 'table' for 'segments' into device
/// memory from 'mr' with bulk copies on 'stream', merging adjacent segments
/// into one copy. 'segments' must be ascending with only the last one possibly
/// short. The result holds those segments in order as segments 0 to n-1, so
/// decodeColumn() reads it with segment ids 0 to n-1. Copies issued from
/// pinned host memory are asynchronous; keep the result alive until work on
/// 'stream' that reads it has completed.
std::unique_ptr<EncodedTable> stageSegments(
    const EncodedTable& table,
    const std::vector<std::string>& columnNames,
    std::span<const int32_t> segments,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Process-wide store of encoded tables.
class EncodedColumnCache {
 public:
  static EncodedColumnCache& instance();

  /// Reads 'columnNames' of the Parquet file(s) of table 'name' a few row
  /// groups at a time, in file order, and encodes them in segments of
  /// 'segmentRows' rows, so tables larger than device memory load; sort the
  /// files beforehand to cluster them. Columns that are already loaded are
  /// skipped, and the rest are added to the table. An empty 'columnNames'
  /// loads every column of the files. Only the named columns are read; no
  /// row is filtered.
  ///
  /// Integer-like columns are bit-packed. String columns take the dictionary
  /// codec while they have at most 65,536 distinct values, and fewer than one
  /// per 8 rows at the end. Other columns are kPlain.
  ///
  /// The packed streams and kPlain value columns are stored in memory from
  /// 'storageMr'. With 'hostResident' that is host memory device kernels can
  /// address, and the table is marked EncodedTable::hostResident.
  /// Dictionaries use 'mr'. If loading throws, the table keeps the columns it
  /// had before.
  void loadColumns(
      const std::string& name,
      const std::vector<std::string>& parquetFiles,
      const std::vector<std::string>& columnNames,
      int32_t segmentRows,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr,
      rmm::device_async_resource_ref storageMr,
      bool hostResident);

  /// Drops the columns of table 'name' that are not in 'keep', and the table
  /// if none are left.
  void retainColumns(
      const std::string& name,
      const std::unordered_set<std::string>& keep);

  /// Returns the table, or null if it is not loaded.
  const EncodedTable* findTable(const std::string& name) const;

  const EncodedTable& table(const std::string& name) const;

  std::vector<std::string> tableNames() const;

  /// Bytes the loaded tables hold in the host storage resource.
  int64_t hostBytes() const;

  /// Bytes the loaded tables hold in device memory.
  int64_t deviceBytes() const;

  void clear();

 private:
  std::unordered_map<std::string, std::unique_ptr<EncodedTable>> tables_;
};

} // namespace facebook::velox::cudf_velox::connector::encoded
