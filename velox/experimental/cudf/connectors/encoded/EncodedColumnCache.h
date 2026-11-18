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

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
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
  /// Sorted distinct values plus one code per row, the codes stored as
  /// kForBitpack. Used for low-cardinality strings.
  kDictionary,
};

/// Decode parameters of one segment of a bit-packed stream. A frame-of-
/// reference segment packs value - 'base', 'base' being its minimum. A delta
/// segment, whose 'flags' has kSegmentDelta, packs value[i] - value[i - 1],
/// zero for its first row, 'base' being that first value. Differences and
/// their sums wrap in unsigned 64-bit arithmetic, so every segment decodes to
/// the exact bits it was encoded from. A scaled segment, whose 'flags' has
/// kSegmentScale100, is a frame-of-reference segment that packs
/// (value - 'base') / 100: it is chosen only when every such difference was
/// proven a multiple of 100, so 'base' + packed * 100 is each value again.
struct SegmentParams {
  int64_t base;
  uint64_t bitOffset;
  uint32_t bitWidth;
  uint32_t flags;
};

/// SegmentParams::flags bit of a delta segment. Only a non-decreasing segment
/// whose differences need fewer bits than its range is delta-coded.
constexpr uint32_t kSegmentDelta = 1;

/// SegmentParams::flags bit of a scaled frame-of-reference segment of a
/// DECIMAL32 or DECIMAL64 column with scale -2. Never set with kSegmentDelta.
constexpr uint32_t kSegmentScale100 = 2;

/// The characters of a kPlain strings column of a host-resident table,
/// LZ4-compressed in blocks of 'blockBytes' uncompressed bytes. A scan copies
/// only the compressed blocks its rows cover and decompresses them on the GPU.
struct CompressedChars {
  /// Uncompressed bytes per block; only the last block is shorter.
  int64_t blockBytes{0};
  /// Uncompressed bytes of all blocks, the column's character count.
  int64_t numChars{0};
  /// The compressed blocks back to back, each starting at a multiple of 8
  /// bytes, in the table's storage memory.
  std::unique_ptr<rmm::device_buffer> blocks;
  /// Start in 'blocks' and compressed size of each block.
  std::vector<int64_t> blockStarts;
  std::vector<int64_t> blockSizes;
};

/// Checks the LZ4 blocks decompressed for a staged column before it is read.
class DecompressionCheck;

/// One column of an encoded table. Every column is split at the same segment
/// boundaries, every 'segmentRows' rows.
struct EncodedColumn {
  std::string name;
  /// The type the column decodes to.
  cudf::data_type type;
  Codec codec;

  /// kForBitpack values or kDictionary codes: packed 64-bit words, segments
  /// concatenated, each starting on a word boundary, plus one padding word.
  rmm::device_uvector<uint64_t> packed;
  rmm::device_uvector<SegmentParams> segmentParams;
  /// Per-segment minimum and maximum of the decoded integer value, for
  /// pruning. Empty for strings.
  std::vector<int64_t> segmentMin;
  std::vector<int64_t> segmentMax;
  std::vector<uint32_t> segmentBitWidth;
  /// Per segment of a bit-packed stream, 1 if it is delta-coded, so decoding
  /// knows without reading device params. Empty without bit-packed streams.
  std::vector<uint8_t> segmentDelta;
  /// Per segment of a bit-packed stream, 1 if it is scaled (kSegmentScale100).
  /// For statistics; decoding reads the flag from the segment params.
  std::vector<uint8_t> segmentScaled;

  /// kPlain: the whole column. kDictionary: the sorted distinct values.
  std::unique_ptr<cudf::column> values;

  /// kPlain strings of a host-resident table loaded with LZ4 compression,
  /// where compression shrinks them: 'values' is then null, 'stringOffsets'
  /// holds the offsets (INT32 or INT64, one per row plus one) and
  /// 'compressedChars' the characters. stageSegments() stages a complete
  /// strings column of the staged rows from them.
  std::unique_ptr<cudf::column> stringOffsets;
  std::unique_ptr<CompressedChars> compressedChars;
  /// Set on a column staged from 'compressedChars'. decodeColumn() verifies
  /// the status and size of each block decompressed for it before reading it.
  std::shared_ptr<DecompressionCheck> decompressionCheck;

  /// Host copy of 'segmentParams' of a host-resident table.
  std::vector<SegmentParams> hostSegmentParams;

  EncodedColumn(cuda::stream_ref stream, rmm::device_async_resource_ref mr);

  /// Device bytes held by the column.
  int64_t encodedBytes() const;
};

/// A table held in device memory in encoded form.
struct EncodedTable {
  std::string name;
  int64_t numRows{0};
  int32_t segmentRows{0};
  int32_t numSegments{0};
  /// Set by EncodedColumnCache::loadTable() with 'hostResident': the packed
  /// streams and kPlain value columns are in host memory, and scans stage
  /// them with stageSegments().
  /// Dictionaries and segment params stay in device memory, and
  /// EncodedColumn::hostSegmentParams holds a host copy of the params.
  bool hostResident{false};
  std::vector<std::unique_ptr<EncodedColumn>> columns;
  std::unordered_map<std::string, int32_t> columnIndex;

  const EncodedColumn& column(const std::string& name) const;

  /// Number of rows in 'segment'; only the last segment is short.
  int32_t segmentSize(int32_t segment) const;
};

/// Decodes dictionary indices, preserving the dictionary's index domain.
/// Used to apply predicates to dictionary values once rather than decoding
/// a string for every row. The returned numeric column owns its buffers.
std::unique_ptr<cudf::column> decodeDictionaryCodes(
    const EncodedTable& table,
    const EncodedColumn& column,
    std::span<const int32_t> segments,
    const rmm::device_uvector<int32_t>& deviceSegments,
    std::optional<cudf::column_view> rows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Decodes 'column' for the rows of 'segments', in ascending segment order.
/// With 'rows', decodes only those rows, given as offsets into the
/// concatenation of 'segments'. 'segments' is a device copy of segment ids.
/// A delta segment decodes by a prefix sum over its rows, so with 'rows' and a
/// delta segment among 'segments', every row of 'segments' is decoded into a
/// temporary first and 'rows' gathered from it. A column holding
/// 'compressedChars' is decoded only once staged by stageSegments(), and a
/// staged one only after its decompression is verified.
std::unique_ptr<cudf::column> decodeColumn(
    const EncodedTable& table,
    const EncodedColumn& column,
    std::span<const int32_t> segments,
    const rmm::device_uvector<int32_t>& deviceSegments,
    std::optional<cudf::column_view> rows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// The values of 'column', a kPlain column of a table returned by
/// stageSegments(), as a view of every staged row in staged order, or nullopt
/// if 'column' does not hold plain values. Verifies the decompression the
/// column was staged with first, throwing if a block failed or decompressed to
/// an unexpected size. The view is valid while the staged table lives, and
/// work reading it must be ordered after the staging.
std::optional<cudf::column_view> stagedPlainValues(const EncodedColumn& column);

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

/// Process-wide store of encoded tables, resident in device memory.
class EncodedColumnCache {
 public:
  static EncodedColumnCache& instance();

  /// Reads the Parquet file(s) of a table a few row groups at a time, in file
  /// order, and encodes it in segments of 'segmentRows' rows, so tables larger
  /// than device memory load; sort the files beforehand to cluster them.
  /// Integer-like columns are bit-packed. String columns take the dictionary
  /// codec when they have at most 65,536 distinct values and fewer than one
  /// per 8 rows: each piece read is coded against its own distinct values and
  /// recoded to the table's dictionary at the end. Other columns are kPlain.
  /// The packed streams, those interim codes and kPlain value columns are
  /// stored in memory from 'storageMr'. With 'hostResident' that is pinned
  /// host memory and the table is marked EncodedTable::hostResident.
  /// Dictionaries and segment params use 'mr'. With
  /// 'deltaEncodeSortedSegments', a segment of an integer-like column that is
  /// non-decreasing and whose consecutive differences need fewer bits than its
  /// range stores those differences (kSegmentDelta); dictionary codes and every
  /// other segment keep the frame of reference. Off by default. With
  /// 'compressPlainStrings' and 'hostResident', the characters of each kPlain
  /// strings column are LZ4-compressed with nvCOMP in 64 KiB blocks when the
  /// blocks and their metadata are smaller; offsets stay uncompressed. Off by
  /// default. With 'scale100Decimals', a segment of a DECIMAL32 or DECIMAL64
  /// column with scale -2 whose values minus the segment minimum are all
  /// proven on the GPU to be multiples of 100 packs those differences divided
  /// by 100 (kSegmentScale100) when that is narrower than ordinary frame of
  /// reference and no wider than delta coding. Segment minimums and maximums
  /// stay those of the values. Off by default.
  void loadTable(
      const std::string& name,
      const std::vector<std::string>& parquetFiles,
      int32_t segmentRows,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr,
      rmm::device_async_resource_ref storageMr,
      bool hostResident,
      bool deltaEncodeSortedSegments = false,
      bool compressPlainStrings = false,
      bool scale100Decimals = false);

  /// Returns the table, or null if it is not loaded.
  const EncodedTable* findTable(const std::string& name) const;

  const EncodedTable& table(const std::string& name) const;

  void clear();

 private:
  std::unordered_map<std::string, std::unique_ptr<EncodedTable>> tables_;
};

} // namespace facebook::velox::cudf_velox::connector::encoded
