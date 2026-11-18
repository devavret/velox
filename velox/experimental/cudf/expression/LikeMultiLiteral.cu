/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Copyright 2026, Sirius Contributors.
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

// Adapted from Sirius (https://github.com/sirius-db/sirius), file
// src/cuda/sirius_like_multiliteral.cu at commit
// 629e1eaff154480d8782ba1702e5ab0d493e7da9 (Apache License 2.0).
//
// The region between "clang-format off" and "clang-format on" is lines 35-226
// of that file, with three changes, each marked "Velox:":
//   - max_literals takes its value from kLikeMultiLiteralMaxLiterals;
//   - like_multiliteral_kernel has no 'invert' parameter;
//   - so it writes 'match' (NOT LIKE stays a separate Velox expression).
// The host functions after the region follow Sirius's like_multiliteral_pattern
// constructor, classify_like_multiliteral and like_multiliteral, with the
// differences noted where they occur.

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/expression/LikeMultiLiteral.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace facebook::velox::cudf_velox {

// clang-format off
namespace {

constexpr int max_literals      = kLikeMultiLiteralMaxLiterals;  // Velox: was like_multiliteral_max_literals
constexpr int max_chunks        = detail::like_multiliteral_max_chunks;
constexpr int threads_per_block = 256;
constexpr int word_shift        = 3;
constexpr int bytes_per_word    = 1 << word_shift;
constexpr int word_mask         = bytes_per_word - 1;
constexpr int bits_per_byte     = 8;
constexpr int bits_per_word     = bytes_per_word * bits_per_byte;

static_assert(bytes_per_word == sizeof(uint64_t));

using literal_desc = detail::like_multiliteral_literal_desc;
using pattern_desc = detail::like_multiliteral_pattern_desc;

/// SWAR (SIMD Within A Register) byte-equality candidate mask: a SUPERSET of the equal-byte
/// positions of @p x vs the broadcast byte of @p bcast, as bit 7 per byte. Every byte equal to
/// bcast is flagged, but borrow propagation from the subtraction can set spurious bits on unequal
/// bytes — callers MUST verify candidates against the actual bytes before acting on them.
__device__ __forceinline__ uint64_t byte_eq_mask(uint64_t x, uint64_t bcast)
{
  /// @p bcast is the target byte replicated 8 times
  /// The result uses the high bit of each byte as a marker
  auto const v = x ^ bcast;  // equal bytes become 0
  /// SISD: (z - 1) & ~z & 0x80 would identify a zero byte. The computation below is a 'SIMD'
  /// version that will produce false positives due to underflow.
  return (v - 0x0101010101010101ULL) & ~v & 0x8080808080808080ULL;
}

/// Load one aligned little-endian word without reading past the chars child (here: word = 8 bytes
/// [uint64_t])
__device__ __forceinline__ uint64_t load_aligned_word(uint64_t const* __restrict__ words,
                                                      int64_t chars_size,
                                                      int64_t word_index)
{
  auto const byte_pos = word_index << word_shift;
  if (byte_pos >= chars_size) { return 0; }
  if (chars_size - byte_pos >= static_cast<int64_t>(bytes_per_word)) { return words[word_index]; }
  auto const* tail = reinterpret_cast<unsigned char const*>(words) + byte_pos;
  uint64_t value   = 0;
#pragma unroll
  for (int i = 0; i < bytes_per_word; ++i) {
    if (byte_pos + i >= chars_size) { break; }
    value |= static_cast<uint64_t>(tail[i]) << (i * bits_per_byte);
  }
  return value;
}

/**
 * @brief Unaligned little-endian u64 window at byte position @p p of an aligned u64 stream.
 *
 * Bytes at or beyond @p chars_size read as zero.
 */
__device__ __forceinline__ uint64_t read_u64_window(uint64_t const* __restrict__ words,
                                                    int64_t chars_size,
                                                    int64_t p)
{
  auto const w       = p >> word_shift;
  auto const bit_off = static_cast<int>(p & word_mask) * bits_per_byte;
  auto const lo      = load_aligned_word(words, chars_size, w);
  if (bit_off == 0) { return lo; }
  auto const hi = load_aligned_word(words, chars_size, w + 1);
  return (lo >> bit_off) | (hi << (bits_per_word - bit_off));
}

/// Candidate positions for a literal within the aligned word @p cur: a SUPERSET of the byte
/// positions k where the literal's first byte is at k and (len >= 2) its second byte follows
/// (possibly in @p next), as bit 7 per byte — byte_eq_mask false positives carry through, so
/// every candidate must be verified. @p b0 / @p b1 are the literal's first two bytes
/// broadcast to all 8 lanes.
__device__ __forceinline__ uint64_t
candidate_mask(uint64_t cur, uint64_t next, uint64_t b0, uint64_t b1, int32_t len)
{
  auto const c1 = byte_eq_mask(cur, b0);
  if (len == 1) { return c1; }
  auto const c2  = byte_eq_mask(cur, b1);
  auto const c2n = byte_eq_mask(next, b1);
  return c1 & ((c2 >> bits_per_byte) | (c2n << (bits_per_word - bits_per_byte)));
}

/// Full literal verification at byte position @p p (caller guarantees p + lit.len fits the row).
__device__ __forceinline__ bool verify_literal(uint64_t const* __restrict__ words,
                                               int64_t chars_size,
                                               int64_t p,
                                               literal_desc const& lit)
{
#pragma unroll
  for (int c = 0; c < max_chunks; ++c) {
    if (c == lit.nchunks) { break; }
    auto const win = read_u64_window(words, chars_size, p + bytes_per_word * c);
    if ((win & lit.chunk_mask[c]) != lit.chunk_val[c]) { return false; }
  }
  return true;
}

/**
 * @brief Thread-per-row multi-literal LIKE matcher.
 *
 * Each thread streams its row's aligned u64 words exactly once, SWAR-detects digram
 * candidates for the literal it currently needs, and verifies full literals only at
 * candidate hits. Greedy leftmost occurrence per literal with the next search starting at
 * the end of the previous match — exact for `%lit1%...%litN%` semantics.
 *
 * All positions are absolute byte positions into the chars buffer (offsets are stored that
 * way even for sliced views). The sliced view's final offset bounds memory loads; each row's
 * offsets independently bound candidate matching.
 */
template <typename OffT>
__global__ void like_multiliteral_kernel(uint64_t const* __restrict__ words,
                                         OffT const* __restrict__ offsets,
                                         cudf::size_type nrows,
                                         cudf::bitmask_type const* __restrict__ null_mask,
                                         cudf::size_type mask_offset,
                                         pattern_desc pat,
                                         bool* __restrict__ out)  // Velox: no 'bool invert'
{
  auto const row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= nrows) { return; }
  auto const chars_size = static_cast<int64_t>(offsets[nrows]);

  bool match = false;
  bool const is_valid =
    null_mask == nullptr ||
    cudf::bit_is_set(null_mask, mask_offset + static_cast<cudf::size_type>(row));
  if (is_valid) {
    auto const s = static_cast<int64_t>(offsets[row]);
    auto const e = static_cast<int64_t>(offsets[row + 1]);
    if (e - s >= pat.total_len) {  // rows shorter than the literals cannot match
      int li       = 0;
      auto min_pos = s;  // next literal may start no earlier than this
      // Hot fields of the current literal, kept in registers.
      auto b0  = pat.literals[0].b0_bcast;
      auto b1  = pat.literals[0].b1_bcast;
      auto len = pat.literals[0].len;

      auto w        = s >> word_shift;
      auto const we = (e + word_mask) >> word_shift;  // exclusive
      auto cur      = load_aligned_word(words, chars_size, w);
      for (; w < we && !match; ++w) {
        auto const next = load_aligned_word(words, chars_size, w + 1);
        auto const base = w << word_shift;  // byte position of aligned word index
        auto cand       = candidate_mask(cur, next, b0, b1, len);
        while (cand) {
          // Extract the earliest candidate
          auto const k = __ffsll(static_cast<long long>(cand)) - 1;
          // Clear the lowest set bit in the candidate mask
          cand &= cand - 1;
          auto const p = base + (k >> word_shift);  // convert bit index into byte lane
          // Bytes before the row start / previous match, or literals overrunning the row
          // end (including digram second-bytes that belong to the next row), are rejected
          // here — candidate detection may fire on them, verification never sees them.
          if (p < min_pos || p + len > e) { continue; }
          if (!verify_literal(words, chars_size, p, pat.literals[li])) { continue; }
          min_pos = p + len;
          ++li;
          if (li == pat.n) {
            match = true;
            break;
          }
          b0  = pat.literals[li].b0_bcast;
          b1  = pat.literals[li].b1_bcast;
          len = pat.literals[li].len;
          // Rescan the current word for the next literal (its match may start here too;
          // positions before min_pos are filtered above).
          cand = candidate_mask(cur, next, b0, b1, len);
        }
        cur = next;
      }
    }
  }
  out[row] = match;  // Velox: was 'match != invert'
}

/// Build the kernel-side literal descriptor from a host literal.
literal_desc make_literal_desc(std::string const& lit)
{
  literal_desc d{};
  d.len     = static_cast<int>(lit.size());
  d.nchunks = (d.len + word_mask) / bytes_per_word;
  for (int i = 0; i < d.len; ++i) {
    auto const byte = static_cast<uint64_t>(static_cast<unsigned char>(lit[i]));
    d.chunk_val[i / bytes_per_word] |= byte << (bits_per_byte * (i % bytes_per_word));
    d.chunk_mask[i / bytes_per_word] |= 0xFFULL << (bits_per_byte * (i % bytes_per_word));
  }
  d.b0_bcast = 0x0101010101010101ULL * static_cast<unsigned char>(lit[0]);
  d.b1_bcast = d.len >= 2 ? 0x0101010101010101ULL * static_cast<unsigned char>(lit[1]) : 0;
  return d;
}

}  // namespace
// clang-format on

namespace {

std::atomic<uint64_t> launchCount{0};

} // namespace

// Sirius's like_multiliteral_pattern constructor.
LikeMultiLiteralPattern::LikeMultiLiteralPattern(
    std::vector<std::string> literals)
    : literals_(std::move(literals)) {
  if (literals_.empty() ||
      literals_.size() > static_cast<size_t>(max_literals)) {
    return;
  }
  for (const auto& literal : literals_) {
    if (literal.empty() ||
        literal.size() >
            static_cast<size_t>(kLikeMultiLiteralMaxLiteralBytes)) {
      return;
    }
  }

  descriptor_.n = static_cast<int>(literals_.size());
  for (size_t i = 0; i < literals_.size(); ++i) {
    descriptor_.literals[i] = make_literal_desc(literals_[i]);
    descriptor_.total_len += descriptor_.literals[i].len;
  }
}

// Sirius's classify_like_multiliteral, except that an escape the pattern never
// uses qualifies instead of refusing the pattern.
std::optional<LikeMultiLiteralPattern> classifyLikeMultiLiteral(
    std::string_view pattern,
    std::optional<std::string_view> escape) {
  if (escape.has_value() &&
      (escape->size() != 1 ||
       pattern.find(escape->front()) != std::string_view::npos)) {
    return std::nullopt; // escape semantics in play: cudf path
  }
  if (pattern.size() < 2 || pattern.front() != '%' || pattern.back() != '%') {
    return std::nullopt; // anchored (prefix/suffix/exact) patterns: cudf path
  }
  std::vector<std::string> literals;
  std::string current;
  for (const auto ch : pattern) {
    if (ch == '%') {
      if (!current.empty()) {
        if (current.size() >
                static_cast<size_t>(kLikeMultiLiteralMaxLiteralBytes) ||
            literals.size() ==
                static_cast<size_t>(kLikeMultiLiteralMaxLiterals)) {
          return std::nullopt;
        }
        literals.push_back(std::move(current));
        current.clear();
      }
      continue; // consecutive '%' collapse
    }
    const auto byte = static_cast<unsigned char>(ch);
    if (ch == '_' || byte == 0 || byte >= 0x80) {
      return std::nullopt; // '_' wildcard / NUL / non-ASCII: cudf path
    }
    current.push_back(ch);
  }
  // The pattern ends with '%', so no literal is pending here.
  if (literals.empty()) {
    return std::nullopt; // pure '%'/'%%': cudf path
  }
  return LikeMultiLiteralPattern(std::move(literals));
}

// Sirius's like_multiliteral without 'invert'. The null mask is copied with
// an explicit temporary resource, as elsewhere in Velox.
std::unique_ptr<cudf::column> likeMultiLiteral(
    const cudf::strings_column_view& input,
    const LikeMultiLiteralPattern& pattern,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const auto& literals = pattern.literals();
  const auto n = static_cast<int64_t>(literals.size());
  CUDF_EXPECTS(
      n >= 1 && n <= max_literals,
      "likeMultiLiteral: pattern was not classified out",
      std::logic_error);
  for (const auto& literal : literals) {
    CUDF_EXPECTS(
        !literal.empty() &&
            literal.size() <=
                static_cast<size_t>(kLikeMultiLiteralMaxLiteralBytes),
        "likeMultiLiteral: literal was not classified out",
        std::logic_error);
  }

  const auto nrows = input.size();
  if (nrows == 0) {
    return cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::BOOL8},
        0,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
  }

  // Column-layout eligibility: the kernel needs an 8-byte-aligned chars base
  // for its aligned u64 loads, and offsets of a known width. Anything else
  // falls back to cudf.
  const auto offsetsId = input.offsets().type().id();
  if (offsetsId != cudf::type_id::INT32 && offsetsId != cudf::type_id::INT64) {
    return nullptr;
  }
  const char* charsBase = input.chars_begin(stream);
  if (reinterpret_cast<uintptr_t>(charsBase) & word_mask) {
    return nullptr;
  }

  const auto& pat = pattern.descriptor();

  auto result = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::BOOL8},
      nrows,
      cudf::copy_bitmask(
          input.parent(), stream, cudf::memory_resources{mr, get_temp_mr()}),
      input.null_count(),
      stream,
      mr);
  auto* out = result->mutable_view().data<bool>();

  const auto* nullMask = input.null_count() > 0 ? input.null_mask() : nullptr;
  const auto* words = reinterpret_cast<const uint64_t*>(charsBase);
  const auto numBlocks = static_cast<uint32_t>(
      (static_cast<int64_t>(nrows) + threads_per_block - 1) /
      threads_per_block);

  if (offsetsId == cudf::type_id::INT32) {
    const auto* offsets = input.offsets().data<int32_t>() + input.offset();
    like_multiliteral_kernel<int32_t>
        <<<numBlocks, threads_per_block, 0, stream.get()>>>(
            words, offsets, nrows, nullMask, input.offset(), pat, out);
  } else {
    const auto* offsets = input.offsets().data<int64_t>() + input.offset();
    like_multiliteral_kernel<int64_t>
        <<<numBlocks, threads_per_block, 0, stream.get()>>>(
            words, offsets, nrows, nullMask, input.offset(), pat, out);
  }
  CUDF_CUDA_TRY(cudaGetLastError());
  launchCount.fetch_add(1, std::memory_order_relaxed);
  return result;
}

uint64_t likeMultiLiteralLaunchCountForTesting() {
  return launchCount.load(std::memory_order_relaxed);
}

} // namespace facebook::velox::cudf_velox
