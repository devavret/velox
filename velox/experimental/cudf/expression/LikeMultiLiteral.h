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
// src/expression_evaluator/like_multiliteral.hpp at commit
// 629e1eaff154480d8782ba1702e5ab0d493e7da9 (Apache License 2.0). Changed for
// Velox: Velox names for the public API, an ESCAPE character that the pattern
// never uses qualifies, NOT LIKE is not fused (no 'invert'), the shared
// classification cache is dropped (each expression keeps its own parsed
// pattern) and a launch counter for tests is added. The descriptor structs in
// 'detail' are Sirius's unchanged.

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Maximum number of literals the fast-path kernel supports.
inline constexpr int kLikeMultiLiteralMaxLiterals = 4;
/// Maximum byte length of each literal the fast-path kernel supports.
inline constexpr int kLikeMultiLiteralMaxLiteralBytes = 32;

namespace detail {

/// Max number of uint64_ts in a literal
inline constexpr int like_multiliteral_max_chunks =
    kLikeMultiLiteralMaxLiteralBytes / 8;

/// Descriptor for a single literal in the fast-path kernel.
struct like_multiliteral_literal_desc {
  uint64_t chunk_val[like_multiliteral_max_chunks];
  uint64_t chunk_mask[like_multiliteral_max_chunks];
  uint64_t b0_bcast;
  uint64_t b1_bcast;
  int len;
  int nchunks;
};

/// Descriptor for the whole multi-literal pattern
struct like_multiliteral_pattern_desc {
  like_multiliteral_literal_desc literals[kLikeMultiLiteralMaxLiterals];
  int n;
  int total_len;
};

} // namespace detail

class LikeMultiLiteralPattern;

/// Classifies a constant LIKE pattern for the multi-literal fast path.
///
/// Accepts exactly '%lit1%lit2%...%litN%':
/// - the pattern starts and ends with '%', and consecutive '%' collapse;
/// - it contains no '_' wildcard;
/// - every literal byte is ASCII in [0x01, 0x7F], so a NUL byte refuses it;
/// - it has 1..kLikeMultiLiteralMaxLiterals literals of at most
///   kLikeMultiLiteralMaxLiteralBytes bytes each.
///
/// 'escape' is std::nullopt when the LIKE has no ESCAPE argument. Otherwise
/// the pattern qualifies only when the escape is exactly one byte that does
/// not occur in the pattern, where it has no effect (as on the CPU path); an
/// escape that the pattern uses refuses it.
///
/// @return The parsed literals, or std::nullopt when the caller must use
///         cudf::strings::like.
std::optional<LikeMultiLiteralPattern> classifyLikeMultiLiteral(
    std::string_view pattern,
    std::optional<std::string_view> escape);

/// Parsed form of an eligible '%lit1%lit2%...%litN%' pattern: the literals and
/// the kernel descriptor built from them, never any result.
class LikeMultiLiteralPattern {
 public:
  /// The literals in pattern order, each 1..kLikeMultiLiteralMaxLiteralBytes
  /// ASCII bytes in [0x01, 0x7F] other than '%' and '_'.
  const std::vector<std::string>& literals() const noexcept {
    return literals_;
  }

  const detail::like_multiliteral_pattern_desc& descriptor() const noexcept {
    return descriptor_;
  }

 private:
  explicit LikeMultiLiteralPattern(std::vector<std::string> literals);

  std::vector<std::string> literals_;
  detail::like_multiliteral_pattern_desc descriptor_{};

  friend std::optional<LikeMultiLiteralPattern> classifyLikeMultiLiteral(
      std::string_view pattern,
      std::optional<std::string_view> escape);
};

/// Evaluates a classified pattern over 'input'. Row i is true iff the literals
/// occur in row i in order and without overlapping. Bytes are compared and
/// never decoded, so any byte sequence is handled, and the result equals the
/// CPU substrings search (functions/lib/Re2Functions.cpp,
/// matchSubstringsPattern). For valid UTF-8 this is also what
/// cudf::strings::like returns. The null mask of 'input' is copied and null
/// rows hold false, as cudf::strings::like does.
///
/// Allocates only the result, from 'mr', and enqueues the work on 'stream'
/// without synchronizing. 'input' must stay valid until that work completes.
///
/// @return The BOOL8 result, or nullptr when the column layout is ineligible
///         (characters not 8-byte aligned, or offsets neither INT32 nor INT64);
///         the caller must then use cudf::strings::like.
std::unique_ptr<cudf::column> likeMultiLiteral(
    const cudf::strings_column_view& input,
    const LikeMultiLiteralPattern& pattern,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Number of likeMultiLiteral() kernel launches in this process, for tests.
uint64_t likeMultiLiteralLaunchCountForTesting();

} // namespace facebook::velox::cudf_velox
