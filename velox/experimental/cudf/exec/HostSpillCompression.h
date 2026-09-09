/* Copyright (c) Facebook, Inc. and its affiliates.
 * Licensed under the Apache License, Version 2.0. */
#pragma once

#include "velox/common/base/Exceptions.h"
#include <lz4.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace facebook::velox::cudf_velox {

// Lossless storage compression; metadata continues to describe the unpacked
// cuDF buffer. A zero rawBytes denotes an uncompressed payload.
inline void compressHostSpill(std::vector<uint8_t>& data, size_t& rawBytes) {
  if (data.empty() || data.size() > LZ4_MAX_INPUT_SIZE) {
    return;
  }
  const auto size = static_cast<int>(data.size());
  std::vector<uint8_t> scratch(LZ4_compressBound(size));
  const auto compressed = LZ4_compress_default(
      reinterpret_cast<const char*>(data.data()),
      reinterpret_cast<char*>(scratch.data()), size, static_cast<int>(scratch.size()));
  if (compressed > 0 && static_cast<size_t>(compressed) < data.size() * 9 / 10) {
    rawBytes = data.size();
    // Allocate at compressed size: resizing scratch would retain its raw-size capacity.
    std::vector<uint8_t> compact(scratch.begin(), scratch.begin() + compressed);
    data.swap(compact);
  }
}

inline std::vector<uint8_t> decompressHostSpill(
    const std::vector<uint8_t>& data, size_t rawBytes) {
  std::vector<uint8_t> out(rawBytes);
  const auto actual = LZ4_decompress_safe(
      reinterpret_cast<const char*>(data.data()), reinterpret_cast<char*>(out.data()),
      static_cast<int>(data.size()), static_cast<int>(rawBytes));
  VELOX_CHECK_EQ(actual, rawBytes, "Corrupt host-spill buffer");
  return out;
}

} // namespace facebook::velox::cudf_velox
