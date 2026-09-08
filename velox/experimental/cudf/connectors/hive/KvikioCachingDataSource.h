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

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileIds.h"

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <future>
#include <memory>
#include <string>
#include <vector>

namespace folly {
class Executor;
}

namespace facebook::velox::cudf_velox::connector::hive {

/// Wraps the cuDF KvikIO datasource and optionally caches remote file bytes in
/// Velox's AsyncDataCache.
///
/// Cache-only prefetches call host_read() on the KvikIO delegate directly into
/// a contiguous cache entry. Device reads subsequently stage a cache hit
/// through pinned memory and enqueue H2D on the consumer stream. When no
/// AsyncDataCache exists, every operation passes through to the delegate.
class KvikioCachingDataSource final : public cudf::io::datasource {
 public:
  KvikioCachingDataSource(
      std::unique_ptr<cudf::io::datasource> delegate,
      std::string path,
      folly::Executor* executor);

  ~KvikioCachingDataSource() override;

  [[nodiscard]] size_t size() const override;

  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size)
      override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  [[nodiscard]] bool supports_device_read() const override;

  [[nodiscard]] bool is_device_read_preferred(size_t size) const override;

  std::future<size_t> device_read_async(
      size_t offset,
      size_t size,
      uint8_t* dst,
      rmm::cuda_stream_view stream) override;

  size_t device_read(
      size_t offset,
      size_t size,
      uint8_t* dst,
      rmm::cuda_stream_view stream) override;

  std::unique_ptr<datasource::buffer> device_read(
      size_t offset,
      size_t size,
      rmm::cuda_stream_view stream) override;

  [[nodiscard]] bool cacheEnabled() const {
    return cache_ != nullptr;
  }

  /// Starts cache-only reads for the merged byte ranges. The returned futures
  /// own completion; no device memory is allocated and no H2D is enqueued.
  std::vector<std::future<size_t>> prefetchRanges(
      cudf::host_span<const cudf::io::text::byte_range_info> byteRanges);

 private:
  cache::CachePin pinRange(uint64_t offset, uint64_t size, bool prefetch);

  void readThroughCache(size_t offset, size_t size, uint8_t* dst);

  size_t prefetchRange(size_t offset, size_t size);

  std::unique_ptr<cudf::io::datasource> delegate_;
  const size_t fileSize_;
  folly::Executor* const executor_;
  cache::AsyncDataCache* const cache_;
  StringIdLease fileNum_;
};

} // namespace facebook::velox::cudf_velox::connector::hive
