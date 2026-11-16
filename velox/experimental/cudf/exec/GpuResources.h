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

#include <cudf/detail/utilities/stream_pool.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

namespace facebook::velox::cudf_velox {

extern std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> mr_;
extern std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>>
    output_mr_;

/// Statistics adaptors wrapping the main (and, if distinct, output) memory
/// resources so that live-allocated GPU bytes can be reported.
/// statistics_resource_adaptor is a cuda::mr::shared_resource: it is copyable
/// and copies share the same counter state, so the copies placed in
/// mr_/output_mr_ increment the same counters that these read.
extern std::optional<rmm::mr::statistics_resource_adaptor> statsMr_;
extern std::optional<rmm::mr::statistics_resource_adaptor> outputStatsMr_;

/// Returns the memory resource designated for temporary allocations: the
/// temporary resource of the active ScopedCudfMemoryResources on this thread,
/// or the current device resource when there is none.
rmm::device_async_resource_ref get_temp_mr();

/// Returns the memory resource designated for output vector allocations: the
/// output resource of the active ScopedCudfMemoryResources on this thread, or
/// the process output resource when there is none.
rmm::device_async_resource_ref get_output_mr();

/// Returns the owner installed by the active ScopedCudfMemoryResources on this
/// thread, or nullptr. A holder of device memory retains it so that whatever
/// the scoped resources refer to outlives the memory.
std::shared_ptr<void> get_memory_resource_owner();

/// Creates the process-wide resource to install as the current device
/// resource. cuDF APIs without a temporary-resource parameter allocate from
/// the current device resource. While a ScopedCudfMemoryResources is active on
/// the allocating thread, such an allocation goes to its temporary resource,
/// which is retained until the matching deallocation on any thread. Outside a
/// scope allocations use 'fallback'.
[[nodiscard]] cuda::mr::any_resource<cuda::mr::device_accessible>
createThreadLocalTemporaryMemoryResource(
    cuda::mr::any_resource<cuda::mr::device_accessible> fallback);

/// Installs the temporary and output resources that get_temp_mr() and
/// get_output_mr() return on this thread until destruction. Scopes nest. The
/// resources carry their own accounting identity, so memory stays attributed
/// when cuDF frees it on another thread or stream.
class ScopedCudfMemoryResources {
 public:
  /// Installs nothing: the resources and owner already in effect stay.
  ScopedCudfMemoryResources() = default;

  /// Keeps the owner of the enclosing scope, if any. 'tempMr' must not be the
  /// current device resource made by createThreadLocalTemporaryMemoryResource()
  /// itself, which would forward to its own scope.
  ScopedCudfMemoryResources(
      rmm::device_async_resource_ref tempMr,
      rmm::device_async_resource_ref outputMr);

  /// Installs 'owner' as well, which get_memory_resource_owner() returns.
  ScopedCudfMemoryResources(
      rmm::device_async_resource_ref tempMr,
      rmm::device_async_resource_ref outputMr,
      std::shared_ptr<void> owner);

  ~ScopedCudfMemoryResources();

  ScopedCudfMemoryResources(const ScopedCudfMemoryResources&) = delete;
  ScopedCudfMemoryResources& operator=(const ScopedCudfMemoryResources&) =
      delete;
  ScopedCudfMemoryResources(ScopedCudfMemoryResources&&) = delete;
  ScopedCudfMemoryResources& operator=(ScopedCudfMemoryResources&&) = delete;

 private:
  const bool installed_{false};
  std::shared_ptr<void> previousOwner_;
  std::optional<rmm::device_async_resource_ref> previousTempMr_;
  std::optional<rmm::device_async_resource_ref> previousOutputMr_;
};

/// Live bytes currently allocated through the cuDF/RMM memory resource(s), or
/// -1 if cuDF is not registered. Counts the bytes callers hold rather than the
/// device memory a pool resource retains for reuse, so the value drops as
/// queries free their allocations.
int64_t cudfAllocatedBytes();

/// Creates a memory resource based on the given mode.
///
/// @param mode rmm::mr::pool_memory_resource mode.
/// @param percent The initial percent of GPU memory to allocate for memory
/// resource.
[[nodiscard]] cuda::mr::any_resource<cuda::mr::device_accessible>
createMemoryResource(std::string_view mode, int percent);

/// Returns the global CUDA stream pool used by cudf.
[[nodiscard]] cudf::detail::cuda_stream_pool& cudfGlobalStreamPool();

/// Records the CUDA device owning the primary context, so worker threads can
/// bind it via `ensureCudaContextForThread()`.
///
/// @param device The CUDA device ordinal that was current during registration.
void setCudfContextDevice(int device);

/// Binds the primary CUDA context on the calling thread for the registered
/// device, rebinding when cuDF is registered on another device.
/// Needed as `cuda::stream_ref` uses the driver API, which does not lazily
/// create a context like the runtime API does.
void ensureCudaContextForThread();

/// Ensures that the calling thread has a CUDA context and returns the
/// `cudf::get_default_stream()`. Use it for host-thread stream work outside
/// cuDF operators.
[[nodiscard]] cuda::stream_ref getDefaultStreamForCurrentThread();

} // namespace facebook::velox::cudf_velox
