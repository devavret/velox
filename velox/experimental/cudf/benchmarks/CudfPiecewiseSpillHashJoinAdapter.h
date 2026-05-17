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

#include <cudf/types.hpp>

namespace facebook::velox::cudf_velox {

enum class SpillHostMemoryKind;

/// Registers two `OperatorAdapter`s — one for `HashBuild`, one for
/// `HashProbe` — that overwrite the default cuDF hash-join adapters and
/// instead substitute the piecewise host-spill operators
/// (`CudfPiecewiseSpillHashJoinBuild` / `CudfPiecewiseSpillHashJoinProbe`).
///
/// Must be called after `registerCudf()` has run, so that the default
/// cuDF adapters are present in the registry and can be overwritten.
///
/// `pieceTargetRows` is captured by the build adapter and used as the
/// target piece size when slicing the build side.
void registerPiecewiseSpillHashJoinAdapter(
    cudf::size_type pieceTargetRows,
    SpillHostMemoryKind spillHostMemoryKind);

} // namespace facebook::velox::cudf_velox
