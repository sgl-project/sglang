/*
 * Copyright (c) 2026 SGLang Team
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

namespace sm90::nvfp4 {

static constexpr int kLatentDim = 512;
static constexpr int kRopeDim = 64;
static constexpr int kBlockSize = 16;
static constexpr int kPackedLatentBytes = kLatentDim / 2;
static constexpr int kScaleBytes = kLatentDim / kBlockSize;
static constexpr int kRopeBytes = kRopeDim * 2;
static constexpr int kBytesPerToken = kPackedLatentBytes + kScaleBytes + kRopeBytes;

static_assert(kPackedLatentBytes == 256);
static_assert(kScaleBytes == 32);
static_assert(kBytesPerToken == 416);

}  // namespace sm90::nvfp4
