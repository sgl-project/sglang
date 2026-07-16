// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TOKENSPEED_FUSED_TOPK_TOPP_H_
#define TOKENSPEED_FUSED_TOPK_TOPP_H_

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace fused_topk_topp {

using SizeType32 = int32_t;

constexpr int K_TOPK_MAX = 128;

// One workspace covers both pipelines + intermediate top-K results.
size_t getWorkspaceSize(SizeType32 batchSize, SizeType32 vocabSize);

// Renorm-only fused kernel.
//
// Inputs (all on device):
//   probs[bs, V]   — already softmax'd probabilities.
//   topKs[bs]      — int32, per-row K. K_TOPK_MAX is hard upper bound for the
//                    top-k path; K >= V (e.g. (1<<30)) routes the row through
//                    the radix top-p path.
//   topPs[bs]      — float, per-row P in (0, 1].
//
// Output:
//   outProbs[bs, V] — same shape as probs; non-selected positions are 0; kept
//                     positions are renormalized so the row sums to 1.
//
// CUDA-graph safe: every kernel launch has fixed grid/block; per-row mode is
// resolved by the kernels themselves via topKs[row].
void invokeFusedTopKTopP(
    float const* probs,
    SizeType32 const* topKs,
    float const* topPs,
    float* outProbs,
    void* workspace,
    SizeType32 batchSize,
    SizeType32 vocabSize,
    cudaStream_t mainStream,
    cudaStream_t memsetStream);

}  // namespace fused_topk_topp

#endif  // TOKENSPEED_FUSED_TOPK_TOPP_H_
