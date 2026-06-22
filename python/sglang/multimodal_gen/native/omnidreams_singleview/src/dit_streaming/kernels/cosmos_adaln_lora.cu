// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// cosmos_adaln_lora.cu — fused adaln-LoRA (down + up) projection helper.
//
// Cosmos has three adaln sub-layers per block (self_attn, cross_attn, mlp),
// each backed by its own LoRA stack:
//
//     down: [lora_dim, K]
//     up:   [3K, lora_dim]
//
// The math the bridge previously did per sub-layer was:
//
//     h    = SiLU(t_emb)               # [B, K]
//     h2   = h @ down.T                # [B, lora_dim]   <- ATen matmul
//     mods = h2 @ up.T + adaln_lora_3D # [B, 3K]         <- ATen matmul + add
//     shift, scale, gate = mods.chunk(3, -1)
//
// That is FOUR ATen ops per sub-layer × 3 sub-layers × 28 blocks = 336
// ATen launches per forward just for the adaln mods. We collapse this to:
//
//   silu_inplace_bf16(t_emb_silu_buf)
//   GEMM (down)              -> lora_hidden  [B, lora_dim]
//   GEMM (up) with bias=adaln_lora_3D row -> mods_out [B, 3K]
//
// We can use the bias path of the second GEMM to fold the
// "+ adaln_lora_3D" add into the GEMM's own bias broadcast, BUT only when
// `adaln_lora_3D` is a single row [1, 3K] -- which it is in the streaming
// path (B=1 typically, or B=2 for CFG with the same lora component
// broadcast). For B>1 we fall back to a separate add kernel.
//
// `mods_out` is then carved into (shift, scale, gate) views by the caller
// using simple pointer arithmetic; no copy.

#include "cosmos_block.cuh"
#include "dtype_utils.cuh"

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>

namespace omnidreams_singleview {

// ---------------------------------------------------------------------------
// In-place SiLU on bf16/fp16 buffer.
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
// ---------------------------------------------------------------------------
template <typename ElementT>
__global__ void cosmos_silu_inplace_kernel(ElementT* __restrict__ data, int64_t numel) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) return;
  float v = omnidreams_singleview::to_float(data[idx]);
  float s = v / (1.f + __expf(-v));
  data[idx] = omnidreams_singleview::from_float<ElementT>(s);
}

// ---------------------------------------------------------------------------
// Add adaln_lora_3D (the per-block bias-style component) into mods_out.
// adaln_lora_3D is broadcast across rows of mods_out:
//   - When adaln_lora_3D has stride 0 across batch (i.e. shape [1, 3K]),
//     same row added to every output row.
//   - When stride = 3K, per-row add.
// ---------------------------------------------------------------------------
template <typename ElementT>
__global__ void cosmos_add_lora_3d_kernel(
    ElementT* __restrict__ mods,           // [B, 3K]
    const ElementT* __restrict__ lora_3d,  // [B, 3K] or [1, 3K]
    int B, int three_K,
    int lora_row_stride)                   // 0 = broadcast, 3K = per-row
{
  int b = blockIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B || j >= three_K) return;
  size_t pos_dst = size_t(b) * three_K + j;
  size_t pos_src = size_t(b) * lora_row_stride + j;
  float a = omnidreams_singleview::to_float(mods[pos_dst]);
  float l = omnidreams_singleview::to_float(lora_3d[pos_src]);
  mods[pos_dst] = omnidreams_singleview::from_float<ElementT>(a + l);
}

// ---------------------------------------------------------------------------
// Host launcher
//
// Note: `down` and `up` are passed in PyTorch's row-major layout
//   - down: [lora_dim, K]    PyTorch nn.Linear(K, lora_dim).weight
//   - up:   [3K, lora_dim]   PyTorch nn.Linear(lora_dim, 3K).weight
//
// `cutlass_linear_layer_rrr_bf16` accepts these PyTorch-layout weights
// directly because internally it uses CUTLASS's RCR layout (RowMajor input
// × ColumnMajor weight × RowMajor output). PyTorch's `[out, in]` row-major
// is byte-equivalent to CUTLASS's `[in, out]` column-major, so the GEMM
// computes `output = input @ weight^T` -- matching PyTorch's nn.Linear
// math directly. No pre-transpose pass needed (unlike WAN's path, which
// does `weight.transpose(0, 1).contiguous()` offline in
// `native/common/weight_utils.py`).
//
// Math expansion:
//   h        [B, K]                      = SiLU(t_emb)            (caller-applied)
//   lora_h   [B, lora_dim]               = h @ down^T
//   mods_pre [B, 3K]                     = lora_h @ up^T
//   mods_out [B, 3K]                     = mods_pre + adaln_lora_3D
// ---------------------------------------------------------------------------

template <typename ElementT>
cudaError_t cosmos_adaln_lora_split(
    const ElementT* t_emb,
    const ElementT* down_weight,
    const ElementT* up_weight,
    const ElementT* adaln_lora_3D,
    ElementT* lora_hidden_buf,
    ElementT* mods_out,
    int B, int K, int lora_dim,
    cudaStream_t stream)
{
  if (B <= 0 || K <= 0 || lora_dim <= 0) return cudaSuccess;

  // 1) Copy t_emb -> lora_hidden_buf (we apply SiLU into the buffer rather
  //    than mutating t_emb in-place; t_emb is shared across all three
  //    sub-layers in a block).
  //
  //    Optimization: we can SiLU directly into a temporary scratch we then
  //    feed as the GEMM input. But to keep the helper allocation-free, we
  //    SiLU-into the bottom of `lora_hidden_buf` (we have B*lora_dim of
  //    space; we need B*K which is much larger... so this approach needs
  //    a separate scratch).
  //
  //    Simpler: SiLU in-place on t_emb is NOT safe (shared). Instead we
  //    fuse SiLU + GEMM by using cutlass_linear_layer_rrr_silu_bf16, which
  //    computes `SiLU(A @ B + bias)` -- but that's SiLU AFTER the GEMM,
  //    not before. The cosmos op order is SiLU(emb) THEN GEMM, so we need
  //    SiLU-before-GEMM.
  //
  //    Solution: have the caller pre-compute SiLU(t_emb) once per forward
  //    and pass that pre-silu'd buffer here. (One SiLU launch per forward,
  //    not per sub-layer.) The bridge does this and passes the same
  //    `t_emb_silu` to all three sub-layers within a block.

  // 2) GEMM: lora_hidden = t_emb_silu @ down  (PyTorch row-major down ==
  //    transposed-for-math: interpret as [K, lora_dim])
  //    Shape: [B, K] x [K, lora_dim] -> [B, lora_dim]
  cudaError_t err = cutlass_linear_layer_rrr_bf16(
      reinterpret_cast<const cutlass::bfloat16_t*>(t_emb),
      reinterpret_cast<const cutlass::bfloat16_t*>(down_weight),
      /*bias=*/nullptr,
      reinterpret_cast<cutlass::bfloat16_t*>(lora_hidden_buf),
      B, K, lora_dim, stream);
  if (err != cudaSuccess) return err;

  // 3) GEMM: mods = lora_hidden @ up
  //    Shape: [B, lora_dim] x [lora_dim, 3K] -> [B, 3K]
  const ElementT* up_bias = (B == 1) ? adaln_lora_3D : nullptr;
  err = cutlass_linear_layer_rrr_bf16(
      reinterpret_cast<const cutlass::bfloat16_t*>(lora_hidden_buf),
      reinterpret_cast<const cutlass::bfloat16_t*>(up_weight),
      reinterpret_cast<const cutlass::bfloat16_t*>(up_bias),
      reinterpret_cast<cutlass::bfloat16_t*>(mods_out),
      B, lora_dim, 3 * K, stream);
  if (err != cudaSuccess) return err;

  // 4) Add adaln_lora_3D into mods_out.
  //    Streaming path always has B small (1 or 2) and adaln_lora_3D has
  //    shape [B, 3K] (per-batch). The add is cheap; one launch.
  if (adaln_lora_3D != nullptr && B != 1) {
    int three_K = 3 * K;
    int lora_row_stride = three_K;  // per-row add (one row per batch)
    dim3 block(256);
    dim3 grid((three_K + block.x - 1) / block.x, B);
    cosmos_add_lora_3d_kernel<ElementT><<<grid, block, 0, stream>>>(
        mods_out, adaln_lora_3D, B, three_K, lora_row_stride);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  return cudaSuccess;
}

// In-place SiLU launcher (called once per forward by the bridge to compute
// SiLU(t_emb) once and reuse across all sub-layers).
template <typename ElementT>
cudaError_t cosmos_silu_inplace(
    ElementT* data,
    int64_t numel,
    cudaStream_t stream)
{
  if (numel <= 0) return cudaSuccess;
  int threads = 256;
  int64_t blocks = (numel + threads - 1) / threads;
  cosmos_silu_inplace_kernel<ElementT><<<blocks, threads, 0, stream>>>(data, numel);
  return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t cosmos_adaln_lora_split<cutlass::bfloat16_t>(
    const cutlass::bfloat16_t*, const cutlass::bfloat16_t*, const cutlass::bfloat16_t*,
    const cutlass::bfloat16_t*, cutlass::bfloat16_t*, cutlass::bfloat16_t*,
    int, int, int, cudaStream_t);
template cudaError_t cosmos_adaln_lora_split<cutlass::half_t>(
    const cutlass::half_t*, const cutlass::half_t*, const cutlass::half_t*,
    const cutlass::half_t*, cutlass::half_t*, cutlass::half_t*,
    int, int, int, cudaStream_t);

template cudaError_t cosmos_silu_inplace<cutlass::bfloat16_t>(
    cutlass::bfloat16_t*, int64_t, cudaStream_t);
template cudaError_t cosmos_silu_inplace<cutlass::half_t>(
    cutlass::half_t*, int64_t, cudaStream_t);

} // namespace omnidreams_singleview
