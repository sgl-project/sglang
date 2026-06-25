#pragma once

// Fused Q dynamic per-tensor FP8 quantization + paged KV-cache FP8 write,
// for the TRTLLM-GEN MHA decode / target-verify / draft-extend path.
//
// One launch replaces the whole per-layer preamble that previously fired as a
// string of tiny islands between the projection and the attention kernel:
//   - q.contiguous()                -> direct_copy
//   - torch.zeros(1) scale buffer   -> FillFunctor
//   - per_tensor_absmax_kernel      -> Q amax reduction
//   - per_tensor_quant_fp8_kernel   -> Q quantization
//   - 1/k_scale, 1/v_scale          -> reciprocal_kernel x2
//   - q_scale * k_scale * scaling   -> MulFunctor x2
//   - _fused_fp8_set_kv_buffer      -> separate Triton KV write
//
// The per-tensor Q amax is a grid-wide reduction, so the kernel is launched
// cooperatively (cudaLaunchAttributeCooperative) and uses cg::grid_group::sync
// to fence the amax atomic before quantization. The grid is sized to the actual
// work and capped at resident capacity, so tiny decode batches launch a handful
// of blocks (cheap sync) while large verify/extend batches saturate the GPU --
// avoiding the serial single-CTA cliff. Cooperative launch is CUDA-graph
// capturable (CUDA 12+), so the captured decode/target-verify path is fine.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/cta.cuh>      // For cta::reduce_max
#include <sgl_kernel/math.cuh>     // For math::max/min/abs, FP8_E4M3_MAX
#include <sgl_kernel/runtime.cuh>  // For runtime::get_blocks_per_sm / get_sm_count
#include <sgl_kernel/utils.cuh>    // For LaunchKernel, fp8_e4m3_t, type aliases
#include <sgl_kernel/vec.cuh>      // For AlignedVector

#include <cooperative_groups.h>
#include <cstdint>

namespace {

namespace cg = cooperative_groups;

constexpr uint32_t kFusedBlockSize = 256;

// Clip to the representable FP8 E4M3 range, then narrow.
SGL_DEVICE fp8_e4m3_t quant_one(float x, float inv_scale) {
  using namespace device;
  const float scaled = x * inv_scale;
  const float clipped = math::min(math::max(scaled, -math::FP8_E4M3_MAX), math::FP8_E4M3_MAX);
  return static_cast<fp8_e4m3_t>(clipped);
}

template <typename T, int kVec>
__global__ void fused_q_quant_kv_write_kernel(
    // Q: dynamic per-tensor quant
    const T* __restrict__ q_in,
    fp8_e4m3_t* __restrict__ q_out,
    int64_t n_q,
    // BMM1 descale out = (amax_q / 448) * (k_scale * attn_scaling).
    // Doubles as the grid-wide amax scratch during the reduction.
    float* __restrict__ bmm1_out,
    float bmm1_extra,
    // K/V: static per-tensor quant + scatter into paged cache by slot
    const T* __restrict__ k_in,
    const T* __restrict__ v_in,
    fp8_e4m3_t* __restrict__ k_cache,
    fp8_e4m3_t* __restrict__ v_cache,
    const int64_t* __restrict__ cache_loc,
    float inv_k_scale,
    float inv_v_scale,
    int num_tokens,
    int kv_row,
    int64_t k_in_row_stride,
    int64_t v_in_row_stride,
    int64_t cache_row_stride) {
  using namespace device;
  using in_vec_t = AlignedVector<T, kVec>;
  using out_vec_t = AlignedVector<fp8_e4m3_t, kVec>;

  cg::grid_group grid = cg::this_grid();
  const int64_t gtid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t gstride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t n_qvec = n_q / kVec;

  // Clear the grid-wide amax scratch (reuses bmm1_out).
  if (grid.thread_rank() == 0) {
    *bmm1_out = 0.0f;
  }
  grid.sync();

  // ---- Phase 1: per-tensor amax over the whole Q tensor ----
  float thread_max = 0.0f;
  for (int64_t vi = gtid; vi < n_qvec; vi += gstride) {
    in_vec_t v;
    v.load(q_in, vi);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      thread_max = math::max(thread_max, math::abs(static_cast<float>(v[i])));
    }
  }
  for (int64_t i = n_qvec * kVec + gtid; i < n_q; i += gstride) {
    thread_max = math::max(thread_max, math::abs(static_cast<float>(q_in[i])));
  }

  __shared__ float smem[kFusedBlockSize / 32];
  cta::reduce_max(thread_max, smem);
  __syncthreads();
  if (threadIdx.x == 0) {
    // amax is non-negative, so the IEEE bit pattern orders as a signed int.
    atomicMax(reinterpret_cast<int*>(bmm1_out), __float_as_int(smem[0]));
  }
  grid.sync();

  const float amax = *bmm1_out;
  const float inv_q = math::FP8_E4M3_MAX / amax;  // = 1 / q_scale
  grid.sync();                                    // all blocks read amax before rank 0 overwrites the scratch
  if (grid.thread_rank() == 0) {
    *bmm1_out = (amax / math::FP8_E4M3_MAX) * bmm1_extra;
  }

  // ---- Phase 2: quantize Q ----
  for (int64_t vi = gtid; vi < n_qvec; vi += gstride) {
    in_vec_t v;
    v.load(q_in, vi);
    out_vec_t o;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      o[i] = quant_one(static_cast<float>(v[i]), inv_q);
    }
    o.store(q_out, vi);
  }
  for (int64_t i = n_qvec * kVec + gtid; i < n_q; i += gstride) {
    q_out[i] = quant_one(static_cast<float>(q_in[i]), inv_q);
  }

  // ---- Phase 3: quantize + scatter K and V into the paged cache ----
  // The contiguous row (num_kv_heads * head_dim) divides kVec, so a vector
  // never straddles a head boundary; the cache is slot-major NHD, so the
  // write target for token t is simply slot = cache_loc[t].
  const int64_t row_vecs = kv_row / kVec;
  const int64_t total_vecs = static_cast<int64_t>(num_tokens) * row_vecs;
  for (int64_t idx = gtid; idx < total_vecs; idx += gstride) {
    const int64_t t = idx / row_vecs;
    const int64_t jv = idx % row_vecs;
    const int64_t slot = cache_loc[t];

    in_vec_t k_vec;
    k_vec.load(k_in + t * k_in_row_stride, jv);
    out_vec_t k_o;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      k_o[i] = quant_one(static_cast<float>(k_vec[i]), inv_k_scale);
    }
    k_o.store(k_cache + slot * cache_row_stride, jv);

    in_vec_t v_vec;
    v_vec.load(v_in + t * v_in_row_stride, jv);
    out_vec_t v_o;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      v_o[i] = quant_one(static_cast<float>(v_vec[i]), inv_v_scale);
    }
    v_o.store(v_cache + slot * cache_row_stride, jv);
  }
}

template <typename T>
void fused_q_quant_kv_write(
    tvm::ffi::TensorView q_in,
    tvm::ffi::TensorView q_out,
    tvm::ffi::TensorView bmm1_out,
    tvm::ffi::TensorView k_in,
    tvm::ffi::TensorView v_in,
    tvm::ffi::TensorView k_cache,
    tvm::ffi::TensorView v_cache,
    tvm::ffi::TensorView cache_loc,
    double inv_k_scale,
    double inv_v_scale,
    double bmm1_extra,
    int64_t num_tokens,
    int64_t kv_row,
    int64_t k_in_row_stride,
    int64_t v_in_row_stride,
    int64_t cache_row_stride) {
  using namespace host;

  auto N = SymbolicSize{"n_q"};
  auto one = SymbolicSize{"one"};
  auto nt = SymbolicSize{"num_tokens"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();

  TensorMatcher({N}).with_dtype<T>().with_device(device_).verify(q_in);
  TensorMatcher({N}).with_dtype<fp8_e4m3_t>().with_device(device_).verify(q_out);
  TensorMatcher({one}).with_dtype<float>().with_device(device_).verify(bmm1_out);
  TensorMatcher({nt}).with_dtype<int64_t>().with_device(device_).verify(cache_loc);

  const int64_t n_q = static_cast<int64_t>(N.unwrap());
  const DLDevice device = device_.unwrap();
  RuntimeCheck(
      static_cast<int64_t>(nt.unwrap()) == num_tokens,
      "fused_q_quant_kv_write: cache_loc length ",
      nt.unwrap(),
      " != num_tokens ",
      num_tokens);

  constexpr int kVec = 16 / sizeof(T);
  RuntimeCheck(kv_row % kVec == 0, "fused_q_quant_kv_write: kv_row ", kv_row, " must be a multiple of ", kVec);

  auto* kernel = fused_q_quant_kv_write_kernel<T, kVec>;

  // Size the cooperative grid to the work, capped at resident capacity so all
  // blocks co-exist (required for grid.sync). Tiny decode batches -> few blocks.
  const int64_t q_vecs = div_ceil(n_q, static_cast<int64_t>(kVec));
  const int64_t kv_vecs = num_tokens * (kv_row / kVec);
  const int64_t work = q_vecs > kv_vecs ? q_vecs : kv_vecs;
  const uint32_t per_sm = runtime::get_blocks_per_sm(kernel, kFusedBlockSize);
  const uint32_t max_grid = per_sm * runtime::get_sm_count(device.device_id);
  const int64_t want = div_ceil(work, static_cast<int64_t>(kFusedBlockSize));
  uint32_t grid = static_cast<uint32_t>(want < 1 ? 1 : (want > max_grid ? max_grid : want));

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(grid);
  config.blockDim = dim3(kFusedBlockSize);
  config.dynamicSmemBytes = 0;
  config.stream = LaunchKernel::resolve_device(device);
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeCooperative;
  attr.val.cooperative = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  RuntimeDeviceCheck(
      ::cudaLaunchKernelEx(
          &config,
          kernel,
          static_cast<const T*>(q_in.data_ptr()),
          static_cast<fp8_e4m3_t*>(q_out.data_ptr()),
          n_q,
          static_cast<float*>(bmm1_out.data_ptr()),
          static_cast<float>(bmm1_extra),
          static_cast<const T*>(k_in.data_ptr()),
          static_cast<const T*>(v_in.data_ptr()),
          static_cast<fp8_e4m3_t*>(k_cache.data_ptr()),
          static_cast<fp8_e4m3_t*>(v_cache.data_ptr()),
          static_cast<const int64_t*>(cache_loc.data_ptr()),
          static_cast<float>(inv_k_scale),
          static_cast<float>(inv_v_scale),
          static_cast<int>(num_tokens),
          static_cast<int>(kv_row),
          k_in_row_stride,
          v_in_row_stride,
          cache_row_stride));
}

}  // namespace
