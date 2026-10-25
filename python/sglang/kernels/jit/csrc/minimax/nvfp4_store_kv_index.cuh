#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize/DType/Device
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil, pointer::offset

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, PDL helpers
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp8.h>

namespace sglang {

/// Elements sharing one E4M3 block scale in the NVFP4 KV format.
inline constexpr int64_t kNvfp4BlockSize = 16;
/// Packed payload bytes produced by one block (two e2m1 codes per byte).
inline constexpr int64_t kNvfp4BlockBytes = kNvfp4BlockSize / 2;

namespace minimax_nvfp4 {

/// \brief `rcp.approx.ftz.f32`, the reciprocal the reference quantizer uses.
///
/// Deliberately not IEEE division. The MiniMax NVFP4 KV reference
/// (`flashinfer::nvfp4_kv_quantize`) forms both the block scale and the
/// quantization multiplier through this instruction, and its ~1 ulp error
/// crosses e2m1 round-to-nearest ties: substituting an exact divide changes
/// roughly 0.1% of output nibbles on random BF16 input. Measured on B300 /
/// flashinfer 0.6.18 against the six cache buffers.
///
/// This is also why `deepseek_v4::fp4` is not reused here -- that path pins
/// itself to IEEE division and additionally clamps the scale to `[2^-9, 448]`,
/// whereas this reference applies no floor and lets small blocks underflow the
/// scale to zero.
SGL_DEVICE float rcp_approx(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

/// \brief Convert eight scaled fp32 values to eight packed e2m1 codes.
///
/// `cvt.rn.satfinite.e2m1x2.f32 d, hi, lo` puts `lo` in the low nibble, so the
/// first element of each pair occupies the low nibble of its byte. Requires the
/// arch-specific target (sm_100a / sm_103a).
SGL_DEVICE uint32_t f32x8_to_e2m1x8(const float (&x)[8]) {
  uint32_t out;
  asm volatile(
      "{\n"
      ".reg .b8 b0;\n.reg .b8 b1;\n.reg .b8 b2;\n.reg .b8 b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}"
      : "=r"(out)
      : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
  return out;
}

/// \brief Quantize one 16-element BF16 block to NVFP4.
///
/// Reproduces the reference operation order exactly:
///   `sf  = rcp(gs) * (amax * rcp(6))`, rounded to E4M3 and read back,
///   `osc = amax != 0 ? rcp(sf * gs) : 0`,
///   each value scaled by `osc` then `cvt.rn.satfinite.e2m1x2.f32`.
/// `amax` is exact regardless of reduction order because `fabsf` of a BF16
/// value is representable in BF16.
///
/// \param src Sixteen contiguous BF16 elements.
/// \param global_scale The pool's outer scale (unit in production).
/// \param scale_out Receives the E4M3 scale byte.
/// \return The eight packed payload bytes.
SGL_DEVICE uint64_t quantize_block(const bf16_t* __restrict__ src, float global_scale, uint8_t* scale_out) {
  float v[kNvfp4BlockSize];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kNvfp4BlockSize; ++i) {
    v[i] = static_cast<float>(src[i]);
    const float a = fabsf(v[i]);
    amax = a > amax ? a : amax;
  }

  float sf = rcp_approx(global_scale) * (amax * rcp_approx(6.0f));
  const __nv_fp8_e4m3 sf_e4m3 = __nv_fp8_e4m3(sf);
  *scale_out = sf_e4m3.__x;
  sf = static_cast<float>(sf_e4m3);
  const float osc = amax != 0.0f ? rcp_approx(sf * global_scale) : 0.0f;

  float lo[8], hi[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    lo[i] = v[i] * osc;
    hi[i] = v[i + 8] * osc;
  }
  return (static_cast<uint64_t>(f32x8_to_e2m1x8(hi)) << 32) | f32x8_to_e2m1x8(lo);
}

}  // namespace minimax_nvfp4

/// Branch index into the three quantized streams.
enum Nvfp4Branch : uint32_t { kMainK = 0, kMainV = 1, kIndexK = 2 };

struct Nvfp4StoreParams {
  const void* __restrict__ in_ptrs[3];  // main K, main V, index K (BF16)
  int64_t in_token_stride[3];           // elements
  int64_t in_head_stride[3];            // elements; unused for index K

  void* __restrict__ payload_ptrs[3];
  void* __restrict__ scale_ptrs[3];
  int64_t payload_strides[3][3];  // [branch][page, head, row], bytes
  int64_t scale_strides[3][3];    // [branch][page, head, row], bytes

  const void* __restrict__ indices;
  const float* __restrict__ global_scale;
  uint32_t num_kv_heads;
  uint32_t num_total_heads;  // 2 * num_kv_heads + 1
  uint32_t page_size;
  uint32_t total_jobs;  // num_tokens * num_total_heads
};

template <int64_t kHeadDim, bool kUsePDL>
struct Nvfp4StoreTrait {
  static_assert(kHeadDim % kNvfp4BlockSize == 0, "head dim must be a multiple of 16");
  /// One thread per 16-element block; the whole head is one worker group.
  static constexpr uint32_t kWorkerSize = kHeadDim / kNvfp4BlockSize;

  template <typename T>
  SGL_DEVICE static void forward(const Nvfp4StoreParams& params) {
    using namespace device;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t work_id = tid / kWorkerSize;   // which (token, head) job
    const uint32_t block_id = tid % kWorkerSize;  // which 16-element block
    if (work_id >= params.total_jobs) return;

    const uint32_t num_kv_heads = params.num_kv_heads;
    const uint32_t token_id = work_id / params.num_total_heads;
    const uint32_t slot_id = work_id % params.num_total_heads;

    uint32_t branch, head_id;
    if (slot_id < num_kv_heads) {
      branch = kMainK;
      head_id = slot_id;
    } else if (slot_id < 2 * num_kv_heads) {
      branch = kMainV;
      head_id = slot_id - num_kv_heads;
    } else {
      branch = kIndexK;
      head_id = 0;
    }

    const auto src = static_cast<const bf16_t*>(params.in_ptrs[branch]) + token_id * params.in_token_stride[branch] +
                     head_id * params.in_head_stride[branch] + block_id * kNvfp4BlockSize;

    // Every global read, `indices` included, stays behind the PDL wait.
    PDLWaitPrimary<kUsePDL>();
    const auto loc = static_cast<int64_t>(static_cast<const T*>(params.indices)[token_id]);
    const int64_t page = loc / params.page_size;
    const int64_t row = loc % params.page_size;

    const auto& pay_s = params.payload_strides[branch];
    const auto& sc_s = params.scale_strides[branch];
    const auto payload = pointer::offset(
        params.payload_ptrs[branch], page * pay_s[0], head_id * pay_s[1], row * pay_s[2], block_id * kNvfp4BlockBytes);
    const auto scale =
        pointer::offset(params.scale_ptrs[branch], page * sc_s[0], head_id * sc_s[1], row * sc_s[2], block_id);

    uint8_t scale_byte;
    const uint64_t packed = minimax_nvfp4::quantize_block(src, *params.global_scale, &scale_byte);
    *static_cast<uint64_t*>(payload) = packed;
    *static_cast<uint8_t*>(scale) = scale_byte;
    PDLTriggerSecondary<kUsePDL>();
  }
};

template <typename Trait, typename T>
__global__ void nvfp4_store_kv_index_kernel(const __grid_constant__ Nvfp4StoreParams params) {
  Trait::template forward<T>(params);
}

/// \brief Quantize main K/V and index K to NVFP4 and scatter into their caches.
///
/// Replaces the three `contiguous` copies, three `nvfp4_kv_quantize` calls, four
/// page/row arithmetic kernels and six scatters of the eager path with a single
/// launch, writing all six cache buffers byte-identically to that path.
///
/// \tparam kHeadDim Head dimension shared by the main and index caches.
/// \tparam kUsePDL Whether to emit the PDL wait/trigger pair.
/// \param k Main K, `[tokens, num_kv_heads, kHeadDim]` BF16; any outer strides.
/// \param v Main V, same shape and dtype as `k`.
/// \param idx_k Index K, `[tokens, kHeadDim]` BF16.
/// \param indices Physical slots per token, int32 or int64.
/// \param global_scale One-element fp32 outer scale on the same device.
/// \param page_size Rows per cache page.
template <int64_t kHeadDim, bool kUsePDL>
void nvfp4_store_kv_index(
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView idx_k,
    tvm::ffi::TensorView k_payload,
    tvm::ffi::TensorView k_scale,
    tvm::ffi::TensorView v_payload,
    tvm::ffi::TensorView v_scale,
    tvm::ffi::TensorView idx_k_payload,
    tvm::ffi::TensorView idx_k_scale,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView global_scale,
    int64_t page_size) {
  using namespace host;
  auto T = SymbolicSize{"tokens"};
  auto H = SymbolicSize{"kv_heads"};
  auto P = SymbolicSize{"pages"};
  auto R = SymbolicSize{"page_size"};
  auto in_dtype = SymbolicDType{};
  auto indice_dtype = SymbolicDType{};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  constexpr int64_t kPayloadBytes = kHeadDim / 2;
  constexpr int64_t kScaleBytes = kHeadDim / kNvfp4BlockSize;

  // The reference quantizer is BF16-only on this path; FP16 stays on the eager
  // route until it has its own bit-exact acceptance.
  TensorMatcher({T, H, kHeadDim})  //
      .with_strides({-1, -1, 1})
      .with_dtype<bf16_t>(in_dtype)
      .with_device(device)
      .verify(k)
      .verify(v);
  TensorMatcher({T, kHeadDim})  //
      .with_strides({-1, 1})
      .with_dtype<bf16_t>(in_dtype)
      .with_device(device)
      .verify(idx_k);

  // Cache views may sit inside wider per-page rows; only each token row must be packed.
  TensorMatcher({P, H, R, kPayloadBytes})  //
      .with_strides({-1, -1, kPayloadBytes, 1})
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(k_payload)
      .verify(v_payload);
  TensorMatcher({P, H, R, kScaleBytes})  //
      .with_strides({-1, -1, kScaleBytes, 1})
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(k_scale)
      .verify(v_scale);
  TensorMatcher({P, R, kPayloadBytes})  //
      .with_strides({-1, kPayloadBytes, 1})
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(idx_k_payload);
  TensorMatcher({P, R, kScaleBytes})  //
      .with_strides({-1, kScaleBytes, 1})
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(idx_k_scale);
  // Payload blocks are written as one 64-bit store.
  for (const auto& payload : {k_payload, v_payload}) {
    CHECK_HOST(payload.stride(0) % kNvfp4BlockBytes == 0 && payload.stride(1) % kNvfp4BlockBytes == 0)
        << "main payload page and head strides must be multiples of " << kNvfp4BlockBytes;
  }
  CHECK_HOST(idx_k_payload.stride(0) % kNvfp4BlockBytes == 0)
      << "index payload page stride must be a multiple of " << kNvfp4BlockBytes;
  for (const auto& payload : {k_payload, v_payload, idx_k_payload}) {
    CHECK_HOST(reinterpret_cast<uintptr_t>(payload.data_ptr()) % kNvfp4BlockBytes == 0)
        << "payload base must be " << kNvfp4BlockBytes << "-byte aligned";
  }

  TensorMatcher({T})  //
      .with_dtype<int32_t, int64_t>(indice_dtype)
      .with_device(device)
      .verify(indices);
  TensorMatcher({1})  //
      .with_dtype<float>()
      .with_device(device)
      .verify(global_scale);

  CHECK_HOST(R.unwrap() == page_size) << "page size " << page_size << " does not match the cache row count "
                                      << R.unwrap();

  const int64_t num_tokens = T.unwrap();
  const int64_t num_kv_heads = H.unwrap();
  const int64_t num_total_heads = 2 * num_kv_heads + 1;

  auto params = Nvfp4StoreParams{};
  params.in_ptrs[kMainK] = k.data_ptr();
  params.in_ptrs[kMainV] = v.data_ptr();
  params.in_ptrs[kIndexK] = idx_k.data_ptr();
  params.in_token_stride[kMainK] = k.stride(0);
  params.in_token_stride[kMainV] = v.stride(0);
  params.in_token_stride[kIndexK] = idx_k.stride(0);
  params.in_head_stride[kMainK] = k.stride(1);
  params.in_head_stride[kMainV] = v.stride(1);
  params.in_head_stride[kIndexK] = 0;

  params.payload_ptrs[kMainK] = k_payload.data_ptr();
  params.payload_ptrs[kMainV] = v_payload.data_ptr();
  params.payload_ptrs[kIndexK] = idx_k_payload.data_ptr();
  params.scale_ptrs[kMainK] = k_scale.data_ptr();
  params.scale_ptrs[kMainV] = v_scale.data_ptr();
  params.scale_ptrs[kIndexK] = idx_k_scale.data_ptr();

  // Index caches carry no head axis, so their head stride is zero.
  const tvm::ffi::TensorView main_payload[2] = {k_payload, v_payload};
  const tvm::ffi::TensorView main_scale[2] = {k_scale, v_scale};
  for (int b = 0; b < 2; ++b) {
    params.payload_strides[b][0] = main_payload[b].stride(0);
    params.payload_strides[b][1] = main_payload[b].stride(1);
    params.payload_strides[b][2] = main_payload[b].stride(2);
    params.scale_strides[b][0] = main_scale[b].stride(0);
    params.scale_strides[b][1] = main_scale[b].stride(1);
    params.scale_strides[b][2] = main_scale[b].stride(2);
  }
  params.payload_strides[kIndexK][0] = idx_k_payload.stride(0);
  params.payload_strides[kIndexK][1] = 0;
  params.payload_strides[kIndexK][2] = idx_k_payload.stride(1);
  params.scale_strides[kIndexK][0] = idx_k_scale.stride(0);
  params.scale_strides[kIndexK][1] = 0;
  params.scale_strides[kIndexK][2] = idx_k_scale.stride(1);

  params.indices = indices.data_ptr();
  params.global_scale = static_cast<const float*>(global_scale.data_ptr());
  params.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
  params.num_total_heads = static_cast<uint32_t>(num_total_heads);
  params.page_size = static_cast<uint32_t>(page_size);
  params.total_jobs = static_cast<uint32_t>(num_tokens * num_total_heads);
  if (params.total_jobs == 0) return;

  using Trait = Nvfp4StoreTrait<kHeadDim, kUsePDL>;
  constexpr uint32_t kBlockSize = 256u;
  const uint32_t num_blocks = div_ceil(params.total_jobs * Trait::kWorkerSize, kBlockSize);
  const auto kernel = indice_dtype.is_type<int32_t>()  //
                          ? nvfp4_store_kv_index_kernel<Trait, int32_t>
                          : nvfp4_store_kv_index_kernel<Trait, int64_t>;
  LaunchKernel(num_blocks, kBlockSize, device.unwrap())  //
      .enable_pdl(kUsePDL)(kernel, params);
}

}  // namespace sglang
