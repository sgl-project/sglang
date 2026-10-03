#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <cuda_fp8.h>

namespace sglang {

struct FusedStoreCacheParam {
  const void* __restrict__ input;
  void* __restrict__ cache;
  const void* __restrict__ indices;
  uint32_t num_tokens;
};

[[maybe_unused]]
SGL_DEVICE float fp8_e4m3_clip(float val) {
  namespace math = device::math;
  return math::max(math::min(val, kFP8E4M3Max), -kFP8E4M3Max);
}

[[maybe_unused]]
SGL_DEVICE fp8x2_e4m3_t pack_fp8(float x, float y) {
  return fp8x2_e4m3_t{fp32x2_t{fp8_e4m3_clip(x), fp8_e4m3_clip(y)}};
}

// Quantize one fp32 value to an E2M1 4-bit nibble stored in a uint8.
// Bucket boundaries [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] mirror torch's
// `bucketize(..., right=False)`; sign bit is set only for non-zero codes.
[[maybe_unused]]
SGL_DEVICE uint8_t e2m1_nibble(float x) {
  float a = fminf(fabsf(x), 6.0f);
  uint8_t code;
  if (a <= 0.25f)
    code = 0;
  else if (a <= 0.75f)
    code = 1;
  else if (a <= 1.25f)
    code = 2;
  else if (a <= 1.75f)
    code = 3;
  else if (a <= 2.5f)
    code = 4;
  else if (a <= 3.5f)
    code = 5;
  else if (a <= 5.0f)
    code = 6;
  else
    code = 7;
  uint8_t sign = (x < 0.0f && code != 0) ? 1u : 0u;
  return code | static_cast<uint8_t>(sign << 3);
}

// Pack two fp32 values into one e2m1x2 byte via PTX hardware instruction.
// lo nibble = e2m1(lo_val), hi nibble = e2m1(hi_val).
[[maybe_unused]]
SGL_DEVICE uint8_t e2m1x2_pack_ptx(float hi_val, float lo_val) {
  uint8_t tmp;
  asm("cvt.rn.satfinite.e2m1x2.f32 %0, %1, %2;" : "=r"(tmp) : "f"(hi_val), "f"(lo_val));
  return tmp;
}

// 1 / 2^(exp - 127) as fp32. Equivalent to `1.0f / __uint_as_float(exp << 23)`.
[[maybe_unused]]
SGL_DEVICE float inv_scale_ue8m0(int32_t exp) {
  return __uint_as_float((127 + 127 - exp) << 23);
}

template <typename KeyT, typename IndicesT, uint32_t kPageBits, bool kUsePDL>
__global__ void fused_store_indexer_cache(const __grid_constant__ FusedStoreCacheParam param) {
  using namespace device;

  /// NOTE: 132 = 128 + 4
  constexpr int64_t kPageBytes = 132 << kPageBits;

  // each warp handles 128 elements, each block handles multiple rows
  const auto& [input, cache, indices, num_tokens] = param;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto global_wid = global_tid / 32;
  const auto lane_id = threadIdx.x % 32;

  if (global_wid >= num_tokens) return;

  PDLWaitPrimary<kUsePDL>();  // wait for primary kernel

  // prefetch the index
  const auto index = static_cast<const IndicesT*>(indices)[global_wid];
  // always load the value from input (don't store if invalid)
  using KeyT2 = packed_t<KeyT>;
  using InStorage = AlignedVector<KeyT2, 2>;
  using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;
  const auto elems = static_cast<const InStorage*>(input)[global_tid];
  const auto [x0, x1] = cast<fp32x2_t>(elems[0]);
  const auto [y0, y1] = cast<fp32x2_t>(elems[1]);
  const auto local_max = fmaxf(fmaxf(fabs(x0), fabs(x1)), fmaxf(fabs(y0), fabs(y1)));
  const auto abs_max = warp::reduce_max(local_max);
  // use normal fp32 scale
  const auto scale = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
  const auto inv_scale = 1.0f / scale;
  const int32_t page = index >> kPageBits;
  const int32_t offset = index & ((1 << kPageBits) - 1);
  const auto page_ptr = pointer::offset(cache, page * kPageBytes);
  const auto value_ptr = pointer::offset(page_ptr, offset * 128);
  const auto scale_ptr = pointer::offset(page_ptr, 128 << kPageBits, offset * 4);
  OutStorage result;
  result[0] = pack_fp8(x0 * inv_scale, x1 * inv_scale);
  result[1] = pack_fp8(y0 * inv_scale, y1 * inv_scale);
  static_cast<OutStorage*>(value_ptr)[lane_id] = result;
  static_cast<float*>(scale_ptr)[0] = scale;

  PDLTriggerSecondary<kUsePDL>();  // launch secondary kernel
}

template <typename KeyT, typename IndicesT, uint32_t kPageBits, bool kUsePDL>
__global__ void fused_store_indexer_mxfp4_cache(const __grid_constant__ FusedStoreCacheParam param) {
  using namespace device;

  // MXFP4: head_dim=128, block=32 elements (4 blocks/token), 2 nibbles/byte.
  // Per token: 64 packed FP4 bytes + 4 ue8m0 scale bytes = 68 bytes.
  constexpr int kHeadDim = 128;
  constexpr int kMxfp4Block = 32;
  constexpr int kNumScaleBlocks = kHeadDim / kMxfp4Block;      // 4
  constexpr int kPackedBytesPerToken = kHeadDim / 2;           // 64
  constexpr int kScaleBytesPerToken = kNumScaleBlocks;         // 4
  constexpr int kElemsPerLane = 4;                             // 4 fp32 per lane
  constexpr int kLanesPerBlock = kMxfp4Block / kElemsPerLane;  // 8
  constexpr int64_t kPageBytes = static_cast<int64_t>((kPackedBytesPerToken + kScaleBytesPerToken) << kPageBits);

  const auto& [input, cache, indices, num_tokens] = param;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto global_wid = global_tid / 32;
  const auto lane_id = threadIdx.x % 32;
  // Each warp handles one token; lane k holds elems [4k, 4k+3].
  // Block id = lane_id / 8; lanes 0..7 → block 0, 8..15 → block 1, etc.
  const auto block_idx_in_token = lane_id / kLanesPerBlock;

  if (global_wid >= num_tokens) return;

  PDLWaitPrimary<kUsePDL>();

  const auto index = static_cast<const IndicesT*>(indices)[global_wid];

  using KeyT2 = packed_t<KeyT>;
  using InStorage = AlignedVector<KeyT2, 2>;
  const auto elems = static_cast<const InStorage*>(input)[global_tid];
  const auto [x0, x1] = cast<fp32x2_t>(elems[0]);
  const auto [y0, y1] = cast<fp32x2_t>(elems[1]);

  // Per-block (8-lane group) absmax → ue8m0 = ceil(log2(amax / 6.0)) + 127.
  const float local_max = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(y0), fabsf(y1)));
  float amax = warp::reduce_max<kLanesPerBlock>(local_max);
  amax = fmaxf(amax, 1e-4f);
  // ceil(log2(amax / 6.0)) using fp32 bit trick:
  //   (float_to_bits(ratio) + 0x007FFFFF) & 0x7F800000
  const float ratio = amax / 6.0f;
  uint32_t dequant_scale_exp = (__float_as_uint(ratio) + 0x007FFFFFu) & 0x7F800000u;
  int32_t exp_biased = static_cast<int32_t>(dequant_scale_exp >> 23);
  exp_biased = exp_biased > 254 ? 254 : exp_biased;
  const uint8_t ue8m0 = static_cast<uint8_t>(exp_biased);
  const float inv_scale = __uint_as_float(static_cast<uint32_t>(254 - exp_biased) << 23);

#if __CUDA_ARCH__ >= 890
  // ── PTX cvt.rn.satfinite.e2m1x2.f32 (requires sm_89+ / PPU >= 150) ──
  const float s0 = x0 * inv_scale;
  const float s1 = x1 * inv_scale;
  const float s2 = y0 * inv_scale;
  const float s3 = y1 * inv_scale;
  const uint8_t byte0 = e2m1x2_pack_ptx(s1, s0);
  const uint8_t byte1 = e2m1x2_pack_ptx(s3, s2);
#else
  // ── Fallback: manual bucket quantization ──
  const uint8_t n0 = e2m1_nibble(x0 * inv_scale);
  const uint8_t n1 = e2m1_nibble(x1 * inv_scale);
  const uint8_t n2 = e2m1_nibble(y0 * inv_scale);
  const uint8_t n3 = e2m1_nibble(y1 * inv_scale);
  const uint8_t byte0 = n0 | static_cast<uint8_t>(n1 << 4);
  const uint8_t byte1 = n2 | static_cast<uint8_t>(n3 << 4);
#endif
  const uint16_t packed = static_cast<uint16_t>(byte0) | (static_cast<uint16_t>(byte1) << 8);

  // Page layout (kPageSize tokens):
  //   [0, kPageSize * 64):           packed FP4 (64 bytes/token)
  //   [kPageSize * 64, +kPageSize*4): ue8m0 scales (4 bytes/token)
  // DeepGEMM treats each token's 4 ue8m0 bytes as a single int32 (kv_sf).
  const int32_t page = index >> kPageBits;
  const int32_t offset = index & ((1 << kPageBits) - 1);
  const auto page_ptr = pointer::offset(cache, page * kPageBytes);
  const auto value_ptr = pointer::offset(page_ptr, offset * kPackedBytesPerToken);
  const auto scale_ptr = pointer::offset(page_ptr, kPackedBytesPerToken << kPageBits, offset * kScaleBytesPerToken);

  // Streaming store: bypass L1/L2 cache for large sequential KV writes.
  {
    const uint16_t* store_addr = reinterpret_cast<const uint16_t*>(value_ptr) + lane_id;
    asm volatile("st.global.cs.u16 [%0], %1;" ::"l"(store_addr), "h"(packed) : "memory");
  }
  // The first lane in each 8-lane group writes its block's ue8m0 byte.
  if ((lane_id % kLanesPerBlock) == 0) {
    const uint8_t* scale_addr = static_cast<const uint8_t*>(scale_ptr) + block_idx_in_token;
    asm volatile("st.global.cs.u8 [%0], %1;" ::"l"(scale_addr), "r"((uint32_t)ue8m0) : "memory");
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <typename KeyT, typename IndicesT, uint32_t kPageSize, bool kUsePDL>
struct FusedStoreCacheIndexerMXFP4Kernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  // 64 packed FP4 bytes + 4 ue8m0 bytes per token = 68 bytes.
  static constexpr int64_t kPageBytes = 68 * kPageSize;
  static constexpr auto kernel = fused_store_indexer_mxfp4_cache<KeyT, IndicesT, kLogSize, kUsePDL>;

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogSize == kPageSize);

  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView cache, tvm::ffi::TensorView indices) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, 128})  // input
        .with_dtype<KeyT>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({-1, -1})  // cache
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({N})  // indices
        .with_dtype<IndicesT>()
        .with_device(device_)
        .verify(indices);
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = FusedStoreCacheParam{
        .input = input.data_ptr(),
        .cache = cache.data_ptr(),
        .indices = indices.data_ptr(),
        .num_tokens = num_tokens,
    };
    const auto kBlockSize = 128;
    const auto num_blocks = div_ceil(num_tokens * 32, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

template <typename KeyT, typename IndicesT, uint32_t kPageSize, bool kUsePDL>
struct FusedStoreCacheIndexerKernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  /// NOTE: 132 = 128 + 4 (128 represent K and 4 represent scale)
  static constexpr int64_t kPageBytes = 132 * kPageSize;
  static constexpr auto kernel = fused_store_indexer_cache<KeyT, IndicesT, kLogSize, kUsePDL>;

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogSize == kPageSize);

  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView cache, tvm::ffi::TensorView indices) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, 128})  // input
        .with_dtype<KeyT>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({-1, -1})  // cache
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({N})  // indices
        .with_dtype<IndicesT>()
        .with_device(device_)
        .verify(indices);
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = FusedStoreCacheParam{
        .input = input.data_ptr(),
        .cache = cache.data_ptr(),
        .indices = indices.data_ptr(),
        .num_tokens = num_tokens,
    };
    const auto kBlockSize = 128;
    const auto num_blocks = div_ceil(num_tokens * 32, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
