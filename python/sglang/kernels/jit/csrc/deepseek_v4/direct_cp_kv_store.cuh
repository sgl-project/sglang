#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <cuda_fp8.h>

namespace {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

struct DirectCPKVStoreParam {
  const void* __restrict__ input;
  const void* __restrict__ indices;
  uint64_t cache_multicast;
  uint32_t num_tokens;
  uint32_t num_pages;
};

SGL_DEVICE uint16_t fp8x2_bits(fp8x2_e4m3_t value) {
  static_assert(sizeof(value) == sizeof(uint16_t));
  return *reinterpret_cast<const uint16_t*>(&value);
}

SGL_DEVICE uint32_t bf16x2_bits(bf16x2_t value) {
  static_assert(sizeof(value) == sizeof(uint32_t));
  return *reinterpret_cast<const uint32_t*>(&value);
}

SGL_DEVICE void multimem_store_v4_u32(
    void* ptr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile(
      "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
      :
      : "l"(ptr), "f"(__uint_as_float(x)), "f"(__uint_as_float(y)),
        "f"(__uint_as_float(z)), "f"(__uint_as_float(w))
      : "memory");
}

SGL_DEVICE void multimem_store_v2_u32(void* ptr, uint32_t x, uint32_t y) {
  asm volatile(
      "multimem.st.relaxed.sys.global.v2.f32 [%0], {%1, %2};"
      :
      : "l"(ptr), "f"(__uint_as_float(x)), "f"(__uint_as_float(y))
      : "memory");
}

template <typename Float, typename IndicesT, uint32_t kPageBits>
__global__ __launch_bounds__(256, 2) void direct_cp_store_flashmla_cache(
    DirectCPKVStoreParam param) {
  using namespace device;
  constexpr int64_t kPageBytes = host::div_ceil(584 << kPageBits, 576) * 576;

  const uint32_t bid = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t wid = tid / 32;
  const uint32_t lane = tid % 32;
  __shared__ uint8_t scales[8];

  const int64_t index =
      static_cast<int64_t>(static_cast<const IndicesT*>(param.indices)[bid]);
  if (index < 0 || index >= static_cast<int64_t>(param.num_pages) << kPageBits) {
    __trap();
  }
  using Float2 = packed_t<Float>;
  const auto elems = static_cast<const Float2*>(param.input)[tid + bid * 256];
  auto* cache = reinterpret_cast<uint8_t*>(param.cache_multicast);
  const int32_t page = index >> kPageBits;
  const int32_t offset = index & ((1 << kPageBits) - 1);
  auto* page_ptr = cache + static_cast<int64_t>(page) * kPageBytes;
  auto* value_ptr = page_ptr + static_cast<int64_t>(offset) * 576;

  if (wid != 7) {
    const auto [x, y] = cast<fp32x2_t>(elems);
    const auto abs_max = warp::reduce_max(fmaxf(fabs(x), fabs(y)));
    const auto scale_raw = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
    const auto scale_ue8m0 = cast_to_ue8m0(scale_raw);
    const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
    const uint32_t packed = fp8x2_bits(pack_fp8(x * inv_scale, y * inv_scale));

    const uint32_t pair = packed | (__shfl_down_sync(0xffffffff, packed, 1) << 16);
    const uint32_t group_base = lane & ~7u;
    const uint32_t r0 = __shfl_sync(0xffffffff, pair, group_base);
    const uint32_t r1 = __shfl_sync(0xffffffff, pair, group_base + 2);
    const uint32_t r2 = __shfl_sync(0xffffffff, pair, group_base + 4);
    const uint32_t r3 = __shfl_sync(0xffffffff, pair, group_base + 6);
    if ((lane & 7u) == 0) {
      multimem_store_v4_u32(value_ptr + tid * 2, r0, r1, r2, r3);
    }
    if (lane == 0) scales[wid] = static_cast<uint8_t>(scale_ue8m0);
  } else {
    const uint32_t packed = bf16x2_bits(cast<bf16x2_t>(elems));
    const uint32_t group_base = lane & ~3u;
    const uint32_t r0 = __shfl_sync(0xffffffff, packed, group_base);
    const uint32_t r1 = __shfl_sync(0xffffffff, packed, group_base + 1);
    const uint32_t r2 = __shfl_sync(0xffffffff, packed, group_base + 2);
    const uint32_t r3 = __shfl_sync(0xffffffff, packed, group_base + 3);
    if ((lane & 3u) == 0) {
      multimem_store_v4_u32(value_ptr + 448 + lane * 4, r0, r1, r2, r3);
    }
  }

  if (tid == 0) scales[7] = 0;
  __syncthreads();
  if (tid == 0) {
    const uint32_t lo = static_cast<uint32_t>(scales[0]) |
                        (static_cast<uint32_t>(scales[1]) << 8) |
                        (static_cast<uint32_t>(scales[2]) << 16) |
                        (static_cast<uint32_t>(scales[3]) << 24);
    const uint32_t hi = static_cast<uint32_t>(scales[4]) |
                        (static_cast<uint32_t>(scales[5]) << 8) |
                        (static_cast<uint32_t>(scales[6]) << 16);
    auto* scale_ptr = page_ptr + (576 << kPageBits) + static_cast<int64_t>(offset) * 8;
    multimem_store_v2_u32(scale_ptr, lo, hi);
  }
}

template <typename Float, typename IndicesT, uint32_t kPageSize>
struct DirectCPKVStoreKernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);

  static void run(
      tvm::ffi::TensorView input,
      tvm::ffi::TensorView cache,
      int64_t cache_multicast,
      tvm::ffi::TensorView indices) {
    using namespace host;
    static_assert(std::has_single_bit(kPageSize));
    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, 512}).with_dtype<Float>().with_device(device_).verify(input);
    TensorMatcher({-1, -1}).with_dtype<uint8_t>().with_device(device_).verify(cache);
    TensorMatcher({N}).with_dtype<IndicesT>().with_device(device_).verify(indices);
    RuntimeCheck(cache_multicast != 0, "cache multicast pointer is null");
    RuntimeCheck(cache.size(1) >= host::div_ceil(584 * kPageSize, 576) * 576,
                 "cache page stride is smaller than the packed DSV4 layout");
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    if (num_tokens == 0) return;
    const auto params = DirectCPKVStoreParam{
        .input = input.data_ptr(),
        .indices = indices.data_ptr(),
        .cache_multicast = static_cast<uint64_t>(cache_multicast),
        .num_tokens = num_tokens,
        .num_pages = static_cast<uint32_t>(cache.size(0)),
    };
    LaunchKernel(num_tokens, 256, device_.unwrap())(
        direct_cp_store_flashmla_cache<Float, IndicesT, kLogSize>, params);
  }
};

}  // namespace
