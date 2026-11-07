#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <tvm/ffi/extra/cuda/device_guard.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <map>
#include <mutex>

namespace sglang {
namespace fp4_candidates {

using namespace cute;
using BF16 = cutlass::bfloat16_t;

constexpr int kHeads = 32;
constexpr int kDim = 128;
constexpr int kCandidates = 64;
constexpr int kThreads = 128;

// Each warp covers all 32 heads for 16 candidates (two N8 groups). Both
// operands are natural D128 and K-major, so ldmatrix loads them without
// a transpose. Keeping each candidate within one warp permits the
// exact H32 epilogue reduction without a shared-memory round trip.
using TiledMMA = decltype(make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, Layout<Shape<_1, _4, _1>>{}));
using QLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_32, _128>, Stride<_128, _1>>{}));
using KLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _128>, Stride<_128, _1>>{}));

struct Params {
  const BF16* q;
  const BF16* weights;
  const void* requests;
  const void* req_table;
  const void* lengths;
  const uint8_t* cache;
  const void* blocks;
  float* out;
  int64_t cache_stride;
  int64_t table_stride;
  int64_t block_stride;
  int64_t width;
  int64_t max_len;
  int32_t page_size;
  int32_t ratio;
  bool requests_i64;
  bool table_i64;
  bool lengths_i64;
  bool blocks_i64;
};

struct SharedStorage {
  alignas(128) BF16 q[kHeads * kDim];
  alignas(128) BF16 k[kCandidates * kDim];
  bool valid[kCandidates];
  // The 64-byte BF16 row occupies the original structure's trailing padding.
  BF16 weights[kHeads];
};

__device__ __forceinline__ int64_t load_index(const void* ptr, int64_t offset, bool is_i64) {
  return is_i64 ? static_cast<const int64_t*>(ptr)[offset] : static_cast<const int32_t*>(ptr)[offset];
}

__device__ __forceinline__ float round_bf16(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ float e2m1(uint32_t code) {
  const uint32_t magnitude = code & 7;
  // The E2M1 nonnegative values are {0, .5, 1, 1.5, 2, 3, 4, 6}.
  const uint32_t exponent = magnitude >> 1;
  const float value =
      exponent == 0 ? float(magnitude) * 0.5f : (1.0f + float(magnitude & 1) * 0.5f) * float(1u << (exponent - 1));
  // Match Triton's +0-value sign operation, including code 8 -> +0.
  return code & 8 ? 0.0f - value : value;
}

__device__ __forceinline__ float ue8m0(uint32_t exponent) {
  // Reproduce the pinned Triton PTX, including its FTZ behavior. The
  // mathematical UE8M0 byte 0 is 2^-127, but ex2.approx.ftz flushes it to
  // zero *before* E2M1 multiplication; this differs from exact dequantization.
  float scale;
  const float power = float(exponent) - 127.0f;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(scale) : "f"(power));
  return scale;
}

__device__ __forceinline__ uint32_t decode_fp4_normal_bf16_bits(uint32_t code, uint32_t exponent) {
  const uint32_t magnitude = code & 7u;
  // The scalar E2M1 sign operation returns +0 for both zero codes.
  if (magnitude == 0u) return 0u;
  const uint32_t sign = (code & 8u) << 12;
  const int adjustment = magnitude == 1u ? -1 : int(magnitude >> 1) - 1;
  const uint32_t biased_exponent = uint32_t(int(exponent) + adjustment);
  const uint32_t mantissa = (magnitude > 1u ? magnitude & 1u : 0u) << 6;
  return sign | (biased_exponent << 7) | mantissa;
}

__device__ __forceinline__ bool is_normal_fp4_exponent(uint32_t exponent) {
  return exponent >= 2u && exponent <= 252u;
}

__device__ __forceinline__ uint32_t decode_fp4_pair_fast_prmt(uint32_t packed_byte, uint32_t exp_bias) {
  // At exponent2, the eight magnitude bits divided by64 are
  // {0,2,4,5,6,7,8,9}. Select both magnitudes and zero-fill their high bytes.
  const uint32_t selector = (packed_byte & 7u) | ((packed_byte & 0x70u) << 4);
  const uint32_t base = __byte_perm(0x05040200u, 0x09080706u, selector) << 6;
  const uint32_t low_base = base & 0xffffu;
  const uint32_t high_base = base >> 16;
  // Both FP4 zero codes remain +0. Exceptional exponents never enter here.
  const uint32_t low = low_base ? (((packed_byte & 8u) << 12) | (low_base + exp_bias)) : 0u;
  const uint32_t high = high_base ? (((packed_byte & 0x80u) << 8) | (high_base + exp_bias)) : 0u;
  return low | (high << 16);
}

__device__ __forceinline__ uint32_t decode_fp4_pair_scaled_bf16x2(uint32_t packed_byte, float scale) {
  // Actual scalar FTZ/subnormal/overflow arithmetic, including signed zeros
  // and zero*infinity NaNs. The segment caller computes its scale once.
  const BF16 low(e2m1(packed_byte & 15u) * scale);
  const BF16 high(e2m1((packed_byte >> 4) & 15u) * scale);
  return uint32_t(low.raw()) | (uint32_t(high.raw()) << 16);
}

__device__ __forceinline__ uint32_t decode_fp4_pair_to_bf16x2(uint32_t packed_byte, uint32_t exponent) {
  return is_normal_fp4_exponent(exponent) ? decode_fp4_pair_fast_prmt(packed_byte, (exponent - 2u) << 7)
                                          : decode_fp4_pair_scaled_bf16x2(packed_byte, ue8m0(exponent));
}

template <bool kNormal>
__device__ __forceinline__ void
store_fp4_segment_impl(uint4 payload, uint32_t exp_bias, float scale, BF16* shared_k, int n, int d0) {
  CUTE_UNROLL
  for (int word = 0; word < 4; ++word) {
    const uint32_t value = word == 0 ? payload.x : word == 1 ? payload.y : word == 2 ? payload.z : payload.w;
    uint32_t decoded[4];
    CUTE_UNROLL
    for (int byte = 0; byte < 4; ++byte) {
      const uint32_t packed = (value >> (8 * byte)) & 255u;
      if constexpr (kNormal) {
        decoded[byte] = decode_fp4_pair_fast_prmt(packed, exp_bias);
      } else {
        decoded[byte] = decode_fp4_pair_scaled_bf16x2(packed, scale);
      }
    }
    // Swizzle preserves these eight BF16 values as one aligned16-byte chunk.
    const int offset = KLayout{}(make_coord(n, 2 * (d0 + word * 4)));
    *reinterpret_cast<uint4*>(shared_k + offset) = make_uint4(decoded[0], decoded[1], decoded[2], decoded[3]);
  }
}

// One outlined call per exceptional 16-byte segment, never per packed byte.
// Pass payload words by value so no pointer to a thread-local array escapes.
__device__ __noinline__ void store_fp4_segment_slow(uint4 payload, uint32_t exponent, BF16* shared_k, int n, int d0) {
  store_fp4_segment_impl<false>(payload, 0u, ue8m0(exponent), shared_k, n, d0);
}

struct Payload32 {
  uint32_t word[8];
};

__device__ __forceinline__ Payload32 load_payload_32(const uint8_t* ptr) {
  Payload32 payload;
  // Cache tensor bases and page strides can be unaligned. Both 16-byte
  // accesses are valid only when this actual half-payload address is aligned.
  if ((reinterpret_cast<uintptr_t>(ptr) & 15u) == 0u) {
    const uint4 first = *reinterpret_cast<const uint4*>(ptr);
    const uint4 second = *reinterpret_cast<const uint4*>(ptr + 16);
    payload.word[0] = first.x;
    payload.word[1] = first.y;
    payload.word[2] = first.z;
    payload.word[3] = first.w;
    payload.word[4] = second.x;
    payload.word[5] = second.y;
    payload.word[6] = second.z;
    payload.word[7] = second.w;
  } else {
    CUTE_UNROLL
    for (int word = 0; word < 8; ++word) {
      uint32_t value = 0;
      CUTE_UNROLL
      for (int byte = 0; byte < 4; ++byte) {
        value |= uint32_t(ptr[4 * word + byte]) << (8 * byte);
      }
      payload.word[word] = value;
    }
  }
  return payload;
}

__device__ __forceinline__ void copy_q16_async(const BF16* src, BF16* dst) {
  const uint32_t shared_address = uint32_t(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(shared_address), "l"(src) : "memory");
}

template <int kPageSize, int kRatio, int kTilesPerCTA = 1>
__global__ __launch_bounds__(kThreads) void score_kernel(const __grid_constant__ Params p) {
  __shared__ SharedStorage smem;
  struct RowMetadata {
    int64_t visible;
    int64_t request;
  };
  static_assert(sizeof(RowMetadata) == 16);
  // T1 has no live reference to this object.
  __shared__ RowMetadata row_metadata;
  const int tid = threadIdx.x;
  const int64_t row = blockIdx.x;
  // Page geometry and compression ratio are host-known ABI inputs. The
  // generic instantiation retains runtime signed division for other pages.
  const int64_t page_size = kPageSize == 0 ? p.page_size : kPageSize;
  const int64_t ratio = kRatio == 0 ? p.ratio : kRatio;

  auto sQ = make_tensor(make_smem_ptr(smem.q), QLayout{});
  auto sK = make_tensor(make_smem_ptr(smem.k), KLayout{});

  static_assert(kTilesPerCTA >= 1 && kTilesPerCTA <= 4);
  // Q is staged once per CTA; K remains a single reusable [64, 128] tile.
  // Each K tile is shared by all heads, never expanded to [64, 32, 128].
  const BF16* q_row = p.q + row * kHeads * kDim;
  // Contiguous Q views can retain a two-byte storage offset. Such views
  // use the original scalar copy; the 16-byte path checks the actual base.
  const bool q_aligned = (reinterpret_cast<uintptr_t>(q_row) & 15u) == 0u;
  if (q_aligned) {
    constexpr int kQChunksPerThread = kHeads * kDim / (kThreads * 8);
    static_assert(kHeads * kDim % (kThreads * 8) == 0);
    CUTE_UNROLL
    for (int chunk = 0; chunk < kQChunksPerThread; ++chunk) {
      const int i = tid * 8 + chunk * kThreads * 8;
      // The swizzle preserves each eight-BF16 chunk's low three bits.
      const int offset = QLayout{}(make_coord(i / kDim, i % kDim));
      copy_q16_async(q_row + i, smem.q + offset);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  } else {
    for (int i = tid; i < kHeads * kDim; i += kThreads) {
      const int h = i / kDim;
      const int d = i % kDim;
      sQ(h, d) = p.q[(row * kHeads + h) * kDim + d];
    }
  }

  // Copy weights without conversion once per CTA. The first tile's existing
  // publication barrier makes them visible to every epilogue reader.
  if (tid < kHeads) smem.weights[tid] = p.weights[row * kHeads + tid];

  if constexpr (kTilesPerCTA > 1) {
    // Tensor rows are valid even for all-masked candidates. Only thread 0
    // produces these scalars; table/cache dereferences remain valid-masked.
    if (tid == 0) {
      const int64_t length = load_index(p.lengths, row, p.lengths_i64);
      row_metadata.visible = length < p.max_len ? length : p.max_len;
      row_metadata.request = load_index(p.requests, row, p.requests_i64);
    }
    // Metadata is consumed before K staging's publication barrier, so this
    // initial barrier is required. It does not complete the Q async copies.
    __syncthreads();
  }

  // Keep one decoder/MMA body instead of unrolling the large tile loop.
  // The tail condition is CTA-uniform; no thread exits a required barrier.
#pragma unroll 1
  for (int tile = 0; tile < kTilesPerCTA; ++tile) {
    const int64_t first = (int64_t(blockIdx.y) * kTilesPerCTA + tile) * kCandidates;
    if constexpr (kTilesPerCTA > 1) {
      if (first >= p.width) break;
    }

    // Two threads stage each candidate. Keep its mapping, payload and scale
    // values local so payload reads need no shared-metadata producer barrier.
    const int n = tid >> 1;
    const int half = tid & 1;
    const int64_t compact = first + n;
    const int64_t block =
        compact < p.width ? load_index(p.blocks, row * p.block_stride + compact / 8, p.blocks_i64) : -1;
    const int64_t logical = block * 8 + compact % 8;
    int64_t visible = 0;
    if constexpr (kTilesPerCTA > 1) {
      visible = row_metadata.visible;
    } else {
      const int64_t length = load_index(p.lengths, row, p.lengths_i64);
      visible = length < p.max_len ? length : p.max_len;
    }
    const bool valid = compact < p.width && block >= 0 && logical >= 0 && logical < visible;
    Payload32 payload{};
    uint32_t exponent0 = 127;
    uint32_t exponent1 = 127;
    if (valid) {
      int64_t request = 0;
      if constexpr (kTilesPerCTA > 1) {
        request = row_metadata.request;
      } else {
        request = load_index(p.requests, row, p.requests_i64);
      }
      const int64_t slot = load_index(p.req_table, request * p.table_stride + logical * ratio, p.table_i64) / ratio;
      const int64_t page_base = (slot / page_size) * p.cache_stride;
      const int64_t off = slot % page_size;
      const uint8_t* payload_ptr = p.cache + page_base + off * 64 + half * 32;
      const uint8_t* scale_ptr = p.cache + page_base + page_size * 64 + off * 4 + half * 2;
      payload = load_payload_32(payload_ptr);
      exponent0 = scale_ptr[0];
      exponent1 = scale_ptr[1];
    }
    if constexpr (kTilesPerCTA > 1) {
      // These candidate loads only produced private register values. Wait
      // for all prior K/valid readers before overwriting either shared array.
      if (tile != 0) __syncthreads();
    }
    if (half == 0) smem.valid[n] = valid;

    // Natural K coordinates keep each decoded BF16 pair contiguous and
    // aligned for one 32-bit shared store. Boundary decoding stays unchanged.
    CUTE_UNROLL
    for (int segment = 0; segment < 2; ++segment) {
      const uint32_t exponent = segment == 0 ? exponent0 : exponent1;
      const uint4 words = make_uint4(
          payload.word[segment * 4],
          payload.word[segment * 4 + 1],
          payload.word[segment * 4 + 2],
          payload.word[segment * 4 + 3]);
      const int d0 = half * 32 + segment * 16;
      if (is_normal_fp4_exponent(exponent)) {
        const uint32_t exp_bias = (exponent - 2u) << 7;
        store_fp4_segment_impl<true>(words, exp_bias, 0.0f, smem.k, n, d0);
      } else {
        store_fp4_segment_slow(words, exponent, smem.k, n, d0);
      }
    }
    // Complete the one-time Q copies before the first publication barrier.
    // Every iteration publishes its new K tile and validity to all MMA readers.
    if (q_aligned && tile == 0) asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();

    TiledMMA mma;
    auto thr_mma = mma.get_slice(tid);
    auto tQs = thr_mma.partition_A(sQ);
    auto tKs = thr_mma.partition_B(sK);
    auto rQ = thr_mma.make_fragment_A(tQs);
    auto rK = thr_mma.make_fragment_B(tKs);
    auto acc = partition_fragment_C(mma, Shape<_32, _64>{});
    clear(acc);

    auto copy_q = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto copy_k = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, BF16>{}, mma);
    auto thr_copy_q = copy_q.get_slice(tid);
    auto thr_copy_k = copy_k.get_slice(tid);
    auto qs = thr_copy_q.partition_S(sQ);
    auto ks = thr_copy_k.partition_S(sK);
    auto qr = thr_copy_q.retile_D(rQ);
    auto kr = thr_copy_k.retile_D(rK);

    // Experimental natural D128 accumulation changes the pinned parity
    // reference's K order, while retaining BF16 operands and epilogue rounding.
    CUTE_UNROLL
    for (int k = 0; k < size<2>(rQ); ++k) {
      copy(copy_q, qs(_, _, k), qr(_, _, k));
      copy(copy_k, ks(_, _, k), kr(_, _, k));
      gemm(mma, rQ(_, _, k), rK(_, _, k), acc);
    }

    const auto identity = make_identity_tensor(Shape<_32, _64>{});
    const auto coords = thr_mma.partition_C(identity);
    CUTE_STATIC_ASSERT_V(size(acc) == Int<16>{});
    CUTE_UNROLL
    for (int i = 0; i < size(acc); ++i) {
      const int h = get<0>(coords(i));
      const float dot = round_bf16(acc(i));
      const float relu = fmaxf(dot, 0.0f);
      const float weight = float(smem.weights[h]);
      acc(i) = round_bf16(relu * weight);
    }

    // With the 1x4 warp layout, lane l owns heads g=l/4 and g+8,+16,+24
    // for columns warp*8 + 2*(l%4) + {0,1,32,33}. For each column the four
    // head values occupy fragment offsets base+{0,2,4,6}, with base={0,1,8,9}.
    // Retain the reference's left-associated per-group additions before
    // reducing groups g with xor 4,2,1 (lane xor 16,8,4 respectively).
    CUTE_UNROLL
    for (int column = 0; column < 4; ++column) {
      const int base = (column & 1) + (column >> 1) * 8;
      float value = acc(base) + acc(base + 2);
      value += acc(base + 4);
      value += acc(base + 6);
      value += __shfl_xor_sync(0xffffffffu, value, 16);
      value += __shfl_xor_sync(0xffffffffu, value, 8);
      value += __shfl_xor_sync(0xffffffffu, value, 4);
      const int n = get<1>(coords(base));
      if ((tid & 31) < 4 && first + n < p.width) {
        p.out[row * p.width + first + n] = smem.valid[n] ? round_bf16(value) : __uint_as_float(0xff800000u);
      }
    }
  }
}

}  // namespace fp4_candidates

struct FP4IndexerCandidatesKernel {
  template <int kTilesPerCTA>
  static auto select_kernel(int64_t page_size, int64_t ratio) {
    static_assert(kTilesPerCTA >= 1 && kTilesPerCTA <= 4);
    using namespace fp4_candidates;
    return page_size == 64 ? (ratio == 1 ? score_kernel<64, 1, kTilesPerCTA> : score_kernel<64, 2, kTilesPerCTA>)
                           : score_kernel<0, 0, kTilesPerCTA>;
  }

  // Read-only attributes of the same kernel used by run_fixed<T>. Intended
  // for controlled calibration; querying does not launch the kernel.
  // field: 0=SM count, 1=active CTAs/SM, 2=registers/thread,
  //        3=static shared bytes, 4=local bytes/thread, 5=max block threads.
  template <int kTilesPerCTA>
  static int64_t query_fixed(int64_t device_id, int64_t page_size, int64_t ratio, int64_t field) {
    using namespace host;
    RuntimeCheck(device_id >= 0 && device_id <= INT32_MAX, "device_id out of range");
    RuntimeCheck(page_size > 0 && page_size <= INT32_MAX, "page_size out of range");
    RuntimeCheck(ratio == 1 || ratio == 2, "invalid ratio");
    const tvm::ffi::CUDADeviceGuard guard(static_cast<int32_t>(device_id));
    const auto kernel = select_kernel<kTilesPerCTA>(page_size, ratio);
    if (field == 0) return runtime::get_sm_count(static_cast<int32_t>(device_id));
    if (field == 1) return runtime::get_blocks_per_sm(kernel, fp4_candidates::kThreads, 0);
    cudaFuncAttributes attributes{};
    RuntimeDeviceCheck(cudaFuncGetAttributes(&attributes, kernel));
    switch (field) {
      case 2:
        return attributes.numRegs;
      case 3:
        return attributes.sharedSizeBytes;
      case 4:
        return attributes.localSizeBytes;
      case 5:
        return attributes.maxThreadsPerBlock;
      default:
        RuntimeCheck(false, "invalid resource query field");
        return 0;
    }
  }

  struct PreparedLaunch {
    fp4_candidates::Params params;
    DLDevice device;
    int64_t rows;
  };

  struct DeviceStats {
    int32_t sm_count;
    int32_t resident_ctas;
  };

  static uint64_t ceil_div_u64(uint64_t value, uint64_t divisor) {
    return value / divisor + (value % divisor != 0);
  }

  // Calibrated page64 policy: T3 at rho=3 (24/8), otherwise T1.
  // Both CUDA counters are positive int32 values, so their uint64 product
  // times three cannot overflow. Other page geometries retain T1.
  static constexpr uint64_t kRho = 3;
  static_assert(kRho > 0 && kRho <= 4);

  static int choose_tiles_from_stats(int64_t rows, int64_t width, int64_t page_size, const DeviceStats& stats) {
    // Only page64 has performance calibration. Generic fixed variants remain
    // available for testing, while their public automatic path stays at T1.
    if (rows <= 0 || width <= 128 || page_size != 64) return 1;
    if (stats.sm_count <= 0 || stats.resident_ctas <= 0) return 1;
    const uint64_t n64 = ceil_div_u64(static_cast<uint64_t>(width), 64);
    if (n64 < 3) return 1;
    const uint64_t groups = ceil_div_u64(n64, 3);
    const uint64_t sms = static_cast<uint64_t>(stats.sm_count);
    const uint64_t scaled = sms * static_cast<uint64_t>(stats.resident_ctas) * kRho;
    const uint64_t required_ctas = scaled > sms ? scaled : sms;
    // Avoid rows*groups and overflow-prone (a+b-1)/b formulations.
    return static_cast<uint64_t>(rows) >= ceil_div_u64(required_ctas, groups) ? 3 : 1;
  }

  template <typename Kernel>
  static DeviceStats cached_device_stats(Kernel kernel, int32_t device_id) {
    // The kernel pointer identifies T and its page/ratio specialization.
    // Block size128 and dynamic shared0 are fixed; no carveout is changed.
    static std::mutex mutex;
    static std::map<std::pair<const void*, int32_t>, DeviceStats> cache;
    const std::lock_guard<std::mutex> lock(mutex);
    const auto key = std::make_pair(reinterpret_cast<const void*>(kernel), device_id);
    if (const auto it = cache.find(key); it != cache.end()) return it->second;
    const tvm::ffi::CUDADeviceGuard guard(device_id);
    DeviceStats stats{};
    host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&stats.sm_count, cudaDevAttrMultiProcessorCount, device_id));
    host::RuntimeDeviceCheck(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&stats.resident_ctas, kernel, fp4_candidates::kThreads, 0));
    host::RuntimeCheck(stats.sm_count > 0 && stats.resident_ctas > 0, "invalid device capacity");
    return cache.emplace(key, stats).first->second;
  }

  static int choose_tiles(int32_t device_id, int64_t rows, int64_t width, int64_t page_size, int64_t ratio) {
    if (rows <= 0 || width <= 128 || page_size != 64) return 1;
    const auto kernel = select_kernel<3>(page_size, ratio);
    return choose_tiles_from_stats(rows, width, page_size, cached_device_stats(kernel, device_id));
  }

  // Test-only export of the actual chooser used by run; no scorer clone.
  static int64_t query_dispatch(int64_t device_id, int64_t rows, int64_t width, int64_t page_size, int64_t ratio) {
    using namespace host;
    RuntimeCheck(device_id >= 0 && device_id <= INT32_MAX, "device_id out of range");
    RuntimeCheck(rows >= 0 && width >= 0, "negative dispatch shape");
    RuntimeCheck(page_size > 0 && page_size <= INT32_MAX, "page_size out of range");
    RuntimeCheck(ratio == 1 || ratio == 2, "invalid ratio");
    return choose_tiles(static_cast<int32_t>(device_id), rows, width, page_size, ratio);
  }

  static PreparedLaunch prepare(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView weights,
      const tvm::ffi::TensorView requests,
      const tvm::ffi::TensorView req_table,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView cache,
      const tvm::ffi::TensorView blocks,
      const tvm::ffi::TensorView out,
      const int64_t page_size,
      const int64_t max_len,
      const int64_t ratio) {
    using namespace host;
    using namespace fp4_candidates;
    auto batch = SymbolicSize{"batch"};
    auto nblocks = SymbolicSize{"candidate_blocks"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();
    TensorMatcher({batch, kHeads, 128}).with_dtype<bf16_t>().with_device(device).verify(q);
    TensorMatcher({batch, kHeads}).with_dtype<bf16_t>().with_device(device).verify(weights);
    TensorMatcher({batch}).with_dtype<int32_t, int64_t>().with_device(device).verify(requests);
    TensorMatcher({batch}).with_dtype<int32_t, int64_t>().with_device(device).verify(lengths);
    TensorMatcher({-1, -1}).with_strides({-1, 1}).with_dtype<int32_t, int64_t>().with_device(device).verify(req_table);
    TensorMatcher({batch, nblocks}).with_dtype<int32_t, int64_t>().with_device(device).verify(blocks);
    TensorMatcher({-1, page_size * 68}).with_strides({-1, 1}).with_dtype<uint8_t>().with_device(device).verify(cache);
    TensorMatcher({batch, nblocks.unwrap() * 8}).with_dtype<float>().with_device(device).verify(out);
    RuntimeCheck(page_size > 0 && page_size <= INT32_MAX, "page_size out of range");
    RuntimeCheck(
        (ratio == 1 || ratio == 2) && max_len >= 0 && max_len <= req_table.size(1) / ratio, "invalid ratio or max_len");
    const int64_t rows = batch.unwrap();
    const int64_t width = nblocks.unwrap() * 8;
    const Params params{
        .q = static_cast<const BF16*>(q.data_ptr()),
        .weights = static_cast<const BF16*>(weights.data_ptr()),
        .requests = requests.data_ptr(),
        .req_table = req_table.data_ptr(),
        .lengths = lengths.data_ptr(),
        .cache = static_cast<const uint8_t*>(cache.data_ptr()),
        .blocks = blocks.data_ptr(),
        .out = static_cast<float*>(out.data_ptr()),
        .cache_stride = cache.stride(0),
        .table_stride = req_table.stride(0),
        .block_stride = blocks.stride(0),
        .width = width,
        .max_len = max_len,
        .page_size = static_cast<int32_t>(page_size),
        .ratio = static_cast<int32_t>(ratio),
        .requests_i64 = is_type<int64_t>(requests.dtype()),
        .table_i64 = is_type<int64_t>(req_table.dtype()),
        .lengths_i64 = is_type<int64_t>(lengths.dtype()),
        .blocks_i64 = is_type<int64_t>(blocks.dtype()),
    };
    return {params, device.unwrap(), rows};
  }

  template <int kTilesPerCTA>
  static void launch_fixed(const PreparedLaunch& prepared) {
    using namespace host;
    using namespace fp4_candidates;
    const Params& params = prepared.params;
    if (prepared.rows == 0 || params.width == 0) return;
    const auto kernel = select_kernel<kTilesPerCTA>(params.page_size, params.ratio);
    LaunchKernel(
        dim3(prepared.rows, div_ceil(params.width, int64_t(kCandidates) * kTilesPerCTA)), kThreads, prepared.device)
        .launch(kernel, params);
  }

  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView weights,
      const tvm::ffi::TensorView requests,
      const tvm::ffi::TensorView req_table,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView cache,
      const tvm::ffi::TensorView blocks,
      const tvm::ffi::TensorView out,
      const int64_t page_size,
      const int64_t max_len,
      const int64_t ratio) {
    const PreparedLaunch prepared =
        prepare(q, weights, requests, req_table, lengths, cache, blocks, out, page_size, max_len, ratio);
    if (prepared.rows == 0 || prepared.params.width == 0) return;
    const int tiles = choose_tiles(prepared.device.device_id, prepared.rows, prepared.params.width, page_size, ratio);
    if (tiles == 3)
      launch_fixed<3>(prepared);
    else
      launch_fixed<1>(prepared);
  }

  template <int kTilesPerCTA>
  static void run_fixed(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView weights,
      const tvm::ffi::TensorView requests,
      const tvm::ffi::TensorView req_table,
      const tvm::ffi::TensorView lengths,
      const tvm::ffi::TensorView cache,
      const tvm::ffi::TensorView blocks,
      const tvm::ffi::TensorView out,
      const int64_t page_size,
      const int64_t max_len,
      const int64_t ratio) {
    launch_fixed<kTilesPerCTA>(
        prepare(q, weights, requests, req_table, lengths, cache, blocks, out, page_size, max_len, ratio));
  }
};

}  // namespace sglang
