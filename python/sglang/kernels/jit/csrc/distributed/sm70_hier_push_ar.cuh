// SPDX-License-Identifier: Apache-2.0
// Two-step fp16 all-reduce for the 8xV100 hybrid NVLink mesh (NVLink quads
// {0-3}, {4-7}, bridged i <-> i+4) in one launch: reduce inside the quad, then
// across the pair. Each rank pushes 16-byte vectors into its peers' receive
// slots and polls its own; a slot word still holding kEmpty has not arrived.
// Each step equals custom_all_reduce.cuh cross_device_reduce_1stage bitwise:
// fp32 accumulation over the group ranks in order, one fp16 rounding.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_fp16.h>

#include <cstdint>

namespace sglang::sm70_hier_push_ar {

inline constexpr int kQuad = 4;
inline constexpr int kPair = 2;
inline constexpr int kHalves = 8;  // fp16 values per 16-byte vector
// Two fp16 NaNs; a kernel restores it in every slot word it consumed.
inline constexpr uint32_t kEmpty = 0xffffffffu;
// Every launch uses the full grid so every block flips its epoch on every call,
// keeping the parity equal across blocks whatever the size.
inline constexpr int kThreads = 256;
inline constexpr int kBlocks = 16;

struct Params {
  // Receive workspaces of the group ranks, [2 epochs][group size][slot_bytes].
  uint8_t* quad_ws[kQuad];
  uint8_t* pair_ws[kPair];
  uint32_t* epochs;  // [kBlocks], rank-local
  uint32_t slot_bytes;
  int32_t quad_rank;
  int32_t pair_rank;
};

SGL_DEVICE uint4 ld_relaxed(const uint8_t* addr) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(addr)
               : "memory");
  return v;
}

SGL_DEVICE void st_relaxed(uint8_t* addr, uint4 v) {
  asm volatile("st.relaxed.sys.global.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w),
               "l"(addr)
               : "memory");
}

SGL_DEVICE bool arrived(uint4 v) {
  return v.x != kEmpty && v.y != kEmpty && v.z != kEmpty && v.w != kEmpty;
}

/// A payload word equal to kEmpty (two NaNs) becomes two other NaNs.
SGL_DEVICE uint4 not_empty(uint4 v) {
  constexpr uint32_t kNaNs = 0xfe00fe00u;
  v.x = v.x == kEmpty ? kNaNs : v.x;
  v.y = v.y == kEmpty ? kNaNs : v.y;
  v.z = v.z == kEmpty ? kNaNs : v.z;
  v.w = v.w == kEmpty ? kNaNs : v.w;
  return v;
}

/// Spins until the slot holds a payload, then marks it empty again.
SGL_DEVICE uint4 take(uint8_t* slot) {
  uint4 v;
  do {
    v = ld_relaxed(slot);
  } while (!arrived(v));
  *reinterpret_cast<uint4*>(slot) = make_uint4(kEmpty, kEmpty, kEmpty, kEmpty);
  return v;
}

/// packed_reduce of cross_device_reduce_1stage: upcast slot 0, add the rest in
/// order in fp32, round once to fp16.
template <int kN>
SGL_DEVICE uint4 reduce(const uint4 (&v)[kN]) {
  float acc[kHalves];
  const __half* h0 = reinterpret_cast<const __half*>(&v[0]);
#pragma unroll
  for (int i = 0; i < kHalves; ++i) {
    acc[i] = __half2float(h0[i]);
  }
#pragma unroll
  for (int s = 1; s < kN; ++s) {
    const __half* hs = reinterpret_cast<const __half*>(&v[s]);
#pragma unroll
    for (int i = 0; i < kHalves; ++i) {
      acc[i] += __half2float(hs[i]);
    }
  }
  uint4 out;
  __half* ho = reinterpret_cast<__half*>(&out);
#pragma unroll
  for (int i = 0; i < kHalves; ++i) {
    ho[i] = __float2half(acc[i]);
  }
  return out;
}

__global__ __launch_bounds__(kThreads) void hier_push_kernel(const Params p,
                                                             const uint4* in,
                                                             uint4* out,
                                                             uint32_t num_vecs) {
  const uint32_t epoch = p.epochs[blockIdx.x] & 1;
  const uint32_t stride = gridDim.x * blockDim.x;
  const uint32_t first = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t quad_base = static_cast<size_t>(epoch) * kQuad * p.slot_bytes;
  const size_t pair_base = static_cast<size_t>(epoch) * kPair * p.slot_bytes;
  const size_t quad_src = quad_base + static_cast<size_t>(p.quad_rank) * p.slot_bytes;
  const size_t pair_src = pair_base + static_cast<size_t>(p.pair_rank) * p.slot_bytes;

  for (uint32_t vid = first; vid < num_vecs; vid += stride) {
    const uint4 v = not_empty(in[vid]);
#pragma unroll
    for (int q = 0; q < kQuad; ++q) {
      st_relaxed(p.quad_ws[q] + quad_src + vid * 16ull, v);
    }
  }
  uint8_t* const quad_rx = p.quad_ws[p.quad_rank] + quad_base;
  for (uint32_t vid = first; vid < num_vecs; vid += stride) {
    uint4 v[kQuad];
#pragma unroll
    for (int s = 0; s < kQuad; ++s) {
      v[s] = take(quad_rx + s * static_cast<size_t>(p.slot_bytes) + vid * 16ull);
    }
    const uint4 sum = not_empty(reduce(v));
#pragma unroll
    for (int r = 0; r < kPair; ++r) {
      st_relaxed(p.pair_ws[r] + pair_src + vid * 16ull, sum);
    }
  }
  uint8_t* const pair_rx = p.pair_ws[p.pair_rank] + pair_base;
  for (uint32_t vid = first; vid < num_vecs; vid += stride) {
    uint4 v[kPair];
#pragma unroll
    for (int s = 0; s < kPair; ++s) {
      v[s] = take(pair_rx + s * static_cast<size_t>(p.slot_bytes) + vid * 16ull);
    }
    out[vid] = reduce(v);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    p.epochs[blockIdx.x] = epoch ^ 1;
  }
}

/// out = sum over the 8 ranks of in, fp16 [n] with n * 2 a multiple of 16
/// bytes and at most slot_bytes; out may alias in. Every rank of both groups
/// must make the same sequence of calls.
void hier_push(tvm::ffi::TensorView in,
               tvm::ffi::TensorView out,
               tvm::ffi::TensorView epochs,
               int64_t quad_ws0,
               int64_t quad_ws1,
               int64_t quad_ws2,
               int64_t quad_ws3,
               int64_t pair_ws0,
               int64_t pair_ws1,
               int64_t slot_bytes,
               int64_t quad_rank,
               int64_t pair_rank) {
  using namespace host;
  SymbolicSize n = {"n"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({n}).with_dtype<fp16_t>().with_device<kDLCUDA>(device_).verify(in).verify(out);
  TensorMatcher({kBlocks}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(epochs);
  const int64_t bytes = n.unwrap() * 2;
  CHECK_HOST(bytes > 0 && bytes % 16 == 0 && bytes <= slot_bytes && slot_bytes % 16 == 0);
  CHECK_HOST(reinterpret_cast<uintptr_t>(in.data_ptr()) % 16 == 0);
  CHECK_HOST(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0);
  CHECK_HOST(quad_rank >= 0 && quad_rank < kQuad && pair_rank >= 0 && pair_rank < kPair);
  CHECK_HOST(slot_bytes <= 0xffffffffll / (2 * kQuad));
  Params p{};
  const int64_t quad[kQuad] = {quad_ws0, quad_ws1, quad_ws2, quad_ws3};
  const int64_t pair[kPair] = {pair_ws0, pair_ws1};
  for (int i = 0; i < kQuad; ++i) {
    CHECK_HOST(quad[i] != 0 && quad[i] % 16 == 0);
    p.quad_ws[i] = reinterpret_cast<uint8_t*>(quad[i]);
  }
  for (int i = 0; i < kPair; ++i) {
    CHECK_HOST(pair[i] != 0 && pair[i] % 16 == 0);
    p.pair_ws[i] = reinterpret_cast<uint8_t*>(pair[i]);
  }
  p.epochs = static_cast<uint32_t*>(epochs.data_ptr());
  p.slot_bytes = static_cast<uint32_t>(slot_bytes);
  p.quad_rank = static_cast<int32_t>(quad_rank);
  p.pair_rank = static_cast<int32_t>(pair_rank);
  LaunchKernel(kBlocks, kThreads, device_.unwrap())(hier_push_kernel,
                                                    p,
                                                    static_cast<const uint4*>(in.data_ptr()),
                                                    static_cast<uint4*>(out.data_ptr()),
                                                    static_cast<uint32_t>(bytes / 16));
}

}  // namespace sglang::sm70_hier_push_ar
