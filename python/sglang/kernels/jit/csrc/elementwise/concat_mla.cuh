#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace sglang {

// ======================= Memory Utilities =======================
// Adapted from DeepEP: https://github.com/deepseek-ai/DeepEP/blob/main/csrc/kernels/utils.cuh

SGL_DEVICE int get_lane_id() {
  int lane_id;
  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
  return lane_id;
}

SGL_DEVICE void st_na_global_v1(const int* ptr, int v) {
  asm volatile("st.global.L1::no_allocate.s32 [%0], %1;" ::"l"(ptr), "r"(v) : "memory");
}

SGL_DEVICE void st_na_global_v2(const int2* ptr, const int2& v) {
  asm volatile("st.global.L1::no_allocate.v2.s32 [%0], {%1, %2};" ::"l"(ptr), "r"(v.x), "r"(v.y) : "memory");
}

SGL_DEVICE int ld_na_global_v1(const int* ptr) {
  int r;
  asm volatile("ld.global.nc.L1::no_allocate.s32 %0, [%1];" : "=r"(r) : "l"(ptr));
  return r;
}

SGL_DEVICE int2 ld_na_global_v2(const int2* ptr) {
  int2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.s32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(ptr));
  return r;
}

SGL_DEVICE void prefetch_L2(const void* p) {
#if defined(ENABLE_L2_PREFETCH)
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
#endif
}

// ======================= concat_mla_k Kernel =======================

constexpr int NUM_LOCAL_HEADS = 128;
constexpr int QK_NOPE_HEAD_DIM = 128;
constexpr int QK_ROPE_HEAD_DIM = 64;
constexpr int K_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;

constexpr int HEAD_CHUNK_SIZE = 16;
constexpr int NUM_HEAD_CHUNKS = NUM_LOCAL_HEADS / HEAD_CHUNK_SIZE;

__global__ void concat_mla_k_kernel(
    bf16_t* __restrict__ k,
    const bf16_t* __restrict__ k_nope,
    const bf16_t* __restrict__ k_rope,
    const int num_tokens,
    const int64_t k_stride_0,
    const int k_stride_1,
    const int64_t k_nope_stride_0,
    const int k_nope_stride_1,
    const int64_t k_rope_stride_0) {
  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int token_id = flat_warp_id / NUM_HEAD_CHUNKS;
  const int head_chunk_id = flat_warp_id % NUM_HEAD_CHUNKS;
  const int lane_id = get_lane_id();
  if (token_id >= num_tokens) return;

  using NopeVec = int2;  // 8B/thread, 32 threads = 256B/row
  using RopeVec = int;   // 4B/thread, 32 threads = 128B/row
  static_assert(sizeof(NopeVec) * 32 == QK_NOPE_HEAD_DIM * sizeof(bf16_t), "nope vec mismatch");
  static_assert(sizeof(RopeVec) * 32 == QK_ROPE_HEAD_DIM * sizeof(bf16_t), "rope vec mismatch");

  const int head_row0 = head_chunk_id * HEAD_CHUNK_SIZE;

  const int2* __restrict__ nope_src =
      reinterpret_cast<const int2*>(k_nope + token_id * k_nope_stride_0 + head_row0 * k_nope_stride_1) + lane_id;

  int2* __restrict__ nope_dst = reinterpret_cast<int2*>(k + token_id * k_stride_0 + head_row0 * k_stride_1) + lane_id;

  int* __restrict__ rope_dst =
      reinterpret_cast<int*>(k + token_id * k_stride_0 + head_row0 * k_stride_1 + QK_NOPE_HEAD_DIM) + lane_id;

  const int nope_src_stride_v = (k_nope_stride_1 >> 2);  // int2 covers 4 bf16
  const int nope_dst_stride_v = (k_stride_1 >> 2);
  const int rope_dst_stride_v = (k_stride_1 >> 1);  // int covers 2 bf16

  const int* rope_base = reinterpret_cast<const int*>(k_rope + token_id * k_rope_stride_0);
  const RopeVec rope_val = ld_na_global_v1(rope_base + lane_id);

  prefetch_L2(nope_src);
  NopeVec cur = ld_na_global_v2(nope_src);

#pragma unroll
  for (int i = 0; i < HEAD_CHUNK_SIZE; ++i) {
    NopeVec next;
    if (i + 1 < HEAD_CHUNK_SIZE) {
      const int2* next_src = nope_src + nope_src_stride_v;
      prefetch_L2(next_src);
      next = ld_na_global_v2(next_src);
    }

    st_na_global_v2(nope_dst, cur);
    st_na_global_v1(rope_dst, rope_val);

    nope_src += nope_src_stride_v;
    nope_dst += nope_dst_stride_v;
    rope_dst += rope_dst_stride_v;

    cur = next;
  }
}

struct ConcatMlaKKernel {
  static void run(tvm::ffi::TensorView k, tvm::ffi::TensorView k_nope, tvm::ffi::TensorView k_rope) {
    using namespace host;

    // The widest access is `v2.s32` (8B), and the `>> 2` stride stepping needs
    // each row stride to be a whole number of int2's -- which is the same 8B.
    constexpr int64_t kAlignNope = 8;
    constexpr int64_t kAlignRope = 4;  // rope is read with `v1.s32`

    auto N = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, NUM_LOCAL_HEADS, K_HEAD_DIM})  //
        .with_strides({-1, -1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(kAlignNope)
        .verify(k);
    TensorMatcher({N, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM})  //
        .with_strides({-1, -1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(kAlignNope)
        .verify(k_nope);
    TensorMatcher({N, 1, QK_ROPE_HEAD_DIM})  //
        .with_strides({-1, -1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(kAlignRope)
        .verify(k_rope);

    const auto num_tokens = static_cast<int>(N.unwrap());
    constexpr uint32_t kNumWarpsPerBlock = 32;
    const auto grid_size = div_ceil(static_cast<uint32_t>(num_tokens * NUM_HEAD_CHUNKS), kNumWarpsPerBlock);

    LaunchKernel(grid_size, kNumWarpsPerBlock * 32, device.unwrap())(
        concat_mla_k_kernel,
        static_cast<bf16_t*>(k.data_ptr()),
        static_cast<const bf16_t*>(k_nope.data_ptr()),
        static_cast<const bf16_t*>(k_rope.data_ptr()),
        num_tokens,
        k.stride(0),
        static_cast<int>(k.stride(1)),
        k_nope.stride(0),
        static_cast<int>(k_nope.stride(1)),
        k_rope.stride(0));
  }
};

// ======================= concat_mla_absorb_q Kernel =======================

constexpr int A_LAST_DIM = 512;
constexpr int B_LAST_DIM = 64;
constexpr int OUT_LAST_DIM = A_LAST_DIM + B_LAST_DIM;

template <bool kUsePDL>
__global__ void concat_mla_absorb_q_kernel(
    const bf16_t* a,
    const bf16_t* b,
    bf16_t* out,
    const int num_items,
    const int dim_1,
    const int64_t a_stride_0,
    const int a_stride_1,
    const int64_t b_stride_0,
    const int b_stride_1,
    const int64_t out_stride_0,
    const int out_stride_1) {
  using namespace device;
  using enum warp::LoadStorePattern::type;
  constexpr int64_t kABytes = A_LAST_DIM * sizeof(bf16_t);
  constexpr int64_t kBBytes = B_LAST_DIM * sizeof(bf16_t);

  PDLWaitPrimary<kUsePDL>();

  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpThreads;
  if (flat_warp_id >= num_items) return;

  const int idx_0 = flat_warp_id / dim_1;
  const int idx_1 = flat_warp_id % dim_1;
  const auto out_row = out + idx_0 * out_stride_0 + idx_1 * out_stride_1;

  const auto b_val = warp::load_bytes<kBBytes, WARP_UNIFORM_16B>(b + idx_0 * b_stride_0 + idx_1 * b_stride_1);
  const auto a_val = warp::load_bytes<kABytes, WARP_UNIFORM_16B>(a + idx_0 * a_stride_0 + idx_1 * a_stride_1);

  PDLTriggerSecondary<kUsePDL>();

  warp::store_bytes<kBBytes, WARP_UNIFORM_16B>(out_row + A_LAST_DIM, b_val);
  warp::store_bytes<kABytes, WARP_UNIFORM_16B>(out_row, a_val);
}

template <bool kUsePDL>
struct ConcatMlaAbsorbQKernel {
  static void run(tvm::ffi::TensorView a, tvm::ffi::TensorView b, tvm::ffi::TensorView out) {
    using namespace host;
    using device::warp::LoadStorePattern;
    using enum LoadStorePattern::type;

    constexpr int64_t kAlignA = LoadStorePattern::get_vec_bytes<A_LAST_DIM * sizeof(bf16_t), WARP_UNIFORM_16B>();
    constexpr int64_t kAlignB = LoadStorePattern::get_vec_bytes<B_LAST_DIM * sizeof(bf16_t), WARP_UNIFORM_16B>();
    // `out` carries both halves, so it must satisfy the wider one; the B half
    // starts A_LAST_DIM in, which must not break B's own alignment.
    constexpr int64_t kAlignOut = std::max(kAlignA, kAlignB);
    static_assert(A_LAST_DIM * sizeof(bf16_t) % kAlignB == 0, "A width must not misalign the B half");

    // a, b and out share the leading [dim_0, dim_1]: the kernel indexes all three with one (idx_0, idx_1).
    auto N0 = SymbolicSize{"dim_0"};
    auto N1 = SymbolicSize{"dim_1"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N0, N1, A_LAST_DIM})  //
        .with_strides({-1, -1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(kAlignA)
        .verify(a);
    TensorMatcher({N0, N1, B_LAST_DIM})  //
        .with_strides({-1, -1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(kAlignB)
        .verify(b);
    TensorMatcher({N0, N1, OUT_LAST_DIM})  //
        .with_strides({-1, -1, 1})
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(kAlignOut)
        .verify(out);

    const auto num_items = static_cast<int>(N0.unwrap() * N1.unwrap());
    const auto dim_1 = static_cast<int>(N1.unwrap());
    constexpr uint32_t kNumWarpsPerBlock = 32;
    const auto grid_size = div_ceil(static_cast<uint32_t>(num_items), kNumWarpsPerBlock);

    LaunchKernel(grid_size, kNumWarpsPerBlock * 32, device.unwrap())
        .enable_pdl(kUsePDL)(
            concat_mla_absorb_q_kernel<kUsePDL>,
            static_cast<const bf16_t*>(a.data_ptr()),
            static_cast<const bf16_t*>(b.data_ptr()),
            static_cast<bf16_t*>(out.data_ptr()),
            num_items,
            dim_1,
            a.stride(0),
            static_cast<int>(a.stride(1)),
            b.stride(0),
            static_cast<int>(b.stride(1)),
            out.stride(0),
            static_cast<int>(out.stride(1)));
  }
};

}  // namespace sglang
