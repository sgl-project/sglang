#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

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

    auto N = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"num_heads"};
    auto D = SymbolicSize{"k_head_dim"};
    auto D_nope = SymbolicSize{"nope_head_dim"};
    auto D_rope = SymbolicSize{"rope_head_dim"};
    auto S0_k = SymbolicSize{"k_stride_0"};
    auto S1_k = SymbolicSize{"k_stride_1"};
    auto S0_k_nope = SymbolicSize{"k_nope_stride_0"};
    auto S1_k_nope = SymbolicSize{"k_nope_stride_1"};
    auto S0_k_rope = SymbolicSize{"k_rope_stride_0"};
    auto device = SymbolicDevice{};

    // Set known fixed values
    H.set_value(NUM_LOCAL_HEADS);
    D.set_value(K_HEAD_DIM);
    D_nope.set_value(QK_NOPE_HEAD_DIM);
    D_rope.set_value(QK_ROPE_HEAD_DIM);

    // Verify k: [num_tokens, num_heads, k_head_dim]
    TensorMatcher({N, H, D}).with_strides({S0_k, S1_k, 1}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(k);

    // Verify k_nope: [num_tokens, num_heads, nope_head_dim]
    TensorMatcher({N, H, D_nope})
        .with_strides({S0_k_nope, S1_k_nope, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(k_nope);

    // Verify k_rope: [num_tokens, 1, rope_head_dim]
    TensorMatcher({N, 1, D_rope})
        .with_strides({S0_k_rope, -1, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(k_rope);

    // Check alignment
    RuntimeCheck(reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0, "Tensor k must be 16-byte aligned");
    RuntimeCheck(reinterpret_cast<uintptr_t>(k_nope.data_ptr()) % 16 == 0, "Tensor k_nope must be 16-byte aligned");
    RuntimeCheck(reinterpret_cast<uintptr_t>(k_rope.data_ptr()) % 16 == 0, "Tensor k_rope must be 16-byte aligned");

    const int num_tokens = static_cast<int>(N.unwrap());

    constexpr int num_warps_per_block = 32;
    const int grid_size = div_ceil(num_tokens * NUM_HEAD_CHUNKS, num_warps_per_block);
    const int block_size = num_warps_per_block * 32;

    LaunchKernel(grid_size, block_size, device.unwrap())(
        concat_mla_k_kernel,
        static_cast<bf16_t*>(k.data_ptr()),
        static_cast<const bf16_t*>(k_nope.data_ptr()),
        static_cast<const bf16_t*>(k_rope.data_ptr()),
        num_tokens,
        S0_k.unwrap(),
        static_cast<int>(S1_k.unwrap()),
        S0_k_nope.unwrap(),
        static_cast<int>(S1_k_nope.unwrap()),
        S0_k_rope.unwrap());
  }
};

// ======================= concat_mla_absorb_q Kernel =======================

constexpr int A_LAST_DIM = 512;
constexpr int B_LAST_DIM = 64;
constexpr int OUT_LAST_DIM = A_LAST_DIM + B_LAST_DIM;

__global__ void concat_mla_absorb_q_kernel(
    bf16_t* a,
    bf16_t* b,
    bf16_t* out,
    const int num_items,
    const int dim_1,
    const int64_t a_stride_0,
    const int a_stride_1,
    const int64_t b_stride_0,
    const int b_stride_1,
    const int64_t out_stride_0,
    const int out_stride_1) {
  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane_id = get_lane_id();

  const int idx_0 = flat_warp_id / dim_1;
  const int idx_1 = flat_warp_id % dim_1;

  if (flat_warp_id >= num_items) {
    return;
  }

  using ABufType = int4;
  constexpr int A_NUM_UNROLL = 2;
  static_assert(sizeof(ABufType) * A_NUM_UNROLL == A_LAST_DIM * sizeof(a[0]) / 32);
  ABufType a_buf[A_NUM_UNROLL];

  using BBufType = int;
  constexpr int B_NUM_UNROLL = 1;
  static_assert(sizeof(BBufType) * B_NUM_UNROLL == B_LAST_DIM * sizeof(b[0]) / 32);
  BBufType b_buf;

  {
    const BBufType* base_addr = reinterpret_cast<BBufType*>(b + idx_0 * b_stride_0 + idx_1 * b_stride_1);
    b_buf = *(base_addr + lane_id);
  }

#pragma unroll
  for (int i = 0; i < A_NUM_UNROLL; ++i) {
    const ABufType* base_addr = reinterpret_cast<ABufType*>(a + idx_0 * a_stride_0 + idx_1 * a_stride_1);
    a_buf[i] = *(base_addr + i * 32 + lane_id);
  }

  {
    BBufType* base_addr = reinterpret_cast<BBufType*>(out + idx_0 * out_stride_0 + idx_1 * out_stride_1 + A_LAST_DIM);
    *(base_addr + lane_id) = b_buf;
  }

#pragma unroll
  for (int i = 0; i < A_NUM_UNROLL; ++i) {
    ABufType* base_addr = reinterpret_cast<ABufType*>(out + idx_0 * out_stride_0 + idx_1 * out_stride_1);
    *(base_addr + i * 32 + lane_id) = a_buf[i];
  }
}

struct ConcatMlaAbsorbQKernel {
  static void run(tvm::ffi::TensorView a, tvm::ffi::TensorView b, tvm::ffi::TensorView out) {
    using namespace host;

    auto N0_a = SymbolicSize{"a_dim_0"};
    auto N1_a = SymbolicSize{"a_dim_1"};
    auto D_a = SymbolicSize{"a_last_dim"};
    auto N0_b = SymbolicSize{"b_dim_0"};
    auto N1_b = SymbolicSize{"b_dim_1"};
    auto D_b = SymbolicSize{"b_last_dim"};
    auto N0_out = SymbolicSize{"out_dim_0"};
    auto N1_out = SymbolicSize{"out_dim_1"};
    auto D_out = SymbolicSize{"out_last_dim"};
    auto S0_a = SymbolicSize{"a_stride_0"};
    auto S1_a = SymbolicSize{"a_stride_1"};
    auto S0_b = SymbolicSize{"b_stride_0"};
    auto S1_b = SymbolicSize{"b_stride_1"};
    auto S0_out = SymbolicSize{"out_stride_0"};
    auto S1_out = SymbolicSize{"out_stride_1"};
    auto device = SymbolicDevice{};

    // Set known fixed values
    D_a.set_value(A_LAST_DIM);
    D_b.set_value(B_LAST_DIM);
    D_out.set_value(OUT_LAST_DIM);

    // Verify a: [dim_0, dim_1, A_LAST_DIM]
    TensorMatcher({N0_a, N1_a, D_a})
        .with_strides({S0_a, S1_a, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(a);

    // Verify b: [dim_0, dim_1, B_LAST_DIM]
    TensorMatcher({N0_b, N1_b, D_b})
        .with_strides({S0_b, S1_b, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(b);

    // Verify out: [dim_0, dim_1, OUT_LAST_DIM]
    TensorMatcher({N0_out, N1_out, D_out})
        .with_strides({S0_out, S1_out, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(out);

    // Check alignment
    RuntimeCheck(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0, "Tensor a must be 16-byte aligned");
    RuntimeCheck(reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 == 0, "Tensor b must be 16-byte aligned");
    RuntimeCheck(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "Tensor out must be 16-byte aligned");

    // Verify dimensions match: a.size(0) * a.size(1) == b.size(0) * b.size(1)
    RuntimeCheck(
        N0_a.unwrap() * N1_a.unwrap() == N0_b.unwrap() * N1_b.unwrap(),
        "Dimension mismatch: a.size(0) * a.size(1) must equal b.size(0) * b.size(1)");
    RuntimeCheck(N1_a.unwrap() == N1_b.unwrap(), "Dimension mismatch: a.size(1) must equal b.size(1)");

    const int num_items = static_cast<int>(N0_a.unwrap() * N1_a.unwrap());
    const int dim_1 = static_cast<int>(N1_a.unwrap());

    constexpr int num_warps_per_block = 32;
    const int grid_size = div_ceil(num_items, num_warps_per_block);
    const int block_size = num_warps_per_block * 32;

    LaunchKernel(grid_size, block_size, device.unwrap())(
        concat_mla_absorb_q_kernel,
        static_cast<bf16_t*>(a.data_ptr()),
        static_cast<bf16_t*>(b.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        num_items,
        dim_1,
        S0_a.unwrap(),
        static_cast<int>(S1_a.unwrap()),
        S0_b.unwrap(),
        static_cast<int>(S1_b.unwrap()),
        S0_out.unwrap(),
        static_cast<int>(S1_out.unwrap()));
  }
};

}  // namespace
