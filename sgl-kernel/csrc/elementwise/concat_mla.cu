#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <cuda_runtime.h>

#include "pytorch_extension_utils.h"

constexpr int NUM_LOCAL_HEADS = 128;
constexpr int QK_NOPE_HEAD_DIM = 128;
constexpr int QK_ROPE_HEAD_DIM = 64;
constexpr int K_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;

constexpr int HEADS_PER_BLOCK = 16;
static_assert(NUM_LOCAL_HEADS % HEADS_PER_BLOCK == 0, "HEADS_PER_BLOCK must divide NUM_LOCAL_HEADS");

__forceinline__ __device__ int get_lane_id() {
  int lane_id;
  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
  return lane_id;
}

int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

__global__ void concat_mla_k_kernel(
    nv_bfloat16* k,
    nv_bfloat16* k_nope,
    nv_bfloat16* k_rope,
    const int num_tokens,
    const int k_stride_0,
    const int k_stride_1,
    const int k_nope_stride_0,
    const int k_nope_stride_1,
    const int k_rope_stride_0) {
  const int token_id = blockIdx.x;
  const int head_chunk = blockIdx.y;
  if (token_id >= num_tokens) return;

  const int head_base = head_chunk * HEADS_PER_BLOCK;
  constexpr int VEC_ELEMS = 16 / sizeof(nv_bfloat16);               // 8
  constexpr int NOPE_VEC_PER_HEAD = QK_NOPE_HEAD_DIM / VEC_ELEMS;   // 16
  constexpr int ROPE_VEC_PER_TOKEN = QK_ROPE_HEAD_DIM / VEC_ELEMS;  // 8

  nv_bfloat16* __restrict__ k_tok = k + (int64_t)token_id * k_stride_0;
  const nv_bfloat16* __restrict__ kn_tok = k_nope + (int64_t)token_id * k_nope_stride_0;
  const nv_bfloat16* __restrict__ kr_tok = k_rope + (int64_t)token_id * k_rope_stride_0;

  // read rope (64 bf16 = 128B) to shared memory
  __shared__ int4 rope_smem_v[ROPE_VEC_PER_TOKEN];  // 8 * 16B = 128B
  for (int v = threadIdx.x; v < ROPE_VEC_PER_TOKEN; v += blockDim.x) {
    const int4* src = reinterpret_cast<const int4*>(kr_tok + v * VEC_ELEMS);
    rope_smem_v[v] = *src;
  }
  __syncthreads();

  // copy nope -> k[..., :QK_NOPE_HEAD_DIM]
  // 16 heads * 16 vec/head = 256 vec
  const int total_nope_vec = HEADS_PER_BLOCK * NOPE_VEC_PER_HEAD;
  for (int idx = threadIdx.x; idx < total_nope_vec; idx += blockDim.x) {
    const int h_local = idx / NOPE_VEC_PER_HEAD;
    const int v = idx % NOPE_VEC_PER_HEAD;

    nv_bfloat16* __restrict__ k_head = k_tok + (head_base + h_local) * k_stride_1;
    const nv_bfloat16* __restrict__ kn_head = kn_tok + (head_base + h_local) * k_nope_stride_1;

    int4* dst = reinterpret_cast<int4*>(k_head + v * VEC_ELEMS);
    const int4* src = reinterpret_cast<const int4*>(kn_head + v * VEC_ELEMS);
    *dst = *src;
  }

  // broadcast rope -> k[..., QK_NOPE_HEAD_DIM:]
  // 16 heads * 8 vec = 128 vec
  const int total_rope_vec = HEADS_PER_BLOCK * ROPE_VEC_PER_TOKEN;
  for (int idx = threadIdx.x; idx < total_rope_vec; idx += blockDim.x) {
    const int h_local = idx / ROPE_VEC_PER_TOKEN;
    const int v = idx % ROPE_VEC_PER_TOKEN;

    nv_bfloat16* __restrict__ k_head = k_tok + (head_base + h_local) * k_stride_1;
    int4* dst = reinterpret_cast<int4*>(k_head + QK_NOPE_HEAD_DIM + v * VEC_ELEMS);
    const int4 src = rope_smem_v[v];
    *dst = src;
  }
}

inline void check_tensor(const at::Tensor& t, int64_t shape0, int64_t shape1, int64_t shape2, c10::ScalarType dtype) {
  TORCH_CHECK_EQ(t.dim(), 3);
  TORCH_CHECK_EQ(t.size(0), shape0);
  TORCH_CHECK_EQ(t.size(1), shape1);
  TORCH_CHECK_EQ(t.size(2), shape2);
  TORCH_CHECK_EQ(t.dtype(), dtype);
  TORCH_CHECK(t.device().is_cuda());
  TORCH_CHECK_EQ(((int64_t)t.data_ptr()) % 16, 0);  // alignment
}

void concat_mla_k(at::Tensor k, at::Tensor k_nope, at::Tensor k_rope) {
  const int num_tokens = k.size(0);

  check_tensor(k, num_tokens, NUM_LOCAL_HEADS, K_HEAD_DIM, at::kBFloat16);
  check_tensor(k_nope, num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM, at::kBFloat16);
  check_tensor(k_rope, num_tokens, 1, QK_ROPE_HEAD_DIM, at::kBFloat16);
  TORCH_CHECK_EQ(k.stride(2), 1);
  TORCH_CHECK_EQ(k_nope.stride(2), 1);
  TORCH_CHECK_EQ(k_rope.stride(2), 1);

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int grid_size = num_tokens * NUM_LOCAL_HEADS;
  const int block_size = 256;

  concat_mla_k_kernel<<<grid_size, block_size, 0, stream>>>(
      reinterpret_cast<nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<nv_bfloat16*>(k_nope.data_ptr()),
      reinterpret_cast<nv_bfloat16*>(k_rope.data_ptr()),
      num_tokens,
      k.stride(0),
      k.stride(1),
      k_nope.stride(0),
      k_nope.stride(1),
      k_rope.stride(0));
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

// ============================== concat_mla_absorb_q ==============================

// TODO give a name prefix, also maybe refactor code above
constexpr int A_LAST_DIM = 512;
constexpr int B_LAST_DIM = 64;

__global__ void concat_mla_absorb_q_kernel(
    nv_bfloat16* a,
    nv_bfloat16* b,
    nv_bfloat16* out,
    const int num_items,
    const int dim_1,
    const int a_stride_0,
    const int a_stride_1,
    const int b_stride_0,
    const int b_stride_1,
    const int out_stride_0,
    const int out_stride_1) {
  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane_id = get_lane_id();

  if (flat_warp_id >= num_items) {
    return;
  }

  const int idx_0 = flat_warp_id / dim_1;
  const int idx_1 = flat_warp_id % dim_1;

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

inline void check_tensor_concat_mla_absorb_q(const at::Tensor& t, int64_t shape2) {
  TORCH_CHECK_EQ(t.dim(), 3);
  TORCH_CHECK_EQ(t.size(2), shape2);
  TORCH_CHECK_EQ(t.stride(2), 1);
  TORCH_CHECK_EQ(t.dtype(), at::kBFloat16);
  TORCH_CHECK(t.device().is_cuda());
  TORCH_CHECK_EQ(((int64_t)t.data_ptr()) % 16, 0);  // alignment
}

// TODO further optimize it later
void concat_mla_absorb_q(at::Tensor a, at::Tensor b, at::Tensor out) {
  check_tensor_concat_mla_absorb_q(a, A_LAST_DIM);
  check_tensor_concat_mla_absorb_q(b, B_LAST_DIM);
  check_tensor_concat_mla_absorb_q(out, A_LAST_DIM + B_LAST_DIM);

  const auto stream = at::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(a.size(0) * a.size(1), b.size(0) * b.size(1));
  TORCH_CHECK_EQ(a.size(1), b.size(1));
  const int num_items = a.size(0) * a.size(1);

  constexpr int num_warps_per_block = 32;
  const int grid_size = ceil_div(num_items, num_warps_per_block);
  const int block_size = num_warps_per_block * 32;

  concat_mla_absorb_q_kernel<<<grid_size, block_size, 0, stream>>>(
      reinterpret_cast<nv_bfloat16*>(a.data_ptr()),
      reinterpret_cast<nv_bfloat16*>(b.data_ptr()),
      reinterpret_cast<nv_bfloat16*>(out.data_ptr()),
      num_items,
      a.size(1),
      a.stride(0),
      a.stride(1),
      b.stride(0),
      b.stride(1),
      out.stride(0),
      out.stride(1));
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}
