#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <cuda_runtime.h>

#include "pytorch_extension_utils.h"

constexpr int NUM_LOCAL_HEADS = 128;
constexpr int QK_NOPE_HEAD_DIM = 128;
constexpr int QK_ROPE_HEAD_DIM = 64;
constexpr int K_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;

constexpr int HEAD_CHUNK_SIZE = 16;
constexpr int NUM_HEAD_CHUNKS = NUM_LOCAL_HEADS / HEAD_CHUNK_SIZE;

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
  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int token_id = flat_warp_id / NUM_HEAD_CHUNKS;
  const int head_chunk_id = flat_warp_id % NUM_HEAD_CHUNKS;
  const int lane_id = get_lane_id();

  if (token_id >= num_tokens) {
    return;
  }

  using KNopeBufType = int2;
  static_assert(sizeof(KNopeBufType) == QK_NOPE_HEAD_DIM * sizeof(k[0]) / 32);
  KNopeBufType k_nope_buf[HEAD_CHUNK_SIZE];

  using KRopeBufType = int;
  static_assert(sizeof(KRopeBufType) == QK_ROPE_HEAD_DIM * sizeof(k[0]) / 32);
  KRopeBufType k_rope_buf;

  {
    const int* base_addr = reinterpret_cast<int*>(k_rope + token_id * k_rope_stride_0);
    k_rope_buf = *(base_addr + lane_id);
  }

#pragma unroll
  for (int i = 0; i < HEAD_CHUNK_SIZE; ++i) {
    const int head_id = head_chunk_id * HEAD_CHUNK_SIZE + i;
    const int2* base_addr = reinterpret_cast<int2*>(k_nope + token_id * k_nope_stride_0 + head_id * k_nope_stride_1);
    k_nope_buf[i] = *(base_addr + lane_id);
  }

#pragma unroll
  for (int i = 0; i < HEAD_CHUNK_SIZE; ++i) {
    const int head_id = head_chunk_id * HEAD_CHUNK_SIZE + i;

    {
      int2* base_addr = reinterpret_cast<int2*>(k + token_id * k_stride_0 + head_id * k_stride_1);
      *(base_addr + lane_id) = k_nope_buf[i];
    }
    {
      int* base_addr = reinterpret_cast<int*>(k + token_id * k_stride_0 + head_id * k_stride_1 + QK_NOPE_HEAD_DIM);
      *(base_addr + lane_id) = k_rope_buf;
    }
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

  constexpr int num_warps_per_block = 32;
  const int grid_size = ceil_div(num_tokens * NUM_HEAD_CHUNKS, num_warps_per_block);
  const int block_size = num_warps_per_block * 32;

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
