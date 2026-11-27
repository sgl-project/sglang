#pragma once

#include "../sparse_mla_q8kv8_prefill_sm90/config.h"
#include "../sparse_mla_q8kv8_prefill_sm90/helpers.h"
#include <cuda.h>
#include <cuda_fp8.h>

using namespace cute;

#include "../sparse_mla_q8kv8_prefill_sm90/dense_fp8_transpose_v.h"
#include "../sparse_mla_q8kv8_prefill_sm90/dense_fp8_utils.h"

namespace sglang {
namespace q8kv8_sm90 {

constexpr int kHeadDim = 128;
constexpr int kBlockSize = 128;
constexpr int kMmaRows = 64;
constexpr int kTokenTile = 64;
constexpr int kNumStages = 2;
constexpr int kWarpGroupSize = cutlass::NumThreadsPerWarpGroup;

enum NamedBarriers : uint32_t {
  kWarpGroupSync = 0,
};

using fp8_t = cutlass::float_e4m3_t;
using bf16_t = cutlass::bfloat16_t;

struct TmaParams {
  CUtensorMap k;
  CUtensorMap v;
};

using SmemLayoutQ = decltype(coalesce(
    tile_to_shape(GMMA::Layout_K_SW64_Atom<fp8_t>{}, Shape<Int<kMmaRows>, Int<kHeadDim>>{}, Step<_1, _2>{}),
    Shape<_1, _1>{}));
using SmemLayoutK = decltype(coalesce(
    tile_to_shape(GMMA::Layout_K_SW64_Atom<fp8_t>{}, Shape<Int<kTokenTile>, Int<kHeadDim>>{}, Step<_1, _2>{}),
    Shape<_1, _1>{}));
using SmemLayoutVt = decltype(coalesce(
    tile_to_shape(GMMA::Layout_K_SW64_Atom<fp8_t>{}, Shape<Int<kHeadDim>, Int<kTokenTile>>{}, Step<_1, _2>{}),
    Shape<_1, _1>{}));
using TiledMmaQK = decltype(make_tiled_mma(GMMA::MMA_64x64x32_F32E4M3E4M3_SS_TN<>{}, Layout<Shape<_1, _1, _1>>{}));
using TiledMmaPV = decltype(make_tiled_mma(GMMA::MMA_64x128x32_F32E4M3E4M3_RS_TN<>{}, Layout<Shape<_1, _1, _1>>{}));

struct SharedStorage {
  array_aligned<fp8_t, cosize_v<SmemLayoutQ>> q;
  array_aligned<fp8_t, kNumStages * cosize_v<SmemLayoutK>> k;
  array_aligned<fp8_t, kNumStages * cosize_v<SmemLayoutK>> v;
  array_aligned<fp8_t, kNumStages * cosize_v<SmemLayoutVt>> vt;
  bool valid[kNumStages][kTokenTile];
  alignas(8) uint64_t kv_ready[kNumStages];
  int32_t batch;
  int32_t q_position;
  int32_t seq_len;
  int32_t physical_page_start;
  int64_t request;
};

template <typename T>
__device__ __forceinline__ uint32_t to_shared(T* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* barrier) {
  asm volatile("mbarrier.init.shared.b64 [%0], 1;" ::"r"(to_shared(barrier)));
}

__device__ __forceinline__ void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void warpgroup_sync() {
  cutlass::arch::NamedBarrier::arrive_and_wait(kWarpGroupSize, NamedBarriers::kWarpGroupSync);
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* barrier, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(to_shared(barrier)), "r"(bytes));
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* barrier, uint32_t phase) {
  asm volatile(
      "{\n\t.reg .pred complete;\n\t"
      "WAIT_%=: mbarrier.try_wait.parity.shared.b64 complete, [%0], %1;\n\t"
      "@!complete bra WAIT_%=;\n\t}\n" ::"r"(to_shared(barrier)),
      "r"(phase));
}

__device__ __forceinline__ void
tma_load_3d(fp8_t* dst, const CUtensorMap* tensor_map, int32_t x, int32_t y, int32_t z, uint64_t* barrier) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3, %4}], [%5];" ::"r"(to_shared(dst)),
      "l"(tensor_map),
      "r"(x),
      "r"(y),
      "r"(z),
      "r"(to_shared(barrier))
      : "memory");
}

__device__ __forceinline__ void copy_q_16B(fp8_t* dst, const fp8_t* src, bool pred) {
  if (pred && (reinterpret_cast<uintptr_t>(src) & 0xf) == 0) {
    *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
    return;
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    dst[i] = pred ? src[i] : fp8_t(0.0f);
  }
}

__device__ __forceinline__ void load_kv_tile(
    SharedStorage& storage,
    const TmaParams& tma,
    const int32_t* req_to_token,
    int stage,
    int selected_block,
    int tile,
    int kv_head,
    int num_kv_heads,
    int max_slots,
    int req_stride,
    int tid) {
  Tensor sK = make_tensor(make_smem_ptr(storage.k.data() + stage * cosize_v<SmemLayoutK>), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(storage.v.data() + stage * cosize_v<SmemLayoutK>), SmemLayoutK{});

  if (tid < kTokenTile) {
    const int logical_position = selected_block * kBlockSize + tile * kTokenTile + tid;
    storage.valid[stage][tid] =
        selected_block >= 0 && logical_position < storage.seq_len && logical_position <= storage.q_position;
  }
  if (tid == 0) {
    // A sparse block is exactly one physical page. Resolve that page once,
    // then reuse it for the two 64-token TMA stages.
    if (tile == 0) {
      const int logical_block_start = selected_block * kBlockSize;
      int first_slot = 0;
      if (selected_block >= 0) {
        first_slot = req_to_token[storage.request * static_cast<int64_t>(req_stride) + logical_block_start];
        if (first_slot < 0) {
          first_slot += max_slots;
        } else if (first_slot >= max_slots) {
          first_slot -= max_slots;
        }
      }
      // The native provider requires page_size == block_size == 128, so the
      // first logical token of a sparse block is already the physical page
      // base. Do not round here: rounding would hide a broken page contract.
      storage.physical_page_start = first_slot;
    }
    const int physical_slot = storage.physical_page_start + tile * kTokenTile;
    constexpr uint32_t kTransactionBytes = 2 * kTokenTile * kHeadDim * sizeof(fp8_t);
    mbarrier_arrive_expect_tx(&storage.kv_ready[stage], kTransactionBytes);
    tma_load_3d(&sK(0, 0), &tma.k, 0, physical_slot, kv_head * 2, &storage.kv_ready[stage]);
    tma_load_3d(&sK(0, kHeadDim / 2), &tma.k, 0, physical_slot, kv_head * 2 + 1, &storage.kv_ready[stage]);
    tma_load_3d(&sV(0, 0), &tma.v, 0, physical_slot, kv_head * 2, &storage.kv_ready[stage]);
    tma_load_3d(&sV(0, kHeadDim / 2), &tma.v, 0, physical_slot, kv_head * 2 + 1, &storage.kv_ready[stage]);
  }
}

__global__ void fp8_mha_fwd_q8kv8_kernel(
    bf16_t* __restrict__ output,
    const fp8_t* __restrict__ q,
    const fp8_t* __restrict__ k_cache,
    const fp8_t* __restrict__ v_cache,
    const int32_t* __restrict__ req_to_token,
    const int64_t* __restrict__ slot_ids,
    const int32_t* __restrict__ topk_idx,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ prefix_lens,
    __grid_constant__ const TmaParams tma,
    int total_q,
    int num_q_heads,
    int num_kv_heads,
    int max_slots,
    int req_stride,
    int topk,
    int batch_size,
    int64_t q_stride_0,
    int64_t q_stride_1,
    int64_t q_stride_2,
    float effective_sm_scale,
    float v_scale) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
  const int q_idx = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int group_size = num_q_heads / num_kv_heads;

  extern __shared__ char smem[];
  SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem);
  Tensor sQ = make_tensor(make_smem_ptr(storage.q.data()), SmemLayoutQ{});

  if (tid == 0) {
    int batch = 0;
    while (batch + 1 < batch_size && q_idx >= cu_seqlens[batch + 1]) {
      ++batch;
    }
    storage.batch = batch;
    storage.q_position = prefix_lens[batch] + q_idx - cu_seqlens[batch];
    storage.seq_len = seq_lens[batch];
    storage.request = slot_ids[batch];
    mbarrier_init(&storage.kv_ready[0]);
    mbarrier_init(&storage.kv_ready[1]);
    fence_mbarrier_init();
  }

  constexpr int kVectorsPerRow = kHeadDim / 16;
  for (int vector_idx = tid; vector_idx < kMmaRows * kVectorsPerRow; vector_idx += blockDim.x) {
    const int row = vector_idx / kVectorsPerRow;
    const int col = (vector_idx % kVectorsPerRow) * 16;
    fp8_t* dst = &sQ(row, col);
    if (row < group_size) {
      const int q_head = kv_head * group_size + row;
      const int64_t q_offset = static_cast<int64_t>(q_idx) * q_stride_0 + static_cast<int64_t>(q_head) * q_stride_1 +
                               static_cast<int64_t>(col) * q_stride_2;
      copy_q_16B(dst, q + q_offset, true);
    } else {
      copy_q_16B(dst, nullptr, false);
    }
  }
  warpgroup_sync();
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tma.k) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&tma.v) : "memory");
  }

  Tensor rP = partition_fragment_C(TiledMmaQK{}, Shape<Int<kMmaRows>, Int<kTokenTile>>{});
  Tensor rO = partition_fragment_C(TiledMmaPV{}, Shape<Int<kMmaRows>, Int<kHeadDim>>{});
  cute::fill(rO, 0.0f);

  using RP8Layout = decltype(flash::convert_layout_acc_Aregs<TiledMmaPV>(rP.layout()));
  Tensor rP8 = make_tensor<fp8_t>(RP8Layout{});
  float running_max[2] = {-INFINITY, -INFINITY};
  float running_sum[2] = {0.0f, 0.0f};

  SmemTransposeFp8_64x64<kTokenTile, kHeadDim> transpose;
  using Transpose = SmemTransposeFp8_64x64<kTokenTile, kHeadDim>;
  using SrcLayout = typename Transpose::SmemLayoutTransposeV;
  using DstLayout = typename Transpose::SmemLayoutTransposeVt;
  constexpr int kTilesPerBlock = kBlockSize / kTokenTile;
  const int total_tiles = topk * kTilesPerBlock;
  if (total_tiles > 0) {
    const int selected_block = topk_idx[(kv_head * total_q + q_idx) * topk];
    load_kv_tile(storage, tma, req_to_token, 0, selected_block, 0, kv_head, num_kv_heads, max_slots, req_stride, tid);
    mbarrier_wait(&storage.kv_ready[0], 0);
    warpgroup_sync();
  }

  for (int tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
    const int stage = tile_idx % kNumStages;
    Tensor sK = make_tensor(make_smem_ptr(storage.k.data() + stage * cosize_v<SmemLayoutK>), SmemLayoutK{});
    Tensor sVt = make_tensor(make_smem_ptr(storage.vt.data() + stage * cosize_v<SmemLayoutVt>), SmemLayoutVt{});

    const int next_tile_idx = tile_idx + 1;
    if (next_tile_idx < total_tiles) {
      const int next_selected_idx = next_tile_idx / kTilesPerBlock;
      const int next_tile = next_tile_idx % kTilesPerBlock;
      const int next_selected_block = topk_idx[(kv_head * total_q + q_idx) * topk + next_selected_idx];
      load_kv_tile(
          storage,
          tma,
          req_to_token,
          stage ^ 1,
          next_selected_block,
          next_tile,
          kv_head,
          num_kv_heads,
          max_slots,
          req_stride,
          tid);
    }

    Tensor sVSrc = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(storage.v.data() + stage * cosize_v<SmemLayoutK>), SrcLayout{}));
    Tensor sVtDst = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(storage.vt.data() + stage * cosize_v<SmemLayoutVt>), DstLayout{}));
    transpose.transpose_pair(
        flatten(sVSrc(_, 0, 0)), flatten(sVtDst(_, 0, 0)), flatten(sVSrc(_, 0, 1)), flatten(sVtDst(_, 0, 1)));
    fence_view_async_shared();
    warpgroup_sync();

    sm90::gemm_ss(true, TiledMmaQK{}, sQ, sK, rP, tid);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(rP);
    warpgroup_fence_operand(rO);
    warpgroup_fence_operand(rP8);

#pragma unroll
    for (int row_idx = 0; row_idx < 2; ++row_idx) {
#pragma unroll
      for (int i = row_idx * 2; i < size(rP); i += 4) {
        const int col = 8 * (i / 4) + (tid % 4) * 2;
        if (!storage.valid[stage][col]) {
          rP(i) = -INFINITY;
        }
        if (!storage.valid[stage][col + 1]) {
          rP(i + 1) = -INFINITY;
        }
      }
    }

#pragma unroll
    for (int row_idx = 0; row_idx < 2; ++row_idx) {
      float tile_max = -INFINITY;
#pragma unroll
      for (int i = row_idx * 2; i < size(rP); i += 4) {
        tile_max = max(tile_max, max(rP(i), rP(i + 1)));
      }
      tile_max = max(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 1));
      tile_max = max(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 2));
      tile_max *= effective_sm_scale;
      const float new_max = max(running_max[row_idx], tile_max);
      const float old_scale = expf(running_max[row_idx] - new_max);
      float tile_sum = 0.0f;
#pragma unroll
      for (int i = row_idx * 2; i < size(rP); i += 4) {
        rP(i) = expf(rP(i) * effective_sm_scale - new_max);
        rP(i + 1) = expf(rP(i + 1) * effective_sm_scale - new_max);
        tile_sum += rP(i) + rP(i + 1);
      }
#pragma unroll
      for (int i = row_idx * 2; i < size(rO); i += 4) {
        rO(i) *= old_scale;
        rO(i + 1) *= old_scale;
      }
      running_sum[row_idx] = running_sum[row_idx] * old_scale + tile_sum;
      running_max[row_idx] = new_max;
    }

    flash::permute_Cregs_fp8(rP);
    Tensor rPAcc = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(rP.layout()));
    flash::convert_type_out(rPAcc, rP8);

    sm90::gemm_rs(false, TiledMmaPV{}, rP8, sVt, rO, tid);
    warpgroup_commit_batch();

    if (next_tile_idx < total_tiles) {
      const int next_stage = next_tile_idx % kNumStages;
      const int next_phase = (next_tile_idx / kNumStages) & 1;
      mbarrier_wait(&storage.kv_ready[next_stage], next_phase);
      warpgroup_sync();
    }
  }

  warpgroup_wait<0>();
  warpgroup_fence_operand(rO);
  warpgroup_fence_operand(rP8);

  running_sum[0] += __shfl_xor_sync(0xffffffff, running_sum[0], 1);
  running_sum[0] += __shfl_xor_sync(0xffffffff, running_sum[0], 2);
  running_sum[1] += __shfl_xor_sync(0xffffffff, running_sum[1], 1);
  running_sum[1] += __shfl_xor_sync(0xffffffff, running_sum[1], 2);

  warpgroup_fence_operand(rO);
#pragma unroll
  for (int row_idx = 0; row_idx < 2; ++row_idx) {
    const int row = sm90::get_AorC_row_idx(row_idx, tid);
    if (row < group_size) {
      const int q_head = kv_head * group_size + row;
      const float scale = running_sum[row_idx] > 0.0f ? v_scale / running_sum[row_idx] : 0.0f;
#pragma unroll
      for (int i = row_idx * 2; i < size(rO); i += 4) {
        const int col = 8 * (i / 4) + (tid % 4) * 2;
        const int64_t output_offset = (static_cast<int64_t>(q_idx) * num_q_heads + q_head) * kHeadDim + col;
        output[output_offset] = static_cast<bf16_t>(rO(i) * scale);
        output[output_offset + 1] = static_cast<bf16_t>(rO(i + 1) * scale);
      }
    }
  }
#else
  if (cute::thread0()) {
    CUTE_INVALID_CONTROL_PATH("fp8_mha_fwd_q8kv8_kernel requires sm90");
  }
#endif
}

inline void launch_fp8_mha_fwd_q8kv8_sm90(
    bf16_t* output,
    const fp8_t* q,
    const fp8_t* k_cache,
    const fp8_t* v_cache,
    const int32_t* req_to_token,
    const int64_t* slot_ids,
    const int32_t* topk_idx,
    const int32_t* cu_seqlens,
    const int32_t* seq_lens,
    const int32_t* prefix_lens,
    int total_q,
    int num_q_heads,
    int num_kv_heads,
    int max_slots,
    int req_stride,
    int topk,
    int batch_size,
    int64_t q_stride_0,
    int64_t q_stride_1,
    int64_t q_stride_2,
    float effective_sm_scale,
    float v_scale,
    cudaStream_t stream) {
  TmaParams tma;
  uint64_t size[3] = {
      static_cast<uint64_t>(kHeadDim / 2), static_cast<uint64_t>(max_slots), static_cast<uint64_t>(2 * num_kv_heads)};
  uint64_t stride[2] = {
      static_cast<uint64_t>(kHeadDim) * static_cast<uint64_t>(num_kv_heads) * sizeof(fp8_t),
      static_cast<uint64_t>((kHeadDim / 2) * sizeof(fp8_t))};
  // Two adjacent 64x64 SW64 slices form each logical 64x128 K/V tile.
  uint32_t box_size[3] = {kHeadDim / 2, kTokenTile, 1};
  uint32_t elem_stride[3] = {1, 1, 1};
  auto encode_tma = [&](CUtensorMap* tensor_map, const fp8_t* cache) {
    CUresult result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        3,
        const_cast<fp8_t*>(cache),
        size,
        stride,
        box_size,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    KU_ASSERT(result == CUDA_SUCCESS);
  };
  encode_tma(&tma.k, k_cache);
  encode_tma(&tma.v, v_cache);

  auto kernel = &fp8_mha_fwd_q8kv8_kernel;
  constexpr size_t smem_size = sizeof(SharedStorage);
  KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  kernel<<<dim3(total_q, num_kv_heads), 128, smem_size, stream>>>(
      output,
      q,
      k_cache,
      v_cache,
      req_to_token,
      slot_ids,
      topk_idx,
      cu_seqlens,
      seq_lens,
      prefix_lens,
      tma,
      total_q,
      num_q_heads,
      num_kv_heads,
      max_slots,
      req_stride,
      topk,
      batch_size,
      q_stride_0,
      q_stride_1,
      q_stride_2,
      effective_sm_scale,
      v_scale);
}

}  // namespace q8kv8_sm90
}  // namespace sglang
