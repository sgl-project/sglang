// Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
// (fastvideo-kernel/csrc/attention/block_sparse_launch_sm100a.cuh, Apache-2.0). Inference-only
// forward; the sm_103a device pass is admitted alongside sm_100a.

#ifndef BLOCK_SPARSE_VSA_LAUNCH_SM100A_CUH
#define BLOCK_SPARSE_VSA_LAUNCH_SM100A_CUH

// Launch surface for the sm_100a VSA block-sparse FMHA forward.
//
// Everything a caller needs: a POD argument struct, a predicate saying whether this build can
// run those arguments, and one launch entry point. The benchmark in
// block_sparse_bench_sm100a.cu and the torch binding both go through here, so there is
// one tensormap construction and one launch configuration rather than two that can drift.
//
// Two compile-time knobs select the four builds:
//   VSA_BLK128  false -> 64-token sparse blocks, true -> 128-token
//   VSA_BHSD    false -> [token][head][dim] (BSHD), true -> [batch][head][token][dim] (BHSD)

#include "block_sparse_kernel_sm100a.cuh"

namespace VSA_NAMESPACE {

struct BlockSparseVsaArgs {
  const __nv_bfloat16* q;
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;      // natural layout; only blk128 reads it (blk64 still needs v_t)
  const __nv_bfloat16* v_t;    // unused: kept so the bench's V_T buffer still binds
  __nv_bfloat16* o;
  float* lse;                  // [batch, num_heads, seqlen] fp32, or nullptr

  const int* q2k_idx;              // [batch*num_heads*num_blocks, max_kv] int32
  const int* q2k_num;              // [batch*num_heads*num_blocks] int32
  const int* variable_block_sizes; // [num_blocks] int32, valid tokens per block

  int batch;
  int num_heads;
  int seqlen;
  int head_dim;
  int num_blocks;
  int max_kv;
  float sm_scale;
};

// cudaSuccess iff this build can run `a`. Deliberately conservative: the caller is expected
// to fall back to its own implementation rather than get a wrong answer.
__host__ inline cudaError_t block_sparse_supported(const BlockSparseVsaArgs& a) {
  if (a.head_dim != HEAD_DIM) return cudaErrorInvalidValue;      // compile-time in the kernel
  if (a.num_blocks % 2 != 0) return cudaErrorInvalidValue;       // a CTA owns an adjacent pair
  if (a.seqlen != a.num_blocks * BLOCK) return cudaErrorInvalidValue;
  if (a.max_kv < 1 || a.num_blocks < 1) return cudaErrorInvalidValue;
  if (a.q == nullptr || a.k == nullptr || a.o == nullptr) return cudaErrorInvalidValue;
  if (a.q2k_idx == nullptr || a.q2k_num == nullptr) return cudaErrorInvalidValue;
  // FastVideo always supplies this; without it padded keys would be attended as real zeros.
  if (a.variable_block_sizes == nullptr) return cudaErrorInvalidValue;
  // V is read MN-major at BOTH block sizes now, so no pre-transposed V_T is ever needed.
  if (a.v == nullptr) return cudaErrorInvalidValue;
  return cudaSuccess;
}

__host__ inline cudaError_t launch_block_sparse_sm100a(const BlockSparseVsaArgs& a,
                                                           cudaStream_t stream) {
  const cudaError_t sup = block_sparse_supported(a);
  if (sup != cudaSuccess) return sup;

  const int B = a.batch, H = a.num_heads, S = a.seqlen, hd = a.head_dim;
  const int num_blocks = a.num_blocks, max_kv = a.max_kv;
  const long tq = (long)B * S;
  const int packed_mtiles_per_seq = num_blocks / 2;
  const int total_work = B * H * packed_mtiles_per_seq;
  constexpr bool BHSD = VSA_BHSD;

  CUtensorMap tq_, tk_, tvt_, tv_, to_;
  {
    uint64_t gd[4] = { (uint64_t)SUB_COLS_BF16, BHSD ? (uint64_t)((long)B * H) : (uint64_t)H,
                       BHSD ? (uint64_t)S : (uint64_t)tq, (uint64_t)Q_SUBTILES };
    uint64_t gs[3] = { BHSD ? (uint64_t)((long)S * hd) * 2u : (uint64_t)hd * 2u,
                       BHSD ? (uint64_t)hd * 2u : (uint64_t)((long)H * hd) * 2u,
                       (uint64_t)SUB_COLS_BF16 * 2u };
    uint32_t bd[4] = { (uint32_t)SUB_COLS_BF16, 1u, (uint32_t)M_TILE, (uint32_t)Q_SUBTILES };
    uint32_t es[4] = { 1u, 1u, 1u, 1u };
    if (cuTensorMapEncodeTiled(&tq_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
                               const_cast<__nv_bfloat16*>(a.q), gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
    if (cuTensorMapEncodeTiled(&to_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, a.o, gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
  }
  {
    uint64_t gd[4] = { (uint64_t)SUB_COLS_BF16,
                       BHSD ? (uint64_t)S : (uint64_t)tq,
                       BHSD ? (uint64_t)(hd / SUB_COLS_BF16)
                            : (uint64_t)((long)H * hd / SUB_COLS_BF16),
                       (uint64_t)((long)B * H) };
    uint64_t gs[3] = { BHSD ? (uint64_t)hd * 2u : (uint64_t)((long)H * hd) * 2u,
                       (uint64_t)SUB_COLS_BF16 * 2u,
                       (uint64_t)((long)S * hd) * 2u };
    uint32_t bd[4] = { (uint32_t)SUB_COLS_BF16, (uint32_t)BLOCK,
                       BLK128 ? (uint32_t)K_SUBTILES : 1u, 1u };
    uint32_t es[4] = { 1u, 1u, 1u, 1u };
    if (cuTensorMapEncodeTiled(&tk_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, BHSD ? 4 : 3,
                               const_cast<__nv_bfloat16*>(a.k), gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
    // V map is byte-for-byte the K map over a.v: MN-major V needs no transpose (blk128).
    const __nv_bfloat16* vbase = a.v ? a.v : a.k;
    if (cuTensorMapEncodeTiled(&tv_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, BHSD ? 4 : 3,
                               const_cast<__nv_bfloat16*>(vbase), gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
  }
  // V_T map: blk64 only. Unused at blk128 but must still be a valid tensormap to pass by value.
  {
    const __nv_bfloat16* vt = a.v_t ? a.v_t : a.k;
    if constexpr (BLK128) {
      uint64_t gd[3] = { (uint64_t)SUB_COLS_BF16, (uint64_t)((long)H * hd),
                         (uint64_t)((long)tq / SUB_COLS_BF16) };
      uint64_t gs[2] = { (uint64_t)tq * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
      uint32_t bd[3] = { (uint32_t)SUB_COLS_BF16, (uint32_t)hd, (uint32_t)V_SUBTILES };
      uint32_t es[3] = { 1u, 1u, 1u };
      if (cuTensorMapEncodeTiled(&tvt_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
                                 const_cast<__nv_bfloat16*>(vt), gd, gs, bd, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
        return cudaErrorInvalidValue;
    } else {
      if (make_tma_2d_tiled(&tvt_, const_cast<__nv_bfloat16*>(vt), (long)H * hd, (int)tq, hd,
                            SUB_COLS_BF16, 2, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                            CU_TENSOR_MAP_SWIZZLE_128B) != cudaSuccess)
        return cudaErrorInvalidValue;
    }
  }

  const size_t smem =
        (size_t)2 * Q_TILE_BYTES + NUM_KV_STAGES * KV_RING_SLOT_BYTES
      + (size_t)2 * M_TILE * HEAD_DIM * sizeof(__nv_bfloat16)
      + (2 * NUM_KV_STAGES + 22) * 8
      + (size_t)CLC_STAGES * (2 * 8 + 16) + 16
      + 8
      + (size_t)2 * STAT_REGIONS * STATS * sizeof(float)
      + 256;

#ifndef VSA_NAMED_BAR
#define VSA_NAMED_BAR false
#endif
#ifndef VSA_THROTTLE
#define VSA_THROTTLE false
#endif
#ifndef VSA_USE_CLC
#define VSA_USE_CLC true
#endif
  constexpr bool FULL_NAMED_BAR = VSA_NAMED_BAR, EX2_EMU = true, SPLIT_P = true,
                 SOFTMAX_THROTTLE = VSA_THROTTLE, USE_CLC = VSA_USE_CLC,
                 Q_RASTER = true, MHA = true;
  auto kfn = &fmha_context_bf16_gen_kernel<32, FULL_NAMED_BAR, EX2_EMU, SPLIT_P,
                                           SOFTMAX_THROTTLE, USE_CLC, Q_RASTER, MHA,
                                           /*RESCALE_THRESHOLD=*/8, /*BHSD=*/VSA_BHSD>;
  cudaError_t e = cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  if (e != cudaSuccess) return e;

  const unsigned long long magic0 = make_magic((unsigned)(H * packed_mtiles_per_seq));
  const unsigned long long magic1 = make_magic((unsigned)H);
  const unsigned long long magic2 = make_magic((unsigned)packed_mtiles_per_seq);
  const float scale_log2 = a.sm_scale * (float)M_LOG2E;

  int numSM = 0;
  e = cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);
  if (e != cudaSuccess) return e;
  const int num_ctas = USE_CLC ? total_work : (total_work < numSM ? total_work : numSM);
  dim3 grid(num_ctas, 1, 1), block(N_WARPS * 32, 1, 1);

  if (USE_CLC) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = stream;
    cudaLaunchAttribute cfgAttr[1];
    cfgAttr[0].id = cudaLaunchAttributeClusterDimension;
    cfgAttr[0].val.clusterDim.x = 1; cfgAttr[0].val.clusterDim.y = 1;
    cfgAttr[0].val.clusterDim.z = 1;
    cfg.attrs = cfgAttr; cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kfn, tq_, tk_, tvt_, tv_, to_, S, H, scale_log2, B,
                              num_blocks, packed_mtiles_per_seq, max_kv, magic0, magic1, magic2,
                              a.q2k_idx, a.q2k_num, a.variable_block_sizes, a.lse);
  }
  kfn<<<grid, block, smem, stream>>>(tq_, tk_, tvt_, tv_, to_, S, H, scale_log2, B, num_blocks,
                                     packed_mtiles_per_seq, max_kv, magic0, magic1, magic2,
                                     a.q2k_idx, a.q2k_num, a.variable_block_sizes, a.lse);
  return cudaGetLastError();
}

}  // namespace VSA_NAMESPACE

// Callers (the bench, the torch binding) keep using unqualified names; each translation unit
// only ever sees the one configuration its VSA_BLK128 selected.
using namespace VSA_NAMESPACE;

#endif  // BLOCK_SPARSE_VSA_LAUNCH_SM100A_CUH
