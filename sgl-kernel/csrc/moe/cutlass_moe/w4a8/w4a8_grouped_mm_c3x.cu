#include <c10/cuda/CUDAGuard.h>
#include <cudaTypedefs.h>
#include <torch/all.h>

#include "cutlass/cutlass.h"
#include "w4a8_grouped_mm_c3x.cuh"

using namespace cute;

namespace {

#define JOIN_STRUCT_NAME(m, n, k, a, b, c) sm90_fp8_config##_##m##_##n##_##k##_##a##_##b##_##c

#define JOIN_STRUCT_NAME_CO(m, n, k, a, b, c) sm90_fp8_co_config##_##m##_##n##_##k##_##a##_##b##_##c

#define GENERATE_SM90_W4A8_PP_CONFIG(M, N, K, A, B, C)                                                               \
  struct JOIN_STRUCT_NAME(M, N, K, A, B, C) {                                                                        \
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;                                  \
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;                                  \
    using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;                                         \
    using ClusterShape = cute::Shape<cute::Int<A>, cute::Int<B>, cute::Int<C>>;                                      \
                                                                                                                     \
    using Cutlass3xW4A8Gemm = cutlass_3x_w4a8_group_gemm<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>; \
  };

#define GENERATE_SM90_W4A8_CO_CONFIG(M, N, K, A, B, C)                                                               \
  struct JOIN_STRUCT_NAME_CO(M, N, K, A, B, C) {                                                                     \
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;                               \
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;                               \
    using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;                                         \
    using ClusterShape = cute::Shape<cute::Int<A>, cute::Int<B>, cute::Int<C>>;                                      \
                                                                                                                     \
    using Cutlass3xW4A8Gemm = cutlass_3x_w4a8_group_gemm<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>; \
  };

GENERATE_SM90_W4A8_PP_CONFIG(64, 16, 512, 1, 1, 1)
GENERATE_SM90_W4A8_PP_CONFIG(64, 32, 512, 2, 1, 1)

GENERATE_SM90_W4A8_CO_CONFIG(128, 16, 512, 1, 1, 1)
GENERATE_SM90_W4A8_CO_CONFIG(128, 16, 512, 2, 1, 1)
GENERATE_SM90_W4A8_CO_CONFIG(128, 32, 512, 1, 1, 1)
GENERATE_SM90_W4A8_CO_CONFIG(128, 32, 512, 2, 1, 1)
GENERATE_SM90_W4A8_CO_CONFIG(128, 64, 512, 1, 1, 1)

void dispatch_w4a8_moe_mm_sm90(
    torch::Tensor& d_tensors,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& a_strides,
    torch::Tensor const& b_strides,
    torch::Tensor const& d_strides,
    torch::Tensor const& s_strides,
    int64_t chunk_size,
    int64_t topk) {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

  uint32_t const m = a_tensors.size(0) / topk;
  uint32_t const n = d_tensors.size(1);
  uint32_t const k = a_tensors.size(1);

  if (n == 4096 && k == 7168) {
    // group gemm 1
    if (m <= 4) {
      using Cutlass3xW4A8GemmSelected = typename JOIN_STRUCT_NAME(64, 32, 512, 2, 1, 1)::Cutlass3xW4A8Gemm;
      cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmSelected>(
          d_tensors,
          a_tensors,
          b_tensors,
          a_scales,
          b_scales,
          expert_offsets,
          problem_sizes,
          a_strides,
          b_strides,
          d_strides,
          s_strides,
          chunk_size);
    } else if (m <= 16) {
      using Cutlass3xW4A8GemmSelected = typename JOIN_STRUCT_NAME_CO(128, 16, 512, 2, 1, 1)::Cutlass3xW4A8Gemm;
      cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmSelected>(
          d_tensors,
          a_tensors,
          b_tensors,
          a_scales,
          b_scales,
          expert_offsets,
          problem_sizes,
          a_strides,
          b_strides,
          d_strides,
          s_strides,
          chunk_size);
    } else if (m <= 256) {
      using Cutlass3xW4A8GemmSelected = typename JOIN_STRUCT_NAME_CO(128, 16, 512, 1, 1, 1)::Cutlass3xW4A8Gemm;
      cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmSelected>(
          d_tensors,
          a_tensors,
          b_tensors,
          a_scales,
          b_scales,
          expert_offsets,
          problem_sizes,
          a_strides,
          b_strides,
          d_strides,
          s_strides,
          chunk_size);
    } else if (m <= 1024) {
      using Cutlass3xW4A8GemmSelected = typename JOIN_STRUCT_NAME_CO(128, 32, 512, 2, 1, 1)::Cutlass3xW4A8Gemm;
      cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmSelected>(
          d_tensors,
          a_tensors,
          b_tensors,
          a_scales,
          b_scales,
          expert_offsets,
          problem_sizes,
          a_strides,
          b_strides,
          d_strides,
          s_strides,
          chunk_size);
    } else {
      using Cutlass3xW4A8GemmSelected = typename JOIN_STRUCT_NAME_CO(128, 64, 512, 1, 1, 1)::Cutlass3xW4A8Gemm;
      cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmSelected>(
          d_tensors,
          a_tensors,
          b_tensors,
          a_scales,
          b_scales,
          expert_offsets,
          problem_sizes,
          a_strides,
          b_strides,
          d_strides,
          s_strides,
          chunk_size);
    }
  } else if (n == 7168 && k == 2048) {
    // group gemm 2
    if (m <= 8) {
      using Cutlass3xW4A8GemmSelected = typename JOIN_STRUCT_NAME(64, 16, 512, 1, 1, 1)::Cutlass3xW4A8Gemm;
      cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmSelected>(
          d_tensors,
          a_tensors,
          b_tensors,
          a_scales,
          b_scales,
          expert_offsets,
          problem_sizes,
          a_strides,
          b_strides,
          d_strides,
          s_strides,
          chunk_size);
    } else if (m <= 512) {
      using Cutlass3xW4A8GemmSelected = typename JOIN_STRUCT_NAME_CO(128, 32, 512, 1, 1, 1)::Cutlass3xW4A8Gemm;
      cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmSelected>(
          d_tensors,
          a_tensors,
          b_tensors,
          a_scales,
          b_scales,
          expert_offsets,
          problem_sizes,
          a_strides,
          b_strides,
          d_strides,
          s_strides,
          chunk_size);
    } else {
      using Cutlass3xW4A8GemmSelected = typename JOIN_STRUCT_NAME_CO(128, 64, 512, 1, 1, 1)::Cutlass3xW4A8Gemm;
      cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmSelected>(
          d_tensors,
          a_tensors,
          b_tensors,
          a_scales,
          b_scales,
          expert_offsets,
          problem_sizes,
          a_strides,
          b_strides,
          d_strides,
          s_strides,
          chunk_size);
    }
  } else {
    using Cutlass3xW4A8GemmSelected = typename JOIN_STRUCT_NAME_CO(128, 32, 512, 1, 1, 1)::Cutlass3xW4A8Gemm;
    cutlass_w4a8_group_gemm_caller<Cutlass3xW4A8GemmSelected>(
        d_tensors,
        a_tensors,
        b_tensors,
        a_scales,
        b_scales,
        expert_offsets,
        problem_sizes,
        a_strides,
        b_strides,
        d_strides,
        s_strides,
        chunk_size);
  }
}

}  // namespace

void cutlass_w4a8_moe_mm_sm90(
    torch::Tensor& d_tensors,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& a_strides,
    torch::Tensor const& b_strides,
    torch::Tensor const& d_strides,
    torch::Tensor const& s_strides,
    int64_t chunk_size,
    int64_t topk) {
  dispatch_w4a8_moe_mm_sm90(
      d_tensors,
      a_tensors,
      b_tensors,
      a_scales,
      b_scales,
      expert_offsets,
      problem_sizes,
      a_strides,
      b_strides,
      d_strides,
      s_strides,
      chunk_size,
      topk);
}
