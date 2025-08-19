#include <c10/cuda/CUDAGuard.h>
#include <cudaTypedefs.h>
#include <torch/all.h>

#include <type_traits>

#include "cutlass/cutlass.h"
#include "w4a8_grouped_mm_c3x.cuh"

using namespace cute;

namespace {

enum class Sched { PP, CO };

template <int M, int N, int K, int A, int B, int C, Sched S>
struct SM90W4A8Config {
  using KernelSchedule = std::conditional_t<
      S == Sched::PP,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;

  using EpilogueSchedule = std::conditional_t<
      S == Sched::PP,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>;

  using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
  using ClusterShape = cute::Shape<cute::Int<A>, cute::Int<B>, cute::Int<C>>;
  using Cutlass3xW4A8Gemm = cutlass_3x_w4a8_group_gemm<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>;
};

template <int M, int N, int K, int A, int B, int C>
using SM90_PP = SM90W4A8Config<M, N, K, A, B, C, Sched::PP>;

template <int M, int N, int K, int A, int B, int C>
using SM90_CO = SM90W4A8Config<M, N, K, A, B, C, Sched::CO>;

template <typename Config>
inline void invoke_gemm(
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
    int64_t chunk_size) {
  using GemmT = typename Config::Cutlass3xW4A8Gemm;
  cutlass_w4a8_group_gemm_caller<GemmT>(
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
  uint32_t const m = a_tensors.size(0) / topk;
  uint32_t const n = d_tensors.size(1);
  uint32_t const k = a_tensors.size(1);

  if (n == 4096 && k == 7168) {
    // group gemm 1
    if (m <= 4) {
      invoke_gemm<SM90_PP<64, 32, 512, 2, 1, 1>>(
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
      invoke_gemm<SM90_CO<128, 16, 512, 2, 1, 1>>(
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
      invoke_gemm<SM90_CO<128, 16, 512, 1, 1, 1>>(
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
      invoke_gemm<SM90_CO<128, 32, 512, 2, 1, 1>>(
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
      invoke_gemm<SM90_CO<128, 64, 512, 1, 1, 1>>(
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
      invoke_gemm<SM90_PP<64, 16, 512, 1, 1, 1>>(
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
      invoke_gemm<SM90_CO<128, 32, 512, 1, 1, 1>>(
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
      invoke_gemm<SM90_CO<128, 64, 512, 1, 1, 1>>(
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
  } else if (n == 512 && k == 7168) {
    // group gemm 1 for tp
    if (m <= 4) {
      invoke_gemm<SM90_PP<64, 32, 512, 2, 1, 1>>(
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
      invoke_gemm<SM90_CO<128, 16, 512, 2, 1, 1>>(
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
      invoke_gemm<SM90_CO<128, 16, 512, 2, 1, 1>>(
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
      invoke_gemm<SM90_CO<128, 32, 512, 2, 1, 1>>(
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
      invoke_gemm<SM90_CO<128, 64, 512, 1, 1, 1>>(
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
  } else if (n == 7168 && k == 256) {
    // group gemm 2 for tp
    if (m <= 8) {
      invoke_gemm<SM90_PP<64, 16, 128, 1, 1, 1>>(
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
      invoke_gemm<SM90_PP<128, 32, 128, 2, 1, 1>>(
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
      invoke_gemm<SM90_PP<128, 64, 128, 1, 1, 1>>(
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
    if (k % 512 == 0) {
      invoke_gemm<SM90_CO<128, 32, 512, 1, 1, 1>>(
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
      invoke_gemm<SM90_PP<128, 64, 128, 1, 1, 1>>(
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
