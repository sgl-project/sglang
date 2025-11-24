#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cassert>
#include <iostream>
#include <string>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "es_fp8_blockwise_functor.cuh"

namespace expert_specialization {

using namespace cute;

template <typename T>
void es_sm90_fp8_blockwise_scaled_group_mm_pre_compute(
    // Output
    torch::Tensor& out_ptrs,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
    torch::Tensor& layout_sfa,
    torch::Tensor& layout_sfb,
    torch::Tensor& lm_problem_sizes,
    torch::Tensor& mm_problem_sizes,
    torch::Tensor& hm_problem_sizes,
    // Input
    torch::Tensor& out_tensors,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& expert_offsets,
    bool is_h20_device,
    cudaStream_t stream) {
  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  // Creat Scale Factor Layout Functor
  using LayoutSFA = typename PerfConfigMiddleMH20::LayoutSFA;
  using LayoutSFB = typename PerfConfigMiddleMH20::LayoutSFB;
  struct Fp8BlockwiseGroupedGemmSFLayoutFunctor<PerfConfigMiddleMH20> sf_layout(
      reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()), reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr()));

  int num_experts = (int)expert_offsets.size(0);
  TORCH_CHECK(num_experts <= 1024, "Expert more than 1024");  // Max threads per block is 1024

  struct Fp8BlockwiseGroupedGemmOffsetFunctor<cutlass::float_e4m3_t, float, T> of(
      static_cast<int*>(expert_offsets.data_ptr()),
      static_cast<cutlass::float_e4m3_t*>(a_tensors.data_ptr()),
      static_cast<cutlass::float_e4m3_t*>(b_tensors.data_ptr()),
      static_cast<T*>(out_tensors.data_ptr()),
      static_cast<float*>(a_scales.data_ptr()),
      static_cast<float*>(b_scales.data_ptr()),
      static_cast<cutlass::float_e4m3_t**>(a_ptrs.data_ptr()),
      static_cast<cutlass::float_e4m3_t**>(b_ptrs.data_ptr()),
      static_cast<float**>(a_scales_ptrs.data_ptr()),
      static_cast<float**>(b_scales_ptrs.data_ptr()),
      static_cast<T**>(out_ptrs.data_ptr()));
  if (!is_h20_device) {
    struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigLowMHx00> lm_psf(
        static_cast<int*>(lm_problem_sizes.data_ptr()));
    struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigMiddleMHx00> mm_psf(
        static_cast<int*>(mm_problem_sizes.data_ptr()));
    struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigHighMHx00> hm_psf(
        static_cast<int*>(hm_problem_sizes.data_ptr()));
    groupedGemmPreComputeKernel<<<1, num_experts, 0, stream>>>(
        static_cast<int*>(problem_sizes.data_ptr()), of, sf_layout, lm_psf, mm_psf, hm_psf);
  } else {
    struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigLowMH20> lm_psf(
        static_cast<int*>(lm_problem_sizes.data_ptr()));
    struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigMiddleMH20> mm_psf(
        static_cast<int*>(mm_problem_sizes.data_ptr()));
    struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigHighMH20> hm_psf(
        static_cast<int*>(hm_problem_sizes.data_ptr()));
    groupedGemmPreComputeKernel<<<1, num_experts, 0, stream>>>(
        static_cast<int*>(problem_sizes.data_ptr()), of, sf_layout, lm_psf, mm_psf, hm_psf);
  }
}

template <typename GemmTraits>
void launch_sm90_fp8_blockwise_scaled_group_mm(
    torch::Tensor& out_ptrs,
    const torch::Tensor& a_ptrs,
    const torch::Tensor& b_ptrs,
    const torch::Tensor& a_scales_ptrs,
    const torch::Tensor& b_scales_ptrs,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_d,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& workspace,
    cudaStream_t stream) {
  using ElementA = typename GemmTraits::ElementA;
  using StrideA = typename GemmTraits::StrideA;
  using ElementB = typename GemmTraits::ElementB;
  using StrideB = typename GemmTraits::StrideB;
  using ElementAccumulator = typename GemmTraits::ElementAccumulator;
  using LayoutSFA = typename GemmTraits::LayoutSFA;
  using LayoutSFB = typename GemmTraits::LayoutSFB;
  using ElementD = typename GemmTraits::ElementD;
  using StrideD = typename GemmTraits::StrideD;
  using UnderlyingProblemShape = typename GemmTraits::ProblemShape::UnderlyingProblemShape;
  using Gemm = typename GemmTraits::Gemm;
  using GemmKernel = typename GemmTraits::GemmKernel;

  int num_experts = (int)problem_sizes.size(0);
  Gemm gemm_op;

  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementA**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(stride_a.data_ptr()),
      static_cast<const ElementB**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(stride_b.data_ptr()),
      static_cast<const ElementAccumulator**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),
      static_cast<const ElementAccumulator**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr())};

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = c10::cuda::current_device();
  hw_info.sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  typename GemmKernel::EpilogueArguments epilogue_args{
      {}, nullptr, nullptr, static_cast<ElementD**>(out_ptrs.data_ptr()), static_cast<StrideD*>(stride_d.data_ptr())};

  UnderlyingProblemShape* problem_sizes_as_shapes = static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info};

  auto can_implement_status = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess, "Failed to implement GEMM");

  auto status = gemm_op.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm_op.run(stream, nullptr, true);  // Enable PDL
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}

template <typename OutType>
void es_sm90_fp8_blockwise_scaled_group_mm_distpatch_out_dtype(
    torch::Tensor& out_ptrs,
    const torch::Tensor& a_ptrs,
    const torch::Tensor& b_ptrs,
    const torch::Tensor& a_scales_ptrs,
    const torch::Tensor& b_scales_ptrs,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_d,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& lm_problem_sizes,
    const torch::Tensor& mm_problem_sizes,
    const torch::Tensor& hm_problem_sizes,
    const torch::Tensor& workspace,
    bool is_h20_device,
    cudaStream_t stream) {
  using LowMGemmH20Traits =
      ExpertSpecializationSm90FP8BlockwiseGroupedGemmTraits<OutType, cutlass::layout::ColumnMajor, PerfConfigLowMH20>;
  using LowMGemmHx00Traits =
      ExpertSpecializationSm90FP8BlockwiseGroupedGemmTraits<OutType, cutlass::layout::ColumnMajor, PerfConfigLowMHx00>;
  using MiddleMGemmH20Traits =
      ExpertSpecializationSm90FP8BlockwiseGroupedGemmTraits<OutType, cutlass::layout::RowMajor, PerfConfigMiddleMH20>;
  using MiddleMGemmHx00Traits = ExpertSpecializationSm90FP8BlockwiseGroupedGemmTraits<
      OutType,
      cutlass::layout::ColumnMajor,
      PerfConfigMiddleMHx00>;
  using HighMGemmH20Traits =
      ExpertSpecializationSm90FP8BlockwiseGroupedGemmTraits<OutType, cutlass::layout::RowMajor, PerfConfigHighMH20>;
  using HighMGemmHx00Traits =
      ExpertSpecializationSm90FP8BlockwiseGroupedGemmTraits<OutType, cutlass::layout::RowMajor, PerfConfigHighMHx00>;

  if (!is_h20_device) {
    launch_sm90_fp8_blockwise_scaled_group_mm<LowMGemmHx00Traits>(
        out_ptrs,
        b_ptrs,
        a_ptrs,
        b_scales_ptrs,
        a_scales_ptrs,
        stride_b,
        stride_a,
        stride_d,
        layout_sfb,
        layout_sfa,
        lm_problem_sizes,
        workspace,
        stream);
  } else {
    launch_sm90_fp8_blockwise_scaled_group_mm<LowMGemmH20Traits>(
        out_ptrs,
        b_ptrs,
        a_ptrs,
        b_scales_ptrs,
        a_scales_ptrs,
        stride_b,
        stride_a,
        stride_d,
        layout_sfb,
        layout_sfa,
        lm_problem_sizes,
        workspace,
        stream);
  }

  if (!is_h20_device) {
    launch_sm90_fp8_blockwise_scaled_group_mm<MiddleMGemmHx00Traits>(
        out_ptrs,
        b_ptrs,
        a_ptrs,
        b_scales_ptrs,
        a_scales_ptrs,
        stride_b,
        stride_a,
        stride_d,
        layout_sfb,
        layout_sfa,
        mm_problem_sizes,
        workspace,
        stream);
  } else {
    launch_sm90_fp8_blockwise_scaled_group_mm<HighMGemmHx00Traits>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_d,
        layout_sfa,
        layout_sfb,
        mm_problem_sizes,
        workspace,
        stream);
  }

  if (!is_h20_device) {
    launch_sm90_fp8_blockwise_scaled_group_mm<HighMGemmHx00Traits>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_d,
        layout_sfa,
        layout_sfb,
        hm_problem_sizes,
        workspace,
        stream);
  } else {
    launch_sm90_fp8_blockwise_scaled_group_mm<HighMGemmH20Traits>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_d,
        layout_sfa,
        layout_sfb,
        hm_problem_sizes,
        workspace,
        stream);
  }
}

}  // namespace expert_specialization
