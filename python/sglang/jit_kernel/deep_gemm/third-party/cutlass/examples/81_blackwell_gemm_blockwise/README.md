# Blockwise and Groupwise GEMM and Grouped GEMM on Blackwell

Blockwise and Groupwise GEMM and Grouped GEMM implement software scaling by the accumulator type.
The examples in this directory aim to demonstrate how we can instantiate this kernel and run it.
The profiler enables instantiating and profiling different kernel configurations for Blockwise and Groupwise GEMM
to determine the best performing kernel for your workload.

## Introduction
Blockwise and Groupwise GEMM operations enable fine-grained numerical precision control by applying scale factors at configurable granularities. This is particularly useful for quantized neural networks where different regions of tensors may have different scaling requirements.

For a GEMM $D = \alpha A B + \beta C$, we introduce two scale factor tensors, SFA
and SFB. This leads to a GEMM $D = \alpha \text{SFA} * A \text{ SFB} * B + \beta C$.

## Scale Factor Tensors
- *SFA*: Broadcast the same scale within a block defined by _scale granularity m_ and _scale granularity k_ when scaling A.
  - Scale granularity m and scale granularity k are also referred to as _scale vector m_ and _k_ respectively.
- *SFB*: Broadcast the same scale within a block defined by _scale granularity n_ and _scale granularity k_ when scaling B.
  - Scale granularity n and scale granularity k are also referred to as _scale vector n_ and _k_ respectively.

These can be represented in CuTe as:
- *SFA Layout*: $((\text{scale granularity M}, M / \text{scale granularity M}), (\text{scale granularity K}, K / \text{scale granularity K})) : ((0, int), (0, int))$
- *SFB Layout*: $((\text{scale granularity N}, M / \text{scale granularity M}), (\text{scale granularity K}, K / \text{scale granularity K})) : ((0, int), (0, int))$

The 0 element stride ensures the same group of coordinates to map to the same element in the scale factors.

## Configuration

For convenience the Blockwise and Groupwise implementation provide 
`cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>`
to deduce layouts and manage compact tensors. 

`cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>` by default makes
every tensor major the M/N mode, but can be configured. For example:
`cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK, UMMA::Major::K, UMMA::Major::MN>`
denotes SFA will be major in the K dimension but SFB will be major in the N dimension.

## Integration with Other Frameworks

If translating from frameworks like Torch where SFA has shape 
(M / ScaleGranularityM, K / ScaleGranularityK) and SFB has a shape (K / ScaleGranularityK, N / ScaleGranularityN),
ensure to transpose SFB and B to fit into the canonical CuTe layout form. This ensures K is always the second mode.
Use strides can be used to determine if each tensor is MN or K major to correctly form the layouts either directly
or with the convenience wrappers.


## Kernel Selection and Profiling 

To determine the most performance Blockwise/Groupwise GEMM or Grouped GEMM kernel for your use case, you can utilize the
[CUTLASS profiler](../../media/docs/cpp/profiler.md).

All Blockwise/Groupwise GEMMs and Group GEMMs with `f32` scaling of `e4m3` or runtime `f8` types can be selected by 
selecting a subset of kernels when configuring with CMake by passing:
`-DCUTLASS_LIBRARY_KERNELS="cutlass3x*f32xe4m3_*f32xe4m3*,cutlass3x*f32xf8_*f32xf8*"`.

The simplest way to use the profiler is to pass `m`, `n`, and `k` as well as your `scale_vec_size_m`, 
`scale_vec_size_n`, and `scale_vec_size_k`. Passing `enable-best-kernel-for-fixed-shape` will do some autotuning
per kernel to determine best rasterization orders, swizzles, and cluster sizes. Passing `blockwiseGemm`
or `GroupedGemm` through the operation flag will determine which set of operations will be profiled.

For examle, this command using the cutlass profiler will dump the performance of all compiled kernels which support scale
granularity m = 1, scale granularity n = 128, and scale granularity k = 128 for the problem size 8192x8192x8192:
```
cutlass_profiler --operation=blockwiseGemm \
                 --enable-best-kernel-for-fixed-shape \
                 --m=8192 --n=8192 --k=8192 \
                 --scale_vec_size_m=1 --scale_vec_size_n=128 --scale_vec_size_k=128 \
                 --verification-enabled=false
```

### Kernel Naming Convention

The naming of the blockwise and groupwise kernels includes the following new pattern: for each tensor scalar pair we have
`<scale_granularity_m or scale_granularity_n>x<scale_granularity_k><accumulator type>x<scaled tensor type>`. For example
`cutlass3x_sm100_tensorop_gemm_64x128f32xe4m3_1x128f32xe4m3_f32_f16_f16_64x128x128_1x1x1_0_nnn_align16_1sm` would denote:
- A CUTLASS 3 GEMM for SM100 that uses tensor cores.
- SFA is f32 with a 64 element scale granularity m and a 128 element scale granularity k.
- The A matrix is e4m3.
- SFB is f32 with a 1 element scale granularity n and a 128 element scale granularity k.
- The B matrix is e4m3. 
- The epilogue is done in f32. 
- The C matrix is f16.
- The D matrix is f16.
- The MMA tile shape is 64x128x128. 
- The cluster shape is 1x1x1. 
- A, B, C, and D are all column major. 
- The alignment of the major modes are 16 elements for A, B, C, and D. 
- The MMA variant is a 1SM instruction.

It is also worthwhile to note that C can be void if scaling by beta is not needed.

## Performance Tips and Tricks

- *MMA Dimensions*: in both Blackwell and Hopper tensor cores it is worthwhile to note that the smallest `MMA_M` dimension is 64, but `MMA_N`
dimension can be as small as 8 for some instructions. For problem sizes where M is small consider computing $D^T = \alpha B^T A^T + \beta C^T$ instead.
  - When computing after swapping A and B and transposing the N dimension is now our small dimension. With a small `MMA_N` we can more effectively tile without performing unecessary computation.
- *Layout Swapping*: When optimizing with the profiler swap `m` and `n` inputs and adjust layouts to reflect this swapping and transposing.
  - For example if we have a row-major A, column-major B, and row-major D, we can swap tensors and run a kernel with:
    - The left hand matrix as row-major (since B transposed is row-major)
    - A right hand matrix as column-major (since A transposed is column-major)
    - A column-major output (since D transposed is column-major).

When using blockwise and groupwise GEMM we must swap the scale vector sizes when doing this optimization. If we have a 1 element scale granularity M
and a 128 element scale granularity N, we must run a kernel with a 128 element scale granularity M and a 1 element scale granularity
N.
