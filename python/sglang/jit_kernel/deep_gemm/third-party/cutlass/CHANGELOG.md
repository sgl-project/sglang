# Changelog

# CUTLASS 4.x

## [4.2.1](https://github.com/NVIDIA/cutlass/releases/tag/v4.2.1) (2025-09-22)

### CuTe DSL
* Bug fixings and improvements
    - Fixed an issue when running DSL codes with cuda-python 13.0
    - Fixed an issue when running inductor with DSL codes
    - Fixed an issue with unexpected logging when running DSL codes in FlashInfer
    - Fixed the issue reported in https://github.com/NVIDIA/cutlass/issues/2647
    - Fixed an issue when conditional define of variables outside of dynamic control flow

### CUTLASS C++
* Bypass EVT for nosmem blockwise kernels on Blackwell.
* Rename cutlass/python/cutlass directory to cutlass/python/cutlass_cppgen.

## [4.2.0](https://github.com/NVIDIA/cutlass/releases/tag/v4.2.0) (2025-09-15)

### CuTe DSL
* More Python versions are now supported for both x86-64 and aarch64, including
    - Python 3.10, 3.11, 3.12, and 3.13
* Added new example and updated notebook to get started with CuTe DSL
    - [Call kernels with dlpack bypassed](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/ampere/call_bypass_dlpack.py)
    - Updates on [TensorSSA demonstration](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks/tensorssa.ipynb)
      + Added a section for introducing the broadcast
* API updates
    - Please refer to [DSL API changelog](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_api/changelog.html) for details
* Bug fixings and improvements
    - Fixed ``cute.print_tensor`` for coordinate tensor
    - Fixed `cute.print` for tuple of layouts
    - Fixed frozen object is not properly updated after fully assigned in dynamic control flow
    - Fixed assign tuple/list element in a dynamic control flow may cause compilation failure
    - Improved error message when CUDA context is not initialized
    - Improved docstring of congruent and weakly_congruent

### CUTLASS C++
* Support for Blackwell SM103 kernels for B300 GPUs.
    - Collective mainloop codes: [Blockscaled datatypes with support for dense GEMM mainloop](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm103_blockscaled_mma_warpspecialized.hpp)
    - New [GEMM](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/dispatch_policy.hpp) and [epilogue](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/dispatch_policy.hpp) dispatch policies for collectives, kernel layers, and builders.
    - Kernel codes: [Blockscaled datatypes with support for dense GEMM kernel](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm103_blockscaled_gemm_tma_warpspecialized.hpp).
* Set of examples that demonstrate the usage of the 3.x API for targeting Blackwell SM103 architecture:
    - [Blockscaled ultra fp4 dense GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/89_sm103_fp4_ultra_gemm/).
    - [Blockscaled ultra fp4 dense grouped GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/90_sm103_fp4_ultra_grouped_gemm).
* Set of unit tests that demonstrate the usage of Blackwell SM103 blockscaled GEMM
    - Unit test files with prefix name of `sm103_` under [GEMM device unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/).
* Support for Blackwell SM121 kernels for DGX Spark GPUs.
    - Share the major codes with Blackwell SM120 kernels.
* Add support for heuristics-based kernel filtering and autotuning using `nvidia-matmul-heuristics` to find the best kernels for a given scenario.
    - Details please refer to [heuristics doc](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cpp/heuristics.md).
* Further enhance Blackwell SM100 Attention kernels in [example 77](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha/).
    - Add fused reduction kernel support for cutlass MLA.
    - Add softmax skip correction.
    - Support for GQA in FMHA backward kernel.
    - Fix an issue where `get_unmasked_trip_count` may return a negative value.
    - Fix an issue where mbarriers are initialized with a zero arrival count.
    - Fix a corner case issue where the sequence length of q is not a multiple of tile_q.
    - Remove tma padding for forward kernel inputs.
* Add Blackwell SM100 kernels for MoEs (focusing on Low-Latency inference performance): [example 92](https://github.com/NVIDIA/cutlass/tree/main/examples/92_blackwell_moe_gemm/).  It uses TMA (for weights) and CPASYNC (for tokens) to load input matrices and allow only one problem dimension to vary across groups/experts, unlike general Grouped GEMMs.  Note: further API simplifications and kernel improvements are upcoming. Any feedback on API is welcome.
* Further enhance blockwise and groupwise GEMMs on Hopper and Blackwell
    - On Blackwell SM120, a blockwise gemm kernel is added: [example 87](https://github.com/NVIDIA/cutlass/tree/main/examples/87_blackwell_geforce_gemm_blockwise/).
    - On Hopper, add K major scale factor support for SM90 blockwise kernels.
    - On Hopper, relax the restriction that the k dimension of the problem size has to be the multiple of the k dimension of the tile size.
    - On Hopper, grouped version supports the case when k = 0.
* Support for Blackwell SM100 fp4 gemv kernels.
    - Kernel codes: [Gemv kernel](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/gemv_blockscaled.h).
    - Example codes: [example 91](https://github.com/NVIDIA/cutlass/tree/main/examples/91_fp4_gemv/)
* Support for Blackwell SM100 legacy mixed input GEMM kernels.
    - Collective mainloop codes: [Mixed input mainloop](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm100_mma_warpspecialized_mixed_input.hpp).
    - Kernel codes: [Mixed input kernel](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized_mixed_input_transform.hpp).
    - Example codes: [example 86](https://github.com/NVIDIA/cutlass/tree/main/examples/86_blackwell_mixed_dtype_gemm/).
* Support for Blackwell SM100 cpasync kernel.
    - Collective mainloop codes: [cpasync mainloop](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm100_mma_cpasync_warpspecialized.hpp).
    - Kernel codes: [cpasync kernel](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm100_gemm_cpasync_warpspecialized.hpp).
* Support Blackwell SM120 mixed input blockscaled grouped GEMM.
* Instantiating more Blackwell kernels in profiler.
    - Blackwell SM100 and SM103 kernels support `CUTLASS_LIBRARY_INSTANTIATION_LEVEL` to instantiate all possible combinations.
    - To use this feature, `CUTLASS_LIBRARY_KERNELS` must be non-empty. Profiler will combine `CUTLASS_LIBRARY_KERNELS` and `CUTLASS_LIBRARY_INSTANTIATION_LEVEL` to instantiate specific kernels.
    - Details please check [Profiler Doc](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cpp/profiler.md).
* Fix some profiler issues:
    - Modify default cluster callback values to none 0 to avoid profiler failure when these values are not set in command line.
    - Fix some no output and timeout issues.
    - Fix Pingpong Blockwise Hopper library generation.
* From CUDA 13.0, the Blackwell SM101 for Thor GPUs is renamed to SM110.
    - For CUDA toolkit version < 13.0, SM101 is still used for Thor GPUs.
    - For CUDA toolkit version >= 13.0, SM110 is used for Thor GPUs and SM101 is no longer valid.
* Rename legacy Python API package from `cutlass` to `cutlass_cppgen` and add Blackwell EVT support to legacy Python interface.
    - Restructuring the C++ Blackwell SM100 Collective Epilogue Builder to work with the Python interface's `EpilogueDescriptors`.
    - Added Blackwell SM100 EVT Emitter on the Python side and routed most emission through Hopper SM90 Emitter.
    - Added some support for running SM100 kernels via the Python interface.
* CuTe changes:
    - Fix inaccurate GridDim calculation under [CuTe tutorial](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/blackwell/).
    - Add [movmatrix](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-movmatrix) support.
    - Fix smallest MMA-N allowed for Blackwell fp8 and fp16 gemm kernels.
    - Support fp16 accmulator for sm89 fp8 mma.
    - Shorten `nullspace` implementation.
    - Isolate and comment on `cosize` hacks.
    - Important documentation correction: `E<0,1> == 1@0@1`.
* Fix some kernel issues:
    - Fix Hopper SM90 group gemm kernel to only use the commit group and wait group instead of also waiting on mbarriers.
    - Fix a tiny bug when K is large for Blackwell SM103 fp4 grouped GEMM kernel.
* Add following unit tests:
    - [fp16 accmulator for sm89 fp8 mma](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute/ampere/cooperative_gemm.cu)
    - [movmatrix test](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute/turing/movm.cu)
    - [fp8 narrow mma n](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_tensorop_gemm/f16_f16_void_f32_narrow_mma_n.cu) and [fp16 narrow mma n](test/unit/gemm/device/sm100_tensorop_gemm/f8_f8_void_bf16_narrow_mma_n.cu)
* Various improvements and fixes from the community and CUTLASS team. Thanks to everyone who submitted PRs!
* Optimal code generation with CUDA toolkit versions 13.0U1.

## [4.1.0](https://github.com/NVIDIA/cutlass/releases/tag/v4.1.0) (2025-07-16)

### CuTe DSL
* Add aarch64 support, you can now pip install `nvidia-cutlass-dsl` on GB200 systems!
* More examples demonstrating how to use CuTe DSL to write peak-performance kernels
    - [Blackwell Mamba2 SSD](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/mamba2_ssd/mamba2_ssd.py)
    - [Blackwell SM100 persistent dense blockscaled GEMM with static scheduling](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py)
* API updates
    - Please refer to [DSL API changelog](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_api/changelog.html) for details

### CUTLASS C++
* Further enhance Blackwell SM100 Attention kernels in [example 77](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha/).
    - Add variable sequence length support for FMHA Backward kernel.
    - Add varlen test support to Backward runner.
    - Codes support empty batch sequences.
* Replace `subbyte_iterator` with `cute::recast_ptr` when constructing logical iterators/arrays.
* CuTe changes:
    - Rewrite ArithTuple and ScaledBasis for robustness and clarity.
    - Remove buggy and kludgy `get_layoutA|B|C_MN` and friends from Atoms/TiledX.
    - Factor out `print_latex` and friends and rewrite.
    - Factor out `print_svg` and friends and rewrite.
* Support Blackwell SM100 SIMT packed fp32x2 kernels.
* Support residual add for implicit gemm kernels.
* Various fixes for CUTLASS C++ Python interface's EVT tracer:
    - Add verifier for sm90 to report the invalid input.
    - When adding an edge to the graph, if the edge already exists, add an identity compute node to avoid having multiple parallel edges.
    - Register operations of tanh, sigmoid, exp, gelu to the python ast frontend.
    - Replace the NotImplemented Error by packing all nodes into a single topological visitor node as a fallback.
* Fix profiler bugs in exhaustive perf search.
    - Fix incorrect cluster shape output issue when doing exhaustive search.
    - Fix a bug in profiler grouped GEMM for setting tile scheduler swizzles, cluster shapes, and raster orders.
* Fix some profiler issues.
    - Complete the reference for Blackwell blockwise gemm kernels.
    - Fix incorrect regex logic for L1 test.
* Various improvements and fixes from the community and CUTLASS team. Thanks to everyone who submitted PRs!
* Optimal code generation with CUDA toolkit versions 12.9.

## [4.0.0](https://github.com/NVIDIA/cutlass/releases/tag/v4.0.0) (2025-06-03)

### CuTe DSL
* CuTe DSL, a Python DSL centered around CuTe's abstractions
    - [Core DSL implementation files](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL)
    - [DSL quick start](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/quick_start.html)
    - [DSL Overview](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/overview.html)
* [Overhauled documentation with a new dedicated website](https://docs.nvidia.com/cutlass)
* Set of examples demonstrating how to use CuTe DSL to write peak-performance kernels
    - [Blackwell SM100 persistent dense GEMM with static scheduling](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py)
    - [Blackwell SM100 grouped GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/grouped_gemm.py)
    - [Blackwell SM100 fused multi-head attention forward pass](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/fmha.py)
    - [Hopper GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/hopper/dense_gemm.py)
    - [Ampere GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/ampere/tensorop_gemm.py)
    - [FlashAttention-2 implementation targeting Ampere and Ada class GPUs (SM80, SM86, SM89)](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py)
    - [SmemAllocator to facilitate shared memory allocation and management](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/ampere/smem_allocator.py)
    - [C-structure based customized interface between JIT function and user codes](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/cute/ffi/jit_argument.py)
* [Educational notebooks for getting started with CuTe DSL](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks)
* API updates
    - Please refer to [DSL API changelog](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_api/changelog.html) for details

### CUTLASS C++
* Support [Family Specific Architecture Features](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/) which was introduced in CUDA 12.9
  - 100f, 101f, 120f were added to support Family Specific Architecture Features which allows running the same binary on different chips belonging to the same Family (e.g. sm100) without recompiling.  Note 101a is supported since CUTLASS 3.9
* Instruction shapes and redundant accumulation type have been removed from CUTLASS 3.x-style library kernel names to disambiguate kernels and shorten names.
  - For example:
    + `(old) cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x1x1_0_tnn_align8_warpspecialized_cooperative_epi_tma`
    + `(new) cutlass3x_sm90_tensorop_gemm_bf16_bf16_f32_bf16_bf16_128x256x64_1x1x1_0_tnn_align8_warpspecialized_cooperative_epi_tma`
   - If you are using the CUTLASS library kernel names directly (e.g. to compile a subset of the CUTLASS library with `-DCUTLASS_LIBRARY_KERNELS`, filter kernels in the CUTLASS profiler with `--kernels`), please update your uses accordingly, this is a breaking change.
* Further improved [Blockwise](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu) and [Groupwise](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling.cu) GEMMs on Hopper and Blackwell.
  - Added non-power-of-two tile sizes.
  - Improved performance for K-major scale factors.
  - The argument `mma_promotion_interval` has been removed from non-grouped GEMM to align with the grouped and Blackwell SM100 versions.
* Enhance Blackwell SM100 Attention kernels in [example 77](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha/).
  - Support LSE output in FMHA Forward kernel.
  - Enhance performance measurement: support of different warmup iterations; buffer rotation to keep L2 cold; separate testing of persistent and non-persistent.
  - Enhance testing of variable sequence length.
  - Disable B2B mode in MLA to simplify the sample.
  - Clarify that `fmha_gen`  sample only supports head dim 128.
  - Fixes for split-kv output in MLA.
* Improve Blackwell and Hopper grouped GEMM performance, functionality, and profiler support.
  - Enable runtime datatype for Blackwell SM100 grouped GEMM. Profiler support is also added.
  - Enable kernel parameter exploration for Blackwell SM100 grouped GEMM - raster_order, swizzle.
* Add [Blackwell SM100 implicit GEMM conv fprop/dgrad/wgrad unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device_3x/).
* Add dynamic and preferred cluster support for convolution Blackwell SM100 kernels.
* Fix profiler issues which cause no output or not supported error for some kernels.
* Optimizations for Blackwell SM100 and SM120 block scaled kernels.
* Support for Blackwell SM120 blockwise dense gemm in CUTLASS library and profiler.
* New [Hopper SM90 FMHA example](https://github.com/NVIDIA/cutlass/tree/main/examples/88_hopper_fmha/), similar in design to the existing [Blackwell FMHA](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha/).
* CuTe changes:
    - Rework `cute::copy_if` so that the predicate tensor is also a true CuTe Tensor rather than a lambda and introduces transform-tensors to avoid any extra register or load/store overhead in using bool-tensors.
    - New [CuTe tutorial](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/tiled_copy_if.cu) to show the usage of copy_if in tile copy.
    - Add [CuTe C++ reduce op](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/tensor_reduce.hpp).
        - Add several [unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute/core/tensor_algs.cpp) for CuTe tensor algorithms.
* Various improvements and fixes from the community and CUTLASS team. Thanks to everyone who submitted PRs!
* Optimal code generation with CUDA toolkit versions 12.9.


# CUTLASS 3.x

## [3.9.2](https://github.com/NVIDIA/cutlass/releases/tag/v3.9.2) (2025-05-03)
* Fixed [Blockwise](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu) and [Groupwise](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling.cu) GEMM hang issue when problem size K is 128.
* Optimal code generation with CUDA toolkit versions 12.9.

## [3.9.1](https://github.com/NVIDIA/cutlass/releases/tag/v3.9.1) (2025-04-30)
* Fixed Group Gemm hang issue in CUTLASS 3.x
* Improved Hopper [Blockwise](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu) and [Groupwise](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling.cu) GEMM performance.

## [3.9.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.9.0) (2025-04-24)

* Support for Blackwell SM120 kernels for GeForce GPUs in CUTLASS 3.x API:
  - Collective mainloops that target for:
    * [Blockscaled datatypes with support for dense GEMM](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp)
    * [Blockscaled datatypes with support for sparse GEMM](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm120_blockscaled_sparse_mma_tma.hpp)
  - New [GEMM](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/dispatch_policy.hpp) and [epilogue](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/dispatch_policy.hpp) dispatch policies for collectives, kernel layers, and builders.
  - [Blackwell SM120 epilogue](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp) and [full set of EVT fusions](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp).
* Set of examples that demonstrate the usage of the 3.x API for targeting Blackwell SM120 architecture:
  - [Blockscaled GEMM with NVFP4 input datatype and BF16 output tensor](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu).
  - [Blockscaled GEMM with NVFP4 input datatype and NVFP4 output tensor with scale factor generation](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu).
  - [Blockscaled GEMM with mixed input datatype (MXFP8 and MXFP6) and BF16 output tensor](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu).
  - [Grouped GEMM with nvfp4 datatype](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu).
  - [Sparse Blockscaled GEMM with mxfp8 input datatype and BF16 output tensor](https://github.com/NVIDIA/cutlass/tree/main/examples/80_blackwell_geforce_sparse_gemm/80a_blackwell_geforce_mxfp8_bf16_sparse_gemm.cu).
  - [Sparse Blockscaled GEMM with NVFP4 input datatype and NVFP4 output tensor](https://github.com/NVIDIA/cutlass/tree/main/examples/80_blackwell_geforce_sparse_gemm/80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm.cu).
* Set of unit tests that demonstrate the usage of both [sparse](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm120_blockscaled_sparse_tensorop_gemm/) and [dense](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/) Blackwell SM120 blockscaled GEMM.
* Support for Blackwell SM100 Sparse kernels:
  - Collective mainloop that target for
    * [SM100 Sparse GEMM](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm100_sparse_mma_warpspecialized.hpp)
* Set of example that demonstrate the usage of the 3.x API for targeting Blackwell SM100 Sparse GEMM:
  - [Sparse GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/83_blackwell_sparse_gemm/83_blackwell_sparse_gemm.cu)
  - [Blockscaled Sparse GEMM with NVFP4 input data type](https://github.com/NVIDIA/cutlass/tree/main/examples/84_blackwell_narrow_precision_sparse_gemm/84a_blackwell_nvfp4_bf16_sparse_gemm.cu)
  - [Blockscaled Sparse GEMM with mixed input data type (MXFP8 and MXFP4)](https://github.com/NVIDIA/cutlass/tree/main/examples/84_blackwell_narrow_precision_sparse_gemm/84b_blackwell_mixed_mxfp8_bf16_sparse_gemm.cu)
* Set of unit tests that demonstrate the usage of [sparse](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_sparse_tensorop_gemm) and [blockscaled sparse](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm100_blockscaled_sparse_tensorop_gemm) Blackwell SM100 GEMM.
* A new Multi-head Latent Attention (MLA) for SM100 Blackwell architecture in CUTLASS [example](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha/) covers the flashMLA-like weight-absorbed decoding use-case.
* A new FMHA Backward kernel for SM100 Blackwell architecture extends CUTLASS [example](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha/) to show how the five backward pass MMAs can be fused into a single kernel to achieve high performance.
* A new [distributed GEMM example](https://github.com/NVIDIA/cutlass/tree/main/examples/82_blackwell_distributed_gemm/82_blackwell_distributed_gemm.cu) for SM100 Blackwell architecture.
* Enhancement and new support of block-wise and group-wise GEMM for Hopper and Blackwell architectures:
  - Enhancement of [blockwise GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu) for Hopper architecture.
  - Enhancement of [groupwise GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling.cu) for Hopper architecture.
  - Support for [grouped GEMM with blockwise and groupwise scaling](https://github.com/NVIDIA/cutlass/tree/main/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling/) for Hopper architecture.
  - Support for [grouped-wise GEMM](https://github.com/NVIDIA/cutlass/tree/main/tools/profiler/src/blockwise_gemm_operation_profiler.cu) in CUTLASS profiler.
  - Support for [blockwise GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/81_blackwell_gemm_blockwise/81_blackwell_gemm_blockwise.cu) for Blackwell architecture.
  - Support for [groupwise GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/81_blackwell_gemm_blockwise/81_blackwell_gemm_groupwise.cu) for Blackwell architecture.
  - Support for [grouped GEMM with blockwise](https://github.com/NVIDIA/cutlass/tree/main/examples/81_blackwell_gemm_blockwise/81_blackwell_grouped_gemm_blockwise.cu) and [groupwise scaling](https://github.com/NVIDIA/cutlass/tree/main/examples/81_blackwell_gemm_blockwise/81_blackwell_grouped_gemm_groupwise.cu) for Blackwell architecture.
* Added support for enhanced kernel performance search (auto-tuning) in CUTLASS profiler:
  - Sorting performance results by GFLOPs/second: Users can now sort the final performance report based on GFLOPs/second, making it easier to identify the most efficient kernels.
  - Exhaustive search for best kernel performance in GFLOPs/second: The profiler now searches for the best-performing kernel across a range of problem sizes, swizzle sizes, rasterization orders, and dynamic cluster configurations to maximize performance.
  - Performance search under a fixed GEMM shape: Enables exhaustive tuning within a fixed GEMM shape, exploring various kernel parameters to find the best configuration.
  - More detailed introductions and examples to leverage this feature can be found in [profiler.md](https://docs.nvidia.com/cutlass/media/docs/cpp/profiler.html#exhaustive-search-mode-and-top-k-output-ranking-according-to-performance-in-gflopss).
* Support `void` as the D element in sm100 kernel epilogues.
* Various improvements and fixes from the community and CUTLASS team. Thanks to everyone who submitted PRs!
* Optimal code generation with CUDA toolkit versions 12.8U1.

## [3.8.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.8.0) (2025-01-25)

* Support for new CuTe building blocks specifically for Blackwell SM100 architecture:
  - [5th generation Blackwell Tensor Core instructions (TCGen05)](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom/mma_traits_sm100.hpp) via CuTe MMA atoms.
  - Extensions to [Tensor Memory Accelerator](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom/copy_traits_sm100_tma.hpp) via CuTe Copy atoms.
  - Exposure of Blackwell's new tensor memory (note: distinct from TMA) as [`tmem`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/pointer.hpp) across CuTe as a first class data locale.
  - Exposure of [`tmem->rmem`, `rmem->tmem` and `smem->tmem data movement instructions`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom/copy_traits_sm100.hpp) as copy atoms in CuTe.
  - [`make_tmem_copy()`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom/copy_traits_sm100.hpp) utility method to ease creation of tiled copies for tmem copy atoms.
  - Support for [new variants of LDSM on Blackwell](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom/copy_traits_sm100.hpp) via CuTe Copy atoms.
* Support for new CUTLASS building blocks specifically for Blackwell SM100 architecture:
  - Various narrow precision [FP4, FP6, and FP8](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/exmy_base.h) formats as well as their [block-scaled variants NVFP4, MXFP4, MXFP6, and MXFP8](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/float_subbyte.h)
  - [Pipelines that implement Blackwell specific synchronization](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/pipeline/sm100_pipeline.hpp).
  - [Cluster launch control API supporting preferred and fallback cluster shapes](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/cluster_launch.hpp).
  - Data types including NVFP4, MXFP4, MXFP6, and MXFP8 and all their supported element and scale factor types.
  - Tile schedulers using [Blackwell's Cluster Launch Control (CLC) feature](https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_cluster_launch_control.html) to implement dynamic persistence scheduling for [GEMMs](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp), and [stream-K](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm100_tile_scheduler_stream_k.hpp).
  - Extensions to testbeds and reference check code for unit tests and CUTLASS profiler.
* Full support for Blackwell SM100 kernels in CUTLASS 3.x API:
  - [Blackwell specific kernel layers](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp) that
    + Implement a new warp-specialization recipe tuned specifically for Blackwell SM100 architecture.
    + Leverage all the new features such as CLC based tile scheduling, preferred cluster, and TMEM based double buffering of accumulators.
    + Support stream-K load balancing for all kernel types everywhere via composable scheduler support.
  - Blackwell collective mainloops that target the TCGen05 MMA instructions (both SS and TS) for
    * [Non-block scaled data types without support for pointer array and grouped GEMM with TMA](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp)
    * [Non-block scaled data types with support for pointer array and grouped GEMM with TMA](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm100_mma_array_warpspecialized.hpp)
    * [Block scaled data types without support for pointer array and grouped GEMM with TMA](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp)
    * [Block scaled data types with support for pointer array and grouped GEMM with TMA](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm100_blockscaled_mma_array_warpspecialized.hpp)
  - Blackwell [collective mainloop for convolution kernels](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/conv/collective/sm100_implicit_gemm_umma_warpspecialized.hpp) supporting non-block scaled data types for fprop, dgrad, and wgrad.
  - New [GEMM](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/dispatch_policy.hpp), [convolution](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/conv/dispatch_policy.hpp), and [epilogue](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/dispatch_policy.hpp) dispatch policies for collectives, kernel layers, and builders.
  - [Blackwell epilogue that supports loading accumulators from `tmem`](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp) and full set of EVT fusions.
* CUTLASS library and profiler integration for block scaled data types for kernel emission, profiling, and verification.
  - Support for preferred and fallback cluster shapes via profiler command line arguments parsing to set dynamic cluster shapes.
  - Support for dynamic datatypes by parsing profiler via profiler command line arguments parsing to set dynamic datatype setting in TCGen05 MMA instruction descriptors.
  - Support for mixed input GEMM kernels on Hopper in the profiler.
* New CUTLASS profiler flag `use-cuda-graphs` to reduce overheads when benchmarking launch-bound kernels.
* A new 3.x version of grouped GEMM to the CUTLASS library and generates kernels for Hopper and Blackwell. Now grouped GEMM support is enabled in the CUTLASS profiler (`./cutlass_profiler --operation=GroupedGemm --help` for details).
* Set of examples that demonstrate the usage of the 3.x API for targeting Blackwell SM100 architecture:
  - [Basic FP16 and FP8 GEMMs with minimal changes from Hopper examples](https://github.com/NVIDIA/cutlass/tree/main/examples/70_blackwell_gemm/), demonstrating ease of migration for off the shelf kernels using the 3.x collective builder API.
  - GEMM with [opt-in collective builder schedules showcasing available recipes](https://github.com/NVIDIA/cutlass/tree/main/examples/71_blackwell_gemm_with_collective_builder/71_blackwell_gemm_with_collective_builder.cu) for Blackwell.
  - Block scaled data type GEMMs targeting Blackwell's native block scaled Tensor Cores:
    + [NVFP4 inputs with BF16 output](https://github.com/NVIDIA/cutlass/tree/main/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu)
    + [NVFP4 inputs with NVFP4 output](https://github.com/NVIDIA/cutlass/tree/main/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu)
    + [Mixed MXFP8 and MXFP6 inputs with BF16 output](https://github.com/NVIDIA/cutlass/tree/main/examples/72_blackwell_narrow_precision_gemm/72c_blackwell_mixed_mxfp8_bf16_gemm.cu)
  - GEMM example demonstrating [Blackwell's new preferred cluster support via dynamic cluster shapes](https://github.com/NVIDIA/cutlass/tree/main/examples/73_blackwell_gemm_preferred_cluster/blackwell_gemm_preferred_cluster.cu) for increased occupancy.
  - [GEMM with CLC based StreamK scheduler for load balancing](https://github.com/NVIDIA/cutlass/tree/main/examples/74_blackwell_gemm_streamk/blackwell_gemm_streamk.cu).
  - Grouped GEMM for [vanilla FP8 data inputs](https://github.com/NVIDIA/cutlass/tree/main/examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm.cu) and [NVFP4 block scaled inputs](https://github.com/NVIDIA/cutlass/tree/main/examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm_block_scaled.cu).
  - Convolution kernels for [fprop](https://github.com/NVIDIA/cutlass/tree/main/examples/76_blackwell_conv/76_blackwell_conv_fprop.cu), [dgrad](https://github.com/NVIDIA/cutlass/tree/main/examples/76_blackwell_conv/76_blackwell_conv_dgrad.cu), and [wgrad](https://github.com/NVIDIA/cutlass/tree/main/examples/76_blackwell_conv/76_blackwell_conv_wgrad.cu).
  - [Fused multi-head attention fprop kernel](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha/77_blackwell_fmha.cu) supporting fp16/bf16/fp8 data types across head dims of 32,64, and 128.
  - A new BF16x9 GEMM [kernel](https://github.com/NVIDIA/cutlass/tree/main/examples/78_blackwell_emulated_bf16x9_gemm/78_blackwell_emulated_bf16x9_gemm.cu) that emulates FP32 GEMM (SGEMM) using BF16 operations.
* Set of examples that demonstrate the usage of the 3.x API for targeting Hopper architecture:
  - A set of new [Hopper grouped GEMM kernels](https://github.com/NVIDIA/cutlass/tree/main/examples/69_hopper_mixed_dtype_grouped_gemm/) that support mixed A and B datatypes.
  - A new [Hopper FP8 GEMM with groupwise scaling](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling.cu).
* Documentation updates:
  - [Quickstart - instantiating a Blackwell block-scaled GEMM](https://docs.nvidia.com/cutlass/media/docs/cpp/quickstart.html#instantiating-a-blackwell-sm100-gemm-kernel).
  - Detailed [Blackwell block-scaled GEMM functionality documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html)
  - A new [functionality documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/functionality.html) specifically for 3.x API comprehensively documenting all supported kernel types, data types, kernel features, minimum CUDA tookit support etc for 3.x supported architectures.
  - Updates to [compatibility](https://docs.nvidia.com/cutlass/overview.html#compatibility) section regarding supported compilers, operating systems, CUDA Toolkits, Hardware Architectures, and [Target Architecture](https://docs.nvidia.com/cutlass/overview.html#target-architecture).
  - Updates to [profiler documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/profiler.html) for testing mixed input GEMM kernels on Hopper.

## [3.7.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.7.0) (2025-01-11)
- [Hopper blockwise scaling FP8 GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu) uses 2D scaling tensor, assigning one value per threadblock.  This allows a finer-grained scaling to be applied for each output tile per gemm-k iteration. The operands and scaling tensors are loaded from global memory to shared memory using TMA and cp_async, respectively. The scaling is applied inside the mainloop.  Details with figures are [here](https://github.com/NVIDIA/cutlass/pull/1932#issue-2645398439).
- [Distributed GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/65_distributed_gemm/65_distributed_gemm.cu) is a new (experimental) API which can turn existing CUTLASS GEMM kernels into pipelined Tensor Parallel GEMMs that run efficiently on NVLink-based network of GPUs. Its pipelining schedules can hide most of the communication behind computation, and relies on point-to-point communication, which can simply use CUDA runtime's peer device access feature. It also utilizes remote TMA loads and memcopies with CUDA graphs to handle communication primarily through the Copy Engine, leaving all SMs free for Hopper's persistent kernels.  For more details you can refer to the [DistGEMM blog post](https://blog.shi-labs.com/distributed-gemm-88be6a481e2b).
- Improved persistent grid launch for Hopper kernels with large cluster sizes (>= size of 4) using the new `make_kernel_hardware_info` API as shown in [example 48](https://github.com/NVIDIA/cutlass/tree/main/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu).
- Enabled high precision accumulation for Hopper FP8 Sparse GEMM.
- Potential API breaking changes:
  + Fix `cute::UniversalCopy` for type safety.
  + No longer implicitly select `cute::SM80_CP_ASYNC_*` based on input tensors. This avoids implicit downstream synchronization requirements. To use `SM80_CP_ASYNC`, users must explicitly select the appropriate CopyAtom.
  + Fix `cute::SM80_CP_ASYNC_CACHEALWAYS`, `cute::SM80_CP_ASYNC_CACHEGLOBAL`, `cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL`, `cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL` to avoid implicitly selecting `ZFILL` behavior on predication.
  + Remove `cute::copy_vec<T>` in favor of `cute::copy_aligned` and `cute::copy(AutoVectorizingCopyWithAssumedAlignment<NumBits>,...)`.
  + A refactor of default epilogue struct `DefaultEpilogue` [API](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/collective/default_epilogue.hpp) to avoid reading non-void `ElementC` value for `ElementC = void` kernel.
- New CUTLASS profiler flags: `profiling-duration`, `min-iterations`, and `kernels-file` documented in [profiler.md](https://docs.nvidia.com/cutlass/media/docs/cpp/profiler.html#cutlass-profiler).
- Various improvements and fixes from the community and CUTLASS team. Thanks to everyone who submitted PRs!
- Optimal code generation with CUDA toolkit versions 12.6.

## [3.6.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.6.0) (2024-10-03)

- [Hopper structured sparse GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/62_hopper_sparse_gemm/62_hopper_sparse_gemm.cu).
  + [FP16](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_sparse_gemm_f16_f16_f32_tensor_op_f32.cu)
  + [FP8](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_sparse_gemm_f8_f8_f32_tensor_op_f32.cu)
  + [INT8](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_sparse_gemm_s8_s8_s32_tensor_op_s32.cu)
  + [TF32](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_sparse_gemm_tf32_tf32_f32_tensor_op_f32.cu)
- A refactor to the CUTLASS 3.x convolution `kernel::ConvUniversal` [API](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/conv/kernel/sm90_implicit_gemm_tma_warpspecialized.hpp) to bring it in line with `gemm::GemmUniversal`. Now the 3.x convolution API is no longer considered as a beta API.
- [An improved mixed input GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm/README.md) and a [lookup table implementation](https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm.cu) for `INT4`x`FP8` scale-only mode.
- [EVT nodes for Top-K selection and softmax](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/fusion/sm90_visitor_topk_softmax.hpp) and [GEMM example using those](https://github.com/NVIDIA/cutlass/tree/main/examples/61_hopper_gemm_with_topk_and_softmax/61_hopper_gemm_with_topk_and_softmax.cu).
- [Programmatic Dependent Launch](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/arch/grid_dependency_control.h) (PDL) that leverages a new Hopper feature to speedup two back-to-back kernels, and its corresponding [documentations](https://docs.nvidia.com/cutlass/media/docs/cpp/dependent_kernel_launch.html).
- [A new debugging tool, synclog](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/arch/synclog.hpp), for dumping out all synchronization events from within a kernel to a file. Please see [synclog documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/utilities.html#debugging-asynchronous-kernels-with-cutlasss-built-in-synclog-tool) for details.
- A new TMA-enabled [epilogue](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/collective/sm90_epilogue_array_tma_warpspecialized.hpp) for grouped GEMM that brings significant performance improvement, as well as its EVT support.
- A SIMT-enabled pointer-array [epilogue](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/collective/sm70_epilogue_vectorized_array.hpp).
- A new [Ping-Pong kernel schedule for Grouped GEMM](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_pingpong.hpp) and some other optimizations.
- [A new instantiation strategy for CUTLASS profiler kernels](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/sm90_shapes.py) along with [improved documentation for instantiation level in CUTLASS profiler](https://docs.nvidia.com/cutlass/media/docs/cpp/profiler.html#instantiating-more-kernels-with-hopper).
- A new hardware support for comparisons and computations of [`cutlass::bfloat16_t`](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/bfloat16.h)
- Fixed use of isnan on Windows for [`half_t`](https://github.com/NVIDIA/cutlass/tree/main/test/unit/core/functional.cu).
- Various improvements and fixes from the community and CUTLASS team. Thanks to everyone who submitted PRs!
- Optimal code generation with CUDA toolkit versions 12.6.

## [3.5.1](https://github.com/NVIDIA/cutlass/releases/tag/v3.5.1) (2024-07-25)

- [Minimal SM90 WGMMA + TMA GEMM example in 100 lines of code](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/wgmma_sm90.cu)
- [Exposure of L2 `cache_hint`s in TMA copy atoms](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch/copy_sm90_tma.hpp#L48)
- Exposure of raster order and tile swizzle extent in [CUTLASS library profiler](./media/docs/cpp/profiler.md#gemm), and
[example 48](https://github.com/NVIDIA/cutlass/tree/main/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu).
- [TMA store based and EVT supported epilogues](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/collective/sm90_epilogue_array_tma_warpspecialized.hpp) for [Hopper pointer array batched kernels](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_f16_f16_f16_tensor_op_f32_ptr_array.cu).
- A new [`GemmSparseUniversal` API for CUTLASS 2.x Ampere kernels](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/device/gemm_sparse_universal.h) to enable serial and parallel split-k for sparse tensor cores and new tiny tile sizes to better support LLM inferrence:
  + [FP16 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16t_f16n_f32t_tensor_op_f32_sparse_sm80.cu#L269-L393) and [NT](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16t_f32t_tensor_op_f32_sparse_sm80.cu#L269-L411).
  + [int8 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s8t_s8n_s32t_tensor_op_s32_sparse_sm80.cu#L264-L452).
  + [int4 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s4t_s4n_s32t_tensor_op_s32_sparse_sm80.cu#L264-L452).
  + [FP32 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f32t_f32n_f32t_tensor_op_f32_sparse_sm80.cu#L427-L642) and [NT](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f32n_f32t_f32t_tensor_op_f32_sparse_sm80.cu#L427-L456).
- [CUDA host adapter](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/cuda_host_adapter.hpp) extensions to support TMA descriptor construction driver APIs.
- Inclusion of more [Hopper fprop, dgrad, and wgrad convolution kernels in CUTLASS library and profiler](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/generator.py).
- Support for residual add (beta != 0) in convolution kernels.
- A new convolution [epilogue](https://github.com/NVIDIA/cutlass/tree/main/examples/16_ampere_tensorop_conv2dfprop/ampere_tensorop_conv2dfprop.cu#L269) for CUTLASS 2.x to support non-packed NHWC output.
- A refactor of [include files throughout CUTLASS core directories](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/collective_mma_decl.hpp) to reduce circular dependencies and [tests to guard against them](https://github.com/NVIDIA/cutlass/tree/main/test/self_contained_includes/CMakeLists.txt).
- [A guide for setting up VSCode to work well with CUTLASS](https://docs.nvidia.com/cutlass/media/docs/cpp/ide_setup.html) and [expanded code style guide](https://docs.nvidia.com/cutlass/media/docs/cpp/programming_guidelines.html).
- Better support for MSVC as a host compiler.
- Many performance optimizations, improvements, and bug fixes including fixes for FlashAttention-2.
- Optimal code generation with CUDA toolkit versions 12.4 and 12.5u1.

## [3.5.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.5.0) (2024-04-09)

- Implicit GEMM Convolutions targeting Hopper SM90A via WGMMA + [TMA im2col](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom/copy_traits_sm90_im2col.hpp)
  + Native implementation in CUTLASS 3.x using CuTe, mirroring the [same design hierarchy as that of GEMMs](https://docs.nvidia.com/cutlass/media/docs/cpp/gemm_api_3x.html).
  + Support for 1D, 2D, and 3D convolutions in a [rank-agnostic fashion](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/conv/convnd_problem_shape.hpp).
  + Support for [Fprop](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device_3x/fprop/sm90_conv3d_fprop_implicit_gemm_s8_s8_s32_tensorop_s32.cu), [Dgrad](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device_3x/dgrad/sm90_conv2d_dgrad_implicit_gemm_f16_f16_f32_tensorop_f16.cu), and [Wgrad](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device_3x/wgrad/sm90_conv1d_wgrad_implicit_gemm_f16_f16_f32_tensorop_f16.cu) algorithms
  + [CUTLASS profiler support](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/conv3x_emitter.py) for 2D and 3D convolutions implemented via the 3.x API.
  + NOTE: this is a beta release. Further updates to CUTLASS will include major performance improvements, feature enablement, and possible breaking changes to the API until 3.7 release. Your feedback is welcome on the design!
- Support for [Ada (SM89) FP8 tensor cores via the 2.x API](https://github.com/NVIDIA/cutlass/tree/main/examples/58_ada_fp8_gemm/ada_fp8_gemm.cu). Requires CUDA 12.4 or newer.
- [Ampere gather/scatter convolution example](https://github.com/NVIDIA/cutlass/tree/main/examples/59_ampere_gather_scatter_conv/README.md) in CuTe and CUTLASS 3.x
  + Showcasing how custom kernels can be written and optimized using CUTLASS 3.x and CuTe and the general strategy for implementing convolutions as specializations of GETTs.
  + Implementation of a coarse grained sparse gather/scatter kernel achieving peak performance on Ampere class tensor cores.
- 32x and 16x tile sizes are added to CUTLASS 2.x to improve the performance of narrow-tall and wide-short matrices.
  + [Ampere FP16 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16t_f16n_f16t_tensor_op_f32_sm80.cu) and [NT](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16t_f16t_tensor_op_f32_sm80.cu#L227-L301), [Ampere INT8 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s8t_s8n_s8t_tensor_op_s32_sm80.cu#L392-L1342), [Ampere INT4 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s4t_s4n_s4t_tensor_op_s32_sm80.cu#L372-L934).
  + [Turing FP16 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16t_f16n_f16t_tensor_op_f32_sm75.cu#L55-L394), [Turing INT8 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s8t_s8n_s8t_tensor_op_s32_sm75.cu#L166-L537), [Turing INT4 TN](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s4t_s4n_s4t_tensor_op_s32_sm75.cu#L310-L564).
- Updates to CuTe documentation for [`cute::Tensor<>`](./media/docs/cpp/cute/03_tensor.md), [MMA atoms](./media/docs/cpp/cute/0t_mma_atom.md), and an overhauled [CuTe GEMM tutorial series](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial).
- Extensions to CuTe to support [L2 prefetching](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/prefetch.hpp) and [TMA store+reductions](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch/copy_sm90_tma.hpp#L1337).
- Remove C++11 requirement on a few CUTLASS 2.x API header files. All CUTLASS files now require C++17.
- Fixes to greatly reduce build warnings.
- Updates and bugfixes from the community (thanks!)

## [3.4.1](https://github.com/NVIDIA/cutlass/releases/tag/v3.4.1) (2024-02-14)

- Statically available [CUTLASS Version macros](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/version.h) that allow for handling API changes between CUTLASS releases on the users' side.
- Improvements for Hopper [Group-GEMMs](https://github.com/NVIDIA/cutlass/tree/main/examples/57_hopper_grouped_gemm) and [Pointer-Array Batched GEMMs](https://github.com/NVIDIA/cutlass/tree/main/examples/56_hopper_ptr_array_batched_gemm).
- Updates and bugfixes from the community (thanks!).

## [3.4.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.4.0) (2024-01-12)
* Expanded [Mixed-input Hopper GEMMs](https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm) support covering {16-bit, 8-bit} x {8-bit, 4-bit} input types with fast numerical converters and group scaling factors.
* Performance improvements to [Mixed-input Hopper GEMMs](https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm)
* Beta release of [Pointer-Array Batched GEMMs](https://github.com/NVIDIA/cutlass/tree/main/examples/56_hopper_ptr_array_batched_gemm) now available on Hopper GPUs utilizing TMA and WGMMA (requires CUDA 12.3 or above).
* Beta release of [Group-GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/57_hopper_grouped_gemm) utilizing TMA and WGMMA (requires CUDA 12.3 or above).
* [Ampere Sparse GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/15_ampere_sparse_tensorop_gemm/ampere_sparse_tensorop_gemm_with_visitor.cu) supports Epilogue Visitor Tree (EVT) now.
* NamedBarriers usability improvement and list of [ReservedNamedBarriers](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/arch/barrier.h) has been officially released.
* Improved CuTe documentation including improved clarity and depth of [Quickstart](./media/docs/cpp/cute/00_quickstart.md), [CuTe Layout](./media/docs/cpp/cute/01_layout.md), and [CuTe Layout Algebra](./media/docs/cpp/cute/02_layout_algebra.md). Associated code comments, post-conditions, and details in [CuTe Core Unit Tests](./test/unit/cute/core/) also improved.

## [3.3](https://github.com/NVIDIA/cutlass/releases/tag/v3.3.0) (2023-10-31)
* [Mixed-input Hopper GEMMs](https://github.com/NVIDIA/cutlass/tree/main/examples/55_hopper_mixed_dtype_gemm) support covering 16-bit x 8-bit input operand types.
* [Mixed-input Ampere GEMMs](https://github.com/NVIDIA/cutlass/pull/1084) with support for canonical layouts (TN). The implementation supports upcast on operandB {fp16, bf16} x {s8, u8}, and upcast on operandA {s8, u8} x {fp16, bf16}.
* [Copy Async based Hopper GEMMs](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_bf16_bf16_bf16_alignx_tensor_op_f32_warpspecialized_cooperative.cu) - which support lower than 16B aligned input tensors.
* Kernel schedules and Builder support for mixed precision and Copy Async GEMMs with < 16B aligned input tensors.
* Profiler support for lower-aligned Hopper GEMMs.
* Performance Improvements to [Scatter-Gather Hopper Example](https://github.com/NVIDIA/cutlass/tree/main/examples/52_hopper_gather_scatter_fusion).
* Sub-Byte type fixes and improvements.
* EVT Support for RELU with Aux bitmap tensor store (used in dRELU). See [SM90 EVT fusions](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp) for details.
* Fusion support for backprop fusions including drelu, dgelu, and dbias.
* Support for void-C kernels and SM80 mixed-input GEMMs in the CUTLASS Python interface

## [3.2.2](https://github.com/NVIDIA/cutlass/releases/tag/v3.2.2) (2023-10-25)
* Minor patch for issue/1138

## [3.2.1](https://github.com/NVIDIA/cutlass/releases/tag/v3.2.1) (2023-09-22)
* Python support SM90 Epilogue Visitor Tree (EVT) on top of the C++ support released in 3.2.0.
* SM80 EVT support in C++ and Python.
* Other SM90 epilogue improvements.
* Splitting CUTLASS library into smaller units based on operation, arch and datatypes. See [1105](https://github.com/NVIDIA/cutlass/discussions/1105) for details.
* Making `tools/library/scripts` packageable - `tools/library/scripts` is now moving to `python/cutlass_library`. See the Python [README](https://github.com/NVIDIA/cutlass/tree/main/python/README.md) for details.
* SM90 TF32 kernel improvements for all layouts.
* SM90 rasterization direction support in the CUTLASS profiler.
* Improvement for CUTLASS profiler build times.
* Remove Python-C++ bindings.

## [3.2.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.2.0) (2023-08-03)

* New warp-specialized persistent FP8 GEMM kernel [kernel schedules](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp) and [mainloops](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp)  targeting Hopper architecture that achieve great performance with TMA, WGMMA, and threadblock clusters. An example showcasing [Hopper warp-specialized FP8 GEMMs](https://github.com/NVIDIA/cutlass/tree/main/examples/54_hopper_fp8_warp_specialized_gemm). FP8 GEMMs come with a fast accumulation mode. When enabled, problem execution might be faster but at the cost of lower accuracy because intermediate results will not periodically be promoted to a higher precision.
* New [Epilogue Visitor Tree (EVT)](https://github.com/NVIDIA/cutlass/tree/main/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu) support for Hopper TMA epilogues. EVTs allows for user-defined customized epilogue fusion patterns without having to write a new epilogue.
* [Stream-K](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp) feature for Hopper. Note that this is only a functional implementation of stream-K, and should not be used for performance comparison. Optimizations are expected in a future release.
* Improved CTA rasterization and support for CTA swizzling for Hopper kernels using the [Tile Scheduler](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp).
* Improved performance for [warp-specialized TensorFloat-32 (TF32) GEMM kernels](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_tf32_tf32_f32_tensor_op_f32_gmma_rs_cluster_warpspecialized.cu) targeting Hopper TMA.
* [Hopper GEMM+Permute](https://github.com/NVIDIA/cutlass/tree/main/examples/53_hopper_gemm_permute/53_hopper_gemm_permute.cu), an example of fusing tensor reordering (permutation) with GEMM mainloop or epilogue.
* New CUTLASS 2D Convolution Python interface. New [example](https://github.com/NVIDIA/cutlass/tree/main/examples/python/03_basic_conv2d.ipynb) here.
* Support for Windows (MSVC) builds. Tested with Visual Studio 2019 v16.11.27 on Windows 10.0.
* Optimal performance using [**CUDA 12.2u1**](https://developer.nvidia.com/cuda-downloads)
* Updates and bugfixes from the community (thanks!)

## [3.1.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.1.0) (2023-04-14)
* New CUTLASS Python interface that aims to provide an ease-of-use interface for instantiating, emitting, compiling, and running CUTLASS kernels via Python. More details [here](https://github.com/NVIDIA/cutlass/tree/main/python/README.md) and new [examples](https://github.com/NVIDIA/cutlass/tree/main/examples/python).
* New [efficient epilogues](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_f16_f16_f16_tensor_op_f32_cluster_warpspecialized_cooperative.cu#L783) using TMA for Hopper.
* Support for [fused epilogues](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_f16_f16_f16_tensor_op_f32_cluster_warpspecialized_cooperative_bias_elementwise.cu), such Bias, ReLU and GELU, using the new efficient epilogues.
* New [warp-specialized TensorFloat-32 (TF32) GEMM kernels](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_tf32_tf32_f32_tensor_op_f32_gmma_rs_cluster_warpspecialized.cu) targeting Hopper TMA.
* New [*warp-specialized persistent cooperative*](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp) kernel design that allows for larger tile sizes and improves performance on Hopper.
* An [example](https://github.com/NVIDIA/cutlass/tree/main/examples/51_hopper_gett) showcasing GEMM-Like Tensor-Tensor Contraction (GETT) capability on Hopper.
* Epilogue builders. Similar to mainloop builders (see [example 49](https://github.com/NVIDIA/cutlass/tree/main/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu)), epilogue builders aim to generate the best-possible epilogue while exposing incremental opt-ins for greater customization.
* Profiler support for overriding kernel and epilogue builder auto schedules for 3.x API kernels, allowing specific policies to be run in the CUTLASS profiler.
* Performance optimizations for the [*warp-specialized persistent ping-pong*](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp) kernel.
* Changes to the [GEMM API 3.x](./media/docs/cpp/gemm_api_3x.md), involving the host-facing arguments and the underlying `Params` structs.
* [FMHA Backward Pass](https://github.com/NVIDIA/cutlass/tree/main/examples/41_fused_multi_head_attention/fused_multi_head_attention_backward.cu) from Meta xFormers.
* [Streamk GEMM with Broadcast](https://github.com/NVIDIA/cutlass/tree/main/examples/47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk_broadcast.cu) enables epilogue broadcast with StreamK GEMM.
* [Batched B2B GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/13_two_tensor_op_fusion) now can run multiple Back-to-Back GEMM with the same problem size in parallel.
* [Batched Strided GEMV](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemv.cu) support both row major and column major input matrix.
* [Permute + GEMM fusion](https://github.com/NVIDIA/cutlass/tree/main/examples/39_gemm_permute) can fuse Permute with following GEMM now.  Before, we only support fusing GEMM with Permute in the epilogue.
* [Row Broadcast](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/threadblock/predicated_tile_iterator_row_broadcast.h) can be fused in the epilogue.
* The GitHub branch is renamed from `master` to `main` in this release.
* Optimal performance using [**CUDA 12.1**](https://developer.nvidia.com/cuda-downloads)
* Updates and bugfixes from the community (thanks!)

## [3.0.0](https://github.com/NVIDIA/cutlass/releases/tag/v3.0.0) (2023-01-23)
* [CuTe](./media/docs/cpp/cute/00_quickstart.md), a [new core library and backend](./include/cute) for CUTLASS 3.0 that defines a single Layout vocabulary type and an associated algebra of layouts for a much more expressive and composable abstraction for tensors, sets of parallel agents, and operations by said agents on tensors.
* [A new conceptual operation hierarchy](./media/docs/cpp/cutlass_3x_design.md) that replaces the architecture-centric hierarchy of CUTLASS 2.x and [documentation for CUTLASS 3.0's GEMM API changes](./media/docs/cpp/gemm_api_3x.md).
* Strict API backwards compatibility that exposes both 2.x and 3.x API kernels through the same [`device::GemmUniversalAdapter`](./include/cutlass/gemm/device/gemm_universal_adapter.h) and [`kernel::GemmUniversal`](./include/cutlass/gemm/kernel/gemm_universal.hpp) types, allowing users to include both APIs in the same translation units. More information can be found in the [3.x backwards compatibility section](./media/docs/cpp/cutlass_3x_backwards_compatibility.md).
* Updates to [Functionality](./media/docs/cpp/functionality.md) which directs users on which kernels are supported via CUTLASS-2 and CUTLASS-3.
* Updates to [Compatibility](./README.md#compatibility) Section regarding supported compilers, operating systems, CUDA Toolkits, Hardware Architectures and [Target Architecture](./README.md#target-architecture).
* New warp-specialized GEMM [kernel schedules](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp) and [mainloops](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp) targeting Hopper architecture that achieve great performance with TMA, WGMMA, and threadblock clusters.
* Extensions to CUTLASS profiler to support threadblock cluster shapes in library and profiler tile configurations.
* [CUTLASS library integration](https://github.com/NVIDIA/cutlass/tree/main/tools/library/src/gemm_operation_3x.hpp) for 3.x API kernels built through the new `CollectiveBuilder` API, enabling CUTLASS profiler.
* Support for [Hopper GEMMs](https://github.com/NVIDIA/cutlass/tree/main/examples/48_hopper_warp_specialized_gemm) through the new 3.0 API with CuTe-based exposure of the Hopper [Tensor Memory Accelerator](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor) and [WGMMA Tensor Core](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions) features.
* Set of examples that demonstrate the usage of the new 3.0 API to easily build GEMM kernels targeting Hopper: examples [48](https://github.com/NVIDIA/cutlass/tree/main/examples/48_hopper_warp_specialized_gemm), [49](https://github.com/NVIDIA/cutlass/tree/main/examples/49_hopper_gemm_schedules_with_collective_builder), and [50](https://github.com/NVIDIA/cutlass/tree/main/examples/50_hopper_gemm_with_epilogue_swizzle).

# CUTLASS 2.x

## [2.11.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.11.0) (2022-11-19)
* [Stream-K](https://github.com/NVIDIA/cutlass/tree/main/examples/47_ampere_gemm_universal_streamk), which is a new general way to do split-K.  It can not only improve performance, but can also significantly reduce the number of tile sizes that need to be profiled to find the best one.
* [Fused multi-head attention Kernel](https://github.com/NVIDIA/cutlass/tree/main/examples/41_fused_multi_head_attention).  It has two variants: one uses batched GEMM for the fixed sequence length, and the other one uses group GEMM for the variable sequence length.  Both versions just need one kernel.
* [Dual GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/45_dual_gemm), which can fuse A x B and A x C into one kernel. Two GEMMs has no producer-consumer dependency.
* Hopper improves [double precision matrix multiplication](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f64n_f64t_f64t_tensor_op_f64_sm90.cu) by 2x compared to Ampere at iso-clocks. It is supported since CUDA 11.8.
* [BLAS3](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/hemm_cf64_cf64_cf64_tensor_op_f64_sm90.cu) functions with Hoppers new double precision matrix multiplication instructions.
* [ELL Block Sparse GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/43_ell_block_sparse_gemm), which uses an [ELL matrix](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/) to describe the sparsity of A matrix.  B and output matrices are still dense. The block size can be arbitary.
* Optimized [Group Conv](https://github.com/NVIDIA/cutlass/tree/main/examples/42_ampere_tensorop_group_conv) for SingleGroup mode, which requires that the output channel per group is a multiple of Threadblock tile N.
* [Optimized DepthWise Conv](https://github.com/NVIDIA/cutlass/tree/main/examples/46_depthwise_simt_conv2dfprop/depthwise_simt_conv2dfprop.cu).  Two new modes are added
  * [kOptimized](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/depthwise_conv2d_fprop_direct_conv_f16nhwc_f16nhwc_f16nhwc_simt_f16_sm60.cu) - use direct conv to compute instead of implicit GEMM.
    *  The restrictions are: 1) input ,output channel and group number should be multiple of (128 / sizeof(input element)). 2) The input filter size should be the same as the template parameter configuration.
  * [kFixedStrideDilation](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/depthwise_conv2d_fprop_direct_conv_fixed_stride_dilation_f16nhwc_f16nhwc_f16nhwc_simt_f16_sm60.cu) - which puts stride and dilation into templates to further improve the performance. In this mode, kernel persistents some inputs into register to squeeze more performance, so large filter/stride/dilation is not recommanded.
    * The restrictions are: 1) input, output channel and group number should be multiple of (128 / sizeof(input element)). 2) input filter size, stride, dilation should same as the template parameter configuration.
* [Scripts](https://github.com/NVIDIA/cutlass/tree/main/examples/44_multi_gemm_ir_and_codegen) to fuse multiple back-to-back GEMM.  Its implementation was discussed in a GTC'22 Spring [talk](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41606/).
* [FP8 data type definition](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/float8.h) and [conversion routines](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/numeric_conversion.h#L1274-2115).
* Updates and bugfixes from the community (thanks!).  Big shout out to Meta's [xFormers](https://github.com/facebookresearch/xformers).

* **Deprecation announcement:** CUTLASS plans to deprecate the following:
  * Maxwell and Pascal GPU architectures
  * Ubuntu 16.04
  * CUDA 10.2

## [2.10.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.10.0) (2022-08-23)
* [CUTLASS Python](https://github.com/NVIDIA/cutlass/tree/main/examples/40_cutlass_py) now supports GEMM, CONV, Group GEMM for different data types as well as different epilogue flavours.
* Optimizations for CUTLASS's [Grouped GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/24_gemm_grouped/gemm_grouped.cu) kernel.  Threadblock scheduling part is improved.  Some computation can be moved to the host side if applicable.  [Grouped Syr2k](https://github.com/NVIDIA/cutlass/tree/main/examples/38_syr2k_grouped/syr2k_grouped.cu) kernels are added, too.
* Optimizations for [GEMM+Softmax](https://github.com/NVIDIA/cutlass/tree/main/examples/35_gemm_softmax).  All the reduction computation is fused into the previous GEMM.  More template arguments are provided to fine tune the performance.
* [Grouped GEMM for Multihead Attention](https://github.com/NVIDIA/cutlass/tree/main/examples/41_multi_head_attention).  This general group gemm based MHA does not require the sequence length of all GEMMs to be the same which makes it most useful for natural language processing.
* [GEMM + Layer norm fusion for Ampere](https://github.com/NVIDIA/cutlass/tree/main/examples/37_gemm_layernorm_gemm_fusion/) splits the layernorm into two parts and both of them can be fused into the GEMMs before and after separately.  In addition to use square sum to compute variance of layernorm, [Shift-K](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data) is provided if square sum raise numerical issues.
* [GEMM Epilogue Permutation Fusion](https://github.com/NVIDIA/cutlass/tree/main/examples/39_gemm_permute) can apply user provided permutation layout mapping in the GEMM epilogue.
* [Grouped convolution targeting implicit GEMM](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/group_conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_sm80.cu) introduces the first group convolution implementation to CUTLASS.  It is an Analytical implementation, not an Optimized.  The restrictions are: 1) input and output channel number should be multiple of group number. 2) split-K is not supported.  The implementation has 2 modes:
  * kSingleGroup: output channel per group is multiple of Threadblock tile N.
  * kMultipleGroup: Threadblock tile N is multiple of output channel per group.
* [Depthwise separable convolution](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/depthwise_conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_simt_f16_sm60.cu) introduces the first depthwise convolution which is also Analytical for now.  The restrictions are: 1) SIMT only 2) No split-K 3) input channel equals to output channel equals to group number.
* Standalone [Layernorm](https://github.com/NVIDIA/cutlass/tree/main/tools/util/include/cutlass/util/device_layernorm.h) and [Pooling](https://github.com/NVIDIA/cutlass/tree/main/tools/util/include/cutlass/util/device_nhwc_pooling.h) kernels.
* [Back-to-back GEMM/CONV](https://github.com/NVIDIA/cutlass/tree/main/examples/13_two_tensor_op_fusion) relaxes the requirement that the first GEMM K dimension needs to be the multiple of Threadblock Tile K dimension.
* Optimal performance using [**CUDA 11.6u2**](https://developer.nvidia.com/cuda-downloads)
* Updates and bugfixes from the community (thanks!)

## [2.9.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.9.0) (2022-04-21)

* [First layer Convolution kernels](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_fixed_channels_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_sm80.cu) specialized for small channel counts and reduced alignment
  * [Few channels](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_few_channels.h) specialization for reduced alignment capabilities
  * [Fixed channels](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_fixed_channels.h) further specialized when channel count perfectly matches the access vector size
  * [Unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_few_channels_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_sm80.cu)
  * [Python-based instance emitter](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/generator.py) in the CUTLASS Library and support in the Profiler
* [BLAS3](https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-3-function-reference) operators accelerated by Tensor Cores
  * Supported types: f32, cf32, f64, cf64, tf32x3, complex tf32x3
  * [HERK](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/her2k_cf32h_cf32n_tensor_op_fast_f32_sm80.cu) with [emitter](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/rank_k_operation.py)
  * [SYRK](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/syrk_f32n_f32t_tensor_op_fast_f32_sm80.cu) with [emitter](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/rank_k_operation.py)
  * [SYMM](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/symm_f32n_f32n_tensor_op_fast_f32_ls_sm80.cu) with [emitter](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/symm_operation.py)
  * [TRMM](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/trmm_f32n_f32t_f32t_tensor_op_fast_f32_ls_sm80.cu) with [emitter](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/trmm_operation.py)
  * [Unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/testbed_rank_k_universal.h)
* [CUTLASS Python](https://github.com/NVIDIA/cutlass/tree/main/examples/40_cutlass_py) demonstrating JIT compilation of CUTLASS kernels and a Python-based runtime using [CUDA Python](https://developer.nvidia.com/cuda-python)
  * [Python-based runtime](https://github.com/NVIDIA/cutlass/tree/main/tools/library/scripts/rt.py) interoperable with existing emitters
* [GEMM + Softmax example](https://github.com/NVIDIA/cutlass/tree/main/examples/35_gemm_softmax)
* [Gather and Scatter Fusion with GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/36_gather_scatter_fusion) can gather inputs and scatters outputs based on indices vectors in the same GEMM kernel.
  * It can select random rows in a row major matrix.
  * It can select random columns in a column major matrix.
* [Back-to-back GEMM/CONV](https://github.com/NVIDIA/cutlass/tree/main/examples/13_two_tensor_op_fusion) fully supports buffering the first GEMM/CONV results in the shared memory for the latter one to use.  It can eliminate register spill when the tile size is big.  Additionally, bias vector add is supported in the first GEMM/CONV.
  * Supported kernels: GEMM and CONV.
  * Supported types: fp16 and int8.
  * Supported architectures: Turing and Ampere.
* [Transposed Convolution](https://github.com/NVIDIA/cutlass/tree/main/examples/34_transposed_conv2d) (a.k.a Deconvolution) support which reuses Dgrad implementation.
* [Utility functions](https://github.com/NVIDIA/cutlass/tree/main/tools/util/include/cutlass/util) that can pad NHWC and convert between NCHW and NHWC.
* [Small alignment implicit gemm](https://github.com/NVIDIA/cutlass/issues/242) support for Fprop/Dgrad/Wgrad so that padding is no longer mandated to use tensor cores in these kernels.
* Epilogue enhancement:
  * Eliminate bank conflicts in int8 tensor core kernels.
  * Half2 usage if epilogue compute type is fp16.
  * More activation functions: Silu, Hardswish, Leaky Relu.
  * New elementwise fusion pattern for [residual block](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/thread/linear_combination_residual_block.h).
* [Group GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/24_gemm_grouped) thread block number calculation fix which helps to launch the intended number of threadblocks to fully occupy the GPUs.
* [Parallel GEMM splitk](https://github.com/NVIDIA/cutlass/pull/277) support in the CUTLASS profiler.
* Optimal performance using [**CUDA 11.6u2**](https://developer.nvidia.com/cuda-downloads)
* Updates and bugfixes from the community (thanks!)


## [2.8.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.8.0) (2021-11-19)

* **TF32x3:** emulated single-precision using Tensor Cores
  * 45+ TFLOPs on NVIDIA A100
  * [GEMM SDK example](https://github.com/NVIDIA/cutlass/tree/main/examples/27_ampere_3xtf32_fast_accurate_tensorop_gemm/27_ampere_3xtf32_fast_accurate_tensorop_gemm.cu) (real)
  * [COMPLEX GEMM SDK example](https://github.com/NVIDIA/cutlass/tree/main/examples/29_ampere_3xtf32_fast_accurate_tensorop_complex_gemm/29_3xtf32_complex_gemm.cu) (complex)
  * [Implicit GEMM Convolution SDK example](https://github.com/NVIDIA/cutlass/tree/main/examples/28_ampere_3xtf32_fast_accurate_tensorop_fprop/ampere_3xtf32_fast_accurate_tensorop_fprop.cu)
* **Mainloop fusion for Convolution:** convolution with fused per-channel scale-bias-relu
  * [Conv Fprop SDK example](https://github.com/NVIDIA/cutlass/tree/main/examples/25_ampere_fprop_mainloop_fusion/ampere_fprop_mainloop_fusion.cu)
  * [Conv WGrad SDK example](https://github.com/NVIDIA/cutlass/tree/main/examples/26_ampere_wgrad_mainloop_fusion/ampere_wgrad_mainloop_fusion.cu)
  * [cutlass::conv::device::ImplicitGemmConvolutionFusion](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/conv/device/implicit_gemm_convolution_fusion.h)
* **Grouped GEMM:** similar to batched GEMM with distinct problem size per group
  * [SDK example](https://github.com/NVIDIA/cutlass/tree/main/examples/24_gemm_grouped) with performance comparison with Batched Strided GEMM
  * [cutlass::gemm::device::GemmGrouped](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/device/gemm_grouped.h)
* [Implicit GEMM Convolution fusion](https://github.com/NVIDIA/cutlass/tree/main/examples/13_two_tensor_op_fusion/) supports staging 1st convolution's output accumulator in the shared memory on Turing. This allows more flexible warp tile sizes and less regsiter pressue.
* Optimal performance using [**CUDA 11.5**](https://developer.nvidia.com/cuda-downloads)
* Updates from the community (thanks!)

* **Deprecation announcement:** CUTLASS plans to deprecate the following:
  * Maxwell and Pascal GPU architectures
  * Ubuntu 16.04
  * CUDA 10.2

## [2.7.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.7.0) (2021-09-24)
  * Mainloop fusion for GEMM: [summation over A or B](https://github.com/NVIDIA/cutlass/tree/main/examples/23_ampere_gemm_operand_reduction_fusion/ampere_gemm_operand_reduction_fusion.cu)
  * [Strided DGRAD (optimized iterators)](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/conv/kernel/default_conv2d_dgrad.h)
  * [Half-precision GELU_taylor activation functions](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/thread/activation.h#L196)
    * Use these when accumulation and epilogue compute types are all `cutlass::half_t`
  * Tuning and bug fixes to [fused GEMM + GEMM example](https://github.com/NVIDIA/cutlass/tree/main/examples/13_two_tensor_op_fusion/)
  * Support for smaller than 128b aligned Convolutions: [see examples](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm80.cu#L272)
  * Caching of results to accelerate Convolution [unit tests](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/cache_testbed_output.h)
    * Can be enabled or disabled by running `cmake .. -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=OFF`
  * Corrections and bug fixes reported by the CUTLASS community
    * Thank you for filing these issues!

## [2.6.1](https://github.com/NVIDIA/cutlass/releases/tag/v2.6.1) (2021-09-03)
  * Arbitrary padding and striding for CUTLASS Strided DGRAD Convolution operator (Analytic Iterators)
  * Tuning for GEMMs fused with partial reductions
  * Corrections and bug fixes reported by the CUTLASS community
    * Thank you for filing these issues!

## [2.6.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.6.0) (2021-07-22)
  * Optimal performance when compiled with the [CUDA 11.4 Toolkit](https://developer.nvidia.com/cuda-toolkit)
    * Adopt the new L2 prefetch feature in [cp.async](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/arch/memory.h) and [global load](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/arch/memory_sm80.h)
  * Fused operators with GEMM and Convolution
    * [Fused broadcast in epilogue](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_with_broadcast_f16n_f16n_f16n_tensorop_f32_sm75.cu)
    * [Fused partial reduction in epilogue](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_with_reduction_f16n_f16n_f16n_tensorop_f32_sm75.cu)
  * 64b tensor strides and leading dimensions support for GEMMs
  * Affine rank=2 matrix layouts
    * Row stride and column stride for matrices using [cutlass::layout::AffineRank2](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/layout/matrix.h)
    * Support [FP64 tensor core](https://github.com/NVIDIA/cutlass/tree/main/examples/18_ampere_fp64_tensorop_affine2_gemm/ampere_fp64_tensorop_affine2_gemm.cu) and SIMT GEMM.
  * [Batched GEMV](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemv.cu) preview implementation
  * [New strided Dgrad](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_strided_dgrad_implicit_gemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_sm80.cu) implementation
    * Accelerates over previous implementation by cutting down redundant math by 4x
    * Support using new `Dy` and `w` analytic iterators and existing `cutlass::conv::device::ImplicitGemmConvolution` interface
  * Quaternion-valued GEMM and Convolution in single- and double-precision (targeting CUDA Cores)
    * Updates to [quaternion.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/quaternion.h) and [functional.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/functional.h)
    * SDK Example for [GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/21_quaternion_gemm/quaternion_gemm.cu) and [Convolution](https://github.com/NVIDIA/cutlass/tree/main/examples/22_quaternion_conv/quaternion_conv.cu)
    * [Unit tests for GEMM](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/simt_qgemm_nn_sm50.cu) and [Convolution](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_qf32nhwc_qf32nhwc_qf32nhwc_simt_f32_sm50.cu)
  * Many improvements to the epilogue.
    * Provide an [option](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/threadblock/epilogue.h) to not fully unroll the epilogue to reduce the code size and improve the performance when using complicated elementwise operations
    * Performance improvement for FP16 tensor core kernels
    * Bug fixes
  * Enhanced Clang support and the combination of Clang 13 and CUDA 11.4 can build and run kernels from Pascal and Ampere.
  * Updated minimum CUDA Toolkit requirement to 10.2
    * [CUDA 11.4 Toolkit](https://developer.nvidia.com/cuda-toolkit) recommended
  * Corrections and bug fixes reported by the CUTLASS community
    * Thank you for filing these issues!

## [2.5.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.5.0) (2021-02-26)
  * Tensor reductions
    * _m_-to-_n_ reductions of tensors with affine layout
    * [Specializations](https://github.com/NVIDIA/cutlass/tree/main/test/unit/reduction/device/tensor_reduce_contiguous.cu) for reductions including contiguous dimension
    * [Specializations](https://github.com/NVIDIA/cutlass/tree/main/test/unit/reduction/device/tensor_reduce_strided.cu) for reductions excluding contiguous dimension
    * Custom reduction functors such as `cutlass::logical_and`
    * Large tensor support, up to 2^63 elements (however, each dimension is limited to an extent of 2^31)
  * Optimizations for 3-D convolution
    * [Optimized tile iterators](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/conv/threadblock/conv3d_fprop_activation_tile_access_iterator_optimized.h) using precomputed delta table for 3-D convolution
    * Full coverage of [forward](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv3d_fprop_implicit_gemm_f16ndhwc_f16ndhwc_f32ndhwc_tensor_op_f32_sm80.cu) and [backwards](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv3d_dgrad_implicit_gemm_f16ndhwc_f16ndhwc_f32ndhwc_tensor_op_f32_sm80.cu) passes for 3D convolution
  * [Fused Convolution+Convolution example](https://github.com/NVIDIA/cutlass/tree/main/examples/13_two_tensor_op_fusion/README.md)
  * Corrections and bug fixes reported by the CUTLASS community
    * Thank you for filing these issues!


## [2.4.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.4.0) (2020-11-19)
  * Implicit GEMM convolution kernels supporting CUDA and Tensor Cores on NVIDIA GPUs
    * Operators: forward (Fprop), backward data gradient (Dgrad), and backward weight gradient (Wgrad) convolution
    * Data type: FP32, complex<FP32>, Tensor Float 32 (TF32), BFloat16 (BF16), Float16, Int4, Int8, Int32
    * Spatial dimensions: 1-D, 2-D, and 3-D
    * Layout: NHWC, NCxHWx
  * Implicit GEMM convolution components:
    * Global memory iterators supporting Fprop, Dgrad, and Wgrad
    * `MmaMultistage` for implicit GEMM convolution for NVIDIA Ampere architecture
    * `MmaPipeline` for implicit GEMM convolution for NVIDIA Volta and Turing architectures
    * [Documentation](./media/docs/cpp/implicit_gemm_convolution.md) describing Implicit GEMM Convolution algorithm and implementation

## [2.3.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.3.0) (2020-09-23)
 * [NVIDIA Ampere Architecture features](https://devblogs.nvidia.com/nvidia-ampere-architecture-in-depth/)
   * [Sparse Tensor Core GEMM kernels](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16n_f32t_tensor_op_f32_sparse_sm80.cu):
     * Direct access to Sparse Tensor Cores and maximum performance via [`mma.sp.sync`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma-and-friends)
   * Fast SGEMM targeting GeForce RTX 30-series CUDA Cores
 * Minor Features:
   * [Activation functions](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/thread/activation.h) such as [GeLU](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/thread/linear_combination_gelu.h) and [Sigmoid](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/thread/linear_combination_sigmoid.h)
   * Small [matrix](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/matrix.h) and [quaternion](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/quaternion.h) template classes in device code
   * [Floating-point constants](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/constants.h)
 * NVIDIA Ampere GPU Architecture examples and documentation:
   * [Tensor Float 32](https://github.com/NVIDIA/cutlass/tree/main/examples/14_ampere_tf32_tensorop_gemm/ampere_tf32_tensorop_gemm.cu) and
   * [Sparse Tensor Cores](https://github.com/NVIDIA/cutlass/tree/main/examples/15_ampere_sparse_tensorop_gemm/ampere_sparse_tensorop_gemm.cu)
   * Documentation added on CUTLASS [efficient row-major epilogue](./media/docs/cpp/gemm_api.md#efficient-epilogue)

## [2.2.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.2.0) (2020-06-08)
 * [NVIDIA Ampere Architecture features](https://devblogs.nvidia.com/nvidia-ampere-architecture-in-depth/)
   * Fast Tensor Core operations:
    * Maximum performance via [`mma.sync`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma-and-friends)
    * Tensor Float 32, BFloat16, and double-precision data types
    * Mixed integer data types (int8, int4, bin1)
   * Asynchronous copy for deep software pipelines via [`cp.async`](https://docs.nvidia.com/cuda/parallel-thread-execution)
   * Described in [GTC 2020 Webinar (SR 21745)](https://developer.nvidia.com/gtc/2020/video/s21745) (free registration required)
 * Features:
   * SDK examples showing GEMM fused with bias+relu and fused GEMM+GEMM
   * Complex-valued GEMMs targeting NVIDIA Ampere Tensor Cores in double-precision and Tensor Float 32
   * Gaussian complex GEMMs using 3m complex multiply algorithm
   * Universal GEMM kernel supporting two batch modes and two algorithms for parallel reductions
 * Policy updates:
   * [CUDA 11 Toolkit](https://developer.nvidia.com/cuda-toolkit) needed to enable NVIDIA Ampere Architecture features
   * Disabled F16C by default for compatibility - enable on cmake command line with `-DCUTLASS_ENABLE_F16C=ON`

## [2.1.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.1.0) (2020-04-06)
 * BLAS-style host-side API added to [CUTLASS Library](./media/docs/cpp/quickstart.md#cutlass-library)
    * API to launch compiled kernel instances for GEMM and planar complex GEMM
 * Planar Complex GEMM kernels targeting Volta and Turing Tensor Cores
    * Computes complex matrix products on matrices stored as disjoint real and imaginary parts
    * [SDK Examples of Planar Complex GEMMs](https://github.com/NVIDIA/cutlass/tree/main/examples/10_planar_complex/planar_complex.cu)
 * Minor enhancements and bug fixes

## [2.0.0](https://github.com/NVIDIA/cutlass/releases/tag/v2.0.0) (2019-11-19)
 * Substantially refactored for
    * Better performance, particularly for native Turing Tensor Cores
    * Robust and durable templates spanning the design space
    * Encapsulated functionality embodying modern C++11 programming techniques
    * Optimized containers and data types for efficient, generic, portable device code
  * Updates to:
    * [Quick start guide](./media/docs/cpp/quickstart.md)
    * [Documentation](./README.md#documentation)
    * [Utilities](./media/docs/cpp/utilities.md)
    * [CUTLASS Profiler](./media/docs/cpp/profiler.md)
 * Native Turing Tensor Cores
    * Efficient GEMM kernels targeting Turing Tensor Cores
    * Mixed-precision floating point, 8-bit integer, 4-bit integer, and binarized operands
 * Coverage of existing CUTLASS functionality
    * GEMM kernels targeting CUDA and Tensor Cores in NVIDIA GPUs
    * Volta Tensor Cores through native mma.sync and through WMMA API
    * Optimizations such as parallel reductions, threadblock rasterization, and intra-threadblock reductions
    * Batched GEMM operations
    * Complex-valued GEMMs
 * **Note: a host compiler supporting C++11 or greater is required.**

# CUTLASS 1.x

## [1.3.2](https://github.com/NVIDIA/cutlass/releases/tag/v1.3.2) (2019-07-09)
 * Performance improvement for Volta Tensor Cores TN and TT layouts.

## [1.3.1](https://github.com/NVIDIA/cutlass/releases/tag/v1.3.1) (2019-04-09)
 * Corrected NVRTC unit tests.

## [1.3.0](https://github.com/NVIDIA/cutlass/releases/tag/v1.3.0) (2019-03-20)
 * Efficient GEMM kernel targeting Volta Tensor Cores via `mma.sync` instruction added in CUDA 10.1.

## [1.2.0](https://github.com/NVIDIA/cutlass/releases/tag/v1.2.0) (2018-10-26)
 * Parallelized reductions across threadblocks ("Split-K")
   * Improved IGEMM performance
 * Batched strided WMMA GEMMs

## [1.1.0](https://github.com/NVIDIA/cutlass/releases/tag/v1.1.0) (2018-09-19)
  * Turing Features
    * WMMA GEMM targeting TensorCores - INT8, INT4, 1-bit
  * Batched Strided GEMM
  * Threadblock rasterization strategies
    * Improved performance for adverse problem sizes and data layouts
  * Extended CUTLASS Core comonents
    * Tensor views support arbitrary matrix and tensor layouts
    * Zip iterators for structuring multiple data streams
  * Enhanced CUTLASS utilities
    * Reference code for tensor operations in host and device code
    * Added HostMatrix<> for simplified matrix creation
  * Examples
    * Basic GEMM, tensor views, CUTLASS utilities, batched GEMM, WMMA GEMM

## [1.0.1](https://github.com/NVIDIA/cutlass/releases/tag/v1.0.1) (2018-06-11)

  * Intra-threadblock reduction added for small threadblock tile sizes
    * sgemm_64x128x16, sgemm_128x128x16, sgemm_128x64x16, sgemm_128x32x16, sgemm_64x64x16, sgemm_64x32x16
    * igemm_32x32x128
  * GEMM _K_ residue handled during prologue prior to mainloop
  * Replaced Google Test copy with submodule. Use `git submodule init --recursive --update`

## [1.0.0](https://github.com/NVIDIA/cutlass/commit/2028ebe120aab22bfd0b2baf8902d4c9627eb33f) (2018-05-16)

  * Substantial rewrite to accommodate new architecture
  * Kernels: SGEMM, DGEMM, IGEMM, HGEMM, WMMA GEMM
  * Unit and performance tests

## [0.0.1](https://github.com/NVIDIA/cutlass/commit/d08ba8ac46e2fa3f745e070c390182edb56b2e91) (2017-12-04)

  * Initial release


## Copyright

Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
