![ALT](../../images/gemm-hierarchy-with-epilogue-no-labels.png "CUTLASS Profiler")

# CUTLASS Profiler

The CUTLASS Profiler is a command-line driven test and profiling environment for CUTLASS computations
defined in the CUTLASS Instance Library. The CUTLASS Profiler is capable of executing each GEMM, Sparse Gemm,
Conv2d, and Conv3d kernel.

The CUTLASS Profiler may be compiled with:
```bash
$ make cutlass_profiler -j
```

To limit compilation time, only one tile size (typically 128x128) and threadblock cluster size (typically 2x1x1) is instantiated for each data type,
math instruction, and layout. To instantiate all sizes, set the following environment variable when running CMake from an
empty `build/` directory.
```bash
$ cmake .. -DCUTLASS_NVCC_ARCHS="70;75;80" -DCUTLASS_LIBRARY_KERNELS=all  -DCUTLASS_UNITY_BUILD_ENABLED=ON
...
$ make cutlass_profiler -j
```
Enabling the unity build places multiple kernel instances in one compilation unit, thereby reducing size of the compiled
binary and avoiding linker limitations on some platforms.

The CUTLASS Profiler sources are stored in:

```bash
tools/
  profiler/
```

# Emitting kernels via `emit_kernel_listing.py`

We provide a Python script `emit_kernel_listing.py` that allows a user to selectively test a subset of profiler-based kernels stamped out in `generator.py`. A unique benefit to generate kernels and test via this script is that it can feed a series of runtime arguments, such as different `M`/`N`/`K` and `alpha`/`beta`, to each kernel, instead of relying on a single default value. It also properly generates runtime datatype and cluster shapes for certain kernels to help reduce the generated kernel count and accordingly the total compilation time. An interested user may refer to [emit_kernel_listing.py](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/emit_kernel_listing.py) for details. To enable this new feature, a user should add `-DCUTLASS_BUILD_FOR_PROFILER_REGRESSIONS=ON` when building CUTLASS profiler.

## Instantiating more kernels with Hopper
With Hopper (SM90), you will need to use an additional flag,
`CUTLASS_LIBRARY_INSTANTIATION_LEVEL`, in order to instantiate all possible combinations,
which unlike previous architectures, will be in the order of millions of kernels.
Due to this, `CUTLASS_LIBRARY_KERNELS` must be non-empty, since generating and filtering these
kernels alone can take hours.
You must also exercise caution, because not all of these configs are tested, and some may fail to
compile or fail to launch at runtime.

```bash
$ cmake .. \
  -DCUTLASS_NVCC_ARCHS="90a" \
  -DCUTLASS_LIBRARY_KERNELS="cutlass3x_sm90_tensorop_gemm_f16_f16_f32_void_f32_*" \
  -DCUTLASS_LIBRARY_INSTANTIATION_LEVEL="max" \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON
```

The CUTLASS profiler employs a four-digit integer level (global instantiation level) mechanism to manage the generation of kernel configurations. This global instantiation level decides the behavior of multiple "generators" by defining how many and which combinations of configurations are produced. If a global instantiation level contains fewer than four digits, it can be padded with leading zeros to ensure it is four digits long. Each of the four digits in the global level corresponds to a specific category that influences kernel generation, from right to left:

0. **Instruction Shape**
1. **MMA Shape Multiplier**
2. **Cluster Shape**
3. **Schedule Pruning**

Cluster shape levels define the number of CTAs (Cooperative Thread Arrays) included in the kernel generation:

- **Level 0**: Only `(1, 2, 1)` cluster shape.
- **Level 1**: Clusters with 2 CTAs.
- **Level 2**: Clusters with 1 or 2 CTAs.
- **Level 3**: Clusters with 1, 2, or 4 CTAs.
- **Level 4**: Clusters with 1, 2, 4, or 8 CTAs.
- **Level 5**: Clusters with 1, 2, 4, 8, or 16 CTAs.

The MMA multipliers are combined with MMA instruction shapes (WGMMA shapes) to form CTA shapes. The levels for MMA multipliers determine the configurations generated for different data types.
- **Levels [0, 3]**: Control the specific configurations generated for various data types.
- **Level 9**: Activates exhaustive mode, generating all possible configurations.

Higher levels encompass a broader range of CTA configurations, resulting in more comprehensive kernel generation.

Instruction shape levels control the selection of WGMMA shapes used in kernel generation:

- **Level 0**: Generates the "default" shape only.
- **Level 1**: Includes additional shapes for unpruned cases, specifically for TF32 data type.
- **Level 2**: Includes shapes that are powers of 2.
- **Level 3**: Includes all other shapes.

The detailed definition of the three instantiation levels controlling cluster shape, MMA shape multiplier, and instruction shape can be found in [sm90_shapes.py](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/sm90_shapes.py).

Schedule pruning levels decide the epilogue schedule and mainloop schedule to stamp out a kernel instance. As defined in `get_valid_schedules` in [sm90_utils.py](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/sm90_utils.py),

- **Level >= 1**: Indicates that no pruning is being applied.
- **Level 0**: Indicates pruning according to existing [generator.py](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/generator.py) behavior.

An instantiation level `500`, which is padded to `0500`, thus indicates:

- **Instruction Shapes**: At level 0, generating only the "default" shape.
- **MMA Multipliers**: At level 0, generating only one multiplier, `(2, 1, 4)`.
- **Cluster Sizes**: At level 5, allowing for clusters with 1, 2, 4, 8, or 16 CTAs.
- **Schedule Pruning**: At level 0, where pruning is applied according to the existing `generator.py` behavior.

## Instantiating more MMA shapes with Hopper

When instantiating more tile shapes, specially non-power-of-2 Tile-N shapes, make sure to enable `CUTLASS_ENABLE_SM90_EXTENDED_MMA_SHAPES`. 
This may lead to some increase in per-kernel compilation times.
When `CUTLASS_LIBRARY_INSTANTIATION_LEVEL` is set, then `CUTLASS_ENABLE_SM90_EXTENDED_MMA_SHAPES` is enabled by default. 

## Mixed input data type kernels for Hopper

With Hopper (SM90), the kernel generator will generate the following combinations of mixed input data types ("mixed dtype"):

| dtype(A) | dtype(B)   |
| -------- | ---------- |
| e4m3     | f16, bf16  |
| e5m2     | f16, bf16  |
| int8     | f16, bf16  |
| uint8    | f16, bf16  |
| int4     | f16, bf16  |
| int4     | e4m3, e5m2 |
| uint4    | f16, bf16  |
| int2     | f16, bf16  |
| uint2    | f16, bf16  |

For each mixed dtype kernel, the kernel generator will generate combinations of three different running modes:
* Convert-only
* Scale-only
* Scale-with-zero-point-shifting

For {4-bits-dtype, 8-bits-dtype} x 16-bits-dtype, the kernel generator will further generate kernels using shuffled layouts for the narrow data type matrix, which may have a better performance compared to its non-shuffle counter parts.

## Instantiating more kernels with Blackwell
Blackwell (SM100) and Blackwell Ultra similarly support
`CUTLASS_LIBRARY_INSTANTIATION_LEVEL`, in order to instantiate all possible combinations.
Due to this, `CUTLASS_LIBRARY_KERNELS` must be non-empty, since generating and filtering these
kernels alone can take hours.
You must also exercise caution, because not all of these configs are tested, and some may fail to
compile or fail to launch at runtime.

```bash
$ cmake .. \
  -DCUTLASS_NVCC_ARCHS="100f" \
  -DCUTLASS_LIBRARY_KERNELS="cutlass3x_sm100_tensorop_gemm_f16_f16_f32_void_f32_*" \
  -DCUTLASS_LIBRARY_INSTANTIATION_LEVEL="max" \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON
```

The CUTLASS profiler uses the same four-digit integer level (global instantiation level) mechanism to manage the generation of kernel configurations for Blackwell as well:

0. **Instruction Shape**
1. **MMA Shape Multiplier**
2. **Cluster Shape**
3. **Data Type and Schedule Pruning**

Note for Blackwell kernels an MMA shape multiplier is no longer necessary since Blackwell kernels do not have a different
ping pong or cooperative schedule. The profiler ignores this digit when instantiating.

Cluster shape levels define the number of CTAs (Cooperative Thread Arrays) included in the kernel generation:

- **Level 0**: Only dynamic cluster shapes.
- **Level 1**: For 1SM kernels `(1, 1, 1)` and `(2, 1, 1)` for 2SM kernels.
- **Level 2**: For 1SM kernels we also have `(1, 2, 1)` and for 2SM we have `(2, 2, 1)` and `(4, 1, 1)`.
- **Level 3**: For 1SM kernels we have `(1, 4, 1)` and for 2SM we have `(2, 4, 1)` and `(4, 2, 1)`.
- **Level 4**: For 1SM kernels we have `(4, 4, 1)` and for 2SM we have `(4, 4, 1)`.
- **Level 5**: For 1SM kernels we have `(2, 1, 1)`.
- **Level 6**: For 1SM kernels we have `(2, 2, 1)` and `(4, 1, 1)` and for 2SM kernels we have `(8, 1, 1)`.
- **Level 7**: For 1SM kernels we have `(2, 4, 1)` and `(4, 2, 1)`
- **Level 8**: For 1SM kernels we have `(1, 8, 1)` and `(8, 1, 1)`

Instruction shape levels control the selection of MMA shapes used in kernel generation:

- **Level 0**: Generates the "default" shape only.
- **Level 1**: Includes additional shapes for FP8, FP6, and FP4 as well as MX and NVFP4.
- **Level 2**: Includes small tile shapes.
- **Level 3**: Includes some non-power of 2 shapes.
- **Level 4**: Includes further small tile shapes and non-power of 2 shapes.
- **Level 5**: Includes all shapes.

The detailed definition of the three instantiation levels controlling cluster shape and instruction shape can be found in [sm100_shapes.py](https://github.com/NVIDIA/cutlass/tree/main/python/cutlass_library/sm100_shapes.py).

## CUTLASS Profiler usage

The CUTLASS Profiler usage statement may be obtained by executing `cutlass_profiler --help` and appears as follows.
```bash
CUTLASS Performance Tool
usage:

    cutlass_profiler [options]

  --help

  --mode=<string>                                  Cutlass profiler execution mode.
                                                    --mode=profile    regular verification and profiling (default)
                                                    --mode=dry_run    no kernels are launched or workspaces allocated
                                                    --mode=enumerate  lists all operation kind and operations
                                                    --mode=trace      executes a single device-side computation with
                                                                       no other kernel launches

  --device-info                                    Prints information on all GPUs present in the system

  --operation=<operation_kind>                     CUTLASS operation to profile.

  --kernels=<string_list>                          Filter operations by kernel names. For example, call all kernels with
                                                   ("s1688" and "nt") or ("s844" and "tn" and "align8") in their
                                                   operation name using --kernels="s1688*nt, s884*tn*align8"

  --kernels-file=<path>                            Same behavior as `kernels`, but kernel names are specified in a file with
                                                   one kernel name on each line. Set of profiled kernels is the union of kernels
                                                   specified here and those specified in `kernels`.

  --ignore-kernels=<string_list>                   Excludes kernels whose names match anything in this list.

Device:
  --device=<int>                                   CUDA Device ID

  --compute-capability=<int>                       Override the compute capability.

  --llc-capacity=<capacity in KiB>                 Capacity of last-level cache in kilobytes. If this is non-zero,
                                                   profiling phases cycle through different input tensors to induce
                                                   capacity misses in the L2.

  --allocations=<name>:<device>,<name>:<device>    Pairs of allocation names to devices. If <device> is negative,
                                                   the execution device is used


Initialization:
  --initialization=<bool>                          Enables initialization (default: true). If false, device memory is
                                                   not initialized after allocation.

  --initialization-provider=<provider>             Selects initialization provider {host, device*}. (default: '*')

  --dist=<distribution>                            Data distribution of input tensors {uniform*, gaussian, identity, sequential}
                                                    --dist=uniform,min:<double>,max:<double>,scale:<integer>
                                                    --dist=gaussian,mean:<double>,stddev:<double>,scale:<integer>
                                                    --dist=sequential,start:<double>,delta:<double>,scale:<integer>
                                                    --dist=identity

  --seed=<int>                                     Random number generator seed. Used to enforce deterministic
                                                   initialization.


Library:
  --library-algo-mode=<mode>                       Indicates algorithm mode used to call libraries such as cuBLAS and cuDNN.
                                                   mode={default*,matching,best}

  --library-algos=<range-list>                     If --algorithm-mode=best, permits specifying a selection of algorithms.


Profiling:
  --workspace-count=<workspace count>              Number of discrete workspaces maintained to avoid cache-resident
                                                 If zero (default), the amount is chosen for each workload based on
                                                 capacity of the last-level cache.

  --profiling-iterations=<iterations>              Number of iterations to profile each kernel. If zero, kernels
                                                   are launched up to the profiling duration. If non-zero, this
                                                   overrides `profiling-duration` and `min-iterations`.

  --profiling-duration=<duration>                  Time to spend profiling each kernel (ms). Overriden by
                                                   `profiling-iterations` when `profiling-iterations` != 0.
                                                   Note that `min-iterations` must also be satisfied.

  --min-iterations=<iterations>                    Minimum number of iterations to spend profiling each kernel, even if
                                                   `profiling-duration` has been met.

  --warmup-iterations=<iterations>                 Number of iterations to execute each kernel prior to profiling (default: 10).

  --use-cuda-graphs=<bool>                         If true, kernels are launched in a CUDA graph. Useful when the kernel launch time is a bottleneck.

  --sleep-duration=<duration>                      Number of ms to sleep between profiling periods (ms).

  --profiling-enabled=<bool>                       If true, profiling is actually conducted.

Verification:
  --verification-enabled=<bool>                    Whether to perform verification checks.

  --epsilon=<error>                                Error threshold. Setting to zero (default) requires
                                                   bit-level equivalence.

  --nonzero-floor=<floor>                          Results whose absolute value is less than this quantity
                                                   are treated as zero for comparisons.

  --save-workspace=<string>                        Specifies when to save the GEMM inputs and results to the filesystem.
                                                    --save-workspace=never      never save workspace (default)
                                                    --save-workspace=incorrect  save workspace for incorrect results
                                                    --save-workspace=always     always save workspace

  --verification-providers=<providers>             List of providers used to verify result. (default: '*')
                                                   Gemm verification-providers {cublas*}
                                                   Conv2d verification-providers {cudnn*, device*, host}


Report:
  --append=<bool>                                  If true, result is appended to possibly existing file. Otherwise,
                                                   any existing file is overwritten.

  --output=<path>                                  Path to output file for machine readable results. Operation kind and '.csv' is appended.

  --junit-output=<path>                            Path to junit output file for result reporting. Operation kind and '.junit.xml' is appended.

  --report-not-run=<bool>                          If true, reports the status of all kernels including those that
                                                   do not satisfy the given arguments.

  --tags=<column:tag,...>                          Inserts leading columns in output table and uniform values for each
                                                   column. Useful for generating pivot tables.

  --verbose=<bool>                                 Prints human-readable text to stdout. If false, nothing is written to stdout.


About:
  --version                                        CUTLASS 2.4.0 built on Nov 19 2020 at 11:59:00


Operations:

     gemm                                          General matrix-matrix product. D = alpha * A*B + beta * C
     spgemm                                        Structured sparse GEMM. D = alpha * A*B + beta * C
     conv2d                                        Conv2d operation. Output(Tensor4D) = alpha * Input(Tensor4D) * Filter(Tensor4D) + beta * Input(Tensor4D)
     conv3d                                        Conv3d operation. Output(Tensor5D) = alpha * Input(Tensor5D) * Filter(Tensor5D) + beta * Input(Tensor5D)


For details about a particular function, specify the function name with --help.

Example:

  $ cutlass_profiler --operation=Gemm --help

  $ cutlass_profiler --operation=Conv3d --help

  $ cutlass_profiler --operation=Conv2d --help

```

# GEMM

The CUTLASS Profiler is capable of executing GEMM and Sparse GEMM problems.

The CUTLASS Profiler can be built with cuBLAS enabled to use as a reference implementation. If CMake detects
the cuBLAS library available in the system, it is included as a dependency. This may be explicitly overridden
with CMake flag `CUTLASS_ENABLE_CUBLAS`.

## GEMM Arguments

The complete set of arguments available to each operation may be viewed by specifying the operation name
in addition to `--help`. The argument flags and their aliases usable for GEMM appear as follows.

```bash
$ ./tools/profiler/cutlass_profiler --operation=gemm --help

GEMM

  [enum]      --gemm_kind                                       Variant of GEMM (e.g. universal, gemm, planar_complex, planar_complex_array)
  [int]       --m,--problem-size::m                             M dimension of the GEMM problem space
  [int]       --n,--problem-size::n                             N dimension of the GEMM problem space
  [int]       --k,--problem-size::k                             K dimension of the GEMM problem space
  [tensor]    --A                                               Tensor storing the A operand
  [tensor]    --B                                               Tensor storing the B operand
  [tensor]    --C                                               Tensor storing the C operand
  [scalar]    --alpha,--epilogue::alpha                         Epilogue scalar alpha
  [scalar]    --beta,--epilogue::beta                           Epilogue scalar beta
  [enum]      --split_k_mode,--split-k-mode                     Variant of split K mode(serial, parallel)
  [int]       --split_k_slices,--split-k-slices                 Number of partitions of K dimension
  [int]       --batch_count,--batch-count                       Number of GEMMs computed in one batch
  [enum]      --op_class,--opcode-class                         Class of math instruction (simt, tensorop, wmmatensorop, wmma).
  [enum]      --accum,--accumulator-type                        Math instruction accumulator data type
  [int]       --cta_m,--threadblock-shape::m                    Threadblock shape in the M dimension
  [int]       --cta_n,--threadblock-shape::n                    Threadblock shape in the N dimension
  [int]       --cta_k,--threadblock-shape::k                    Threadblock shape in the K dimension
  [int]       --cluster_m,--cluster-shape::m                    Cluster shape in the M dimension
  [int]       --cluster_n,--cluster-shape::n                    Cluster shape in the N dimension
  [int]       --cluster_k,--cluster-shape::k                    Cluster shape in the K dimension
  [int]       --cluster_m_fallback,--cluster-shape-fallback::m  Fallback cluster shape in the M dimension
  [int]       --cluster_n_fallback,--cluster-shape-fallback::n  Fallback cluster shape in the N dimension
  [int]       --cluster_k_fallback,--cluster-shape-fallback::k  Fallback cluster shape in the K dimension
  [int]       --stages,--threadblock-stages                     Number of stages of threadblock-scoped matrix multiply
  [int]       --warps_m,--warp-count::m                         Number of warps within threadblock along the M dimension
  [int]       --warps_n,--warp-count::n                         Number of warps within threadblock along the N dimension
  [int]       --warps_k,--warp-count::k                         Number of warps within threadblock along the K dimension
  [int]       --inst_m,--instruction-shape::m                   Math instruction shape in the M dimension
  [int]       --inst_n,--instruction-shape::n                   Math instruction shape in the N dimension
  [int]       --inst_k,--instruction-shape::k                   Math instruction shape in the K dimension
  [int]       --min_cc,--minimum-compute-capability             Minimum device compute capability
  [int]       --max_cc,--maximum-compute-capability             Maximum device compute capability
  [enum]      --raster_order={heuristic|H|along_m|M|along_n|N}  If supported by kernel, sets the tile raster direction
  [int]       --swizzle_size={1,2,4,8}                          If supported by kernel, sets the 2D tile swizzle extent (In Hopper, other values will be rounded down to the nearest supported value)
  [int]       --use_pdl,--use-pdl                               Use PDL (true, false)
  [int]       --enable_sm90_mixed_dtype_shuffle_test            If true, the profiler will test SM90 mixed input kernels that can use shuffled input layouts for better performance
  [enum]      --runtime_input_datatype_a                        Runtime data type for A matrix, narrow-precision only (e4m3, e5m2, e3m2, e2m3, e2m1)
  [enum]      --runtime_input_datatype_b                        Runtime data type for B matrix, narrow-precision only (e4m3, e5m2, e3m2, e2m3, e2m1)

Examples:

Profile a particular problem size:
  $ cutlass_profiler --operation=Gemm --m=1024 --n=1024 --k=128

Schmoo over problem size and beta:
  $ cutlass_profiler --operation=Gemm --m=1024:4096:256 --n=1024:4096:256 --k=128:8192:128 --beta=0,1,2.5

Schmoo over accumulator types:
  $ cutlass_profiler --operation=Gemm --accumulator-type=f16,f32

Run when A is f16 with column-major and B is any datatype with row-major (For column major, use column, col, or n. For row major use, row or t):
  $ cutlass_profiler --operation=Gemm --A=f16:column --B=*:row

Using various input value distribution:
  $ cutlass_profiler --operation=Gemm --dist=uniform,min:0,max:3
  $ cutlass_profiler --operation=Gemm --dist=gaussian,mean:0,stddev:3
  $ cutlass_profiler --operation=Gemm --dist=sequential,start:0,delta:1

Using CUTLASS 3.x GEMM kernel with a tile scheduler that supports runtime tile remapping and raster mode order:
  $ cutlass_profiler --operation=Gemm --m=2048 --n=2048 --k=2048 --raster_order=M --swizzle_size=2

Run a kernel with cta tile size of 256x128x32 and save workspace if results are incorrect (note that --cta-tile::k=32 is default cta-tile size):
 $ cutlass_profiler --operation=Gemm --cta_m=256 --cta_n=128  --cta_k=32 --save-workspace=incorrect

Test your changes to gemm kernels with a quick functional test and save results in functional-test.csv:
 $ cutlass_profiler  --operation=Gemm \
   --m=8,56,120,136,256,264,512,520,1024,1032,4096,8192,16384 \
   --n=8,56,120,136,256,264,512,520,1024,1032,4096,8192,16384 \
   --k=8,16,32,64,128,256,288,384,504,512,520 \
   --beta=0,1,2 --profiling-iterations=1 \
   --providers=cutlass --output=functional-test.csv

Profile when execution is performed on device 0 and the C tensor is located on a device 1 and D on device 2:
  $ cutlass_profiler --device=0 --allocations=C:1,D:2 --operation=Gemm --m=1024 --n=1024 --k=128
```

The format of tensor argument is followed by `<type>:<layout>`. The type could be `f32` as 32-bit floating point, `s8` as 8-bit signed integer, etc. The available types can be referred to the `NumericTypeID_enumerants` in [util.cu](https://github.com/NVIDIA/cutlass/tree/main/tools/library/src/util.cu). The layout could be `row` or `column`. If `--enable_sm90_mixed_dtype_shuffle_test=true` is used, the actual layout of the narrow data type matrix is a shuffled layout, neither `row` nor `column`.

In addition to encoded data types, CUTLASS profiler allows non-encoded generic data types, namely `f8`, `f6`, and `f4`, with corresponding encoding specified through GEMM input argument: `--runtime_input_datatype_a` and `--runtime_input_datatype_b`. Currently, six encoding schemes are supported: `e4m3`, `e5m2`, `e3m2`, `e2m3`, and `e2m1`.

Cluster shapes can be statically set to `Shape<int,int,_1>;` and specified via runtime arguments: `cluster_m`, `cluster_n` and `cluster_k` in CUTLASS profiler.  In addition to preferred cluster shapes, a user can also specify fallback cluster shapes via runtime arguments: `cluster_m_fallback`, `cluster_n_fallback` and `cluster_k_fallback` in CUTLASS profiler. Those fallback cluster shapes are smaller shapes than the preferred ones for the hardware to assign when there is no chance to issue a larger preferred CGA cluster to the GPU. There are several rules for using a flexible CGA: 1) Preferred CGA size should be divisible by fallback CGA size. 2) Grid dim should be divisible by preferred CGA size. 3) Preferred CGA and fallback CGA must have the same depth (cluster_dim.z must be equal). One may refer to our CUTLASS Example [73_blackwell_gemm_flexible_cluster](https://github.com/NVIDIA/cutlass/tree/main/examples/73_blackwell_gemm_preferred_cluster/blackwell_gemm_preferred_cluster.cu) for more details of the this feature. 
Please be noted that this feature (flexible cluster shapes within a single grid) is only applicable to `sm100a` kernels. The hardware will rasterize into a single cluster shape for those kernels that do not support this feature even with preferred or fallback cluster shapes assigned.

CUTLASS 3.x kernels for Hopper and Blackwell also support a new feature called programatic dependent launch (PDL). This can be enabled with `--use-pdl`, and can overlap the epilogue of the prior kernel with the prologue of the next kernel. This can effectively hide kernel prologues. Using PDL can improve performance for back to back GEMMs. See [dependent kernel launch](dependent_kernel_launch.md) for more information. CUDA graphs can also be used (`--use-cuda-graphs`) with PDL to ensure that smaller kernels are enqueued back-to-back on a stream.

## Exhaustive search mode and top-k output ranking according to performance in GFLOPS/s

CUTLASS also allows a few options to enable searching best performing kernel in a broader parameter space.

1. **Sorting Performance Results by GFLOPs/second**  
   A new option enables users to sort the final performance report based on GFLOPs/second, making it easier to identify the most efficient kernels.

2. **Exhaustive Search for Best Kernel Performance in GFLOPs/second**  
   This feature allows the profiler to search for the best-performing kernel across a range of problem sizes, swizzle sizes, rasterization orders, and dynamic cluster configurations. It ensures that all viable configurations are considered to maximize performance.

3. **Performance Search Under a Fixed GEMM Shape**  
   This option enables exhaustive performance tuning for a specific problem size. Unlike the previous feature, this restricts the search to a fixed GEMM shape while still exploring various kernel parameters to find the best configuration.

### Usage Examples

#### 1. Finding the Best Performing Kernel

Use the following command to conduct an exhaustive search and sort results by GFLOPs/second:

```bash
cutlass_profiler --kernels=*gemm* --enable-kernel-performance-search --sort-results-flops-per-sec
```

#### 2. Performance Optimization for a Fixed GEMM Shape

To optimize kernel performance for a specific GEMM problem size:

```bash
cutlass_profiler --kernels=*gemm* --enable-best-kernel-for-fixed-shape --m=6144 --n=6144 --k=6144 --sort-results-flops-per-sec
```

To search optimized kernel performance for a series of GEMM shapes (m, n, k = 1024, 2048):

```bash
cutlass_profiler --kernels=*gemm* --enable-best-kernel-for-fixed-shape --m=1024,2048 --n=1024,2048 --k=1024,2048 --sort-results-flops-per-sec
```

It is worth noting that by enabling exhaustive performance search via `--enable-kernel-performance-search`, a user is still able and responsible to decide parameters like data distribution in argument list, for which a user can choose `--dist=uniform,min:-1,max:1,scale:-1` to initialize a tensor with floating point numbers in uniform distribution. Otherwise, those parameters will be initialized to their default values.

For examples above, one can change the kernel filtering regex according to their own use cases.

## Example CUDA Core GEMM Operation

Example command line for profiling SGEMM kernels is as follows:
```bash
$ ./tools/profiler/cutlass_profiler --kernels=sgemm --m=3456 --n=4096 --k=4096



=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass_simt_sgemm_128x128_8x2_nn_align1

          Status: Success
    Verification: ON
     Disposition: Passed

          cuBLAS: Passed

       Arguments: --m=3456 --n=4096 --k=4096 --A=f32:column --B=f32:column --C=f32:column --alpha=1 --beta=0 --split_k_slices=1  \
                  --batch_count=1 --op_class=simt --accum=f32 --cta_m=128 --cta_n=128 --cta_k=8 --stages=2 --warps_m=4  \
                  --warps_n=2 --warps_k=1 --inst_m=1 --inst_n=1 --inst_k=1 --min_cc=50 --max_cc=1024

           Bytes: 180355072  bytes
           FLOPs: 115992428544  flops

         Runtime: 6.73655  ms
          Memory: 24.934 GiB/s

            Math: 17218.4 GFLOP/s
```

Note, the arguments which appear in the output may be used as command line parameters for subsequent invocations.


## Example Tensor Core GEMM Operations

To execute kernels targeting Tensor Core operations, supply the flag `--op_class=tensorop` in the command line.
```bash
$ ./tools/profiler/cutlass_profiler --op_class=tensorop --m=3456 --n=4096 --k=8192



=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass_tensorop_s16816gemm_f16_256x128_32x3_nn_align8

          Status: Success
    Verification: ON
     Disposition: Passed

          cuBLAS: Passed

       Arguments: --m=3456 --n=4096 --k=8192 --A=f16:column --B=f16:column --C=f32:column --alpha=1 --beta=0 --split_k_slices=1  \
                  --batch_count=1 --op_class=tensorop --accum=f32 --cta_m=256 --cta_n=128 --cta_k=32 --stages=3 --warps_m=4  \
                  --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=16 --min_cc=80 --max_cc=1024

           Bytes: 180355072  bytes
           FLOPs: 231956545536  flops

         Runtime: 0.98647  ms
          Memory: 170.272 GiB/s

            Math: 235138 GFLOP/s
```

## Covering the problem space

All arguments may have single values or comma-delimited set of values. Integers may also be specified
as an inclusive range with the following syntax `start:end:increment` or simply `start:end`.

For example, the following sweeps over the range of the GEMM K dimension from 8 to 4096 in increments
of 8 elements.

```bash
$ ./tools/profiler/cutlass_profiler --kernels=cutlass_simt_sgemm_128x128_nn --m=4352 --n=4096 --k=8:4096:8
```

## Output

By default, runtime and computed GFLOP/s are reported for each operation and problem size. Additionally,
a table of comma separated values are reported at the end of the execution. This may be output to a file
with the `--output=<filename.csv>` command line option as shown:

```bash
$ ./tools/profiler/cutlass_profiler --kernels=cutlass_simt_sgemm_128x128_nn            \
                                    --m=3456 --n=4096 --k=8:4096:8 --output=report.csv
```

To faclitate generation of pivot tables and charts, additional columns may be prepended with the
`--tags=<column>:<value>` option. One or more tags may be specified using a comma-delimited list.

```bash
$ ./tools/profiler/cutlass_profiler --kernels=cutlass_simt_sgemm_128x128_nn            \
                                    --m=3456 --n=4096 --k=8:4096:8 --output=report.csv \
                                    --tags=cutlass:2.2,date:2020-06-08
```

## CUTLASS 3.0 GEMM procedural names

CUTLASS 3.0 introduces a new naming convention for GEMMs used by the profiler targeting the NVIDIA
Hopper architecture and beyond so as to indicate new features of the kernel within the name
(e.g., the cluster shape).

To best illustrate this naming convention, we will walk through the meaning of each of the components
in a GEMM kernel used by the profiler:

```
cutlass3x_sm90_tensorop_gemm_f16_f16_f32_f16_f32_{optional-mixed-dtype-config}_128x128x64_2x1x1_0_ntn_align8
```

The components within this name are as follows:

* `cutlass3x`: indicates that the kernel was generated through the CUTLASS 3.0 API
* `sm90`: indicates that the kernel targets NVIDIA GPUs with compute capability 90
* `tensorop`: indicates that the kernel makes use of NVIDIA Tensor Cores
(as opposed to `simt`, which indicates the use of "CUDA cores")
* `s`: indicates that the Tensor Core instruction being used accumulates in single precision
(as opposed to `h`, which indicates half precision)
* `64x128x16gemm`: indicates that the shape of the Tensor Core instruction being used (MxNxK) is 64x128x16
* `f16_f16_f32_f16_f16`: indicates that the data types for operands A, B, Accumulator, C and D (in that order).
* `optional-mixed-dtype-config`: optional, will be empty if this is not a mixed dtype kernel. For mixed dtype kernels, it contains `_cvt`, `_scl`, `_sclzr`, respectively, for convert-only, scale-only, scale-with-zero-point running modes. It further contains `_shfl` if the kernel uses a shuffled layout for the narrow data type input matrix.
* `128x128x64`: indicates that the thread block shape used in the GEMM (MxNxK) is 128x128x64
* `2x1x1`: indicates that the cluster shape being used is 2x1x1
* `0`: indicates that the kernel uses the CollectiveBuilder's automatic stage calculation to determine the
number of pipeline stages in the kernel. Note that `0` does not mean that no stages are used. A nonzero value indicates that automatic stage calculation is not performed and indicates the number of pipeline stages to be used.
This 0 is only added to the kernel's procedural name, the profiler will still report the actual stage count
when printing the kernel argument details (`--stages=N`) and kernel discovery will still support filtering through the `--stages` argument.
* `ntn`: indicates that the layouts for operands A, B, and C are column major ("n"; non-transposed),
row major ("t"; transposed), and column major, respectively.
* `align8`: indicates that the maximum alignment between operands A and B is 8.

Note that in some special cases where the input A/B types do not match that of the MMA
instruction's, the MMA facing input type is added to the instruction string as well.

```
cutlass3x_sm90_tensorop_tf32gemm_f32_f32_f32_f32_f32_128x128x32_2x1x1_0_tnn_align4
```

* `s64x128x8tf32gemm`: indicates that the MMA consumes inputs in `tf32` format, and therefore
the kernel performs rounding of the `f32` values in global memory while loading them into shared memory.

For custom mainloop or epilogue schedules, details of the opted-in schedule are appended to the end of the
kernel name. For example,

```
cutlass3x_sm90_tensorop_gemm_f16_f16_f16_void_f16_128x128x64_1x1x1_0_nnn_align8_warpspecialized_cooperative_epi_tma
```

* `warpspecialized_cooperative`: Mainloop employs a persistent warp-specialized mainloop and kernel schedule.
* `epi_tma`: Kernel epilogue employs TMA based vectorization.
* `f16_f16_f16_void_f16`: In this case, C type is set to `void`, indicating that residual matrix support
is disabled.

## Further Documentation

For documentation on profiling blockwise and groupwise (software scaled) GEMMs see the [example 81 README](https://github.com/NVIDIA/cutlass/blob/main/examples/81_blackwell_gemm_blockwise/README.md).

# Convolution

The CUTLASS Profiler is capable of executing 2-D and 3-D convolution problems for forwards and backwards
operator variants.

The CUTLASS Profiler can be built with cuDNN enabled to use as a reference implementation. If CMake detects
the cuDNN library available in the system, it is included as a dependency. This may be explicitly overridden
with CMake flag `CUTLASS_ENABLE_CUDNN`.

```bash
$ cmake .. -DCUTLASS_LIBRARY_OPERATIONS=conv2d -DCUTLASS_ENABLE_CUDNN=OFF
...
$ make -j16 cutlass_profiler
```


## Convolution Arguments

```bash
$ ./tools/profiler/cutlass_profiler --help --operation=Conv2d

Conv2d

  [enum]      --conv_kind                                       Convolutional operator (fprop, dgrad, wgrad)
  [int]       --n,--input_n                                     Input N dimension of the Conv2d problem space
  [int]       --h,--input_h                                     Input H dimension of the Conv2d problem space
  [int]       --w,--input_w                                     Input W dimension of the Conv2d problem space
  [int]       --c,--input_c                                     Input C dimension of the Conv2d problem space
  [int]       --k,--filter_k                                    Filter K dimension of the Conv2d problem space
  [int]       --r,--filter_r                                    Filter R dimension of the Conv2d problem space
  [int]       --s,--filter_s                                    Filter S dimension of the Conv2d problem space
  [int]       --p,--output_p                                    Output P dimension of the Conv2d problem space
  [int]       --q,--output_q                                    Output Q dimension of the Conv2d problem space
  [int]       --g,--groups                                      Number of convolution groups
  [int]       --pad_h                                           Padding in H direction
  [int]       --pad_w                                           Padding in W direction
  [int]       --stride_h                                        Stride in H direction
  [int]       --stride_w                                        Stride in W direction
  [int]       --dilation_h                                      Dilation in H direction
  [int]       --dilation_w                                      Dilation in W direction
  [tensor]    --Activation                                      Tensor storing the Activation operand
  [tensor]    --Filter                                          Tensor storing the Filter operand
  [tensor]    --Output                                          Tensor storing the Output operand
  [enum]      --conv_mode                                       Convolution filter mode (conv, cross)
  [enum]      --iterator_algorithm,--iterator_algo              Convolution iterator algorithm (analytic, optimized)
  [scalar]    --alpha,--epilogue::alpha                         Epilogue scalar alpha
  [scalar]    --beta,--epilogue::beta                           Epilogue scalar beta
  [enum]      --split_k_mode,--split-k-mode                     SplitK mode for serial or parallel reduction (serial, parallel)
  [int]       --split_k_slices,--split-k-slices                 Number of partitions of K dimension
  [enum]      --eq_gemm_provider,--eq-gemm-provider             Enable profiling equivalent gemm by the following providers (cutlass)
  [enum]      --op_class,--opcode-class                         Class of math instruction (simt, tensorop, wmmatensorop, wmma)
  [enum]      --accum,--accumulator-type                        Math instruction accumulator data type
  [int]       --cta_m,--threadblock-shape::m                    Threadblock shape in the M dimension
  [int]       --cta_n,--threadblock-shape::n                    Threadblock shape in the N dimension
  [int]       --cta_k,--threadblock-shape::k                    Threadblock shape in the K dimension
  [int]       --cluster_m,--cluster-shape::m                    Cluster shape in the M dimension
  [int]       --cluster_n,--cluster-shape::n                    Cluster shape in the N dimension
  [int]       --cluster_k,--cluster-shape::k                    Cluster shape in the K dimension
  [int]       --cluster_m_fallback,--cluster-shape-fallback::m  Fallback cluster shape in the M dimension
  [int]       --cluster_n_fallback,--cluster-shape-fallback::n  Fallback cluster shape in the N dimension
  [int]       --cluster_k_fallback,--cluster-shape-fallback::k  Fallback cluster shape in the K dimension
  [int]       --stages,--threadblock-stages                     Number of stages of threadblock-scoped matrix multiply
  [int]       --warps_m,--warp-count::m                         Number of warps within threadblock along the M dimension
  [int]       --warps_n,--warp-count::n                         Number of warps within threadblock along the N dimension
  [int]       --warps_k,--warp-count::k                         Number of warps within threadblock along the K dimension
  [int]       --inst_m,--instruction-shape::m                   Math instruction shape in the M dimension
  [int]       --inst_n,--instruction-shape::n                   Math instruction shape in the N dimension
  [int]       --inst_k,--instruction-shape::k                   Math instruction shape in the K dimension
  [int]       --min_cc,--minimum-compute-capability             Minimum device compute capability
  [int]       --max_cc,--maximum-compute-capability             Maximum device compute capability

Examples:

Profile a particular convolution (specify all the convolution parameters):
 $ cutlass_profiler --operation=Conv2d --Activation=f16:nhwc --Filter=f16:nhwc --Output=f16 --accumulator-type=f32 --n=32 --h=14 --w=14 --c=8 --k=64 --r=3 --s=3 --pad_h=1 --pad_w=1 --stride_h=1 --stride_w=1 --dilation_h=1 --dilation_w=1

```

## Example CUDA Core Convolution Operation

Example command line for profiling forward propagation convolution kernels on CUDA cores is as follows:
```bash
$ ./tools/profiler/cutlass_profiler --kernels=simt_sfprop  --verification-providers=device --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3


=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: conv2d
       Operation: cutlass_simt_sfprop_optimized_128x128_8x2_nhwc

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed

       Arguments: --conv_kind=fprop --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3 --p=224 --q=224 --pad_h=1 --pad_w=1  \
                  --stride_h=1 --stride_w=1 --dilation_h=1 --dilation_w=1 --Activation=f32:nhwc --Filter=f32:nhwc --Output=f32:nhwc  \
                  --conv_mode=cross --iterator_algorithm=optimized --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1  \
                  --eq_gemm_provider=none --op_class=simt --accum=f32 --cta_m=128 --cta_n=128 --cta_k=8 --stages=2 --warps_m=4  \
                  --warps_n=2 --warps_k=1 --inst_m=1 --inst_n=1 --inst_k=1 --min_cc=50 --max_cc=1024

           Bytes: 2055798784  bytes
           FLOPs: 118482796544  flops

         Runtime: 8.13237  ms
          Memory: 235.431 GiB/s

            Math: 14569.3 GFLOP/s

```

## Example Tensor Core Convolution Operation

Example command line for profiling forward propagation convolution kernels runing on Tensor Cores is as follows:
```bash
$ ./tools/profiler/cutlass_profiler --kernels=tensorop*fprop  --verification-providers=device --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3



=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: conv2d
       Operation: cutlass_tensorop_s16816fprop_optimized_f16_128x128_64x4_nhwc

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed

       Arguments: --conv_kind=fprop --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3 --p=224 --q=224 --pad_h=1 --pad_w=1  \
                  --stride_h=1 --stride_w=1 --dilation_h=1 --dilation_w=1 --Activation=f16:nhwc --Filter=f16:nhwc --Output=f32:nhwc  \
                  --conv_mode=cross --iterator_algorithm=optimized --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1  \
                  --eq_gemm_provider=none --op_class=tensorop --accum=f32 --cta_m=128 --cta_n=128 --cta_k=64 --stages=4  \
                  --warps_m=2 --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=16 --min_cc=80 --max_cc=1024

           Bytes: 1130659840  bytes
           FLOPs: 118482796544  flops

         Runtime: 0.945071  ms
          Memory: 1114.21 GiB/s

            Math: 125369 GFLOP/s


```

# Copyright

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
