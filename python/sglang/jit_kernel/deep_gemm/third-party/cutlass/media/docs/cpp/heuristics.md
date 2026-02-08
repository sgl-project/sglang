
# GEMM Heuristics

## Overview

Gemm heuristics in `cutlass_library` aim to reduce the search space for runtime autotuning, so that only a subset of valid kernels need to be built and profiled for a given set of GEMM problems. This implementation uses Nvidia's `nvidia-matmul-heuristics`, an analytical heuristic that ranks GEMM kernels by estimated performance given a problem size and hardware SKU. You can find more info in [the docs](https://docs.nvidia.com/cuda/nvidia-matmul-heuristics).

## Coverage

Gemm heuristics in `cutlass_library` is an experimental feature and exhaustive functional or performance coverage is not guaranteed. It currently supports the following.

Problem space:
- Plain dense gemm for `f8`, `f16`, `f32`

Hardware:
- Hopper (sm9x)
- Blackwell (sm10x)

## Usage / Quick Start

### Install Dependencies

Using the wheel is recommended:
```
pip install nvidia-matmul-heuristics
```

### Prepare Input File

Prepare a list of gemm problem definitions, in the form of a json list, to be evaluated by the heuristic. Here is a sample file with two problems:
```
[
{
     "m" : 4096,
     "n" : 4096,
     "k" : 4096,
     "batch_count" : 1,
     "layout" : "tnn",
     "dtype_a" : "f16",
     "dtype_b" : "f16",
     "dtype_c" : "f16",
     "dtype_acc" : "f32",
     "dtype_d" : "f16",
     "beta" : 0.0,
     "use_fast_acc": false
},
{
     "m" : 4096,
     "n" : 4096,
     "k" : 4096,
     "batch_count" : 1,
     "layout": "tnn",
     "dtype_a" : "e5m2",
     "dtype_b" : "e5m2",
     "dtype_c" : "f32",
     "dtype_acc" : "f32",
     "dtype_d" : "e5m2",
     "beta" : 0.0,
     "use_fast_acc": true
}
]
```

Note: `use_fast_acc` only needs to be specified for FP8 kernels on SM90. Otherwise, it is ignored.

### Build

Build CUTLASS using CMake as normal, providing heuristics-specific options to CMake. Note that hardware details are detected automatically. For offline builds, use `-DCUTLASS_LIBRARY_HEURISTICS_GPU`.
For example, here is a minimal command for Nvidia's Hopper Architecture (sm90):

```
$ cmake .. \
    -DCUTLASS_NVCC_ARCHS=90a \
    -DCUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE=<path_to_your_problem_list.json> \
    -DCUTLASS_LIBRARY_HEURISTICS_CONFIGS_PER_PROBLEM=<number of configurations to build per problem> 
...
...

$ make cutlass_profiler -j

```

This will produce a csv testlist which provides all testcases that need be run to perform autotuning over the built configurations, including kernel runtime options. The location of this file can be changed by the CMake option `-DCUTLASS_LIBRARY_HEURISTICS_TESTLIST_FILE`.

CUTLASS CMake currently supports the following for heuristics:
- `CUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE`: Path to the file containing a json list of GEMM problems
- `CUTLASS_LIBRARY_HEURISTICS_CONFIGS_PER_PROBLEM`: Max number of configurations the heuristic will return for each GEMM problem. The same configuration or kernel can be suggested for multiple problems.
- `CUTLASS_LIBRARY_HEURISTICS_RESTRICT_KERNELS`: Limits the build to only the set of kernels instantiated by the default CUTLASS CMake build flow, composing with other options such as `CUTLASS_LIBRARY_INSTANTIATION_LEVEL`. Set this to `ON` as a workaround if the heuristic suggests kernel configurations that do not build on your platform (possible for some unsupported or experimental use cases). This option is set to `OFF` by default, which builds all of the suggested configurations.
- `CUTLASS_LIBRARY_HEURISTICS_TESTLIST_FILE`: Path to the output CSV which will contain the testcases to be used for autotuning, consumable by `cutlass_profiler`.
- `CUTLASS_LIBRARY_HEURISTICS_GPU`: The GPU to use for heuristics; for instance, `H100_SXM5`. Used for offline builds. If unset, the hardware properties will be auto-detected using the Cuda Driver APIs. See `generator.py` for valid GPU strings

### Profile

Use the emitted testlist CSV with `cutlass_profiler` to collect performance data, which can be used to determine the fastest built kernel configuration for each of the input problems. Example which profiles each testcase for a fixed 50ms:
```
cutlass_profiler --operation=Gemm --testlist-file=<path_to_your_testlist.csv> --profiling-iterations=0 --profiling-duration=50 --verification-enabled=false --output=<path_to_outfile>
```

## Direct Usage in Python

If you have pre-built CUTLASS kernels or custom CUTLASS emitters, you can use the Python APIs directly to select kernels to build or profile. See `filter_manifest_and_write_heuristics_file()` in `heuristics.py` for example usage.

