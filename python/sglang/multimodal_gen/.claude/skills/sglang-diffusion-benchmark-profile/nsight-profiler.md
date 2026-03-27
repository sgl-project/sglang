---
name: nsight-profiler
description: Expert skill for NVIDIA Nsight Systems and Nsight Compute profiling tools. Configure profiling sessions, analyze kernel reports, interpret occupancy metrics, roofline model data, memory bandwidth bottlenecks, and warp execution efficiency.
allowed-tools: Bash(*) Read Write Edit Glob Grep WebFetch
metadata:
  author: babysitter-sdk
  version: "1.0.0"
  category: performance-profiling
  backlog-id: SK-002
  source: "Adapted from https://github.com/lobehub/lobehub (.agents/skills/nsight-profiler)"
---

> **Source**: This skill is adapted from the [lobehub/lobehub](https://github.com/lobehub/lobehub) open-source repository (`.agents/skills/nsight-profiler`). Original author: `babysitter-sdk`.

# nsight-profiler

You are **nsight-profiler** - a specialized skill for NVIDIA Nsight Systems and Nsight Compute profiling tools. This skill provides expert capabilities for performance analysis and optimization of GPU applications.

## Overview

This skill enables AI-powered GPU profiling operations including:
- Configure and execute Nsight Systems profiling sessions
- Analyze Nsight Compute kernel reports
- Interpret occupancy metrics and SM utilization
- Parse and visualize roofline model data
- Identify memory bandwidth bottlenecks
- Analyze warp execution efficiency
- Generate optimization recommendations from profiler data
- Compare kernel performance across different configurations

## Prerequisites

- NVIDIA Nsight Systems 2023.1+
- NVIDIA Nsight Compute 2023.1+
- CUDA Toolkit 11.0+
- GPU with compute capability 7.0+ (for full profiling features)

## Capabilities

### 1. Nsight Systems Profiling

System-wide performance analysis:

```bash
# Basic system profile
nsys profile -o report ./cuda_program

# Profile with CUDA API tracing
nsys profile -t cuda,nvtx,osrt -o report ./cuda_program

# Capture GPU metrics
nsys profile --gpu-metrics-device=all -o report ./cuda_program

# Profile specific duration
nsys profile -d 10 -o report ./cuda_program

# Export to multiple formats (one type per command)
nsys export -t sqlite report.nsys-rep
nsys export -t json report.nsys-rep

# Generate summary statistics
nsys stats report.nsys-rep
```

Child-process note:
- If the target launches worker processes, add `--trace-fork-before-exec=true`.
- For `sglang generate`, this is usually required to capture the real worker trace.

### 2. Nsight Compute Profiling

Detailed kernel analysis:

```bash
# Profile all kernels
ncu --target-processes all -o profile ./cuda_program

# Profile specific kernel
ncu --target-processes all --kernel-name myKernel -o profile ./cuda_program

# Full metric collection
ncu --target-processes all --set full -o profile ./cuda_program

# Roofline analysis
ncu --target-processes all --set roofline -o profile ./cuda_program

# Memory analysis
ncu --target-processes all --section MemoryWorkloadAnalysis -o profile ./cuda_program

# Import an existing report
ncu --import profile.ncu-rep --page details --csv
```

Child-process note:
- If the target forks or spawns subprocesses, use `--target-processes all`.
- For `sglang generate`, this is the safer default.

### 3. Occupancy Analysis

Analyze and optimize occupancy:

```bash
# Collect occupancy metrics
ncu --section Occupancy -o occupancy ./cuda_program

# Key metrics to analyze:
# - Achieved Occupancy
# - Theoretical Occupancy
# - Block Limit (registers, shared memory, warps)
# - Occupancy Limiter
```

```cuda
// Query occupancy in code
int numBlocks;
int blockSize = 256;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocks, myKernel, blockSize, sharedMemSize);

float occupancy = (numBlocks * blockSize) /
    (float)deviceProp.maxThreadsPerMultiProcessor;
printf("Theoretical Occupancy: %.2f%%\n", occupancy * 100);
```

### 4. Roofline Model Analysis

Performance bound analysis:

```bash
# Generate roofline data
ncu --set roofline -o roofline ./cuda_program

# Key metrics:
# - Achieved FLOP/s
# - Achieved Memory Bandwidth
# - Arithmetic Intensity (FLOP/byte)
# - Ridge Point
```

Interpretation guide:
- Below memory roofline: Memory bound
- Below compute roofline: Compute bound
- At peak: Optimal utilization

### 5. Memory Bandwidth Analysis

Identify memory bottlenecks:

```bash
# Memory analysis sections
ncu --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Chart \
    --section MemoryWorkloadAnalysis_Tables \
    -o memory ./cuda_program
```

Key metrics:
- Global Load/Store Throughput
- L1/L2 Cache Hit Rate
- Shared Memory Bandwidth
- Memory Transactions per Request

### 6. Warp Execution Analysis

Analyze warp efficiency:

```bash
# Warp state analysis
ncu --section WarpStateStatistics -o warp ./cuda_program

# Scheduler statistics
ncu --section SchedulerStatistics -o scheduler ./cuda_program
```

Key metrics:
- Warp Cycles Per Issued Instruction
- Eligible Warps Per Active Cycle
- Active Warps Per Scheduler
- Stall Reasons (memory, sync, execution)

### 7. Kernel Comparison

Compare kernel variants:

```bash
# Step 1: Profile baseline
ncu --target-processes all --set full -o baseline ./program_v1

# Step 2: Profile optimized version
ncu --target-processes all --set full -o optimized ./program_v2

# Step 3: Export both profiles to CSV, then compare with Python (no GUI needed)
# Note: --import can only be specified once; --page diff is not a valid page value.
ncu --import baseline.ncu-rep --page details --csv > baseline_details.csv
ncu --import optimized.ncu-rep --page details --csv > optimized_details.csv

python3 -c "
import csv
def load(p):
    return {r.get('Metric Name',''): r.get('Metric Value','')
            for r in csv.DictReader(open(p))}
b = load('baseline_details.csv')
o = load('optimized_details.csv')
for k in sorted(set(b) | set(o)):
    bv, ov = b.get(k,''), o.get(k,'')
    if bv != ov:
        print(f'{k[:55]:<55} {bv} -> {ov}')
"
```

### 8. Performance Recommendations

Automated analysis:

```bash
# Get optimization recommendations
ncu --section SpeedOfLight \
    --section SpeedOfLight_RooflineChart \
    -o speedoflight ./cuda_program

# Export with recommendations
ncu --import profile.ncu-rep --page details --csv > details.csv
```

## Common Profiling Workflows

### Workflow 1: Initial Performance Assessment

```bash
# Step 1: System overview
nsys profile -t cuda -o system_overview ./program
nsys stats system_overview.nsys-rep

# Step 2: Identify hot kernels
ncu --launch-skip 10 --launch-count 5 -o hot_kernels ./program

# Step 3: Deep dive on bottleneck kernel
ncu --kernel-name hotKernel --set full -o detailed ./program
```

### Workflow 2: Memory Optimization

```bash
# Analyze memory access patterns
ncu --section SourceCounters \
    --section MemoryWorkloadAnalysis \
    --kernel-name targetKernel \
    -o memory_analysis ./program

# Check for coalescing issues
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
    -o coalescing ./program
```

### Workflow 3: Occupancy Optimization

```bash
# Profile with occupancy focus
ncu --section Occupancy \
    --section LaunchStatistics \
    -o occupancy ./program
```

**Interpreting occupancy limiters** (from the `Occupancy` section report):

| Limiter shown | Fix |
|---------------|-----|
| `Registers` | Reduce register pressure: use fewer local variables, add `maxnreg` hint |
| `Shared Memory` | Decrease shared memory allocation or use 32-bit instead of 64-bit |
| `Block Size` | Increase threads per block; ensure block size is a multiple of warp size (32) |
| `Warp Limit` | Already at theoretical max for this SM; no action needed |

> **For Triton kernels**: block sizes are controlled via `@triton.autotune` configs, not CLI flags. To test occupancy at different block sizes, add or modify the `triton.Config({"BLOCK_C": N}, num_warps=W)` entries in the autotune list and re-run. Do **not** pass `--block-size` as a CLI argument — the Triton benchmark script does not accept it.

## Dependencies

- Nsight Systems 2023.1+
- Nsight Compute 2023.1+
- CUDA Toolkit 11.0+

## Constraints

- Full profiling requires root/admin privileges
- Some metrics only available on specific GPU architectures
- Profiling adds overhead; results may differ from production
- Nsight Compute profiles one kernel invocation at a time by default
