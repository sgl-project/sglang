# Error Analysis & Optimization Guide

This reference defines the protocol for the **optimize sub-agent** to analyze errors and plan optimizations for overlap kernels. It covers error diagnosis, optimization strategies, and version management.

---

## Error Diagnosis Taxonomy

When the optimize sub-agent receives a result file from the benchmark sub-agent, it first reads the unified JSON and classifies the error (if any) into one of the following categories:

| Category | Typical Symptoms | Root Cause Patterns | Fix Strategy |
|----------|-----------------|--------------------|--------------|
| **Compile Error** | Triton JIT compilation fails; `CompilationError`, `RuntimeError` during kernel import | Syntax errors, unsupported Triton ops, missing `tl.constexpr`, type mismatches | Fix code syntax; ensure all Triton constraints are met |
| **CUDA Illegal Memory Access** | `CUDA error: an illegal memory access was encountered`; silent corruption or crash | Out-of-bounds pointer, unmasked PTX inline asm, power-of-2 lane overflow, incorrect stride calculation | Add address clamping, boundary masks, validate pointer offsets |
| **CUDA Misaligned Address** | `CUDA error: a misaligned address was encountered` | Accessing `symm_mem` buffer at non-16B-aligned offset; `BLOCK_SIZE` not matching alignment requirements | Ensure 16B alignment for all buffer accesses; use `tl.multiple_of` hints |
| **Hang / Deadlock** | `timeout 30` kills the process (exit code 124); no output after 30s; GPU shows 100% utilization on nvidia-smi but no progress; torchrun processes linger after timeout | **Most common overlap kernel hang causes:** (1) Signal polling spin-wait without timeout: `while ld_sys(signal_addr) != 1: pass` — if CE never writes the signal, the thread spins forever; (2) Barrier mismatch across ranks: one rank hits `dist.barrier()` but another rank is stuck in the kernel; (3) Missing signal reset between iterations: `progress.fill_(0)` not called before relaunching, so kernel reads stale `1` values and proceeds before data is ready, or CE reads stale `0` and signals are never written; (4) `break`/`continue` in Triton JIT loop body — Triton doesn't support these control flow constructs at the thread level, causing CTA divergence; (5) Circular barrier ordering: rank 0 waits for rank 1's signal, rank 1 waits for rank 0's signal; (6) Asymmetric CTA count: if num_tiles is not divisible across ranks, some CTAs never reach the barrier | **Fix by root cause:** (1) Add bounded spin-wait with timeout counter to signal polling loops; (2) Ensure all ranks call `dist.barrier()` symmetrically; (3) Always reset signal tensor to 0 before each kernel launch with a barrier before and after; (4) Replace `break`/`continue` with `while not cond` pattern; (5) Use rotated peer ordering so each rank pulls from different peers first; (6) Pad num_tiles to be divisible by world_size. **Also add host-side timeout** wrapping all torchrun commands with `timeout 30` to detect hangs automatically |
| **Correctness Mismatch** | Output differs from reference beyond tolerance | Accumulator dtype (fp16 instead of fp32), wrong reduction order, incorrect scaling, wrong output shape | Use fp32 accumulator; verify cast placement; check output shape matches collective semantics |
| **Environment Failure** | `torch.cuda.is_available() == False`, missing packages, wrong GPU count | Pod doesn't have required GPUs, missing symm_mem support, incompatible torch version | Cannot fix via kernel optimization — report to user |

---

## Diagnosis Procedure

### Step 1: Read the Result File

Read `<kernel_name>_result_v<version>.json` completely. This is a unified file containing both performance data and error info. Extract:
- **Correctness status** (`correctness.status`: PASSED or FAILED)
- **Error type** (from `errors.error_type` if `errors` is not null — using the taxonomy above)
- **Error details** (from `errors.error_details` + `errors.traceback` if available)
- **Environment info** (from `errors.env_info` if available)
- **Performance data** (from `performance` if correctness passed — speedup, latency, efficiency)

### Step 2: Read the Kernel Source

Read the current version of the kernel file (`<kernel_name>_v<version>.py`). Cross-reference the error with specific code sections:
- For compile errors: locate the exact line/operation that Triton rejected
- For CUDA errors: locate pointer arithmetic, inline asm, mask logic
- For hangs: locate barrier calls, signal patterns, loop structures
- For correctness: locate accumulator types, reduction logic, output shapes

### Step 2b: Hang-Specific Diagnosis (if error_type is "hang_deadlock")

When the benchmark timed out with exit code 124, perform this additional structured diagnosis:

1. **Locate signal polling loops**: Search for `while` loops that poll signal values (e.g., `while ld_sys(signal_addr) != 1: pass`). These are the #1 cause of hangs. Check:
   - Is there a timeout/bound on the loop? If no, the thread will spin forever if CE never writes.
   - Is the signal tensor reset to 0 before each kernel launch? (`progress.fill_(0)`)
   - Is there a `dist.barrier()` after the reset to ensure all ranks see it?

2. **Check barrier symmetry**: Ensure every `dist.barrier()` in the host code is reached by ALL ranks. An asymmetric barrier (one rank calls it, another doesn't) causes deadlock.

3. **Check `break`/`continue` in Triton JIT**: Triton doesn't support Python `break` or `continue` in JIT-compiled loops at the thread level. Replace with conditional logic (`while not cond: pass` or flag variables).

4. **Check CTA count vs world_size alignment**: If `num_tiles` is not evenly divisible by the chunk scheduling, some CTAs may be assigned to empty chunks and skip signal polling, causing other CTAs to wait forever.

5. **Check peer ordering**: In full-mesh pull AllGather, each rank must pull from all peers. If the rotation order is wrong, a rank may wait for a signal that is only written after the rank itself completes, creating a circular dependency.

6. **Examine the partial log**: Read `<kernel_name>_results/correctness_v<version>.log` from the pod for any partial output. Sometimes the kernel prints partial results before hanging, which can identify which rank/stage is stuck.

### Step 3: Read the Performance Info (If Available)

If the result file has `correctness.status == "PASSED"` and `performance` is not null, read the performance data from the same JSON to understand:
- Which parts worked (compute-only passed? comm-only passed?)
- Where the bottleneck is (compute-heavy? comm-heavy?)

This helps the optimize sub-agent prioritize which dimension to optimize.

---

## Optimization Strategies

### Strategy Selection Matrix

Based on the error category and performance profile, select the optimization strategy:

| Error Category | Performance Profile | Primary Strategy | Secondary Strategy |
|---------------|--------------------|-----------------|--------------------|
| Compile Error | N/A (can't run) | Fix syntax/compilation issues | N/A |
| CUDA Illegal Access | N/A (crashes) | Fix memory safety (address clamping, masks) | Reduce grid size as safety measure |
| CUDA Misaligned | N/A (crashes) | Fix alignment (16B boundaries, `tl.multiple_of`) | Adjust BLOCK_SIZE to power-of-2 aligned values |
| Hang / Deadlock | N/A (timeout after 30s, exit code 124) | Fix the specific hang root cause (signal polling, barrier symmetry, signal reset, CTA divergence) | Add bounded spin-wait with timeout counter in kernel; verify `progress.fill_(0)` + barrier pattern before each launch |
| Correctness Mismatch | Low speedup (< 1.5x) | Fix precision (fp32 accumulator, cast placement) | Adjust block sizes for better tiling |
| Correctness Pass, Low Performance | Low speedup (< 1.5x) | Tune block sizes and SM allocation | Reduce register pressure via chunking |
| Correctness Pass, Good Performance | Good speedup (> 1.5x) | Fine-tune for marginal gains | Try alternative overlap mode |

### Parameter Tuning Parameters

These are the tunable parameters the optimize sub-agent can adjust (do NOT change the overlap algorithm itself unless correctness is broken):

| Parameter | Default | Range | Impact | How to Tune |
|-----------|---------|-------|--------|-------------|
| `BLOCK_M` | auto-computed | 16–128 | CTA tile size in M dimension; affects register pressure and occupancy | Increase for compute-heavy, decrease for memory-bound |
| `BLOCK_N` | auto-computed | 32–512 | CTA tile size in N dimension; affects shared memory usage | Must be `next_power_of_2(N_per_chunk)`; increase chunk size to increase |
| `BLOCK_K` | 128 (for GEMM) | 64–256 | Inner loop tile for GEMM kernels | Keep at 128 for fp8; 64 for bf16 with high register pressure |
| `num_warps` | 32 | 4–32 | Threads per CTA; affects memory bandwidth saturation | Always try 32 first for memory-bound kernels; reduce to 16 if register pressure is high |
| `num_stages` | 1 | 0–4 | Software pipelining depth | Keep at 1 for overlap kernels (compute and comm interleave naturally) |
| `num_comm_sms` | 8 | 4–16 | SMs dedicated to communication in inter-SM mode | Increase if comm is bottleneck; decrease if compute needs more SMs |
| `N_CHUNKS` | `N // 1024` | 2–16 | Number of chunks in N dimension; reduces register pressure | Increase if occupancy is low; decrease if chunk overhead dominates |

### Code-Level Optimization Strategies

When parameter tuning alone doesn't achieve the target speedup, the optimize sub-agent can apply code-level optimizations:

| Strategy | When to Apply | How |
|----------|--------------|-----|
| **Reduce register pressure** | Occupancy < 50%; compiler reports high register count | Split K-dimension into chunks via `tl.range`; use `tl.constexpr` for strides; avoid large intermediate tensors |
| **Vectorize memory access** | Kernel is memory-bound but BW utilization is low | Add `tl.multiple_of(ptr, 16)` hints; use `tl.load` with `eviction_policy=evict_last` for read-once data |
| **Eliminate redundant computation** | Compute time > expected; repeated calculations in loop | Pre-compute offsets outside loops; use incremental pointer update (`ptr += stride` instead of `base + i * stride`) |
| **Optimize barrier pattern** | Barrier overhead > 10% of total time | Replace 3-barrier pattern with master CTA barrier; use PTX-level CAS spin-loop instead of Python `while` |
| **Switch overlap mode** | Current mode has fundamental limitations | Consider switching: intra-SM → inter-SM (if register pressure too high), inter-SM → without-SM (if SM cost dominates) |

---

## Version Management Rules

### Naming Convention

Each optimization iteration produces a **new kernel file** with a version suffix. The original kernel is `v1`, and each subsequent optimization is `v2`, `v3`, etc.

```
<kernel_name>_v1.py   ← initial generation (from sglang-overlap-kernel-generation)
<kernel_name>_v2.py   ← first optimization iteration
<kernel_name>_v3.py   ← second optimization iteration
...
```

### File Organization

All kernel versions coexist in the same directory. The optimize sub-agent:
- **NEVER overwrites** an existing version file
- **Always creates** a new file with the next version number
- **Preserves** the original version for comparison and rollback

### Benchmark File Correspondence

Each kernel version has a corresponding benchmark file:

```
<kernel_name>_v1_benchmark.py
<kernel_name>_v2_benchmark.py
...
```

The benchmark file for each version imports from its corresponding kernel version. The optimize sub-agent updates the import path in the benchmark file when creating a new version.

### Error and Performance File Correspondence

All data is stored in **unified result files** (not separate perf/error files):

```
<kernel_name>_result_v1.json   ← v1 result: perf + errors in one file
<kernel_name>_result_v2.json   ← v2 result: perf + errors in one file
...
```

If v1 correctness passed: `errors` = null, `performance` populated.
If v2 correctness failed: `errors` populated, `performance` = null.
All data for a version is in one place — no separate log files to cross-reference.

---

## Optimization Iteration Protocol

### Step 1: Read Current State

1. Read the **result JSON file** from the latest benchmark run (`<kernel_name>_result_v<version>.json`) — this unified file contains both performance data and error info
2. Read the **current kernel source** (the version that produced the result)
4. Read **previous result files** for comparison (what changed, what was tried)

### Step 2: Diagnose and Plan

1. Classify the error using the taxonomy above
2. Identify the root cause in the kernel source
3. Select the optimization strategy from the strategy matrix
4. Plan specific code changes (which parameters to tune, which code sections to modify)
5. Document the planned changes in a brief optimization note

### Step 3: Implement Optimization

1. Create a **new kernel file** with the next version number (`v<N+1>.py`)
2. Copy the current version's code as the starting point
3. Apply the planned changes
4. Update the benchmark file import to reference the new version
5. Ensure the kernel remains self-contained (no cross-imports between versions)

### Step 4: Generate Optimization Note

Write a brief note documenting what was changed and why:

```
=== Optimization Note: <kernel_name> v<N+1> ===
Based on: v<N> error/performance data
Error type: <category or "none — performance optimization">
Changes:
  1. <specific change with line reference>
  2. <specific change with line reference>
Rationale: <why these changes should improve the issue>
Expected impact: <predicted improvement in correctness or performance>
```

This note is saved alongside the kernel file for traceability.

---

## Optimization Stop Criteria

The optimize loop terminates when any of these conditions is met:

1. **Target speedup achieved**: overlap speedup ≥ user-specified target (default: 1.5x)
2. **Max iterations reached**: user-specified number of optimization rounds completed
3. **No progress detected**: last 3 iterations show no performance improvement (speedup delta < 0.05x)
4. **Regression detected**: new version's performance is worse than the previous version by > 0.2x

When the loop terminates due to stop criteria 3 or 4, the optimize sub-agent should:
- Note the stop reason in its report
- Recommend the best-performing version found so far
- NOT attempt further optimization without user guidance
