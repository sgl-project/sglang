# SGLang Overlap Kernel Optimize — Multi-Agent Orchestration Skill

## Overview

This skill orchestrates an iterative overlap kernel optimization loop using **four sequential sub-agents**. It is a **lead agent** that never directly generates kernel code, runs benchmarks, or performs integration — all work is delegated to specialized sub-agents that invoke existing skills.

The optimization loop:
1. **Generate** an overlap kernel and benchmark(via `sglang-overlap-kernel-generation` skill)
2. **Benchmark** the kernel on a remote pod (via kubectl exec)
3. **Optimize** the kernel based on benchmark results / error info (tune parameters, optimize kernel code, fix errors)
4. **Repeat** steps 2–3 for the user-specified number of iterations
5. **Integrate** the best-performing version into SGLang (via `sglang-overlap-kernel-integration` skill)

---

## Architecture

### Lead Agent Responsibilities

The lead agent (this skill) is responsible for:
- **Orchestrating the loop**: dispatching sub-agents in the correct sequence
- **Tracking progress**: maintaining the performance progress table across iterations
- **Version management**: ensuring each optimization creates a new versioned file (never overwriting)
- **Best-version selection**: selecting the highest-speedup correctness-passing version after all iterations
- **Handoff preparation**: assembling the integration context package for the final sub-agent
- **User communication**: presenting iteration results and final recommendation

The lead agent **never** writes kernel code, runs benchmarks, or modifies SGLang framework code directly.

### Sub-Agent Sequence

Sub-agents are dispatched **sequentially** (not parallel) because each step depends on the output of the previous step:

| Step | Sub-Agent | Invoked Skill / Action | Input | Output |
|------|-----------|----------------------|-------|--------|
| 1 | **Generation** | `sglang-overlap-kernel-generation` | User requirements (mode, ops, shapes) | Kernel file `v1.py` + benchmark file `v1_benchmark.py` |
| 2 | **Benchmark** | kubectl exec + run benchmark on remote pod | Kernel + benchmark files, pod info | `<kernel_name>_result_v1.json` (unified: perf + errors in one file) |
| 3 | **Optimize** | Error analysis + parameter tuning + code fixes | Result JSON file + current kernel source | New kernel `v<N+1>.py` + updated benchmark file |
| 4 | **Benchmark** | Same as Step 2 | New kernel version files | `<kernel_name>_result_v<N+1>.json` |
| ... | **Loop** Steps 3–4 | Until max iterations reached or target speedup achieved | ... | ... |
| Final | **Integration** | `sglang-overlap-kernel-integration` | Best kernel file + integration context package | SGLang framework changes |

---

## Required User Inputs

Before starting, the lead agent must collect these inputs from the user:

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| **Overlap mode** | Yes | — | `inter-sm`, `intra-sm`, or `without-sm` (passed to generation skill) |
| **Compute operator** | Yes | — | GEMM, attention, etc. (passed to generation skill) |
| **Communication collective** | Yes | — | all-gather, reduce-scatter, all-reduce (passed to generation skill) |
| **Problem shape (M, N, K)** | Yes | — | Tensor dimensions for the overlap kernel |
| **Data type** | Yes | bf16 | bf16, fp16, fp8, etc. |
| **Pod name or namespace+selector** | Yes | — | Kubernetes pod for benchmark execution |
| **World size** | Yes | 2 | Number of GPUs / ranks for distributed benchmark |
| **Max optimization iterations** | Yes | 5 | Number of optimize→benchmark loops to run |
| **Comm time savings target** | No | 50% | Overlap kernel must save > this % of pure communication time for early termination. Formula: `(non_overlap_us - overlap_us) > target * comm_only_us` |
| **Multimem / NVLS** | No | false | Whether to use NVLS for communication |
| **FP8 GEMM** | No | false | Whether compute kernel uses block-wise FP8 |

---

## Workflow

### Step 0: Input Collection & Planning

Collect all required user inputs listed above. If any required input is missing, ask the user before proceeding.

Then output the **Optimization Plan** to the user:

```
=== Overlap Kernel Optimization Plan ===
Mode: <overlap_mode>
Compute: <compute_op>  Comm: <comm_collective>
Shape: M=<M>, N=<N>, K=<K>  Dtype: <dtype>
Pod: <pod_name>  World size: <world_size>
Max iterations: <max_iterations>  Comm savings target: ><target>% of comm_only time
Multimem: <yes/no>  FP8 GEMM: <yes/no>

Sub-agent sequence:
  1. Generation (sglang-overlap-kernel-generation) → v1 kernel
  2. Benchmark (kubectl exec on <pod>) → v1 result JSON
  3. Optimize (error analysis + tuning) → v2 kernel
  4. Benchmark → v2 result JSON
  ... (repeat until <max_iterations> or comm_savings > <target>%)
  Final. Integration (sglang-overlap-kernel-integration) → SGLang changes
```

Wait for user confirmation before proceeding.

---

### Step 1: Generation Sub-Agent

Dispatch a sub-agent to invoke the **`sglang-overlap-kernel-generation`** skill.

#### Dispatch Prompt

```
You are the Generation sub-agent. Your task is to generate an overlap kernel using the sglang-overlap-kernel-generation skill.

Requirements:
- Overlap mode: <overlap_mode>
- Compute operator: <compute_op>
- Communication collective: <comm_collective>
- Problem shape: M=<M>, N=<N>, K=<K>
- Data type: <dtype>
- World size: <world_size>
- Multimem/NVLS: <yes/no>
- FP8 GEMM: <yes/no>

Follow the sglang-overlap-kernel-generation SKILL.md exactly, including:
1. Read the appropriate mode reference file
2. Read references/primitives.md before emitting any PTX helpers
3. Generate the kernel file following the four-section format
4. Run the correctness & hang check verification
5. Generate the benchmark test file

Output files:
- Kernel file: save as <kernel_name>_v1.py
- Benchmark file: save as <kernel_name>_v1_benchmark.py

These files will be sent to a remote pod for benchmark testing.
```

#### Expected Output

- `<kernel_name>_v1.py` — the initial overlap kernel
- `<kernel_name>_v1_benchmark.py` — the corresponding benchmark test

The lead agent records these file paths for the benchmark sub-agent.

---

### Step 2: Benchmark Sub-Agent

Dispatch a sub-agent to execute the benchmark on the remote pod using `kubectl exec`.

#### Dispatch Prompt

```
You are the Benchmark sub-agent. Your task is to run the overlap kernel benchmark on a remote Kubernetes pod.

Pod info:
- Pod name: <pod_name>
- Namespace: <namespace> (if provided)
- World size: <world_size>

Files to test:
- Kernel: <kernel_name>_v<version>.py
- Benchmark: <kernel_name>_v<version>_benchmark.py

Remote work directory: /tmp/overlap_kernel_bench
All files on the pod MUST be placed under this directory.

Follow the benchmark execution protocol in references/benchmark-protocol.md exactly:
1. Validate the pod environment (GPU, torch, triton, symm_mem)
2. Create remote work directory: kubectl exec <pod> -- mkdir -p /tmp/overlap_kernel_bench/results
3. Copy kernel and benchmark files to the pod:
   kubectl cp <local_kernel> <pod>:/tmp/overlap_kernel_bench/<kernel_name>_v<version>.py
   kubectl cp <local_benchmark> <pod>:/tmp/overlap_kernel_bench/<kernel_name>_v<version>_benchmark.py
4. Run correctness test FIRST with 30s timeout:
   kubectl exec <pod> -- bash -c 'cd /tmp/overlap_kernel_bench && timeout 30 env NVSHMEM_DISABLE_CUDA_VMM=0 torchrun --nproc_per_node=<world_size> <benchmark_file>.py --case correctness --M <M> --N <N> --K <K>'
5. If correctness PASSES (exit code 0), run performance benchmark with 30s timeout:
   kubectl exec <pod> -- bash -c 'cd /tmp/overlap_kernel_bench && timeout 30 env NVSHMEM_DISABLE_CUDA_VMM=0 torchrun --nproc_per_node=<world_size> <benchmark_file>.py --case performance --M <M> --N <N> --K <K> --warmup 10 --iters 100'
6. Collect results back to local machine: kubectl cp <pod>:/tmp/overlap_kernel_bench/results/ <local_results_dir>/
7. Clean up: kill lingering torchrun processes and remove version-specific files on the pod
8. Write results to the unified result file: <kernel_name>_result_v<version>.json (use `scripts/record_benchmark_result.py`)

HANG DETECTION (CRITICAL):
- ALL benchmark commands MUST be wrapped with `timeout 30`. Exit code 124 = timeout = hang.
- If exit code is 124, the kernel has hung/deadlocked. Record this as error_type "hang_deadlock".
- On hang: capture partial log output, check GPU state (nvidia-smi), kill zombie processes, do NOT proceed to performance test.
- Common hang causes in overlap kernels: signal polling spin-wait without timeout, barrier mismatch across ranks, missing signal reset between iterations, break/continue in Triton JIT loops.

IMPORTANT:
- Correctness is a hard gate. Do NOT run performance if correctness fails or times out.
- Record FULL error output in the result file's `errors` field (including traceback, CUDA errors, environment info).
- For hangs: record error_type as "hang_deadlock" with error_details explaining the timeout behavior.
- Use `scripts/record_benchmark_result.py` to write the unified result JSON.
- Timeout: 30s for BOTH correctness and performance. No exceptions.
```

#### Expected Output

- **Unified result file**: `<kernel_name>_result_v<version>.json` — single file per version containing both performance data and error info (via `errors` field)

After the benchmark sub-agent returns, the lead agent:
1. Reads the result file
2. Updates the **Performance Progress Table** (use `scripts/select_best_version.py` or see `references/performance-tracking.md`)
3. Presents the current iteration results to the user
4. Decides whether to continue the optimization loop or terminate early

---

### Step 3: Optimize Sub-Agent

Dispatch a sub-agent to analyze errors / performance and produce an improved kernel version.

#### Dispatch Prompt Template

```
You are the Optimize sub-agent. Your task is to analyze the benchmark results and produce an optimized version of the overlap kernel.

Current state:
- Current kernel: <kernel_name>_v<version>.py
- Benchmark result: <result_file_path> (unified JSON with perf + errors)
- Previous versions' performance: <summary of all prior iterations' speedup and key changes>

Follow the error analysis and optimization guide in references/error-analysis.md exactly:
1. Read the error/performance file to understand what needs to be fixed or improved
2. Read the current kernel source to identify root causes
3. Classify the error (if any) using the taxonomy
4. Select the appropriate optimization strategy
5. Create a NEW kernel file: <kernel_name>_v<next_version>.py (NEVER overwrite existing versions)
6. Copy the current version's code as starting point, then apply targeted changes
7. Update the benchmark file import to reference the new version: <kernel_name>_v<next_version>_benchmark.py
8. Write an optimization note documenting what was changed and why

VERSION MANAGEMENT RULES:
- NEVER overwrite or delete existing kernel version files
- Always create a new file with the next version number
- The new version must be self-contained (no cross-imports between versions)

Optimization strategies (prioritize in this order):
- If kernel FAILED correctness: fix the root cause (memory safety, barriers, precision)
- If kernel PASSED but below target speedup: tune parameters (BLOCK_M/N/K, num_warps, num_comm_sms, N_CHUNKS)
- If kernel PASSED and near target: fine-tune for marginal gains (vectorization hints, barrier optimization)
```

#### Expected Output

- `<kernel_name>_v<next_version>.py` — the optimized kernel
- `<kernel_name>_v<next_version>_benchmark.py` — updated benchmark file
- Optimization note (brief description of changes and rationale)

After the optimize sub-agent returns, the lead agent dispatches the benchmark sub-agent again (Step 2) with the new version.

---

### Step 4: Loop Control

After each benchmark sub-agent return, the lead agent evaluates the loop termination conditions:

**Comm savings metric**: `comm_savings_pct = (non_overlap_us - overlap_us) / comm_only_us * 100%`

This measures how much of the pure communication time the overlap kernel saves. The overlap kernel overlaps compute and comm, so the ideal overlap time ≈ `max(compute_only_us, comm_only_us)`. The time saved compared to sequential execution (`non_overlap_us`) is `non_overlap_us - overlap_us`, and we compare this savings against the communication cost (`comm_only_us`) to quantify overlap effectiveness.

| Condition | Action |
|-----------|--------|
| **Comm savings target achieved** (`comm_savings_pct > target%`) | Stop loop; proceed to Step 5 (integration) |
| **Max iterations reached** | Stop loop; proceed to Step 5 with best available version |
| **All versions failed correctness** | Stop loop; report failure to user; do NOT proceed to integration |
| **No progress for 3 iterations** (last 3 passing versions show < 5% improvement in `comm_savings_pct`) | Stop loop; proceed to Step 5 with best available version |
| **Regression** (`comm_savings_pct` drops > 10% from best) | Note regression; continue loop but flag the regression |
| **Otherwise** | Continue loop: dispatch optimize sub-agent → benchmark sub-agent |

The lead agent presents the current **Performance Progress Table** to the user after each iteration (use `scripts/select_best_version.py` to generate):

```
Iteration | Version | Correctness | Speedup | Overlap (us) | Comm Savings (%) | Key Change
---------------------------------------------------------------------------------------------------------
    1     |   v1    |   PASSED    |  1.20   |    3456.7    |       25.0       | Initial generation
    2     |   v2    |   FAILED    |   N/A   |     N/A      |        N/A       | Reduced BLOCK_M, clamping
    3     |   v3    |   PASSED    |  1.80   |    2100.0    |       55.0       | Increased BLOCK_N, 32 warps
---------------------------------------------------------------------------------------------------------
Best so far: v3 (speedup=1.80, comm_savings=55.0%)
Comm savings target: >50% — ACHIEVED ✓

Proceeding to integration with v3.
```

---

### Step 5: Best Version Selection

After the loop terminates, the lead agent selects the best version for integration.

**Selection procedure** (see `references/performance-tracking.md` §Best Version Selection):

1. Filter out all versions that failed correctness
2. Sort remaining versions by `comm_savings_pct` DESC, then speedup DESC, overlap_us ASC
3. Select the first entry
4. Report the selection to the user with rationale

**Edge case handling**:
- If NO version passed correctness → report failure, recommend manual debugging, do NOT integrate
- If best `comm_savings_pct` < target → inform user, ask whether to proceed with integration anyway

After selection, the lead agent prepares the **Integration Handoff Package**:

```json
{
  "kernel_file": "<kernel_name>_v<best>.py",
  "kernel_version": "v<best>",
  "overlap_mode": "<mode>",
  "compute_op": "<compute_op>",
  "comm_op": "<comm_collective>",
  "symm_mem_mechanism": "<mechanism>",
  "target_model_layers": "<layers>",
  "forward_mode": "<prefill|decode|both>",
  "speedup": "<best_speedup>",
  "overlap_efficiency": "<best_efficiency>",
  "comm_savings_pct": "<best_comm_savings_pct>",
  "config": {
    "M": "<M>", "N": "<N>", "K": "<K>",
    "dtype": "<dtype>",
    "num_comm_sms": "<value>",
    "world_size": "<world_size>"
  }
}
```

---

### Step 6: Integration Sub-Agent

Dispatch a sub-agent to invoke the **`sglang-overlap-kernel-integration`** skill with the best-performing kernel.

#### Dispatch Prompt

```
You are the Integration sub-agent. Your task is to integrate the best-performing overlap kernel into the SGLang framework using the sglang-overlap-kernel-integration skill.

Integration handoff package:
<full JSON of the handoff package>

Kernel file to integrate: <kernel_name>_v<best_version>.py
Performance data: <kernel_name>_perf_summary.json (generated by `scripts/select_best_version.py --output_summary`)

Follow the sglang-overlap-kernel-integration SKILL.md exactly, executing all 7 patterns in order:
  P0: Profile-driven identification + semantic gate
  P1: Kernel understanding
  P2: Self-contained kernel placement (migrate to symm_mem_kernels/)
  P3: Communicator setup
  P4: Env-var gating
  P5: Fused fast-path + fallback
  P6: Redundant communication bypass

IMPORTANT:
- Use the BEST version (<best_version>) of the kernel for integration
- The kernel must be self-contained — no cross-imports from versioned files
- Follow the integration checklist at the end of the skill
- Verify all patterns before declaring completion
```

#### Expected Output

The integration sub-agent produces the full set of SGLang framework changes as described in the `sglang-overlap-kernel-integration` skill. The lead agent reviews the changes and presents a summary to the user.

---

## Final Output Format

After all sub-agents have completed, the lead agent produces a summary report:

```
=== Overlap Kernel Optimization Complete ===

Optimization Loop Summary:
  Total iterations: <N>
  Best version: v<best> (speedup=<speedup>x, comm_savings=<comm_savings_pct>%)
  Comm savings target: ><target>% — ACHIEVED / NOT ACHIEVED

Performance Progress:
  <Full progress table generated by scripts/select_best_version.py>

Best Kernel:
  File: <kernel_name>_v<best>.py
  Speedup: <speedup>x over sequential compute+comm
  Overlap latency: <overlap_us> us
  Comm savings: <comm_savings_pct>% of comm_only time overlapped away

Integration Status:
  <Summary of integration sub-agent's output — files modified, patterns applied>

All Result Files:
  v1: <path>/<kernel_name>_result_v1.json — speedup=<val>, comm_savings=<val>% / FAILED (errors: <error_type>)
  v2: <path>/<kernel_name>_result_v2.json — speedup=<val>, comm_savings=<val>% / FAILED (errors: <error_type>)
  ...
  v<best>: <path>/<kernel_name>_result_v<best>.json — speedup=<val>, comm_savings=<val>% — SELECTED FOR INTEGRATION
```

---

## References

| File | When to Read |
|------|-------------|
| `references/benchmark-protocol.md` | Step 2 — benchmark sub-agent dispatch; defines pod execution protocol, file transfer, unified result file format |
| `references/error-analysis.md` | Step 3 — optimize sub-agent dispatch; defines error taxonomy, optimization strategies, version management rules |
| `references/performance-tracking.md` | Step 4–5 — loop control and best-version selection; defines unified result file schema, progress table, selection criteria |
| `scripts/record_benchmark_result.py` | Step 2 — write unified result JSON files; used by benchmark sub-agent to record both perf and error data |
| `scripts/select_best_version.py` | Step 4–5 — read all result files, print progress table, select best version, optionally generate summary JSON |
| `../sglang-overlap-kernel-generation/SKILL.md` | Step 1 — generation sub-agent invokes this skill for initial kernel creation |
| `../sglang-overlap-kernel-generation/references/primitives.md` | Step 1 — sub-agent reads for PTX primitives before generating kernel |
| `../sglang-overlap-kernel-generation/references/benchmark_template.md` | Step 1 — sub-agent reads for benchmark test file format |
| `../sglang-overlap-kernel-generation/references/<mode>.md` | Step 1 — sub-agent reads the selected overlap mode reference |
| `../sglang-overlap-integration/SKILL.md` | Step 6 — integration sub-agent invokes this skill for SGLang framework integration |

---

## Constraints

- **Lead agent never writes kernel code** — all kernel generation and optimization is done by sub-agents
- **Lead agent never runs benchmarks** — all remote execution is done by benchmark sub-agents
- **Lead agent never modifies SGLang framework** — all integration is done by integration sub-agent
- **Never overwrite existing versions** — each optimization creates a new versioned file
- **Correctness is a hard gate** — performance benchmark never runs on a kernel that failed correctness
- **Self-contained kernels** — each kernel version file imports only from standard packages; no cross-imports between versions
- **Pod access required** — the user must provide a valid Kubernetes pod for benchmark execution; the skill cannot proceed without it
- **30-second timeout mandatory** — all benchmark commands on the pod MUST be wrapped with `timeout 30`; if a command does not return within 30 seconds, it is killed (exit code 124) and recorded as a hang/deadlock failure
- **Hang = hard failure** — if any benchmark phase times out (exit code 124), the kernel is treated as FAILED regardless of whether it might eventually complete; do NOT retry without code fixes
- **Remote work directory** — all files on the pod MUST be placed under `/tmp/overlap_kernel_bench/`; this path is fixed and must not be changed
- **Cleanup after each run** — kill lingering torchrun processes and remove version-specific files on the pod after collecting results, to avoid stale state affecting subsequent runs
