# Benchmark Execution Protocol

This document defines the end-to-end protocol for executing overlap kernel benchmarks on a remote Kubernetes pod. The benchmark sub-agent MUST follow these steps exactly in order.

---

## Remote Work Directory Convention

All kernel and benchmark files on the pod are placed under a fixed directory:

```
REMOTE_WORK_DIR=/tmp/overlap_kernel_bench
```

Files on the pod follow this layout:

```
/tmp/overlap_kernel_bench/
├── <kernel_name>_v<version>.py
├── <kernel_name>_v<version>_benchmark.py
└── results/
    └── <kernel_name>_result_v<version>.json
```

The benchmark sub-agent MUST use this path for all `kubectl cp` and `kubectl exec` operations. Create it if it does not exist.

---

## Step 1: Pod Environment Validation

Before any benchmark, validate the pod has the required hardware and software:

```bash
kubectl exec <pod> -n <namespace> -- bash -c '
  echo "=== GPU Check ===" && \
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && \
  echo "=== GPU Count ===" && \
  nvidia-smi --list-gpus | wc -l && \
  echo "=== Python Check ===" && \
  python3 -c "import torch; print(f\"torch={torch.__version__}, cuda={torch.cuda.is_available()}\")" && \
  echo "=== Triton Check ===" && \
  python3 -c "import triton; print(f\"triton={triton.__version__}\")" && \
  echo "=== Symmetric Memory Check ===" && \
  python3 -c "import torch.distributed._symmetric_memory as sm; print(\"symm_mem=OK\")" && \
  echo "=== NVSHMEM Check ===" && \
  python3 -c "import nvshmem; print(\"nvshmem=OK\")" 2>/dev/null || echo "nvshmem=NOT_AVAILABLE"
'
```

**Validation gates** (must ALL pass before proceeding):

| Check | Requirement | Failure Action |
|-------|------------|----------------|
| GPU available | `torch.cuda.is_available() == True` | Record env failure, STOP |
| GPU count | `>= world_size` | Record env failure, STOP |
| Triton | Version >= 3.0 | Record env failure, STOP |
| Symmetric memory | `import` succeeds | Record env failure, STOP |
| NVSHMEM | `import` succeeds (optional for without-sm mode) | Warning only; only required for inter-SM / intra-SM modes |

If validation fails, record the error using the `environment_failure` error type and STOP. Do NOT proceed to benchmark.

---

## Step 2: File Transfer to Pod

Copy kernel and benchmark files from the local machine to the pod.

### 2.1 Create remote work directory

```bash
kubectl exec <pod> -n <namespace> -- mkdir -p /tmp/overlap_kernel_bench/results
```

### 2.2 Copy files

```bash
# Copy kernel file
kubectl cp <local_kernel_path> <pod>:/tmp/overlap_kernel_bench/<kernel_name>_v<version>.py -n <namespace>

# Copy benchmark file
kubectl cp <local_benchmark_path> <pod>:/tmp/overlap_kernel_bench/<kernel_name>_v<version>_benchmark.py -n <namespace>
```

### 2.3 Verify file transfer

```bash
kubectl exec <pod> -n <namespace> -- ls -la /tmp/overlap_kernel_bench/<kernel_name>_v<version>*.py
```

Confirm both files exist and have non-zero size. If transfer fails, record error and STOP.

---

## Step 3: Run Correctness Test (with 30s Timeout)

Correctness is a **hard gate** — performance benchmark MUST NOT run if correctness fails.

### 3.1 Execute correctness test

```bash
kubectl exec <pod> -n <namespace> -- bash -c '
  cd /tmp/overlap_kernel_bench && \
  timeout 30 env NVSHMEM_DISABLE_CUDA_VMM=0 torchrun \
    --nproc_per_node=<world_size> \
    <kernel_name>_v<version>_benchmark.py \
    --case correctness \
    --M <M> --N <N> --K <K>
' 2>&1 | tee /tmp/overlap_kernel_bench/results/correctness_v<version>.log
```

### 3.2 Timeout & hang detection

**CRITICAL: 30-second timeout is mandatory.** The `timeout 30` wrapper ensures the process is killed if it does not return within 30 seconds. This is the primary mechanism for detecting kernel hangs.

After the command completes, check the exit code:

| Exit code | Meaning | Action |
|-----------|---------|--------|
| 0 | Correctness test passed | Proceed to Step 4 |
| 1 (or non-zero, non-124) | Correctness test failed (logic error, CUDA error, etc.) | Parse error output, record as `FAILED`, STOP |
| 124 | **Timeout — process killed after 30s** | This indicates a **kernel hang/deadlock**. Record as `HANG` error type, STOP |
| 137 | Process killed by SIGKILL (OOM or similar) | Record as `environment_failure`, STOP |

**HANG DETECTION PROTOCOL**: When exit code is 124 (timeout), the benchmark sub-agent MUST:

1. Record the error as `error_type: "hang_deadlock"` in the unified result file
2. Capture any partial output from the log file:
   ```bash
   kubectl exec <pod> -n <namespace> -- cat /tmp/overlap_kernel_bench/results/correctness_v<version>.log
   ```
3. Check for GPU state after hang (nvidia-smi may show GPU still occupied):
   ```bash
   kubectl exec <pod> -n <namespace> -- nvidia-smi
   ```
4. Attempt to clean up zombie processes:
   ```bash
   kubectl exec <pod> -n <namespace> -- bash -c "pkill -9 -f 'torchrun.*<kernel_name>'"
   ```
5. **Do NOT proceed to performance benchmark** — a hanging kernel is a hard failure
6. Report the hang to the lead agent for the optimize sub-agent to diagnose root cause

### 3.3 Parse correctness results

If the correctness test passed (exit code 0), extract diff metrics from the output:

```bash
kubectl exec <pod> -n <namespace> -- grep -E "(PASSED|FAILED|max_diff|mean_diff|diff=)" /tmp/overlap_kernel_bench/results/correctness_v<version>.log
```

Record `max_diff` and `mean_diff` values for the unified result file.

---

## Step 4: Run Performance Benchmark (with 30s Timeout)

Only execute if correctness test PASSED (exit code 0 in Step 3).

### 4.1 Execute performance benchmark

```bash
kubectl exec <pod> -n <namespace> -- bash -c '
  cd /tmp/overlap_kernel_bench && \
  timeout 30 env NVSHMEM_DISABLE_CUDA_VMM=0 torchrun \
    --nproc_per_node=<world_size> \
    <kernel_name>_v<version>_benchmark.py \
    --case performance \
    --M <M> --N <N> --K <K> \
    --warmup 10 --iters 100
' 2>&1 | tee /tmp/overlap_kernel_bench/results/performance_v<version>.log
```

### 4.2 Timeout & hang detection (same as Step 3.2)

The same 30-second timeout applies. Exit code 124 indicates a hang during the performance benchmark phase. This is less common than correctness-phase hangs (since correctness already verified the kernel can complete at least once), but can still occur due to:

- Non-deterministic hangs (race conditions in signal polling)
- GPU thermal throttling causing timing-dependent stalls
- Memory pressure from multiple iterations causing OOM-triggered hangs

If timeout occurs during performance benchmark:
1. Record as `error_type: "hang_deadlock"` with `error_details: "Performance benchmark hung after correctness passed — possible race condition or resource exhaustion"`
2. The correctness result still counts as PASSED, but performance data is unavailable
3. Set `performance: null` in the result file
4. Attempt cleanup (same as Step 3.2)

### 4.3 Parse performance results

If the performance benchmark completed (exit code 0), extract metrics:

```bash
kubectl exec <pod> -n <namespace> -- grep -E "(Compute-only|Comm-only|Non-overlap|Overlap kernel|Speedup|efficiency|savings)" /tmp/overlap_kernel_bench/results/performance_v<version>.log
```

Extract these values from the output:
- `compute_only_us` — standalone GEMM latency
- `comm_only_us` — NCCL AllGather latency
- `non_overlap_us` — sequential compute+comm latency
- `overlap_us` — overlap kernel latency
- `speedup` — (compute_time + comm_time) / overlap_time
- `comm_savings_pct` — (non_overlap - overlap) / comm_only * 100

---

## Step 5: Result Collection & Recording

After benchmark execution, collect output files back to local machine:

```bash
# Copy result logs and any output files from the pod
kubectl cp <pod>:/tmp/overlap_kernel_bench/results/ <local_results_dir>/ -n <namespace>
```

### Unified Result File Format

All benchmark results (both successful and failed) are recorded in a **single unified JSON file** per version. There is no separate error log file — error information is embedded in the same JSON via the `errors` field.

The file name convention is:

```
<kernel_name>_result_v<version>.json
```

Use the `scripts/record_benchmark_result.py` script to write result files:

```bash
# Successful benchmark
python3 scripts/record_benchmark_result.py \
    --kernel_name <kernel_name> \
    --version v1 \
    --pod <pod_name> \
    --iteration 1 \
    --status PASSED \
    --max_diff 0.0001 --mean_diff 0.00005 \
    --compute_only_us 1234.5 --comm_only_us 5678.9 \
    --non_overlap_us 6913.4 --overlap_us 3456.7 \
    --speedup 2.0 --overlap_efficiency 0.85 \
    --config "M=1024,N=7168,K=2048,dtype=bf16,world_size=2,num_comm_sms=8" \
    --tuning_changes "Initial generation" \
    --output_dir ./results

# Failed benchmark (correctness error)
python3 scripts/record_benchmark_result.py \
    --kernel_name <kernel_name> \
    --version v2 \
    --pod <pod_name> \
    --iteration 2 \
    --status FAILED \
    --max_diff 0.5 --mean_diff 0.2 \
    --error_type cuda_illegal_access \
    --error_details "CUDA error: an illegal memory access was encountered at kernel line 89" \
    --error_traceback "<full traceback string>" \
    --error_env_info "<nvidia-smi + version info>" \
    --config "M=1024,N=7168,K=2048,dtype=bf16,world_size=2,num_comm_sms=8" \
    --tuning_changes "Reduced BLOCK_M, added address clamping" \
    --output_dir ./results

# Hang / timeout (exit code 124)
python3 scripts/record_benchmark_result.py \
    --kernel_name <kernel_name> \
    --version v3 \
    --pod <pod_name> \
    --iteration 3 \
    --status FAILED \
    --error_type hang_deadlock \
    --error_details "Correctness test hung — process killed after 30s timeout. Possible signal polling deadlock or barrier mismatch." \
    --error_traceback "N/A — process timed out, no traceback available. Partial log captured in correctness_v3.log" \
    --error_env_info "<nvidia-smi output showing GPU state after hang>" \
    --config "M=1024,N=7168,K=2048,dtype=bf16,world_size=2,num_comm_sms=8" \
    --tuning_changes "Added barrier, changed num_warps from 4 to 32" \
    --output_dir ./results
```

### Unified Result JSON Schema

```json
{
  "kernel_name": "<op>_overlap",
  "version": "v1",
  "timestamp": "2026-06-26T14:30:00Z",
  "pod": "<pod_name>",
  "iteration": 1,
  "config": {
    "M": 1024,
    "N": 7168,
    "K": 2048,
    "dtype": "bf16",
    "world_size": 2,
    "num_comm_sms": 8,
    "block_m": 64,
    "block_n": 256,
    "block_k": 128,
    "num_warps": 32,
    "n_chunks": 7
  },
  "correctness": {
    "status": "PASSED",
    "max_diff": 0.0001,
    "mean_diff": 0.00005,
    "rtol": 0.01,
    "atol": 0.01
  },
  "performance": {
    "compute_only_us": 1234.5,
    "comm_only_us": 5678.9,
    "non_overlap_us": 6913.4,
    "overlap_us": 3456.7,
    "speedup": 2.0,
    "overlap_efficiency": 0.85
  },
  "errors": null,
  "tuning_changes": "Initial generation — no tuning applied"
}
```

When correctness **fails** (logic/CUDA error), the `performance` field is `null` and the `errors` field is populated:

```json
{
  "kernel_name": "<op>_overlap",
  "version": "v2",
  "timestamp": "2026-06-26T14:35:00Z",
  "pod": "<pod_name>",
  "iteration": 2,
  "config": { ... },
  "correctness": {
    "status": "FAILED",
    "max_diff": 0.5,
    "mean_diff": 0.2
  },
  "performance": null,
  "errors": {
    "error_type": "cuda_illegal_access",
    "error_details": "CUDA error: an illegal memory access was encountered at kernel line 89",
    "traceback": "<full Python/CUDA traceback>",
    "env_info": "<nvidia-smi output, Python/Torch/Triton versions, GPU model>"
  },
  "tuning_changes": "Reduced BLOCK_M from 64 to 32, added address clamping"
}
```

When the test **hangs** (30s timeout), the result file records it as a hang error:

```json
{
  "kernel_name": "<op>_overlap",
  "version": "v3",
  "timestamp": "2026-06-26T14:40:00Z",
  "pod": "<pod_name>",
  "iteration": 3,
  "config": { ... },
  "correctness": {
    "status": "FAILED",
    "max_diff": null,
    "mean_diff": null
  },
  "performance": null,
  "errors": {
    "error_type": "hang_deadlock",
    "error_details": "Correctness test timed out after 30s — process killed by timeout (exit code 124). Indicates kernel hang/deadlock, likely caused by signal polling spin-wait, barrier mismatch across ranks, or missing signal reset between iterations.",
    "traceback": "N/A — process timed out before producing traceback. Partial stdout captured in correctness_v3.log.",
    "env_info": "<nvidia-smi output after hang showing GPU state>"
  },
  "tuning_changes": "Changed signal polling from infinite while loop to bounded spin"
}
```

Key design principles:
- **One file per version**: no separate `perf_*.json` and `errors_*.log` — everything goes into `<kernel_name>_result_v<version>.json`
- **`errors` field**: `null` when normal (no errors), populated with structured error info when something went wrong
- **`performance` field**: `null` when correctness failed or timed out, populated with all metrics when correctness passed
- **Hang detection**: exit code 124 from the `timeout` command is the canonical hang indicator — it means the process did not complete within the 30-second budget
- **Read via script**: use `scripts/select_best_version.py` to read all result files, print progress table, and select best version

---

## Step 6: Cleanup

After collecting results, clean up remote artifacts to avoid stale state affecting subsequent runs:

```bash
# Kill any lingering torchrun processes
kubectl exec <pod> -n <namespace> -- bash -c "pkill -9 -f 'torchrun.*overlap_kernel_bench' 2>/dev/null || true"

# Remove version-specific files (keep the work directory for future runs)
kubectl exec <pod> -n <namespace> -- rm -f /tmp/overlap_kernel_bench/<kernel_name>_v<version>.py
kubectl exec <pod> -n <namespace> -- rm -f /tmp/overlap_kernel_bench/<kernel_name>_v<version>_benchmark.py
kubectl exec <pod> -n <namespace> -- rm -f /tmp/overlap_kernel_bench/results/correctness_v<version>.log
kubectl exec <pod> -n <namespace> -- rm -f /tmp/overlap_kernel_bench/results/performance_v<version>.log
```

This cleanup ensures that the next benchmark iteration starts from a clean state and avoids import conflicts if the kernel module name changes between versions.

---

## Summary: Timeout & Hang Detection Rules

| Phase | Timeout | Hang indicator | On hang |
|-------|---------|---------------|---------|
| Pod validation | 30s | No response from kubectl exec | Record `environment_failure`, STOP |
| File transfer | 30s per file | kubectl cp stalls | Retry once, then record error, STOP |
| Correctness test | **30s** | Exit code 124 from `timeout 30` | Record `hang_deadlock`, cleanup, STOP |
| Performance benchmark | **30s** | Exit code 124 from `timeout 30` | Record `hang_deadlock`, cleanup, STOP |
| Result collection | 30s | kubectl cp stalls | Retry once, then warn and continue |

**Why 30 seconds?** A correctly functioning overlap kernel correctness test should complete in 5-10 seconds on modern GPUs. A performance benchmark (10 warmup + 100 iterations) should complete in 15-25 seconds. The 30-second budget provides reasonable headroom while ensuring hangs are detected promptly. If a test cannot complete in 30 seconds, it almost certainly indicates a kernel bug (infinite spin-wait, barrier deadlock, missing signal) rather than a legitimate slow execution.