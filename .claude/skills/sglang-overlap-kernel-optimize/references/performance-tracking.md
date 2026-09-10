# Performance Tracking & Best Version Selection

This reference defines the format for tracking performance across optimization iterations and the procedure for selecting the best-performing kernel version for integration. All benchmark data (both successful and failed) is stored in **unified result JSON files** — there is no separate error log file.

---

## Unified Result File Format

Each benchmark iteration produces a **single unified JSON file** that contains both performance data and error info:

```
<kernel_name>_result_v<version>.json
```

Use the `scripts/record_benchmark_result.py` script to write result files, and `scripts/select_best_version.py` to read them and select the best version.

### Schema — Successful Benchmark

```json
{
  "kernel_name": "allgather_gemm_overlap",
  "version": "v1",
  "timestamp": "2026-06-26T14:30:00Z",
  "pod": "sglang-test-pod-abc123",
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
    "overlap_efficiency": 0.85,
    "compute_bw_gb_s": 120.5,
    "overlap_bw_gb_s": 95.3
  },
  "errors": null,
  "tuning_changes": "Initial generation — no tuning applied"
}
```

### Schema — Failed Benchmark

For versions that fail correctness, `performance` is `null` and `errors` is populated:

```json
{
  "kernel_name": "allgather_gemm_overlap",
  "version": "v2",
  "timestamp": "2026-06-26T14:35:00Z",
  "pod": "sglang-test-pod-abc123",
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
    "env_info": "<nvidia-smi output, Python/Torch/Triton versions>"
  },
  "tuning_changes": "Reduced BLOCK_M from 64 to 32, added address clamping for inline asm"
}
```

**Key design**: `errors` field is `null` when everything is normal, and populated with structured error info when something went wrong. No separate error log file exists.

---

## Comm Savings Metric

The primary optimization target is **comm savings percentage** — how much of the pure communication time the overlap kernel manages to "hide" behind compute:

```
comm_savings_pct = (non_overlap_us - overlap_us) / comm_only_us * 100%
```

- `non_overlap_us` = `compute_only_us + comm_only_us` (sequential compute + comm, no overlap)
- `overlap_us` = latency of the overlap kernel (compute and comm running concurrently)
- `comm_only_us` = pure communication time (NCCL collective alone)
- The time saved by overlap = `non_overlap_us - overlap_us`
- We compare this savings against the communication cost to quantify overlap effectiveness

**Interpretation**:
- `comm_savings_pct > 50%` means the overlap kernel hides more than half of the communication time → target achieved
- `comm_savings_pct = 100%` means communication is fully hidden (ideal: overlap time ≈ compute_only time)
- `comm_savings_pct ≈ 0%` means no overlap benefit at all (overlap time ≈ sequential time)

The `speedup` metric is still recorded for reference: `speedup = (compute_only_us + comm_only_us) / overlap_us`, but the loop termination condition is based on `comm_savings_pct`.

---

## Performance Summary File

After all optimization iterations complete, use `scripts/select_best_version.py` to consolidate all result files into a single summary:

```bash
python3 scripts/select_best_version.py \
    --kernel_name <kernel_name> \
    --results_dir ./results \
    --comm_savings_target_pct <target_pct> \
    --max_iterations <max_iterations> \
    --output_summary
```

This produces `<kernel_name>_perf_summary.json` in the results directory:

### Schema

```json
{
  "kernel_name": "allgather_gemm_overlap",
  "total_iterations": 5,
  "comm_savings_target_pct": 50,
  "target_iterations": 5,
  "versions": [
    {
      "version": "v1",
      "correctness": "PASSED",
      "speedup": 1.2,
      "overlap_us": 3456.7,
      "overlap_efficiency": 0.6,
      "comm_savings_pct": 25.0,
      "tuning_changes": "Initial generation"
    },
    {
      "version": "v2",
      "correctness": "FAILED",
      "speedup": null,
      "overlap_us": null,
      "overlap_efficiency": null,
      "comm_savings_pct": null,
      "tuning_changes": "Reduced BLOCK_M, added address clamping"
    },
    {
      "version": "v3",
      "correctness": "PASSED",
      "speedup": 1.8,
      "overlap_us": 2100.0,
      "overlap_efficiency": 0.88,
      "comm_savings_pct": 55.0,
      "tuning_changes": "Increased BLOCK_N to 512, switched to 32 warps, added tl.multiple_of hints"
    },
    {
      "version": "v4",
      "correctness": "PASSED",
      "speedup": 1.9,
      "overlap_us": 1980.0,
      "overlap_efficiency": 0.91,
      "comm_savings_pct": 60.0,
      "tuning_changes": "Optimized barrier pattern (master CTA), reduced N_CHUNKS to 4"
    },
    {
      "version": "v5",
      "correctness": "PASSED",
      "speedup": 1.85,
      "overlap_us": 2050.0,
      "overlap_efficiency": 0.89,
      "comm_savings_pct": 55.5,
      "tuning_changes": "Tried num_comm_sms=12 (regression — too many SMs for comm)"
    }
  ],
  "best_version": "v4",
  "best_speedup": 1.9,
  "best_overlap_efficiency": 0.91,
  "best_comm_savings_pct": 60.0,
  "convergence_status": "converged|no_progress|regression|max_iterations_reached|all_failed"
}
```

---

## Best Version Selection Procedure

After all optimization iterations, the lead agent selects the best-performing version for integration.

### Selection Criteria (Priority Order)

1. **Correctness MUST pass** — any version that failed correctness is excluded from consideration
2. **Highest comm_savings_pct** — among correctness-passing versions, select the one that saves the most communication time
3. **Tiebreaker: highest speedup** — if two versions have the same `comm_savings_pct`, prefer the one with higher speedup
4. **Tiebreaker: lowest overlap latency** — if still tied, prefer the one with lower `overlap_us`

### Selection Algorithm

```
1. Filter: remove all versions where correctness.status != "PASSED" (read from result JSON files)
2. Compute comm_savings_pct for each passing version: (non_overlap_us - overlap_us) / comm_only_us * 100
3. Sort: by comm_savings_pct DESC, then speedup DESC, then overlap_us ASC
4. Select: the first entry in the sorted list
5. Report: the selected version, its comm_savings_pct, speedup, and the rationale
```

This algorithm is implemented in `scripts/select_best_version.py`.

### Edge Cases

| Case | Handling |
|------|----------|
| **All versions failed correctness** | Report failure; recommend manual intervention; do NOT proceed to integration |
| **Only one version passed** | Select that version regardless of metrics; note in report that it's the only viable option |
| **Best version regressed from previous** | Still select the highest comm_savings_pct; note the regression context in the report |
| **Comm savings below target** | Select the best available; note that target was not met; proceed to integration only with user confirmation |

---

## Performance Progress Table

During the optimization loop, the lead agent maintains a running progress table for the user. This is printed automatically by `scripts/select_best_version.py`:

```bash
python3 scripts/select_best_version.py \
    --kernel_name <kernel_name> \
    --results_dir ./results \
    --comm_savings_target_pct <target_pct>
```

Example output:

```
Iteration | Version | Correctness | Speedup | Overlap (us) | Comm Savings (%) | Key Change
---------------------------------------------------------------------------------------------------------
    1     |   v1    |   PASSED    |  1.20   |    3456.7    |       25.0       | Initial generation
    2     |   v2    |   FAILED    |   N/A   |     N/A      |        N/A       | Reduced BLOCK_M, clamping
    3     |   v3    |   PASSED    |  1.80   |    2100.0    |       55.0       | Increased BLOCK_N, 32 warps
    4     |   v4    |   PASSED    |  1.90   |    1980.0    |       60.0       | Master CTA barrier, N_CHUNKS=4
    5     |   v5    |   PASSED    |  1.85   |    2050.0    |       55.5       | num_comm_sms=12 (regression)
---------------------------------------------------------------------------------------------------------
Best: v4 (speedup=1.90, comm_savings=60.0%)
Comm savings target: >50% — ACHIEVED
```

This table is updated after each benchmark sub-agent returns results, and presented to the user at the end of each iteration.

---

## Integration Handoff Package

When the best version is selected, the lead agent prepares an **integration handoff package** for the integration sub-agent. This package contains:

1. **Best kernel file**: `<kernel_name>_v<best_version>.py` — the selected kernel
2. **Best benchmark file**: `<kernel_name>_v<best_version>_benchmark.py` — the corresponding benchmark
3. **Performance summary**: `<kernel_name>_perf_summary.json` — all iteration data
4. **Integration context**: a JSON document describing:

```json
{
  "kernel_file": "allgather_gemm_overlap_v4.py",
  "kernel_version": "v4",
  "overlap_mode": "inter-sm",
  "compute_op": "GEMM",
  "comm_op": "all-gather",
  "symm_mem_mechanism": "torch.distributed._symmetric_memory",
  "target_model_layers": ["model.layers.*.mlp"],
  "forward_mode": "decode",
  "speedup": 1.9,
  "overlap_efficiency": 0.91,
  "comm_savings_pct": 60.0,
  "config": {
    "M": 1024,
    "N": 7168,
    "K": 2048,
    "dtype": "bf16",
    "num_comm_sms": 8,
    "world_size": 2
  }
}
```

The integration sub-agent uses this package to invoke the `sglang-overlap-integration` skill with all necessary context.
