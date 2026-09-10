# ascend_bench: configuration-driven benchmark sweeps for SGLang

`ascend_bench` automates "launch → warm up → benchmark → verify → rank" for
SGLang deployments, with first-class support for Ascend NPU hosts. One YAML
file describes a whole sweep: the server argument grid, the workload grid,
SLA thresholds, and the repeat policy. The tool expands the grid into cells,
executes them one by one, hard-verifies that HBM is freed between cells, and
produces ranked, reproducible reports.

Core features:

- **Config-driven sweeps** — cartesian product over server/workload axes;
  every key in the YAML is a literal `sglang.launch_server` /
  `sglang.bench_serving` CLI flag.
- **Hard HBM gate** — after each cell the tool kills the server process
  group and polls `npu-smi info` until per-device HBM is back near the
  pre-run baseline; orphaned workers holding tens of GB are killed and the
  cell is flagged instead of poisoning the next one.
- **SLA filtering + ranking** — cells violating any configured percentile
  threshold are eliminated; survivors are ranked by output throughput with
  TPOT p99 as tiebreak; repeats produce mean ± std, and cells whose
  variance exceeds `cv_max` are marked unrankable rather than trusted.
- **Accuracy smoke gate** — optional gsm8k run per cell (reuses
  `benchmark/gsm8k/bench_sglang.py`); below-floor accuracy flags the perf
  data as untrustworthy.
- **Regression diff** — `compare.py` joins two runs by `cell_hash` and
  reports deltas for baseline/regression verification.
- **Provenance** — every run records sglang/torch_npu/CANN/driver versions,
  git sha, NPU topology, env vars, and the exact server argv.

## Quick start

```bash
# from a sglang checkout on the benchmark host
python benchmark/ascend_bench/run_sweep.py \
    --config benchmark/ascend_bench/configs/cookbook/qwen3-8b-bf16.yaml \
    --workdir /mnt/ascend_bench_runs

# inspect what would run without launching anything
python benchmark/ascend_bench/run_sweep.py --config <cfg> --dry-run

# resume an interrupted run: pass the printed run_id again
python benchmark/ascend_bench/run_sweep.py --config <cfg> --run-id <run_id> \
    --workdir /mnt/ascend_bench_runs

# compare two runs (regression check)
python benchmark/ascend_bench/compare.py RUN_A RUN_B \
    --metrics output_throughput p99_ttft_ms p99_tpot_ms
```

Artifacts per run live under `<workdir>/<run_id>/`:

```
manifest.jsonl            append-only: provenance header + per-cell transitions
cells/<cell_id>/          server.log, bench.jsonl, gsm8k.out per cell
report.md / report.json   ranked table, SLA matrix, compatibility matrix
```

Exit code is `0` only when every cell reached `done`.

## Config schema (see `configs/cookbook/` for complete examples)

```yaml
name: my-sweep
model:    {path, dtype, quantization, trust_remote_code}
server:   {args: {--flag: value}, env: {VAR: value}, axes: {--flag: [v1, v2]}}
workload: {dataset_name, args, axes, num_prompts_mult}
sla:      {thresholds: {p99_ttft_ms: 2000}, cv_max: 0.15}
run:      {repeats, seed, warmup_requests, timeouts, python, gsm8k}
```

`axes` are the search grid; axis values override base `args` for that flag.
On Ascend NPU, `--cuda-graph-bs` is validated to ≤ 10 entries at load time
(graph-capture stream-conflict crash).

## Hardware-free tests

```bash
python -m pytest benchmark/ascend_bench/tests -q
```

The core package (`asc_bench/`) imports neither `torch` nor `sglang` — a
unit test enforces it — so the logic gate runs in any CI. All NPU-specific
parsing lives in `asc_bench/npu.py`, the single relocation seam.
