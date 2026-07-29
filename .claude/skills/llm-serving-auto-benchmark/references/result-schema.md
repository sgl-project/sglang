# Result Schema

Write one JSON object per candidate. Keep failed candidates in the same file so
the final summary explains what was tried.

## SLA Key Convention

One canonical naming across this skill. Config files and normalized result rows
must agree.

| Key | Where | Type |
| --- | --- | --- |
| `max_p99_ttft_ms` | both | float, milliseconds, p99 |
| `max_p99_tpot_ms` | both | float, milliseconds, p99 |
| `min_success_rate` | both | float in [0, 1] |
| `passed` | result only | bool; recomputed after the run |

Do not use `max_ttft_ms` or `max_tpot_ms` without the `p99_` prefix; those names
hide whether the target is a mean or a tail. Older cookbook configs used mean
latency targets by accident and have been migrated to the p99 names above.

The config-level SLA block lives under `benchmark.sla` (cookbook configs) or at
the top level (example plan). Either location is acceptable, but the key names
must match this table.

## JSONL Row

The values below (`gpu_model`, `gpu_count`, file paths, numeric metrics, etc.)
are illustrative. Replace them with the actual target hardware and measured
values; this schema is not tied to H100.

```json
{
  "framework": "sglang",
  "framework_version": "0.5.0",
  "framework_commit": "abcdef0",
  "candidate_id": "sglang-tp8-flashinfer",
  "model": "meta-llama/Llama-3.1-70B-Instruct",
  "status": "ok",
  "failure_reason": "",
  "hardware": {
    "gpu_model": "NVIDIA H100 80GB HBM3",
    "gpu_count": 8,
    "visible_devices": "0,1,2,3,4,5,6,7"
  },
  "workload": {
    "kind": "custom",
    "scenario": "chat",
    "dataset_path": "/bench/workload.autobench.jsonl",
    "input_len": 2048,
    "output_len": 512,
    "input_len_p50": 1800,
    "input_len_p95": 4096,
    "output_len_p50": 384,
    "output_len_p95": 1024,
    "num_prompts": 1000,
    "request_rate": 16,
    "max_concurrency": 256,
    "endpoint": "/v1/chat/completions"
  },
  "sla": {
    "max_p99_ttft_ms": 2000,
    "max_p99_tpot_ms": 80,
    "min_success_rate": 0.99,
    "passed": true
  },
  "metrics": {
    "request_throughput": 15.8,
    "output_token_throughput": 12500.0,
    "total_token_throughput": 42000.0,
    "mean_ttft_ms": 430.0,
    "p99_ttft_ms": 1550.0,
    "mean_tpot_ms": 26.0,
    "p99_tpot_ms": 72.0,
    "mean_e2e_ms": 8200.0,
    "p99_e2e_ms": 19000.0,
    "success_rate": 0.995
  },
  "server_command": "python -m sglang.launch_server ...",
  "benchmark_command": "python -m sglang.bench_serving ...",
  "validated_cli_flags": {
    "server": ["tp_size", "attention_backend"],
    "benchmark": ["dataset_name", "request_rate", "max_concurrency"]
  },
  "artifacts": {
    "server_log": "/bench/sglang/server.log",
    "raw_result": "/bench/sglang/results.jsonl",
    "server_help": "/bench/sglang/help_launch_server.txt",
    "benchmark_help": "/bench/sglang/help_bench_serving.txt"
  }
}
```

`input_len` and `output_len` are the representative scenario lengths used for
synthetic workloads or a named bucket. For custom production-like datasets,
also include p50/p95 buckets when available. These fields let
`sglang-sota-performance` pass the slow benchmark shape directly into
`llm-torch-profiler-analysis`:

- prefill profile: `--prefill-input-len <slow input len>` and
  `--prefill-output-len 1`
- decode profile: `--decode-input-len 1` and
  `--decode-output-len <slow output len>`

## Status Values

- `ok`: benchmark finished and metrics are trustworthy
- `failed`: command failed for a known non-OOM reason
- `oom`: model or candidate exhausted GPU/host memory
- `timeout`: server or benchmark timed out
- `skipped`: intentionally not run, with a reason in `failure_reason`

## Ranking Rule

The default ranking is:

1. `status == "ok"`
2. `sla.passed == true`
3. higher `metrics.request_throughput`
4. higher `metrics.output_token_throughput`
5. lower `metrics.mean_ttft_ms`
6. lower `metrics.mean_tpot_ms`
7. lower `hardware.gpu_count`

If the user cares more about token throughput than request throughput, swap
steps 3 and 4 and state that in the final report.

This ranking rule does not change the SLA gate. Keep `sla.max_p99_ttft_ms` and
`sla.max_p99_tpot_ms` as the tail-latency constraints; use mean TTFT and mean
TPOT only for default winner selection among rows that have already passed SLA.

Missing metric semantics:

- If `metrics.mean_ttft_ms` is absent from a row, the ranking script treats it
  as the worst possible value, so that row falls below any candidate with a
  real mean-TTFT measurement. Do not write `0` as a placeholder for "no
  measurement"; leave the field out or set it to `null`.
- If `metrics.mean_tpot_ms` is absent from a row, the ranking script treats it
  as the worst possible value, so that row falls below any candidate with a
  real mean-TPOT measurement. Do not write `0` as a placeholder for "no
  measurement"; leave the field out or set it to `null`.
- If `metrics.request_throughput` or `metrics.output_token_throughput` is
  missing, the row ranks below any candidate with a real measurement in those
  keys. A failed candidate that still produced partial metrics should keep the
  metrics it did produce.

## Final Report Tables

The markdown summary must include these sections:

1. `Best Commands By Framework`: one table per framework. Each table has one row
   per workload scenario and includes the best candidate, SLA result, throughput,
   latency metrics, GPU count, exact server command, and artifacts.
2. `Cross-Framework Best Comparison`: one table that compares the best SGLang,
   vLLM, and TensorRT-LLM command for each scenario. Sort each scenario by the
   ranking rule above so the best deployment choice is first.
3. `Failed Or SLA-Failing Candidates`: include this table when any candidate
   failed, was skipped, or completed without passing SLA. This table records
   tried configs that were not selected. Keep each reason concrete enough to
   tell whether the candidate needs a retry, lower concurrency, a parameter fix,
   or no further action.
