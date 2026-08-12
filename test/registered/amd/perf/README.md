# AMD nightly performance benchmarks

Throughput benchmarks that run in `nightly-test-amd-rocm720.yml` (ROCm 7.2) and
`nightly-test-amd.yml` (ROCm 7.0). Each one launches a server, runs
`sglang.bench_one_batch_server` over a batch-size sweep at a fixed input/output
length, and writes a markdown table to the job's step summary plus a
`results_*.json` under `test/performance_results_*`.

## Regression gating

A benchmark that only prints its numbers cannot fail when those numbers get
worse. `sglang.test.perf_baseline` compares measured output throughput against
a recorded per-batch-size baseline and fails the test when a batch size lands
below its floor, so the drop shows up as a red job instead of a table nobody
read.

```python
from sglang.test.perf_baseline import ThroughputBaseline, check_output_throughput

PERF_BASELINE = ThroughputBaseline(
    {1: 115.4, 8: 794.5, 32: 2840.1},          # batch size -> tok/s
    recorded_from="median of 11 MI35x nightly runs, 2026-07-30..2026-08-11",
)

check = check_output_throughput(results, PERF_BASELINE, "model [MI35x]")
report += check.markdown          # publish the table first,
if not check.ok:                  # then fail on it
    self.fail(check.failure_message())
```

`PERF_BASELINE = None` reports without gating, which is where a new benchmark
starts until it has run enough nights to have a baseline.

### Recording or refreshing a baseline

1. Pull the output throughput per batch size from the last ~10 nightly runs of
   the job that owns the test (`gh run list --workflow=nightly-test-amd-rocm720.yml
   --event=schedule`, then `gh api repos/sgl-project/sglang/actions/jobs/<id>/logs`).
2. Take the median per batch size as the baseline, and check the largest
   deviation from that median across the window. It should sit well inside the
   tolerance; if it does not, the benchmark is too noisy to gate as configured.
3. Keep a `recorded_from` string naming the window, so the next person can tell
   a stale baseline from a fresh one.

Refresh a baseline whenever an intentional change moves the numbers — a new
AITER or ROCm image, a different server configuration, a kernel swap. The gate
is there to make that a deliberate edit rather than a silent drift.

### Why output throughput, and why 15%

Output throughput at a fixed batch size and input/output length is the stable
half of the benchmark. Over the 11 nightly DeepSeek-V4 MI35x runs used for the
baselines above, the worst deviation from the median was 7.0%, so the default
15% tolerance carries roughly 2x headroom while still catching the tens of
percent a toolchain regression costs. Input throughput on the same runs swung
up to 24%: it is computed from TTFT and reflects scheduling jitter more than
kernel speed, so it is reported but not gated.

The AMD nightly runs without `--enable-retry`, so a gate that fires fails the
job on the first occurrence. That is the reason for the headroom.

## Coverage

Gated on a recorded baseline:

| Benchmark | Hardware | Job |
|---|---|---|
| DeepSeek-V4-Flash FP8 + FP4 | MI35x | `nightly-8-gpu-mi35x-deepseek-v4-flash-rocm720` |
| DeepSeek-V4-Pro FP8 + FP4 | MI35x | `nightly-8-gpu-mi35x-deepseek-v4-pro-rocm720` |
| DeepSeek-V4-Pro FP4 MTP, bs=1 decode | MI35x | `nightly-8-gpu-mi35x-deepseek-v4-pro-mtp-rocm720` |

Reporting, baseline not recorded yet:

| Benchmark | Hardware | Job |
|---|---|---|
| GPT-OSS 20b + 120b (MXFP4) | MI35x | `nightly-perf-8-gpu-mi35x-gpt-oss-rocm720` |
| GPT-OSS 20b + 120b (bf16) | MI30x | `nightly-perf-8-gpu-gpt-oss-rocm720` |
| Kimi-K3 | MI35x | `nightly-perf-8-gpu-mi35x-kimi-k3-rocm720` |

Reporting only, no gate: DeepSeek-V3.1, DeepSeek-V3.2 (basic + MTP),
DeepSeek-R1-MXFP4 (plain, KV-FP8, all-reduce fusion), GLM-5.1, GLM-5-MXFP4,
Grok1-INT4, Grok2, Qwen3.5-FP8, MiniMax-M2.7, and the 2-GPU text and VLM
sweeps. Each of these can be gated by recording a baseline the same way.

## Gaps found while evaluating the benchmark dashboard

Checked against
[the CI benchmark dashboard](https://michaelzhang-ai.github.io/sglang-ci/benchmark-dashboard/)
on 2026-08-12. The dashboard's collector lives outside this repository, so the
notes below describe what this repository publishes, not what the collector
does with it.

**Perf tests that no job runs.** Nine benchmarks are registered but wired to no
workflow job, so they produce nothing to plot: `test_deepseek_v3_perf.py`,
`test_glm5_perf_amd.py`, `test_glm5_perf_mi35x.py`, `test_grok1_fp8_perf.py`,
`test_kimi_k26_perf_amd.py`, `test_kimi_k26_perf_mi35x.py`,
`test_minimax_m25_perf_amd.py`, `test_minimax_m25_perf_mi35x.py`,
`test_minimax_m27_perf_mi35x.py`. Kimi-K2.6 is the notable one: both jobs that
serve it run accuracy only, even though the perf test exists on both
architectures.

**Models with accuracy coverage but no perf benchmark.** DeepSeek-R1 HiCache,
DeepSeek-R1-MXFP4 TP2/TP4, DeepSeek-V4-Pro DSpark, GLM-5.1-MXFP4, MiniMax-M2.5
and M3 on 4 GPUs, and Qwen3.5 Triton DCP. A serving-stack regression that only
affects one of these shapes has no CI signal.

**`test_bs_1_speed` measurements that gate on a liveness floor.** A dozen AMD
model tests measure single-request decode speed alongside accuracy, and several
assert a hardcoded floor (`self.assertGreater(speed, 12)` for DeepSeek-V3 on
AMD) rather than a value recorded from that test's own history. A fixed floor
that far below the working number reports a broken server, not a slowdown.
Each is a candidate for the same `ThroughputBaseline({1: ...})` treatment
DeepSeek-V4-Pro MTP now uses.

**Result JSON is written but never collected.** Every `NightlyBenchmarkRunner`
benchmark writes `results_*.json`, and `scripts/ci/utils/save_metrics.py` knows
how to turn those into the metrics artifact the CUDA nightly uploads — but no
AMD workflow calls it, so the AMD side has no structured artifact, only step
summary markdown. Wiring `save_metrics.py` plus an artifact upload into the AMD
perf jobs would give the dashboard the same input the NVIDIA jobs provide.

**DeepSeek-V4 reports in a different shape.** The four DeepSeek-V4 tests write
their own table (`### test_perf_8k_1k (deepseek-v4-pro-fp8, unified_kv_triton)`)
with no model path, no GPU config, and no input-length column, and they write
their benchmark JSON to `/tmp` rather than to a `performance_results_*`
directory. Anything that keys off the shape every other perf test emits
(`### <model path> [<gpu config>]` over a table that carries input length) will
skip all four DeepSeek-V4 configurations, which would explain a dashboard that
shows no DeepSeek-V4 series at all.
