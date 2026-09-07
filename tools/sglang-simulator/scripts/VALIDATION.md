# InferCast simulator accuracy validation

Compare simulator-side InferCast predictions against measured serving traces.

## Prerequisites

- InferCast FIDB slice matching `scheduler.backend_version`
- Measured serving `results.jsonl` at concurrency 1 (no-cache)
- SGLang Simulator branch with InferCast predictor

## Quick micro-forward check (no GPU)

Uses the same per-forward UMD calls as `InferCastTimePredictor`, replaying
concurrency=1 EXTEND + DECODE steps for fixed ISL/OSL:

```bash
python tools/sglang-simulator/scripts/micro_simulate_infercast.py \
  --systems-root /path/to/perfdb \
  --system mi355x \
  --version 0.5.17 \
  --tp-size 8 \
  --real-root tools/sglang-simulator/scripts/real_bench_mi355x
```

## Full simulator validation (recommended for PR)

```bash
export SGLANG_SIMULATOR_CONFIG_PATH=tools/sglang-simulator/examples/sim_configs/infercast_silicon.json
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE
python benchmark/simulator/bench_runner.py ...
```

Read TTFT/TPOT/duration from server `metrics.json`, not client wall clock.

## MI355X reference run (2026-09-07)

Ground truth: `/home/jxing/workspace/bench/QPDX_q355/Qwen_*` on
`smci355-ccs-aus-m15-17`, Qwen3-32B-FP8, tp8, concurrency 1.

| Workload | TTFT err | TPOT err | ITL err | Input TPS err | Duration err |
|:---|---:|---:|---:|---:|---:|
| 512/512 | 75.6% | 10.4% | 10.4% | 13.0% | 88.9% |
| 1024/512 | 77.7% | 10.3% | 10.3% | 126.0% | 88.9% |
| 1024/1024 | 77.9% | 10.6% | 10.6% | 12.6% | 88.9% |
| 2048/1024 | 97.7% | 10.4% | 10.4% | 146.6% | 89.9% |

Decode-step TPOT tracks silicon within ~10%. TTFT/duration gaps are expected
until full SGLang Simulator OFFLINE runs include scheduler CPU overhead and
FIDB prefill calibration catches long-context prefill.
