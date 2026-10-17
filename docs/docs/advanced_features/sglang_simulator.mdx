---
title: "SGLang Simulator"
metatags:
    description: "Run SGLang scheduling and KV-cache simulations without loading model weights or executing GPU kernels."
---

SGLang Simulator reuses SGLang's scheduler, request lifecycle, and KV-cache implementation while replacing model forward execution with a latency predictor. Use it to compare scheduling and cache configurations on timestamped or synthetic workloads without loading model weights.

## Supported scope

SGLang Simulator tracks the current `main` branch and recent SGLang releases. The current integration is validated with `v0.5.16`, `v0.5.17`, `v0.5.18`, and `main`.

The initial upstream scope uses one simulated worker with `tp_size=1`, `ep_size=1`, `dp_size=1`, and `pp_size=1`. A simulator configuration can describe a larger target system for latency prediction, but the SGLang runtime process topology remains single-worker.

The simulator supports:

- synthetic request rates, ShareGPT workloads, and timestamped Autobench traces;
- OFFLINE logical-time simulation and BLOCKING wall-clock replay;
- AIConfigurator, ML, and replay latency predictors;
- SGLang prefix caching and [HiCache](/docs/advanced_features/hicache); and
- serving-compatible TTFT, TPOT, ITL, throughput, and cache-hit metrics.

## Install from the SGLang repository

Use the simulator and SGLang source from the same monorepo checkout:

```bash
python3 -m pip install -e tools/sglang-simulator
export PYTHONPATH="$PWD/tools/sglang-simulator/src:$PWD/python"
```

AIConfigurator is optional. Install the validated extra only when you use an AIConfigurator predictor:

```bash
python3 -m pip install -e "tools/sglang-simulator[aic]"
```

## Start a simulator server

Choose a fresh output directory for every run. The server owns the simulation mode and writes metrics to this directory.

```bash
export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE
export SGLANG_SIMULATOR_OUTPUT_DIR=/tmp/sglang-simulator-quickstart

python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path tools/sglang-simulator/test/assets/qwen3-8b \
  --tokenizer-path tools/sglang-simulator/examples/assets/tokenizer \
  --sim-config-path tools/sglang-simulator/examples/sim_configs/replay.json \
  --port 30000
```

`OFFLINE` advances the simulator's logical clock without sleeping. `BLOCKING` also sleeps for predicted forward and cache-load latency, which is useful when a client must observe simulated wall-clock pacing.

## Send a workload

In another terminal, export the same output directory and run the simulator-aware serving benchmark from the repository root:

```bash
export PYTHONPATH="$PWD/tools/sglang-simulator/src:$PWD/python"
export SGLANG_SIMULATOR_OUTPUT_DIR=/tmp/sglang-simulator-quickstart

python3 benchmark/simulator/bench_serving.py \
  --simulator-mode offline \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model tools/sglang-simulator/test/assets/qwen3-8b \
  --tokenizer tools/sglang-simulator/examples/assets/tokenizer \
  --dataset-name sharegpt \
  --dataset-path tools/sglang-simulator/examples/workloads/sharegpt-example.json \
  --sharegpt-output-len 4 \
  --num-prompts 3 \
  --output-file /tmp/sglang-simulator-quickstart/benchmark.json
```

The benchmark injects logical arrival metadata into each request and displays the server-side simulator metrics. For timestamped traffic, use the simulator-owned Autobench JSONL format and add `--use-trace-timestamps`.

## Read the results

The output directory contains:

- `metrics.json`: aggregate latency, throughput, and cache metrics;
- `request.jsonl`: per-request timing and cache information; and
- `iteration.jsonl`: scheduler batch composition and predicted iteration latency.

Use a unique output directory for each run so metrics from separate experiments are not mixed. See the [SGLang Simulator source README](https://github.com/sgl-project/sglang/tree/main/tools/sglang-simulator) for simulator configuration fields, predictor examples, and maintained tests.
