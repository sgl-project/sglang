# SGLang Simulator

SGLang Simulator reuses SGLang's scheduler and cache implementation while
replacing model forward execution with a latency predictor. It supports
timestamped trace replay, synthetic workloads, hierarchical cache simulation,
and serving-compatible metrics without loading model weights.

## Requirements

- A compatible SGLang checkout. The simulator uses the SGLang source from the
  same monorepo checkout.
- A local model directory containing model configuration files. Tokenizer files
  are also required unless tokenizer initialization is disabled.
- Predictor data for AIConfigurator, ML, or replay mode.

Use an official SGLang image matching the checkout when validating GPU and
runtime compatibility.

## Installation

From the SGLang repository:

```bash
pip install -e tools/sglang-simulator
```

The simulator does not install or pin a second `sglang` package. Run it from a
checkout whose `python/sglang` package is available on `PYTHONPATH`, or from a
matching official SGLang image.

AIConfigurator is optional. Install it separately when using the
`aiconfigurator` predictor.

## Quick start

The repository includes a deterministic replay example:

```bash
export SGLANG_ROOT=/absolute/path/to/sglang
export PYTHONPATH="${SGLANG_ROOT}/python:${PYTHONPATH:-}"

# Update model_path in server_args.json before running.
python3 "${SGLANG_ROOT}/tools/sglang-simulator/examples/run_inprocess.py" \
  --server-args "${SGLANG_ROOT}/tools/sglang-simulator/examples/replay/server_args.json" \
  --sim-config "${SGLANG_ROOT}/tools/sglang-simulator/examples/replay/simulator.json" \
  --mode OFFLINE \
  --workload trace \
  --dataset "${SGLANG_ROOT}/tools/sglang-simulator/examples/replay/trace.jsonl" \
  --num-prompts 3 \
  --output-dir /tmp/sglang-simulator-replay
```

The run writes:

- `result.metrics.json`
- `result.request.jsonl`
- `result.iteration.jsonl`
- resolved copies of the server and simulator configuration

## Serving mode

Set the simulator mode and output directory before starting the server:

```bash
export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE
export SGLANG_SIMULATOR_OUTPUT_DIR=/tmp/sglang-simulator-serving

python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path /absolute/path/to/model \
  --sim-config-path /absolute/path/to/simulator.json \
  --port 30000
```

Send timestamped traffic with the simulator-aware benchmark adapter:

```bash
python3 -m sglang_simulator.simulation.bench_serving \
  --simulator-mode offline \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model /absolute/path/to/model \
  --dataset-name autobench \
  --dataset-path /absolute/path/to/trace.jsonl \
  --use-trace-timestamps \
  --num-prompts 100 \
  --warmup-requests 0 \
  --profile
```

Server options are normal SGLang command-line arguments. The in-process example
constructs the same `ServerArgs` object from JSON.

## Simulation modes

| Mode | Behavior |
|---|---|
| `OFFLINE` | Advances the simulator's logical clock without sleeping. |
| `BLOCKING` | Sleeps for predicted forward and visible L2-to-L1 load latency. |

Use server-side simulator metrics for comparisons. Client wall-clock duration is
not the simulated timeline in `OFFLINE` mode.

## Configuration

A simulator configuration has three sections:

```json
{
  "platform": {
    "accelerator": {"name": "h20_sxm"},
    "disk_read_bandwidth_gb": 8,
    "disk_write_bandwidth_gb": 8,
    "memory_read_bandwidth_gb": 64,
    "memory_write_bandwidth_gb": 64,
    "num_device_per_node": 1
  },
  "predictor": {
    "name": "replay",
    "database_path": "/absolute/path/to/replay_table.json"
  },
  "scheduler": {
    "tp_size": 1,
    "ep_size": 1,
    "dp_size": 1,
    "data_type": "BF16",
    "kv_cache_data_type": "BF16",
    "backend_name": "sglang"
  }
}
```

- `platform` describes the simulated accelerator and storage bandwidth.
- `predictor` selects forward-latency prediction.
- `scheduler` describes target parallelism and backend metadata. It does not
  launch physical tensor-parallel workers.

Supported predictors:

| Predictor | Purpose |
|---|---|
| `aiconfigurator` | Operator and module performance-database estimation. |
| `ml` | A trained sklearn-compatible 18-feature latency model. |
| `replay` | Exact or nearest-neighbor batch-composition replay. |

Relative predictor paths are resolved from the simulator configuration location.
Environment variables in paths use `${NAME}` syntax.

## Workload formats

The Autobench trace format uses timestamps in milliseconds:

```json
{"prompt":[1,2,3],"prompt_len":3,"output_len":1,"timestamp":200}
```

Random and ShareGPT workloads are also supported by the in-process and serving
benchmark paths.

## Validation

Run the CPU compatibility and unit tests from the repository root:

```bash
pip install -e tools/sglang-simulator
python3 -m pytest -q tools/sglang-simulator/test/unit
```

Run repository checks before submitting:

```bash
git ls-files -z tools/sglang-simulator | \
  xargs -0 env SKIP=no-commit-to-branch pre-commit run --files
```

Runtime changes should also be validated in a matching official SGLang image
with both `OFFLINE` and `BLOCKING` modes. Predictor changes should report
step-level error, and scheduler or cache changes should compare request latency,
throughput, and prefix-cache reuse against measured traces.
