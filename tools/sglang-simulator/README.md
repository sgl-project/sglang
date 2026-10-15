# SGLang Simulator

SGLang Simulator reuses SGLang's scheduler and cache implementation while
replacing model forward execution with a latency predictor. It supports
timestamped trace replay, synthetic workloads, hierarchical cache simulation,
and serving-compatible metrics without loading model weights.

See the [SGLang Simulator advanced-feature guide](../../docs/docs/advanced_features/sglang_simulator.mdx)
for the user-facing setup and serving workflow.

## Compatibility

SGLang Simulator tracks the current SGLang `main` branch and maintains compatibility
with recent SGLang releases. The current integration is validated with `v0.5.16`,
`v0.5.17`, `v0.5.18`, and `main`. Compatibility code uses API and capability
checks instead of branching on version numbers.

## Requirements

- A compatible SGLang checkout. The simulator uses the SGLang source from the
  same monorepo checkout.
- A local model directory containing model configuration files. Tokenizer files
  are also required unless tokenizer initialization is disabled.
- Predictor data for AISimulate, ML, or replay mode.

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

### Estimator SDK

Install the AISimulate estimator SDK through the extra below. The current
simulator shared import and memory-estimation paths also load this SDK when
using ML or replay predictors:

```bash
pip install -e "tools/sglang-simulator[aisimulate]"
```

The extra pins `aisimulate==0.12.1`, the release this integration is
validated against; other AISimulate versions are not validated. AISimulate
0.12.1 requires Python 3.11–3.13. The integration uses the estimator SDK
only — AISimulate Replay and Sweeper are out of scope. Pair this dependency
with the
`examples/sim_configs/aisimulate_sol.json` and `aisimulate_silicon.json`
example configurations, which set `scheduler.backend_version` to `0.5.10`,
the backend slot the AISimulate performance databases publish for these
systems. That field selects performance data; it is not the SGLang checkout
version.

AISimulate is the successor to AIConfigurator and is the current home of its
estimator SDK, which was originally contributed by the AIConfigurator
project; see the
[upstream transition policy](https://github.com/ai-dynamo/aisimulate#aiconfigurator-repository-transition).
In the pinned 0.12.1 release the SDK is published under the
`aiconfigurator.sdk` compatibility namespace *inside* the `aisimulate`
distribution — `aisimulate.sdk` does not exist in that release — so the
predictor adapter imports from `aiconfigurator.sdk` on purpose. The adapter
verifies distribution provenance at import time: the namespace must be
provided by the `aisimulate` distribution and the standalone legacy
`aiconfigurator` distribution must not be installed. Environments that still
carry the legacy distribution fail with an actionable error instead of
silently running unvalidated estimator code.
[`test/test_aisimulate_import.py`](test/test_aisimulate_import.py)
covers this provenance contract, and
[`test/test_aisimulate_predictor.py`](test/test_aisimulate_predictor.py)
covers the real-SDK predictor behavior.

### Migrating from the legacy AIConfigurator setup

The legacy `[aic]` extra (`aiconfigurator==0.10.0`), the `aic_sol.json` and
`aic_silicon.json` examples, and the `aiconfigurator` predictor name have
been removed; this is a breaking configuration change. Migrate an existing
setup as follows:

1. Create a fresh environment and install the AISimulate extra:

   ```bash
   pip install -e "tools/sglang-simulator[aisimulate]"
   ```

   Do not install `aiconfigurator` into it: the adapter rejects environments
   where the legacy distribution is present.

2. In the simulator configuration, rename the predictor and switch the
   database slot to the one the AISimulate databases publish:

   ```json
   "predictor": {"name": "aiconfigurator", "database_mode": "SILICON"}
   ```

   becomes

   ```json
   "predictor": {"name": "aisimulate", "database_mode": "SILICON"}
   ```

   For the supplied `a100_sxm` / `sglang` examples, move
   `scheduler.backend_version` from `0.5.9` to `0.5.10`: the pinned SDK
   provides the latter database slot for this combination. For other
   hardware or backends, select a slot actually available in their database.
   A mismatched slot fails at server startup rather than being skipped. The
   `aisimulate_sol.json` and `aisimulate_silicon.json` examples show the
   resulting configuration.

3. Re-run the predictor and compatibility tests after migrating.

Upgrade the AISimulate pin only after rerunning the predictor and
compatibility tests.

## Quick start

The maintained tests define the supported first-version scope:

- [`test/test_simulation_sglang_runner.py`](test/test_simulation_sglang_runner.py):
  direct Python use of the repository-level
  [`SGLangBenchmarkRunner`](../../benchmark/simulator/bench_runner.py);
- [`test/test_simulation_sglang_serving.py`](test/test_simulation_sglang_serving.py):
  server plus benchmark-client use through the HTTP serving path with the
  installed estimator SDK, ML, and replay predictors and ShareGPT or
  timestamped traffic;
- [`test/test_simulation_offline_blocking.py`](test/test_simulation_offline_blocking.py):
  equivalent logical results in `OFFLINE` and `BLOCKING` modes;
- [`test/test_simulation_cache_hit_ratio.py`](test/test_simulation_cache_hit_ratio.py):
  reusable-prefix accounting and cache-tier hit metrics across repeated runs;
- [`test/test_aisimulate_import.py`](test/test_aisimulate_import.py):
  estimator-SDK namespace and distribution-provenance selection;
- [`test/test_aisimulate_predictor.py`](test/test_aisimulate_predictor.py):
  real-SDK prefill/decode prediction and predictor configuration registry.

From `tools/sglang-simulator`:

```bash
python3 -m pytest -q test/test_simulation_sglang_runner.py
python3 -m pytest -q test/test_simulation_sglang_serving.py
```

Read these tests as the minimal maintained examples for constructing a dataset,
running a benchmark, starting a simulator server, sending programmatic, ShareGPT,
or timestamped traffic, comparing execution modes, and collecting request,
latency, throughput, and prefix-cache metrics.

The serving and offline-blocking tests use the AISimulate example
configurations `aisimulate_sol` and `aisimulate_silicon`, which pin the
`0.5.10` backend slot that the AISimulate performance databases publish. With
no estimator SDK installed, the SDK-dependent cases fail at server startup;
it is not skipped. The CPU CI workflow runs the full suite in a single
AISimulate environment with the real SDK installed.

## Serving mode

Choose a fresh output directory and export it in the server terminal before
starting the server:

```bash
export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE
export SIMULATOR_OUTPUT_DIR=/tmp/sglang-simulator-serving-001
test ! -e "$SIMULATOR_OUTPUT_DIR"
export SGLANG_SIMULATOR_OUTPUT_DIR="$SIMULATOR_OUTPUT_DIR"

python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path /absolute/path/to/model \
  --sim-config-path /absolute/path/to/simulator.json \
  --port 30000
```

In the benchmark terminal, export the same output directory before sending
timestamped traffic with the simulator-aware benchmark adapter:

```bash
cd /path/to/sglang
export SIMULATOR_OUTPUT_DIR=/tmp/sglang-simulator-serving-001
export SGLANG_SIMULATOR_OUTPUT_DIR="$SIMULATOR_OUTPUT_DIR"

python3 benchmark/simulator/bench_serving.py \
  --simulator-mode offline \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model /absolute/path/to/model \
  --dataset-name autobench \
  --dataset-path /absolute/path/to/trace.jsonl \
  --use-trace-timestamps \
  --num-prompts 100 \
  --profile \
  --output-file "$SIMULATOR_OUTPUT_DIR/benchmark.json"
```

The server and benchmark are separate processes, so exporting
`SGLANG_SIMULATOR_OUTPUT_DIR` in the server terminal does not configure the
benchmark terminal. The benchmark adapter reads `metrics.json` from this path
after profiling and uses those server-side logical-time metrics for its serving
table and output file. If the benchmark points at another directory, it may show
unrelated stale metrics or client wall-clock values. Use the same fresh path in
both terminals for every run.

The simulator always runs the SGLang runtime with `tp_size=ep_size=dp_size=pp_size=1`
and both attention/decode context-parallel sizes set to `1`. Parallel CLI options
accepted by SGLang are therefore ignored by this simulator entry point. This keeps
simulator-only CPU work single-process; it does not change the modeled deployment.
Set the real deployment topology under `scheduler` in `--sim-config-path`. That
topology drives predictor and cache-resource modeling without launching physical
parallel workers.

Other server options are normal SGLang command-line arguments. For direct Python
integration, see
[`test_simulation_sglang_runner.py`](test/test_simulation_sglang_runner.py); for the
process/HTTP path, see
[`test_simulation_sglang_serving.py`](test/test_simulation_sglang_serving.py).

## Simulation modes

| Mode | Behavior |
|---|---|
| `OFFLINE` | Advances the simulator's logical clock without sleeping. |
| `BLOCKING` | Sleeps for predicted forward and visible L2-to-L1 load latency. |

Use server-side simulator metrics for comparisons. Client wall-clock duration is
not the simulated timeline in `OFFLINE` mode. When using the benchmark adapter,
make sure its `SGLANG_SIMULATOR_OUTPUT_DIR` matches the server's output directory
so the printed table and `benchmark.json` are sourced from the current run's
`metrics.json`.

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
    "tp_size": 4,
    "ep_size": 4,
    "dp_size": 1,
    "pp_size": 1,
    "cp_size": 1,
    "cp_style": "none",
    "data_type": "BF16",
    "kv_cache_data_type": "BF16",
    "backend_name": "sglang"
  }
}
```

- `platform` describes the simulated accelerator and storage bandwidth.
- `predictor` selects forward-latency prediction.
- `scheduler` describes the real target deployment topology and backend metadata.
  `tp_size`, `ep_size`, `dp_size`, `pp_size`, and `cp_size` are modeled values;
  they do not launch physical workers. For the AISimulate predictor, `tp_size`
  is converted to attention TP after removing modeled DP and CP, while
  `cp_size` is passed as AISimulate context parallelism. `cp_style` uses the
  AISimulate values such as `none`, `allgather`, `ulysses`, or `ring`.
  Decode-only `dcp_size` has no separate AISimulate field and is not modeled
  yet.

### Prefix-cache accuracy

Prefix-cache hit accuracy is highly sensitive to `max_total_tokens`. It controls
the simulated device KV-cache capacity and participates in hierarchical host-cache
sizing, so a mismatch changes eviction timing and device, host, and storage hit
attribution. For deployment-faithful results, copy `max_total_num_tokens=N` from
the real SGLang server startup log and launch the simulator with
`--max-total-tokens N`. Avoid relying on a separately estimated capacity when
comparing the simulator with production traces.

Supported predictors:

| Predictor | Purpose |
|---|---|
| `aisimulate` | Operator and module performance-database estimation. |
| `ml` | A trained sklearn-compatible 18-feature latency model. |
| `replay` | Exact or nearest-neighbor batch-composition replay. |

Relative predictor paths are resolved from the simulator configuration location.
Environment variables in paths use `${NAME}` syntax.

## Workload formats

The Autobench trace format uses timestamps in milliseconds:

```json
{"prompt":[1,2,3],"prompt_len":3,"output_len":1,"timestamp":200}
```

Random and ShareGPT workloads are also supported by the runner API and serving
benchmark paths.

## Validation

Run the CPU compatibility and unit tests from the repository root, with the
`python/sglang` package importable from the same checkout (see Installation)
and one estimator extra installed:

```bash
pip install -e "tools/sglang-simulator[aisimulate]"
pip install pytest
python3 -m pytest -q tools/sglang-simulator/test/test_simulation_sglang_runner.py
python3 -m pytest -q tools/sglang-simulator/test/test_simulation_sglang_serving.py
python3 -m pytest -q tools/sglang-simulator/test/test_simulation_cache_hit_ratio.py
python3 -m pytest -q tools/sglang-simulator/test/test_simulation_offline_blocking.py
python3 -m pytest -q tools/sglang-simulator/test/test_aisimulate_import.py
python3 -m pytest -q tools/sglang-simulator/test/test_aisimulate_predictor.py
```

Run each file as a separate pytest command because the runner test installs
process-global simulator hooks and state.

Run repository checks before submitting:

```bash
git ls-files -z tools/sglang-simulator | \
  xargs -0 env SKIP=no-commit-to-branch pre-commit run --files
```

Runtime changes should also be validated in a matching official SGLang image
with both `OFFLINE` and `BLOCKING` modes. Predictor changes should report
step-level error, and scheduler or cache changes should compare request latency,
throughput, and prefix-cache reuse against measured traces.
