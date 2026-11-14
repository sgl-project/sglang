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

Install the AISimulate estimator SDK in a compatible environment:

```bash
pip install -e "tools/sglang-simulator[aisimulate]"
```

The extra pins `aisimulate==0.13.0`. The official wheels require Python
3.11Ã¢â‚¬â€œ3.13 and cover Linux x86_64/aarch64 and macOS arm64. This release exposes
the modern `aisimulate.sdk` namespace.

AISimulate 0.13.0 requires NumPy>=2.1,<3, replacing the NumPy 1.26.4 pin
used by standalone AIConfigurator 0.10.0. Use an environment compatible with
both the SDK and your SGLang runtime. If your existing runtime requires
NumPy 1.x, prepare a separate compatible environment.

For a new environment, install the SGLang runtime and its dependencies from
the same checkout before installing the estimator extra above. The extra
installs the simulator and estimator SDK, not the SGLang runtime dependencies;
adding source directories to `PYTHONPATH` does not install those dependencies.
No old-package uninstall is needed in a fresh environment.

Use the `aisimulate` predictor name for the existing static op-level path,
including phase scale factors.
The default memory-estimation path also uses this SDK for ML/replay predictors.
An explicit token capacity or the optional resource budget below avoids that path.

#### Migrating from AIConfigurator

To upgrade an existing compatible environment, activate that environment and
remove the standalone distributions before installing the pinned SDK extra,
as required by the AISimulate 0.13.0 migration instructions:

```bash
python3 -m pip uninstall -y aiconfigurator aiconfigurator-core
python3 -m pip install -e "tools/sglang-simulator[aisimulate]"
```

Replace each AIConfigurator artifact with its AISimulate counterpart:

| Old (AIConfigurator) | New (AISimulate) |
| --- | --- |
| Install extra `[aic]` | Install extra `[aisimulate]` |
| Predictor name `"aiconfigurator"` | Predictor name `"aisimulate"` |
| `sglang_simulator.time_predictor.aiconfigurator` | `sglang_simulator.time_predictor.aisimulate` |
| `AIConfiguratorTimePredictor` | `AISimulateTimePredictor` |
| `examples/sim_configs/aic_sol.json` | `examples/sim_configs/aisimulate_sol.json` |
| `examples/sim_configs/aic_silicon.json` | `examples/sim_configs/aisimulate_silicon.json` |

There is no compatibility layer: a simulator configuration that still uses the
`"aiconfigurator"` predictor name fails with an explicit error naming the
replacement. Update the configuration instead of relying on a fallback.

The migration keeps the SGLang-side parameter construction, uneven-batch
compensation, and phase scale factors. It does not promise unchanged numerical
results between AIConfigurator and AISimulate performance databases; re-baseline
against your own measurements.

The supplied SOL/SILICON examples select backend data version `0.5.10`; this
is a performance-database slot, not the version of the SGLang checkout running
the simulator.

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
- [`test/test_aisimulate_predictor.py`](test/test_aisimulate_predictor.py):
  real-SDK prefill/decode prediction and prefix reuse in SOL and SILICON modes.

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

## Collect baselines and train an ML predictor

Use [collect_baseline](src/sglang_simulator/tools/collect_baseline/README.md) to capture request traces
and synchronized batch steps from a real SGLang server. Then use
[train_ml_time_predictor](src/sglang_simulator/tools/train_ml_time_predictor/README.md) to train an
18-feature latency model. Installing the package provides `sglang-simulator-collect`
and `sglang-simulator-train`, including a small collected training example.

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

### Explicit CPU bootstrap

CPU compatibility is active only when both `SGLANG_SIMULATOR_BOOTSTRAP=1`
and `SGLANG_USE_CPU_ENGINE=1` are set by the simulator worker entry point.
It keeps native quantization configuration validation, while avoiding physical
accelerator discovery and import-time AITER loading for dummy model execution.
Without both flags, the CPU bootstrap wrappers remain inactive. The separate
Quark change defers its optional AITER import until the quantization call in all
execution modes; that call keeps its original arguments, result and dependency
error. This changes import timing without replacing the quantizer.

The CPU pool plumbing translates an explicit logical token capacity to the
native pool byte budget. Its bounded search rejects unreachable layouts rather
than spinning indefinitely, preserving native sizing and rounding.

### Optional SDK resource budget

`memory_budget.provider: "simopus"` selects a separate structural resource
estimate for the declared target deployment. It does not select the latency
predictor: replay, ML or another predictor can use this budget independently.
The configuration pins the SDK revision, model identity and an explicit policy.
Unsupported layouts fail rather than falling back to host memory or guessed
capacity. An explicit `--max-total-tokens` takes precedence and does not load
the resource SDK. With no `memory_budget`, existing target capacity estimation
is unchanged. Successful optional budgets are included in `metrics.json` with
their resource, policy and topology provenance.

#### Replay with an independent resource budget

[`replay_simopus_budget.json`](examples/sim_configs/replay_simopus_budget.json)
combines the existing replay table with an SDK resource budget for Qwen3-8B.
It needs the CPU bootstrap and shared SDK loader, not the SimOpus latency or
native staged provider. Install the simulator and a SimOpus SDK build exposing
`build_model_resources` and `estimate_memory_budget`; AISimulate is not required
for this example. The replay table and its 1 ms miss fallback are demonstration
latencies, not measured model performance.

From the repository root, create a local configuration pinned to the SDK that
is actually installed. Keep it beside the example so its relative replay-table
path remains valid:

```bash
python3 - <<'PY'
import json
import re
from pathlib import Path

import simopus

source = Path("tools/sglang-simulator/examples/sim_configs/replay_simopus_budget.json")
config = json.loads(source.read_text())
revision = simopus.__revision__
assert re.fullmatch(r"[0-9a-f]{40}", revision), "Install a revision-stamped SDK build"
config["memory_budget"]["provider_revision"] = revision
source.with_name("replay_simopus_budget.local.json").write_text(
    json.dumps(config, indent=2) + "\n"
)
PY

SGLANG_USE_CPU_ENGINE=1 CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" \
python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path "$PWD/tools/sglang-simulator/test/assets/qwen3-8b" \
  --tokenizer-path "$PWD/tools/sglang-simulator/examples/assets/tokenizer" \
  --sim-config-path tools/sglang-simulator/examples/sim_configs/replay_simopus_budget.local.json \
  --max-running-requests 8 --disable-overlap-schedule --page-size 1 \
  --skip-server-warmup --device cpu --port 30000
```

Do **not** pass `--max-total-tokens`: it deliberately bypasses automatic sizing.
The local model fixture supplies configuration without downloading weights;
`model_revision: null` makes no Hugging Face revision claim. The SDK records its
configuration SHA and resource digest. For an actual local Qwen3-8B checkpoint,
use its model/tokenizer directories and set `memory_budget.model_revision` to
that checkpoint's full commit rather than inventing a revision for the fixture.
The checked-in example pins the SDK revision used by its resource integration
test; the preparation command above explicitly replaces that pin for your build.

`GET /server_info` reports the selected `max_total_num_tokens`. After a profiled
request, `metrics.json.memory_budget` records the same capacity and SDK/model/
policy/topology provenance. This is a capacity simulation, not a physical OOM
guarantee; speculative, sliding-window and other unsupported pool layouts still
fail explicitly.
