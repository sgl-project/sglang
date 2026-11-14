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
3.11–3.13 and cover Linux x86_64/aarch64 and macOS arm64. This release exposes
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
The shared memory-estimation path also uses this SDK for ML/replay predictors.

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
- [`test/test_simulation_pd_e2e.py`](test/test_simulation_pd_e2e.py):
  1P1D and 2P2D runs through the repository-level
  [`PDDisaggBenchmarkRunner`](../../benchmark/simulator/pd_bench_runner.py).

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

## Prefill-decode disaggregation

[`PDDisaggBenchmarkRunner`](../../benchmark/simulator/pd_bench_runner.py) runs
one workload on one or more prefill instances and one or more decode instances
in the same process. All instances go through SGLang's disaggregation queues,
but no KV cache moves between them: the prefill sender and the decode receiver
are SGLang's fake implementations, and the transfer time is modelled in between.

```python
import os

os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = "/absolute/path/to/simulator.json"

from sglang_simulator.simulation.benchmark import BenchmarkConfig
from benchmark.simulator.pd_bench_runner import PDDisaggBenchmarkRunner
from sglang.srt.server_args import ServerArgs

common = dict(model_path="/path/to/model", load_format="dummy",
              chunked_prefill_size=8192, page_size=256)
runner = PDDisaggBenchmarkRunner(
    ServerArgs(**common, disaggregation_mode="prefill",
               disaggregation_transfer_backend="mooncake"),
    ServerArgs(**common, disaggregation_mode="decode"),
    output_dir="/tmp/sglang-simulator-pd",
)
try:
    # `dataset` is any simulator dataset, such as a timestamped trace.
    metrics = runner.benchmark(
        BenchmarkConfig(ignore_request_timestamp=False), dataset
    )
finally:
    runner.shutdown()
```

For several instances, pass a list of `ServerArgs` per role, each prefill
instance with its own `disaggregation_bootstrap_port`, and optionally a
`PDRouter`. Like the SGLang PD router, it picks the prefill and the decode
instance of each request independently, in arrival order. It runs the gateway's
own policies (`round_robin`, the default, `random`, `cache_aware`,
`power_of_two`, `manual`, `bucket` for prefill, `consistent_hashing` and
`prefix_hash`) through `sglang_router`, which must be built from
`sgl-model-gateway/bindings/python` in this tree. Keyword arguments are the
router's policy flags, with `launch_router`'s defaults:

```python
from benchmark.simulator.pd_bench_runner import PDRouter

runner = PDDisaggBenchmarkRunner(
    [ServerArgs(**common, disaggregation_mode="prefill",
                disaggregation_transfer_backend="mooncake",
                disaggregation_bootstrap_port=8998 + i) for i in range(2)],
    [ServerArgs(**common, disaggregation_mode="decode") for _ in range(2)],
    router=PDRouter(prefill_policy="cache_aware", decode_policy="power_of_two",
                    cache_threshold=0.3),
)
```

As in the gateway, a request counts against both its instances from routing
until it completes, and `cache_aware`, `power_of_two` and `prefix_hash` read
that count. Routing precedes the simulation, so completion times come from
`PDRouter(end_times={rid: seconds})`, e.g. a measured run, or else the runner
simulates the workload `routing_rounds` times (default 2), routing each round
with the completion times of the round before. The router passes the prompt
text when a policy needs it and, like the HTTP router, no tokens; `manual` and
`consistent_hashing` read the routing key from `custom_params["headers"]`.
Wall-clock parts of the policies, such as bucket boundary adjustment and cache
tree eviction, do not follow the simulated clock.

The run has two passes with a transfer step between them:

1. The prefill pass replays the workload's timestamps, or arrivals generated
   once from `request_rate` when `ignore_request_timestamp` is set. Each prefill
   instance runs its own requests, concurrently with the others, and records
   when each EXTEND chunk of a request finishes.
2. The runner turns those chunks into the KV sends SGLang makes: one send per
   chunk after the decode bootstrap completes, page-aligned except for the last.
   A prefill prefix hit already on the device is sent on its own when its batch
   starts, ahead of the forward, as SGLang does by default. A send moves whole
   pages, so a partial last page is charged as a full one. Sends of one request
   are serial. As in SGLang, each prefill instance shards its sends over
   `transfer_queues` queues by decode instance, and each queue sends one wave at
   a time. Queues moving data out of one prefill instance, or into one decode
   instance, share its bandwidth.
3. The decode pass always starts each request at the end of its own prefill and holds
   it in the transfer queue until the part of the transfer that prefill did not
   hide has elapsed.

`metrics` reports client-visible TTFT and TPOT across both passes, plus lists of
per-instance metrics under `prefill` and `decode`; the prefill metrics include
the `kv_layout` (page size, KV bytes per token per rank, chunked prefill size)
the waves were built from. Each request statistic also carries
`prefill_instance`, `decode_instance`, `kv_xfer_send_s` (summed send time),
`kv_xfer_visible_s` (the wait decode observed), `kv_xfer_n_waves`, and
`kv_xfer_wave_tokens`. All times are on the workload's timeline.

Requirements:

- The prefill instance needs a real transfer backend because SGLang rejects
  `fake` on a prefill server. With `mooncake`, the Mooncake transfer engine is
  initialised at startup, so `mooncake-transfer-engine` must be installed, but
  nothing is sent.
- The decode instance always uses the `fake` backend, so decode radix cache is
  unavailable and the whole prompt is transferred.

Configure the transfer model with `kv_transfer_config` in the simulator
configuration:

```json
"kv_transfer_config": {
  "network": "EXAMPLE_RDMA",
  "network_profiles": "../network_profiles/example_profiles.json",
  "seed": 0
}
```

A send costs `alpha + bytes / bandwidth` plus a sampled excursion.

| Key | Meaning |
|---|---|
| `network` | Profile name looked up in `network_profiles`; see below. Explicit keys below override it. |
| `network_profiles` | Path to a JSON file of profiles by name, or the mapping itself. A relative path is tried from the working directory, then from the configuration file's directory and its parents. Required with `network`. |
| `alpha_us` | Per-send fixed cost. Default `0`. |
| `bandwidth_gbps` | Effective bandwidth of one rank's share of a send. Default: the accelerator's nominal inter-node bandwidth. |
| `bootstrap_alpha_s` | Time after arrival at which the decode bootstrap completes. Nothing is sent before it: earlier chunks are merged into the next send, and a prefill that finishes first waits for it. Default `0`, for a bootstrap that completes before a request's first batch. |
| `excursion` | Delays above the alpha-beta line, in the profile format below. |
| `max_inflight_waves` | Sends of different requests that may overlap in one transfer queue. Default `1`, as a Mooncake queue sends one chunk at a time. |
| `transfer_queues` | Transfer queues per prefill instance (SGLang's `SGLANG_DISAGGREGATION_QUEUE_SIZE`). Decode instance `j` uses queue `j % transfer_queues`. Default `4`. |
| `visible_scale` | Scales only the wait decode observes; send times are unchanged. Default `1.0`. |
| `chunked_prefill` | `false` sends each request in one wave. |
| `seed` | Seed for excursion sampling, derived per request so runs are reproducible. |

Without `kv_transfer_config`, decode starts as soon as prefill finishes. An
unknown network or an invalid value raises an error.

No profiles are bundled, because fitted values describe the hardware, model
and parallelism they were measured on.
[`examples/network_profiles/example_profiles.json`](examples/network_profiles/example_profiles.json)
shows the format with made-up values, and
[`examples/sim_configs/pd_ml.json`](examples/sim_configs/pd_ml.json) uses it;
replace its predictor paths with bundles trained for your deployment. A profile
file maps each network name to:

| Key | Meaning |
|---|---|
| `alpha_us` | Fixed cost of one send, in microseconds. |
| `bandwidth_gbps` | Effective bandwidth of one rank's share of a send, in GB/s. |
| `excursion.bands` | Delay distributions by send size. A send uses the first band, by ascending `max_bytes` (bytes per rank), that it is smaller than; a band without `max_bytes` has no limit. With `probability`, the send is delayed by a draw from `magnitude_s`, a `{seconds: weight}` distribution. |
| `excursion.later_wave_exc_scale` | Scales the delay of every wave after a request's first. Default `1`. |

To fit a profile, record the bytes per rank and the elapsed time of every KV
send on the deployment you simulate; a send takes as long as its slowest rank.

1. Send prompts of several fixed lengths one at a time, and fit
   `elapsed = alpha + bytes / bandwidth` to the median send time of each size.
2. Replay a representative trace and take each send's excursion, its elapsed
   time above that line. Per size band, `probability` is the share of sends
   whose excursion exceeds a small threshold, and `magnitude_s` is the
   histogram of those excursions.
3. Set `later_wave_exc_scale` to the mean excursion of later waves divided by
   that of first waves.

For a DeepSeek-V4 model, the KV bytes per token are measured from the KV pool,
whose padded c4, c128 and indexer sub-pools the model config cannot describe.
Other models keep the size derived from the model config.

The `ml` predictor makes two independent choices. By instance role, a prefill
instance loads `prefill_database_path` when it is set, and every other instance
loads `database_path`. By batch phase, a bundle with separate `prefill` and
`decode` regressors runs `decode` for batches whose requests all extend by one
token and `prefill` for every other batch; `prefill_scale_factor` and
`decode_scale_factor` scale the two phases, as they do for `aisimulate`.
Set `scheduler.ignore_cpu_overhead` to `true` to keep the simulator's own host
time off the simulated clock.

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
