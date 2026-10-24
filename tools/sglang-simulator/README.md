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

The native `_build_dsv4_kv_pool` factory, when available, retains its original
dimensions, layout checks and real CPU buffers. Use a small explicit token
capacity for metadata checks. This path does not qualify older packed factories
without that seam, virtualize payload storage or establish large-capacity model
support.

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
`aiconfigurator` predictor or automatic device-memory profiling. Replay, including
its KNN fallback, can use an explicit `max_total_tokens` without importing AIC.
The `aic` extra pins AIConfigurator to the exact release
validated with the simulator so upstream API changes cannot silently alter an
installation. In a clean virtual environment, install the extra with:

```bash
pip install -e "tools/sglang-simulator[aic]"
```

In an existing SGLang image, install the same pin without dependency resolution
to avoid replacing its NumPy/CUDA stack:

```bash
pip install --no-deps "aiconfigurator==0.10.0"
```

Upgrade this pin only after rerunning the AIC predictor and compatibility tests.

## Quick start

The maintained tests define the supported first-version scope:

- [`test/test_simulation_sglang_runner.py`](test/test_simulation_sglang_runner.py):
  direct Python use of the repository-level
  [`SGLangBenchmarkRunner`](../../benchmark/simulator/bench_runner.py);
- [`test/test_simulation_sglang_serving.py`](test/test_simulation_sglang_serving.py):
  server plus benchmark-client use through the HTTP serving path with AIC, ML,
  and replay predictors and ShareGPT or timestamped traffic;
- [`test/test_simulation_offline_blocking.py`](test/test_simulation_offline_blocking.py):
  equivalent logical results in `OFFLINE` and `BLOCKING` modes;
- [`test/test_simulation_cache_hit_ratio.py`](test/test_simulation_cache_hit_ratio.py):
  reusable-prefix accounting and cache-tier hit metrics across repeated runs.

From `tools/sglang-simulator`:

```bash
python3 -m pytest -q test/test_simulation_sglang_runner.py
python3 -m pytest -q test/test_simulation_sglang_serving.py
```

Read these tests as the minimal maintained examples for constructing a dataset,
running a benchmark, starting a simulator server, sending programmatic, ShareGPT,
or timestamped traffic, comparing execution modes, and collecting request,
latency, throughput, and prefix-cache metrics.

## Serving mode

### Closed-loop workloads

In a closed-loop workload, request completion or a tool timer determines the next
request. The **workload driver** owns these dependencies, its dataset parser,
concurrency, warmup, and stopping conditions. The **simulator** owns model request
scheduling, cache state, and predicted execution time. No dataset schema or agent
framework is required by the server.

Clients can use either existing execution mode:

| | `BLOCKING`: real time | `OFFLINE` with coordinator: fast-forward |
|---|---|---|
| Client | An ordinary HTTP workload driver. | A driver using the coordinator protocol. |
| Timing | Predicted batch latency is spent sleeping; client timers use real time. | Client and server advance virtual time together. |
| Metrics | Client measurements include HTTP, tokenization, and scheduling overhead. | The driver uses simulator timestamps. |

Both paths use the same scheduler, cache, and latency predictor. The model output
is synthetic: the driver must supply response history and workload dependencies
when replaying a trace. These runs do not evaluate generated answers or agents
choosing tools from model output. Results depend on the predictor data;
the bundled replay table is an example, not a hardware performance calibration.

#### Real-time HTTP clients

Start a small example server from the repository root:

```bash
export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
export SGLANG_SIMULATOR_OUTPUT_MODE=BLOCKING
unset SGLANG_SIMULATOR_COORDINATOR

python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path tools/sglang-simulator/test/assets/qwen3-8b \
  --tokenizer-path tools/sglang-simulator/examples/assets/tokenizer \
  --chat-template chatml \
  --sim-config-path tools/sglang-simulator/examples/sim_configs/replay.json \
  --max-total-tokens 8192 \
  --disable-overlap-schedule \
  --skip-server-warmup \
  --port 30000
```

Point an HTTP driver at `http://127.0.0.1:30000`. It can use `/generate` or the
OpenAI-compatible API and issue dependent requests after responses and tool
waits. No coordinator integration is needed. Requests can run concurrently, but
a five-second tool wait takes five real seconds. For actual traces, replace the
example model configuration and tokenizer and set cache capacity for the workload.
Compare client wall-time metrics separately from simulator logical-time metrics.

#### Fast-forward clients

Restart the server with the same launch command and these environment settings:

```bash
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE
export SGLANG_SIMULATOR_COORDINATOR=tcp://127.0.0.1:30001
```

Run the two-request example:

```bash
python3 tools/sglang-simulator/examples/closed_loop.py --tool-delay 5
```

The tool delay advances virtual time without sleeping. For parallel branches
and a join, restart the server and run
`python3 tools/sglang-simulator/examples/fork_join.py --tool-delay 5`.
Both examples use `/generate`; see [the example guide](examples/README.md) to
adapt the event loop to another dataset. OpenAI HTTP/SSE requests use the same
protocol by including `rid` in the body. One server handles one run; restarting
gives the next run a clean clock and cache.

Plain `OFFLINE` cannot account for a client's future requests or tool waits.
Use `BLOCKING` for clients without coordinator support. Coordination itself
requires `OFFLINE`.

#### Driver contract

The protocol uses conservative event synchronization: the server advances only
when the driver has accounted for all work at the current virtual time. The
transport is JSON over a loopback ZeroMQ REQ/REP connection, separate from HTTP.
[`AsyncCoordinatorClient`](src/sglang_simulator/simulation/client.py) implements
the exchange; it does not execute workloads.

The adapter interface is `AsyncCoordinatorClient` and the `Ready`, `Submit`,
`Cancel`, `Grant`, and `Terminal` message types in
[`simulation.coordinator`](src/sglang_simulator/simulation/coordinator.py).
The server-side `Coordinator`, scheduler bridge, and native request objects are
implementation details; workload adapters should not depend on them.

Protocol version 1 has two messages:

| Message | Fields |
|---|---|
| `Ready` | Stable `run_id`, sequential `round` starting at 0, ordered `operations` (`{"type":"submit","rid":"..."}` or `{"type":"cancel","rid":"..."}`), optional `next_wakeup_ns`, and `finished`. A cancel may include `reason`. |
| `Grant` | Matching `run_id` and `round`, `now_ns`, `terminals` (`rid`, `kind`, `at_ns`, optional `reason`), and `status` (`CONTINUE`, `FINISHED`, or `ERROR` with `error`). Terminal kinds are `COMPLETED`, `CANCELLED`, and `ERROR`. |

Both carry `protocol_version: 1`; only one `Ready` may be outstanding. All times
are integer nanoseconds relative to the run. The earliest pending client timer
is `next_wakeup_ns`; omit it when no timer is pending. Each round follows these rules:

1. Process current completions and timers, including work their callbacks create.
   Start HTTP requests concurrently with `Ready`; its submits form the input
   barrier, so the server waits for every active submitted payload before advancing.
2. Read `Grant` and consume HTTP results for its completed requests before
   releasing dependent work. A timer-only grant needs no HTTP result; other
   requests may remain pending across rounds. Do not wait for all pending HTTP
   requests before sending the next `Ready`, or release dependencies on HTTP
   arrival alone.
3. When stopping, submit required cancels and set `finished`. Continue consuming
   terminals and sending finished rounds until `FINISHED` confirms native cleanup.

Use unique, equal-length request IDs within a run: SGLang's native cancellation
matches ID prefixes. The HTTP pool must admit every request in a round without
waiting for earlier responses; the examples use a dedicated unlimited aiohttp
pool and control workload concurrency explicitly. Protocol watchdogs use real
time to detect failures. They do not advance simulated time, and a timed-out
exchange must not be retried on the same run.

The initial scope is one local driver and scheduler, text generation with
ordinary radix cache, and completion-triggered workflows. HiCache, PD, explicit
sessions, LoRA, speculative decoding, and asynchronous grammar are rejected.
Coordinated runs preserve native `--prefill-decode-interval` countdowns when the
runtime exposes the validated cadence helper and schedule-pass surfaces. A
cadence-only pass with pending native work consumes no modeled forward time:
it neither advances the clock nor grants a future client timer. Unsupported
native cadence surfaces fail before coordinator binding. Other compatibility
guards, including speculative decoding, DP attention and overlapping execution,
remain unchanged.

Phase deadlines are driver-owned virtual timers; native wall-time waiting/running
timeouts must be disabled. HTTP/SSE transport and predictor execution time do not
contribute to virtual CPU overhead. Unmodeled frontend/CPU costs are zero.
Predicted batch durations must be positive at nanosecond resolution; a zero-cost
fallback is rejected rather than allowing an endless workload at one timestamp.

Startup generation warmup is skipped. The driver may perform workload/cache
warmup, then start its measurement window without resetting cache state. Warmup
requests and time must be excluded from benchmark results.
Administrative resets and profiling requests are rejected. Harness-specific
phase scheduling and metric aggregation remain in that harness's adapter.

#### Output and metrics

For fixed-length trace replay, request the desired output length with
`ignore_eos: true`. Prompt lengths depend on the supplied tokenizer and chat
template; use actual server token and cache counts for metrics. Trace-declared
lengths and synthetic output text do not establish cache reuse by themselves.

Coordinated `/generate` responses include a timing snapshot under
`meta_info.sglang_simulator`. Streaming chat (`/v1/chat/completions`, `n=1`)
emits the same snapshot as an additional SSE event before each native content output:

```text
event: sglang_simulator
data: {"version":1,"rid":"request0","arrival_ns":0,"first_token_ns":10000000,"last_token_ns":30000000,"commit_ns":30000000,"terminal_ns":30000000,"prompt_tokens":7,"completion_tokens":3,"cached_tokens":0}
```

All times are integer nanoseconds relative to the coordinated run. Arrival is
the `Submit` time; first/last token times are native visible-token commits after
the modeled batch finishes. `commit_ns` is the current result's commit time.
`terminal_ns` is null until native completion; first/last are null until a token
is committed. A result with no new tokens leaves first/last unchanged. Counts
come from native prompt, visible output and cache state; cache reporting is
enabled automatically for coordinated runs.

Snapshots use constant space per request and preserve first/last times when
streaming chunks are coalesced or emitted less frequently. They do not contain
per-token timestamps, so they support TTFT, completion latency and mean TPOT,
not per-token ITL percentiles. The initial empty role chunk can precede the first
snapshot. For streaming chat metrics, request `stream_options: {"include_usage": true}`
and validate the terminal snapshot against final usage before consuming `[DONE]`.
Non-streaming `/generate` returns its snapshot and counts in the JSON response.
Cancellation before a token commit retains the native abort response.

The ordinary HTTP path is covered by `test_simulation_blocking_agentic.py`;
`test_simulation_blocking_pacing.py` checks cache-load response timing.
Focused coordinator tests are `test_simulation_coordinator.py`,
`test_simulation_timing_metadata.py`, `test_simulation_coordinated_bridge.py`, and
`test_simulation_coordinated_serving.py`. Run test files in separate processes,
as simulator hooks cache configuration at import time.

### Timestamped serving workloads

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
  they do not launch physical workers. For AIConfigurator, `tp_size` is converted
  to attention TP after removing modeled DP and CP, while `cp_size` is passed as
  AIConfigurator context parallelism. `cp_style` uses the AIConfigurator values
  such as `none`, `allgather`, `ulysses`, or `ring`. Decode-only `dcp_size` has no
  separate AIConfigurator field and is not modeled yet.

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
| `aiconfigurator` | Operator and module performance-database estimation. |
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

Run the CPU compatibility and unit tests from the repository root:

```bash
pip install -e tools/sglang-simulator
python3 -m pytest -q tools/sglang-simulator/test/test_simulation_sglang_runner.py
python3 -m pytest -q tools/sglang-simulator/test/test_simulation_sglang_serving.py
```

Run the two files as separate pytest commands because the runner test installs
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

## Optional committed-output ledger

For coordinated OFFLINE runs, set
`SGLANG_SIMULATOR_TOKEN_EVENTS=/absolute/path/to/token_events.jsonl` to retain
native visible output-token commits. The parent directory must exist and the
file must be new. With the variable unset, the recorder is disabled and does no
file I/O. Recording does not change virtual batch costs or scheduling.

The JSONL stream starts with
`{"event":"start","version":1,"clock":"virtual_ns"}`. Each output-producing
batch writes an `output_commit` containing integer `at_ns` and `commits` entries
with `rid`, zero-based `output_index`, and positive incremental `token_count`.
Counts come from native visible output progression, including the first prefill
token and multi-token commits. Empty/chunk-only/aborted results add no tokens.
Previously committed tokens remain in the stream when their request is later
cancelled. This measures output, not speculative or cancelled computation whose
tokens were discarded before becoming visible.

Only successful coordinator completion writes a `finish` footer containing
`run_id`, integer `at_ns`, `status: "FINISHED"`, and `total_output_tokens`.
Interrupted or failed streams have no success footer and must not be treated as
complete results. Writes are flushed per batch; a write failure aborts the
coordinated run. Existing files are never overwritten.

The ledger contains no workload phases. A caller supplies exact integer virtual
phase boundaries and request ownership separately. For fixed-duration throughput,
sum the selected profiling requests' commits in `[start_ns, start_ns + duration_ns)`
and divide by that configured duration, retaining output from requests later
cancelled. Exclude warmup by request ownership as well as time: warmup completion
and profiling start can share a timestamp. Report grace-period output and successful
request latency cohorts separately. An early-ended measurement is incomplete and
must not be presented as a full-duration throughput point. Legacy request statistics'
requested `output_length` are not a substitute for this ledger.

The AgentX reference output-token throughput counts output from successful
profiling requests over their observed earliest-start/latest-end span. It uses a
different output cohort and denominator; report it separately from fixed-window
committed throughput.
