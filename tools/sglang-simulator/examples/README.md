# SGLang Simulator examples

The example assets are organized by purpose:

- `sim_configs/`: standalone AIC SOL, AIC SILICON, ML, and replay simulator configs;
- `assets/`: the small illustrative ML model, replay table, and test tokenizer;
- `workloads/`: ShareGPT and timestamped simulator/Autobench workload examples;

The ML model is an illustrative constant-latency sklearn model, not a calibrated
hardware predictor. Rebuild it and the tokenizer with:

```bash
python3 examples/build_example_assets.py
```

Only load pickle/joblib assets from sources you trust.

For maintained direct-run and serving examples, see
[`test_simulation_sglang_runner.py`](../test/test_simulation_sglang_runner.py) and
[`test_simulation_sglang_serving.py`](../test/test_simulation_sglang_serving.py).

Start a server with any example config:

```bash
python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path /path/to/model \
  --sim-config-path examples/sim_configs/aic_sol.json \
  --port 30000
```

Run a ShareGPT workload with at least four output tokens so decode and TPOT are
measured:

```bash
cd /path/to/sglang
python3 benchmark/simulator/bench_serving.py \
  --simulator-mode=offline \
  --backend=sglang \
  --base-url=http://127.0.0.1:30000 \
  --model=/path/to/model \
  --tokenizer=/path/to/model \
  --dataset-name=sharegpt \
  --dataset-path=examples/workloads/sharegpt-example.json \
  --sharegpt-output-len=4 \
  --num-prompts=3 \
  --profile
```

The timestamp trace uses the simulator-owned Autobench JSONL contract. Its
`timestamp` values are request-arrival times in milliseconds.

## Closed-loop drivers

Start the CPU example server in the main README's
[closed-loop guide](../README.md#closed-loop-workloads), using `OFFLINE` and
`SGLANG_SIMULATOR_COORDINATOR=tcp://127.0.0.1:30001`. From the repository root:

```bash
python3 tools/sglang-simulator/examples/closed_loop.py --tool-delay 5
```

This sends a request, waits five virtual seconds, then sends its dependent
request. Restart the server to run the fork/join example:

```bash
python3 tools/sglang-simulator/examples/fork_join.py --tool-delay 5
```

Its workflow is `root → tool timer → {left, right} → join`. The branches have
different output lengths, so the driver must handle separate completion events
and submit the join only after both branches finish. The simulator determines
completion times from its scheduler and latency predictor.

These are explicit event loops built on `AsyncCoordinatorClient`, with ordinary
HTTP for model requests. To use another dataset:

1. Parse its records into request bodies, dependencies, and tool delays in the
   driver. Preserve recorded history when subsequent prompts depend on it.
2. At each grant, consume the matching completed HTTP results, update dependency
   state, and submit newly runnable requests. Keep other HTTP requests pending.
3. Register the earliest workload timer in `Ready`, and finish only after all
   work and cancellation cleanup have drained.

No server change or shared dataset format is needed. A larger harness can keep
its existing scheduler and implement the same
[driver contract](../README.md#driver-contract) in an adapter. The server models
request execution; parsing a dataset, choosing branches, separating warmup, and
aggregating workload metrics remain client responsibilities.
