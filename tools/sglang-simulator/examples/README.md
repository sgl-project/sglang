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
