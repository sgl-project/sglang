# AISimulate forward-pass estimation

Use the `aisimulate` predictor to estimate forward-pass latency through the
AISimulate SDK. For installation and the serving workflow, see the
[simulator README](../README.md).

## Choose a mode

Select the mode in the configuration file's `predictor` section:

| Use | `predictor` settings | Input |
| --- | --- | --- |
| Default op-level, no learned correction | Omit `estimation_mode` (or set `"op_level"`) | SDK systems database; no observation file |
| Calibrated op-level | `estimation_mode: "op_level"`, `estimator_config: {"correction": {"enabled": true}}`, `tune_observations` | Collected observations JSON |
| Regression | `estimation_mode: "fpm_regression"`, `tune_observations` | Collected observations JSON |
| Whole-forward interpolation | `estimation_mode: "fpm_interpolation"`, matching `systems_paths` | Materialized systems root |
| Automatic selection | `estimation_mode: "auto"` | Inputs of whichever mode the SDK selects |

Default `op_level` disables SDK correction and uses SGLang's uneven-prefill
adjustment. Enabling SDK correction disables that local adjustment, even
before tuning. Regression and interpolation use the SDK estimate directly.
`auto` selects by availability, not accuracy, and follows the selected mode's
behavior. Optional phase scales and latency floors apply afterward in all modes.

## Minimal launch

Use [aisimulate_sol.json](../examples/sim_configs/aisimulate_sol.json) for the
default mode:

```bash
SGLANG_USE_CPU_ENGINE=1 CUDA_VISIBLE_DEVICES="" \
SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE \
python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path /path/to/model \
  --sim-config-path tools/sglang-simulator/examples/sim_configs/aisimulate_sol.json \
  --port 30000
```

The configuration selects the predictor; the model comes from `--model-path`.
The simulated accelerator, precision and topology come from the
configuration's `platform` and `scheduler` sections — edit them to match the
deployment. `--sim-config-path` and `SGLANG_SIMULATOR_CONFIG_PATH` are
interchangeable.

## Collecting observations

Calibration connects two files with different roles:

- `/tmp/aisim-example/simulator_config.json` — your configuration. Copy the
  repository's [aisimulate_corrected.json](../examples/sim_configs/aisimulate_corrected.json)
  and edit it for your deployment. Pass it to `--sim-config-path`.
- `/tmp/aisim-example/calibration/TP0.ais_observations.json` — generated
  measurement data. The `sglang-simulator-collect` tool writes it at
  `/stop_profile` when AIS export is enabled. The configuration's
  `predictor.tune_observations` points to it, and the SDK reads it during
  initialization.

Observation paths are resolved relative to the configuration file's
directory, so the copied configuration's `tune_observations` value already
references the collector output.

1. Copy and edit the configuration. The sample targets A100 80 GB, TP1, SOL;
   edit `platform` and `scheduler` to match your accelerator, precision,
   topology and SGLang version, and use the same model for collection and
   simulation.

   ```bash
   mkdir -p /tmp/aisim-example
   cp tools/sglang-simulator/examples/sim_configs/aisimulate_corrected.json \
     /tmp/aisim-example/simulator_config.json
   ```

2. Start the collector with AIS export enabled. Use a fresh, empty output
   directory and a fresh server per run:

   ```bash
   sglang-simulator-collect \
     --collection-output-dir /tmp/aisim-example/calibration \
     --model-path /path/to/model \
     --collection-export-ais-observations \
     --disable-overlap-schedule --host 127.0.0.1 --port 30000
   ```

   Follow the warmup → `/start_profile` → requests → `/stop_profile` sequence
   in the [collect_baseline README](../src/sglang_simulator/tools/collect_baseline/README.md).
   Starting the server alone produces no observations; the file is written
   on successful `/stop_profile`. Cover the prefill, decode and mixed workloads
   you want to predict; regression reports an error if its data is not ready.

   Use one TP rank's file, normally `TP0`; do not combine duplicated ranks.
   `wall_time` is synchronized `Scheduler.run_batch` time in seconds, including
   CPU work inside that boundary, not HTTP latency or pure GPU kernel time.
   Collection needs neither Dynamo nor AISimulate; the SDK is needed when
   the simulator reads the observations.

3. Stop the collector and launch the simulator with the configuration file:

   ```bash
   SGLANG_USE_CPU_ENGINE=1 CUDA_VISIBLE_DEVICES="" \
   SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE \
   python3 -m sglang_simulator.simulation.sglang.launch_server \
     --model-path /path/to/model \
     --sim-config-path /tmp/aisim-example/simulator_config.json \
     --port 30001
   ```

For regression, keep the same `tune_observations` file, set
`estimation_mode` to `"fpm_regression"` and remove the `correction` setting —
no new measurement file or conversion is required.

## Interpolation profiles

`systems_paths` entries point to systems **roots** containing the system YAML
and paired `fpm_forward_perf.parquet` / `fpm_forward_perf.metadata.json` files.
Paths are relative to the configuration file; do not point them at observations
JSON or a single parquet file. Download and verify an existing matching profile
with the SDK v0.13 dataset CLI; it prints the systems root, but does not collect data:

```bash
python -m aisimulate_core.sdk.fpm_dataset /path/to/manifest.json PROFILE_KEY --cache-dir /path/to/fpm-cache
```

The manifest and profile key must match your deployment; see the bundled
[DeepSeek-V4.1 TP4 example](https://github.com/ai-dynamo/aisimulate/blob/v0.13.0/python/aisimulate/src/aisimulate_core/systems/profiles/dsv41_fpm/four_gpu_hf_dataset.json).
AISimulate v0.13's whole-forward profile collector supports only vLLM. This PR
does not add an SGLang interpolation-profile collector: the export above
produces observations, not interpolation profiles. SOL still requires profile
data; missing or out-of-domain data errors do not silently select another mode.

## Supported scope and configuration notes

- This bridge supports text-only serving with PP = attention-DP = CP = 1;
  TP/EP are passed through. Multimodal, speculative decoding and FP16 are
  unsupported.
- Set matching scheduler precision/quantization options when serving arguments
  cannot identify the SDK recipe. INT8 requires `moe_quant_mode_override`.
- `prefill_scale_factor` and `decode_scale_factor` (dimensionless) and
  `prefill_min_latency` (milliseconds) are optional post-processing applied to
  the SDK estimate after any learned correction; the defaults `1`/`1`/`0` are
  neutral.
- `database_path` is an alias for a single-entry `systems_paths`. Setting both
  fails with an explicit error.
- `enable_oom_check` accepts only `false` (or omitted); `true` is unsupported.
- `workload_distribution` cannot be passed through this predictor and fails
  with a specific error. Latency estimation uses the SDK's `power_law` default
  rather than the legacy `balanced`; memory sizing is unchanged.
