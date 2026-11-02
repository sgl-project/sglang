# AISimulate native forward-pass estimation

All modes use the same `aisimulate` installation, predictor and simulator
entry points. The simulator supplies deployment and scheduled-batch inputs;
AISimulate performs op-level estimation, correction, regression and interpolation.

## Choose a mode

| Use | `predictor` settings |
| --- | --- |
| Existing op-level behavior | Omit `estimation_mode`; existing phase scale factors still work. |
| Native op-level, uncorrected | `estimation_mode: "op_level"`, `estimator_config: {"correction": {"enabled": false}}` |
| Native op-level with calibration | `estimation_mode: "op_level"`, `estimator_config: {"correction": {"enabled": true}}`, `tune_observations` |
| Native regression | `estimation_mode: "fpm_regression"`, `tune_observations` |
| Whole-forward interpolation | `estimation_mode: "fpm_interpolation"`, matching `systems_paths` |
| SDK automatic selection | `estimation_mode: "auto"`; inspect the logged SDK diagnostics for its selection. |

`auto` follows the SDK's availability order, not a comparison of accuracy.
`estimator_config` is the SDK's native configuration object. For interpolation,
correction follows the SDK default; set `correction.enabled` explicitly when
an uncorrected result is desired. Regression has its own native fit controls.

Start with [this complete configuration](../examples/sim_configs/aisimulate_native.json),
then change its `predictor` section, for example:

```json
{
  "name": "aisimulate",
  "estimation_mode": "op_level",
  "estimator_config": {"correction": {"enabled": true}},
  "tune_observations": "calibration/iterations.json"
}
```

Pass the configuration using the existing `--sim-config-path` or
`SGLANG_SIMULATOR_CONFIG_PATH` entry. Model path comes from the normal model
argument; the simulated accelerator and topology come from `platform` and
`scheduler`. There is no second predictor name or installation extra.

The native path snapshots the actual scheduler phase, request membership,
new-token counts and prior KV before the forward call. This preserves mixed
batches, prefix hits and single-token prefill. Native op-level can therefore
differ from the older static path's averaged-batch approximation. Omitting
`estimation_mode` retains that older calculation; an SDK upgrade is not a
promise of identical values across SDK versions.

## Calibration inputs

`tune_observations` points to JSON containing the SDK's native iterations:
`[[iteration0_rank0], [iteration1_rank0], ...]` for the supported single
attention-DP rank. For example, one observed prefill iteration is:

```json
[[{
  "scheduled_requests": {
    "num_prefill_requests": 1,
    "sum_prefill_tokens": 128,
    "sum_prefill_kv_tokens": 0,
    "var_prefill_length": 0.0,
    "num_decode_requests": 0,
    "sum_decode_kv_tokens": 0,
    "var_decode_kv_tokens": 0.0
  },
  "wall_time": 0.008
}]]
```

This is a format example, not measured ground truth or enough data to make
every regression store ready. `wall_time` is measured forward-pass time in
**seconds**, excluding serving/HTTP latency. The adapter passes the parsed
object directly to `tune_with_fpms`; it does not define another observation
schema or train model weights. Regression returns no estimate until the
queried workload store is ready; the simulator then raises a clear error.
Correction readiness and empty-bucket behavior are determined by the SDK.

Prepare observations for the matching model, hardware, quantization, topology
and timing boundary, and keep evaluation coordinates separate from calibration.
Those identity/hash/split checks belong to offline data preparation: the native
iterations format does not make the simulator verify data provenance.
Calibration is replayed at initialization. The current SDK interface does not
provide a complete fitted-state save/load facility through this adapter.

## Profiles and supported scope

Prepare whole-forward profiles with AISimulate's own dataset tooling and pass
the materialized systems roots in `systems_paths`. Paths, including observation
paths, are relative to the simulator configuration file. Missing data and
out-of-domain SDK errors propagate; explicit modes do not silently fall back.
See the [AISimulate SDK](https://github.com/ai-dynamo/aisimulate) for its native
configuration and dataset APIs.

Native modes currently support one scheduler view with PP=attention-DP=CP=1;
TP/EP remain deployment parameters. Unsupported execution phases and topology
are rejected. FP16 is not silently relabeled BF16. Quantized serving requires
explicit, matching scheduler precision/quantization settings when the serving
arguments cannot identify the complete SDK recipe.

Legacy phase scaling, minimum latency, OOM-check and `database_path` settings
are confined to the default path. Native modes use `estimator_config` and
`systems_paths` instead. The native SDK returns milliseconds; the adapter
converts once to the simulator's seconds and adds no host overhead.

The focused tests compare the adapter with direct SDK calls and exercise the
CPU runner with mixed work and warm prefixes. The optional interpolation test
uses `SGLANG_SIMULATOR_TEST_FPM_SYSTEMS_ROOT` for an externally prepared official
B300 TP2 profile. These are integration tests, not GPU accuracy measurements.
