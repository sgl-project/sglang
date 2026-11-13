# Collect a baseline

Install the simulator package in an environment with the matching SGLang version
and its serving dependencies. From a source checkout, use
`pip install ./tools/sglang-simulator`; a wheel installation provides the same
command, which can run from any directory:

```bash
sglang-simulator-collect \
  --collection-output-dir /tmp/aisim-example/calibration \
  --model-path /path/to/model \
  --disable-overlap-schedule --host 127.0.0.1 --port 30000
```

Other flags are standard SGLang server arguments. The output directory must be
new or empty. After warmup, collect a request with:

```bash
curl -f -X POST http://127.0.0.1:30000/start_profile -H 'Content-Type: application/json' -d '{}'
curl -f http://127.0.0.1:30000/generate -H 'Content-Type: application/json' \
  -d '{"text":"Hello","sampling_params":{"max_new_tokens":16,"ignore_eos":true}}'
curl -f -X POST http://127.0.0.1:30000/stop_profile -H 'Content-Type: application/json' -d '{}'
```

Replace the generate call with your benchmark; wait for all requests before stop.
Warm up representative prompt lengths, batch sizes and mixed-request patterns
before `/start_profile`, so compilation and first-use costs do not dominate
calibration. These endpoints control collection instead of the PyTorch profiler. Hooks reuse
the simulator infrastructure without patching SGLang source. Records stay in
memory until `stop_profile`; use a fresh output directory/server for each run.

Each scheduler writes three JSONL files (`TP0` shown):

| File | Contents |
| --- | --- |
| `TP0.request.jsonl` | Request ID, Unix arrival time in milliseconds, input/output lengths, disjoint device/host/disk cache-hit lengths. |
| `TP0.raw_request.jsonl` | Arrival/queue times in seconds, completion flag, token IDs, output length and token emission latencies in seconds. |
| `TP0.schedule_batch.jsonl` | Forward mode, per-request extend/past-KV lengths and synchronized `iter_latency` in seconds. |

Collection and JSON export require neither Dynamo nor the AISimulate SDK.
The SDK is needed when the simulator consumes the observations. Add
`--collection-export-ais-observations` to the server command above; each
scheduler then also writes
`TP0.ais_observations.json` on `stop_profile`: native AISimulate tuning
observations as `[[observation], ...]`, one entry per exported step, each
holding the SDK's `scheduled_requests` metrics and a `wall_time` in seconds
equal to that step's `iter_latency`. The snapshot is taken before execution,
so mixed steps keep their scheduler-order per-request arrays and decode rows
exclude the current token from prior KV. Beam-tail rows count as computed
decode rows in the observation, while `schedule_batch.jsonl` request rows stay
tied to the batch's actual requests. The export requires PP =
attention-DP = CP = 1, a text-only model, and no speculative decoding;
unsupported setups fail `start_profile` with a clear error instead of
exporting misleading samples.
Serve it to the AISimulate predictor with the `tune_observations` option (see
[the ForwardPass guide](../../../../docs/aisimulate_fpm.md)); each TP rank writes
its own file with duplicated scheduler work, so tune from a single rank
(normally `TP0`) rather than concatenating ranks.

The generated `TP0.ais_observations.json` contains measurement data, not simulator
settings. You separately copy/edit a simulator configuration file and pass that
file to `--sim-config-path`; its `predictor.tune_observations` points to the
generated observations. No manual observation file or renaming is needed.
See the [configuration-to-observations example](../../../../docs/aisimulate_fpm.md#collecting-observations)
for the directory layout, complete configuration file and launch command.

DP/PP/EP ranks are appended when their parallel size exceeds one. Cache counts
come from SGLang's first scheduled chunk. The first token latency starts at server
arrival; subsequent entries measure time between output-producing steps.
`client_created_time` in sampling `custom_params`, when supplied in Unix seconds,
is preferred as the normalized arrival timestamp.

Step timing synchronizes the device before and after `Scheduler.run_batch`.
Features are captured before execution; start/end step timestamps use
`perf_counter`. This requires `--disable-overlap-schedule` and changes execution
timing. The initial scope is ordinary text generation, not speculative decoding
or multimodal feature modeling. Health probes are excluded from request traces;
steps containing any health probe are excluded from training data. The observed
`wall_time` is that synchronized `Scheduler.run_batch` span — including CPU
work inside the boundary, not pure GPU kernel time or HTTP latency.

Read files after a successful stop response. Export never overwrites existing
files and retains records on failure; remove any logged partial files before
retrying. Raw traces include token IDs: review them before sharing. Continue with
[the training guide](../train_ml_time_predictor/README.md).

For a minimal GPU smoke test with a small dummy model and four requests, run:

```bash
python3 -m pytest -q tools/sglang-simulator/test/test_baseline_collection_sglang.py
```

The export test in that file exercises the AIS observation export with real
mixed-chunk steps (`--enable-mixed-chunk`, a short request decoding while a
long request prefills).
