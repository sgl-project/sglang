# Collect a baseline

Install the simulator package in an environment with the matching SGLang version
and its serving dependencies. From a source checkout, use
`pip install ./tools/sglang-simulator`; a wheel installation provides the same
command, which can run from any directory:

```bash
sglang-simulator-collect \
  --collection-output-dir /tmp/baseline-001 \
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
These endpoints control collection instead of the PyTorch profiler. Hooks reuse
the simulator infrastructure without patching SGLang source. Records stay in
memory until `stop_profile`; use a fresh output directory/server for each run.

Each scheduler writes three JSONL files (`TP0` shown):

| File | Contents |
| --- | --- |
| `TP0.request.jsonl` | Request ID, Unix arrival time in milliseconds, input/output lengths, disjoint device/host/disk cache-hit lengths. |
| `TP0.raw_request.jsonl` | Arrival/queue times in seconds, completion flag, token IDs, output length and token emission latencies in seconds. |
| `TP0.schedule_batch.jsonl` | Forward mode, per-request extend/past-KV lengths and synchronized `iter_latency` in seconds. |

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
steps containing any health probe are excluded from training data.

Read files after a successful stop response. Export never overwrites existing
files and retains records on failure; remove any logged partial files before
retrying. Raw traces include token IDs: review them before sharing. Continue with
[the training guide](../train_ml_time_predictor/README.md).

For a minimal GPU smoke test with a small dummy model and four requests, run:

```bash
python3 -m pytest -q tools/sglang-simulator/test/test_baseline_collection_sglang.py
```
