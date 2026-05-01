# Example: TTFT Spike With Low Queue Time

Use this case when:

- the service feels slow
- `/health` and `/health_generate` stay green
- queue growth is not obvious
- you want replay before trace or profiling

Use this loop:

```text
baseline bundle
  -> save the slow request
  -> replay the same request
  -> trace or profile only if replay still points to compute-side ownership
```

## 1. Collect a Bundle

```bash
python3 scripts/incident_artifact_tool.py collect-bundle \
  --base-url http://127.0.0.1:30000 \
  --outdir /tmp/incident_bundle_ttft_case

python3 scripts/incident_artifact_tool.py summarize-bundle \
  /tmp/incident_bundle_ttft_case
```

One summary looked like:

```text
Health: /health=ok /health_generate=ok
Point-in-time load: running=1 waiting=0 total=1 token_usage=0.410 throughput=29.800 cache_hit_rate=0.970
Metrics: requests=2 prompt_tokens=1540 generation_tokens=128 avg_ttft_s=3.210 avg_e2e_s=4.150 avg_queue_s=0.030
Stage Averages (max across TP ranks): prefill_forward=2.900s, request_process=0.090s
```

First signal:

- `waiting=0`
- queue time is tiny
- TTFT is still high
- `prefill_forward` dominates

That is enough to rule out queue pressure as the first explanation.

## 2. Save and Replay the Slow Request

```bash
python3 -m sglang.srt.managers.configure_logging \
  --url http://127.0.0.1:30000 \
  --dump-requests-folder /tmp/sglang_request_dump \
  --dump-requests-threshold 1
```

Replay:

```bash
python3 scripts/playground/replay_request_dump.py \
  --input-folder /tmp/sglang_request_dump \
  --parallel 1
```

Replay should preserve the same symptom:

- TTFT stays high
- queue time stays low
- the issue still looks compute-side

## Expected Result

1. the service is healthy
2. queue pressure is not the main explanation
3. the same slow request shape is reproducible through replay
4. the next step is trace or compute profiling, not queue debugging
