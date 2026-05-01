# Example: Replay-First TP Communication Hang

Use this case when:

- one request hangs instead of returning
- the symptom looks like a generic serving stall
- you want the same replay-first flow used by the crash case

Use this loop:

```text
baseline bundle
  -> save the trigger request
  -> replay on a clean target
  -> collect replay-time bundle and stacks
  -> switch to debug-distributed-hang
```

## Fault Injection

Injected shape:

1. rank 0 arms a one-shot flag only when a real extend batch satisfies
   `extend_num_tokens == 769`
2. the next TP logits `all_gather` on rank 0 is skipped
3. the peer TP rank still enters the real collective

That creates a real collective mismatch:

- rank 0 returns local data
- rank 1 waits in the collective
- the request stops making progress

One trigger prompt was:

```text
"hello " * 768
```

which tokenized to:

```text
prompt_tokens = 769
```

## 1. Collect a Healthy Bundle

```bash
python3 scripts/incident_artifact_tool.py collect-bundle \
  --base-url http://127.0.0.1:30000 \
  --outdir /tmp/incident_bundle_ok

python3 scripts/incident_artifact_tool.py summarize-bundle \
  /tmp/incident_bundle_ok
```

One baseline summary looked like:

```text
Health: /health=ok /health_generate=ok
Point-in-time load: running=0 waiting=0 total=0 token_usage=0.000 throughput=0.000
```

## 2. Save and Replay the Trigger Request

```bash
python3 -m sglang.srt.managers.configure_logging \
  --url http://127.0.0.1:30000 \
  --dump-requests-folder /tmp/sglang_request_dump_hang \
  --dump-requests-threshold 1
```

After the live hang is captured, restart a clean debug target with the same
model path and the same injection, then replay:

```bash
python3 scripts/playground/replay_request_dump.py \
  --input-folder /tmp/sglang_request_dump_hang \
  --parallel 1
```

On the replay run, the request hit the same serving path:

```text
Prefill batch, #new-seq: 1, #new-token: 769, #cached-token: 0
```

and then hung again.

## 3. Collect Replay-Time Bundle And Stacks

```bash
python3 scripts/incident_artifact_tool.py collect-bundle \
  --base-url http://127.0.0.1:30000 \
  --outdir /tmp/incident_bundle_hang
```

One replay-time bundle looked like:

```text
health.txt.error.json:
  TimeoutError: timed out

health_generate.txt.error.json:
  TimeoutError: timed out

loads_all.json:
  ConnectionResetError: [Errno 104] Connection reset by peer

loads_core_queues_disagg.json:
  URLError: <urlopen error [Errno 111] Connection refused>
```

Then let the watchdog capture the first useful stack. One TP rank showed:

```text
cuEventSynchronize
cudaEventSynchronize
synchronize (torch/cuda/streams.py:231)
process_batch_result_prefill
process_batch_result
event_loop_overlap
```

## Expected Result

1. the server is healthy before the trigger request
2. the same trigger request is saved and replayed
3. replay reproduces the same hang
4. replay-time bundle and watchdog stacks point at a distributed stall
5. the next step is `debug-distributed-hang`, not profiling
