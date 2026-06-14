---
name: measure-prescheduler-overhead
description: Measure SGLang request-lifetime overhead BEFORE the scheduler (HTTP, tokenizer, ZMQ) by polling Prometheus /metrics during bench_serving. Use when bench_serving alone is insufficient and the user wants to attribute TTFT to tokenize/IPC vs GPU compute.
disable-model-invocation: true
---

# Measure Pre-Scheduler Overhead

`bench_serving` reports only client-side TTFT/E2E. To attribute it to CPU work in the TokenizerManager process vs GPU compute, decompose TTFT using server `/metrics` (requires `--enable-metrics` on the server). With the histograms from this skill installed, TTFT splits exhaustively:

```
TTFT  =  chat_template  +  hf_tokenize  +  zmq_wire  +  other_pre_scheduler   ← pre-scheduler
      +  request_process  +  queue  +  prefill_forward                          ← scheduler / GPU
```

The first four buckets together are the "pre-scheduler overhead" — everything that happens to a request inside the TokenizerManager process before the scheduler picks it up. `other_pre_scheduler` is derived to close any gap; the other three are explicit Prometheus histograms.

Before the patch, only the residual `TTFT − queue − request_process − prefill_forward` is available as a single lump.

## Workflow

1. **Verify metrics endpoint and stages**
   ```bash
   python3 sgl_metrics.py peek --url http://<host>:<port>/metrics --filter sglang
   curl -s http://<host>:<port>/metrics | grep -oE 'stage="[^"]+"' | sort -u
   ```
   Expect stages: `request_process`, `prefill_forward`, `chunked_prefill`.

2. **Flush cache and health-check** — before snapshotting, clear any prior state so the `/metrics` deltas reflect only this run, and confirm the server is responsive.
   ```bash
   curl -s -X POST http://<host>:<port>/flush_cache
   curl -sf http://<host>:<port>/health_generate > /dev/null || { echo "server not healthy"; exit 1; }
   ```

3. **Snapshot before, poll during, snapshot after**
   ```bash
   curl -s http://<host>:<port>/metrics > metrics_before.txt
   nohup python3 sgl_metrics.py poll --url http://<host>:<port>/metrics \
       --output run.jsonl --interval 0.2 --print > poller.log 2>&1 &
   POLL_PID=$!

   python3 -m sglang.bench_serving --base-url http://<host>:<port> ... 2>&1 | tee bench.log

   curl -s http://<host>:<port>/metrics > metrics_after.txt
   kill -INT $POLL_PID
   ```

4. **Diff the histograms** — for each of `e2e_request_latency_seconds`, `time_to_first_token_seconds`, `queue_time_seconds`, and every `per_stage_req_latency_seconds{stage=...}`, compute `(sum_after - sum_before) / (count_after - count_before)` to get the avg over the run. The 4 tp_ranks all report identical scheduler-side numbers; use `tp_rank=0`. **Skip stages with `n=0`** (e.g. `chunked_prefill` when no request was chunked) — those are per-event metrics, not per-request, and can't be subtracted alongside per-request stages.

5. **Decompose TTFT so it sums to 100%.** With the three new pre-scheduler histograms (see next section) plus the existing scheduler-side ones, TTFT splits cleanly:

   ```
   TTFT  =  chat_template  +  hf_tokenize  +  zmq_wire  +  other_pre_scheduler
         +  request_process  +  queue  +  prefill_forward
   ```

   where `other_pre_scheduler` is a *derived* bucket that closes the gap:

   ```
   other_pre_scheduler = TTFT
                       - chat_template - hf_tokenize - zmq_wire           # explicit pre-scheduler
                       - request_process - queue - prefill_forward         # scheduler-side
   ```

   It covers `_send_one_request` preamble + `wrap_shm_features` + the bit of `_init_req_state` not bracketed by any histogram. Expect it to be small (<1% of TTFT). If `other_pre_scheduler` is large, you have unexplained tokenizer-side work — go look at `tokenizer_manager.generate_request`. The whole *pre-scheduler total* is `chat_template + hf_tokenize + zmq_wire + other_pre_scheduler` (equivalently `TTFT − request_process − queue − prefill_forward`).

6. **Sanity check with client side** — `bench duration` ≈ `Δ e2e_sum − warmup contribution`. If client E2E ≫ server E2E, suspect HTTP/network transport; otherwise the overhead is fully inside the TokenizerManager process.

## Sub-attribution via Prometheus histograms

The skill's residual (`TTFT − queue − request_process − prefill_forward`) tells you *how big* the pre-scheduler overhead is, but not which substage dominates. To split it, add three new entries to `sglang:per_stage_req_latency_seconds` — observed automatically through the same `--enable-metrics` pipeline you already use, no log parsing, no extra flags.

| New stage | Latency captured |
|---|---|
| `chat_template` | from request entry (`received_time`) to right after `_convert_to_internal_request` returns. For OpenAI chat completions this includes both chat-template rendering AND HF tokenization, because `apply_chat_template(..., tokenize=True, ...)` does both in one call on the handler thread. For native `/generate` it's just validation. |
| `hf_tokenize` | from `prompt_render_finish_time` to `_tokenize_one_request` returning. For OpenAI chat this is the TokenizerManager post-conversion bookkeeping (mm-processor checks etc.) — usually small. For `/generate` with text input this is the actual HF tokenize. |
| `zmq_wire` | from `api_server_dispatch_time` (right before `send_pyobj` on the tokenizer) to `scheduler_recv_time` (after pickle deserialize on the scheduler). Includes pickle + ZMQ send + wire + deserialize. We can't split this further from the scheduler side: `api_server_dispatch_finish_time` is set *after* `send_pyobj` returns, so it isn't in the pickled payload. |

### The patch (one file in `observability/`, light glue in three others)

All three new histograms are observed from a single site: `SchedulerReqTimeStats.set_scheduler_recv_time` (called once per request when the scheduler picks it up). The source timestamps live on `APIServerReqTimeStats` on the tokenizer side and are propagated forward via `__getstate__` — `__setstate__` already handles cross-process clock-drift via `convert_time_cross_thread`.

Concretely:

1. In `python/sglang/srt/observability/req_time_stats.py`:
   - Add a `prompt_render_finish_time: float = 0.0` field + `set_prompt_render_finish_time` setter on `APIServerReqTimeStats`.
   - Extend `APIServerReqTimeStats.__getstate__` to propagate the four perf_counter timestamps the scheduler needs (`created_time`, `prompt_render_finish_time`, `tokenize_finish_time`, `api_server_dispatch_time`).
   - Add the matching four fields on `SchedulerReqTimeStats` so `new_from_obj` copies them.
   - Register three new `RequestStage` entries (`CHAT_TEMPLATE`, `HF_TOKENIZE`, `ZMQ_WIRE`) with `metrics_is_observed=True`.
   - In `SchedulerReqTimeStats.set_scheduler_recv_time`, after setting `scheduler_recv_time`, observe the three latencies.

2. In `python/sglang/srt/entrypoints/openai/serving_base.py`, set the chat-template boundary right after `_convert_to_internal_request`:
   ```python
   adapted_request.received_time = received_time
   adapted_request.prompt_render_finish_time = monotonic_time()
   ```

3. In `python/sglang/srt/managers/io_struct.py`, add `prompt_render_finish_time: Optional[float] = None` to `GenerateReqInput` and `EmbeddingReqInput`, and propagate it on batch sub-requests (next to `received_time`).

4. In `python/sglang/srt/managers/tokenizer_manager.py` `_init_req_state`, just after `set_created_time`:
   ```python
   if getattr(obj, "prompt_render_finish_time", None):
       time_stats.set_prompt_render_finish_time(obj.prompt_render_finish_time)
   ```

After restart, `/metrics` exposes:

```
sglang:per_stage_req_latency_seconds{stage="chat_template"}
sglang:per_stage_req_latency_seconds{stage="hf_tokenize"}
sglang:per_stage_req_latency_seconds{stage="zmq_wire"}
```

alongside the existing `request_process`, `prefill_forward`, `chunked_prefill`. The same `sgl_metrics.py` before/after diff (step 3 in the Workflow above) now decomposes pre-scheduler overhead automatically.

### Caveats worth knowing

- **`chat_template` includes HF tokenize for OpenAI chat completions.** `_convert_to_internal_request` calls `apply_chat_template(..., tokenize=True, ...)`, which renders the template *and* runs the HF tokenizer in one call. So for `/v1/chat/completions` the `chat_template` histogram is the heavy bucket (~chat template + HF tokenize). The `hf_tokenize` histogram for that path is then the post-conversion bookkeeping inside `_tokenize_one_request` (mm-processor checks, etc.) and is usually small.
- **For the native `/generate` endpoint** the labeling is more accurate: `chat_template` ≈ validation only, `hf_tokenize` ≈ the real HF tokenizer run.
- **ZMQ wire can't be split further from the scheduler side.** `api_server_dispatch_finish_time` is set after `send_pyobj` returns, so it isn't in the pickled payload that reaches the scheduler. `zmq_wire` therefore covers pickle + send + wire + deserialize together; if you need the split, observe it on the tokenizer side using a separate `TokenizerMetricsCollector` per-stage histogram.

## Notes

- Warmup requests are counted in `/metrics` deltas but excluded from bench_serving's client-side numbers — account for this when comparing.
- For long-prompt + high-cache-hit workloads (e.g., 73k shared prefix), CPU tokenization frequently exceeds GPU prefill by 2-3×.
- The per-stage `request_process` histogram is the scheduler's `RecvReq` handler (~0.02 ms), NOT the tokenizer manager — don't confuse the two.

## Tool

`sgl_metrics.py` (this directory): subcommands `peek`, `poll` (JSONL @ N hz), `summarize` (histogram avg + gauge min/mean/max over the run).
