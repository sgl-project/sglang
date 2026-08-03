---
name: debug-serving-fault
description: Iterate quickly on a serving-time exception in the SGLang scheduler using --debug-mode, which discards the failed batch and keeps the process (weights + captured CUDA graphs) alive instead of crashing. Use when reproducing a NaN, sampling, attention, or batch-result bug that raises a Python exception during a forward pass, and each crash costs a full weight load + graph capture.
---

# Debugging a serving-time fault without restarting the server

## Goal

A Python exception raised while a batch runs (NaN guard, sampling assertion, attention
metadata mismatch, batch-result bookkeeping) normally kills the scheduler process. On a
large model that means paying weight load + CUDA graph capture again for every single
reproduction, which makes iterating on the bug very slow.

`--debug-mode` turns that crash into a per-batch abort: the failed batch's requests are
aborted, the loop resumes, and the process keeps its weights and captured graphs. You
send the next request and see the next reproduction in seconds.

## When this applies

Use it when **all** of these hold:

- The failure is a **CPU-side Python exception** raised inside `run_batch` or
  `process_batch_result`.
- You can reproduce it by sending requests to a server you control.
- Restart cost is what is slowing you down.

Do **not** reach for it when:

| Symptom | Use instead |
|---|---|
| CUDA illegal memory access, device-side assert | [`debug-cuda-crash`](../debug-cuda-crash/SKILL.md) — the CUDA context is poisoned, the process cannot continue |
| Hang / no progress, no exception | [`debug-distributed-hang`](../debug-distributed-hang/SKILL.md) |
| Numerical drift with no exception | [`msprobe`](../../../docs_new/docs/developer_guide/msprobe_debugging_guide.mdx) |
| Production incident triage | [`sglang-prod-incident-triage`](../sglang-prod-incident-triage/SKILL.md) |

`--debug-mode` is **dev-only**. Never enable it on a production server: it converts a
loud crash into an aborted request, which is exactly the wrong trade-off when you want
a restart and an alert.

## Step 1: Launch with `--debug-mode`

```bash
python3 -m sglang.launch_server \
    --model-path MODEL_PATH \
    --debug-mode
```

Constraints enforced at startup (the flag errors out rather than silently doing nothing):

- `tp_size == pp_size == dp_size == 1`. Multi-rank runs are in lockstep, so one rank
  discarding a batch would desync the others.
- No PD disaggregation, no PD multiplexing — they run different event loops.
- Overlap schedule is **force-disabled** (you will see a warning). Only
  `event_loop_normal` handles a failed batch.

If your bug only reproduces under TP > 1 or with the overlap scheduler, `--debug-mode`
cannot help; fall back to the normal crash-and-restart loop.

## Step 2: Reproduce, read the log, repeat

Each failure logs the traceback plus the rids it aborted:

```
Batch forward failed with 3 request(s) in flight. --debug-mode is on; aborting the
batch's requests and resuming the event loop instead of tearing down the process.
Traceback (most recent call last):
  ...
RuntimeError: <your bug>

Discarded the failed batch (aborted rids: ['abc...', 'def...']); resuming the event loop.
```

The client that sent a request in the failed batch receives a normal abort
(`finish_reason.type == "abort"`), so your repro script can loop without hanging.

**Requests that were not in the failed batch keep running.** The fault is scoped to the
batch: the waiting queue, the memory pools and the rest of the scheduler state are
untouched. This matters for reproduction — you can keep a background load running while
you poke at the failing input.

## Step 3: Narrow the input

Because the process stays warm, bisecting the trigger is cheap. Practical loop:

1. Send the failing request. Note the traceback.
2. Send a variant (shorter prompt, different sampling params, `max_new_tokens=1`,
   no logprobs, no grammar). If it survives, the difference is a candidate cause.
3. Keep halving until you have the minimal request that still raises.

Combine with the usual instrumentation — the process surviving is what makes these
practical to iterate on:

```bash
# eager execution through the graph path, to rule the graph in or out
--debug-cuda-graph

# per-forward tensor dumps
--debug-tensor-dump-output-folder DIR --debug-tensor-dump-layers N
```

## Step 4: Verify you have not been misled by a leak

The discard releases each aborted request's KV through the normal
`release_kv_cache` path, but a bug in your own patch can still leak. Arm the
scheduler's own invariant checker so a leak is a hard failure rather than a warning:

```bash
SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1
```

The checker runs in `on_idle`, which is **outside** the debug-mode `try`, so a leak
still crashes the process — the flag does not hide it. If your run survives a fault and
then goes idle without raising, the pools are intact.

To compare quantitatively, read `token_capacity` from `/get_internal_state` once the
server is idle, before and after the fault. It must be identical.

## What is deliberately not recovered

Only the batch forward is guarded. Exceptions in request intake
(`process_input_requests`) or batch scheduling (`get_next_batch_to_run`) still tear the
process down, because a half-built batch has no owner for the requests it already
popped off the waiting queue — there is no honest way to resume. If your bug is in
scheduling rather than the forward, you are back to crash-and-restart.

## Injecting a fault (for testing the mechanism itself)

There is no built-in fault injector. To exercise the path — e.g. when changing the
discard logic — add a temporary raise at the top of `Scheduler.run_batch` guarded by an
env var, and remove it before committing:

```python
# TEMPORARY -- do not commit
_inject = os.environ.get("SGLANG_TEST_INJECT_FAULT_AT")
if _inject and self.forward_ct == int(_inject):
    raise RuntimeError(f"injected fault at forward_ct={self.forward_ct}")
```

A single one-shot fault (`SGLANG_TEST_INJECT_FAULT_AT=40`) is more informative than a
recurring one: it lets you check that requests outside the failed batch still complete.
Cover a decode batch, a prefill batch, and a mid-chunk chunked prefill (small
`--chunked-prefill-size` plus a long prompt) — the chunked case is the one that exercises
the `chunked_req` pointer.

## Where the code lives

- `python/sglang/srt/managers/scheduler.py` — `event_loop_normal` (the guarded block)
  and `_discard_failed_batch` (which scheduler fields are cleared).
- `python/sglang/srt/managers/scheduler_components/debug_fault_handler.py` — the
  per-request teardown.
- `test/registered/unit/managers/test_scheduler_debug_mode.py` — CPU unit tests.

If you extend the teardown, keep it scoped to the batch. A whole-scheduler reset has to
enumerate every loop-carried field (queues, chunked slots, pending health-check IPCs,
dLLM staging, ...) and rots as scheduler state grows; a batch already carries its own
request list. The `test_discards_batch_and_clears_only_pointers_to_it` test pins this.
