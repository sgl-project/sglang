# [Experimental] ThreadedEngine for Free-Threaded Python (no-GIL)

## Motivation

CPython 3.14t ships with a stable free-threaded build (PEP 703) that removes the GIL.
SGLang's default architecture runs the scheduler and detokenizer as **separate processes**
communicating via ZMQ IPC + shared memory. Under free-threaded Python, these components
can safely run as **threads** within a single process, eliminating all serialization overhead.

This PR adds an opt-in `ThreadedEngine` that replaces ZMQ IPC with in-process
`queue.SimpleQueue` channels, yielding measurable latency improvements for
multimodal workloads where CPU-side image processing is significant.

## Design

### Architecture (tp=1)

```
Standard Engine (multi-process):
  Main process ──ZMQ──▶ Scheduler process ──ZMQ──▶ Detokenizer process
       ▲                                                    │
       └────────────────── ZMQ ◀────────────────────────────┘

ThreadedEngine (single-process):
  Main thread ──queue──▶ Scheduler thread ──queue──▶ Detokenizer thread
       ▲                                                     │
       └────────────────── queue ◀───────────────────────────┘
```

### Key decisions

- **tp=1 only.** With tp>1 the scheduler thread contends with the
  HTTP/tokenizer threads for CPU, which slows NCCL sync and regresses
  throughput (~3× in measurements). When `SGLANG_THREADED_ENGINE=1` is
  set with tp>1, the server **fails fast** unless
  `SGLANG_THREADED_ENGINE_ALLOW_FALLBACK=1` is also set, in which case
  it falls back to the standard multi-process Engine with a warning.
- **Minimal source-file modifications.** The threaded mode lives in two
  new files (`engine_threaded.py`, `channel.py`) plus a small env-var
  branch in `http_server.py`. A handful of runtime monkey-patches in
  `engine_threaded.py` (forced `spawn` start method, multimodal SHM
  bypass, piecewise CUDA graph disable) are gated behind an explicit
  `enable_threaded_engine()` call so they only take effect when the
  threaded engine is actually used.
- **Piecewise CUDA graph disabled.** `torch.compile` inductor workers
  can deadlock in a multi-threaded free-threaded process. Regular CUDA
  graphs still work fine.
- **SHM bypass.** Since scheduler and tokenizer share an address space,
  multimodal tensors are passed by reference (zero-copy) instead of
  being copied through `SharedMemory`.

## Files changed

| File | Description |
|------|-------------|
| `python/sglang/srt/entrypoints/engine_threaded.py` | New. `ThreadedEngine`, `enable_threaded_engine()`, threaded scheduler/detokenizer/tokenizer-manager subclasses. |
| `python/sglang/srt/managers/channel.py` | New. `queue.SimpleQueue`-based IPC channel replacements (`QueueSender`, `QueueReceiver`, `AsyncQueueReceiver`, `SyncChannelPair`, `AsyncChannelPair`, `ChannelHub`). |
| `python/sglang/srt/entrypoints/http_server.py` | Env-var branch in `launch_server` that routes to `launch_threaded_server`. |
| `test/srt/test_channel.py` | New. Unit tests for the channel layer (pure-Python, runnable on 3.12+). |

## Usage

```bash
# Requires CPython 3.14t (free-threaded build)
PYTHON_GIL=0 SGLANG_THREADED_ENGINE=1 python -m sglang.launch_server \
    --model-path <model> --tp 1
```

For tp>1 the server refuses to start unless you also set
`SGLANG_THREADED_ENGINE_ALLOW_FALLBACK=1`, which makes it fall back to
the standard multi-process Engine with a warning.

If you construct `ThreadedEngine` directly (bypassing
`launch_threaded_server`), call `enable_threaded_engine(server_args)`
first — `ThreadedEngine.__init__` will refuse to run otherwise.

## Benchmark results

**Setup:** Qwen3-VL-8B-Instruct, tp=1, 4×360p images per request,
QPS=2, 200 requests, `--disable-radix-cache`, `--mem-fraction-static 0.7`.
Each configuration run 3 times and averaged.


### Triton attention backend

| Metric | Py3.12 Engine | Py3.14t ThreadedEngine | Change |
|--------|:---:|:---:|:---:|
| Mean TTFT | 290.5 ms | 269.5 ms | **-7.2%** |
| Median TTFT | 272.8 ms | 249.8 ms | **-8.5%** |
| Mean E2E | 542.5 ms | 516.9 ms | **-4.7%** |
| Median E2E | 483.9 ms | 456.8 ms | **-5.6%** |
| P99 E2E | 1299.8 ms | 1264.9 ms | -2.7% |
| Mean TPOT | 16.80 ms | 16.50 ms | -1.8% |

### FlashInfer attention backend

| Metric | Py3.12 Engine | Py3.14t ThreadedEngine | Change |
|--------|:---:|:---:|:---:|
| Mean TTFT | 290.3 ms | 268.7 ms | **-7.4%** |
| Median TTFT | 269.1 ms | 252.4 ms | **-6.2%** |
| Mean E2E | 542.2 ms | 514.9 ms | **-5.0%** |
| Median E2E | 481.3 ms | 455.5 ms | **-5.4%** |
| P99 E2E | 1299.7 ms | 1264.0 ms | -2.7% |
| Mean TPOT | 16.79 ms | 16.41 ms | -2.3% |

### Summary

- **~7–8% TTFT improvement** across both attention backends
- **~5% E2E latency reduction**
- Gains come from eliminating IPC overhead, independent of attention backend
- TPOT (GPU-bound decode) is unaffected, as expected

## Limitations

- Requires CPython 3.14t (free-threaded build).
- tp=1 only.
- pp>1 and dp>1 not supported.
- Piecewise CUDA graph disabled (regular CUDA graph works).
- Idle sleeper disabled in threaded mode (queue receivers can't poll via
  `zmq.Poller`). The trade-off is a tight scheduler poll loop, which is
  acceptable in single-process mode where there is no idle CPU to save.
- No Ray support.
