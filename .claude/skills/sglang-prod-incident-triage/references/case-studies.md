# Case Studies

Use these examples only after the live bundle and request dump point toward the
same class of failure. They are patterns for how to reason from replayable
evidence, not recipes to copy blindly.

## CUDA Crash: Upstream Top-K Corruption, Downstream MoE OOB

Use when a replayed CUDA crash lands in a MoE align or shared-memory kernel but
the suspicious data was produced by an earlier routing kernel.

Shape that made the original case useful:

- model family: Qwen3 MoE
- visible crash: `moe_align_block_size_kernel`
- likely producer: `topkGatingSoftmax` / MoE top-k routing
- evidence path: crash dump -> replay -> CUDA coredump -> walk one kernel
  upstream from the visible fault

Triage loop:

```text
summarize crash dump
  -> replay the exact request
  -> enable CUDA coredump on the replay target
  -> identify the failing kernel
  -> inspect the immediately preceding producer kernel and tensors
```

Key lesson: a consumer kernel can be the first one to fault even when the bad
index was produced earlier. Preserve the request shape before changing prompts.

## Latency: TTFT Spike With Low Queue Time

Use when `/health` and `/health_generate` are green, queue depth is low, but TTFT
is still high.

Signals from the original case:

- `waiting=0`
- average queue time was tiny
- TTFT was high
- scheduler stage timing pointed to prefill forward time

Triage loop:

```text
collect live bundle
  -> save the slow request
  -> replay the same request on a clean target
  -> profile only after replay reproduces compute-side ownership
```

Key lesson: rule out queue pressure with `/v1/loads`, `/metrics`, and stage
timing before opening a profiler trace.

## Distributed Hang: Request-Shaped TP Collective Mismatch

Use when one request hangs, ranks stop making progress differently, and the
failure looks like a generic serving stall until replay isolates it.

Shape that made the original case useful:

- a prompt tokenized to a specific extend length
- one TP rank skipped a logits `all_gather`
- the peer rank still entered the real collective
- the request never returned

Triage loop:

```text
collect healthy bundle
  -> save the trigger request
  -> replay on a clean target
  -> collect rank stacks and replay-time bundle
  -> switch to debug-distributed-hang
```

Key lesson: once the symptom looks like rank divergence or a collective mismatch,
do not keep profiling kernels. Preserve the replay and move to distributed-hang
debugging.
