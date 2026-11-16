# Experimental adaptive prefill admission

This opt-in prototype extends the existing prefill/decode cadence control.
It does not demonstrate a GPU decode-p99 improvement. Default scheduling is
unchanged when `--adaptive-prefill` is absent.

Initial scope: single-worker, synchronous, text-only chunked generation with
FCFS admission. Overlap, TP/DP/PP/CP, speculative decoding, mixed chunks,
diffusion, dynamic chunk sizing and disaggregated serving are rejected.

For a Linux CUDA serving installation, an example configuration is:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \
  --tp-size 1 --dp-size 1 --pp-size 1 --disable-overlap-schedule \
  --schedule-policy fcfs --chunked-prefill-size 4096 --adaptive-prefill
```

This launch has not been GPU-validated. `--adaptive-prefill-target-ms` defaults
to 40, `--adaptive-prefill-token-ms` to 0.02 and
`--adaptive-prefill-max-decode-rounds` to 8. The existing
`--prefill-decode-interval` becomes a floor (at least one decode turn).

The controller observes launch-to-completed-host-result time, including host
processing. It does not add a CUDA synchronization or estimate device-only
execution time. Non-overlapped execution is required so outstanding work is not
double-counted. Completed prefill cost updates a conservative per-token estimate;
slow prefills increase the bounded decode interval, while inexpensive samples
relax it. Recent decode cost reserves time before admitting more prefill tokens.

The aggregate chunk allowance passes through the existing `PrefillAdder` budgets
and KV feasibility checks. One page is the minimum progress quantum, even if it
exceeds the soft time target. When nothing is decoding, the configured chunk size
is retained and the existing cadence bypass admits prefill immediately.

At most the configured maximum decode rounds separate prefill admission
opportunities. This is not a per-request wait bound: exhausted KV capacity,
overload and the duration of an already-running chunk can still delay progress.
No kernel preemption, hard p99 guarantee or tenant isolation is provided.

## CPU checks

With the repository's CPU test dependencies installed:

```bash
PYTHONPATH=python:. python -m pytest \
  test/registered/unit/managers/test_adaptive_prefill.py \
  test/registered/unit/managers/test_scheduler_prefill_decode_interval.py \
  test/registered/unit/managers/scheduler_components/test_dynamic_chunk_sizer.py -q
```

These exercise real scheduler dispatch methods with synthetic batches and mocked
worker/collective boundaries. They do not run a model. Real GPU correctness,
TTFT, decode p99, throughput and prefill-progress benchmarks remain pending.
Compare baseline and opt-in configurations with identical models, chunk limits,
arrival traces and synchronous execution before making any performance claim.
