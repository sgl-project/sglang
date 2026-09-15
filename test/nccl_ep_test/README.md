# NCCL EP regression fixtures

These fixtures support the registered tests for NCCL EP LL, CUDA Graph
ownership, Triton expert compute, zero-token ranks, shared experts, EPLB,
and the staged SBO/TBO execution paths.

Run from the repository root with SGLang and pytest installed:

```bash
PYTHONPATH=python:test python -m pytest -q \
  test/registered/unit/layers/moe/test_nccl_ep_*.py \
  test/registered/unit/eplb/test_nccl_ep_weight_relocation.py \
  test/registered/unit/batch_overlap/test_tbo_logprob_metadata.py
```

The CUDA tests use one GPU; SM89 is sufficient. They exercise production
Graphs, dispatchers, runner input buffers and expert kernels with a narrow
one-rank replacement for the external EP bindings. They do not validate native
NCCL EP communication. The oracle and prefill fallback tests also run on CPU;
the weight relocation test uses two CPU Gloo processes.

The fixtures retain independent routing/output oracles, temporary tensor
lifetime checks, capture/replay ordering, shutdown/recapture failures, empty
subbatches and per-lane resource ownership. Each registered test file exposes
the repository CI entrypoint.

The full-model benchmarks, native multi-rank experiments, Nsight analysis,
rental-session drivers and their own unit tests are maintained separately.
Their [original versions](https://github.com/Laceprndpm/sglang/tree/abec649bf47333da7d343d38ddca29b51eab9635/test/nccl_ep_test)
remain available with their historical validation results. Those results name
the tested revision; they do not establish a new native pass for this checkout.

## Multistream regression

Add `--enable-nccl-ep-multistream` to a compatible NCCL EP LL configuration
with `--enable-two-batch-overlap` and/or `--enable-single-batch-overlap`.
The option is disabled by default and requires single-node CUDA, DeepSeek
V2/V3 block-128 FP8 experts and the Triton MoE runner. EPLB migration is not
supported with this option.

Each TBO lane retains its own communication stream, group, handle and scratch.
With separate Triton attention children, attention TP=1 and dense TP=1, the
existing staged executor also submits each lane's compute on a separate
stream. Other attention backends keep compute on the existing stream because
their children may share mutable workspace. A shared zero allocator keeps its
backing storage alive across both streams; unsupported allocator types and
TP collective paths retain the existing compute stream. Graph replays and
buckets remain serial. Streams are initialized during warmup and reused
during capture and replay.

The tests cover producer/consumer ordering, independent lane completion,
captured tensor lifetimes, shared allocator slices, dynamic routing and the
existing staged model paths:

```bash
PYTHONPATH=python:test python -m pytest -q \
  test/registered/unit/layers/moe/test_nccl_ep_multistream.py \
  test/registered/unit/layers/moe/test_nccl_ep_multistream_config.py
```

These tests use real CUDA streams and Graphs with external EP replaced.
They detect completion incorrectly waiting for independent compute, but do
not measure native communication overlap or model speedup. The opt-in
batch-invariant router and full-model logit diagnostics are separate changes.
