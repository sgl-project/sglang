# NCCL EP low-latency CUDA Graph tests

Shared fixtures for the registered unit tests and the two-GPU manual harness.
The harness exercises the real dispatcher, FP8 conversion, full Graph backend
and runner input buffers with synthetic experts `f_e(x) = (e + 1) * x`. It does
not load model weights or run an expert GEMM. The CPU oracle checks receive-row
contents and multiplicities independently of receive order, then checks weighted
combine with `rtol=0, atol=0` on exactly representable fixtures.

## Registered tests

From the repository root, with SGLang and pytest installed:

```bash
python -m pytest -q test/registered/unit/layers/moe/test_nccl_ep_*.py
```

`test_nccl_ep_oracle.py` and `test_nccl_ep_prefill_fallback.py` are CPU tests.
The Graph input, lifecycle, configuration and benchmark tests use one CUDA GPU;
SM89 is sufficient. They replace only the external EP communication with a narrow
test double. They do not require the NCCL EP optional dependency. Each registered
test file also supports the CI entrypoint `python file.py -f`.

The inherited four-GPU eager experiment is at
[`../manual/test_nccl_ep_synthetic.py`](../manual/test_nccl_ep_synthetic.py).
It is separate from this harness and is not a registered CI test.

## Two-GPU correctness

Install the versions and NCCL override in the
[NCCL EP guide](../../docs_new/docs/advanced_features/nccl_ep_cuda_graph.mdx).
The reproduction harness deliberately checks the tested dependency versions,
including the official Torch `2.11.0+cu130` wheel and actual loaded NCCL 2.30.7.
Use exactly two visible SM90+ GPUs with direct peer access, on one node. Select a
CUDA 13 Toolkit through `CUDA_HOME` and `PATH`; JIT setup selects the installed
EP/NCCL wheel headers and matching CUDA headers.

```bash
export PYTHONPATH="$PWD/python:$PWD/test${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_EP_REPORT_DIR="$PWD/nccl-ep-results"
python -m nccl_ep_test environment

# Each invocation starts fresh processes and records a report for each rank.
for implementation in native sglang; do
  timeout 15m torchrun --standalone --nproc-per-node=2 --module nccl_ep_test \
    eager --implementation "$implementation"
  timeout 15m torchrun --standalone --nproc-per-node=2 --module nccl_ep_test \
    eager --implementation "$implementation" --identity
  timeout 15m torchrun --standalone --nproc-per-node=2 --module nccl_ep_test \
    capture --implementation "$implementation" --buckets 8 --generations 1
  timeout 15m torchrun --standalone --nproc-per-node=2 --module nccl_ep_test \
    dynamic --implementation "$implementation" --buckets 8 16 32 \
    --layers 2 --generations 2 --replays 1000
done

timeout 15m torchrun --standalone --nproc-per-node=2 --module nccl_ep_test runner
timeout 15m torchrun --standalone --nproc-per-node=2 --module nccl_ep_test cleanup

# Separate native input-contract probes; not claims about Graph inputs.
timeout 15m torchrun --standalone --nproc-per-node=2 --module nccl_ep_test \
  zero_length --implementation native
timeout 15m torchrun --standalone --nproc-per-node=2 --module nccl_ep_test \
  duplicates --implementation native
```

The matrix changes tokens, routes and weights separately and together. It covers
balanced/hotspot routes, padding with nonzero masked weights, empty-effective
ranks, all-masked inputs, ABA bucket changes, eager interleaving, multiple layers,
stream handoffs and recapture. Cleanup injects a controlled host failure after a
completed transaction. Native/GPU/peer faults require a fresh process; `timeout`
bounds failures that block a peer. A capability skip exits 77 and is never PASS.

## Matched measurement and profiling

Run measurement after correctness passes. Keep JIT/debug logging and profiling
disabled during timing. Both modes perform the same copies, two synthetic expert
layers, receive observations and one retained combine output per layer.

```bash
timeout 15m torchrun --standalone --nproc-per-node=2 --module nccl_ep_test \
  benchmark --buckets 8 16 32 --samples 200 --warmups 20 --rounds 4
python -m nccl_ep_test summarize --reports \
  "$NCCL_EP_REPORT_DIR/benchmark-rank0.json" \
  "$NCCL_EP_REPORT_DIR/benchmark-rank1.json" > "$NCCL_EP_REPORT_DIR/summary.json"

# Profile in a separate correctness run, with graph node tracing enabled.
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none \
  --force-overwrite=true -o "$NCCL_EP_REPORT_DIR/dynamic" \
  timeout 15m torchrun --standalone --nproc-per-node=2 --module nccl_ep_test \
  dynamic --buckets 8 16 32 --layers 2 --generations 2 --replays 1000
```

Each timing sample waits for its end CUDA event. The summary takes the maximum
across aligned rank samples before calculating median/p95. These are serial
synthetic-step latencies including host launch gaps, not isolated communication
kernel latency or full-model throughput. Capture, warmup, CPU oracle work and
first-use JIT are excluded from the timed samples. Profiling adds overhead and
must not be used to derive the unprofiled speedup.

See [the recorded validation](validation.md) for the tested snapshot and results.
