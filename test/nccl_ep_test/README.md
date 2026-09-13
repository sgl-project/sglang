# NCCL EP low-latency CUDA Graph tests

Shared fixtures for the registered unit tests and the two-GPU manual harness.
The Triton compute, zero-token rank and independent shared-expert follow-ups
have a separate [validation guide](followup_validation.md) and cost-ordered
server gate. The synthetic expert harness described below remains available.
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

## Fixed-topology EPLB

The local EPLB tests exercise the production updater, FP8 Triton compute,
received-count recording and Graph reuse without native EP. A separate CPU test
moves weights and scales between two Gloo processes, including redundant slots
and updates in layer chunks:

```bash
PYTHONPATH=python:test python -m pytest -q \
  test/registered/unit/layers/moe/test_nccl_ep_eplb.py \
  test/registered/unit/eplb/test_nccl_ep_weight_relocation.py
```

On the pinned two-GPU environment above, validate native LL dispatch/combine,
CUDA weight transfer and Graph reuse after relocation:

```bash
timeout 15m torchrun --standalone --nproc-per-node=2 \
  --module nccl_ep_test.eplb_pair \
  --replays 100 --generations 2 --report-dir "$NCCL_EP_REPORT_DIR/eplb"
```

This gate uses four logical experts, six physical slots, two layers and two
decode buckets. It alternates routing layouts and active/idle senders, verifies
weight/scale bytes after migration, compares real expert outputs to independent
unpadded Triton experts, and checks handle reuse and resource closure. Eager
interleaving uses true zero-length inputs. The native gate requires two SM90+
GPUs; local tests alone do not certify native communication.

For shared-expert/dispatch SBO, run the local model, hook and Graph tests:

```bash
PYTHONPATH=python:test python -m pytest -q \
  test/registered/unit/layers/moe/test_nccl_ep_sbo.py
```

Add `--sbo` to the pair gate to execute DeepSeek's actual `forward_deepep`
shared-expert hooks while rebalancing experts. The gate also interleaves eager
execution and true zero-token ranks. Use a separate report directory to keep
the serial and SBO results:

```bash
timeout 15m torchrun --standalone --nproc-per-node=2 \
  --module nccl_ep_test.eplb_pair --sbo \
  --replays 100 --generations 2 --report-dir "$NCCL_EP_REPORT_DIR/eplb-sbo"
```

Local tests replace EP with a narrow external-library double. They verify
ordering and numerical equivalence, not communication overlap or speedup.
Profile the native gate on the target GPUs to measure the overlap benefit.

For TBO, run the local staged executor, per-subbatch resource ownership, real
Triton/shared-MLP and Graph tests:

```bash
PYTHONPATH=python:test python -m pytest -q \
  test/registered/unit/layers/moe/test_nccl_ep_tbo.py
```

The tests use production split/pad/merge helpers and decode/prefill operation
strategies with identity attention fixtures. They include non-LIFO completion,
unequal/empty children, dynamic routing, EPLB/SBO composition and the decode
runner's TBO input-count updates. They do not load a complete serving model.

On the SM90+ pair, add `--tbo` to the EPLB gate. It uses two separate LL groups
for the staged subbatches, real shared and routed experts, received-count and
output oracles, eager decode/prefill interleaving and persistent decode Graphs:

```bash
timeout 15m torchrun --standalone --nproc-per-node=2 \
  --module nccl_ep_test.eplb_pair --tbo \
  --replays 100 --generations 2 --report-dir "$NCCL_EP_REPORT_DIR/eplb-tbo"
```

Repeat with `--sbo --tbo` for the combined configuration. Run Nsight separately
from timing; for example:

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none \
  -o "$NCCL_EP_REPORT_DIR/eplb-tbo-trace" \
  timeout 15m torchrun --standalone --nproc-per-node=2 \
  --module nccl_ep_test.eplb_pair --tbo \
  --replays 8 --generations 1 --report-dir "$NCCL_EP_REPORT_DIR/eplb-tbo-profile"
```

This correctness gate includes oracle checks and EPLB migration between replays;
its wall time is not a serving throughput benchmark. Native results for the
EPLB/SBO/TBO extensions have not yet been recorded.

For full-model serving with the extensions, first run the `env` and `single`
phases in [the serving guide](followup_validation.md#server-gates-and-cost-order)
on the current commit. Then run a feature-specific native gate and smoke test:

```bash
python -m nccl_ep_test.followup_server extensions \
  --features eplb sbo tbo --reports "$NCCL_EP_REPORT_DIR"
python -m nccl_ep_test.followup_server serve \
  --features eplb sbo tbo --reports "$NCCL_EP_REPORT_DIR"
```

You can select `eplb`, `eplb sbo`, or `eplb tbo` separately. Each configuration
gets its own report directory. Serving requires a successful native gate for
the same commit and feature set. It runs the pinned full DeepSeek-V2-Lite FP8
checkpoint, checks resolved flags and actual eager/Graph decode counters, and
requires a completed EPLB rebalance in the server log. The recipe uses two
redundant slots, rebalance every eight iterations, a four-iteration recording
window and two-layer migration chunks. TBO uses even decode buckets.

The eight serving requests use 32 generated tokens with EOS stopping disabled
to exercise a complete chunked rebalance. Finite logprobs and successful requests
are smoke-test evidence, not a model-quality or throughput benchmark.

For matched MoE timing with real shared/Triton experts, run the separate driver
after native correctness passes. Use fresh torchrun processes and separate
directories for the serial, `--sbo`, `--tbo`, and `--sbo --tbo` configurations:

```bash
timeout 15m torchrun --standalone --nproc-per-node=2 \
  --module nccl_ep_test.overlap_benchmark --sbo --tbo \
  --buckets 8 32 --samples 100 --warmups 20 --rounds 4 \
  --report-dir "$NCCL_EP_REPORT_DIR/timing-sbo-tbo"
python -m nccl_ep_test.overlap_benchmark --summarize \
  "$NCCL_EP_REPORT_DIR/timing-sbo-tbo/overlap-rank0.json" \
  "$NCCL_EP_REPORT_DIR/timing-sbo-tbo/overlap-rank1.json" \
  > "$NCCL_EP_REPORT_DIR/timing-sbo-tbo/summary.json"
```

Repeat the configuration sequence in reverse order to expose order/thermal
effects. Each process alternates eager and Graph measurement blocks, alternates
preloaded input/routing variants within samples, and checks outputs against
independent expert GEMMs outside timing. The summary aligns rank samples before
computing maximum-rank median/p95. Compare like buckets, cases and execution
modes across configurations.

This measures two independent MoE layers with fixed expert placement; it excludes
attention computation, EPLB migration, JIT, warmup, capture, barriers and oracle
checks. It uses no EP audit wrappers or receive snapshots. TBO includes its
staged host execution with identity attention fixtures. It does not establish
full-model TBO throughput. Keep profiling disabled for timing measurements.

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
