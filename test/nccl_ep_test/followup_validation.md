# NCCL EP follow-up validation

These changes build on the persistent Graph lifecycle at
`537b7a17477a6621efd727f5f088848d754ed3b4` (upstream PR #38683, dependent on #32329).

## Triton compute compatibility

Select `--moe-a2a-backend nccl_ep --moe-runner-backend triton` explicitly.
The adapter supports ordinary block-128 FP8, float32 activation scales and
gated SiLU. It preserves the LL wire contract and delegates the original
routing weights to NCCL combine. It does not need the native DeepEP library
or DeepGEMM for expert computation.

This is a compatibility path: dequantization, requantization, alignment and
scratch scale with expert receive capacity. No LL performance parity is claimed.

Local compute tests ran on RTX 4060 Laptop (SM89), Torch `2.13.0+cu130`.
They execute real Triton GEMMs and ordinary CUDA Graphs, without native EP.
CPU checks use fan-in-normalized random weights and `rtol=atol=0.02`.
The higher-gain stress case is separately checked against existing unpadded
single-expert Triton GEMMs: FP8 GEMM rounding amplified by the second
quantization does not satisfy that CPU tolerance for every input. Structured
inputs can also reach this boundary with normalized weights, especially after
routed scaling; the CPU tolerance is not a universal FP8 error bound.
Eager/Graph valid-row comparisons are exact. Invalid payload and scale tails
are poisoned with NaN to verify masking.

Run from the checkout root:

```bash
PYTHONPATH=python:test python -m nccl_ep_test.single_gpu --report /tmp/nccl-ep-compute.json
```

Before two-GPU integration, run the same gate on one PRO 6000, adding
`--require-sm 120 --experts 32 --capacity 128`. This records actual peak memory
and replay time. The completed SM120 and native validation record is below.
Server Torch 2.11 and local Torch 2.13 are separate validation records.

## Zero-token rank integration

The first certified configuration targets one node, TP=DP=EP=2, DP attention,
dense TP=1 and DP LM head. Independent local buckets are retained where
`require_mlp_tp_gather` is false. Gathered configurations retain their existing
common bucket behavior. Logical empty ranks replay the smallest positive bucket
with zero valid tokens and return zero local rows, while continuing EP work.

One fixed-size exchange on the existing EP CPU/Gloo group precedes the runtime
Graph/eager decision, including IDLE and prefill. It coordinates both admission
and capture-hidden mode: no peer may rebuild the native group while another
replays its old generation. This control overhead must be measured on the pair.

Local tests cover real Gloo agreement with two CPU processes, production
ModelRunner branch ordering, actual zero-length IDLE runner inputs, scheduler
idle creation and coordinated recapture. Native independent-bucket interoperability is established separately by the
two-GPU acceptance gate recorded below.

## Serial independent shared experts

For the explicit NCCL EP + Triton combination, DeepSeek's independent shared
MLP runs on the current stream before routed expert dispatch. An existing
auxiliary stream must not implicitly enable overlap. The Blackwell overlap
environment switch is rejected for this combination. Routed scaling remains
owned by the existing model composition; the shared result is never multiplied
by the routed scaling factor.

Single-GPU tests load actual DeepSeek gate/up/down weights into the existing
MLP, including separate gate/up checkpoint shards. They cover one/two shared
experts, BF16/block-FP8, model dimensions, eager/replay agreement, CPU arithmetic,
empty inputs and routed scaling both inside and outside top-k weights.
A combined test runs the real dispatcher, Triton GEMMs, shared MLP and Graph
backend with only the external EP library replaced by the restricted test
double. It checks active/idle transitions and one persistent handle.

## Server gates and cost order

Use the dependency installation and NCCL library override in the
[existing reproduction guide](README.md#two-gpu-correctness). The native harness
pins official Torch `2.11.0+cu130`, nccl4py `0.4.1`, nccl-extensions `0.1.0` and
NCCL `2.30.7`. Reuse the CUDA 13.0 Toolkit; do not build Torch from source.
The server must run the same commit for all phases. A failed or stale preceding
gate blocks the next phase. Each phase records a JSON status and logs.

```bash
export PYTHONPATH="$PWD/python:$PWD/test${PYTHONPATH:+:$PYTHONPATH}"
export EP_FOLLOWUP_REPORTS="$PWD/nccl-ep-followup-results"
python -m nccl_ep_test.followup_server env --reports "$EP_FOLLOWUP_REPORTS"
python -m nccl_ep_test.followup_server single --reports "$EP_FOLLOWUP_REPORTS"
# Continue only after the single SM120 compute and shared-MLP tests pass.
python -m nccl_ep_test.followup_server pair --reports "$EP_FOLLOWUP_REPORTS"
python -m nccl_ep_test.followup_server serve --reports "$EP_FOLLOWUP_REPORTS"
```

`env` records the driver, Toolkit and Torch CUDA separately, checks loaded
NCCL/EP library identity and checks JIT headers/compiler without compiling EP.
`single` exposes only the first selected GPU and requires SM120. Its largest
compute probe uses 32 local experts, hidden 2048, intermediate 1408 and receive
capacity 128 (two ranks times the eager budget of 64). It reports eager/replay
time and peak PyTorch allocation. These are compute-only measurements.

`pair` requires two SM120 GPUs with bidirectional P2P. It runs actual NCCL EP,
Triton GEMMs and a shared MLP through the production ModelRunner branch and
decode runner with controlled model/attention/router fixtures. Cases include
`[8,8]`, `[0,16]`, `[32,0]`, `[0,33]`, `[9,1]`, `[1,17]`, recovery to `[8,8]`,
two layers, two capture generations and 1000 changing replays per generation.
It checks receive contents/multiplicity against the CPU routing oracle and
combined outputs against independent unpadded single-expert Triton GEMMs plus a
standalone shared MLP, with CPU routing and weighting. This arithmetic oracle
does not use the EP adapter or its receive buffers. It retains `rtol=atol=0.02`.
The CPU arithmetic oracle additionally records maximum absolute error and
whether it met 0.02, without treating that diagnostic as the native gate.
It also requires an
idle rank to perform remote expert work. Host admission and forward submission
times include peer waits; they are not isolated GPU kernel times.

`serve` runs eager then Graph on
`gaunernst/DeepSeek-V2-Lite-Chat-FP8`, pinned to revision
`2f6d5dd458e5d9673719d03d30c488866be52e1a`. It checks the resolved server
configuration, sends serial and concurrent requests, requires finite output
logprobs and positive completion lengths, and verifies positive actual eager
or Graph decode counters in `/metrics`. The recipe uses TP=DP=EP=2, DP attention,
dense TP=1, DP LM head, unfused shared experts, Triton attention/FP8 GEMMs,
decode buckets 1/8/16/32 and a per-rank prefill cap of 64 after DP adjustment.
Local configuration tests parse and resolve this exact recipe without downloading
weights; native bindings and SM120 capability are replaced for that test only.

The driver bounds subprocess execution and terminates its worker process group
on timeout/failure. A failed native rank requires a fresh process. The driver
does not connect to, provision or rent a machine.

For profiling, rerun the pair gate separately after correctness passes:

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none \
  -o "$EP_FOLLOWUP_REPORTS/followups" \
  python -m nccl_ep_test.followup_server pair --reports "$EP_FOLLOWUP_REPORTS"
```

Keep profiling disabled when comparing latency. Full-model quality benchmarks,
throughput optimization, fused shared experts, EPLB, SBO/TBO and deterministic
fallback are outside this follow-up's validation scope. Record the exact tested
commit for each hardware result; never reuse a preceding gate from another SHA.

In the local combined structured-input case, receive quantization matched the
CPU wire reference exactly. One routed expert differed from CPU by up to
0.02539; scaling/combination amplified the final difference to 0.05078. The
independent Triton arithmetic oracle passed at 0.02. This limitation is reported
explicitly instead of increasing the CPU tolerance or attributing the mismatch
to communication.

## Local measurement (SM89)

The model-capacity compute probe completed 1000 replays on the local RTX 4060
Laptop with Torch `2.13.0+cu130`: 32 experts, capacity 128, H=2048, I=1408.
CPU reference tolerance was `rtol=atol=0.02`. Indicative eager time was 1.279 ms
and replay time 1.174 ms, both including count-buffer updates. Peak PyTorch
allocation was 388,546,048 bytes (370.5 MiB). These measurements use default
Triton tuning and exclude native EP, model attention, full model memory and
cross-rank control synchronization; they do not establish an LL speedup.


## Native validation (two SM120 GPUs)

On 2026-09-10, all four driver phases passed on
`6afcf2b64e580420df72553107cf2a0ea9c97259`. Hardware: two RTX PRO 6000
Blackwell Server Edition GPUs, NODE topology, bidirectional P2P read/write and
Torch peer access. Ubuntu 22.04.5; driver 595.71.05 (advertised CUDA compatibility
13.2), Toolkit 13.0.88, official Torch 2.11.0+cu130, NCCL runtime 2.30.7,
nccl4py 0.4.1 and nccl-extensions 0.1.0. No Torch source build was used.

| Gate | Observed result |
| --- | --- |
| Single SM120 | 24 Triton/shared-MLP tests passed; E32/C128/H2048/I1408, 1000 replays |
| Compute-only time | Eager 0.268 ms; replay 0.196 ms, including count copies; peak allocation 370.5 MiB |
| Native pair | All seven routing/bucket cases, recovery, two layers and two generations passed; 1000 changing replays per generation |
| Empty ranks | Each rank checked eight snapshots containing remote expert work while locally idle |
| Arithmetic | Independent device oracle passed; CPU composed diagnostic max absolute error 0.0 on both ranks for this fixture |
| Resource lifetime | Per rank: one eager group plus two Graph generations, one persistent handle per generation; all explicitly closed |
| Host admission | Mean 0.283 / 0.279 ms on ranks 0 / 1, including peer wait |
| Host forward submission | Mean 0.541 / 0.502 ms on ranks 0 / 1, including peer wait |
| Eager serving | Eight requests passed; 76 actual `decode_none` passes |
| Graph serving | Eight requests passed; 72 actual `decode_cuda_graph` passes |

Serving used the pinned model and exact recipe above. All requests returned
positive completion lengths and finite logprobs. The median request wall times
were 1.581 s eager and 0.316 s Graph. These are eight-request smoke measurements,
with mixed serial/concurrent requests and cache effects, not a controlled
throughput or speedup benchmark. Raw completions include byte-level marker text;
seven of eight eager/Graph completions matched, and one concurrent completion
differed. Readable generation quality and deterministic equivalence were not
acceptance criteria and are not certified by this smoke.

A separate Nsight Systems 2025.3.1 trace on `bc95943f16a5f4aa6528ec8781f1348d499819dc`
ran the pair harness with 50 replays per generation. It recorded 224 Graph
launches, 16 instantiations/destructions, 1040 native dispatch kernels, 1040
native combine kernels and 1248 Triton `fused_moe_kernel` invocations across
both ranks. Each of those three kernel families had 896 invocations identified
as CUDA Graph nodes. All captured kernels ran on one stream per GPU. The trace
includes warmup, capture, oracle gaps and peer waits; it is execution evidence,
not isolated LL latency. It predates the engine cuMem and serving-port fixes;
the native compute/Graph implementation is identical to the final tested SHA.

The server run exposed and fixed four integration/fixture issues: standalone
shared-MLP construction incorrectly queried uninitialized global TP state;
the synthetic eager IDLE router launched a zero-grid TopK mask kernel; engine
startup disabled cuMem needed by the NCCL Device API; and the serving driver
mistook TCP TIME_WAIT for a live listener. Regression tests cover each issue.
Native group creation fails with cuMem=0 and succeeds with cuMem=1; engine
startup now ensures cuMem is enabled for NCCL EP. The complete local suite at
the tested SHA passed 135 tests on SM89.
