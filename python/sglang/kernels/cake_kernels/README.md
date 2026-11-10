# Cake kernels

This directory owns thin adapters to Cake-generated implementations distributed
by FlashInfer. Generated CUDA sources and JIT caches remain owned by FlashInfer.
As with `kda_kernels/`, runtime consumers import the operator facade under
`sglang.kernels.ops`; importing the namespace must not initialize a backend.

Backend selection must preserve the operator's dtype, layout, state, mutation,
and numerical contract. In particular, Cake top-k-then-top-p sampling is not a
drop-in replacement for joint top-k/top-p filtering. Prepared plans must remain
valid when input contents and buffers change across calls and CUDA Graph replay.

The sampler's softmax fast path accepts FP32 contiguous logits on SM100 and
SM103, small batches and large vocabularies. The sampler retains its existing
filtering and random-number generation, and deterministic inference retains
torch.softmax.
Other operators require their own numerical and real-model performance evidence
before being added here.

The fast path also requires FlashInfer's `cake_blackwell_softmax` module. Older
FlashInfer installations retain Torch softmax; this adapter does not change
SGLang's dependency pins.

The MLA value gate (`attention.mla_value_gate`) is the Cake form of Kimi K3's
`MlaValue`: the W_UV projection of the BF16 attention latent, rounded to BF16 and
multiplied by the FP32 sigmoid of the gate, with every row whose request length is
zero or negative written as zero. Measured on B200 and GB300, its output is
bit-exact to MlaValue.
`prepare_cake_mla_value_gate` binds W_UV outside CUDA Graph capture and returns a
runner whose launches can be captured and replayed with new input contents; a
replaced W_UV storage needs a new runner. Kimi K3 selects it per prepared bucket
with the `k3_flashinfer_mla_value` slot and keeps MlaValue otherwise. It needs
FlashInfer's `cake_mla_value_gate` module, which SGLang's pinned FlashInfer does not
ship yet.

The upstream primitive is merged in FlashInfer
[#8867](https://github.com/flashinfer-ai/flashinfer/pull/8867); the request-length
L1-load update is merged in
[#8904](https://github.com/flashinfer-ai/flashinfer/pull/8904)
(`ba14d1e28bfa89473f3ed25ce450b0e9f7a74091`). Those PRs contain component
validation, not full-model qualification of this adapter. Record the exact
FlashInfer revision and matching package versions when qualifying the slot;
SGLang's pinned `flashinfer_python==0.7.2` retains `MlaValue`.

The official `flashinfer_python-0.7.3-py3-none-any.whl` was also inspected on
November 10, 2026. Its ZIP directory contains neither
`flashinfer/cake_mla_value_gate.py` nor the corresponding experimental package.
Matching Python/cubin/JIT-cache version numbers alone therefore cannot enable
this operator on that release. Selection continues to check for the actual
module. The source/AOT checks below remain distinct from installed-wheel
qualification.

Only an explicit capability refusal (missing module, unsupported architecture or
unavailable program) disables the adapter for the device. Other prepare errors,
including OOM, capture refusal and unclassified JIT/load errors, keep `MlaValue`
for that preparation and allow later buckets to try. Preparation stays outside
graph capture; launch errors propagate rather than switching kernels in a graph.
The slot replaces `MlaValue` in DCP1 and native-MLA paths; fused DCP8 keeps
`DcpValue`. Measure the slot separately with the same profile and workload so its
effect is not mixed into the native-MLA versus fused-attention comparison.

FlashInfer [#8917](https://github.com/flashinfer-ai/flashinfer/pull/8917)
(`f454a17f3f2bae878caf2d49637b1d2b6f407fcd`) adds automatic 8-row / 16-row
program selection during preparation. Keep that selection in FlashInfer. Its
reported PDL-enabled warm-L2 standalone timings regress for some B >= 25 rows,
while the same rows improve behind a PDL predecessor. Qualify the actual
predecessor and complete call before drawing a serving-performance conclusion.

The author's [subsequent assessment](https://github.com/flashinfer-ai/flashinfer/issues/8795#issuecomment-6681591837)
keeps the two shipped programs: gains from other variants do not justify another
program for the default dependent-PDL consumer. Revisit that decision when the
consumer's launch mode or shapes change, rather than adding another kernel here.

SM103 checks with the complete FI `f454a17f` source package used normal FI and
SGLang package initialization, the real slot/environment and operator registry,
and FI's supported AOT loader. A native load failure in a private test-library
copy reproduced the old device-wide refusal after the library was restored;
the corrected adapter recovered and passed B25/q1 and B64/q2 eager and
changed-input graph checks. Layout refusals were already bucket-local in the
original adapter and are not evidence of that regression.

With CuTeDSL 4.8.0, the unchanged `MlaValue` and this FI preparation path also
produced identical output bits at B1/q1, B25/q1 and B64/q2, in eager execution and
changed-input graph replay. The CuTe launch uses the current runtime stream,
including during capture. These are source-package/operator checks; installed
wheel-trio compatibility, the complete K3 executor, TP8/DCP/HiCache, task quality
and matched serving performance still require separate qualification.

### Native K3 consumer segment

The [manual consumer driver](../../../../test/manual/kernels/bench_kimi_k3_native_value_gate.py)
calls the production native value-projection method, K3's gate-stream helpers and
the native BF16 gate kernel, alongside FI's public preparation/launch API. It uses
synthetic TP8-rank-shaped tensors with 12 local heads, a 512-wide latent, 128 value
columns per head and a 7168-wide gate input. It covers changed-input graphs,
invalid rows with NaN gate inputs, a double-gating negative control and new
preparation after replacing weight storage.

On SM103 with Torch 2.14.1+cu130 and CuTe DSL 4.8.0, the complete consumer segment
produced these warm medians in microseconds:

| Rows | Execution | Native BF16 | Native projection + FP32 diagnostic | FI FP32 |
| --- | --- | ---: | ---: | ---: |
| 8 | Python-dispatched eager | 58.998 | 82.510 | 74.261 |
| 8 | Captured graph replay | 12.418 | 21.196 | 12.371 |
| 32 | Python-dispatched eager | 59.543 | 84.516 | 74.071 |
| 32 | Captured graph replay | 16.534 | 24.706 | 18.485 |

Timing inputs are identical, finite and all valid. Each point has five warmups
and three samples of 30 segments on a shared device. The graph uses the native
capture-only gate side stream. The FP32 diagnostic isolates the precision policy
and uses separate Torch elementwise operations. It is a diagnostic reference.
[Measurements and samples](../../../../docs/images/kimi-k3-specialized-decode/native-consumer-measurements.json)
record the complete scope and the earlier resource-guard failure separately.

The FI output's maximum absolute difference from native BF16 was 0.00048828125.
Against the FP32 diagnostic it was at most 0.0000009536743 at B8 and 0.0001220703125
at B32. FI explicitly refuses `sigmoid_precision="bf16"`. The tested native/FI
bridge uses normal source-package imports and JIT compilation of FI-shipped
Cake-generated CUDA through FI's public loader. Its source package reports
`0.0.0+unknown` at the archived `f454a17f` revision. This is source-build evidence.

These points leave the native BF16 serving path in place. A native FI selection
still needs an agreed precision policy, an exactly-once handoff past the existing
gate wrapper, and live valid-row metadata for graph padding. Native graph padding
can have nonzero sequence lengths, and its valid-token scalar is conditional on
engine configuration. The prepared executor's FI adapter remains separately
scoped to its existing `MlaValue` sites.

With a FI build exposing this experimental API and one visible SM103 GPU, run
the driver from the repository root:

```bash
python test/manual/kernels/bench_kimi_k3_native_value_gate.py \
  --output /tmp/k3-native-value-gate.json
```

Use `--flashinfer-revision` to record the source build's commit.
