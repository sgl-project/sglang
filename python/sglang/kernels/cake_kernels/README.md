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

Only an explicit capability refusal (missing module, unsupported architecture or
unavailable program) disables the adapter for the device. Other prepare errors,
including OOM, capture refusal and unclassified JIT/load errors, keep `MlaValue`
for that preparation and allow later buckets to try. Preparation stays outside
graph capture; launch errors propagate rather than switching kernels in a graph.
The slot replaces `MlaValue` in DCP1 and native-MLA paths; fused DCP8 keeps
`DcpValue`. Measure the slot separately with the same profile and workload so its
effect is not mixed into the native-MLA versus fused-attention comparison.
