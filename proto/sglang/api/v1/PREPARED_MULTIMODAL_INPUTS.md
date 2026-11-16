# External prepared multimodal inputs (draft)

**Status:** design seed for discussion. No request fields, generated bindings or
runtime handlers are added by this draft. Existing `api.v1` behavior is unchanged.

## Problem

An out-of-process frontend can already run a model's community image processor
and produce token IDs, pixel tensors and positional metadata. Native
`api.v1.Generate` currently accepts media sources through `MediaInput` and
`MediaRef`, but has no input form for that prepared result. A caller must send
source media and repeat preprocessing inside the engine, or depend on a private
bridge to the scheduler.

The proposed interface is a public SGLang contract for any external frontend.
Deployment-specific protocols and object-store handles are translated by the
caller; they do not become engine dependencies. Tensor representation can be
common, while tensor meaning and required positional metadata remain specific to
the selected model processor.

## First implementation slice

Start with one request's image processor output for one supported Qwen-VL family
in unified serving. The exact processor profile and wire representation need
maintainer agreement before implementation.

- Require already-expanded `input_ids` and an explicit prepared-input mode.
  Reject text input, raw/prepared mixtures and unsupported processor profiles
  before scheduling.
- Carry ordered media items with explicit spans and typed tensors. For the first
  profile these include pixel values and the image grid needed by native
  positional handling. Define one span convention and translate it once to the
  native representation, preserving item boundaries.
- Give each tensor an explicit dtype, shape, layout and bounded inline bytes.
  Validate dimension/byte arithmetic, supported dtypes, item spans and the
  profile's required tensors before tensor materialization or scheduler admission.
- Preserve the processor's token/media alignment. Skip media fetch, decode,
  resize, patch extraction and placeholder expansion; keep the vision encoder
  and generation inside SGLang. Reuse validated native positional handling
  rather than accepting unchecked positional overrides.
- Reuse existing item identity and cache-namespace semantics. Agree on the trust
  boundary for caller-provided identities; do not introduce a second cache-key
  format or infer media identity from shape and placeholder tokens.
- Declare the supported prepared-input profile explicitly. Clients must confirm
  support before submitting it: an older protobuf server can ignore unknown
  fields, so field presence alone cannot establish compatibility.

This slice is CPU-processed encoder input, not post-encoder embeddings. It does
not claim support for every model that accepts a similarly shaped tensor.

## Ownership and compatibility

Keep the normal request admission, generation and cancellation paths. Inline
input storage must remain alive until all native consumers finish using it,
including failure and cancellation paths; RPC acceptance is not consumption.

The draft does not choose protobuf tags or change the shared JSON contract.
Before adding active fields, agree on how the HTTP/Python adapters handle the
new form, including explicit refusal where unsupported, and follow the normal
schema generation and `sglang-api-types` versioning rules.

Shared-memory references, remote transfers and their release protocol are a
separate follow-up after the inline input semantics work. Video/audio,
post-encoder embeddings, EPD/P/D handoff and distributed ownership are also
outside the first slice. This proposal does not extend cancellation retirement
receipts or make a zero-copy or performance claim.

## Existing work to reuse

- [#44159](https://github.com/sgl-project/sglang/issues/44159) defines explicit
  per-item precomputed input semantics. Its merged first step,
  [#44160](https://github.com/sgl-project/sglang/pull/44160), preserves caller
  hashes and offsets. Reuse those semantics rather than replacing the Python
  precomputed-input work.
- [#44171](https://github.com/sgl-project/sglang/issues/44171) tracks media identity
  through native cache reuse and KV events; cache/event redesign stays there.
- [#48705](https://github.com/sgl-project/sglang/issues/48705) tracks Rust
  multimodal parity. [#46291](https://github.com/sgl-project/sglang/pull/46291)
  supplies native image preprocessing through `sglang-processor`; this proposal
  adds a prepared-input ingress rather than another image processor.
- [#38471](https://github.com/sgl-project/sglang/pull/38471) handles pre-sampled
  video and explicitly excludes gRPC video transport. That is adjacent work,
  not an implementation of the interface proposed here.
- SMG's [`MultimodalInputs`](https://github.com/smg-project/smg/blob/main/crates/grpc_client/proto/sglang_scheduler.proto)
  is transport prior art for typed pixel tensors, model-specific tensors and
  placeholders. It is a different service contract, not current `api.v1` support.

## Implementation and validation plan

1. Agree on the first model profile, span/identity semantics, capability discovery
   and shared HTTP/gRPC representation with the API and multimodal owners.
2. Add the typed inline contract and generated bindings, with wire/JSON
   compatibility and malformed-input CPU tests.
3. Connect it to the existing native multimodal item/encoder path. Test that
   prepared input bypasses preprocessing without losing spans or metadata, and
   that rejected and cancelled inputs keep correct ownership.
4. Compare prepared and raw-media requests from the same community processor on
   a real model: positional inputs, output tokens, usage and normal text controls.
   Cover rejection of incompatible profiles and unchanged legacy requests.

No functional, model-accuracy or performance validation has been run for this
design-only draft. Shared-memory optimization should be measured separately
after the input contract is implemented.
