# Proposal: a prepared multimodal input boundary

Status: common structural contract and native little-endian tensor decoding are
implemented in `src/mm/prepared.rs`. Model-semantic import, processor admission,
`api.v1` wire ingress, and serving support for any family are **not**
implemented. The raw `MmProcessor::process` path is unchanged. This document is
not a claim that prepared serving is available.

## Motivation and existing work

An external preprocessor may already have produced model inputs such as pixel
values and grid tensors. A sidecar using native `api.v1` should be able to pass
those inputs to SGLang without decoding the original media again or maintaining
a second implementation of each model's token layout and position rules.

This is an integration gap, not a claim that SGLang has no prepared-input path:

- Python has `processor_output` and precomputed-input handling.
  [#44159](https://github.com/sgl-project/sglang/issues/44159) tracks per-item
  inputs with explicit offsets and hashes;
  [#44160](https://github.com/sgl-project/sglang/pull/44160) preserves these on
  precomputed embeddings. Pre-encoder pixels and post-encoder embeddings must
  remain distinguishable.
- SMG's separate `sglang.grpc.scheduler` protocol already has `TensorData`
  (bytes, shape, dtype), `pixel_values`, `model_specific_tensors`, and
  placeholder ranges. Its servicer constructs `MultimodalProcessorOutput`.
  See the [protocol](https://github.com/smg-project/smg/blob/main/crates/grpc_client/proto/sglang_scheduler.proto)
  and [servicer](https://github.com/smg-project/smg/blob/main/grpc_servicer/smg_grpc_servicer/sglang/servicer.py).
  These are prior art to reuse or align with, not native `api.v1` support.
- The legacy native `sglang.runtime.v1` tokenized `GenerateRequest` carries
  string media inputs, without a typed prepared-tensor envelope. It is a
  different protocol from SMG's scheduler API.
- [#28669](https://github.com/sgl-project/sglang/issues/28669),
  [#40920](https://github.com/sgl-project/sglang/issues/40920), and
  [#48232](https://github.com/sgl-project/sglang/issues/48232) cover standalone
  preprocessing/rendering. Their metadata-only media paths are related, but
  do not themselves define importing externally produced pixel tensors into
  native `api.v1`.
- [#48705](https://github.com/sgl-project/sglang/issues/48705) tracks Rust/Python
  multimodal parity; [#38075](https://github.com/sgl-project/sglang/issues/38075)
  tracks the `api.v1` control plane. This proposal should fit those efforts.

[#50701](https://github.com/sgl-project/sglang/pull/50701) prototypes native
`api.v1` prepared input. This draft separates the processor boundary discussion
from that transport change. Neither PR should create another registry of
`<model>_tensor_adapter` implementations.

**Dependency note.** [#48705](https://github.com/sgl-project/sglang/issues/48705)
and [#44159](https://github.com/sgl-project/sglang/issues/44159) are related
trackers for parity and per-item prepared semantics; they are not merge
prerequisites for this standalone generic structural foundation. The existing
native multimodal processor path already came from merged
[#46291](https://github.com/sgl-project/sglang/pull/46291). Wire integration
remains with [#50701](https://github.com/sgl-project/sglang/pull/50701).

## What is implemented here

`mm::prepared` defines a transport-independent borrowed view
(`PreparedInput` / `PreparedItem` / `PreparedTensor` / `TokenSpan`), caller
budgets (`PreparedLimits`), structural validation
(`PreparedInput::check` → opaque `CheckedPreparedInput`), and per-item native
decoding (`CheckedPreparedInput::decode_tensors` → `NamedTensors`).

The checked view borrows immutable byte slices; that Rust borrow is not a
cross-process shared-memory access or lifetime guarantee. Scalar shapes (`[]`)
have one element; shapes with a zero dimension have none. Only `Float32` and
`Int64` decode today, matching the dependency's native `TensorData` variants.
Spans are half-open `[start, end)` on the supplied token IDs; conversion to the
existing inclusive scheduler offsets remains part of semantic integration.

`check` guarantees only structural properties: profile/fingerprint equality,
nonempty nonnegative `input_ids` within budget, item/tensor/span counts, unique
nonempty tensor names within an item, shape rank and exact little-endian
shape/dtype byte length, total byte budget, and non-overlapping span geometry
with first-span item order. Expected profile/fingerprint values must come from
trusted service/processor configuration, not from the request; string equality
does not prove preprocessing provenance. Passing `check` does **not** grant
semantic safety, processor admission, or serving support. Float finiteness, required tensor
names, geometry-to-token alignment, layout, positions, hashes, and scheduler
items remain outside this module.

`dynamo-multimodal` 0.2.3 has no safe prepared semantic import hook. This crate
does not invent one, does not treat layout/positions as tensor validators, and
does not enable prepared serving for any family.

## Current processor boundary

`MmProcessor::process` currently accepts encoded images and delegates image
processing, token layout, and positions to the existing `MmFamilyProcessor`.
Its result is `MmOutput`: expanded token IDs, features and auxiliary tensors,
item hashes, inclusive item offsets, and positions. The native server already
has a handoff for this result. That raw path is unchanged by the structural
module above.

The dependency's family interface exposes processing from decoded media,
layout, and positions. Those methods are not an import validator: layout and
position calculation trust the supplied item geometry and feature-token count.
The current Rust registry is also narrower than Python's model coverage.
Adding a common entry point cannot automatically make every model supported.

## Preferred boundary

Add one prepared-input entry on the existing processor abstraction once a real
semantic import hook exists. Keep model semantics with the existing processor
implementation, and have both raw and prepared paths converge on the normal
validated output and server handoff. The following responsibilities are
proposed; only the public input/transport structural row is partially realized
by `mm::prepared` today:

| Boundary | Responsibility |
| --- | --- |
| External preprocessor | Produce named tensors and media metadata using the served model's processor configuration; identify the processing stage and configuration. |
| Public input/transport | Check tensor encoding, dtype, rank, shape/byte consistency, size limits, duplicates, and request ownership; resolve bytes independently of model semantics. (`mm::prepared` covers the structural subset of this row.) |
| Existing processor | Check required tensors and their semantic relationships, supported processing stage, item geometry, and token alignment; compute or validate layout and positions. |
| Existing engine handoff | Consume the validated result using the normal scheduler path and resource ownership rules. |

The initial implementation should choose one token-input convention explicitly
(unexpanded placeholders or already-expanded token IDs). If both are later
supported, they must be distinct modes; the receiver must never infer the mode
and accidentally expand twice. Caller-provided offsets, positions, or hashes
must not bypass alignment checks or the engine's cache identity rules.

Unsupported processors should reject prepared input and omit the corresponding
capability. The raw-media path keeps its current behavior. Reuse existing
family code when import validation is required; do not add a parallel family
factory in the server or in a new adapter package.

Named tensors provide transport extensibility, not permission to pass arbitrary
model kwargs. The active processor defines accepted names, dtypes, shapes, and
relationships. The scheduler handoff must be able to represent them before a
model advertises support; a single feature tensor with integer-only auxiliary
metadata is not a universal model contract.

## Alternative: export a complete processor result

A public, validated projection of a processor's complete result could preserve
existing layout and positions and reduce receiver work. SMG's preprocessed path
and Python's `processor_output` are important references for this option.

Exporting private scheduler objects directly would couple the producer to
internal fields and versions. A public projection would still need compatibility
checks, semantic validation, and explicit cache identity and ownership rules.
The preference for a prepared-input entry is to retain those responsibilities
with the existing processor, rather than making each caller reproduce them.
If maintainers prefer an existing validated output-import path, we should reuse
it rather than implement both approaches.

## Scope and decisions to confirm

1. Can an existing Python/SMG import boundary be shared with the native Rust
   path, or should import validation extend the existing `dynamo-multimodal`
   processor interface? `sglang-processor` currently wraps that dependency;
   this proposal does not assume the required hook already exists there.
2. Which parts of the SMG tensor and placeholder contract should `api.v1`
   reuse? How should processing stage and processor/config compatibility be
   represented without exposing private model implementation details?
3. Should the first supported input be pre-encoder pixel tensors, and should
   token IDs be unexpanded or already expanded? Keep embedding input explicit
   rather than treating the two stages as interchangeable.
4. Which existing processor owners should review validation and capability
   rules? Additional model coverage belongs with those processors, not in this
   transport-focused change.

This work does not introduce another renderer service, implement shared
memory, or change cancellation. Inline bytes and a future shared-memory
reference can resolve to the same validated tensor input, but shared memory
needs its own access, lifetime, consumption, cancellation, and restart contract.

## Validation plan for implementation

Structural checks and native decoding are covered by
`tests/prepared_inputs.rs` (synthetic fixtures only). That suite does not claim
model-semantic parity or serving readiness.

When semantic import exists, compare raw and prepared paths at the processor
boundary: tensor bytes, auxiliary tensors, item order, expanded IDs, offsets,
positions, and cache identity. Include multiple items with unequal geometry and
reject malformed shapes, incompatible configurations, unsupported stages, and
misaligned spans. Round-trip the result through the existing scheduler handoff.

Then run model output checks on GPU using pinned model and processor versions.
A Qwen2.5-VL image test is an initial coverage target, not evidence of support
for all vision models or modalities. No runtime or model tests are claimed by
the structural foundation alone.
