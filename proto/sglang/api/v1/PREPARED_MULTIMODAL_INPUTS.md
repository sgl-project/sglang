# External prepared multimodal inputs

**Status:** feature draft. The inline contract, native Rust ingress and the first
`qwen_vl_image_v1` profile are implemented. Tests are included but have not been
run for this revision. Real-model validation is pending; Qwen-VL is the first
validation target. This is not a claim of Step5, MiMo, video or audio support.

## Motivation and design boundary

An external frontend may already have run a community model processor and have
expanded token IDs, pixel tensors and model metadata. Sending the source media
again through `api.v1.Generate` repeats that work. This ingress lets the native
Rust server consume prepared **CPU encoder inputs**, while keeping vision
encoding and generation in SGLang.

Requirements are informed by an internal Step5 sidecar integration and experience
with mixed-media preprocessing. The public schema is independent of that internal
protocol: the caller translates its deployment-specific representation. No
private handles, storage service or internal serializer is required by SGLang.

There are three separate contracts:

| Layer | Responsibility | This revision |
| --- | --- | --- |
| Public input envelope | Named typed tensors, item-local metadata and expanded-token spans | Implemented with inline bytes |
| Model processor profile | Names, dtypes, geometry, positional semantics and compatibility | Qwen-VL image profile implemented |
| Payload transport and ownership | Inline ownership now; shared-memory references and consumption/retirement later | Uses existing owned buffers; no new lease protocol |

Tensor storage can be common across models; tensor meaning cannot. A crop-based image
processor may need several named tensors and crop metadata. An image/video/
audio processor may need multiple token spans and per-item temporal metadata.
Neither is equivalent to Qwen's `image_grid_thw`. The envelope supports named
payloads and multiple spans; **each additional profile still needs its own
validator, native item conversion and positional handling**. Arbitrary metadata is
never blindly forwarded into model kwargs.

## Request contract

`GenerateRequest.prepared_multimodal` contains:

- `profile`: a versioned public processor contract.
- `processor_fingerprint`: the selected server capability's resolved-config
  fingerprint. It is required, so a stale or different processor is refused.
- `items`: ordered `PreparedMediaItem` objects. Each has a modality, named
  `InlineTensor` values, named integer-vector metadata, and half-open token spans
  `[start, end)` in the request's **already-expanded** `input_ids`.

An inline tensor carries `dtype`, `shape` and dense row-major data. The first
profile accepts little-endian `float32`; other dtypes require explicit support by
another profile. gRPC carries binary protobuf `bytes`; the shared JSON codec uses
canonical padded base64. Unknown keys inside the prepared envelope are errors.

The native Rust HTTP `/generate` and gRPC `Generate` use the same normalization.
The Python HTTP adapter explicitly refuses non-null `prepared_multimodal` rather
than silently discarding it. This is not a new OpenAI content-part format.

Only one non-batch input-ID prompt and `n=1` are supported in this slice. Text,
raw media, caller `mm_hashes` and disaggregation fields cannot be mixed with
prepared inputs. Text-only models, P/D serving and incompatible processors
reject prepared requests before scheduler submission. Existing raw-media paths
remain available.

## Capability and processor agreement

Read `GetServerInfo.prepared_input_capabilities` first. An empty or absent list
means unsupported: old protobuf servers can ignore new fields, so clients must
never infer support from successful request parsing. Native HTTP exposes the same
capability in server info. Only an installed compatible processor in unified
multimodal serving advertises a profile.

The first capability includes the image token ID, patch/merge/temporal-patch sizes,
a 64 MiB whole-request inline tensor budget and `processor_fingerprint`. Transport
body/message limits can impose a smaller bound; advertising a tensor budget does
not raise those limits. Use the served model's exact processor configuration,
including normalization and resampling, when preparing input.

For `qwen_vl_image_v1`, the fingerprint is lowercase hexadecimal SHA-256 over:

1. ASCII `sglang-prepared-qwen-vl-image-v1` followed by one zero byte.
2. `image_token_id` as signed 32-bit little-endian.
3. `patch_size`, `merge_size`, `temporal_patch_size`, `min_pixels`, `max_pixels`,
   each as unsigned 64-bit little-endian, in that order.
4. The three `image_mean`, then three `image_std` values as IEEE float32
   little-endian bytes.
5. ASCII `aten_u8` or `pil`, selecting the resolved native resampler.

It is stable across boots for the same resolved spec; `instance_id` remains the
separate runtime boot identity. A matching fingerprint binds negotiation to the
served processor. It cannot prove that a caller applied preprocessing correctly,
and copying it into a request is not an accuracy check.

## First profile: qwen_vl_image_v1

Each item must have:

- `modality = "image"`.
- Exactly one tensor named `pixel_values`, with shape
  `[grid_h * grid_w, 3 * temporal_patch_size * patch_size ** 2]` and finite float32
  values in the served processor's flattened patch order.
- Exactly one integer metadata entry named `image_grid_thw`, equal to
  `[1, grid_h, grid_w]`, with positive spatial dimensions divisible by merge size.
- Exactly one token span of length
  `(grid_h / merge_size) * (grid_w / merge_size)`, containing only the served
  processor's image-pad token. Every image-pad token in the prompt must belong to
  one declared item. Items and spans must be ordered, disjoint and in bounds.

Extra names, duplicate names, unsupported modalities, invalid dimensions, byte
lengths and overflowing arithmetic are rejected. Tensor storage and spans are
checked before native tensor construction. Nonfinite pixel values are refused
while decoding. Oversized prepared prompts are rejected even under
`allow_auto_truncate`: truncation would invalidate spans and positional metadata.

The ingress skips fetch, decode, resize, patch extraction and placeholder
expansion. It reuses `dynamo-multimodal`'s native Qwen image M-RoPE routine and the
existing `MmEncodedEntry`/feature-buffer path. The normal scheduler then performs
vision encoding and generation. No unchecked positional override is accepted.

Full cache identity comes from the existing typed-content identity over the
actual feature and item metadata; no caller identity can substitute for it. The
legacy routing hash is derived from that identity. Existing cache namespaces and
request cancellation remain in use.

## Accuracy lessons and model expansion

Mixed-media integration experience showed how flattening item-scoped processor
options into one request-wide map can apply a video's pixel limit to an image.
That can alter the actual image grid and prompt length, not just usage reporting.
Accordingly:

- Keep tensors and metadata attached to their media item; preserve item and span
  ordering. Do not broadcast one item's options to other media or modalities.
- Prepared inputs are final processor results. The engine must not run another
  resize or placeholder-expansion pass based on global processor kwargs.
- Bind to the resolved processor spec and validate tensor/grid/span consistency.
  The frontend remains responsible for the policy that produced those tensors.
- Compare native input tensors, positions and per-modality token accounting when
  validating a profile. Similar generated text or a successful RPC is insufficient.

For another model, define its profile's allowed names and dtypes, item-to-token
alignment, required masks/crop/temporal metadata and positional rules. Implement
its conversion using the existing native processor/item abstractions, advertise
only the supported configuration, and add raw-versus-prepared parity coverage.
Do not label Step5 or MiMo supported merely because the envelope can represent
some of their inputs. The current built-in scheduler handoff carries one feature
tensor plus integer metadata per item; a profile with extra feature tensors or
masks also needs an explicit native handoff extension, not just a new profile
name. Video/audio, post-encoder embeddings and EPD/P/D handoff remain outside the
implemented profile.

## Ownership and future large-payload transport

Inline request bytes are owned by the engine. Worker conversion produces the
same owned native feature buffers as raw media, and existing buffer lifetime and
cleanup paths apply. This revision adds no borrowed external memory, release ACK,
`cancel completed` guarantee or cancellation-retirement receipt.

The next transport layer can carry shared-memory references instead of inline
bytes while preserving the input/profile contract. It needs explicit validation
of accessible regions, bounds, immutable contents, consumption and release on
failure/cancel. RPC acceptance, an Abort ACK and stream closure must not be
misrepresented as proof that all consumers released external storage. Coordinate
that work with cancellation completion semantics; no private object store is
required by this ingress. No zero-copy or performance claim is made here.

## Community work to reuse

- [#44159](https://github.com/sgl-project/sglang/issues/44159) and merged
  [#44160](https://github.com/sgl-project/sglang/pull/44160): per-item precomputed
  semantics, hashes and offsets.
- [#44171](https://github.com/sgl-project/sglang/issues/44171): media identity/cache
  events; this ingress reuses the existing identity path.
- [#48705](https://github.com/sgl-project/sglang/issues/48705) and merged
  [#46291](https://github.com/sgl-project/sglang/pull/46291): native multimodal
  parity and the image processor used by this implementation.
- [#38471](https://github.com/sgl-project/sglang/pull/38471): pre-sampled video,
  explicitly excluding gRPC video transport.
- SMG's [prepared tensor protocol](https://github.com/smg-project/smg/blob/main/crates/grpc_client/proto/sglang_scheduler.proto):
  transport prior art; a different service contract.

## Pending validation

Rust/Python JSON/wire, normalization, profile rejection, positional layout and
content-identity tests are included **but not executed in this revision**.
Generated Rust/Python bindings have been regenerated from the canonical schema;
`sglang-api-types` moves from 0.2.0 to 0.3.0 for the additive schema change.

The first real-model target is Qwen-VL: compare raw and externally prepared
requests from the same pinned community processor, with single/multiple images,
unequal image sizes, text controls, malformed inputs and cancellation. Check
input IDs, pixel values, grids, M-RoPE, generated token IDs and usage. Separately
exercise supported TP feature transport. Step5/MiMo profiles, mixed modalities,
shared-memory transport and performance require their own follow-ups and evidence.
