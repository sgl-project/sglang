# Native Rust Multimodal Processing

How the Rust tokenizer-manager path (`SGLANG_RUST_SERVER=1`) serves
vision-language models with image preprocessing done **entirely in Rust** —
no Python `mm_processor` in the request path, and no Python fallback.

Currently supported: the Qwen VL family (`qwen2_vl`, `qwen2_5_vl`, `qwen3_vl`,
`qwen3_vl_moe`, `qwen3_5`, `qwen3_5_moe`), images only.

## Data flow

```
HTTP /generate (axum, rust/sglang-server)
  │  GenerateBody.split() fans image/video/audio_data out per item
  │  (message/request.rs, mirrors Python _normalize_*_data)
  ▼
tm-ingress FSM: Received → … → Normalizing
  │  has_multimodal() → Validated(HasMultimodal) → Encoding
  │  request parks in `pending_mm`; its [text, input_ids, image, video,
  │  audio] msgpack payload goes down a bounded channel
  ▼
MM worker pool (sglang-server/src/mm.rs, N plain threads)
  │  sglang-mm driver::process (inline on the worker thread — the
  │  rlib owns no threads; the pool is the concurrency):
  │    fetch (data:/base64/file/http) → decode → smart_resize →
  │    bicubic resample → normalize (u8→f32 LUT) → patchify →
  │    feature hash; then placeholder expansion + image-only M-RoPE
  │  result buffers parked in the rid-keyed mm sidecar
  ▼
TmEvent::MmEncoded { rid, input_ids }   (final placeholder-expanded ids)
  │  ingress resumes the parked request: Encoding → Queued
  │  (skips the tokenizer pool — the ids are already final)
  ▼
ingress ring → Python scheduler
  │  RustServer.drain: take_mm(rid) pops the sidecar entry and
  │  wraps it zero-copy (numpy owns the Rust buffers, torch.from_numpy
  │  wraps them; hashes are worker-precomputed) into
  │  MultimodalProcessorOutput → obj.mm_inputs
  ▼
scheduler / model forward (mm_inputs identical in shape and content to
the Python mm_processor's output — pinned by the parity test suite)
```

The only Python steps are the launch-time spec build and the drain-time
tensor wrapping; both are off the per-image hot path.

## Failure semantics (no Python fallback)

- **Unsupported model family**: a multimodal model whose
  `NativeMmHost.resolve_native_spec()` resolves to `None` (not in the family
  allowlist, unexpected processor class, unrecognized `--mm-process-config`
  overrides) **fails at launch** with a `RuntimeError`. Serve such models
  without `SGLANG_RUST_SERVER`.
- **Per-request errors** — out-of-scope input (video/audio, precomputed
  features, undecodable/PIL-only images) and hard failures (bad URL,
  oversized fetch, preprocess error) alike — are rejected back to the client
  (400/500) with a message saying why.
- **Late results** for rejected/aborted requests are purged from the sidecar
  by the ingress.

## Code map

| Layer | Location | What lives there |
|---|---|---|
| Pipeline core | `rust/sglang-mm/src/driver.rs` | model-independent driver: fetch → decode → preprocess → hash → expand → M-RoPE |
| | `rust/sglang-mm/src/common/{fetch,token_layout,resize,transforms}.rs` | media fetch; placeholder expansion; PIL-exact Lanczos/Bicubic resize |
| | `rust/sglang-mm/src/pipeline.rs` | `MmFamilyProcessor` trait + the carriers (`Tensor`, `TokenLayout`, ...) |
| | `rust/sglang-mm/src/qwen_vl/mod.rs` | `QwenVlProcessor` (`MmFamilyProcessor` impl) + feature-gated parity bindings |
| | `rust/sglang-mm/src/registry.rs` | `pipeline_from_spec` (family dispatch) |
| Server integration | `rust/sglang-server/src/mm.rs` | worker pool + sidecar; drives the sglang-mm driver with the server tokenizer |
| | `rust/sglang-server/src/message/{request,mm_payload}.rs` | mm fields on the wire body, per-item fan-out, mm payload encoding + typed decode (the wire contract has one owner) |
| | `rust/sglang-server/src/tokenizer_manager/ingress.rs`, `fsm.rs` | `Encoding` stage: park/dispatch/resume/reject |
| | `rust/sglang-server/src/lib.rs` | pyo3 surface: `start_mm_workers(spec_json, workers)`, `take_mm(rid)` |
| Python side | `python/sglang/srt/managers/rust_server.py` | `NativeMmHost` (spec build + launch gate), drain-time zero-copy adapter |

The `sglang-mm` crate builds two ways: the pyo3 extension
`sglang.srt.multimodal._core` (features `python,parallel`, requested by the
wheel build — used by parity tests and Python processors) and a pure-Rust
rlib (default features, i.e. neither — no pyo3, no rayon, no threads) linked
by `sglang-server`.

## The spec JSON

`NativeMmHost` (Python) builds the pipeline spec from the **resolved** HF
processor at launch — never from raw HF config — and is conservative: any
unrecognized knob returns `None` and the launch gate fails. For `qwen_vl`:

```json
{
  "family": "qwen_vl",
  "image_token_id": 151655,
  "patch_size": 14, "merge_size": 2, "temporal_patch_size": 2,
  "min_pixels": 3136, "max_pixels": 12845056,
  "image_mean": [0.481, 0.458, 0.408], "image_std": [0.269, 0.261, 0.276]
}
```

`registry::pipeline_from_spec` deserializes it and dispatches on
`family`. `--mm-process-config {"image": {...}}` overrides are honored for
`min_pixels`/`max_pixels` only.

## Tests

| Layer | Where | Run with |
|---|---|---|
| Pure-Rust unit tests (fetch, tokens, driver, qwen_vl geometry; payload in `sglang-server`) | `#[cfg(test)]` in `rust/sglang-mm/src/**`, `rust/sglang-server/src/message/mm_payload.rs` | `cd rust/sglang-mm && cargo test` (and `--features parallel`; CI: `pr-test-rust-exts.yml`) |
| Server framework tests (mm fan-out, ingress Encoding arms, msgpack shapes) | `#[cfg(test)]` in `rust/sglang-server/src/**` | `cd rust && cargo test -p sglang-server` |
| CPU parity suite vs real HF processors (preprocess, prompt geometry, scheduler-boundary output, drain adapter, error paths) | `test/registered/unit/multimodal/rust/{qwen,shared}/` | `python3 <file>` directly, or CI suite `base-a-test-cpu` (`_core` is built by `pip install -e python`) |
| GPU e2e smoke (live sidecar handoff, multi-image, rejection paths) | `test/registered/vlm/test_rust_native_mm.py` | 1 GPU; CI stage `base-b` |
| GPU quality gate (MMMU ≥ 0.30 through the native path, asserts the "native MM pipeline enabled" launch log) | `test/registered/vlm/test_rust_native_mm_mmmu.py` | 1 GPU; CI stage `base-b` |

## Adding a new model family

1. **Implement `MmFamilyProcessor`** (`rust/sglang-mm/src/pipeline.rs`) in
   `rust/sglang-mm/src/<family>/mod.rs`: `process_item` (decoded media →
   named tensors + geometry), `layout` (prompt geometry as a `TokenLayout`
   value — `Repeat` for placeholder expansion, `Explicit` for structured
   tile/marker schemes), and `positions` if the model has a scheme beyond
   1D RoPE. Parse the family's spec struct from the spec JSON
   (`from_spec_json`). Add `#[cfg(test)]` geometry tests. Reuse
   `common::{resize, transforms, tokens}` where the model matches PIL/HF
   semantics. See the design introduction in `rust/sglang-mm/README.md`.
2. **Register the family**: add a match arm in
   `registry::pipeline_from_spec` (`rust/sglang-mm/src/registry.rs`)
   and `pub mod <family>;` in `rust/sglang-mm/src/lib.rs`. If you add parity
   bindings (recommended), feature-gate them like `qwen_vl::python` and
   register the submodule in the `_core` pymodule.
3. **Extend the Python gate**: in
   `python/sglang/srt/managers/rust_server.py`, teach
   `NativeMmHost.resolve_native_spec()` to recognize the model (processor
   class + `model_type` allowlist) and emit the family's spec from the
   *resolved* processor attributes. Stay conservative: return `None` for
   anything the Rust pipeline does not mirror exactly. Extend
   `NativeMmHost.build_native_mm` if the scheduler needs different
   `model_specific_data`.
4. **Add the parity tests**: copy the qwen pattern under
   `test/registered/unit/multimodal/rust/<family>/` — a `_fixtures.py`
   building the real HF processor, then image-preprocess parity,
   prompt-geometry parity, and scheduler-input parity against
   `NativeMmHost.build_native_mm`. These are the contract that the native
   output is indistinguishable from the Python processor's.
5. **Gate quality**: extend or clone the MMMU e2e gate if the family has a
   suitable checkpoint.

Keep the split in mind: `sglang-mm` owns *what* preprocessing computes
(model semantics, parity-tested), `sglang-server` owns *how* it runs
(threading, channels, sidecar, rejection) — a new family should not need any
`sglang-server` change.
