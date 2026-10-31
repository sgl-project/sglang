---
name: processor-mm-parity
description: Add or verify a model family's multimodal (image) preprocessing in rust/sglang-processor's `mm` component, on top of Dynamo's `dynamo-multimodal`, so it matches SGLang's Python path bit for bit at every stage and end to end. Use when adding a multimodal family to sglang-processor or the Rust server, bumping `dynamo-multimodal`, regenerating `mm_parity` fixtures, or debugging a Rust-vs-Python multimodal mismatch.
---

# Processor multimodal parity

## The principle

**Match Python bit for bit, at every stage and end to end.** For the same
request, the Rust path must produce exactly what SGLang's Python path produces:
- the same resolved processor parameters;
- the same decoded pixels;
- the same feature bytes;
- the same expanded ids and offsets;
- the same positions;
- the same model kwargs;
- the same scheduler inputs.

The whole served request must also come out the same. It follows that:
- **No tolerances.** No `allclose`, no "within 1 ULP", no "close enough on
  MMMU". Compare bytes and hashes of bytes.
- **Python is the reference.** Dynamo is the implementation. SGLang code exists
  only where the two differ, and each such piece names the Python source it
  mirrors.
- **Never patch a fixture to match Rust.** Every mismatch gets one of two fixes:
  - fix Dynamo upstream (`ai-dynamo/frontend-crates`, `multimodal/`), or fix
    SGLang's side (the spec gate in `rust_server/multimodal.py`, or
    `src/mm/mod.rs`) when the rule is SGLang's;
  - or record it as a `known_gap`, with the reason and the upstream fix. The
    test keeps reporting it and fails once it matches.
- **Stage parity is necessary, not sufficient.** Each stage is checked on its
  own, so a mismatch is located immediately. The end-to-end check then proves
  the stages compose.

The one sanctioned difference is the content **hash**. Rust hashes the encoded
bytes with blake3; Python hashes the feature with sha256. It is cache identity
only and never reaches the model, so outputs cannot depend on it. Anything else
that cannot match is a known gap, not an exception.

## The reference: Python's CPU path

Python itself takes different paths on different hosts. The reference is the
**CPU** path, which every fixture records:
- PIL decode (`utils/common.py::_load_image`), then `convert("RGB")` in
  `base_processor._load_single_item`. There is no EXIF transpose unless the
  processor sets `smart_rgb_conversion`.
- The HF image processor's CPU kernels: torchvision on a uint8 tensor (the
  default backend), or PIL with `--image-processor-backend pil`.

On a CUDA host, Python's default deviates from that reference in two ways:
- It decodes JPEG with nvJPEG (`gpu_image_decode`, on by default in
  `base_processor`).
- It runs the torchvision image processor on `cuda:<base_gpu_id>`
  (`_fast_image_processor_device`).

Both are Python disagreeing with itself. Measure them and report them; never
copy them into Rust.

The PIL backend's bicubic output changes across Pillow releases, and SGLang
does not pin Pillow. Every fixture records the Pillow version it was generated
under. Regenerate under that version, or regenerate every fixture together
under a new one.

## Stages

| # | Stage | Python source | Rust | Checked by |
|---|---|---|---|---|
| 1 | Spec: processor params from the loaded HF processor plus server args | `get_processor` (`utils/hf_transformers/processor.py`), `--image-processor-backend`, `--mm-process-config` | resolved in Python by `rust_server/multimodal.py::RustMmProcessor.resolve_spec`, handed over as `MmSpec`; Dynamo `registry::build_processor` | `mm_parity` cases record the resolved spec, per backend and per override |
| 2 | Fetch: source to bytes | `get_image_bytes` | host (`sglang-server/src/multi_modality/`) | host unit tests: data URL, bare base64, `file://`, http, limits |
| 3 | Decode: bytes to RGB | `_load_image`, `_load_single_item` | Dynamo `image::decode::decode_rgb` | the corpus (below); a reject must reject |
| 4 | Features: resize, normalize, patchify | the HF image processor | Dynamo `process_item` | `feature_sha256` and `feature_shapes` per item |
| 5 | Expansion: placeholders to feature tokens | `base_processor.build_input_ids`, `get_mm_items_offset` | Dynamo `layout` and `apply_layout` | `input_ids_sha256`, `num_input_ids`, `offsets` |
| 6 | Positions | `MRotaryEmbedding.get_rope_index` or the family's shortcut | Dynamo `positions` | `mrope_sha256`, `mrope_delta` |
| 7 | Model kwargs | `model_specific_data` (e.g. `image_grid_thw`) | `ProcessedItem.aux` | `image_grid_thw` |
| 8 | Chat with media: prompt plus ordered media list | `process_content_for_template_format`, `_apply_jinja_template` | `render/` | `tests/parity.rs` media cases ([`processor-model-parity`](../processor-model-parity/SKILL.md)) |
| 9 | Scheduler handoff: `MultimodalDataItem` fields | the processor's `MultimodalProcessorOutput` | server `multi_modality/` and `rust_server/multimodal.py::wrap_encoded` | server and Python unit tests: every field the scheduler reads |
| E2E | The served request | the Python server | the Rust server (`SGLANG_RUST_SERVER=1`) | "End to end": bit-exact inputs and outputs, MMMU and MMMU-Pro, serving benchmarks |

## Where a family's code goes

- **Dynamo**, upstream: the family's `process_item`, `layout`, `positions`,
  its `ProcessorSpec` arm, and its `spec_from_hf_configs` branch. Anything HF
  ships belongs there: class defaults, kernels, layout.
- **`src/mm/models/<family>.rs`**, until Dynamo's family processes pixels: a
  Dynamo `MmFamilyProcessor` built from Dynamo's planners and kernels.
  `models::build` is the only match on the family; delete the file once Dynamo
  ships the same code.
- **`src/mm/mod.rs` never branches on a family.** It runs Dynamo's stages in
  order.
- **Python `rust_server/multimodal.py`**: one `RUST_MM_FAMILIES` entry, which
  checks that the processor class is SGLang's own, and its `resolve`, which
  reads the family's `ProcessorSpec` fields from the loaded HF processor. The
  server takes that spec as JSON.

**DeepSeek-V4.1 is the exception to "Python is the reference".** The Rust
server resizes its images with deepseek-recipe-image (OpenCV, re-encoded as
WebP; `sglang-server` feature `deepseek-image`), which Python does not, so its
pixels differ from Python's. Recipe resizes to V4.1's default geometry only,
so the gate refuses a config that changes it. Token layout and model kwargs
still match; it has no `mm_parity` fixture, and accuracy (MMMU, MMMU-Pro) is
its end-to-end gate.

## Method

1. **Pick the checkpoints** from the family's cookbook page (`docs/cookbook/`).
   Include one per distinct processor config, e.g. a different patch size or
   `size` style. Pin each revision.
2. **Trace Python stage by stage** for the family (the table above). Write down
   every transformation and every rejection; this is the divergence inventory.
3. **Extend the corpus** in `tests/scripts/generate_mm_parity.py` (`CORPUS`)
   until it covers each inventory item, and add cases in `cases()`. Images
   must be small and deterministic.
4. **Generate fixtures** (below) and run `tests/mm_parity.rs`.
5. **Fix each failure** at the stage it belongs to: Dynamo for HF behavior,
   SGLang's gate or `src/mm/mod.rs` for SGLang's. Until an upstream fix ships, add the case
   to `KNOWN_GAPS` with the reason and the upstream PR.
6. **Run end to end** (below): the bit-exact check, MMMU and MMMU-Pro on both
   servers, and the serving benchmarks, before the host serves the family.

## Corpus checklist

Each item below needs at least one case:
- **Sizes:** every resize boundary (below `min_pixels`, above `max_pixels`,
  sides smaller than the resize factor), and the aspect-ratio limit on both
  sides.
- **Modes:** L, LA, P, P with `transparency`, RGBA, and 16-bit.
- **JPEG:** baseline, progressive, 4:4:4 and subsampled, CMYK, and EXIF
  orientation.
- **Other formats:** animated GIF (first frame), WebP lossy and lossless with
  alpha, BMP.
- **Prompts:** multiple images in one prompt.
- **Server args:** each backend, and each `--mm-process-config` key Python
  honors and each it rejects.
- **Rejections:** every input Python rejects is recorded as `error` and must
  be rejected by Rust.

## Generating fixtures

```sh
cd rust/sglang-processor
pip install --target /tmp/pillow pillow==<the fixtures' "pillow">
PYTHONPATH=/tmp/pillow:$REPO/python \
    python tests/scripts/generate_mm_parity.py <hf-model-id> [--revision <sha>]
```

Every fixture covers the resize shapes, a two-image prompt and the pixel
overrides. Decoding does not depend on the model or the backend, so only one
fixture per family runs the format sweep and the PIL backend (`--formats`); a
new decoder or backend path is added there.

The generator downloads only the processor and tokenizer files. It loads them
through SGLang's `get_processor`, records the spec SGLang's Rust MM gate resolves
(`RustMmProcessor.resolve_spec`), and runs `process_mm_data_async` on the CPU
path, with GPU decode and the GPU image device turned off. It writes
`tests/fixtures/mm_parity/<model>.json` and any missing corpus images. Each case
carries its spec, so `cargo test` runs offline.

## Verify

```sh
cd rust
cargo test -p sglang-processor --locked --test mm_parity --test mm_single_thread
cargo clippy -p sglang-processor --all-targets --locked -- -D warnings
```

`mm_single_thread` guards the host's threading model. `process` runs inline on
the caller's thread; the server's MM worker pool supplies the concurrency.

## End to end

Stage parity proves each piece. End to end proves they compose into the same
served model. Run the Python server and the Rust server (`SGLANG_RUST_SERVER=1`)
on the same checkpoint with the model's cookbook flags, and run every check
below against both. Report each pair of numbers with its harness, settings and
hardware.

Because Python's CUDA defaults deviate from its CPU reference (see "The
reference"), the bit-exact checks run on inputs where the two coincide:
non-JPEG images with `--image-processor-backend pil`. Report the default-config
delta (nvJPEG, CUDA resize) separately, as Python's own.

### 1. Identical inputs and outputs (bit-exact)

Send the same requests to both servers, through `/generate` and through chat.
- **Scheduler inputs:** prompt ids, `image_grid_thw`, offsets and feature bytes
  must be identical. Dump them on both sides.
- **Outputs:** greedy output tokens and their logprobs must be identical, under
  deterministic inference so that batching adds no noise.

### 2. Accuracy benchmarks

Every family runs these before the host serves it, and again after any Dynamo
bump:
- **MMMU.** `simple_eval_mmmu_vlm.MMMUVLMEval` (`run_eval` with
  `eval_name="mmmu"`), the same subset and sampler on both servers. sgl-eval does
  not ship plain MMMU, so this in-tree harness is the allowed exception
  (`.claude/rules/prefer-sgl-eval.md`). `test/registered/vlm/test_rust_native_mm_mmmu.py`
  is the CI form: it runs the Rust server over `/generate` with a gate
  calibrated against Python.
- **MMMU-Pro.** `sgl-eval run mmmu_pro` and `mmmu_pro_vision` against each
  server's `/v1/chat/completions`. Use the version pinned in the `test` extra of
  `python/pyproject.toml`, from its own virtualenv, with the model's
  recommended sampling (`--load-preset-from-model-id`, or pass the settings
  explicitly).

With identical inputs (check 1), Python and Rust scores must agree at
temperature 0. Any gap is a bug: find the first stage that differs. At the
model's recommended sampling, they must agree within run-to-run noise. Measure
that noise with repeated runs on one server; don't assume it. When adding a
family, calibrate its CI gate from these runs and say how in the test.

### 3. Performance benchmarks

Parity must not cost throughput.
- **Serving:** `python -m sglang.bench_serving --dataset-name image` (and
  `--dataset-name mmmu`) against both servers, and against the previous Rust
  server when replacing a pipeline. Compare request throughput, TTFT and
  end-to-end latency at the same concurrency.
- **Preprocessing:** per-image p50/p99 on the MM worker, so a regression in
  decode or resize is visible apart from the model.

## Pitfalls

- A Dynamo bump can change any stage silently. Rerun every fixture after
  bumping `dynamo-multimodal`.
- Pillow is unpinned. A fixture generated under another Pillow version fails
  PIL-backend cases for reasons unrelated to the code under test.
- The Python processor reads `gpu_image_decode` from the class, not the
  instance. Setting it on the instance leaves nvJPEG on.
- Python's `--mm-process-config` image overrides reach the HF processor at call
  time, and Python falls back to decode-and-retokenize when its own
  placeholder count disagrees. The fixture records what Python finally
  produced.
- A test image set limited to PNG hides decoder differences; `sglang-mm`'s JPEG
  mismatch went unnoticed that way. Keep JPEG in every family's cases.
