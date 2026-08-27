# sglang-mm

Rust-accelerated multimodal preprocessing for SGLang. Fused image decode,
fetch, resize, patchify, normalize, and content hash — all parallel and
GIL-released.

Built two ways:

- **PyO3 extension** `sglang.srt.multimodal._core` (features `python,parallel`,
  requested by the wheel build) via setuptools-rust when installing sglang —
  used by Python processors and parity tests.
- **Pure-Rust `rlib`** (default features, i.e. neither) linked by
  `sglang-server`'s MM worker path — that copy needs no pyo3, no libpython, and
  no rayon: it spawns no threads and runs inline on the calling thread, because
  the server supplies concurrency across requests and pins its own cores.
  `tests/rlib_is_single_threaded.rs` guards that from the outside.

## Architecture

```
src/
├── lib.rs                    # module root; PyO3 module (_core) feature-gated
├── pipeline.rs               # the server-pipeline contract: MmFamilyProcessor
│                             #   trait + the carriers (Tensor, TokenLayout, ...)
├── driver.rs                 # model-independent request driver (fetch →
│                             #   decode → process_item → layout → positions)
├── registry.rs               # ImageProcessorSpec registry (Python-facing)
│                             #   + pipeline_from_spec (family factory)
├── common/
│   ├── mod.rs                # thread pool, image decode, content hash, base64
│   ├── fetch.rs              # media source → bytes (data:/base64/file/http)
│   ├── par.rs                # the only fan-out seam (rayon, or inline)
│   ├── resize.rs             # PIL-exact Lanczos + Bicubic resize
│   ├── token_layout.rs       # TokenLayout mechanics (apply_layout + helpers)
│   └── transforms.rs         # reusable primitives: normalize, pad, extract_patches
└── <model>/
    └── mod.rs                # model-specific processor (inkling, qwen_vl, ...)
```

## Server pipeline architecture

`sglang-server`'s MM workers process an image request entirely in Rust.
`driver::process` runs the same fixed steps for every model family:

```
MmInput { text?, input_ids?, images }
  1. per image: fetch_bytes (inline, sequential — see Design notes), then
       fanned out via common::par:
       content hash → decode_rgb → family.process_item()
                                    → ProcessedItem { feature, aux, geometry }
  2. family.layout(input_ids, geometries)   → TokenLayout
       apply_layout: expanded input_ids + per-item (start, end) offsets
  3. family.positions(len, offsets, geoms)  → Rope1D | MRope
  4. Output { input_ids, items: [{feature, aux, hash}], offsets, positions }
```

The driver owns these steps and their failure semantics — any `Err` at any
step rejects the request as a 400 (there is no Python fallback path). A
model family fills in only the `family.*` calls, by implementing
`MmFamilyProcessor` (`pipeline.rs`): it describes its data, it never runs
the request. With qwen as the example:

- **`process_item`** — one decoded image → `ProcessedItem`:
  - `feature`: the model's feature tensor. Qwen: `pixel_values`, from
    smart_resize → bicubic → normalize → patchify. The item identity is the
    driver's hash of the raw encoded source bytes, taken before decode — the
    same role as Python's `hash_feature`, but a different algorithm over
    different input, so never comparable across paths.
  - `aux`: named tensors for the model runner. Qwen: `image_grid_thw`;
    other families: `image_sizes`, `tgt_sizes`, ... (Python:
    `model_specific_data`).
  - `geometry`: whatever this family's `layout`/`positions` need later.
    Qwen: the `[t, h, w]` patch grid.
- **`layout`** — how the prompt expands, described as a value. Example: the
  prompt `[A, <pad>, B]` with one 4-token image becomes

  ```
  [Text(0..1), Media { item: 0, Repeat(<pad> × 4) }, Text(2..3)]
  ```

  which the driver expands to `[A, <pad>, <pad>, <pad>, <pad>, B]` with
  offsets `[(1, 4)]`. Qwen builds this with the `layout_by_placeholder`
  helper; families that interleave tile markers or row separators
  (internvl/minicpm-style) use `Explicit` id sequences instead. Expansion,
  offsets, and position inputs all derive from this one value, so a family
  cannot get them out of sync.
- **`positions`** — `Rope1D` (default: the scheduler needs nothing extra)
  or `MRope` (qwen's image-only fast path).
- **`capabilities`** — which modalities the family accepts; the server
  rejects everything else per family.

Why not give each family the whole request, like Python's per-family
`process_mm_data_async` override? In the server core, every request must
resolve to exactly one accept/reject with its buffers parked in order —
that invariant only holds structurally if the driver owns the flow.

Two things stay in Python permanently: HF config parsing (a family is
configured by a spec JSON of already-resolved params, selected via
`registry::pipeline_from_spec`) and the thin drain adapter mapping
feature/aux tensors to model kwargs. The carriers grow by need, not
speculation: `DecodedMedia` gains a variant per modality (video/audio),
`Geometry` per family style (tile sets), `TensorData` per dtype.

Supported families: `qwen_vl` (Qwen2-VL / 2.5-VL / 3-VL / 3.5; images only).
Adding one = a `MmFamilyProcessor` impl in `src/<model>/mod.rs` plus a
`family` arm in `pipeline_from_spec`.

`common::fetch` matches the Python `get_image_bytes` semantics
(`REQUEST_TIMEOUT` env, `HTTP(S)_PROXY` / `ALL_PROXY` / `NO_PROXY` including
IPv4-CIDR and `host:port` entries) with two deliberate differences: every
source form is capped at 64 MiB — plus 64 items / 256 MiB per request in the
driver — and `file://` URLs actually work (the Python helper passes the
un-stripped URL to `open()`).

## Python API

```python
from sglang.srt.multimodal._core import common, inkling

# Common (model-agnostic)
common.resize_rgb(arr, out_w, out_h)
common.scaled_dims(w, h, rescale_frac, rescale_cap)
common.image_decode_rgb(bytes)          # -> (h, w, ndarray)
common.content_hash(bytes)              # -> u64 (blake3, truncated)
common.fetch_bytes(source)              # -> bytes (data:/base64/file/http)
common.base64_decode(str)               # -> bytes

# Model-specific
inkling.preprocess_images(list[bytes], ps, frac, cap)  # -> [(h, w, bits, hash), ...]
inkling.decode_patchify(bytes, ps, frac, cap)
inkling.decode_patchify_batch(list[bytes], ps, frac, cap)
inkling.patchify_rgb(arr, patch_size)
```

## Adding a new model

1. Create `src/<model_name>/mod.rs`:

```rust
use crate::common;
use crate::common::par;
use crate::registry::ImageProcessorSpec;

pub struct MyModelProcessor;

impl ImageProcessorSpec for MyModelProcessor {
    fn name(&self) -> &'static str {
        "my_model"
    }

    fn preprocess_batch(
        &self,
        datas: &[Vec<u8>],
        patch_size: usize,
        rescale_frac: Option<f64>,
        rescale_cap: Option<i64>,
    ) -> Result<Vec<(usize, usize, Vec<u16>, u64)>, String> {
        // Always fan out through `par`, never rayon directly: that is what
        // keeps the rlib build rayon-free (see Design notes).
        par::try_map(datas, |data| {
            let hash = common::content_hash_u64(data);
            let (rgb, h, w) = common::decode_rescale(data, rescale_frac, rescale_cap)?;
            // Use common::transforms::* or model-specific logic
            let patches = my_patchify(&rgb, h, w, patch_size);
            Ok((h, w, patches, hash))
        })
    }
}
```

2. Register in `src/registry.rs` `default_registry()`.

3. Add PyO3 bindings in `src/<model_name>/mod.rs` with a `register()` function.

4. Wire up in `src/lib.rs`: `mod my_model;` and `my_model::register(m)?;`.

5. Add Python processor class that calls `from sglang.srt.multimodal._core import my_model`.

## Available transform primitives (`common::transforms`)

| Function | Description |
|----------|-------------|
| `normalize_rgb_f32` | Single-pass `(pixel/255 - mean) / std` |
| `pad_to_grid` | Pad HWC image to grid-aligned dimensions |
| `extract_patches_hwc` | Reshape padded image into `[N, ph, pw, C]` patches |
| `patch_grid` | Compute `(nph, npw)` for given image and patch size |

## Design notes

- All fan-out goes through `common::par`, so whether this crate owns threads is
  decided by the `parallel` feature alone. With it on: CPU pool capped at
  `min(8, cores)` (override `SGL_MM_RS_THREADS`). With it off: no rayon, no
  threads, everything inline. Output is bit-identical either way — the fan-outs
  are order-preserving maps and writes into disjoint slices, never reductions.
  Note that sizing a pool to 1 is *not* the same as off: `install` blocks the
  caller and would serialize every concurrent request in the process.
- Media fetch is blocking I/O and deliberately never enters the CPU pool; it
  runs inline and sequentially in `driver::process`. Contract: callers on a
  fixed worker pool (sglang-server) must resolve I/O-backed string sources —
  URLs *and* file paths (a network mount can hang far longer than any HTTP
  timeout) — on their own I/O layer and pass bytes, so workers never block on
  I/O. `data:`/base64 sources are pure CPU and stay on the worker.
- PNG decode is bit-exact vs PIL; JPEG may differ by ±1 LSB. WebP/GIF/BMP also
  decode (GIF: first frame); their parity is not bit-audited. Samples deeper
  than 8 bits are rejected rather than rescaled (PIL clips instead).
- Lanczos and Bicubic resize are bit-exact clones of PIL's fixed-point
  implementations.
- `common::content_hash_u64` is blake3, *not* Python's SHA-256
  `mm_utils.data_hash`. Hashes are consistent within one path only.

## Build

Automatically built when installing sglang:
```bash
pip install -e "python"
```

Or standalone for development (the PyO3 bindings are behind a non-default
feature — see `[features]` in `Cargo.toml` for why):
```bash
cd rust/sglang-mm
pip install maturin
maturin develop --release --features python
```

## Test

```bash
cd rust/sglang-mm
cargo test --no-default-features  # pure-Rust unit tests (CI: pr-test-rust-exts)
python tests/generate_golden.py   # regenerate fixtures
pytest tests/test_golden.py       # regression tests
python bench/bench_parity.py      # parity + benchmark
```

Scheduler-boundary parity tests against the real HF processors live in
`test/registered/unit/multimodal/rust/`.
