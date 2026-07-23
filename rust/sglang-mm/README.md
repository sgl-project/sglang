# sglang-mm

Rust-accelerated multimodal preprocessing for SGLang. Fused image decode,
fetch, resize, patchify, normalize, and content hash — all parallel and
GIL-released.

Built two ways:

- **PyO3 extension** `sglang.srt.multimodal._core` (feature `python`, default)
  via setuptools-rust when installing sglang — used by Python processors and
  parity tests.
- **Pure-Rust `rlib`** (`default-features = false`) linked by `sglang-server`'s
  native MM worker path — no pyo3 in that dependency graph (the server pins
  its own, newer pyo3).

## Architecture

```
src/
├── lib.rs                    # module root; PyO3 module (_core) feature-gated
├── registry.rs               # ImageProcessorSpec registry (Python-facing)
│                             # + VisionProcessor trait / native_pipeline_from_spec
│                             #   (pure-Rust pipeline driven by sglang-server)
├── common/
│   ├── mod.rs                # thread pool, image decode, SHA256 hash, base64
│   ├── fetch.rs              # media source → bytes (data:/base64/file/http)
│   ├── resize.rs             # PIL-exact Lanczos + Bicubic resize
│   ├── tokens.rs             # placeholder-id expansion + per-item offsets
│   └── transforms.rs         # reusable primitives: normalize, pad, extract_patches
└── <model>/
    └── mod.rs                # model-specific processor (inkling, qwen_vl, ...)
```

## Native server pipeline (`VisionProcessor`)

`sglang-server`'s MM workers process image-only requests for supported model
families entirely in Rust: `common::fetch` → decode → the family's
`VisionProcessor` (resize/normalize/patchify + M-RoPE) → `common::tokens`
placeholder expansion. The Python side selects the family by passing a spec
JSON (`{"family": "qwen_vl", ...resolved processor params}`) to
`registry::native_pipeline_from_spec`. Anything outside a family's scope
(video/audio, precomputed features, unknown source shapes, placeholder
mismatches) falls back per-request to the Python `mm_processor` path.

Supported families: `qwen_vl` (Qwen2-VL / 2.5-VL / 3-VL / 3.5; images only).
Adding one = a `VisionProcessor` impl in `src/<model>/mod.rs` plus a `family`
arm in `native_pipeline_from_spec`.

## Python API

```python
from sglang.srt.multimodal._core import common, inkling

# Common (model-agnostic)
common.resize_rgb(arr, out_w, out_h)
common.scaled_dims(w, h, rescale_frac, rescale_cap)
common.image_decode_rgb(bytes)          # -> (h, w, ndarray)
common.data_hash(bytes)                 # -> u64 SHA256
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
use crate::registry::ImageProcessorSpec;
use rayon::prelude::*;

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
        common::pool().install(|| {
            datas.par_iter().map(|data| {
                let hash = common::sha256_u64(data);
                let (rgb, h, w) = common::decode_rescale(data, rescale_frac, rescale_cap)?;
                // Use common::transforms::* or model-specific logic
                let patches = my_patchify(&rgb, h, w, patch_size);
                Ok((h, w, patches, hash))
            }).collect()
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

- Thread pool capped at `min(8, cores)`. Override: `SGL_MM_RS_THREADS`.
- PNG decode is bit-exact vs PIL; JPEG may differ by ±1 LSB.
- Lanczos resize is a bit-exact clone of PIL's fixed-point implementation.

## Build

Automatically built when installing sglang:
```bash
pip install -e "python"
```

Or standalone for development:
```bash
cd rust/sglang-mm
pip install maturin
maturin develop --release
```

## Test

```bash
python bench/generate_golden.py   # regenerate fixtures
pytest bench/test_golden.py       # regression tests
python bench/bench_parity.py      # parity + benchmark
```
