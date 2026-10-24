# sglang-mm

Rust-accelerated image preprocessing for SGLang's Python multimodal processors
(Inkling, DeepSeek-V4.1, Muse Spark): fused decode, resize, patchify,
normalize and content hash, parallel and GIL-released.

Built as the PyO3 extension `sglang.srt.rust_extensions._multimodal` (features
`python,parallel`, requested by the wheel build) via setuptools-rust when
installing sglang. The pure-Rust `rlib` (default features) exists for the
crate's own tests.

The Rust server's multimodal pipeline is not here: it lives in
`sglang-processor` (`src/mm/`, on Dynamo's `dynamo-multimodal`).

## Architecture

```
src/
├── lib.rs                    # module root; PyO3 module (_multimodal) feature-gated
├── registry.rs               # ImageProcessorSpec registry (Python-facing)
├── common/
│   ├── mod.rs                # thread pool, image decode, content hash, base64
│   ├── par.rs                # the only fan-out seam (rayon, or inline)
│   ├── resize.rs             # PIL-exact Lanczos + Bicubic, ATen uint8 resize
│   └── transforms.rs         # reusable primitives: normalize, pad, extract_patches
└── <model>/
    └── mod.rs                # model-specific processor (inkling, dsv41)
```

## Python API

```python
from sglang.srt.rust_extensions._multimodal import common, inkling

# Common (model-agnostic)
common.resize_rgb(arr, out_w, out_h)
common.scaled_dims(w, h, rescale_frac, rescale_cap)
common.image_decode_rgb(bytes)          # -> (h, w, ndarray)
common.content_hash(bytes)              # -> u64 (blake3, truncated)
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

5. Add Python processor class that calls
   `from sglang.srt.rust_extensions._multimodal import my_model`.

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
