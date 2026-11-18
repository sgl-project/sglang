# sglang-mm

Rust-accelerated multimodal preprocessing for SGLang. Fused image decode,
fetch, resize, patchify, normalize, and content hash — all parallel and
GIL-released.

The processing stages and their dependencies are in
[docs/pipeline.md](docs/pipeline.md).

Built two ways:

- **PyO3 extension** `sglang.srt.rust_extensions._multimodal` (features
  `python,parallel`, requested by the wheel build) via setuptools-rust when
  installing sglang — used by Python processors and parity tests.
- **Pure-Rust `rlib`** (default features, i.e. neither) linked by
  `sglang-server`'s MM worker path — that copy needs no pyo3, no libpython, and
  no rayon: it spawns no threads and runs inline on the calling thread, because
  the server supplies concurrency across requests and pins its own cores.
  `tests/rlib_is_single_threaded.rs` guards that from the outside.

## Architecture

```
src/
├── lib.rs                    # module root; PyO3 module (_multimodal) feature-gated
├── tensor.rs                 # Tensor / TensorData carriers
├── common/
│   ├── mod.rs                # thread pool, image decode, content hash
│   ├── fetch.rs              # media source → bytes (data:/base64/file/http)
│   ├── par.rs                # the only fan-out seam (rayon, or inline)
│   ├── resize.rs             # PIL-exact Lanczos + Bicubic resize
│   └── expansion.rs          # Placeholder expansion and media spans in token ids
└── <model>/
    └── mod.rs                # model-specific processor (inkling, qwen_vl, ...)
```

## Media fetch

`common::fetch` matches the Python `get_image_bytes` semantics
(`REQUEST_TIMEOUT` env, `HTTP(S)_PROXY` / `ALL_PROXY` / `NO_PROXY` including
IPv4-CIDR and `host:port` entries) with deliberate safety bounds: every media
source contributes to a shared 1.25 GiB request budget, and each remote I/O
stream has an additional 64 MiB cap. As in Python, there is no fixed item-count
cap. `file://` URLs also work (the Python helper passes the un-stripped URL to
`open()`).

## Python API

```python
from sglang.srt.rust_extensions._multimodal import common, inkling

# Common (model-agnostic)
common.resize_rgb(arr, out_w, out_h)
common.scaled_dims(w, h, rescale_frac, rescale_cap)
common.image_decode_rgb(bytes)          # -> (h, w, ndarray)
common.content_hash(bytes)              # -> u64 (blake3, truncated)
common.content_digest(bytes)            # -> "blake3:<64 hex>" (full blake3)
common.fetch_bytes(source)              # -> bytes (data:/base64/file/http)
common.expand_token_placeholders(input_ids, spec, start_len=0)  # -> list[int]
common.locate_media(input_ids, token_id, expansions)           # -> [[(start, end), ...], ...]

# Model-specific
inkling.preprocess_images(list[bytes], ps, frac, cap)  # -> [(h, w, bits, hash), ...]
inkling.decode_patchify(bytes, ps, frac, cap)
inkling.decode_patchify_batch(list[bytes], ps, frac, cap)
inkling.patchify_rgb(arr, patch_size)
```

## Adding a new model

1. Create `src/<model_name>/mod.rs` with the model's pure-Rust functions.
   Always fan out through `common::par`, never rayon directly: that is what
   keeps the rlib build rayon-free (see Design notes).

2. Add PyO3 bindings in `src/<model_name>/mod.rs` with a `register()` function.

3. Wire up in `src/lib.rs`: `mod my_model;` and `my_model::register(m)?;`.

4. Add Python processor class that calls
   `from sglang.srt.rust_extensions._multimodal import my_model`.

## Design notes

- All fan-out goes through `common::par`, so whether this crate owns threads is
  decided by the `parallel` feature alone. With it on: CPU pool capped at
  `min(8, cores)` (override `SGLANG_MM_RS_THREADS`). With it off: no rayon, no threads,
  everything inline. Output is bit-identical either way — the fan-outs
  are order-preserving maps and writes into disjoint slices, never reductions.
  Note that sizing a pool to 1 is *not* the same as off: `install` blocks the
  caller and would serialize every concurrent request in the process.
- Media fetch is blocking I/O and deliberately never enters the CPU pool.
  Contract: callers on a fixed worker pool (sglang-server) must resolve I/O-backed string sources —
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
