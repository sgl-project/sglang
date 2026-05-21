# VLM preprocessor — Rust prototype + 3-way benchmark

A Rust-side replacement for SGLang's Python VLM image preprocessor (Qwen2-VL /
Qwen3-VL family), benchmarked head-to-head against HuggingFace
`Qwen2VLImageProcessor` (the current sglang CPU path) and `smg`'s
`llm-multimodal` Rust preprocessor.

The pipeline applies six "new tricks" relative to smg's straight HF translation:

1. **Banker's-rounded `smart_resize`** — matches Python `round()` (half-to-even).
   Without this, `image_grid_thw` can be off-by-1 vs HF on certain fractions,
   corrupting downstream token counts. (`bench-vlm-preproc/src/smart_resize.rs`)

2. **libjpeg-turbo scaled iDCT** via the `turbojpeg` crate. 16 fractional
   scaling factors (M/8 for M ∈ 1..16) vs jpeg-decoder's 4 power-of-2 factors,
   plus NEON-accelerated decode. We pick the smallest factor whose output is
   still ≥ smart_resize target.

3. **PNG row-streamed decode** via the `png` crate's `next_row()`, writing RGB
   directly into a pooled buffer (handles RGBA / Grayscale / GrayscaleAlpha
   inline).

4. **Thread-local buffer pools** (`RGB_POOL`, `RESIZED_POOL`) + caller-owned
   `Vec<f32>` output (`preprocess_image_into`) — no per-request allocation
   after warm-up.

5. **Zero-copy resize** — `fast_image_resize::ImageRef::new(w, h, &src, U8x3)`
   borrows the decoded slice directly.

6. **NEON-vectorized fused normalize + patch-write** — `vld3q_u8` does the
   stride-3 RGB deinterleave in one instruction, then `vfmaq_f32` applies the
   precomputed `(scale, bias)` for all 3 channels in parallel and writes
   directly into the final `[num_patches, C·Tp·P·P]` layout. Scalar fallback for
   non-aarch64.

`bench-smg/` is a sibling Cargo project that depends on smg's `llm-multimodal`
via path so we can drive its `Qwen2VLProcessor::preprocess()` on the same image
bytes.

## Layout

```
bench-vlm-preproc/
  Cargo.toml             # turbojpeg + png + fast_image_resize + rayon + bytemuck
  src/
    smart_resize.rs      # banker's-rounded Qwen smart_resize + tests
    preprocess.rs        # decode → resize → normalize+patch (the pipeline)
    main.rs              # bench harness (per-image + rayon batch)
    dump.rs              # writes pixel_values.f32 + .json for cross-validation
  python/
    gen_fixtures.py      # procedural test image generator (deterministic)
    bench_hf.py          # HF transformers Qwen2VLImageProcessor baseline
    validate.py          # diffs Rust output vs HF (shape, grid_thw, pixel diff)
bench-smg/
  Cargo.toml             # path-deps llm-multimodal from ../../../smg
  src/main.rs            # runs smg's Qwen2VLProcessor on real bytes
```

## Repro

The bench needs:

- macOS (NEON path) or Linux aarch64. On x86 the SIMD path is a scalar fallback.
- Rust toolchain (tested on 1.94).
- Homebrew `jpeg-turbo` (for the `turbojpeg` crate's pkg-config lookup).
- Python 3.11+ for the HF baseline.
- smg checked out alongside sglang at `../../smg/` (relative to this scratch dir
  it's `/Users/alex.nails/workplace/smg`).

### 0. One-time setup

```bash
brew install jpeg-turbo
# from scratch/bench-vlm-preproc/
python3 -m venv .venv
source .venv/bin/activate
pip install Pillow numpy torch torchvision transformers
python3 python/gen_fixtures.py    # writes 8 deterministic JPEG/PNG fixtures
```

### 1. Build the Rust bench

```bash
# from scratch/bench-vlm-preproc/
PKG_CONFIG_PATH=/opt/homebrew/opt/jpeg-turbo/lib/pkgconfig cargo build --release
```

### 2. Build the smg bench

```bash
# from scratch/bench-smg/
cargo build --release
```

### 3. Run all three

```bash
# from scratch/bench-vlm-preproc/
./target/release/bench --fixtures fixtures --iters 30 --batch-iters 20
source .venv/bin/activate && python3 python/bench_hf.py

# from scratch/bench-smg/
./target/release/bench-smg ../bench-vlm-preproc/fixtures
```

### 4. Cross-validate Rust output against HF (correctness)

```bash
# from scratch/bench-vlm-preproc/
./target/release/dump --image fixtures/medium_1024x768.jpg --out /tmp/rust_dump
source .venv/bin/activate && python3 python/validate.py \
    fixtures/medium_1024x768.jpg /tmp/rust_dump
```

Expected output: shape `(3996, 1176)`, `grid_thw [1,54,74]`, `mean abs diff
≈ 0.009` (bilinear-kernel variation between `fast_image_resize` and
`torchvision`; HF processor + Rust patch layout are bit-identical to HF outside
of the resize step).

## Results (Apple M-series, 14 cores, 48 GB RAM)

Per-image total time (decode + resize + normalize + patch), µs:

| fixture                 | **Rust (this) ** | HF Python  | smg        |
|-------------------------|-----------------:|-----------:|-----------:|
| small_512×512.jpg       |       **1,698**  |    1,797   |    2,546   |
| small_512×512.png       |       **2,259**  |    3,703   |    2,988   |
| medium_1024×768.jpg     |          5,260   | **4,236**  |    8,576   |
| medium_1024×768.png     |       **7,029**  |   10,142   |    9,673   |
| large_2048×1536.jpg     |      **12,027**  |   12,791   |   17,031   |
| large_2048×1536.png     |      **17,857**  |   35,825   |   23,396   |
| xl_4096×3072.jpg        |      **27,298**  |   43,294   |   44,710   |
| xl_4096×3072.png        |      **66,402**  |  135,719   |   67,879   |

Rust wins 7/8 single-image fixtures.

Batch (8 images, with rayon vs sequential):

| pipeline                  | wall time | per image |
|---------------------------|----------:|----------:|
| **Rust + rayon**          | **70.4 ms** | **8.8 ms** |
| HF Python (sequential)    |   250.6 ms |  31.3 ms  |
| smg (sequential)          |  ~177  ms |  ~22  ms  |

**Rust + rayon is 3.56× faster than HF end-to-end.**

## Stage-by-stage gains over a naive Rust translation (v1 → v3)

| stage          | v1            | v3 (this)        | win source                |
|----------------|--------------:|-----------------:|---------------------------|
| JPEG decode    | 3,399 µs (med) | 1,810 µs (med) | libjpeg-turbo SIMD + finer scaled iDCT |
| patch-write    | 1,175 µs (med) |   451 µs (med) | NEON `vld3q_u8` + parallel-channel `vfmaq_f32` |
