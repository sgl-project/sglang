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

## Results (Apple M-series, 14 cores, 48 GB RAM) — v4

Per-image total time (decode + fused-resize+normalize+patch), µs:

| fixture                 | **Rust (this)** | HF Python  | smg        |
|-------------------------|----------------:|-----------:|-----------:|
| small_512×512.jpg       |         **823** |    1,874   |    2,496   |
| small_512×512.png       |       **1,366** |    3,812   |    2,932   |
| medium_1024×768.jpg     |       **2,108** |    4,161   |    7,875   |
| medium_1024×768.png     |       **3,740** |    9,979   |    9,149   |
| large_2048×1536.jpg     |       **7,794** |   12,957   |   15,796   |
| large_2048×1536.png     |      **14,219** |   34,854   |   21,350   |
| xl_4096×3072.jpg        |      **23,350** |   41,761   |   44,153   |
| xl_4096×3072.png        |      **56,941** |  131,916   |   66,270   |
| xl_4096×3072_rst16.jpg  |      **21,242** |   42,254   |   44,271   |

**Rust wins all 9 fixtures.** vs HF Python: 1.66× – 2.79× faster per image.
vs smg: 1.16× – 3.74× faster.

Batch (9 images, rayon vs sequential):

| pipeline                  | wall time   | per image  |
|---------------------------|------------:|-----------:|
| **Rust + rayon**          | **69.9 ms** | **7.77 ms** |
| HF Python (sequential)    |   291.6 ms  |   32.4 ms  |
| smg (sequential)          |   ~185 ms   |   ~20.5 ms |

**Rust + rayon is 4.17× faster than HF end-to-end.**

## v4 pipeline additions over v3

| change | win |
|---|---|
| **Fused single-pass kernel** (decode → bilinear+normalize+patch in one pass; no intermediate u8 resized buffer) | post-decode stage **5–7× faster** at all sizes |
| **Sub-image rayon work-stealing** (split merge-block-rows across workers) | scales the fused kernel across all cores even for single-image |
| **RST-aware parallel JPEG decode** (when JPEG has `0xFFD0..0xFFD7` markers, decode strips concurrently via `tj3SetCroppingRegion` FFI; falls back to single-thread when no RSTs) | ~10% additional decode win on `*_rst*.jpg`; **bit-identical** to single-thread path |

## Stage gains over a naive Rust translation (v1 → v4)

| stage                          | v1            | v4 (this)        | source                |
|--------------------------------|--------------:|-----------------:|-----------------------|
| JPEG decode (medium)           | 3,399 µs      | 1,660 µs (1.9×) | libjpeg-turbo SIMD + finer M/8 scaled iDCT |
| post-decode (medium)           | 4,044 µs      |   448 µs (9.0×) | fused single-pass kernel + sub-image rayon |
| **end-to-end total (medium)**  | **7,415 µs**  | **2,108 µs (3.5×)** | combined |
