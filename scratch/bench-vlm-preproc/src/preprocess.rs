//! Qwen2-VL preprocessor: bytes -> [num_patches, C*Tp*P*P] f32 + grid_thw.
//!
//! Pipeline (per image):
//!   1. sniff format
//!   2. JPEG: header parse → smart_resize → libjpeg-turbo iDCT at smallest M/8 ≥ target
//!      PNG:  row-streamed decode via `png` crate
//!   3. zero-copy bilinear to exact target via fast_image_resize (TypedImageRef on pool)
//!   4. NEON (aarch64) / scalar fused-normalize patch-write into final f32 buffer
//!
//! Output layout matches HF Qwen2VLImageProcessor pixel_values exactly:
//!   patches[gh_block, gw_block, mh, mw, c, tp, ph, pw]  flattened to
//!   (Hg/M*Wg/M*M*M, C*Tp*P*P). For static images T=1, Tp=2 (duplicated).
//!
//! Pools (thread-local):
//!   - RGB_POOL:     decoded RGB image bytes
//!   - RESIZED_POOL: bilinear-resized RGB bytes
//!   - OUTPUT_POOL:  f32 pixel_values buffer
//!   - RESIZER:      fast_image_resize::Resizer (internal scratch)

use std::cell::RefCell;

use anyhow::{Context, Result};
use fast_image_resize::{
    images::{Image, ImageRef},
    FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
};

use crate::smart_resize::{smart_resize, SmartResizeCfg};

/// Qwen2-VL processor config (CLIP-style normalization).
#[derive(Debug, Clone, Copy)]
pub struct QwenCfg {
    pub patch_size: u32,
    pub merge_size: u32,
    pub temporal_patch_size: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub resize: SmartResizeCfg,
}

impl Default for QwenCfg {
    fn default() -> Self {
        Self {
            patch_size: 14,
            merge_size: 2,
            temporal_patch_size: 2,
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.26130258, 0.27577711],
            resize: SmartResizeCfg::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PreprocessTimings {
    pub decode_ns: u64,
    pub resize_ns: u64,
    pub normpack_ns: u64,
    pub total_ns: u64,
    pub decoded_h: u32,
    pub decoded_w: u32,
    pub target_h: u32,
    pub target_w: u32,
    pub num_patches: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct PreprocessOut {
    pub num_patches: u32,
    pub patch_features: u32,
    pub grid_thw: [u32; 3],
}

// ---------- format sniffing ----------

fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF
}
fn is_png(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && &bytes[..8] == b"\x89PNG\r\n\x1a\n"
}

// ---------- JPEG RST-aware parallel decode ----------
//
// JPEG restart markers (0xFFD0..0xFFD7) appear between independent entropy-coded
// segments. When present, the entropy decoder state resets at each marker, so we
// can decode strips in parallel — each strip's call to libjpeg-turbo starts its
// entropy decode at the closest RST boundary to the requested crop.y, skipping
// upstream work it would otherwise have to redo.
//
// Most JPEGs DO NOT contain RST markers (Pillow default, cameras). For those we
// transparently fall back to the existing single-thread decode path. RST-equipped
// JPEGs (`Pillow.save(..., restart_marker_blocks=N)`, `cjpeg -restart N`) get the
// fast path.

/// Returns true if the JPEG bytes contain at least one RST marker (0xFFD0..0xFFD7).
/// Fast scan: just looks for 0xFF followed by 0xD0..0xD7 (mid-stream). Doesn't try
/// to skip the JPEG header, since RSTs are valid only in entropy-coded data and
/// don't otherwise appear in headers.
fn has_rst_markers(bytes: &[u8]) -> bool {
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == 0xFF {
            let m = bytes[i + 1];
            if (0xD0..=0xD7).contains(&m) {
                return true;
            }
            // 0xFF 0x00 is escaped data; skip both.
            i += 2;
            continue;
        }
        i += 1;
    }
    false
}

/// Parallel-decode a JPEG that has RST markers, into the supplied rgb_pool.
/// Uses raw turbojpeg-sys FFI to call tj3SetCroppingRegion per worker.
/// Falls back to `decode_jpeg_scaled` if any error occurs (e.g. no RSTs aligned).
fn decode_jpeg_parallel_rst(
    bytes: &[u8],
    target_h: u32,
    target_w: u32,
    rgb_pool: &mut Vec<u8>,
    n_strips: usize,
) -> Result<(u32, u32)> {
    use turbojpeg_sys::*;

    // Probe header + pick scaling factor first.
    let header = turbojpeg::read_header(bytes).context("turbojpeg read_header")?;
    let factors = turbojpeg::Decompressor::supported_scaling_factors();
    let mut chosen = turbojpeg::ScalingFactor::new(1, 1);
    for sf in factors.into_iter().rev() {
        if sf.num() > sf.denom() {
            continue;
        }
        let h = sf.scale(header.height);
        let w = sf.scale(header.width);
        if h >= target_h as usize && w >= target_w as usize {
            chosen = sf;
            break;
        }
    }
    let scaled_h = chosen.scale(header.height) as u32;
    let scaled_w = chosen.scale(header.width) as u32;
    let needed = (scaled_h as usize) * (scaled_w as usize) * 3;
    rgb_pool.clear();
    rgb_pool.resize(needed + NEON_TAIL_PAD, 0u8);

    // Compute MCU height after scaling. For 4:2:0 JPEGs MCU is 16x16 in unscaled
    // space → 16*num/denom in scaled space. We treat MCU height as a strip
    // alignment unit; if RSTs aren't aligned, libjpeg-turbo will slow-path.
    // 16 is the common case; if the JPEG uses other subsampling the strips just
    // become misaligned and parallelism degrades to sequential, but it still
    // decodes correctly.
    let mcu_h_scaled =
        ((16usize * chosen.num()) as f32 / chosen.denom() as f32).ceil().max(1.0) as u32;

    // Align strip boundaries to MCU height in scaled output.
    let strip_h = ((scaled_h / n_strips as u32) / mcu_h_scaled).max(1) * mcu_h_scaled;
    if strip_h * (n_strips as u32 - 1) >= scaled_h || strip_h == 0 {
        // Strips would overshoot or degenerate; fall back to sequential.
        return decode_jpeg_scaled(bytes, target_h, target_w, rgb_pool);
    }

    let row_stride = (scaled_w as usize) * 3;
    let out_addr = rgb_pool.as_mut_ptr() as usize;

    let strip_ranges: Vec<(u32, u32)> = (0..n_strips)
        .map(|i| {
            let y0 = i as u32 * strip_h;
            let y1 = if i == n_strips - 1 { scaled_h } else { (i as u32 + 1) * strip_h };
            (y0, y1)
        })
        .collect();

    let result = std::sync::Mutex::new(Ok::<(), String>(()));

    rayon::scope(|s| {
        for &(y0, y1) in &strip_ranges {
            let res = &result;
            s.spawn(move |_| {
                let r = decode_jpeg_strip_ffi(
                    bytes,
                    chosen,
                    scaled_w,
                    scaled_h,
                    y0,
                    y1,
                    row_stride,
                    out_addr,
                );
                if let Err(e) = r {
                    let mut g = res.lock().unwrap();
                    if g.is_ok() {
                        *g = Err(e);
                    }
                }
            });
        }
    });

    if let Err(e) = result.into_inner().unwrap() {
        // Fall back if any strip failed.
        return decode_jpeg_scaled(bytes, target_h, target_w, rgb_pool)
            .context(format!("parallel decode failed ({}); fallback", e));
    }

    Ok((scaled_h, scaled_w))
}

/// Single-strip FFI decode. Out-of-strip pixels in the destination row stride
/// are untouched by this call.
fn decode_jpeg_strip_ffi(
    bytes: &[u8],
    chosen: turbojpeg::ScalingFactor,
    scaled_w: u32,
    _scaled_h: u32,
    y0: u32,
    y1: u32,
    row_stride: usize,
    out_addr: usize,
) -> Result<(), String> {
    use turbojpeg_sys::*;
    unsafe {
        let handle = tj3Init(TJINIT_TJINIT_DECOMPRESS as i32);
        if handle.is_null() {
            return Err("tj3Init".into());
        }

        // Re-read header on this thread's handle.
        let hdr_rc = tj3DecompressHeader(handle, bytes.as_ptr(), bytes.len() as _);
        if hdr_rc != 0 {
            tj3Destroy(handle);
            return Err("tj3DecompressHeader".into());
        }

        let sf = tjscalingfactor {
            num: chosen.num() as i32,
            denom: chosen.denom() as i32,
        };
        if tj3SetScalingFactor(handle, sf) != 0 {
            tj3Destroy(handle);
            return Err("tj3SetScalingFactor".into());
        }

        let crop = tjregion {
            x: 0,
            y: y0 as i32,
            w: scaled_w as i32,
            h: (y1 - y0) as i32,
        };
        if tj3SetCroppingRegion(handle, crop) != 0 {
            tj3Destroy(handle);
            return Err("tj3SetCroppingRegion".into());
        }

        // Destination pointer to the strip's first row.
        let strip_ptr = (out_addr as *mut u8).add((y0 as usize) * row_stride);

        let rc = tj3Decompress8(
            handle,
            bytes.as_ptr(),
            bytes.len() as _,
            strip_ptr,
            row_stride as i32,
            TJPF_TJPF_RGB as i32,
        );
        tj3Destroy(handle);
        if rc != 0 {
            return Err("tj3Decompress8".into());
        }
        Ok(())
    }
}

// ---------- JPEG decode via libjpeg-turbo ----------

fn decode_jpeg_scaled(
    bytes: &[u8],
    target_h: u32,
    target_w: u32,
    rgb_pool: &mut Vec<u8>,
) -> Result<(u32, u32)> {
    let mut decomp = turbojpeg::Decompressor::new().context("tjInit decompress")?;
    let header = decomp
        .read_header(bytes)
        .context("turbojpeg read_header")?;

    // libjpeg-turbo exposes 16 fractional scaling factors (M/8 for M=1..16).
    // libjpeg-turbo returns them sorted **descending by ratio** (2/1, …, 1/8).
    // We want the SMALLEST factor whose scaled output is still >= target — iterate
    // in reverse and pick the first that satisfies.
    let factors = turbojpeg::Decompressor::supported_scaling_factors();
    let mut chosen = turbojpeg::ScalingFactor::new(1, 1);
    for sf in factors.into_iter().rev() {
        if sf.num() > sf.denom() {
            continue; // skip upscaling (>1)
        }
        let h = sf.scale(header.height);
        let w = sf.scale(header.width);
        if h >= target_h as usize && w >= target_w as usize {
            chosen = sf;
            break;
        }
    }
    decomp
        .set_scaling_factor(chosen)
        .context("turbojpeg set_scaling_factor")?;
    let scaled_h = chosen.scale(header.height);
    let scaled_w = chosen.scale(header.width);
    let needed = scaled_h * scaled_w * 3;
    rgb_pool.clear();
    rgb_pool.resize(needed + NEON_TAIL_PAD, 0u8);

    // Image view borrows the first `needed` bytes; NEON in patchify reads up to
    // `needed + 6` bytes (vld3q_u8 of 48 bytes at last patch).
    let image = turbojpeg::Image {
        pixels: &mut rgb_pool[..needed],
        width: scaled_w,
        pitch: scaled_w * 3,
        height: scaled_h,
        format: turbojpeg::PixelFormat::RGB,
    };
    decomp
        .decompress(bytes, image)
        .context("turbojpeg decompress")?;

    Ok((scaled_h as u32, scaled_w as u32))
}

/// Tail padding bytes appended to every RGB buffer so the NEON vld3q_u8 inner
/// loop can safely overread up to 48 bytes at the last patch boundary.
const NEON_TAIL_PAD: usize = 48;

// ---------- PNG decode via row-streamed `png` crate ----------
//
// Tried zune-png 0.5 with new_fast() options; it was 1.6x slower than `png`+
// `miniz_oxide` on aarch64-apple-darwin for our fixtures. Keep `png` for now.

fn decode_png(bytes: &[u8], rgb_pool: &mut Vec<u8>) -> Result<(u32, u32)> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().context("png read_info")?;
    let (w, h, color) = {
        let info = reader.info();
        (info.width, info.height, info.color_type)
    };

    let needed = (w as usize) * (h as usize) * 3;
    rgb_pool.clear();
    rgb_pool.resize(needed + NEON_TAIL_PAD, 0u8);

    let mut out_off = 0usize;
    while let Some(row) = reader.next_row().context("png next_row")? {
        let r: &[u8] = row.data();
        match color {
            png::ColorType::Rgb => {
                rgb_pool[out_off..out_off + r.len()].copy_from_slice(r);
                out_off += r.len();
            }
            png::ColorType::Rgba => {
                let pixels = r.len() / 4;
                for i in 0..pixels {
                    rgb_pool[out_off + i * 3] = r[i * 4];
                    rgb_pool[out_off + i * 3 + 1] = r[i * 4 + 1];
                    rgb_pool[out_off + i * 3 + 2] = r[i * 4 + 2];
                }
                out_off += pixels * 3;
            }
            png::ColorType::Grayscale => {
                for i in 0..r.len() {
                    let v = r[i];
                    rgb_pool[out_off + i * 3] = v;
                    rgb_pool[out_off + i * 3 + 1] = v;
                    rgb_pool[out_off + i * 3 + 2] = v;
                }
                out_off += r.len() * 3;
            }
            png::ColorType::GrayscaleAlpha => {
                for i in 0..(r.len() / 2) {
                    let v = r[i * 2];
                    rgb_pool[out_off + i * 3] = v;
                    rgb_pool[out_off + i * 3 + 1] = v;
                    rgb_pool[out_off + i * 3 + 2] = v;
                }
                out_off += (r.len() / 2) * 3;
            }
            png::ColorType::Indexed => anyhow::bail!("indexed PNG not handled after EXPAND"),
        }
    }

    Ok((h, w))
}

// ---------- thread-local pools ----------

thread_local! {
    static RGB_POOL:     RefCell<Vec<u8>>  = RefCell::new(Vec::with_capacity(1024 * 1024));
    static RESIZED_POOL: RefCell<Vec<u8>>  = RefCell::new(Vec::with_capacity(1024 * 1024));
    static OUTPUT_POOL:  RefCell<Vec<f32>> = RefCell::new(Vec::with_capacity(1024 * 1024));
    static RESIZER:      RefCell<Resizer>  = RefCell::new(Resizer::new());
}

/// Bilinear resize RGB8 in-place using a borrowed src and pool-backed dst.
fn resize_rgb_zero_copy(
    src: &[u8],
    src_h: u32,
    src_w: u32,
    dst_h: u32,
    dst_w: u32,
    dst_pool: &mut Vec<u8>,
) -> Result<()> {
    let needed = (dst_h as usize) * (dst_w as usize) * 3;
    if src_h == dst_h && src_w == dst_w {
        dst_pool.clear();
        dst_pool.resize(needed + NEON_TAIL_PAD, 0u8);
        dst_pool[..needed].copy_from_slice(&src[..needed]);
        return Ok(());
    }
    let src_needed = (src_h as usize) * (src_w as usize) * 3;
    let src_ref = ImageRef::new(src_w, src_h, &src[..src_needed], PixelType::U8x3)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;

    dst_pool.clear();
    dst_pool.resize(needed + NEON_TAIL_PAD, 0u8);
    let mut dst_img =
        Image::from_slice_u8(dst_w, dst_h, &mut dst_pool[..needed], PixelType::U8x3)
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;

    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));
    RESIZER.with(|r| {
        r.borrow_mut()
            .resize(&src_ref, &mut dst_img, &opts)
            .map_err(|e| anyhow::anyhow!("fir resize: {:?}", e))
    })
}

// ---------- SIMD fused normalize + patch-write ----------
//
// Strategy on aarch64: use vld3q_u8 to load 16 RGB triples → three u8x16 vectors
// (R, G, B planar) in a single instruction. Process all 3 channels in parallel.
// Each iter handles 14 of the 16 lanes (patch_size=14) and ignores the last 2.
// This requires the source RGB buffer to have at least 2 bytes of padding after
// each row's last pixel — we ensure this by adding tail-padding when allocating.
//
// Scalar fallback path takes the original fused-loop shape (no gather buffer).

fn patchify_static_image(rgb: &[u8], h: u32, w: u32, cfg: &QwenCfg, out: &mut [f32]) {
    let p = cfg.patch_size as usize;
    let m = cfg.merge_size as usize;
    let tp = cfg.temporal_patch_size as usize;
    let c = 3usize;
    let h_us = h as usize;
    let w_us = w as usize;

    let grid_h = h_us / p;
    let grid_w = w_us / p;
    let grid_h_mblock = grid_h / m;
    let grid_w_mblock = grid_w / m;
    let merge_block = m * m;
    let patch_features = c * tp * p * p;

    let scale: [f32; 3] = [
        1.0 / (255.0 * cfg.std[0]),
        1.0 / (255.0 * cfg.std[1]),
        1.0 / (255.0 * cfg.std[2]),
    ];
    let bias: [f32; 3] = [
        -cfg.mean[0] / cfg.std[0],
        -cfg.mean[1] / cfg.std[1],
        -cfg.mean[2] / cfg.std[2],
    ];

    let pps = p * p;
    let stride_y = w_us * c;
    debug_assert!(tp <= 2);
    debug_assert_eq!(p, 14, "NEON fast path assumes patch_size=14");

    for gh_block in 0..grid_h_mblock {
        let block_y0 = gh_block * m * p;
        for gw_block in 0..grid_w_mblock {
            let block_x0 = gw_block * m * p;
            for mh in 0..m {
                let patch_y0 = block_y0 + mh * p;
                for mw in 0..m {
                    let patch_x0 = block_x0 + mw * p;
                    let patch_idx = gh_block * (grid_w_mblock * merge_block)
                        + gw_block * merge_block
                        + mh * m
                        + mw;
                    let out_off = patch_idx * patch_features;

                    // Output offsets for R, G, B (and their Tp=2 repeats).
                    let r_base = out_off;
                    let g_base = out_off + tp * pps;
                    let b_base = out_off + 2 * tp * pps;

                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        patch_row_neon_p14(
                            rgb, patch_y0, patch_x0, stride_y, scale, bias,
                            out, r_base, g_base, b_base, p,
                        );
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    patch_row_scalar(
                        rgb, patch_y0, patch_x0, stride_y, c, scale, bias,
                        out, r_base, g_base, b_base, p,
                    );

                    if tp == 2 {
                        for ch_base in [r_base, g_base, b_base] {
                            let (lo, hi) = out.split_at_mut(ch_base + pps);
                            hi[..pps].copy_from_slice(&lo[ch_base..ch_base + pps]);
                        }
                    }
                }
            }
        }
    }
}

/// NEON: process P=14 rows of patch with single-instruction RGB deinterleave.
/// SAFETY: caller must ensure `rgb` is large enough at all referenced offsets
/// (vld3q_u8 reads 16 RGB triples = 48 bytes, but we only use 14 lanes).
/// We require rgb.len() >= last_row_end + 48 - 14*3 = last_row_end + 6. The
/// caller arranges padding via Vec::resize beyond stride*h.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn patch_row_neon_p14(
    rgb: &[u8],
    patch_y0: usize,
    patch_x0: usize,
    stride_y: usize,
    scale: [f32; 3],
    bias: [f32; 3],
    out: &mut [f32],
    r_base: usize,
    g_base: usize,
    b_base: usize,
    p: usize,
) {
    use std::arch::aarch64::*;
    let s_r = vdupq_n_f32(scale[0]);
    let s_g = vdupq_n_f32(scale[1]);
    let s_b = vdupq_n_f32(scale[2]);
    let b_r = vdupq_n_f32(bias[0]);
    let b_g = vdupq_n_f32(bias[1]);
    let b_b = vdupq_n_f32(bias[2]);

    let in_ptr = rgb.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for py in 0..p {
        let src_off = (patch_y0 + py) * stride_y + patch_x0 * 3;
        // Load 16 RGB triples (48 bytes). We use 14.
        let rgb_v = vld3q_u8(in_ptr.add(src_off));
        let r16 = rgb_v.0;
        let g16 = rgb_v.1;
        let b16 = rgb_v.2;

        // Macro-style: u8x16 -> two f32x4 batches (low 4 + next 4 = 8 lanes).
        // We need 14 lanes, but vfmaq writes 4 at a time. Do 4+4+4+2.
        let row_r_off = r_base + py * p;
        let row_g_off = g_base + py * p;
        let row_b_off = b_base + py * p;

        // helper to expand u8x16 lanes [a..a+4) to f32x4
        let expand = |v: uint8x16_t, lo: bool| -> (float32x4_t, float32x4_t) {
            let u16x8_v = if lo { vmovl_u8(vget_low_u8(v)) } else { vmovl_high_u8(v) };
            let lo4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16x8_v)));
            let hi4 = vcvtq_f32_u32(vmovl_high_u16(u16x8_v));
            (lo4, hi4)
        };

        let (r_a, r_b4) = expand(r16, true);  // lanes 0..4, 4..8
        let (g_a, g_b4) = expand(g16, true);
        let (b_a, b_b4) = expand(b16, true);
        let (r_c, r_d) = expand(r16, false);  // lanes 8..12, 12..16
        let (g_c, g_d) = expand(g16, false);
        let (b_c, b_d) = expand(b16, false);

        // FMA
        let ra = vfmaq_f32(b_r, r_a, s_r);
        let rb = vfmaq_f32(b_r, r_b4, s_r);
        let rc = vfmaq_f32(b_r, r_c, s_r);
        let rd = vfmaq_f32(b_r, r_d, s_r);
        let ga = vfmaq_f32(b_g, g_a, s_g);
        let gb = vfmaq_f32(b_g, g_b4, s_g);
        let gc = vfmaq_f32(b_g, g_c, s_g);
        let gd = vfmaq_f32(b_g, g_d, s_g);
        let ba = vfmaq_f32(b_b, b_a, s_b);
        let bb = vfmaq_f32(b_b, b_b4, s_b);
        let bc = vfmaq_f32(b_b, b_c, s_b);
        let bd = vfmaq_f32(b_b, b_d, s_b);

        // store lanes 0..4, 4..8, 8..12 directly (12 lanes), then 2 scalar
        vst1q_f32(out_ptr.add(row_r_off), ra);
        vst1q_f32(out_ptr.add(row_r_off + 4), rb);
        vst1q_f32(out_ptr.add(row_r_off + 8), rc);
        vst1q_f32(out_ptr.add(row_g_off), ga);
        vst1q_f32(out_ptr.add(row_g_off + 4), gb);
        vst1q_f32(out_ptr.add(row_g_off + 8), gc);
        vst1q_f32(out_ptr.add(row_b_off), ba);
        vst1q_f32(out_ptr.add(row_b_off + 4), bb);
        vst1q_f32(out_ptr.add(row_b_off + 8), bc);

        // last 2 of P=14 from lanes 12, 13 of rd/gd/bd.
        // Use lane-extracts.
        *out_ptr.add(row_r_off + 12) = vgetq_lane_f32(rd, 0);
        *out_ptr.add(row_r_off + 13) = vgetq_lane_f32(rd, 1);
        *out_ptr.add(row_g_off + 12) = vgetq_lane_f32(gd, 0);
        *out_ptr.add(row_g_off + 13) = vgetq_lane_f32(gd, 1);
        *out_ptr.add(row_b_off + 12) = vgetq_lane_f32(bd, 0);
        *out_ptr.add(row_b_off + 13) = vgetq_lane_f32(bd, 1);
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn patch_row_scalar(
    rgb: &[u8],
    patch_y0: usize,
    patch_x0: usize,
    stride_y: usize,
    c: usize,
    scale: [f32; 3],
    bias: [f32; 3],
    out: &mut [f32],
    r_base: usize,
    g_base: usize,
    b_base: usize,
    p: usize,
) {
    for py in 0..p {
        let src_off = (patch_y0 + py) * stride_y + patch_x0 * c;
        let r_row = r_base + py * p;
        let g_row = g_base + py * p;
        let b_row = b_base + py * p;
        for px in 0..p {
            let r = rgb[src_off + px * c] as f32 * scale[0] + bias[0];
            let g = rgb[src_off + px * c + 1] as f32 * scale[1] + bias[1];
            let b = rgb[src_off + px * c + 2] as f32 * scale[2] + bias[2];
            out[r_row + px] = r;
            out[g_row + px] = g;
            out[b_row + px] = b;
        }
    }
}

// ---------- fused resize+normalize+patch (single pass) ----------
//
// Reads u8 RGB source directly, computes bilinear in f32, applies (scale, bias),
// writes f32 patch tensor in final [num_patches, C*Tp*P*P] layout. No intermediate
// u8 resized buffer.
//
// Each invocation produces patches for output row band [yb_start*m*p, yb_end*m*p).
// Caller (rayon) can run multiple invocations concurrently on disjoint output bands;
// they share the read-only `src` and write to non-overlapping slices of `out`.
//
// Bilinear convention: PIL/torchvision default ALIGN_CORNERS=False, no antialias:
//   src_y_f(y_out) = (y_out + 0.5) * src_h / dst_h - 0.5
//   src_y0 = floor(src_y_f).clamp(0, src_h - 1)
//   src_y1 = (src_y0 + 1).clamp(.., src_h - 1)
//   dy = (src_y_f - src_y0).clamp(0, 1)
// Same for x. Out = sum(w_ij * p_ij) where w_00 = (1-dy)(1-dx) etc.

fn fused_resize_normalize_patch_band(
    src: &[u8],
    src_h: u32,
    src_w: u32,
    target: crate::smart_resize::Target,
    cfg: &QwenCfg,
    yb_start_mblock: usize,
    yb_end_mblock: usize,
    out: &mut [f32],
) {
    let p = cfg.patch_size as usize;
    let m = cfg.merge_size as usize;
    let tp = cfg.temporal_patch_size as usize;
    let pps = p * p;
    let patch_features = 3 * tp * pps;

    let dst_h = target.h as usize;
    let dst_w = target.w as usize;
    let grid_w_mblock = dst_w / (m * p);
    let merge_block = m * m;

    let sh = src_h as f32;
    let sw = src_w as f32;
    let dh = dst_h as f32;
    let dw = dst_w as f32;
    let scale_y = sh / dh;
    let scale_x = sw / dw;

    let s = [
        1.0f32 / (255.0 * cfg.std[0]),
        1.0f32 / (255.0 * cfg.std[1]),
        1.0f32 / (255.0 * cfg.std[2]),
    ];
    let b = [
        -cfg.mean[0] / cfg.std[0],
        -cfg.mean[1] / cfg.std[1],
        -cfg.mean[2] / cfg.std[2],
    ];

    let src_stride = src_w as usize * 3;
    let src_h_us = src_h as usize;
    let src_w_us = src_w as usize;

    // Precompute per-column src indices and weights (shared across all rows / patches).
    // dst_w is small (typically <= 1148) so this scratch is cheap.
    let mut col_idx0 = vec![0usize; dst_w];
    let mut col_idx1 = vec![0usize; dst_w];
    let mut col_w0 = vec![0f32; dst_w];
    let mut col_w1 = vec![0f32; dst_w];
    for x_out in 0..dst_w {
        let fx = (x_out as f32 + 0.5) * scale_x - 0.5;
        let fx_c = fx.clamp(0.0, sw - 1.0);
        let x0 = fx_c.floor() as usize;
        let x1 = (x0 + 1).min(src_w_us - 1);
        let dx = (fx_c - x0 as f32).clamp(0.0, 1.0);
        col_idx0[x_out] = x0;
        col_idx1[x_out] = x1;
        col_w0[x_out] = 1.0 - dx;
        col_w1[x_out] = dx;
    }

    debug_assert!(tp <= 2);

    for gh_block in yb_start_mblock..yb_end_mblock {
        let block_y0 = gh_block * m * p;
        for mh in 0..m {
            for py in 0..p {
                let y_out = block_y0 + mh * p + py;
                let fy = (y_out as f32 + 0.5) * scale_y - 0.5;
                let fy_c = fy.clamp(0.0, sh - 1.0);
                let y0 = fy_c.floor() as usize;
                let y1 = (y0 + 1).min(src_h_us - 1);
                let dy = (fy_c - y0 as f32).clamp(0.0, 1.0);
                let w_y0 = 1.0 - dy;
                let w_y1 = dy;
                let row0 = y0 * src_stride;
                let row1 = y1 * src_stride;

                // Iterate every output column; group by (gw_block, mw, px) for the
                // patch-slot mapping.
                for gw_block in 0..grid_w_mblock {
                    for mw in 0..m {
                        let patch_idx = gh_block * (grid_w_mblock * merge_block)
                            + gw_block * merge_block
                            + mh * m
                            + mw;
                        let out_off = patch_idx * patch_features;
                        let r_row_off = out_off + py * p;
                        let g_row_off = out_off + tp * pps + py * p;
                        let b_row_off = out_off + 2 * tp * pps + py * p;

                        let patch_x0 = (gw_block * m + mw) * p;
                        for px in 0..p {
                            let x_out = patch_x0 + px;
                            let x0 = col_idx0[x_out];
                            let x1 = col_idx1[x_out];
                            let wx0 = col_w0[x_out];
                            let wx1 = col_w1[x_out];

                            // Combined weights for the 4 source pixels.
                            let w00 = w_y0 * wx0;
                            let w01 = w_y0 * wx1;
                            let w10 = w_y1 * wx0;
                            let w11 = w_y1 * wx1;

                            let p00r = src[row0 + x0 * 3] as f32;
                            let p01r = src[row0 + x1 * 3] as f32;
                            let p10r = src[row1 + x0 * 3] as f32;
                            let p11r = src[row1 + x1 * 3] as f32;
                            let p00g = src[row0 + x0 * 3 + 1] as f32;
                            let p01g = src[row0 + x1 * 3 + 1] as f32;
                            let p10g = src[row1 + x0 * 3 + 1] as f32;
                            let p11g = src[row1 + x1 * 3 + 1] as f32;
                            let p00b = src[row0 + x0 * 3 + 2] as f32;
                            let p01b = src[row0 + x1 * 3 + 2] as f32;
                            let p10b = src[row1 + x0 * 3 + 2] as f32;
                            let p11b = src[row1 + x1 * 3 + 2] as f32;

                            let r = (p00r * w00 + p01r * w01 + p10r * w10 + p11r * w11) * s[0] + b[0];
                            let g = (p00g * w00 + p01g * w01 + p10g * w10 + p11g * w11) * s[1] + b[1];
                            let bb = (p00b * w00 + p01b * w01 + p10b * w10 + p11b * w11) * s[2] + b[2];

                            out[r_row_off + px] = r;
                            out[g_row_off + px] = g;
                            out[b_row_off + px] = bb;
                        }
                        if tp == 2 {
                            for ch_off in [out_off, out_off + tp * pps, out_off + 2 * tp * pps] {
                                let dst0 = ch_off + py * p;
                                let dst1 = ch_off + py * p + pps;
                                let (lo, hi) = out.split_at_mut(dst1);
                                hi[..p].copy_from_slice(&lo[dst0..dst0 + p]);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Single-image fused pipeline entry point. Uses rayon for sub-image
/// parallelism over the merge-block-row dimension.
pub fn preprocess_image_fused_into(
    bytes: &[u8],
    cfg: &QwenCfg,
    out: &mut Vec<f32>,
) -> Result<(PreprocessOut, PreprocessTimings)> {
    use rayon::prelude::*;
    let t0 = std::time::Instant::now();

    let (decoded_h, decoded_w, target) = if is_jpeg(bytes) {
        let header = turbojpeg::read_header(bytes).context("read_header")?;
        let target = smart_resize(header.height as u32, header.width as u32, cfg.resize)?;
        // RST-aware parallel decode if the JPEG has restart markers; else fall back
        // to single-thread libjpeg-turbo. Both paths produce bit-identical pixels.
        let use_parallel = has_rst_markers(bytes);
        let n_strips = rayon::current_num_threads().min(4).max(2);
        let (dh, dw) = RGB_POOL.with(|p| {
            let mut pool = p.borrow_mut();
            if use_parallel {
                decode_jpeg_parallel_rst(bytes, target.h, target.w, &mut pool, n_strips)
            } else {
                decode_jpeg_scaled(bytes, target.h, target.w, &mut pool)
            }
        })?;
        (dh, dw, target)
    } else if is_png(bytes) {
        let (h, w) = RGB_POOL.with(|p| {
            let mut pool = p.borrow_mut();
            decode_png(bytes, &mut pool)
        })?;
        let target = smart_resize(h, w, cfg.resize)?;
        (h, w, target)
    } else {
        anyhow::bail!("unsupported image format")
    };

    let t_decode = t0.elapsed().as_nanos() as u64;

    let num_patches = target.num_patches(cfg.patch_size);
    let patch_features = 3 * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    let total = (num_patches * patch_features) as usize;
    out.clear();
    out.resize(total, 0.0f32);

    let t1 = std::time::Instant::now();
    let p = cfg.patch_size as usize;
    let m = cfg.merge_size as usize;
    let grid_h_mblock = (target.h as usize) / (m * p);

    // Split merge-block-rows into chunks for rayon. Aim ~2× chunks per worker so
    // smaller images don't get pinned to one worker.
    let workers = rayon::current_num_threads().max(1);
    let chunks = (workers * 2).min(grid_h_mblock).max(1);
    let chunk_size = grid_h_mblock.div_ceil(chunks);

    // Pass raw pointer as usize to dodge Send/Sync issues; we re-cast inside each spawn.
    // SAFETY: every chunk writes only the patches in its band; bands are disjoint by
    // construction (gh_block ranges are non-overlapping).
    let out_addr = out.as_mut_ptr() as usize;

    // Pull the decoded RGB out of the thread-local for the duration of the parallel
    // section (RefCell::borrow() can't escape its closure). We put it back after.
    let rgb_buf: Vec<u8> = RGB_POOL.with(|p| std::mem::take(&mut *p.borrow_mut()));
    let src_slice: &[u8] = &rgb_buf;

    rayon::scope(|s| {
        for chunk_idx in 0..chunks {
            let cfg_copy = *cfg;
            let target_copy = target;
            s.spawn(move |_| {
                let yb_start = chunk_idx * chunk_size;
                let yb_end = ((chunk_idx + 1) * chunk_size).min(grid_h_mblock);
                if yb_end <= yb_start {
                    return;
                }
                let out_band: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(out_addr as *mut f32, total)
                };
                fused_resize_normalize_patch_band(
                    src_slice,
                    decoded_h,
                    decoded_w,
                    target_copy,
                    &cfg_copy,
                    yb_start,
                    yb_end,
                    out_band,
                );
            });
        }
    });

    // Return buffer to the thread-local pool for reuse.
    RGB_POOL.with(|p| *p.borrow_mut() = rgb_buf);
    let t_fused = t1.elapsed().as_nanos() as u64;

    let grid_thw = [1, target.grid_h(cfg.patch_size), target.grid_w(cfg.patch_size)];
    let info = PreprocessOut {
        num_patches,
        patch_features,
        grid_thw,
    };
    let timings = PreprocessTimings {
        decode_ns: t_decode,
        resize_ns: 0,
        normpack_ns: t_fused,
        total_ns: t_decode + t_fused,
        decoded_h,
        decoded_w,
        target_h: target.h,
        target_w: target.w,
        num_patches,
    };
    Ok((info, timings))
}


// ---------- top-level: pool-aware entry points ----------

/// Preprocess into a caller-owned `out` Vec<f32> (which is grown / resized as needed).
/// This is the pool-friendly variant: caller can stash `out` in a thread-local /
/// CUDA-IPC buffer pool and reuse across requests.
pub fn preprocess_image_into(
    bytes: &[u8],
    cfg: &QwenCfg,
    out: &mut Vec<f32>,
) -> Result<(PreprocessOut, PreprocessTimings)> {
    let t0 = std::time::Instant::now();

    let (decoded_h, decoded_w, target) = if is_jpeg(bytes) {
        // Need full dims for smart_resize; turbojpeg.read_header avoids decoding.
        // We do header + decode in one call below by passing target to decode_jpeg_scaled,
        // but it itself calls read_header internally.
        // To keep smart_resize before turbojpeg's decode, re-read header here.
        let header = turbojpeg::read_header(bytes).context("read_header")?;
        let target = smart_resize(header.height as u32, header.width as u32, cfg.resize)?;
        let (dh, dw) = RGB_POOL.with(|p| {
            let mut pool = p.borrow_mut();
            decode_jpeg_scaled(bytes, target.h, target.w, &mut pool)
        })?;
        (dh, dw, target)
    } else if is_png(bytes) {
        let (h, w) = RGB_POOL.with(|p| {
            let mut pool = p.borrow_mut();
            decode_png(bytes, &mut pool)
        })?;
        let target = smart_resize(h, w, cfg.resize)?;
        (h, w, target)
    } else {
        anyhow::bail!("unsupported image format (JPEG/PNG only in bench)")
    };

    let t_decode = t0.elapsed().as_nanos() as u64;

    // Resize from RGB_POOL → RESIZED_POOL (zero-copy borrow on the src side).
    let t1 = std::time::Instant::now();
    let resize_res: Result<()> = RGB_POOL.with(|src_p| {
        RESIZED_POOL.with(|dst_p| {
            let src_pool = src_p.borrow();
            let mut dst_pool = dst_p.borrow_mut();
            resize_rgb_zero_copy(
                &src_pool,
                decoded_h,
                decoded_w,
                target.h,
                target.w,
                &mut dst_pool,
            )
        })
    });
    resize_res?;
    let t_resize = t1.elapsed().as_nanos() as u64;

    // Patchify into caller's output buffer.
    let t2 = std::time::Instant::now();
    let num_patches = target.num_patches(cfg.patch_size);
    let patch_features = 3 * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    let total = (num_patches * patch_features) as usize;
    out.clear();
    out.resize(total, 0.0f32);

    RESIZED_POOL.with(|p| {
        let pool = p.borrow();
        patchify_static_image(&pool, target.h, target.w, cfg, out);
    });
    let t_normpack = t2.elapsed().as_nanos() as u64;

    let grid_thw = [1, target.grid_h(cfg.patch_size), target.grid_w(cfg.patch_size)];
    let info = PreprocessOut {
        num_patches,
        patch_features,
        grid_thw,
    };
    let timings = PreprocessTimings {
        decode_ns: t_decode,
        resize_ns: t_resize,
        normpack_ns: t_normpack,
        total_ns: t_decode + t_resize + t_normpack,
        decoded_h,
        decoded_w,
        target_h: target.h,
        target_w: target.w,
        num_patches,
    };
    Ok((info, timings))
}

/// Allocating wrapper around `preprocess_image_into`.
pub fn preprocess_image(
    bytes: &[u8],
    cfg: &QwenCfg,
) -> Result<(Vec<f32>, PreprocessOut, PreprocessTimings)> {
    let mut out = Vec::new();
    let (info, t) = preprocess_image_into(bytes, cfg, &mut out)?;
    Ok((out, info, t))
}

/// Parallel batch via rayon. Each worker thread reuses its own pool.
pub fn preprocess_batch(
    images: &[Vec<u8>],
    cfg: &QwenCfg,
) -> Vec<Result<(Vec<f32>, PreprocessOut, PreprocessTimings)>> {
    use rayon::prelude::*;
    images
        .par_iter()
        .map(|b| preprocess_image(b, cfg))
        .collect()
}
