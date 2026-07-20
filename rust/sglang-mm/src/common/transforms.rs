//! Reusable image transform primitives.
//!
//! Model-specific processors compose these to build their preprocessing
//! pipelines. All functions operate on flat RGB byte arrays (HWC layout).

/// Normalize u8 RGB pixels to f32 in a single pass: `(pixel/255 - mean) / std`.
///
/// Writes into `out` which must have length `h * w * 3`.
pub fn normalize_rgb_f32(
    rgb: &[u8],
    h: usize,
    w: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
    out: &mut [f32],
) {
    debug_assert_eq!(rgb.len(), h * w * 3);
    debug_assert_eq!(out.len(), h * w * 3);
    let inv255 = 1.0f32 / 255.0;
    for i in 0..h * w {
        for c in 0..3 {
            let raw = rgb[i * 3 + c] as f32 * inv255;
            out[i * 3 + c] = (raw - mean[c]) / std[c];
        }
    }
}

/// Pad an HWC image to a grid-aligned size, filling padded pixels with `pad_value`.
///
/// Returns the padded buffer and the new (height, width).
pub fn pad_to_grid(
    rgb_f32: &[f32],
    h: usize,
    w: usize,
    channels: usize,
    grid_h: usize,
    grid_w: usize,
    pad_value: &[f32],
) -> (Vec<f32>, usize, usize) {
    let new_h = ((h + grid_h - 1) / grid_h) * grid_h;
    let new_w = ((w + grid_w - 1) / grid_w) * grid_w;
    let mut out = vec![0.0f32; new_h * new_w * channels];
    // Fill with pad value
    for i in 0..new_h * new_w {
        for c in 0..channels {
            out[i * channels + c] = pad_value[c];
        }
    }
    // Copy original data
    for y in 0..h {
        let src_start = y * w * channels;
        let dst_start = y * new_w * channels;
        out[dst_start..dst_start + w * channels]
            .copy_from_slice(&rgb_f32[src_start..src_start + w * channels]);
    }
    (out, new_h, new_w)
}

/// Reshape a padded HWC image into patches of shape `[num_patches, ph, pw, C]`.
///
/// `h` and `w` must be divisible by `ph` and `pw` respectively.
pub fn extract_patches_hwc(
    data: &[f32],
    h: usize,
    w: usize,
    channels: usize,
    ph: usize,
    pw: usize,
) -> Vec<f32> {
    let nph = h / ph;
    let npw = w / pw;
    let patch_size = ph * pw * channels;
    let mut out = vec![0.0f32; nph * npw * patch_size];
    for i in 0..nph {
        for j in 0..npw {
            let patch_idx = i * npw + j;
            for y in 0..ph {
                let src_y = i * ph + y;
                let src_start = (src_y * w + j * pw) * channels;
                let dst_start = patch_idx * patch_size + y * pw * channels;
                out[dst_start..dst_start + pw * channels]
                    .copy_from_slice(&data[src_start..src_start + pw * channels]);
            }
        }
    }
    out
}

/// Compute the patch grid dimensions for a given image size and patch size.
#[inline]
pub fn patch_grid(h: usize, w: usize, patch_h: usize, patch_w: usize) -> (usize, usize) {
    ((h + patch_h - 1) / patch_h, (w + patch_w - 1) / patch_w)
}
