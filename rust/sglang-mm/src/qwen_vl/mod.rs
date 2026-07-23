//! Qwen VL family (Qwen2-VL / 2.5-VL / 3-VL / 3.5) native image processor.
//!
//! Pure-Rust equivalent of the HF `Qwen2VLImageProcessor` pipeline the Python
//! `QwenVLImageProcessor` drives: `smart_resize` → bicubic resize → rescale +
//! normalize → patchify into `[grid_h*grid_w, C*tps*ps*ps]` (HF flatten order:
//! patches by `(gh/m, gw/m, m, m)`, features by `(C, tps, ps, ps)`, temporal
//! copies duplicated for stills) — plus the image-only M-RoPE fast path.
//! All parameters come from the runtime spec; nothing is hardcoded per model.

use rayon::prelude::*;

use crate::common::{self, resize};
use crate::registry::{MropeItem, ProcessedImage, VisionProcessor};

const MAX_RATIO: f64 = 200.0;

/// Resolved processor params, deserialized from the Python-side spec JSON
/// (unknown fields like `family` / token ids are ignored here).
#[derive(Clone, Debug, serde::Deserialize)]
pub struct QwenVlSpec {
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

pub struct QwenVlProcessor {
    spec: QwenVlSpec,
    /// Per-channel u8 → normalized-f32 lookup: `(v/255 - mean) / std`.
    lut: [[f32; 256]; 3],
}

impl QwenVlProcessor {
    pub fn new(spec: QwenVlSpec) -> Result<Self, String> {
        if spec.patch_size == 0 || spec.merge_size == 0 || spec.temporal_patch_size == 0 {
            return Err("qwen_vl spec: sizes must be positive".into());
        }
        let lut = core::array::from_fn(|c| {
            core::array::from_fn(|v| (v as f32 / 255.0 - spec.image_mean[c]) / spec.image_std[c])
        });
        Ok(Self { spec, lut })
    }

    pub fn from_spec_json(json: &str) -> Result<Self, String> {
        let spec: QwenVlSpec =
            serde_json::from_str(json).map_err(|e| format!("qwen_vl spec: {e}"))?;
        Self::new(spec)
    }

    fn factor(&self) -> usize {
        self.spec.patch_size * self.spec.merge_size
    }

    /// HF flatten: patches ordered `(gh/m, gw/m, m, m)`, features `(C, tps,
    /// ps, ps)`; parallel over merged-block rows.
    fn patchify(&self, rgb: &[u8], h: usize, w: usize) -> Vec<f32> {
        let (ps, m, tps) = (
            self.spec.patch_size,
            self.spec.merge_size,
            self.spec.temporal_patch_size,
        );
        let (gh, gw) = (h / ps, w / ps);
        let dim = 3 * tps * ps * ps;
        let block_row = gw * m * dim; // one merged-block row of patches
        let mut out = vec![0.0f32; gh * gw * dim];

        common::pool().install(|| {
            out.par_chunks_mut(block_row)
                .enumerate()
                .for_each(|(i, chunk)| {
                    let mut p = 0;
                    for j in 0..gw / m {
                        for mh in 0..m {
                            for mw in 0..m {
                                let y0 = (i * m + mh) * ps;
                                let x0 = (j * m + mw) * ps;
                                let patch = &mut chunk[p * dim..(p + 1) * dim];
                                for c in 0..3 {
                                    let ch = &mut patch[c * tps * ps * ps..];
                                    for py in 0..ps {
                                        let src = ((y0 + py) * w + x0) * 3 + c;
                                        for px in 0..ps {
                                            ch[py * ps + px] =
                                                self.lut[c][rgb[src + px * 3] as usize];
                                        }
                                    }
                                    // Temporal copies of a still are duplicates.
                                    let (t0, rest) = ch.split_at_mut(ps * ps);
                                    for t in 0..tps - 1 {
                                        rest[t * ps * ps..(t + 1) * ps * ps].copy_from_slice(t0);
                                    }
                                }
                                p += 1;
                            }
                        }
                    }
                });
        });
        out
    }
}

impl VisionProcessor for QwenVlProcessor {
    fn process_image(&self, rgb: &[u8], h: usize, w: usize) -> Result<ProcessedImage, String> {
        let (th, tw) = smart_resize(
            h,
            w,
            self.factor(),
            self.spec.min_pixels,
            self.spec.max_pixels,
        )?;
        let resized;
        let data = if (th, tw) != (h, w) {
            resized = common::pool()
                .install(|| resize::resize_rgb_filter(rgb, h, w, th, tw, resize::Filter::Bicubic));
            &resized
        } else {
            rgb
        };
        let (gh, gw) = (th / self.spec.patch_size, tw / self.spec.patch_size);
        Ok(ProcessedImage {
            pixel_values: self.patchify(data, th, tw),
            grid_thw: [1, gh as u32, gw as u32],
        })
    }

    fn tokens_per_image(&self, grid: &[u32; 3]) -> usize {
        (grid[0] as usize * grid[1] as usize * grid[2] as usize)
            / (self.spec.merge_size * self.spec.merge_size)
    }

    fn feature_dim(&self) -> usize {
        3 * self.spec.temporal_patch_size * self.spec.patch_size * self.spec.patch_size
    }

    fn mrope_image_only(
        &self,
        input_len: usize,
        items: &[MropeItem],
    ) -> Result<(Vec<i64>, i64), String> {
        mrope_image_only(input_len, items, self.spec.merge_size)
    }
}

/// Python-`round()` (round-half-to-even), which `round_by_factor` relies on.
fn round_half_even(x: f64) -> f64 {
    if (x - x.trunc()).abs() == 0.5 {
        (x / 2.0).round() * 2.0
    } else {
        x.round()
    }
}

/// The Qwen `smart_resize`: dims divisible by `factor`, total pixels within
/// `[min_pixels, max_pixels]`, aspect ratio preserved as closely as possible.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), String> {
    let (h, w) = (height as f64, width as f64);
    if height == 0 || width == 0 {
        return Err("empty image".into());
    }
    let ratio = h.max(w) / h.min(w);
    if ratio > MAX_RATIO {
        return Err(format!(
            "absolute aspect ratio must be smaller than {MAX_RATIO}, got {ratio}"
        ));
    }
    let f = factor as f64;
    let mut h_bar = ((round_half_even(h / f) * f) as usize).max(factor);
    let mut w_bar = ((round_half_even(w / f) * f) as usize).max(factor);
    if h_bar * w_bar > max_pixels {
        let beta = (h * w / max_pixels as f64).sqrt();
        h_bar = ((h / beta / f).floor() * f) as usize;
        w_bar = ((w / beta / f).floor() * f) as usize;
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = ((h * beta / f).ceil() * f) as usize;
        w_bar = ((w * beta / f).ceil() * f) as usize;
    }
    Ok((h_bar, w_bar))
}

/// Image-only M-RoPE fast path (the image branch of
/// `MRotaryEmbedding.get_rope_index`, identical across Qwen generations):
/// text runs sequentially on all three rows; each image spans `(t, h/m, w/m)`
/// index grids; positions advance by `max(t, h/m, w/m)` past an image.
/// Returns flattened row-major `[3, input_len]` positions and the delta
/// (`max + 1 - input_len`). `items` must be in prompt order.
pub fn mrope_image_only(
    input_len: usize,
    items: &[MropeItem],
    merge_size: usize,
) -> Result<(Vec<i64>, i64), String> {
    let len = input_len;
    let mut pos = vec![0i64; 3 * len];
    let fill_text = |st: usize, n: usize, base: i64, pos: &mut [i64]| {
        for k in 0..n {
            let v = base + k as i64;
            pos[st + k] = v;
            pos[len + st + k] = v;
            pos[2 * len + st + k] = v;
        }
    };
    let mut st = 0usize;
    let mut next_pos = 0i64;
    for item in items {
        let (start, end) = (item.start as usize, item.end as usize);
        if start < st || end >= len {
            return Err(format!(
                "mrope: item range ({start},{end}) out of order/bounds"
            ));
        }
        fill_text(st, start - st, next_pos, &mut pos);
        next_pos += (start - st) as i64;

        let t = item.grid[0] as usize;
        let gh = item.grid[1] as usize / merge_size;
        let gw = item.grid[2] as usize / merge_size;
        if t * gh * gw != end - start + 1 {
            return Err("mrope: token span does not match grid".into());
        }
        for ti in 0..t {
            for hi in 0..gh {
                for wi in 0..gw {
                    let idx = start + (ti * gh + hi) * gw + wi;
                    pos[idx] = next_pos + ti as i64;
                    pos[len + idx] = next_pos + hi as i64;
                    pos[2 * len + idx] = next_pos + wi as i64;
                }
            }
        }
        next_pos += (t.max(gh).max(gw)) as i64;
        st = end + 1;
    }
    if st < len {
        fill_text(st, len - st, next_pos, &mut pos);
    }
    let max = pos.iter().copied().max().unwrap_or(-1);
    Ok((pos, max + 1 - len as i64))
}

// --- Python bindings (parity tests drive the exact server pipeline) ---

#[cfg(feature = "python")]
mod python {
    use numpy::{IntoPyArray, PyArray1};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use super::*;
    use crate::registry::VisionProcessor;

    /// Run the full native image path on encoded image bytes:
    /// decode → smart_resize → bicubic → normalize → patchify.
    /// Returns `(pixel_values flat f32, (t, h, w))`.
    #[pyfunction]
    fn preprocess<'py>(
        py: Python<'py>,
        data: Vec<u8>,
        spec_json: &str,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, (u32, u32, u32))> {
        let proc = QwenVlProcessor::from_spec_json(spec_json).map_err(PyValueError::new_err)?;
        let out = py
            .allow_threads(move || {
                let (rgb, h, w) = crate::common::decode_rgb(&data)?;
                proc.process_image(&rgb, h, w)
            })
            .map_err(PyValueError::new_err)?;
        let [t, h, w] = out.grid_thw;
        Ok((out.pixel_values.into_pyarray_bound(py), (t, h, w)))
    }

    #[pyfunction]
    fn smart_resize_py(
        height: usize,
        width: usize,
        factor: usize,
        min_pixels: usize,
        max_pixels: usize,
    ) -> PyResult<(usize, usize)> {
        smart_resize(height, width, factor, min_pixels, max_pixels).map_err(PyValueError::new_err)
    }

    /// `(positions flat [3*input_len], delta)` for image-only requests;
    /// `items` = [(start, end_inclusive, t, h, w), ...] in prompt order.
    #[pyfunction]
    fn mrope_image_only_py<'py>(
        py: Python<'py>,
        input_len: usize,
        items: Vec<(u32, u32, u32, u32, u32)>,
        merge_size: usize,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, i64)> {
        let items: Vec<MropeItem> = items
            .into_iter()
            .map(|(start, end, t, h, w)| MropeItem {
                start,
                end,
                grid: [t, h, w],
            })
            .collect();
        let (pos, delta) =
            mrope_image_only(input_len, &items, merge_size).map_err(PyValueError::new_err)?;
        Ok((pos.into_pyarray_bound(py), delta))
    }

    pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let m = PyModule::new_bound(parent.py(), "qwen_vl")?;
        m.add_function(wrap_pyfunction!(preprocess, &m)?)?;
        m.add_function(wrap_pyfunction!(smart_resize_py, &m)?)?;
        m.add_function(wrap_pyfunction!(mrope_image_only_py, &m)?)?;
        parent.add_submodule(&m)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> QwenVlSpec {
        QwenVlSpec {
            patch_size: 2,
            merge_size: 2,
            temporal_patch_size: 2,
            min_pixels: 4,
            max_pixels: 1 << 30,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
        }
    }

    #[test]
    fn smart_resize_matches_python_reference() {
        // Values from the Python `smart_resize` (qwen_vl.py) run offline.
        assert_eq!(
            smart_resize(1365, 2048, 28, 3136, 12845056).unwrap(),
            (1372, 2044)
        );
        assert_eq!(
            smart_resize(100, 100, 28, 3136, 12845056).unwrap(),
            (112, 112)
        );
        // Downscale branch: 4000x3000 exceeds 1280*28*28 → floor_by_factor.
        assert_eq!(
            smart_resize(3000, 4000, 28, 3136, 1003520).unwrap(),
            (840, 1148)
        );
        // Upscale branch: tiny image below min_pixels → ceil_by_factor.
        assert_eq!(smart_resize(20, 20, 28, 3136, 12845056).unwrap(), (56, 56));
        // Qwen3.5 factors (patch 16 * merge 2, min 65536, max 16777216).
        assert_eq!(
            smart_resize(1365, 2048, 32, 65536, 16777216).unwrap(),
            (1376, 2048)
        );
        // Banker's rounding tie: 48/32 = 1.5 rounds to 2 (even), not 1.
        assert_eq!(smart_resize(4000, 48, 32, 4, 1 << 30).unwrap(), (4000, 64));
        // Extreme aspect ratio rejected.
        assert!(smart_resize(10000, 10, 28, 3136, 12845056).is_err());
    }

    #[test]
    fn patchify_layout_matches_hf_order() {
        // 4x8 image, ps=2, m=2, tps=2 → gh=2, gw=4, dim=3*2*2*2=24.
        // Pixel value encodes its (y, x): v = y*16 + x*2 (fits u8).
        let (h, w) = (4usize, 8usize);
        let mut rgb = vec![0u8; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    rgb[(y * w + x) * 3 + c] = (y * 16 + x * 2 + c) as u8;
                }
            }
        }
        let proc = QwenVlProcessor::new(spec()).unwrap();
        let pv = proc.patchify(&rgb, h, w);
        let dim = proc.feature_dim();
        assert_eq!(pv.len(), 2 * 4 * dim);

        // Patch order (gh/m=1, gw/m=2, m, m): patch 0 = block(0,0) offset (0,0),
        // patch 1 = (0,0)+(0,1) → x0=2, patch 2 = (0,0)+(1,0) → y0=2,
        // patch 4 = block(0,1) → x0=4.
        let lut = |y: usize, x: usize, c: usize| ((y * 16 + x * 2 + c) as f32) / 255.0;
        // patch 1, channel 0, t=0, (py=0, px=0) → pixel (0, 2).
        assert_eq!(pv[dim + 0], lut(0, 2, 0));
        // patch 2, channel 0, t=0, (0,0) → pixel (2, 0).
        assert_eq!(pv[2 * dim], lut(2, 0, 0));
        // patch 4, channel 0 → pixel (0, 4).
        assert_eq!(pv[4 * dim], lut(0, 4, 0));
        // Temporal duplicate: t=1 block equals t=0 block.
        let ps2 = 4; // ps*ps
        assert_eq!(pv[dim + ps2], pv[dim]);
        // Channel 1 block of patch 0 → same pixel, c=1.
        assert_eq!(pv[2 * ps2 * 1 + 0], lut(0, 0, 1)); // c stride = tps*ps*ps = 8
    }

    #[test]
    fn mrope_image_only_matches_reference() {
        // 3 text tokens, image of grid [1, 4, 6] (m=2 → 2x3 = 6 tokens), 2 text.
        // input: [T T T I I I I I I T T], len 11.
        let items = [MropeItem {
            start: 3,
            end: 8,
            grid: [1, 4, 6],
        }];
        let (pos, delta) = mrope_image_only(11, &items, 2).unwrap();
        let len = 11;
        // Text prefix 0..3: all rows 0,1,2.
        for k in 0..3 {
            assert_eq!(
                (pos[k], pos[len + k], pos[2 * len + k]),
                (k as i64, k as i64, k as i64)
            );
        }
        // Image tokens: t=0, h in 0..2, w in 0..3, +3 offset.
        assert_eq!((pos[3], pos[len + 3], pos[2 * len + 3]), (3, 3, 3));
        assert_eq!((pos[4], pos[len + 4], pos[2 * len + 4]), (3, 3, 4));
        assert_eq!((pos[6], pos[len + 6], pos[2 * len + 6]), (3, 4, 3));
        // Text tail resumes at 3 + max(1,2,3) = 6.
        assert_eq!((pos[9], pos[len + 9], pos[2 * len + 9]), (6, 6, 6));
        assert_eq!((pos[10], pos[len + 10], pos[2 * len + 10]), (7, 7, 7));
        // delta = max + 1 - len = 7 + 1 - 11.
        assert_eq!(delta, -3);
    }
}
