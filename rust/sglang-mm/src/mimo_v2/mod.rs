//! MiMo-V2 family (`model_type="mimo_v2"`, e.g. MiMo-V2.5) server-pipeline
//! image processor.
//!
//! Pure-Rust equivalent of the image branch of the Python `MiMoProcessor`
//! (`sglang/srt/multimodal/processors/mimo_v2.py`): MiMo's `smart_resize`
//! variant → torch bilinear resize (`F.interpolate`, align_corners=False, no
//! antialias, f32 throughout) → `(v - mean) / std` in the 0..255 pixel scale
//! → the HF Qwen2-VL patchify order. Positions are 1-D RoPE carried in the
//! `[3, input_len]` wire shape the MiMo scheduler contract uses (all rows
//! equal, delta 0). All parameters come from the runtime spec; nothing is
//! hardcoded per model.
//!
//! Known divergence from Python: PIL composites an RGBA image onto a white
//! background (`MiMoProcessor.to_rgb`) while the shared Rust decode drops the
//! alpha channel; opaque images are unaffected.

use crate::common::{par, resize, round_half_even, token_layout};
use crate::pipeline::{
    DecodedMedia, Geometry, MmFamilyProcessor, PositionOutput, ProcessedItem, Tensor, TensorData,
    TokenLayout,
};

const MAX_RATIO: f64 = 200.0;

/// Resolved processor params, deserialized from the Python-side spec JSON
/// (unknown fields like `family` are ignored here). `image_mean`/`image_std`
/// are in the 0..255 pixel scale (MiMo normalizes unrescaled floats).
#[derive(Clone, Debug, serde::Deserialize)]
pub struct MimoV2Spec {
    pub image_token_id: i32,
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

pub struct MimoV2Processor {
    spec: MimoV2Spec,
}

impl MimoV2Processor {
    pub fn new(spec: MimoV2Spec) -> Result<Self, String> {
        if spec.patch_size == 0 || spec.merge_size == 0 || spec.temporal_patch_size == 0 {
            return Err("mimo_v2 spec: sizes must be positive".into());
        }
        if spec.image_std.contains(&0.0) {
            return Err("mimo_v2 spec: image_std must be nonzero".into());
        }
        Ok(Self { spec })
    }

    pub fn from_spec_json(json: &str) -> Result<Self, String> {
        let spec: MimoV2Spec =
            serde_json::from_str(json).map_err(|e| format!("mimo_v2 spec: {e}"))?;
        Self::new(spec)
    }

    fn factor(&self) -> usize {
        self.spec.patch_size * self.spec.merge_size
    }

    fn tokens_per_image(&self, grid: &[u32; 3]) -> usize {
        (grid[0] as usize * grid[1] as usize * grid[2] as usize)
            / (self.spec.merge_size * self.spec.merge_size)
    }

    /// HF Qwen2-VL flatten (patches `(gh/m, gw/m, m, m)`, features `(C, tps,
    /// ps, ps)`, temporal copies of a still duplicated) over an f32 HWC
    /// buffer, normalizing `(v - mean) / std` inline. Python standardizes the
    /// resized float tensor before its pure-permute flatten; folding the same
    /// f32 affine into the copy is bit-identical.
    fn patchify(&self, data: &[f32], h: usize, w: usize) -> Vec<f32> {
        let (ps, m, tps) = (
            self.spec.patch_size,
            self.spec.merge_size,
            self.spec.temporal_patch_size,
        );
        let (mean, std) = (self.spec.image_mean, self.spec.image_std);
        let (gh, gw) = (h / ps, w / ps);
        let dim = 3 * tps * ps * ps;
        let block_row = gw * m * dim; // one merged-block row of patches
        let mut out = vec![0.0f32; gh * gw * dim];

        par::for_chunks_mut(&mut out, block_row, |i, chunk| {
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
                                    ch[py * ps + px] = (data[src + px * 3] - mean[c]) / std[c];
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
        out
    }
}

impl MmFamilyProcessor for MimoV2Processor {
    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, String> {
        let DecodedMedia::Image { rgb, height, width } = media;
        let (h, w) = (*height, *width);
        let (th, tw) = smart_resize(
            h,
            w,
            self.factor(),
            self.spec.min_pixels,
            self.spec.max_pixels,
        )?;
        let data = resize::resize_rgb_f32_torch_bilinear(rgb, h, w, th, tw);
        let (gh, gw) = (th / self.spec.patch_size, tw / self.spec.patch_size);
        // `smart_resize` guarantees both: dims are positive and divisible by
        // `patch_size * merge_size`. `patchify` indexes on that (and the `dim`
        // division below needs a non-empty grid), so fail loudly rather than
        // panic if a future spec change breaks the guarantee.
        if gh == 0 || gw == 0 || gh % self.spec.merge_size != 0 || gw % self.spec.merge_size != 0 {
            return Err(format!(
                "mimo_v2: patch grid {gh}x{gw} is empty or not a multiple of \
                 merge_size {}",
                self.spec.merge_size
            ));
        }
        let pixel_values = self.patchify(&data, th, tw);
        let dim = pixel_values.len() / (gh * gw);
        Ok(ProcessedItem {
            feature: Tensor {
                shape: vec![gh * gw, dim],
                data: TensorData::F32(pixel_values),
            },
            aux: vec![(
                "image_grid_thw".to_string(),
                Tensor {
                    shape: vec![3],
                    data: TensorData::I64(vec![1, gh as i64, gw as i64]),
                },
            )],
            geometry: Geometry::Grid([1, gh as u32, gw as u32]),
        })
    }

    fn layout(&self, input_ids: &[i32], items: &[Geometry]) -> Result<TokenLayout, String> {
        // The tokenized prompt carries `<|vision_start|><|image_pad|>
        // <|vision_end|>` per image, so expanding the single `image_pad`
        // placeholder reproduces Python's `[start] + n*[pad] + [end]` layout.
        let counts = items
            .iter()
            .map(|Geometry::Grid(grid)| self.tokens_per_image(grid))
            .collect::<Vec<_>>();
        token_layout::layout_by_placeholder(input_ids, self.spec.image_token_id, &counts)
    }

    fn positions(
        &self,
        input_len: usize,
        _offsets: &[(u32, u32)],
        _items: &[Geometry],
    ) -> Result<PositionOutput, String> {
        // MiMo-V2 checkpoints without `rope_scaling.mrope_section` (the only
        // ones the Python launcher admits to this pipeline) are 1-D RoPE, but
        // the scheduler contract still carries `[3, input_len]` positions:
        // Python emits `arange(len).expand(3, -1)` with delta 0, and the
        // model reads the plain 1-D row. Same wire shape as M-RoPE.
        let row = 0..input_len as i64;
        let positions = row.clone().chain(row.clone()).chain(row).collect();
        Ok(PositionOutput::MRope {
            positions,
            delta: 0,
        })
    }
}

/// The MiMo `smart_resize` (`MiMoProcessor.smart_resize`). Differs from the
/// Qwen variant: a min-side shorter than `factor` is first upscaled to it —
/// and that branch *skips* the aspect-ratio guard (Python checks the ratio in
/// the `elif` only) — and `round_by_factor` has no `max(factor)` clamp (the
/// upscale branch already guarantees both sides reach it).
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), String> {
    if height == 0 || width == 0 {
        return Err("empty image".into());
    }
    let (mut h, mut w) = (height as f64, width as f64);
    if height.min(width) < factor {
        let scale = factor as f64 / height.min(width) as f64;
        h = round_half_even(h * scale);
        w = round_half_even(w * scale);
    } else if h.max(w) / h.min(w) > MAX_RATIO {
        return Err(format!(
            "absolute aspect ratio must be smaller than {MAX_RATIO}, got {}",
            h.max(w) / h.min(w)
        ));
    }
    let f = factor as f64;
    let mut h_bar = (round_half_even(h / f) * f) as usize;
    let mut w_bar = (round_half_even(w / f) * f) as usize;
    if h_bar * w_bar > max_pixels {
        let beta = (h * w / max_pixels as f64).sqrt();
        h_bar = ((h / beta / f).floor() * f) as usize;
        w_bar = ((w / beta / f).floor() * f) as usize;
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = ((h * beta / f).ceil() * f) as usize;
        w_bar = ((w * beta / f).ceil() * f) as usize;
    }
    // The downscale branch floors without a lower clamp (as Python does), so
    // a very thin image against a small `max_pixels` can floor a side to 0.
    // Python then fails inside `F.interpolate`; here it would reach the
    // patchify geometry math, so reject it as a request error.
    if h_bar == 0 || w_bar == 0 {
        return Err(format!(
            "smart_resize: {height}x{width} degenerates to {h_bar}x{w_bar} at \
             max_pixels={max_pixels}; image is too thin for this pixel budget"
        ));
    }
    Ok((h_bar, w_bar))
}

// --- Python bindings (parity tests drive the exact server pipeline) ---

#[cfg(feature = "python")]
mod python {
    use numpy::{IntoPyArray, PyArray1};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use super::*;
    use crate::pipeline::TensorData;

    /// `(pixel_values flat f32, (t, h, w))` for one preprocessed image.
    type PyProcessedImage<'py> = (Bound<'py, PyArray1<f32>>, (u32, u32, u32));

    /// Run the full native image path on encoded image bytes:
    /// decode → smart_resize → torch bilinear → normalize → patchify.
    #[pyfunction]
    fn preprocess<'py>(
        py: Python<'py>,
        data: Vec<u8>,
        spec_json: &str,
    ) -> PyResult<PyProcessedImage<'py>> {
        let proc = MimoV2Processor::from_spec_json(spec_json).map_err(PyValueError::new_err)?;
        let out = py
            .detach(move || {
                let (rgb, height, width) = crate::common::decode_rgb(&data)?;
                proc.process_item(&DecodedMedia::Image { rgb, height, width })
            })
            .map_err(PyValueError::new_err)?;
        let Geometry::Grid([t, h, w]) = out.geometry;
        let TensorData::F32(pixel_values) = out.feature.data else {
            return Err(PyValueError::new_err("mimo_v2: expected f32 feature"));
        };
        Ok((pixel_values.into_pyarray(py), (t, h, w)))
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

    pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let m = PyModule::new(parent.py(), "mimo_v2")?;
        m.add_function(wrap_pyfunction!(preprocess, &m)?)?;
        m.add_function(wrap_pyfunction!(smart_resize_py, &m)?)?;
        // The native pipeline driver is family-generic (the spec selects the
        // family) and mimo_v2 drains through the same shape as qwen_vl, so
        // re-register the one binding rather than fork it.
        m.add_function(wrap_pyfunction!(
            crate::qwen_vl::python::process_native_mm,
            &m
        )?)?;
        parent.add_submodule(&m)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> MimoV2Spec {
        MimoV2Spec {
            image_token_id: 1,
            patch_size: 2,
            merge_size: 2,
            temporal_patch_size: 2,
            min_pixels: 4,
            max_pixels: 1 << 30,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
        }
    }

    /// Values from the Python `MiMoProcessor.smart_resize` run offline with
    /// MiMo-V2.5's real factor 32 (patch 16 * merge 2) and pixel limits
    /// (8192, 8388608).
    #[test]
    fn smart_resize_matches_python_reference() {
        let mimo = |h, w| smart_resize(h, w, 32, 8192, 8388608).unwrap();
        assert_eq!(mimo(1365, 2048), (1376, 2048));
        // round_by_factor has no max(factor) clamp; 100 rounds *down* to 96.
        assert_eq!(mimo(100, 100), (96, 96));
        // Downscale branch: 3000x4000 exceeds max_pixels → floor_by_factor.
        assert_eq!(mimo(3000, 4000), (2496, 3328));
        // Min-side upscale to factor, then min_pixels upscale.
        assert_eq!(mimo(20, 20), (96, 96));
        // Branch order: a min side below factor upscales *without* the
        // aspect-ratio guard (ratio 300 here), unlike the Qwen variant.
        assert_eq!(mimo(1, 300), (32, 9600));
        // The guard does fire once the min side reaches factor.
        assert!(smart_resize(33, 33 * 201, 32, 8192, 8388608).is_err());
    }

    /// A thin image against a small `max_pixels` floors one side to 0
    /// (Python reference: (10, 2000) at factor 4, max 3136 → (0, 788), which
    /// Python only catches inside `F.interpolate`). The Rust pipeline must
    /// reject it as a request error, never panic in patchify geometry.
    #[test]
    fn degenerate_target_is_rejected_not_panicked() {
        assert!(smart_resize(10, 2000, 4, 4, 3136).is_err());

        let mut spec = spec();
        spec.min_pixels = 4;
        spec.max_pixels = 3136;
        let proc = MimoV2Processor::new(spec).unwrap();
        let err = proc
            .process_item(&DecodedMedia::Image {
                rgb: vec![0u8; 10 * 2000 * 3],
                height: 10,
                width: 2000,
            })
            .err()
            .expect("degenerate geometry must be an Err, never a panic");
        assert!(err.contains("smart_resize"), "unexpected error: {err}");
    }

    /// The server's message layer gates modalities on what a family declares.
    /// Python MiMo handles video/audio; the native pipeline does not — those
    /// requests must be rejected, so the family must not accidentally declare
    /// them.
    #[test]
    fn mimo_declares_images_only() {
        let caps = MimoV2Processor::new(spec()).unwrap().capabilities();
        assert!(!caps.video && !caps.audio);
    }

    /// Patchify must keep the HF flatten order over the f32 buffer and apply
    /// `(v - mean) / std` per channel (values stay in the 0..255 scale — MiMo
    /// never rescales by 1/255).
    #[test]
    fn patchify_layout_and_normalization_match_reference() {
        // 4x8 image, ps=2, m=2, tps=2 → gh=2, gw=4, dim=3*2*2*2=24.
        let (h, w) = (4usize, 8usize);
        let mut data = vec![0.0f32; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    data[(y * w + x) * 3 + c] = (y * 16 + x * 2 + c) as f32;
                }
            }
        }
        let mut spec = spec();
        spec.image_mean = [10.0, 20.0, 30.0];
        spec.image_std = [2.0, 4.0, 8.0];
        let proc = MimoV2Processor::new(spec).unwrap();
        let pv = proc.patchify(&data, h, w);
        let dim = 24; // 3 * tps * ps * ps
        assert_eq!(pv.len(), 2 * 4 * dim);

        let norm = |y: usize, x: usize, c: usize| {
            ((y * 16 + x * 2 + c) as f32 - [10.0, 20.0, 30.0][c]) / [2.0, 4.0, 8.0][c]
        };
        // Patch order (gh/m=1, gw/m=2, m, m): patch 1 = block(0,0)+(0,1) →
        // pixel (0, 2); patch 2 → (2, 0); patch 4 = block(0,1) → (0, 4).
        assert_eq!(pv[dim], norm(0, 2, 0));
        assert_eq!(pv[2 * dim], norm(2, 0, 0));
        assert_eq!(pv[4 * dim], norm(0, 4, 0));
        // Temporal duplicate: t=1 block equals t=0 block.
        let ps2 = 4; // ps*ps
        assert_eq!(pv[dim + ps2], pv[dim]);
        // Channel 1 block of patch 0 → same pixel, c=1 (c stride = tps*ps*ps).
        assert_eq!(pv[2 * ps2], norm(0, 0, 1));
    }

    /// The 1-D positions ride the M-RoPE wire shape: three identical arange
    /// rows, delta 0 — Python's `arange(len).expand(3, -1)` contract.
    #[test]
    fn positions_are_replicated_arange_with_zero_delta() {
        let proc = MimoV2Processor::new(spec()).unwrap();
        let PositionOutput::MRope { positions, delta } = proc
            .positions(5, &[(1, 2)], &[Geometry::Grid([1, 2, 4])])
            .unwrap()
        else {
            panic!("mimo carries [3, len] positions");
        };
        assert_eq!(delta, 0);
        let row: Vec<i64> = (0..5).collect();
        assert_eq!(positions, [row.clone(), row.clone(), row].concat());
    }
}
