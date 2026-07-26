//! Qwen VL family (Qwen2-VL / 2.5-VL / 3-VL / 3.5) server-pipeline image processor.
//!
//! Pure-Rust equivalent of the HF `Qwen2VLImageProcessor` pipeline the Python
//! `QwenVLImageProcessor` drives: `smart_resize` → bicubic resize → rescale +
//! normalize → patchify into `[grid_h*grid_w, C*tps*ps*ps]` (HF flatten order:
//! patches by `(gh/m, gw/m, m, m)`, features by `(C, tps, ps, ps)`, temporal
//! copies duplicated for stills) — plus the image-only M-RoPE fast path.
//! All parameters come from the runtime spec; nothing is hardcoded per model.

use rayon::prelude::*;

use crate::common::{self, resize, token_layout};
use crate::pipeline::{
    DecodedMedia, Geometry, MmFamilyProcessor, PositionOutput, ProcessedItem, Tensor, TensorData,
    TokenLayout,
};

const MAX_RATIO: f64 = 200.0;

/// One media item's placement for M-RoPE: inclusive token range + patch grid.
pub struct MropeItem {
    pub start: u32,
    pub end: u32,
    pub grid: [u32; 3],
}

/// Resolved processor params, deserialized from the Python-side spec JSON
/// (unknown fields like `family` are ignored here).
#[derive(Clone, Debug, serde::Deserialize)]
pub struct QwenVlSpec {
    pub image_token_id: i32,
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

impl QwenVlProcessor {
    fn tokens_per_image(&self, grid: &[u32; 3]) -> usize {
        (grid[0] as usize * grid[1] as usize * grid[2] as usize)
            / (self.spec.merge_size * self.spec.merge_size)
    }
}

impl MmFamilyProcessor for QwenVlProcessor {
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
        let resized;
        let data = if (th, tw) != (h, w) {
            resized = common::pool()
                .install(|| resize::resize_rgb_filter(rgb, h, w, th, tw, resize::Filter::Bicubic));
            &resized
        } else {
            rgb.as_slice()
        };
        let (gh, gw) = (th / self.spec.patch_size, tw / self.spec.patch_size);
        // `smart_resize` guarantees both: dims are positive and divisible by
        // `patch_size * merge_size`. `patchify` indexes on that (and the `dim`
        // division below needs a non-empty grid), so fail loudly rather than
        // panic if a future spec change breaks the guarantee.
        if gh == 0 || gw == 0 || gh % self.spec.merge_size != 0 || gw % self.spec.merge_size != 0 {
            return Err(format!(
                "qwen_vl: patch grid {gh}x{gw} is empty or not a multiple of \
                 merge_size {}",
                self.spec.merge_size
            ));
        }
        let pixel_values = self.patchify(data, th, tw);
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
        let counts = items
            .iter()
            .map(|Geometry::Grid(grid)| self.tokens_per_image(grid))
            .collect::<Vec<_>>();
        token_layout::layout_by_placeholder(input_ids, self.spec.image_token_id, &counts)
    }

    fn positions(
        &self,
        input_len: usize,
        offsets: &[(u32, u32)],
        items: &[Geometry],
    ) -> Result<PositionOutput, String> {
        let mrope_items = offsets
            .iter()
            .zip(items)
            .map(|(&(start, end), Geometry::Grid(grid))| MropeItem {
                start,
                end,
                grid: *grid,
            })
            .collect::<Vec<_>>();
        let (positions, delta) = mrope_image_only(input_len, &mrope_items, self.spec.merge_size)?;
        Ok(PositionOutput::MRope { positions, delta })
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
    // The downscale branch floors without a lower clamp (as Python does), so a
    // very thin image against a small `max_pixels` can floor a side to 0.
    // Python then fails inside PIL's resize; here it would reach the resize
    // coefficient math (overflow panic in debug, garbage in release) and the
    // `dim = len / (gh * gw)` division, so reject it as a request error.
    if h_bar == 0 || w_bar == 0 {
        return Err(format!(
            "smart_resize: {height}x{width} degenerates to {h_bar}x{w_bar} at \
             max_pixels={max_pixels}; image is too thin for this pixel budget"
        ));
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

/// The qwen scheduler-drain shape, extracted from the generic driver
/// [`Output`](crate::driver::Output). Shared by `sglang-server`'s MM worker
/// and the parity binding so the mapping can't drift; replaced by a generic
/// named-tensor handoff once a second family needs a different shape.
pub struct QwenDrain {
    pub input_ids: Vec<i32>,
    /// All items' `pixel_values`, concatenated in prompt order.
    pub features: Vec<f32>,
    pub grids: Vec<[u32; 3]>,
    pub hashes: Vec<u64>,
    pub offsets: Vec<(u32, u32)>,
    pub mrope: Vec<i64>,
    pub mrope_delta: i64,
}

pub fn pack_drain(output: crate::driver::Output) -> Result<QwenDrain, String> {
    use crate::pipeline::PositionOutput;

    let PositionOutput::MRope { positions, delta } = output.positions else {
        return Err("qwen_vl drain: expected M-RoPE positions".into());
    };
    let mut features = Vec::new();
    let mut grids = Vec::with_capacity(output.items.len());
    let mut hashes = Vec::with_capacity(output.items.len());
    for item in output.items {
        let TensorData::F32(pixel_values) = item.feature.data else {
            return Err("qwen_vl drain: expected f32 feature".into());
        };
        features.extend(pixel_values);
        let grid = item
            .aux
            .into_iter()
            .find_map(|(name, tensor)| match (name.as_str(), tensor.data) {
                ("image_grid_thw", TensorData::I64(v)) => Some(v),
                _ => None,
            })
            .ok_or("qwen_vl drain: missing image_grid_thw")?;
        grids.push([grid[0] as u32, grid[1] as u32, grid[2] as u32]);
        hashes.push(item.hash);
    }
    Ok(QwenDrain {
        input_ids: output.input_ids,
        features,
        grids,
        hashes,
        offsets: output.offsets,
        mrope: positions,
        mrope_delta: delta,
    })
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
    /// Full native pipeline output at the scheduler boundary:
    /// `(input_ids, features, grids, hashes, offsets, mrope, mrope_delta)`.
    type PyNativeOutput<'py> = (
        Vec<i32>,
        Bound<'py, PyArray1<f32>>,
        Vec<(u32, u32, u32)>,
        Vec<u64>,
        Vec<(u32, u32)>,
        Bound<'py, PyArray1<i64>>,
        i64,
    );

    /// Run the full native image path on encoded image bytes:
    /// decode → smart_resize → bicubic → normalize → patchify.
    /// Returns `(pixel_values flat f32, (t, h, w))`.
    #[pyfunction]
    fn preprocess<'py>(
        py: Python<'py>,
        data: Vec<u8>,
        spec_json: &str,
    ) -> PyResult<PyProcessedImage<'py>> {
        let proc = QwenVlProcessor::from_spec_json(spec_json).map_err(PyValueError::new_err)?;
        let out = py
            .detach(move || {
                let (rgb, height, width) = crate::common::decode_rgb(&data)?;
                proc.process_item(&DecodedMedia::Image { rgb, height, width })
            })
            .map_err(PyValueError::new_err)?;
        let Geometry::Grid([t, h, w]) = out.geometry;
        let TensorData::F32(pixel_values) = out.feature.data else {
            return Err(PyValueError::new_err("qwen_vl: expected f32 feature"));
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
        Ok((pos.into_pyarray(py), delta))
    }

    /// One image source: a `str` (data:/base64/file/http, resolved by
    /// `common::fetch`) or raw encoded `bytes`.
    #[derive(FromPyObject)]
    enum PyImageSource {
        Str(String),
        Bytes(Vec<u8>),
    }

    /// Drive the same typed native Qwen request pipeline used by
    /// `sglang-server` (whose message layer owns the wire-payload parsing).
    #[pyfunction]
    #[pyo3(signature = (input_ids, images, spec_json))]
    fn process_native_mm<'py>(
        py: Python<'py>,
        input_ids: Option<Vec<i32>>,
        images: Vec<PyImageSource>,
        spec_json: String,
    ) -> PyResult<PyNativeOutput<'py>> {
        let images = images
            .into_iter()
            .map(|source| match source {
                PyImageSource::Str(s) => crate::driver::ImageSource::String(s),
                PyImageSource::Bytes(b) => crate::driver::ImageSource::Bytes(b),
            })
            .collect();
        let input = crate::driver::MmInput {
            text: None,
            input_ids,
            images,
        };
        let drain = py
            .detach(move || {
                let family = crate::registry::pipeline_from_spec(&spec_json)?;
                let output = crate::driver::process(family.as_ref(), input, |_| {
                    Err("native parity API requires input_ids".into())
                })?;
                pack_drain(output)
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            drain.input_ids,
            drain.features.into_pyarray(py),
            drain.grids.into_iter().map(|[t, h, w]| (t, h, w)).collect(),
            drain.hashes,
            drain.offsets,
            drain.mrope.into_pyarray(py),
            drain.mrope_delta,
        ))
    }

    pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let m = PyModule::new(parent.py(), "qwen_vl")?;
        m.add_function(wrap_pyfunction!(preprocess, &m)?)?;
        m.add_function(wrap_pyfunction!(smart_resize_py, &m)?)?;
        m.add_function(wrap_pyfunction!(mrope_image_only_py, &m)?)?;
        m.add_function(wrap_pyfunction!(process_native_mm, &m)?)?;
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

    /// A thin image against a small `max_pixels` floors one side to 0. That
    /// used to reach the resize coefficient math and panic on a worker thread
    /// (`attempt to multiply with overflow`) instead of rejecting the request.
    #[test]
    fn degenerate_target_is_rejected_not_panicked() {
        // Aspect ratio 200 is exactly at MAX_RATIO, so it passes that guard;
        // 10 / beta then floors to 0 with factor 28.
        assert!(smart_resize(10, 2000, 28, 3136, 3136).is_err());

        let mut spec = spec();
        spec.patch_size = 14;
        spec.min_pixels = 3136;
        spec.max_pixels = 3136;
        let proc = QwenVlProcessor::new(spec).unwrap();
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

    /// The server's message layer gates modalities on what a family declares,
    /// so a family gaining video/audio support must not silently inherit the
    /// images-only default.
    #[test]
    fn qwen_declares_images_only() {
        let caps = QwenVlProcessor::new(spec()).unwrap().capabilities();
        assert!(!caps.video && !caps.audio);
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
        let dim = 24; // 3 * tps * ps * ps
        assert_eq!(pv.len(), 2 * 4 * dim);

        // Patch order (gh/m=1, gw/m=2, m, m): patch 0 = block(0,0) offset (0,0),
        // patch 1 = (0,0)+(0,1) → x0=2, patch 2 = (0,0)+(1,0) → y0=2,
        // patch 4 = block(0,1) → x0=4.
        let lut = |y: usize, x: usize, c: usize| ((y * 16 + x * 2 + c) as f32) / 255.0;
        // patch 1, channel 0, t=0, (py=0, px=0) → pixel (0, 2).
        assert_eq!(pv[dim], lut(0, 2, 0));
        // patch 2, channel 0, t=0, (0,0) → pixel (2, 0).
        assert_eq!(pv[2 * dim], lut(2, 0, 0));
        // patch 4, channel 0 → pixel (0, 4).
        assert_eq!(pv[4 * dim], lut(0, 4, 0));
        // Temporal duplicate: t=1 block equals t=0 block.
        let ps2 = 4; // ps*ps
        assert_eq!(pv[dim + ps2], pv[dim]);
        // Channel 1 block of patch 0 → same pixel, c=1.
        assert_eq!(pv[2 * ps2], lut(0, 0, 1)); // c stride = tps*ps*ps = 8
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
