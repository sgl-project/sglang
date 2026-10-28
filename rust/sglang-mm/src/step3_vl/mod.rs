use crate::common::{par, resize};
use crate::pipeline::{
    DecodedMedia, Geometry, MmFamilyProcessor, ProcessedItem, Segment, Tensor, TensorData,
    TokenLayout, TokenPattern,
};

const PREPROCESS_MAX_SIDE: usize = 3024;
const THIN_SIDE: usize = 32;

#[derive(Clone, Debug, serde::Deserialize)]
pub struct Step3VlSpec {
    pub image_token_id: i64,
    pub image_start_id: i64,
    pub image_end_id: i64,
    pub patch_start_id: i64,
    pub patch_end_id: i64,
    pub patch_newline_id: i64,
    pub image_size: usize,
    pub patch_size: usize,
    pub image_feature_tokens: usize,
    pub patch_feature_tokens: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Semantics {
    Pil,
    Tensor,
}

impl Semantics {
    pub fn for_source(jpeg: bool) -> Self {
        if jpeg { Self::Tensor } else { Self::Pil }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub padded_side: Option<usize>,
    pub resized: (usize, usize),
    pub window: usize,
    pub crop_resize: Option<(usize, usize)>,
    pub windows: Vec<Window>,
    pub newlines: Vec<bool>,
}

pub fn determine_window_size(long: usize, short: usize) -> usize {
    let ratio = long as f64 / short as f64;
    if long <= 728 {
        if ratio > 1.5 { short } else { 0 }
    } else if ratio > 4.0 {
        short.min(504)
    } else {
        504
    }
}

pub fn size_for_padding(w: usize, h: usize) -> (usize, usize) {
    let ratio = w as f64 / h as f64;
    if h.min(w) < THIN_SIDE && !(1.0 / 4.0..=4.0).contains(&ratio) {
        let side = h.max(w);
        (side, side)
    } else {
        (w, h)
    }
}

pub fn size_for_preprocess(w: usize, h: usize) -> (usize, usize) {
    let long = h.max(w);
    if long > PREPROCESS_MAX_SIDE {
        let scale = PREPROCESS_MAX_SIDE as f64 / long as f64;
        ((w as f64 * scale) as usize, (h as f64 * scale) as usize)
    } else {
        (w, h)
    }
}

fn crop_axis(size: usize, window: usize) -> usize {
    let ratio = size as f64 / window as f64;
    if ratio < 1.0 {
        return size;
    }
    let decimal = ratio - (size / window) as f64;
    let n = if decimal > 0.2 {
        ratio as usize + 1
    } else {
        ratio as usize
    };
    window * n
}

pub fn size_for_crop(w: usize, h: usize, window: usize) -> (usize, usize) {
    (crop_axis(w, window), crop_axis(h, window))
}

fn window_starts(size: usize, window: usize) -> Vec<usize> {
    let n = if size <= window {
        1
    } else {
        ((size - window) as f64 / window as f64 + 1.0).ceil() as usize
    };
    let mut starts: Vec<usize> = (0..n).map(|i| window * i).collect();
    if starts.len() > 1 && starts[n - 1] + window > size {
        starts[n - 1] = size - window;
    }
    starts
}

pub fn slide_window(width: usize, height: usize, window: usize) -> (Vec<Window>, usize) {
    let xs = window_starts(width, window);
    let ys = window_starts(height, window);
    let windows = ys
        .iter()
        .flat_map(|&y| {
            xs.iter().map(move |&x| Window {
                x,
                y,
                w: window,
                h: window,
            })
        })
        .collect();
    (windows, xs.len())
}

pub fn plan(width: usize, height: usize) -> Plan {
    let (pw, ph) = size_for_padding(width, height);
    let padded_side = ((pw, ph) != (width, height)).then(|| width.max(height));
    let (iw, ih) = padded_side.map_or((width, height), |s| (s, s));
    let (w1, h1) = size_for_preprocess(iw, ih);
    let window = determine_window_size(h1.max(w1), h1.min(w1));
    if window == 0 {
        return Plan {
            padded_side,
            resized: (w1, h1),
            window,
            crop_resize: None,
            windows: Vec::new(),
            newlines: Vec::new(),
        };
    }
    let (w2, h2) = size_for_crop(w1, h1, window);
    let crop_resize = ((w2, h2) != (iw, ih)).then_some((w2, h2));
    let (windows, x_num) = slide_window(w2, h2, window);
    let n = windows.len();
    let mut newlines: Vec<bool> = (0..n).map(|i| (i + 1) % x_num == 0).collect();
    if n > 0 {
        newlines[n - 1] = false;
    }
    Plan {
        padded_side,
        resized: (w1, h1),
        window,
        crop_resize,
        windows,
        newlines,
    }
}

struct Rgb {
    data: Vec<u8>,
    w: usize,
    h: usize,
}

impl Rgb {
    fn square_pad(self, side: usize) -> Rgb {
        let mut data = vec![0u8; side * side * 3];
        for y in 0..self.h {
            let src = &self.data[y * self.w * 3..(y + 1) * self.w * 3];
            data[y * side * 3..y * side * 3 + self.w * 3].copy_from_slice(src);
        }
        Rgb {
            data,
            w: side,
            h: side,
        }
    }

    fn resize(&self, w: usize, h: usize, semantics: Semantics) -> Rgb {
        let data = match semantics {
            Semantics::Pil => resize::resize_rgb(
                &self.data,
                self.h,
                self.w,
                h,
                w,
                resize::Resample::Pil(resize::Filter::Bilinear),
            ),
            Semantics::Tensor => {
                resize::resize_aa_bilinear_rgb_u8(&self.data, self.h, self.w, h, w)
            }
        };
        Rgb { data, w, h }
    }

    fn crop(&self, win: &Window, semantics: Semantics) -> Rgb {
        let (w, h) = match semantics {
            Semantics::Pil => (win.w, win.h),
            Semantics::Tensor => (
                win.w.min(self.w.saturating_sub(win.x)),
                win.h.min(self.h.saturating_sub(win.y)),
            ),
        };
        let mut data = vec![0u8; w * h * 3];
        let copy_w = w.min(self.w.saturating_sub(win.x));
        for row in 0..h {
            let sy = win.y + row;
            if sy >= self.h || copy_w == 0 {
                continue;
            }
            let src = &self.data[(sy * self.w + win.x) * 3..(sy * self.w + win.x + copy_w) * 3];
            data[row * w * 3..row * w * 3 + copy_w * 3].copy_from_slice(src);
        }
        Rgb { data, w, h }
    }
}

pub struct Step3VlProcessor {
    spec: Step3VlSpec,
}

pub struct Step3VlImage {
    pub pixel_values: Vec<f32>,
    pub patch_pixel_values: Vec<f32>,
    pub newlines: Vec<bool>,
}

impl Step3VlProcessor {
    pub fn new(spec: Step3VlSpec) -> Result<Self, String> {
        if spec.image_size == 0
            || spec.patch_size == 0
            || spec.image_feature_tokens == 0
            || spec.patch_feature_tokens == 0
        {
            return Err("step3_vl spec: sizes must be positive".into());
        }
        if spec.image_std.contains(&0.0) {
            return Err("step3_vl spec: image_std must be non-zero".into());
        }
        Ok(Self { spec })
    }

    pub fn from_spec_json(json: &str) -> Result<Self, String> {
        let spec: Step3VlSpec =
            serde_json::from_str(json).map_err(|e| format!("step3_vl spec: {e}"))?;
        Self::new(spec)
    }

    pub fn spec(&self) -> &Step3VlSpec {
        &self.spec
    }

    fn normalized_resize(&self, img: &Rgb, out: usize) -> Vec<f32> {
        let n = img.w * img.h;
        let mut planes = vec![0.0f32; 3 * n];
        for (i, px) in img.data.chunks_exact(3).enumerate() {
            for c in 0..3 {
                planes[c * n + i] =
                    (px[c] as f32 / 255.0 - self.spec.image_mean[c]) / self.spec.image_std[c];
            }
        }
        resize::resize_aa_bilinear_planes(&planes, 3, img.h, img.w, out, out)
    }

    pub fn preprocess(
        &self,
        rgb: Vec<u8>,
        width: usize,
        height: usize,
        semantics: Semantics,
    ) -> Result<Step3VlImage, String> {
        if width == 0 || height == 0 || rgb.len() != width * height * 3 {
            return Err(format!("step3_vl: invalid {width}x{height} RGB buffer"));
        }
        let plan = plan(width, height);
        let mut img = Rgb {
            data: rgb,
            w: width,
            h: height,
        };
        if let Some(side) = plan.padded_side {
            img = img.square_pad(side);
        }
        let img = img.resize(plan.resized.0, plan.resized.1, semantics);
        if img.w == 0 || img.h == 0 {
            return Err(format!("step3_vl: {width}x{height} image resizes to zero"));
        }
        let pixel_values = self.normalized_resize(&img, self.spec.image_size);
        if plan.windows.is_empty() {
            return Ok(Step3VlImage {
                pixel_values,
                patch_pixel_values: Vec::new(),
                newlines: Vec::new(),
            });
        }
        let crop_src = match plan.crop_resize {
            Some((w, h)) => img.resize(w, h, semantics),
            None => img,
        };
        let per_patch = 3 * self.spec.patch_size * self.spec.patch_size;
        let patches = par::try_map(&plan.windows, |win| -> Result<Vec<f32>, String> {
            let patch = crop_src.crop(win, semantics);
            if patch.w == 0 || patch.h == 0 {
                return Err("step3_vl: empty patch crop".into());
            }
            Ok(self.normalized_resize(&patch, self.spec.patch_size))
        })?;
        let mut patch_pixel_values = Vec::with_capacity(patches.len() * per_patch);
        for p in patches {
            patch_pixel_values.extend_from_slice(&p);
        }
        Ok(Step3VlImage {
            pixel_values,
            patch_pixel_values,
            newlines: plan.newlines,
        })
    }

    pub fn tokens_for(&self, newlines: &[bool]) -> Vec<i64> {
        let s = &self.spec;
        let mut ids = Vec::with_capacity(
            newlines.len() * (s.patch_feature_tokens + 3) + s.image_feature_tokens + 2,
        );
        for &newline in newlines {
            ids.push(s.patch_start_id);
            ids.extend(std::iter::repeat_n(
                s.image_token_id,
                s.patch_feature_tokens,
            ));
            ids.push(s.patch_end_id);
            if newline {
                ids.push(s.patch_newline_id);
            }
        }
        ids.push(s.image_start_id);
        ids.extend(std::iter::repeat_n(
            s.image_token_id,
            s.image_feature_tokens,
        ));
        ids.push(s.image_end_id);
        ids
    }

    pub fn num_image_tokens(&self, width: usize, height: usize) -> usize {
        let plan = plan(width, height);
        self.tokens_for(&plan.newlines).len()
    }

    pub fn expand(&self, input_ids: &[i64], items: &[Vec<bool>]) -> Result<TokenLayout, String> {
        let found = input_ids
            .iter()
            .filter(|&&id| id == self.spec.image_token_id)
            .count();
        if found != items.len() {
            return Err(format!(
                "prompt has {found} image placeholder(s) but {} image(s)",
                items.len()
            ));
        }
        let mut segments = Vec::new();
        let mut start = 0usize;
        let mut item = 0usize;
        for (i, &id) in input_ids.iter().enumerate() {
            if id != self.spec.image_token_id {
                continue;
            }
            if i > start {
                segments.push(Segment::Text(start..i));
            }
            segments.push(Segment::Media {
                item,
                pattern: TokenPattern::Explicit(self.tokens_for(&items[item])),
            });
            item += 1;
            start = i + 1;
        }
        if start < input_ids.len() {
            segments.push(Segment::Text(start..input_ids.len()));
        }
        Ok(TokenLayout { segments })
    }
}

pub fn image_token_runs(
    input_ids: &[i64],
    span: (u32, u32),
    image_token_id: i64,
) -> Vec<(u32, u32)> {
    let mut runs = Vec::new();
    let mut run_start: Option<u32> = None;
    for pos in span.0..=span.1 {
        let hit = input_ids.get(pos as usize) == Some(&image_token_id);
        match (hit, run_start) {
            (true, None) => run_start = Some(pos),
            (false, Some(s)) => {
                runs.push((s, pos - 1));
                run_start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = run_start {
        runs.push((s, span.1));
    }
    runs
}

impl MmFamilyProcessor for Step3VlProcessor {
    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, String> {
        let DecodedMedia::Image {
            rgb,
            height,
            width,
            jpeg,
        } = media;
        let out = self.preprocess(rgb.clone(), *width, *height, Semantics::for_source(*jpeg))?;
        let n = out.newlines.len();
        let (s, p) = (self.spec.image_size, self.spec.patch_size);
        Ok(ProcessedItem {
            feature: Tensor {
                shape: vec![1, 3, s, s],
                data: TensorData::F32(out.pixel_values),
            },
            aux: vec![
                (
                    "patch_pixel_values".to_string(),
                    Tensor {
                        shape: vec![n, 3, p, p],
                        data: TensorData::F32(out.patch_pixel_values),
                    },
                ),
                (
                    "patch_newline_mask".to_string(),
                    Tensor {
                        shape: vec![n],
                        data: TensorData::I64(out.newlines.iter().map(|&b| b as i64).collect()),
                    },
                ),
                (
                    "num_patches".to_string(),
                    Tensor {
                        shape: vec![1],
                        data: TensorData::I64(vec![n as i64]),
                    },
                ),
            ],
            geometry: Geometry::Tiles(out.newlines),
        })
    }

    fn layout(&self, input_ids: &[i64], items: &[Geometry]) -> Result<TokenLayout, String> {
        let newlines = items
            .iter()
            .map(|geometry| match geometry {
                Geometry::Tiles(newlines) => Ok(newlines.clone()),
                _ => Err("step3_vl: expected tile geometry".to_string()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.expand(input_ids, &newlines)
    }
}

pub struct Step3VlPackedItem {
    pub pixel_values: Vec<f32>,
    pub patch_pixel_values: Vec<f32>,
    pub num_patches: usize,
    pub newlines: Vec<bool>,
    pub hash: u64,
    pub offsets: Vec<(u32, u32)>,
}

pub struct Step3VlPackedOutput {
    pub input_ids: Vec<i64>,
    pub items: Vec<Step3VlPackedItem>,
}

pub fn pack_output(
    output: crate::driver::Output,
    image_token_id: i64,
) -> Result<Step3VlPackedOutput, String> {
    let input_ids = output.input_ids;
    let mut items = Vec::with_capacity(output.items.len());
    for (item, &span) in output.items.into_iter().zip(&output.offsets) {
        let TensorData::F32(pixel_values) = item.feature.data else {
            return Err("step3_vl: expected f32 pixel_values".into());
        };
        let mut patch_pixel_values = Vec::new();
        let mut newlines = Vec::new();
        let mut num_patches = 0usize;
        for (name, tensor) in item.aux {
            match (name.as_str(), tensor.data) {
                ("patch_pixel_values", TensorData::F32(v)) => patch_pixel_values = v,
                ("patch_newline_mask", TensorData::I64(v)) => {
                    newlines = v.into_iter().map(|b| b != 0).collect()
                }
                ("num_patches", TensorData::I64(v)) => {
                    num_patches = v.first().copied().unwrap_or(0) as usize
                }
                (other, _) => return Err(format!("step3_vl: unexpected aux tensor {other}")),
            }
        }
        items.push(Step3VlPackedItem {
            pixel_values,
            patch_pixel_values,
            num_patches,
            newlines,
            hash: item.hash,
            offsets: image_token_runs(&input_ids, span, image_token_id),
        });
    }
    Ok(Step3VlPackedOutput { input_ids, items })
}

#[cfg(feature = "python")]
mod python {
    use numpy::{IntoPyArray, PyArray1};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use super::*;

    type PyImage<'py> = (
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Vec<bool>,
    );
    type PyPackedItem<'py> = (
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        usize,
        Vec<bool>,
        u64,
        Vec<(u32, u32)>,
    );

    fn parse_semantics(semantics: Option<&str>, data: &[u8]) -> PyResult<Semantics> {
        match semantics {
            None => Ok(Semantics::for_source(crate::common::is_jpeg(data))),
            Some("pil") => Ok(Semantics::Pil),
            Some("tensor") => Ok(Semantics::Tensor),
            Some(other) => Err(PyValueError::new_err(format!(
                "step3_vl: unknown semantics {other:?}"
            ))),
        }
    }

    fn to_py<'py>(py: Python<'py>, out: Step3VlImage) -> PyImage<'py> {
        (
            out.pixel_values.into_pyarray(py),
            out.patch_pixel_values.into_pyarray(py),
            out.newlines,
        )
    }

    #[pyfunction]
    #[pyo3(signature = (data, spec_json, semantics=None))]
    fn preprocess<'py>(
        py: Python<'py>,
        data: Vec<u8>,
        spec_json: &str,
        semantics: Option<&str>,
    ) -> PyResult<PyImage<'py>> {
        let proc = Step3VlProcessor::from_spec_json(spec_json).map_err(PyValueError::new_err)?;
        let semantics = parse_semantics(semantics, &data)?;
        let out = py
            .detach(move || {
                let (rgb, height, width) = crate::common::decode_rgb(&data)?;
                proc.preprocess(rgb, width, height, semantics)
            })
            .map_err(PyValueError::new_err)?;
        Ok(to_py(py, out))
    }

    #[pyfunction]
    #[pyo3(signature = (datas, spec_json, semantics=None))]
    fn preprocess_batch<'py>(
        py: Python<'py>,
        datas: Vec<Vec<u8>>,
        spec_json: &str,
        semantics: Option<Vec<String>>,
    ) -> PyResult<Vec<PyImage<'py>>> {
        let proc = Step3VlProcessor::from_spec_json(spec_json).map_err(PyValueError::new_err)?;
        if semantics.as_ref().is_some_and(|s| s.len() != datas.len()) {
            return Err(PyValueError::new_err(
                "step3_vl: semantics must have one entry per image",
            ));
        }
        let modes = datas
            .iter()
            .enumerate()
            .map(|(i, d)| parse_semantics(semantics.as_ref().map(|s| s[i].as_str()), d))
            .collect::<PyResult<Vec<_>>>()?;
        let outs = py
            .detach(move || {
                let indexed: Vec<(usize, &Vec<u8>)> = datas.iter().enumerate().collect();
                par::try_map(&indexed, |&(i, data)| {
                    let (rgb, height, width) = crate::common::decode_rgb(data)?;
                    proc.preprocess(rgb, width, height, modes[i])
                })
            })
            .map_err(PyValueError::new_err)?;
        Ok(outs.into_iter().map(|o| to_py(py, o)).collect())
    }

    type ItemRuns = Vec<Vec<(u32, u32)>>;
    type WindowPlan = (
        Option<usize>,
        (usize, usize),
        usize,
        Vec<(usize, usize, usize, usize)>,
        Vec<bool>,
    );

    #[pyfunction]
    fn expand_input_ids(
        input_ids: Vec<i64>,
        newlines: Vec<Vec<bool>>,
        spec_json: &str,
    ) -> PyResult<(Vec<i64>, ItemRuns)> {
        let proc = Step3VlProcessor::from_spec_json(spec_json).map_err(PyValueError::new_err)?;
        let layout = proc
            .expand(&input_ids, &newlines)
            .map_err(PyValueError::new_err)?;
        let expanded =
            crate::common::token_layout::apply_layout(&input_ids, &layout, newlines.len())
                .map_err(PyValueError::new_err)?;
        let runs = expanded
            .offsets
            .iter()
            .map(|&span| image_token_runs(&expanded.input_ids, span, proc.spec().image_token_id))
            .collect();
        Ok((expanded.input_ids, runs))
    }

    #[pyfunction]
    fn num_image_tokens(width: usize, height: usize, spec_json: &str) -> PyResult<usize> {
        let proc = Step3VlProcessor::from_spec_json(spec_json).map_err(PyValueError::new_err)?;
        Ok(proc.num_image_tokens(width, height))
    }

    #[pyfunction]
    fn plan_windows(width: usize, height: usize) -> WindowPlan {
        let p = plan(width, height);
        (
            p.padded_side,
            p.resized,
            p.window,
            p.windows.iter().map(|w| (w.x, w.y, w.w, w.h)).collect(),
            p.newlines,
        )
    }

    #[derive(FromPyObject)]
    enum PyImageSource {
        Str(String),
        Bytes(Vec<u8>),
    }

    #[pyfunction]
    #[pyo3(signature = (input_ids, images, spec_json))]
    fn process_mm<'py>(
        py: Python<'py>,
        input_ids: Vec<i64>,
        images: Vec<PyImageSource>,
        spec_json: String,
    ) -> PyResult<(Vec<i64>, Vec<PyPackedItem<'py>>)> {
        let images = images
            .into_iter()
            .map(|source| match source {
                PyImageSource::Str(s) => crate::driver::ImageSource::String(s),
                PyImageSource::Bytes(b) => crate::driver::ImageSource::Bytes(b),
            })
            .collect();
        let input = crate::driver::MmInput { input_ids, images };
        let packed = py
            .detach(move || {
                let proc = Step3VlProcessor::from_spec_json(&spec_json)?;
                let image_token_id = proc.spec().image_token_id;
                let output = crate::driver::process(&proc, input)?;
                pack_output(output, image_token_id)
            })
            .map_err(PyValueError::new_err)?;
        let items = packed
            .items
            .into_iter()
            .map(|item| {
                (
                    item.pixel_values.into_pyarray(py),
                    item.patch_pixel_values.into_pyarray(py),
                    item.num_patches,
                    item.newlines,
                    item.hash,
                    item.offsets,
                )
            })
            .collect();
        Ok((packed.input_ids, items))
    }

    pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let m = PyModule::new(parent.py(), "step3_vl")?;
        m.add_function(wrap_pyfunction!(preprocess, &m)?)?;
        m.add_function(wrap_pyfunction!(preprocess_batch, &m)?)?;
        m.add_function(wrap_pyfunction!(expand_input_ids, &m)?)?;
        m.add_function(wrap_pyfunction!(num_image_tokens, &m)?)?;
        m.add_function(wrap_pyfunction!(plan_windows, &m)?)?;
        m.add_function(wrap_pyfunction!(process_mm, &m)?)?;
        parent.add_submodule(&m)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> Step3VlSpec {
        Step3VlSpec {
            image_token_id: 7,
            image_start_id: 1,
            image_end_id: 2,
            patch_start_id: 3,
            patch_end_id: 4,
            patch_newline_id: 5,
            image_size: 728,
            patch_size: 504,
            image_feature_tokens: 169,
            patch_feature_tokens: 81,
            image_mean: [0.48145466, 0.4578275, 0.40821073],
            image_std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
        }
    }

    #[test]
    fn window_size_matches_python_rules() {
        assert_eq!(determine_window_size(700, 600), 0);
        assert_eq!(determine_window_size(700, 400), 400);
        assert_eq!(determine_window_size(1000, 800), 504);
        assert_eq!(determine_window_size(3000, 600), 504);
        assert_eq!(determine_window_size(3000, 500), 500);
        assert_eq!(determine_window_size(3000, 700), 504);
    }

    #[test]
    fn padding_preprocess_and_crop_sizes() {
        assert_eq!(size_for_padding(400, 20), (400, 400));
        assert_eq!(size_for_padding(100, 20), (100, 100));
        assert_eq!(size_for_padding(100, 40), (100, 40));
        assert_eq!(size_for_preprocess(4000, 3000), (3024, 2268));
        assert_eq!(size_for_preprocess(1000, 800), (1000, 800));
        assert_eq!(size_for_crop(1000, 800, 504), (1008, 1008));
        assert_eq!(size_for_crop(600, 300, 504), (504, 300));
        assert_eq!(size_for_crop(1100, 504, 504), (1008, 504));
        assert_eq!(size_for_crop(1200, 504, 504), (1512, 504));
    }

    #[test]
    fn plan_windows_and_newlines() {
        let p = plan(1000, 800);
        assert_eq!(p.window, 504);
        assert_eq!(p.crop_resize, Some((1008, 1008)));
        assert_eq!(p.windows.len(), 4);
        assert_eq!(p.newlines, vec![false, true, false, false]);
        let p = plan(640, 480);
        assert!(p.windows.is_empty());
        assert_eq!(p.window, 0);
    }

    #[test]
    fn token_layout_matches_python_replacement() {
        let proc = Step3VlProcessor::new(spec()).unwrap();
        let ids = proc.tokens_for(&[false, true, false]);
        assert_eq!(ids.len(), 3 * 83 + 1 + 171);
        assert_eq!(ids[0], 3);
        assert_eq!(ids[82], 4);
        assert_eq!(ids[83], 3);
        assert_eq!(ids[165], 4);
        assert_eq!(ids[166], 5);
        assert_eq!(ids[167], 3);
        assert_eq!(*ids.last().unwrap(), 2);
        assert_eq!(proc.num_image_tokens(1000, 800), 4 * 83 + 1 + 171);
    }

    #[test]
    fn expand_and_runs() {
        let proc = Step3VlProcessor::new(spec()).unwrap();
        let layout = proc.expand(&[9, 7, 9, 7], &[vec![false], vec![]]).unwrap();
        let out = crate::common::token_layout::apply_layout(&[9, 7, 9, 7], &layout, 2).unwrap();
        let first = image_token_runs(&out.input_ids, out.offsets[0], 7);
        assert_eq!(first, vec![(2, 82), (85, 253)]);
        let second = image_token_runs(&out.input_ids, out.offsets[1], 7);
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].1 - second[0].0 + 1, 169);
        assert!(proc.expand(&[9, 7], &[]).is_err());
    }

    #[test]
    fn preprocess_shapes_and_crop_semantics() {
        let proc = Step3VlProcessor::new(spec()).unwrap();
        let (w, h) = (1000usize, 400usize);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
        let pil = proc.preprocess(rgb.clone(), w, h, Semantics::Pil).unwrap();
        let ten = proc.preprocess(rgb, w, h, Semantics::Tensor).unwrap();
        assert_eq!(pil.pixel_values.len(), 3 * 728 * 728);
        assert_eq!(pil.newlines.len(), ten.newlines.len());
        let n = pil.newlines.len();
        assert_eq!(pil.patch_pixel_values.len(), n * 3 * 504 * 504);
        assert_eq!(ten.patch_pixel_values.len(), n * 3 * 504 * 504);
    }

    #[test]
    fn aa_resize_identity_and_constant() {
        let src: Vec<f32> = (0..3 * 16 * 16).map(|i| i as f32 * 0.25).collect();
        assert_eq!(
            resize::resize_aa_bilinear_planes(&src, 3, 16, 16, 16, 16),
            src
        );
        let flat = vec![3.5f32; 3 * 20 * 30];
        let out = resize::resize_aa_bilinear_planes(&flat, 3, 20, 30, 7, 11);
        assert!(out.iter().all(|&v| (v - 3.5).abs() < 1e-5));
    }
}
