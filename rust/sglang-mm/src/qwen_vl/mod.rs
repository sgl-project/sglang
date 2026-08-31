//! SGLang-side Qwen VL pieces: the scheduler-drain packing shared by
//! `sglang-server`'s MM worker and the parity bindings. The family pipeline
//! itself lives in `dynamo_mm_preprocessor::qwen_vl`.

use dynamo_mm_preprocessor::driver;
use dynamo_mm_preprocessor::pipeline::{PositionOutput, TensorData};

/// The qwen scheduler-drain shape, extracted from the generic driver
/// [`Output`](driver::Output). Shared by `sglang-server`'s MM worker
/// and the parity binding so the mapping can't drift. TODO(mm-families):
/// replace with a generic named-tensor handoff once a second family needs a
/// different shape.
pub struct QwenPackedOutput {
    pub input_ids: Vec<i32>,
    /// All items' `pixel_values`, concatenated in prompt order; flattened
    /// `[Σ t·h·w, 3·temporal_patch_size·patch_size²]`.
    pub features: Vec<f32>,
    /// Per item `[t, h, w]` patch grid.
    pub grids: Vec<[u32; 3]>,
    pub hashes: Vec<u64>,
    /// Per item inclusive token range in `input_ids`.
    pub offsets: Vec<(u32, u32)>,
    /// Flattened row-major `[3, input_len]` M-RoPE positions.
    pub mrope: Vec<i64>,
    pub mrope_delta: i64,
}

pub fn pack_output(output: driver::Output) -> Result<QwenPackedOutput, String> {
    let PositionOutput::MRope { positions, delta } = output.positions else {
        return Err("qwen_vl pack: expected M-RoPE positions".into());
    };
    let mut features = Vec::new();
    let mut grids = Vec::with_capacity(output.items.len());
    let mut hashes = Vec::with_capacity(output.items.len());
    for item in output.items {
        let TensorData::F32(pixel_values) = item.feature.data else {
            return Err("qwen_vl pack: expected f32 feature".into());
        };
        features.extend(pixel_values);
        let grid = item
            .aux
            .into_iter()
            .find_map(|(name, tensor)| match (name.as_str(), tensor.data) {
                ("image_grid_thw", TensorData::I64(v)) => Some(v),
                _ => None,
            })
            .ok_or("qwen_vl pack: missing image_grid_thw")?;
        grids.push([grid[0] as u32, grid[1] as u32, grid[2] as u32]);
        hashes.push(item.hash);
    }
    Ok(QwenPackedOutput {
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

    use super::pack_output;
    use dynamo_mm_preprocessor::pipeline::{DecodedMedia, Geometry, TensorData};
    use dynamo_mm_preprocessor::qwen_vl::{
        MropeItem, QwenVlProcessor, mrope_image_only, smart_resize,
    };
    use dynamo_mm_preprocessor::{driver, image, registry};

    /// `(pixel_values flat f32, (t, h, w))` for one preprocessed image.
    type PyProcessedImage<'py> = (Bound<'py, PyArray1<f32>>, (u32, u32, u32));
    /// Full Rust pipeline output at the scheduler boundary:
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
        use dynamo_mm_preprocessor::pipeline::MmFamilyProcessor;

        let proc = QwenVlProcessor::from_spec_json(spec_json).map_err(PyValueError::new_err)?;
        let out = py
            .detach(move || {
                let (rgb, height, width) = image::decode::decode_rgb(&data)?;
                proc.process_item(&DecodedMedia::Image { rgb, height, width })
            })
            .map_err(PyValueError::new_err)?;
        let Geometry::Grid([t, h, w]) = out.geometry else {
            return Err(PyValueError::new_err("qwen_vl: expected grid geometry"));
        };
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

    /// One image source: a `str` (data:/base64/file/http, resolved by the
    /// dynamo crate's `fetch`) or raw encoded `bytes`.
    #[derive(FromPyObject)]
    enum PyImageSource {
        Str(String),
        Bytes(Vec<u8>),
    }

    /// Drive the same typed native Qwen request pipeline used by
    /// `sglang-server` (whose message layer owns the wire-payload parsing).
    #[pyfunction]
    #[pyo3(signature = (input_ids, images, spec_json))]
    fn process_mm<'py>(
        py: Python<'py>,
        input_ids: Option<Vec<i32>>,
        images: Vec<PyImageSource>,
        spec_json: String,
    ) -> PyResult<PyNativeOutput<'py>> {
        let images = images
            .into_iter()
            .map(|source| match source {
                PyImageSource::Str(s) => driver::ImageSource::String(s),
                PyImageSource::Bytes(b) => driver::ImageSource::Bytes(b),
            })
            .collect();
        let input = driver::MmInput {
            text: None,
            input_ids,
            images,
        };
        let packed = py
            .detach(move || {
                let family = registry::pipeline_from_spec(&spec_json)?;
                let output = driver::process_with(
                    family.as_ref(),
                    input,
                    |_| Err("native parity API requires input_ids".into()),
                    &crate::common::fetch_options_from_env().into(),
                )?;
                pack_output(output)
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            packed.input_ids,
            packed.features.into_pyarray(py),
            packed
                .grids
                .into_iter()
                .map(|[t, h, w]| (t, h, w))
                .collect(),
            packed.hashes,
            packed.offsets,
            packed.mrope.into_pyarray(py),
            packed.mrope_delta,
        ))
    }

    pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let m = PyModule::new(parent.py(), "qwen_vl")?;
        m.add_function(wrap_pyfunction!(preprocess, &m)?)?;
        m.add_function(wrap_pyfunction!(smart_resize_py, &m)?)?;
        m.add_function(wrap_pyfunction!(mrope_image_only_py, &m)?)?;
        m.add_function(wrap_pyfunction!(process_mm, &m)?)?;
        parent.add_submodule(&m)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python::register;
