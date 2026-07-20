pub mod resize;
pub mod transforms;

use std::sync::OnceLock;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;


pub fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::env::var("SGL_MM_RS_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(8, |c| c.get().min(8)));
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("sgl-mm-{i}"))
            .build()
            .expect("failed to build rayon pool")
    })
}

pub fn sha256_u64(data: &[u8]) -> u64 {
    let digest = blake3::hash(data);
    u64::from_be_bytes(digest.as_bytes()[..8].try_into().unwrap())
}

pub fn decode_rgb(data: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
    let img = image::load_from_memory(data).map_err(|e| format!("image decode: {e}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    Ok((rgb.into_raw(), h as usize, w as usize))
}

pub fn decode_rescale(
    data: &[u8],
    rescale_frac: Option<f64>,
    rescale_cap: Option<i64>,
) -> Result<(Vec<u8>, usize, usize), String> {
    let (rgb, h, w) = decode_rgb(data)?;
    let (tw, th) = resize::scaled_dims(w, h, rescale_frac, rescale_cap);
    if (tw, th) == (w, h) {
        return Ok((rgb, h, w));
    }
    Ok((resize::resize_lanczos_rgb(&rgb, h, w, th, tw), th, tw))
}

// --- Python-exposed functions ---

#[pyfunction]
pub fn resize_rgb<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray3<'py, u8>,
    out_w: usize,
    out_h: usize,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    if out_w == 0 || out_h == 0 {
        return Err(PyValueError::new_err("output size must be positive"));
    }
    let shape = arr.shape();
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    if c != 3 {
        return Err(PyValueError::new_err(format!(
            "expected HWC RGB array with 3 channels, got {c}"
        )));
    }
    let data = arr
        .as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))?
        .to_vec();
    let out = py.allow_threads(move || {
        pool().install(|| resize::resize_lanczos_rgb(&data, h, w, out_h, out_w))
    });
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (w, h, rescale_frac=None, rescale_cap=None))]
pub fn scaled_dims(
    w: usize,
    h: usize,
    rescale_frac: Option<f64>,
    rescale_cap: Option<i64>,
) -> (usize, usize) {
    resize::scaled_dims(w, h, rescale_frac, rescale_cap)
}

#[pyfunction]
pub fn image_decode_rgb<'py>(
    py: Python<'py>,
    data: Vec<u8>,
) -> PyResult<(usize, usize, Bound<'py, PyArray1<u8>>)> {
    let (rgb, h, w) = py
        .allow_threads(move || decode_rgb(&data))
        .map_err(PyValueError::new_err)?;
    Ok((h, w, rgb.into_pyarray_bound(py)))
}

#[pyfunction]
pub fn data_hash(py: Python<'_>, data: Vec<u8>) -> u64 {
    py.allow_threads(move || {
        let digest = blake3::hash(&data);
        u64::from_be_bytes(digest.as_bytes()[..8].try_into().unwrap())
    })
}

#[pyfunction]
pub fn base64_decode<'py>(
    py: Python<'py>,
    encoded: &str,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    use base64::Engine;
    let decoded = py
        .allow_threads(|| {
            base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|e| format!("base64 decode error: {e}"))
        })
        .map_err(PyValueError::new_err)?;
    Ok(pyo3::types::PyBytes::new_bound(py, &decoded))
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "common")?;
    m.add_function(wrap_pyfunction!(resize_rgb, &m)?)?;
    m.add_function(wrap_pyfunction!(scaled_dims, &m)?)?;
    m.add_function(wrap_pyfunction!(image_decode_rgb, &m)?)?;
    m.add_function(wrap_pyfunction!(data_hash, &m)?)?;
    m.add_function(wrap_pyfunction!(base64_decode, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
