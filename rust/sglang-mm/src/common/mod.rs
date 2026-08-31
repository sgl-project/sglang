//! SGLang-facing surface over the dynamo crate's primitives: re-exports for
//! in-crate consumers (Inkling), the env-var shims, the Inkling decode helper,
//! and the PyO3 bindings.

pub use dynamo_mm_preprocessor::image::resize;
pub use dynamo_mm_preprocessor::{content_hash_u64, fetch, par};

pub use dynamo_mm_preprocessor::image::decode::decode_rgb;

/// Fetch knobs from the SGLang environment — Python parity:
/// `int(os.getenv("REQUEST_TIMEOUT", "3"))` seconds per image GET.
pub fn fetch_options_from_env() -> fetch::FetchOptions {
    let mut opts = fetch::FetchOptions::default();
    if let Some(secs) = std::env::var("REQUEST_TIMEOUT")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        opts.timeout = std::time::Duration::from_secs(secs);
    }
    opts
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

// --- Python bindings (feature-gated: absent from the pure-Rust rlib) ---

#[cfg(feature = "python")]
mod python {
    use numpy::{IntoPyArray, PyArray1, PyReadonlyArray3, PyUntypedArrayMethods};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    use super::{decode_rgb, resize};

    /// `resample` names the implementation to reproduce: `"pil_lanczos"` (the
    /// inkling default), `"pil_bicubic"`, or `"aten_u8"` (torchvision's uint8
    /// antialias bicubic). Exposed so the bit-exactness tests can cover each.
    #[pyfunction]
    #[pyo3(signature = (arr, out_w, out_h, resample="pil_lanczos"))]
    pub fn resize_rgb<'py>(
        py: Python<'py>,
        arr: PyReadonlyArray3<'py, u8>,
        out_w: usize,
        out_h: usize,
        resample: &str,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        if out_w == 0 || out_h == 0 {
            return Err(PyValueError::new_err("output size must be positive"));
        }
        let resample = match resample {
            "pil_lanczos" => resize::Resample::Pil(resize::Filter::Lanczos),
            "pil_bicubic" => resize::Resample::Pil(resize::Filter::Bicubic),
            "aten_u8" => resize::Resample::AtenU8,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown resample {other:?}; expected \"pil_lanczos\", \
                     \"pil_bicubic\" or \"aten_u8\""
                )));
            }
        };
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
        let out = py.detach(move || resize::resize_rgb(&data, h, w, out_h, out_w, resample));
        Ok(out.into_pyarray(py))
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
            .detach(move || decode_rgb(&data))
            .map_err(PyValueError::new_err)?;
        Ok((h, w, rgb.into_pyarray(py)))
    }

    /// Named `content_hash`, not `data_hash`, so it is not mistaken for
    /// `sglang.srt.managers.mm_utils.data_hash` (SHA-256); this is blake3.
    #[pyfunction]
    pub fn content_hash(py: Python<'_>, data: Vec<u8>) -> u64 {
        py.detach(move || super::content_hash_u64(&data))
    }

    #[pyfunction]
    pub fn fetch_bytes<'py>(py: Python<'py>, source: String) -> PyResult<Bound<'py, PyBytes>> {
        use super::fetch;
        let data = py
            .detach(move || {
                fetch::fetch_bytes_budgeted_with(
                    &source,
                    &fetch::ByteBudget::new(fetch::MAX_FETCH_BYTES),
                    &super::fetch_options_from_env(),
                )
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(PyBytes::new(py, &data))
    }

    #[pyfunction]
    pub fn base64_decode<'py>(
        py: Python<'py>,
        encoded: &str,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        use base64::Engine;
        let decoded = py
            .detach(|| {
                base64::engine::general_purpose::STANDARD
                    .decode(encoded)
                    .map_err(|e| format!("base64 decode error: {e}"))
            })
            .map_err(PyValueError::new_err)?;
        Ok(pyo3::types::PyBytes::new(py, &decoded))
    }

    pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let m = PyModule::new(parent.py(), "common")?;
        m.add_function(wrap_pyfunction!(resize_rgb, &m)?)?;
        m.add_function(wrap_pyfunction!(scaled_dims, &m)?)?;
        m.add_function(wrap_pyfunction!(image_decode_rgb, &m)?)?;
        m.add_function(wrap_pyfunction!(content_hash, &m)?)?;
        m.add_function(wrap_pyfunction!(fetch_bytes, &m)?)?;
        m.add_function(wrap_pyfunction!(base64_decode, &m)?)?;
        parent.add_submodule(&m)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python::register;
