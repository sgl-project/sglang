#![allow(deprecated)]

use std::{ffi::CString, sync::Arc};

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64_ENGINE, Engine};
use pyo3::ffi::c_str;
use pyo3::{
    prelude::*,
    types::{PyAny, PyDict, PyList},
    PyErr,
};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use serde_json::Value;

use super::types::MultiModalInputs;

#[cfg(test)]
use image::RgbImage;

static PYTHON_SHIM: &str = include_str!("mm_processor_shim.py");

#[derive(Debug, Clone)]
pub struct EncodedImage {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub mode: &'static str,
    pub bytes: Vec<u8>,
}

impl EncodedImage {
    pub fn from_rgb_bytes(width: u32, height: u32, bytes: Vec<u8>) -> Self {
        Self {
            width,
            height,
            channels: 3,
            mode: "rgb",
            bytes,
        }
    }

    #[cfg(test)]
    pub fn from_rgb_image(image: &RgbImage) -> Self {
        Self::from_rgb_bytes(image.width(), image.height(), image.as_raw().to_vec())
    }
}

pub struct MmProcessRequest {
    pub model_id: String,
    pub prompt: String,
    pub images: Vec<EncodedImage>,
    pub processor_kwargs: Value,
    pub tokenization_kwargs: Value,
    pub mm_uuids: Option<Value>,
}

impl Default for MmProcessRequest {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            prompt: String::new(),
            images: Vec::new(),
            processor_kwargs: Value::Null,
            tokenization_kwargs: Value::Null,
            mm_uuids: None,
        }
    }
}

pub struct PythonMmBridge {
    module: Py<PyModule>,
}

impl PythonMmBridge {
    pub fn new() -> Result<Self> {
        Python::with_gil(|py| -> PyResult<Self> {
            let code = CString::new(PYTHON_SHIM)
                .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
            let module = PyModule::from_code(
                py,
                code.as_c_str(),
                c_str!("mm_processor_shim.py"),
                c_str!("mm_processor_shim"),
            )?;
            Ok(Self {
                module: module.into(),
            })
        })
        .map_err(|err| err.into())
    }

    pub fn process(&self, request: &MmProcessRequest) -> Result<MultiModalInputs> {
        Python::with_gil(|py| -> PyResult<MultiModalInputs> {
            let module = self.module.bind(py);
            let func = module.getattr("process_mm")?;

            let py_mm_data = PyDict::new(py);
            if !request.images.is_empty() {
                let list = PyList::empty(py);
                for image in &request.images {
                    let encoded = BASE64_ENGINE.encode(&image.bytes);
                    let entry = PyDict::new(py);
                    entry.set_item("width", image.width)?;
                    entry.set_item("height", image.height)?;
                    entry.set_item("channels", image.channels)?;
                    entry.set_item("mode", image.mode)?;
                    entry.set_item("data", encoded)?;
                    list.append(entry)?;
                }
                py_mm_data.set_item("image", list)?;
            }

            let processor_kwargs = python_json(py, &request.processor_kwargs)?;
            let tok_kwargs = python_json(py, &request.tokenization_kwargs)?;

            let mm_uuid_value = request
                .mm_uuids
                .as_ref()
                .map(|value| python_json(py, value))
                .transpose()?;

            let args = (
                request.model_id.as_str(),
                request.prompt.as_str(),
                &py_mm_data,
                &processor_kwargs,
                &tok_kwargs,
            );

            let kwargs = PyDict::new(py);
            if let Some(mm_uuid_py) = mm_uuid_value {
                kwargs.set_item("mm_uuids", &mm_uuid_py)?;
            }

            let result: String = func.call(args, Some(&kwargs))?.extract()?;
            let json: Value = serde_json::from_str(&result)
                .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
            let inputs: MultiModalInputs = serde_json::from_value(json)
                .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
            Ok(inputs)
        })
        .map_err(|err| err.into())
    }
}

fn python_json<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    match value {
        Value::Null => Ok(py.None().into_bound(py)),
        _ => {
            let dumped = serde_json::to_string(value)
                .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
            let module = PyModule::import(py, "json")?;
            module.call_method1("loads", (dumped,))
        }
    }
}

pub struct PythonProcessorPool {
    bridge: Arc<PythonMmBridge>,
    pool: ThreadPool,
}

impl PythonProcessorPool {
    pub fn new(num_threads: usize) -> Result<Self> {
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads.max(1))
            .build()
            .context("failed to build Rayon thread pool")?;
        Ok(Self {
            bridge: Arc::new(PythonMmBridge::new()?),
            pool,
        })
    }

    pub fn process_batch(&self, requests: &[MmProcessRequest]) -> Vec<Result<MultiModalInputs>> {
        let bridge = self.bridge.clone();
        self.pool
            .install(|| requests.par_iter().map(|req| bridge.process(req)).collect())
    }
}
