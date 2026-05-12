use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::mpsc::{self, Receiver, Sender};

use crate::tokenizers::RustTokenizer;
use crate::utils::{json_map_to_pydict, py_value_to_json_string};

#[derive(Debug, Clone)]
pub enum ResponseChunk {
    Data(ResponseData),
    Finished(ResponseData),
    Error(String),
}

#[derive(Debug, Clone)]
pub struct ResponseData {
    pub text: Option<String>,
    pub output_ids: Option<Vec<i32>>,
    pub embedding: Option<Vec<f32>>,
    pub json_bytes: Option<Vec<u8>>,
    pub meta_info: HashMap<String, String>,
}

pub const DEFAULT_RESPONSE_CHANNEL_CAPACITY: usize = 64;

fn lock_or_recover<'a, T>(mutex: &'a Mutex<T>, name: &'static str) -> MutexGuard<'a, T> {
    mutex.lock().unwrap_or_else(|poisoned| {
        tracing::warn!(mutex = name, "Recovering from poisoned gRPC bridge mutex");
        poisoned.into_inner()
    })
}

/// Holds a reference to the Python RuntimeHandle and manages per-request channels.
pub struct PyBridge {
    runtime_handle: PyObject,
    channels: Arc<Mutex<HashMap<String, Sender<ResponseChunk>>>>,
    terminal_errors: Arc<Mutex<HashMap<String, String>>>,
    rust_tokenizer: Option<RustTokenizer>,
    context_len: i32,
    response_channel_capacity: usize,
}

impl PyBridge {
    pub fn new(
        runtime_handle: PyObject,
        rust_tokenizer: Option<RustTokenizer>,
        context_len: i32,
        response_channel_capacity: usize,
    ) -> Self {
        Self {
            runtime_handle,
            channels: Arc::new(Mutex::new(HashMap::new())),
            terminal_errors: Arc::new(Mutex::new(HashMap::new())),
            rust_tokenizer,
            context_len,
            response_channel_capacity: response_channel_capacity.max(1),
        }
    }

    /// Access the Rust tokenizer (if available).
    pub fn rust_tokenizer(&self) -> Option<&RustTokenizer> {
        self.rust_tokenizer.as_ref()
    }

    /// Return the model's context length.
    pub fn context_len(&self) -> i32 {
        self.context_len
    }

    // ------------------------------------------------------------------
    // Channel + callback helpers
    // ------------------------------------------------------------------

    fn create_channel(&self, rid: &str) -> PyResult<Receiver<ResponseChunk>> {
        let (sender, receiver) = mpsc::channel(self.response_channel_capacity);
        {
            let mut channels = lock_or_recover(self.channels.as_ref(), "channels");
            if channels.contains_key(rid) {
                return Err(PyRuntimeError::new_err(format!(
                    "Duplicate active gRPC request id: {}",
                    rid
                )));
            }
            channels.insert(rid.to_string(), sender);
        }
        {
            let mut terminal_errors =
                lock_or_recover(self.terminal_errors.as_ref(), "terminal_errors");
            terminal_errors.remove(rid);
        }
        Ok(receiver)
    }

    fn make_chunk_callback(
        &self,
        py: Python<'_>,
        rid: String,
        channels: Arc<Mutex<HashMap<String, Sender<ResponseChunk>>>>,
        terminal_errors: Arc<Mutex<HashMap<String, String>>>,
        runtime_handle: PyObject,
    ) -> PyResult<PyObject> {
        let callback = ChunkCallback {
            rid,
            channels,
            terminal_errors,
            runtime_handle,
        };
        let py_callback = Py::new(py, callback)?;
        Ok(py_callback.into_any())
    }

    fn make_json_callback(
        &self,
        py: Python<'_>,
        rid: String,
        channels: Arc<Mutex<HashMap<String, Sender<ResponseChunk>>>>,
        terminal_errors: Arc<Mutex<HashMap<String, String>>>,
        runtime_handle: PyObject,
    ) -> PyResult<PyObject> {
        let callback = JsonChunkCallback {
            rid,
            channels,
            terminal_errors,
            runtime_handle,
        };
        let py_callback = Py::new(py, callback)?;
        Ok(py_callback.into_any())
    }

    // ------------------------------------------------------------------
    // Consolidated request submission (generate / embed / classify)
    // ------------------------------------------------------------------

    /// Submit a generate or embed request by passing a pre-built dict to Python.
    ///
    /// `req_type` is "generate", "embed", or "classify".
    /// `req_dict` contains fields matching GenerateReqInput or EmbeddingReqInput.
    pub fn submit_request(
        &self,
        rid: &str,
        req_type: &str,
        req_dict: HashMap<String, serde_json::Value>,
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid)?;
        let channels_ref = self.channels.clone();
        let terminal_errors_ref = self.terminal_errors.clone();
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let py_req_dict = json_map_to_pydict(py, &req_dict)?;
            let callback = self.make_chunk_callback(
                py,
                rid_owned,
                channels_ref,
                terminal_errors_ref,
                self.runtime_handle.clone_ref(py),
            )?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("req_type", req_type)?;
            kwargs.set_item("req_dict", py_req_dict)?;
            kwargs.set_item("chunk_callback", callback)?;

            self.runtime_handle
                .call_method(py, "submit_request", (), Some(&kwargs))?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    // ------------------------------------------------------------------
    // Abort
    // ------------------------------------------------------------------

    pub fn abort(&self, rid: &str, abort_all: bool) -> PyResult<()> {
        if abort_all {
            let rids = {
                let channels = lock_or_recover(self.channels.as_ref(), "channels");
                channels.keys().cloned().collect::<Vec<_>>()
            };
            {
                let mut channels = lock_or_recover(self.channels.as_ref(), "channels");
                channels.clear();
            }
            let mut terminal_errors =
                lock_or_recover(self.terminal_errors.as_ref(), "terminal_errors");
            for channel_rid in rids {
                terminal_errors.insert(channel_rid, "Request aborted".to_string());
            }
        } else {
            {
                let mut channels = lock_or_recover(self.channels.as_ref(), "channels");
                channels.remove(rid);
            }
            let mut terminal_errors =
                lock_or_recover(self.terminal_errors.as_ref(), "terminal_errors");
            terminal_errors.insert(rid.to_string(), "Request aborted".to_string());
        }
        Python::with_gil(|py| {
            self.runtime_handle
                .call_method1(py, "abort", (rid, abort_all))?;
            Ok(())
        })
    }

    // ------------------------------------------------------------------
    // Info / control RPCs (synchronous, small data)
    // ------------------------------------------------------------------

    pub fn get_model_info(&self) -> PyResult<String> {
        Python::with_gil(|py| {
            let result = self.runtime_handle.call_method0(py, "get_model_info")?;
            result.extract::<String>(py)
        })
    }

    pub fn get_server_info(&self) -> PyResult<String> {
        Python::with_gil(|py| {
            let result = self.runtime_handle.call_method0(py, "get_server_info")?;
            result.extract::<String>(py)
        })
    }

    pub fn health_check(&self) -> PyResult<bool> {
        Python::with_gil(|py| {
            let result = self.runtime_handle.call_method0(py, "health_check")?;
            result.extract::<bool>(py)
        })
    }

    /// Tokenize via Python (fallback when Rust tokenizer unavailable).
    pub fn tokenize_py(&self, text: &str, add_special_tokens: bool) -> PyResult<String> {
        Python::with_gil(|py| {
            let result =
                self.runtime_handle
                    .call_method1(py, "tokenize", (text, add_special_tokens))?;
            result.extract::<String>(py)
        })
    }

    /// Detokenize via Python (fallback when Rust tokenizer unavailable).
    pub fn detokenize_py(&self, tokens: Vec<i32>) -> PyResult<String> {
        Python::with_gil(|py| {
            let result = self
                .runtime_handle
                .call_method1(py, "detokenize", (tokens,))?;
            result.extract::<String>(py)
        })
    }

    pub fn list_models(&self) -> PyResult<String> {
        Python::with_gil(|py| {
            let result = self.runtime_handle.call_method0(py, "list_models")?;
            result.extract::<String>(py)
        })
    }

    pub fn submit_get_load(
        &self,
        rid: &str,
        dp_rank: Option<i32>,
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid)?;
        let channels_ref = self.channels.clone();
        let terminal_errors_ref = self.terminal_errors.clone();
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let callback = self.make_json_callback(
                py,
                rid_owned,
                channels_ref,
                terminal_errors_ref,
                self.runtime_handle.clone_ref(py),
            )?;
            self.runtime_handle
                .call_method1(py, "get_load", (callback, dp_rank))?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    pub fn submit_flush_cache(&self, rid: &str) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid)?;
        let channels_ref = self.channels.clone();
        let terminal_errors_ref = self.terminal_errors.clone();
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let callback = self.make_json_callback(
                py,
                rid_owned,
                channels_ref,
                terminal_errors_ref,
                self.runtime_handle.clone_ref(py),
            )?;
            self.runtime_handle
                .call_method1(py, "flush_cache", (callback,))?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    pub fn submit_pause_generation(
        &self,
        rid: &str,
        mode: &str,
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid)?;
        let channels_ref = self.channels.clone();
        let terminal_errors_ref = self.terminal_errors.clone();
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let callback = self.make_json_callback(
                py,
                rid_owned,
                channels_ref,
                terminal_errors_ref,
                self.runtime_handle.clone_ref(py),
            )?;
            self.runtime_handle
                .call_method1(py, "pause_generation", (mode, callback))?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    pub fn submit_continue_generation(&self, rid: &str) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid)?;
        let channels_ref = self.channels.clone();
        let terminal_errors_ref = self.terminal_errors.clone();
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let callback = self.make_json_callback(
                py,
                rid_owned,
                channels_ref,
                terminal_errors_ref,
                self.runtime_handle.clone_ref(py),
            )?;
            self.runtime_handle
                .call_method1(py, "continue_generation", (callback,))?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    pub fn submit_start_profile(
        &self,
        rid: &str,
        output_dir: Option<&str>,
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid)?;
        let channels_ref = self.channels.clone();
        let terminal_errors_ref = self.terminal_errors.clone();
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let callback = self.make_json_callback(
                py,
                rid_owned,
                channels_ref,
                terminal_errors_ref,
                self.runtime_handle.clone_ref(py),
            )?;
            self.runtime_handle
                .call_method1(py, "start_profile", (output_dir, callback))?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    pub fn submit_stop_profile(&self, rid: &str) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid)?;
        let channels_ref = self.channels.clone();
        let terminal_errors_ref = self.terminal_errors.clone();
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let callback = self.make_json_callback(
                py,
                rid_owned,
                channels_ref,
                terminal_errors_ref,
                self.runtime_handle.clone_ref(py),
            )?;
            self.runtime_handle
                .call_method1(py, "stop_profile", (callback,))?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    pub fn submit_update_weights(
        &self,
        rid: &str,
        model_path: &str,
        load_format: Option<&str>,
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid)?;
        let channels_ref = self.channels.clone();
        let terminal_errors_ref = self.terminal_errors.clone();
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let callback = self.make_json_callback(
                py,
                rid_owned,
                channels_ref,
                terminal_errors_ref,
                self.runtime_handle.clone_ref(py),
            )?;
            self.runtime_handle.call_method1(
                py,
                "update_weights_from_disk",
                (model_path, load_format, callback),
            )?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    // ------------------------------------------------------------------
    // OpenAI pass-through RPCs
    // ------------------------------------------------------------------

    pub fn submit_openai(
        &self,
        rid: &str,
        method_name: &str,
        json_body: &[u8],
        trace_headers: &HashMap<String, String>,
    ) -> PyResult<Receiver<ResponseChunk>> {
        let receiver = self.create_channel(rid)?;
        let channels_ref = self.channels.clone();
        let terminal_errors_ref = self.terminal_errors.clone();
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let kwargs = PyDict::new(py);
            let py_bytes = PyBytes::new(py, json_body);
            kwargs.set_item("json_body", py_bytes)?;
            if !trace_headers.is_empty() {
                let py_trace_headers = PyDict::new(py);
                for (key, value) in trace_headers {
                    py_trace_headers.set_item(key, value)?;
                }
                kwargs.set_item("trace_headers", py_trace_headers)?;
            }

            let callback = self.make_json_callback(
                py,
                rid_owned,
                channels_ref,
                terminal_errors_ref,
                self.runtime_handle.clone_ref(py),
            )?;
            kwargs.set_item("chunk_callback", callback)?;

            self.runtime_handle
                .call_method(py, method_name, (), Some(&kwargs))?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    pub fn remove_channel(&self, rid: &str) {
        {
            let mut channels = lock_or_recover(self.channels.as_ref(), "channels");
            channels.remove(rid);
        }
        {
            let mut terminal_errors =
                lock_or_recover(self.terminal_errors.as_ref(), "terminal_errors");
            terminal_errors.remove(rid);
        }
    }

    pub fn take_terminal_error(&self, rid: &str) -> Option<String> {
        let mut terminal_errors = lock_or_recover(self.terminal_errors.as_ref(), "terminal_errors");
        terminal_errors.remove(rid)
    }
}

fn close_channel_with_error(
    py: Python<'_>,
    rid: &str,
    channels: &Arc<Mutex<HashMap<String, Sender<ResponseChunk>>>>,
    terminal_errors: &Arc<Mutex<HashMap<String, String>>>,
    runtime_handle: &PyObject,
    error: &str,
) {
    {
        let mut channels = lock_or_recover(channels.as_ref(), "channels");
        channels.remove(rid);
    }
    {
        let mut terminal_errors = lock_or_recover(terminal_errors.as_ref(), "terminal_errors");
        terminal_errors.insert(rid.to_string(), error.to_string());
    }
    let _ = runtime_handle.call_method1(py, "abort", (rid, false));
}

fn try_send_chunk(
    py: Python<'_>,
    rid: &str,
    channels: &Arc<Mutex<HashMap<String, Sender<ResponseChunk>>>>,
    terminal_errors: &Arc<Mutex<HashMap<String, String>>>,
    runtime_handle: &PyObject,
    sender: &Sender<ResponseChunk>,
    msg: ResponseChunk,
) -> PyResult<()> {
    match sender.try_send(msg) {
        Ok(()) => Ok(()),
        Err(TrySendError::Full(msg)) => {
            match py.allow_threads(|| sender.blocking_send(msg).map_err(|_| ())) {
                Ok(()) => Ok(()),
                Err(_) => {
                    close_channel_with_error(
                        py,
                        rid,
                        channels,
                        terminal_errors,
                        runtime_handle,
                        "gRPC client disconnected",
                    );
                    Ok(())
                }
            }
        }
        Err(TrySendError::Closed(_)) => {
            close_channel_with_error(
                py,
                rid,
                channels,
                terminal_errors,
                runtime_handle,
                "gRPC client disconnected",
            );
            Ok(())
        }
    }
}

// ======================================================================
// Typed chunk callback (for SGLang-native RPCs: dict-based chunks)
// ======================================================================

#[pyclass]
struct ChunkCallback {
    rid: String,
    channels: Arc<Mutex<HashMap<String, Sender<ResponseChunk>>>>,
    terminal_errors: Arc<Mutex<HashMap<String, String>>>,
    runtime_handle: PyObject,
}

#[pymethods]
impl ChunkCallback {
    #[pyo3(signature = (chunk, finished=false, error=None))]
    fn __call__(
        &self,
        chunk: &Bound<'_, PyDict>,
        finished: bool,
        error: Option<String>,
    ) -> PyResult<()> {
        let py = chunk.py();
        let channels = lock_or_recover(self.channels.as_ref(), "channels");
        let sender = match channels.get(&self.rid) {
            Some(s) => s.clone(),
            None => return Ok(()),
        };
        drop(channels);

        if let Some(err_msg) = error {
            try_send_chunk(
                py,
                &self.rid,
                &self.channels,
                &self.terminal_errors,
                &self.runtime_handle,
                &sender,
                ResponseChunk::Error(err_msg),
            )?;
            let mut channels = lock_or_recover(self.channels.as_ref(), "channels");
            channels.remove(&self.rid);
            return Ok(());
        }

        let text: Option<String> = chunk
            .get_item("text")?
            .and_then(|v| v.extract::<String>().ok());

        let output_ids: Option<Vec<i32>> = chunk
            .get_item("output_ids")?
            .and_then(|v| v.extract::<Vec<i32>>().ok());

        let embedding: Option<Vec<f32>> = chunk
            .get_item("embedding")?
            .and_then(|v| v.extract::<Vec<f32>>().ok());

        let meta_info = extract_meta_info(chunk);

        let data = ResponseData {
            text,
            output_ids,
            embedding,
            json_bytes: None,
            meta_info,
        };

        let msg = if finished {
            ResponseChunk::Finished(data)
        } else {
            ResponseChunk::Data(data)
        };

        try_send_chunk(
            py,
            &self.rid,
            &self.channels,
            &self.terminal_errors,
            &self.runtime_handle,
            &sender,
            msg,
        )?;

        if finished {
            let mut channels = lock_or_recover(self.channels.as_ref(), "channels");
            channels.remove(&self.rid);
        }

        Ok(())
    }
}

// ======================================================================
// JSON chunk callback (for OpenAI pass-through RPCs: raw bytes)
// ======================================================================

#[pyclass]
struct JsonChunkCallback {
    rid: String,
    channels: Arc<Mutex<HashMap<String, Sender<ResponseChunk>>>>,
    terminal_errors: Arc<Mutex<HashMap<String, String>>>,
    runtime_handle: PyObject,
}

#[pymethods]
impl JsonChunkCallback {
    #[pyo3(signature = (chunk_bytes, finished=false, error=None, status_code=None))]
    fn __call__(
        &self,
        chunk_bytes: &Bound<'_, pyo3::PyAny>,
        finished: bool,
        error: Option<String>,
        status_code: Option<i32>,
    ) -> PyResult<()> {
        let py = chunk_bytes.py();
        let channels = lock_or_recover(self.channels.as_ref(), "channels");
        let sender = match channels.get(&self.rid) {
            Some(s) => s.clone(),
            None => return Ok(()),
        };
        drop(channels);

        if let Some(err_msg) = error {
            try_send_chunk(
                py,
                &self.rid,
                &self.channels,
                &self.terminal_errors,
                &self.runtime_handle,
                &sender,
                ResponseChunk::Error(err_msg),
            )?;
            let mut channels = lock_or_recover(self.channels.as_ref(), "channels");
            channels.remove(&self.rid);
            return Ok(());
        }

        let bytes_data: Vec<u8> = if let Ok(b) = chunk_bytes.extract::<Vec<u8>>() {
            b
        } else if let Ok(s) = chunk_bytes.extract::<String>() {
            s.into_bytes()
        } else {
            vec![]
        };

        let mut meta_info = HashMap::new();
        if let Some(code) = status_code {
            meta_info.insert("status_code".to_string(), code.to_string());
        }

        let data = ResponseData {
            text: None,
            output_ids: None,
            embedding: None,
            json_bytes: Some(bytes_data),
            meta_info,
        };

        let msg = if finished {
            ResponseChunk::Finished(data)
        } else {
            ResponseChunk::Data(data)
        };

        try_send_chunk(
            py,
            &self.rid,
            &self.channels,
            &self.terminal_errors,
            &self.runtime_handle,
            &sender,
            msg,
        )?;

        if finished {
            let mut channels = lock_or_recover(self.channels.as_ref(), "channels");
            channels.remove(&self.rid);
        }

        Ok(())
    }
}

fn extract_meta_info(chunk: &Bound<'_, PyDict>) -> HashMap<String, String> {
    let mut meta = HashMap::new();
    if let Ok(Some(meta_obj)) = chunk.get_item("meta_info")
        && let Ok(meta_dict) = meta_obj.downcast::<PyDict>()
    {
        for (k, v) in meta_dict.iter() {
            if let Ok(key) = k.extract::<String>()
                && let Ok(val) = py_value_to_json_string(&v)
            {
                meta.insert(key, val);
            }
        }
    }
    meta
}
