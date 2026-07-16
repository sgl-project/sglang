use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::runtime::Handle;
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

impl ResponseChunk {
    fn is_terminal(&self) -> bool {
        matches!(self, Self::Finished(_) | Self::Error(_))
    }
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

type BridgeStateRef = Arc<Mutex<BridgeState>>;

#[derive(Default)]
struct BridgeState {
    channels: HashMap<String, Sender<ResponseChunk>>,
    pending_sends: HashSet<String>,
    ready_callbacks: HashMap<String, PyObject>,
    ready_signals: HashSet<String>,
    terminal_errors: HashMap<String, TerminalError>,
}

#[derive(Debug, Clone)]
pub enum TerminalError {
    ChannelFull { rid: String },
    ClientDisconnected { rid: String },
    Aborted { rid: String },
}

impl TerminalError {
    pub fn message(&self) -> String {
        match self {
            Self::ChannelFull { rid } => {
                format!("gRPC response channel full for {rid}: client not consuming")
            }
            Self::ClientDisconnected { rid } => {
                format!("gRPC client disconnected for request {rid}")
            }
            Self::Aborted { rid } => format!("Request aborted: {rid}"),
        }
    }
}

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkSendStatus {
    Ready,
    Pending,
    Closed,
}

fn lock_or_recover<'a, T>(mutex: &'a Mutex<T>, name: &'static str) -> MutexGuard<'a, T> {
    mutex.lock().unwrap_or_else(|poisoned| {
        tracing::warn!(mutex = name, "Recovering from poisoned gRPC bridge mutex");
        poisoned.into_inner()
    })
}

/// Holds a reference to the Python RuntimeHandle and manages per-request channels.
pub struct PyBridge {
    runtime_handle: PyObject,
    state: BridgeStateRef,
    rust_tokenizer: Option<RustTokenizer>,
    context_len: i32,
    response_channel_capacity: usize,
    tokio_handle: Handle,
}

impl PyBridge {
    pub fn new(
        runtime_handle: PyObject,
        rust_tokenizer: Option<RustTokenizer>,
        context_len: i32,
        response_channel_capacity: usize,
        tokio_handle: Handle,
    ) -> Self {
        debug_assert!(
            response_channel_capacity > 0,
            "response_channel_capacity must be normalized by start_server"
        );
        Self {
            runtime_handle,
            state: Arc::new(Mutex::new(BridgeState::default())),
            rust_tokenizer,
            context_len,
            response_channel_capacity,
            tokio_handle,
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
        let mut state = lock_or_recover(self.state.as_ref(), "state");
        if state.channels.contains_key(rid) {
            return Err(PyRuntimeError::new_err(format!(
                "Duplicate active gRPC request id: {}",
                rid
            )));
        }
        state.channels.insert(rid.to_string(), sender);
        state.terminal_errors.remove(rid);
        state.ready_callbacks.remove(rid);
        state.ready_signals.remove(rid);
        state.pending_sends.remove(rid);
        Ok(receiver)
    }

    fn make_chunk_callback(&self, py: Python<'_>, rid: String) -> PyResult<PyObject> {
        let callback = ChunkCallback {
            rid,
            state: self.state.clone(),
            runtime_handle: self.runtime_handle.clone_ref(py),
            tokio_handle: self.tokio_handle.clone(),
        };
        let py_callback = Py::new(py, callback)?;
        Ok(py_callback.into_any())
    }

    fn make_json_callback(&self, py: Python<'_>, rid: String) -> PyResult<PyObject> {
        let callback = JsonChunkCallback {
            rid,
            state: self.state.clone(),
            runtime_handle: self.runtime_handle.clone_ref(py),
            tokio_handle: self.tokio_handle.clone(),
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
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let py_req_dict = json_map_to_pydict(py, &req_dict)?;
            let callback = self.make_chunk_callback(py, rid_owned)?;

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
        if !abort_all && rid.trim().is_empty() {
            return Err(PyValueError::new_err(
                "Abort requires a non-empty rid unless abort_all is true",
            ));
        }

        let should_call_python = if abort_all {
            let mut state = lock_or_recover(self.state.as_ref(), "state");
            let rids = state
                .channels
                .drain()
                .map(|(rid, _)| rid)
                .collect::<Vec<_>>();
            let affected = rids.len();
            state.pending_sends.clear();
            state.ready_callbacks.clear();
            state.ready_signals.clear();
            for channel_rid in rids {
                state.terminal_errors.insert(
                    channel_rid.clone(),
                    TerminalError::Aborted { rid: channel_rid },
                );
            }
            tracing::debug!(affected, "gRPC abort_all cleared active response channels");
            true
        } else {
            let mut state = lock_or_recover(self.state.as_ref(), "state");
            let was_active = remove_channel_refs_locked(&mut state, rid);
            if was_active {
                state
                    .terminal_errors
                    .insert(rid.to_string(), TerminalError::Aborted { rid: rid.into() });
            } else {
                tracing::debug!(rid, "Ignoring abort for inactive gRPC request id");
            }
            was_active
        };

        if !should_call_python {
            return Ok(());
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

    fn submit_json<F>(&self, rid: &str, call: F) -> PyResult<Receiver<ResponseChunk>>
    where
        F: for<'py> FnOnce(Python<'py>, &PyObject, PyObject) -> PyResult<()>,
    {
        // Closure args are: current Python token, RuntimeHandle, and the JSON chunk callback.
        let receiver = self.create_channel(rid)?;
        let rid_owned = rid.to_string();

        let result = Python::with_gil(|py| -> PyResult<()> {
            let callback = self.make_json_callback(py, rid_owned)?;
            call(py, &self.runtime_handle, callback)
        });

        match result {
            Ok(()) => Ok(receiver),
            Err(err) => {
                self.remove_channel(rid);
                Err(err)
            }
        }
    }

    pub fn submit_get_load(
        &self,
        rid: &str,
        dp_rank: Option<i32>,
    ) -> PyResult<Receiver<ResponseChunk>> {
        self.submit_json(rid, move |py, runtime_handle, callback| {
            runtime_handle.call_method1(py, "get_load", (callback, dp_rank))?;
            Ok(())
        })
    }

    pub fn submit_flush_cache(&self, rid: &str) -> PyResult<Receiver<ResponseChunk>> {
        self.submit_json(rid, |py, runtime_handle, callback| {
            runtime_handle.call_method1(py, "flush_cache", (callback,))?;
            Ok(())
        })
    }

    pub fn submit_pause_generation(
        &self,
        rid: &str,
        mode: &str,
    ) -> PyResult<Receiver<ResponseChunk>> {
        self.submit_json(rid, move |py, runtime_handle, callback| {
            runtime_handle.call_method1(py, "pause_generation", (mode, callback))?;
            Ok(())
        })
    }

    pub fn submit_continue_generation(&self, rid: &str) -> PyResult<Receiver<ResponseChunk>> {
        self.submit_json(rid, |py, runtime_handle, callback| {
            runtime_handle.call_method1(py, "continue_generation", (callback,))?;
            Ok(())
        })
    }

    pub fn submit_start_profile(
        &self,
        rid: &str,
        output_dir: Option<&str>,
    ) -> PyResult<Receiver<ResponseChunk>> {
        self.submit_json(rid, move |py, runtime_handle, callback| {
            runtime_handle.call_method1(py, "start_profile", (output_dir, callback))?;
            Ok(())
        })
    }

    pub fn submit_stop_profile(&self, rid: &str) -> PyResult<Receiver<ResponseChunk>> {
        self.submit_json(rid, |py, runtime_handle, callback| {
            runtime_handle.call_method1(py, "stop_profile", (callback,))?;
            Ok(())
        })
    }

    pub fn submit_update_weights(
        &self,
        rid: &str,
        model_path: &str,
        load_format: Option<&str>,
    ) -> PyResult<Receiver<ResponseChunk>> {
        self.submit_json(rid, move |py, runtime_handle, callback| {
            runtime_handle.call_method1(
                py,
                "update_weights_from_disk",
                (model_path, load_format, callback),
            )?;
            Ok(())
        })
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
        self.submit_json(rid, move |py, runtime_handle, callback| {
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

            kwargs.set_item("chunk_callback", callback)?;

            runtime_handle.call_method(py, method_name, (), Some(&kwargs))?;
            Ok(())
        })
    }

    pub fn remove_channel(&self, rid: &str) {
        let mut state = lock_or_recover(self.state.as_ref(), "state");
        remove_channel_refs_locked(&mut state, rid);
        state.terminal_errors.remove(rid);
    }

    pub fn take_terminal_error(&self, rid: &str) -> Option<TerminalError> {
        let mut state = lock_or_recover(self.state.as_ref(), "state");
        state.terminal_errors.remove(rid)
    }
}

fn close_channel_with_error(
    py: Python<'_>,
    rid: &str,
    state: &BridgeStateRef,
    runtime_handle: &PyObject,
    error: TerminalError,
) {
    let mut state = lock_or_recover(state.as_ref(), "state");
    remove_channel_refs_locked(&mut state, rid);
    state.terminal_errors.insert(rid.to_string(), error);
    drop(state);
    let _ = runtime_handle.call_method1(py, "abort", (rid, false));
}

fn remove_channel_refs_locked(state: &mut BridgeState, rid: &str) -> bool {
    let had_channel = state.channels.remove(rid).is_some();
    let had_pending = state.pending_sends.remove(rid);
    let had_callback = state.ready_callbacks.remove(rid).is_some();
    let had_signal = state.ready_signals.remove(rid);
    had_channel || had_pending || had_callback || had_signal
}

fn remove_channel_refs(rid: &str, state: &BridgeStateRef) {
    let mut state = lock_or_recover(state.as_ref(), "state");
    remove_channel_refs_locked(&mut state, rid);
}

fn register_pending_send(rid: &str, state: &BridgeStateRef) -> bool {
    let mut state = lock_or_recover(state.as_ref(), "state");
    state.pending_sends.insert(rid.to_string())
}

fn mark_send_ready(py: Python<'_>, rid: &str, state: &BridgeStateRef) -> Option<PyObject> {
    let mut state = lock_or_recover(state.as_ref(), "state");
    state.pending_sends.remove(rid);
    if let Some(callback) = state.ready_callbacks.get(rid) {
        Some(callback.clone_ref(py))
    } else {
        state.ready_signals.insert(rid.to_string());
        None
    }
}

fn notify_ready(py: Python<'_>, rid: &str, callback: PyObject) {
    if let Err(err) = callback.call0(py) {
        tracing::warn!(rid, "gRPC on_ready callback failed: {}", err);
    }
}

fn set_on_ready_for_rid(
    py: Python<'_>,
    rid: &str,
    state: &BridgeStateRef,
    on_ready: PyObject,
) -> PyResult<()> {
    let should_notify = {
        let mut state = lock_or_recover(state.as_ref(), "state");
        state
            .ready_callbacks
            .insert(rid.to_string(), on_ready.clone_ref(py));
        state.ready_signals.remove(rid)
    };
    if should_notify {
        on_ready.call0(py)?;
    }
    Ok(())
}

fn clear_on_ready_for_rid(rid: &str, state: &BridgeStateRef) {
    // End notifications for this rid. Do not call set_on_ready again for the same rid.
    let mut state = lock_or_recover(state.as_ref(), "state");
    state.ready_callbacks.remove(rid);
    state.ready_signals.remove(rid);
}

fn try_send_chunk(
    py: Python<'_>,
    rid: &str,
    state: &BridgeStateRef,
    runtime_handle: &PyObject,
    tokio_handle: &Handle,
    sender: &Sender<ResponseChunk>,
    msg: ResponseChunk,
) -> PyResult<ChunkSendStatus> {
    let terminal = msg.is_terminal();
    match sender.try_send(msg) {
        Ok(()) => {
            if terminal {
                remove_channel_refs(rid, state);
            }
            Ok(ChunkSendStatus::Ready)
        }
        Err(TrySendError::Full(msg)) => {
            if !register_pending_send(rid, state) {
                tracing::warn!(
                    rid,
                    "gRPC bridge received another chunk before the parked chunk drained; closing stream"
                );
                close_channel_with_error(
                    py,
                    rid,
                    state,
                    runtime_handle,
                    TerminalError::ChannelFull { rid: rid.into() },
                );
                return Ok(ChunkSendStatus::Closed);
            }

            let rid_owned = rid.to_string();
            let state = state.clone();
            let runtime_handle = runtime_handle.clone_ref(py);
            let sender = sender.clone();

            tokio_handle.spawn(async move {
                match sender.send(msg).await {
                    Ok(()) => {
                        if terminal {
                            // Terminal chunks end the producer contract; no further on_ready
                            // signal is fired after a parked Finished/Error drains.
                            remove_channel_refs(&rid_owned, &state);
                            return;
                        }

                        Python::with_gil(|py| {
                            if let Some(callback) = mark_send_ready(py, &rid_owned, &state) {
                                notify_ready(py, &rid_owned, callback);
                            }
                        });
                    }
                    Err(_) => {
                        Python::with_gil(|py| {
                            close_channel_with_error(
                                py,
                                &rid_owned,
                                &state,
                                &runtime_handle,
                                TerminalError::ClientDisconnected {
                                    rid: rid_owned.clone(),
                                },
                            );
                        });
                    }
                }
            });

            Ok(ChunkSendStatus::Pending)
        }
        Err(TrySendError::Closed(_)) => {
            close_channel_with_error(
                py,
                rid,
                state,
                runtime_handle,
                TerminalError::ClientDisconnected { rid: rid.into() },
            );
            Ok(ChunkSendStatus::Closed)
        }
    }
}

// Typed chunk callback for SGLang-native RPCs (dict-based chunks).
#[pyclass]
struct ChunkCallback {
    rid: String,
    state: BridgeStateRef,
    runtime_handle: PyObject,
    tokio_handle: Handle,
}

#[pymethods]
impl ChunkCallback {
    /// Register before producing chunks. If a parked chunk drained before registration,
    /// Rust fires `on_ready` immediately so late registration cannot miss the edge.
    fn set_on_ready(&self, py: Python<'_>, on_ready: PyObject) -> PyResult<()> {
        set_on_ready_for_rid(py, &self.rid, &self.state, on_ready)
    }

    fn clear_on_ready(&self) {
        clear_on_ready_for_rid(&self.rid, &self.state);
    }

    #[pyo3(signature = (chunk, finished=false, error=None))]
    fn __call__(
        &self,
        chunk: &Bound<'_, PyDict>,
        finished: bool,
        error: Option<String>,
    ) -> PyResult<ChunkSendStatus> {
        let py = chunk.py();
        let state = lock_or_recover(self.state.as_ref(), "state");
        let sender = match state.channels.get(&self.rid) {
            Some(s) => s.clone(),
            None => return Ok(ChunkSendStatus::Closed),
        };
        drop(state);

        if let Some(err_msg) = error {
            return try_send_chunk(
                py,
                &self.rid,
                &self.state,
                &self.runtime_handle,
                &self.tokio_handle,
                &sender,
                ResponseChunk::Error(err_msg),
            );
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
            &self.state,
            &self.runtime_handle,
            &self.tokio_handle,
            &sender,
            msg,
        )
    }
}

// JSON chunk callback for OpenAI pass-through RPCs (raw bytes).
#[pyclass]
struct JsonChunkCallback {
    rid: String,
    state: BridgeStateRef,
    runtime_handle: PyObject,
    tokio_handle: Handle,
}

#[pymethods]
impl JsonChunkCallback {
    /// Register before producing chunks. If a parked chunk drained before registration,
    /// Rust fires `on_ready` immediately so late registration cannot miss the edge.
    fn set_on_ready(&self, py: Python<'_>, on_ready: PyObject) -> PyResult<()> {
        set_on_ready_for_rid(py, &self.rid, &self.state, on_ready)
    }

    fn clear_on_ready(&self) {
        clear_on_ready_for_rid(&self.rid, &self.state);
    }

    #[pyo3(signature = (chunk_bytes, finished=false, error=None, status_code=None))]
    fn __call__(
        &self,
        chunk_bytes: &Bound<'_, pyo3::PyAny>,
        finished: bool,
        error: Option<String>,
        status_code: Option<i32>,
    ) -> PyResult<ChunkSendStatus> {
        let py = chunk_bytes.py();
        let state = lock_or_recover(self.state.as_ref(), "state");
        let sender = match state.channels.get(&self.rid) {
            Some(s) => s.clone(),
            None => return Ok(ChunkSendStatus::Closed),
        };
        drop(state);

        if let Some(err_msg) = error {
            return try_send_chunk(
                py,
                &self.rid,
                &self.state,
                &self.runtime_handle,
                &self.tokio_handle,
                &sender,
                ResponseChunk::Error(err_msg),
            );
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
            &self.state,
            &self.runtime_handle,
            &self.tokio_handle,
            &sender,
            msg,
        )
    }
}

fn extract_meta_info(chunk: &Bound<'_, PyDict>) -> HashMap<String, String> {
    let mut meta = HashMap::new();
    if let Ok(Some(meta_obj)) = chunk.get_item("meta_info")
        && let Ok(meta_dict) = meta_obj.downcast::<PyDict>()
    {
        for (k, v) in meta_dict.iter() {
            // The proto schema is map<string, string>; encode each Python value as JSON
            // so clients can recover numbers, booleans, arrays, and objects losslessly.
            if let Ok(key) = k.extract::<String>()
                && let Ok(val) = py_value_to_json_string(&v)
            {
                meta.insert(key, val);
            }
        }
    }
    meta
}

#[cfg(test)]
mod tests;
