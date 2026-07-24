use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::runtime::Handle;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::mpsc::{self, Receiver, Sender};

use crate::tokenizers::RustTokenizer;
use crate::utils::{json_map_to_pydict, py_value_to_json_string, py_value_to_json_value};

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
    pub choice_index: i32,
    pub json_bytes: Option<Vec<u8>>,
    pub meta_info: ResponseMetadata,
}

#[derive(Debug, Clone)]
pub enum ResponseMetadata {
    Legacy(HashMap<String, String>),
    Typed(serde_json::Map<String, serde_json::Value>),
}

impl ResponseMetadata {
    pub fn into_legacy(self) -> Result<HashMap<String, String>, &'static str> {
        match self {
            Self::Legacy(meta) => Ok(meta),
            Self::Typed(_) => Err("expected legacy string metadata"),
        }
    }

    pub fn as_typed(&self) -> Result<&serde_json::Map<String, serde_json::Value>, &'static str> {
        match self {
            Self::Typed(meta) => Ok(meta),
            Self::Legacy(_) => Err("expected typed JSON metadata"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseMetadataMode {
    LegacyStringMap,
    TypedGenerate,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RequestKey {
    rid: String,
    incarnation: u64,
}

impl RequestKey {
    pub fn rid(&self) -> &str {
        &self.rid
    }
}

pub struct SubmittedRequest {
    pub key: RequestKey,
    pub receiver: Receiver<ResponseChunk>,
}

pub const DEFAULT_RESPONSE_CHANNEL_CAPACITY: usize = 64;

type BridgeStateRef = Arc<Mutex<BridgeState>>;

#[derive(Default)]
struct BridgeState {
    channels: HashMap<String, ActiveChannel>,
    abort_all_in_progress: bool,
    pending_sends: HashSet<RequestKey>,
    ready_callbacks: HashMap<RequestKey, Py<PyAny>>,
    ready_signals: HashSet<RequestKey>,
    terminal_errors: HashMap<RequestKey, TerminalError>,
}

struct ActiveChannel {
    incarnation: u64,
    sender: Sender<ResponseChunk>,
    metadata_mode: ResponseMetadataMode,
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

// skip_from_py_object: this enum is only returned to Python, never received
// from it, so it opts out of pyo3's (deprecated-by-default) FromPyObject
// derive for Clone pyclasses.
#[pyclass(eq, eq_int, skip_from_py_object)]
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
    runtime_handle: Py<PyAny>,
    state: BridgeStateRef,
    rust_tokenizer: Option<RustTokenizer>,
    context_len: i32,
    response_channel_capacity: usize,
    tokio_handle: Handle,
    next_incarnation: AtomicU64,
}

impl PyBridge {
    pub fn new(
        runtime_handle: Py<PyAny>,
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
            next_incarnation: AtomicU64::new(1),
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

    fn create_channel(
        &self,
        rid: &str,
        metadata_mode: ResponseMetadataMode,
    ) -> PyResult<SubmittedRequest> {
        let (sender, receiver) = mpsc::channel(self.response_channel_capacity);
        let mut state = lock_or_recover(self.state.as_ref(), "state");
        if state.abort_all_in_progress {
            return Err(PyRuntimeError::new_err(
                "Cannot submit a gRPC request while abort_all is in progress",
            ));
        }
        if state.channels.contains_key(rid) {
            return Err(PyRuntimeError::new_err(format!(
                "Duplicate active gRPC request id: {}",
                rid
            )));
        }
        let key = RequestKey {
            rid: rid.to_string(),
            incarnation: self.next_incarnation.fetch_add(1, Ordering::Relaxed),
        };
        state.channels.insert(
            rid.to_string(),
            ActiveChannel {
                incarnation: key.incarnation,
                sender,
                metadata_mode,
            },
        );
        Ok(SubmittedRequest { key, receiver })
    }

    fn make_chunk_callback(
        &self,
        py: Python<'_>,
        key: RequestKey,
        metadata_mode: ResponseMetadataMode,
    ) -> PyResult<Py<PyAny>> {
        let callback = ChunkCallback {
            key,
            metadata_mode,
            state: self.state.clone(),
            runtime_handle: self.runtime_handle.clone_ref(py),
            tokio_handle: self.tokio_handle.clone(),
        };
        let py_callback = Py::new(py, callback)?;
        Ok(py_callback.into_any())
    }

    fn make_json_callback(&self, py: Python<'_>, key: RequestKey) -> PyResult<Py<PyAny>> {
        let callback = JsonChunkCallback {
            key,
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
        metadata_mode: ResponseMetadataMode,
    ) -> PyResult<SubmittedRequest> {
        let submitted = self.create_channel(rid, metadata_mode)?;

        let result = Python::attach(|py| -> PyResult<()> {
            let py_req_dict = json_map_to_pydict(py, &req_dict)?;
            let callback = self.make_chunk_callback(py, submitted.key.clone(), metadata_mode)?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("req_type", req_type)?;
            kwargs.set_item("req_dict", py_req_dict)?;
            kwargs.set_item("chunk_callback", callback)?;
            kwargs.set_item(
                "typed_generation",
                metadata_mode == ResponseMetadataMode::TypedGenerate,
            )?;

            self.runtime_handle
                .call_method(py, "submit_request", (), Some(&kwargs))?;
            Ok(())
        });

        match result {
            Ok(()) => Ok(submitted),
            Err(err) => {
                self.remove_channel(&submitted.key);
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

        let keys = if abort_all {
            let mut state = lock_or_recover(self.state.as_ref(), "state");
            state.abort_all_in_progress = true;
            state
                .channels
                .iter()
                .map(|(rid, channel)| RequestKey {
                    rid: rid.clone(),
                    incarnation: channel.incarnation,
                })
                .collect::<Vec<_>>()
        } else {
            let state = lock_or_recover(self.state.as_ref(), "state");
            state.channels.get(rid).map_or_else(Vec::new, |channel| {
                vec![RequestKey {
                    rid: rid.to_string(),
                    incarnation: channel.incarnation,
                }]
            })
        };

        if !abort_all && keys.is_empty() {
            tracing::debug!(rid, "Ignoring abort for inactive gRPC request id");
            return Ok(());
        }

        let call_result = Python::attach(|py| {
            self.runtime_handle
                .call_method1(py, "abort", (rid, abort_all))?;
            Ok(())
        });

        let mut state = lock_or_recover(self.state.as_ref(), "state");
        for key in &keys {
            finalize_explicit_abort_locked(&mut state, key);
        }
        if abort_all {
            state.abort_all_in_progress = false;
            tracing::debug!(
                affected = keys.len(),
                "gRPC abort_all cleared active response channels"
            );
        }
        call_result
    }

    pub fn abort_request(&self, key: &RequestKey) -> PyResult<()> {
        let is_active = {
            let state = lock_or_recover(self.state.as_ref(), "state");
            state
                .channels
                .get(key.rid())
                .is_some_and(|channel| channel.incarnation == key.incarnation)
        };
        if !is_active {
            let mut state = lock_or_recover(self.state.as_ref(), "state");
            remove_auxiliary_refs_locked(&mut state, key);
            return Ok(());
        }

        let call_result = Python::attach(|py| {
            self.runtime_handle
                .call_method1(py, "abort", (key.rid(), false))?;
            Ok(())
        });
        let mut state = lock_or_recover(self.state.as_ref(), "state");
        if remove_channel_refs_locked(&mut state, key) {
            state.terminal_errors.insert(
                key.clone(),
                TerminalError::Aborted {
                    rid: key.rid.clone(),
                },
            );
        }
        call_result
    }

    // ------------------------------------------------------------------
    // Info / control RPCs (synchronous, small data)
    // ------------------------------------------------------------------

    pub fn get_model_info(&self) -> PyResult<String> {
        Python::attach(|py| {
            let result = self.runtime_handle.call_method0(py, "get_model_info")?;
            result.extract::<String>(py)
        })
    }

    pub fn get_server_info(&self) -> PyResult<String> {
        Python::attach(|py| {
            let result = self.runtime_handle.call_method0(py, "get_server_info")?;
            result.extract::<String>(py)
        })
    }

    pub fn health_check(&self) -> PyResult<bool> {
        Python::attach(|py| {
            let result = self.runtime_handle.call_method0(py, "health_check")?;
            result.extract::<bool>(py)
        })
    }

    /// Tokenize via Python (fallback when Rust tokenizer unavailable).
    pub fn tokenize_py(&self, text: &str, add_special_tokens: bool) -> PyResult<String> {
        Python::attach(|py| {
            let result =
                self.runtime_handle
                    .call_method1(py, "tokenize", (text, add_special_tokens))?;
            result.extract::<String>(py)
        })
    }

    /// Detokenize via Python (fallback when Rust tokenizer unavailable).
    pub fn detokenize_py(&self, tokens: Vec<i32>) -> PyResult<String> {
        Python::attach(|py| {
            let result = self
                .runtime_handle
                .call_method1(py, "detokenize", (tokens,))?;
            result.extract::<String>(py)
        })
    }

    pub fn list_models(&self) -> PyResult<String> {
        Python::attach(|py| {
            let result = self.runtime_handle.call_method0(py, "list_models")?;
            result.extract::<String>(py)
        })
    }

    fn submit_json<F>(&self, rid: &str, call: F) -> PyResult<SubmittedRequest>
    where
        F: for<'py> FnOnce(Python<'py>, &Py<PyAny>, Py<PyAny>) -> PyResult<()>,
    {
        // Closure args are: current Python token, RuntimeHandle, and the JSON chunk callback.
        let submitted = self.create_channel(rid, ResponseMetadataMode::LegacyStringMap)?;

        let result = Python::attach(|py| -> PyResult<()> {
            let callback = self.make_json_callback(py, submitted.key.clone())?;
            call(py, &self.runtime_handle, callback)
        });

        match result {
            Ok(()) => Ok(submitted),
            Err(err) => {
                self.remove_channel(&submitted.key);
                Err(err)
            }
        }
    }

    pub fn submit_get_load(&self, rid: &str, dp_rank: Option<i32>) -> PyResult<SubmittedRequest> {
        self.submit_json(rid, move |py, runtime_handle, callback| {
            runtime_handle.call_method1(py, "get_load", (callback, dp_rank))?;
            Ok(())
        })
    }

    pub fn submit_flush_cache(&self, rid: &str) -> PyResult<SubmittedRequest> {
        self.submit_json(rid, |py, runtime_handle, callback| {
            runtime_handle.call_method1(py, "flush_cache", (callback,))?;
            Ok(())
        })
    }

    pub fn submit_pause_generation(&self, rid: &str, mode: &str) -> PyResult<SubmittedRequest> {
        self.submit_json(rid, move |py, runtime_handle, callback| {
            runtime_handle.call_method1(py, "pause_generation", (mode, callback))?;
            Ok(())
        })
    }

    pub fn submit_continue_generation(&self, rid: &str) -> PyResult<SubmittedRequest> {
        self.submit_json(rid, |py, runtime_handle, callback| {
            runtime_handle.call_method1(py, "continue_generation", (callback,))?;
            Ok(())
        })
    }

    pub fn submit_start_profile(
        &self,
        rid: &str,
        output_dir: Option<&str>,
    ) -> PyResult<SubmittedRequest> {
        self.submit_json(rid, move |py, runtime_handle, callback| {
            runtime_handle.call_method1(py, "start_profile", (output_dir, callback))?;
            Ok(())
        })
    }

    pub fn submit_stop_profile(&self, rid: &str) -> PyResult<SubmittedRequest> {
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
    ) -> PyResult<SubmittedRequest> {
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
    ) -> PyResult<SubmittedRequest> {
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

    pub fn remove_channel(&self, key: &RequestKey) {
        let mut state = lock_or_recover(self.state.as_ref(), "state");
        remove_channel_refs_locked(&mut state, key);
        state.terminal_errors.remove(key);
    }

    pub fn take_terminal_error(&self, key: &RequestKey) -> Option<TerminalError> {
        let mut state = lock_or_recover(self.state.as_ref(), "state");
        state.terminal_errors.remove(key)
    }
}

fn close_channel_with_error(
    py: Python<'_>,
    key: &RequestKey,
    state: &BridgeStateRef,
    runtime_handle: &Py<PyAny>,
    error: TerminalError,
) {
    let is_active = {
        let state = lock_or_recover(state.as_ref(), "state");
        state
            .channels
            .get(key.rid())
            .is_some_and(|channel| channel.incarnation == key.incarnation)
    };
    if !is_active {
        return;
    }
    let _ = runtime_handle.call_method1(py, "abort", (key.rid(), false));
    let mut state = lock_or_recover(state.as_ref(), "state");
    if remove_channel_refs_locked(&mut state, key) {
        state.terminal_errors.insert(key.clone(), error);
    }
}

fn remove_auxiliary_refs_locked(state: &mut BridgeState, key: &RequestKey) -> bool {
    let had_pending = state.pending_sends.remove(key);
    let had_callback = state.ready_callbacks.remove(key).is_some();
    let had_signal = state.ready_signals.remove(key);
    had_pending || had_callback || had_signal
}

fn remove_channel_refs_locked(state: &mut BridgeState, key: &RequestKey) -> bool {
    let is_active = state
        .channels
        .get(key.rid())
        .is_some_and(|channel| channel.incarnation == key.incarnation);
    if is_active {
        state.channels.remove(key.rid());
    }
    remove_auxiliary_refs_locked(state, key);
    is_active
}

fn finalize_explicit_abort_locked(state: &mut BridgeState, key: &RequestKey) {
    let expects_typed_terminals = state.channels.get(key.rid()).is_some_and(|channel| {
        channel.incarnation == key.incarnation
            && channel.metadata_mode == ResponseMetadataMode::TypedGenerate
    });
    if expects_typed_terminals {
        return;
    }
    if remove_channel_refs_locked(state, key) {
        state.terminal_errors.insert(
            key.clone(),
            TerminalError::Aborted {
                rid: key.rid.clone(),
            },
        );
    }
}

fn remove_channel_refs(key: &RequestKey, state: &BridgeStateRef) {
    let mut state = lock_or_recover(state.as_ref(), "state");
    remove_channel_refs_locked(&mut state, key);
}

fn register_pending_send(key: &RequestKey, state: &BridgeStateRef) -> bool {
    let mut state = lock_or_recover(state.as_ref(), "state");
    state.pending_sends.insert(key.clone())
}

fn mark_send_ready(py: Python<'_>, key: &RequestKey, state: &BridgeStateRef) -> Option<Py<PyAny>> {
    let mut state = lock_or_recover(state.as_ref(), "state");
    state.pending_sends.remove(key);
    if let Some(callback) = state.ready_callbacks.get(key) {
        Some(callback.clone_ref(py))
    } else {
        state.ready_signals.insert(key.clone());
        None
    }
}

fn notify_ready(py: Python<'_>, rid: &str, callback: Py<PyAny>) {
    if let Err(err) = callback.call0(py) {
        tracing::warn!(rid, "gRPC on_ready callback failed: {}", err);
    }
}

fn set_on_ready_for_rid(
    py: Python<'_>,
    key: &RequestKey,
    state: &BridgeStateRef,
    on_ready: Py<PyAny>,
) -> PyResult<()> {
    let should_notify = {
        let mut state = lock_or_recover(state.as_ref(), "state");
        if state
            .channels
            .get(key.rid())
            .is_none_or(|channel| channel.incarnation != key.incarnation)
        {
            return Ok(());
        }
        state
            .ready_callbacks
            .insert(key.clone(), on_ready.clone_ref(py));
        state.ready_signals.remove(key)
    };
    if should_notify {
        on_ready.call0(py)?;
    }
    Ok(())
}

fn clear_on_ready_for_rid(key: &RequestKey, state: &BridgeStateRef) {
    // End notifications for this rid. Do not call set_on_ready again for the same rid.
    let mut state = lock_or_recover(state.as_ref(), "state");
    state.ready_callbacks.remove(key);
    state.ready_signals.remove(key);
}

fn try_send_chunk(
    py: Python<'_>,
    key: &RequestKey,
    state: &BridgeStateRef,
    runtime_handle: &Py<PyAny>,
    tokio_handle: &Handle,
    sender: &Sender<ResponseChunk>,
    msg: ResponseChunk,
) -> PyResult<ChunkSendStatus> {
    let terminal = msg.is_terminal();
    match sender.try_send(msg) {
        Ok(()) => {
            if terminal {
                remove_channel_refs(key, state);
            }
            Ok(ChunkSendStatus::Ready)
        }
        Err(TrySendError::Full(msg)) => {
            if !register_pending_send(key, state) {
                tracing::warn!(
                    rid = key.rid(),
                    "gRPC bridge received another chunk before the parked chunk drained; closing stream"
                );
                close_channel_with_error(
                    py,
                    key,
                    state,
                    runtime_handle,
                    TerminalError::ChannelFull {
                        rid: key.rid.clone(),
                    },
                );
                return Ok(ChunkSendStatus::Closed);
            }

            let key_owned = key.clone();
            let state = state.clone();
            let runtime_handle = runtime_handle.clone_ref(py);
            let sender = sender.clone();

            tokio_handle.spawn(async move {
                match sender.send(msg).await {
                    Ok(()) => {
                        if terminal {
                            // Terminal chunks end the producer contract; no further on_ready
                            // signal is fired after a parked Finished/Error drains.
                            remove_channel_refs(&key_owned, &state);
                            return;
                        }

                        Python::attach(|py| {
                            if let Some(callback) = mark_send_ready(py, &key_owned, &state) {
                                notify_ready(py, key_owned.rid(), callback);
                            }
                        });
                    }
                    Err(_) => {
                        Python::attach(|py| {
                            close_channel_with_error(
                                py,
                                &key_owned,
                                &state,
                                &runtime_handle,
                                TerminalError::ClientDisconnected {
                                    rid: key_owned.rid.clone(),
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
                key,
                state,
                runtime_handle,
                TerminalError::ClientDisconnected {
                    rid: key.rid.clone(),
                },
            );
            Ok(ChunkSendStatus::Closed)
        }
    }
}

// Typed chunk callback for SGLang-native RPCs (dict-based chunks).
#[pyclass]
struct ChunkCallback {
    key: RequestKey,
    metadata_mode: ResponseMetadataMode,
    state: BridgeStateRef,
    runtime_handle: Py<PyAny>,
    tokio_handle: Handle,
}

#[pymethods]
impl ChunkCallback {
    /// Register before producing chunks. If a parked chunk drained before registration,
    /// Rust fires `on_ready` immediately so late registration cannot miss the edge.
    fn set_on_ready(&self, py: Python<'_>, on_ready: Py<PyAny>) -> PyResult<()> {
        set_on_ready_for_rid(py, &self.key, &self.state, on_ready)
    }

    fn clear_on_ready(&self) {
        clear_on_ready_for_rid(&self.key, &self.state);
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
        let sender = match state.channels.get(self.key.rid()) {
            Some(channel) if channel.incarnation == self.key.incarnation => channel.sender.clone(),
            None => return Ok(ChunkSendStatus::Closed),
            Some(_) => return Ok(ChunkSendStatus::Closed),
        };
        drop(state);

        if let Some(err_msg) = error {
            return try_send_chunk(
                py,
                &self.key,
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

        let choice_index = chunk
            .get_item("index")?
            .and_then(|v| v.extract::<i32>().ok())
            .unwrap_or(0);

        let meta_info = match self.metadata_mode {
            ResponseMetadataMode::LegacyStringMap => {
                ResponseMetadata::Legacy(extract_legacy_meta_info(chunk))
            }
            ResponseMetadataMode::TypedGenerate => {
                ResponseMetadata::Typed(extract_typed_meta_info(chunk))
            }
        };

        let data = ResponseData {
            text,
            output_ids,
            embedding,
            choice_index,
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
            &self.key,
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
    key: RequestKey,
    state: BridgeStateRef,
    runtime_handle: Py<PyAny>,
    tokio_handle: Handle,
}

#[pymethods]
impl JsonChunkCallback {
    /// Register before producing chunks. If a parked chunk drained before registration,
    /// Rust fires `on_ready` immediately so late registration cannot miss the edge.
    fn set_on_ready(&self, py: Python<'_>, on_ready: Py<PyAny>) -> PyResult<()> {
        set_on_ready_for_rid(py, &self.key, &self.state, on_ready)
    }

    fn clear_on_ready(&self) {
        clear_on_ready_for_rid(&self.key, &self.state);
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
        let sender = match state.channels.get(self.key.rid()) {
            Some(channel) if channel.incarnation == self.key.incarnation => channel.sender.clone(),
            None => return Ok(ChunkSendStatus::Closed),
            Some(_) => return Ok(ChunkSendStatus::Closed),
        };
        drop(state);

        if let Some(err_msg) = error {
            return try_send_chunk(
                py,
                &self.key,
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
            choice_index: 0,
            json_bytes: Some(bytes_data),
            meta_info: ResponseMetadata::Legacy(meta_info),
        };

        let msg = if finished {
            ResponseChunk::Finished(data)
        } else {
            ResponseChunk::Data(data)
        };

        try_send_chunk(
            py,
            &self.key,
            &self.state,
            &self.runtime_handle,
            &self.tokio_handle,
            &sender,
            msg,
        )
    }
}

fn extract_typed_meta_info(
    chunk: &Bound<'_, PyDict>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut meta = serde_json::Map::new();
    if let Ok(Some(meta_obj)) = chunk.get_item("meta_info")
        && let Ok(meta_dict) = meta_obj.cast::<PyDict>()
    {
        for (k, v) in meta_dict.iter() {
            if let Ok(key) = k.extract::<String>()
                && let Ok(val) = py_value_to_json_value(&v)
            {
                meta.insert(key, val);
            }
        }
    }
    meta
}

fn extract_legacy_meta_info(chunk: &Bound<'_, PyDict>) -> HashMap<String, String> {
    let mut meta = HashMap::new();
    if let Ok(Some(meta_obj)) = chunk.get_item("meta_info")
        && let Ok(meta_dict) = meta_obj.cast::<PyDict>()
    {
        for (k, v) in meta_dict.iter() {
            if let Ok(key) = k.extract::<String>()
                && let Ok(value) = py_value_to_json_string(&v)
            {
                meta.insert(key, value);
            }
        }
    }
    meta
}

#[cfg(test)]
mod tests;
