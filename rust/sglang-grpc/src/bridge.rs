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
use crate::utils::{json_map_to_pydict, py_value_to_json_string};

mod callbacks;
use callbacks::{ChunkCallback, JsonChunkCallback};

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
    pub delta_output_ids: Option<Vec<i32>>,
    pub embedding: Option<Vec<f32>>,
    pub choice_index: i32,
    pub json_bytes: Option<Vec<u8>>,
    pub meta_info: HashMap<String, String>,
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

    fn incarnation(&self) -> u64 {
        self.incarnation
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
    abort_all_in_progress: usize,
    pending_sends: HashSet<RequestKey>,
    ready_callbacks: HashMap<RequestKey, Py<PyAny>>,
    ready_signals: HashSet<RequestKey>,
    terminal_errors: HashMap<RequestKey, TerminalError>,
}

struct ActiveChannel {
    incarnation: u64,
    sender: Option<Sender<ResponseChunk>>,
    preserve_on_explicit_abort: bool,
    scheduler_backed: bool,
    submitting: bool,
    abort_requested: bool,
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
        preserve_on_explicit_abort: bool,
        scheduler_backed: bool,
    ) -> PyResult<SubmittedRequest> {
        let (sender, receiver) = mpsc::channel(self.response_channel_capacity);
        let mut state = lock_or_recover(self.state.as_ref(), "state");
        if state.abort_all_in_progress > 0 {
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
                sender: Some(sender),
                preserve_on_explicit_abort,
                scheduler_backed,
                submitting: scheduler_backed,
                abort_requested: false,
            },
        );
        Ok(SubmittedRequest { key, receiver })
    }

    fn make_chunk_callback(&self, py: Python<'_>, key: RequestKey) -> PyResult<Py<PyAny>> {
        let callback = ChunkCallback {
            key,
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
        choice_aware: bool,
    ) -> PyResult<SubmittedRequest> {
        let submitted = self.create_channel(rid, choice_aware, true)?;

        let result = Python::attach(|py| -> PyResult<()> {
            let py_req_dict = json_map_to_pydict(py, &req_dict)?;
            let callback = self.make_chunk_callback(py, submitted.key.clone())?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("req_type", req_type)?;
            kwargs.set_item("req_dict", py_req_dict)?;
            kwargs.set_item("chunk_callback", callback)?;
            kwargs.set_item("choice_aware", choice_aware)?;
            kwargs.set_item("lifecycle_id", submitted.key.incarnation())?;

            self.runtime_handle
                .call_method(py, "submit_request", (), Some(&kwargs))?;
            Ok(())
        });

        match result {
            Ok(()) => {
                let abort_requested = {
                    let mut state = lock_or_recover(self.state.as_ref(), "state");
                    state
                        .channels
                        .get_mut(submitted.key.rid())
                        .filter(|channel| channel.incarnation == submitted.key.incarnation)
                        .is_some_and(|channel| {
                            channel.submitting = false;
                            channel.abort_requested
                        })
                };
                if abort_requested {
                    if let Err(err) = self.abort_runtime_request(&submitted.key) {
                        let mut state = lock_or_recover(self.state.as_ref(), "state");
                        if let Some(channel) = state
                            .channels
                            .get_mut(submitted.key.rid())
                            .filter(|channel| channel.incarnation == submitted.key.incarnation)
                        {
                            channel.abort_requested = false;
                        }
                        return Err(err);
                    }
                }
                Ok(submitted)
            }
            Err(err) => {
                self.remove_channel(&submitted.key);
                Err(err)
            }
        }
    }

    // ------------------------------------------------------------------
    // Abort
    // ------------------------------------------------------------------

    fn abort_runtime_request(&self, key: &RequestKey) -> PyResult<()> {
        Python::attach(|py| {
            self.runtime_handle.call_method1(
                py,
                "abort",
                (key.rid(), Some(key.incarnation()), false),
            )?;
            Ok(())
        })
    }

    pub fn abort(&self, rid: &str, abort_all: bool) -> PyResult<()> {
        if !abort_all && rid.trim().is_empty() {
            return Err(PyValueError::new_err(
                "Abort requires a non-empty rid unless abort_all is true",
            ));
        }

        let (keys, call_runtime) = if abort_all {
            let mut state = lock_or_recover(self.state.as_ref(), "state");
            state.abort_all_in_progress += 1;
            let keys = state
                .channels
                .iter_mut()
                .map(|(rid, channel)| {
                    channel.abort_requested = true;
                    RequestKey {
                        rid: rid.clone(),
                        incarnation: channel.incarnation,
                    }
                })
                .collect::<Vec<_>>();
            (keys, true)
        } else {
            let mut state = lock_or_recover(self.state.as_ref(), "state");
            state.channels.get_mut(rid).map_or_else(
                || (Vec::new(), false),
                |channel| {
                    channel.abort_requested = true;
                    let should_call = channel.scheduler_backed && !channel.submitting;
                    (
                        vec![RequestKey {
                            rid: rid.to_string(),
                            incarnation: channel.incarnation,
                        }],
                        should_call,
                    )
                },
            )
        };

        if !abort_all && keys.is_empty() {
            tracing::debug!(rid, "Ignoring abort for inactive gRPC request id");
            return Ok(());
        }

        let call_result = if abort_all {
            Python::attach(|py| {
                self.runtime_handle
                    .call_method1(py, "abort", (rid, Option::<u64>::None, true))?;
                Ok(())
            })
        } else if call_runtime {
            self.abort_runtime_request(&keys[0])
        } else {
            Ok(())
        };

        let mut state = lock_or_recover(self.state.as_ref(), "state");
        if call_result.is_ok() {
            for key in &keys {
                finalize_explicit_abort_locked(&mut state, key);
            }
        } else {
            for key in &keys {
                if let Some(channel) = state
                    .channels
                    .get_mut(key.rid())
                    .filter(|channel| channel.incarnation == key.incarnation)
                {
                    channel.abort_requested = false;
                }
            }
        }
        if abort_all {
            state.abort_all_in_progress = state.abort_all_in_progress.saturating_sub(1);
            tracing::debug!(
                affected = keys.len(),
                "gRPC abort_all cleared active response channels"
            );
        }
        call_result
    }

    pub fn abort_request(&self, key: &RequestKey) -> PyResult<()> {
        let (scheduler_backed, submitting) = {
            let mut state = lock_or_recover(self.state.as_ref(), "state");
            let Some(channel) = state
                .channels
                .get_mut(key.rid())
                .filter(|channel| channel.incarnation == key.incarnation)
            else {
                remove_auxiliary_refs_locked(&mut state, key);
                state.terminal_errors.remove(key);
                return Ok(());
            };
            channel.abort_requested = true;
            channel.sender.take();
            let scheduler_backed = channel.scheduler_backed;
            let submitting = channel.submitting;
            remove_auxiliary_refs_locked(&mut state, key);
            state.terminal_errors.remove(key);
            if !scheduler_backed {
                state.channels.remove(key.rid());
            }
            (scheduler_backed, submitting)
        };

        if scheduler_backed && !submitting {
            self.abort_runtime_request(key)?;
        }
        Ok(())
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
        let submitted = self.create_channel(rid, false, false)?;

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
    let (should_abort, had_consumer, scheduler_backed) = {
        let mut state = lock_or_recover(state.as_ref(), "state");
        let Some(channel) = state
            .channels
            .get_mut(key.rid())
            .filter(|channel| channel.incarnation == key.incarnation)
        else {
            return;
        };
        let sender = channel.sender.take();
        let had_consumer = sender.as_ref().is_some_and(|sender| !sender.is_closed());
        let should_abort =
            channel.scheduler_backed && !channel.submitting && !channel.abort_requested;
        channel.abort_requested = true;
        let scheduler_backed = channel.scheduler_backed;
        remove_auxiliary_refs_locked(&mut state, key);
        if had_consumer {
            state.terminal_errors.insert(key.clone(), error);
        }
        if !scheduler_backed {
            state.channels.remove(key.rid());
        }
        (should_abort, had_consumer, scheduler_backed)
    };
    if should_abort
        && let Err(err) =
            runtime_handle.call_method1(py, "abort", (key.rid(), Some(key.incarnation()), false))
    {
        let mut state = lock_or_recover(state.as_ref(), "state");
        if let Some(channel) = state
            .channels
            .get_mut(key.rid())
            .filter(|channel| channel.incarnation == key.incarnation)
        {
            channel.abort_requested = false;
        }
        tracing::warn!(
            rid = key.rid(),
            "Failed to abort closed gRPC request: {err}"
        );
    }
    if !had_consumer && scheduler_backed {
        tracing::debug!(
            rid = key.rid(),
            "Retaining request tombstone until scheduler terminal"
        );
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
    let Some(channel) = state
        .channels
        .get_mut(key.rid())
        .filter(|channel| channel.incarnation == key.incarnation)
    else {
        return;
    };
    if channel.preserve_on_explicit_abort {
        return;
    }
    let had_consumer = channel.sender.take().is_some();
    let scheduler_backed = channel.scheduler_backed;
    remove_auxiliary_refs_locked(state, key);
    if had_consumer {
        state.terminal_errors.insert(
            key.clone(),
            TerminalError::Aborted {
                rid: key.rid.clone(),
            },
        );
    }
    if !scheduler_backed {
        state.channels.remove(key.rid());
    }
}

fn remove_channel_refs(key: &RequestKey, state: &BridgeStateRef) {
    let mut state = lock_or_recover(state.as_ref(), "state");
    remove_channel_refs_locked(&mut state, key);
}

#[cfg(test)]
mod tests;
