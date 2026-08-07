use super::*;

fn register_pending_send(key: &RequestKey, state: &BridgeStateRef) -> bool {
    let mut state = lock_or_recover(state.as_ref(), "state");
    if state
        .channels
        .get(key.rid())
        .is_none_or(|channel| channel.incarnation != key.incarnation || channel.sender.is_none())
    {
        return false;
    }
    state.pending_sends.insert(key.clone())
}

fn mark_send_ready(py: Python<'_>, key: &RequestKey, state: &BridgeStateRef) -> Option<Py<PyAny>> {
    let mut state = lock_or_recover(state.as_ref(), "state");
    state.pending_sends.remove(key);
    if state
        .channels
        .get(key.rid())
        .is_none_or(|channel| channel.incarnation != key.incarnation || channel.sender.is_none())
    {
        return None;
    }
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
        if state.channels.get(key.rid()).is_none_or(|channel| {
            channel.incarnation != key.incarnation || channel.sender.is_none()
        }) {
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
                    Err(_) if terminal => {
                        // The scheduler terminal reached the bridge even though the
                        // consumer disappeared before the parked send drained.
                        remove_channel_refs(&key_owned, &state);
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
            if terminal {
                remove_channel_refs(key, state);
                return Ok(ChunkSendStatus::Closed);
            }
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
pub(super) struct ChunkCallback {
    pub(super) key: RequestKey,
    pub(super) state: BridgeStateRef,
    pub(super) runtime_handle: Py<PyAny>,
    pub(super) tokio_handle: Handle,
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

        let Some(sender) = sender else {
            if finished || error.is_some() {
                remove_channel_refs(&self.key, &self.state);
            }
            return Ok(ChunkSendStatus::Closed);
        };

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

        let delta_output_ids: Option<Vec<i32>> = chunk
            .get_item("delta_output_ids")?
            .and_then(|v| v.extract::<Vec<i32>>().ok());

        let embedding: Option<Vec<f32>> = chunk
            .get_item("embedding")?
            .and_then(|v| v.extract::<Vec<f32>>().ok());

        let choice_index = chunk
            .get_item("index")?
            .and_then(|v| v.extract::<i32>().ok())
            .unwrap_or(0);

        let meta_info = extract_meta_info(chunk);

        let data = ResponseData {
            text,
            output_ids,
            delta_output_ids,
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
pub(super) struct JsonChunkCallback {
    pub(super) key: RequestKey,
    pub(super) state: BridgeStateRef,
    pub(super) runtime_handle: Py<PyAny>,
    pub(super) tokio_handle: Handle,
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

        let Some(sender) = sender else {
            if finished || error.is_some() {
                remove_channel_refs(&self.key, &self.state);
            }
            return Ok(ChunkSendStatus::Closed);
        };

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
            delta_output_ids: None,
            embedding: None,
            choice_index: 0,
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
            &self.key,
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
        && let Ok(meta_dict) = meta_obj.cast::<PyDict>()
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
