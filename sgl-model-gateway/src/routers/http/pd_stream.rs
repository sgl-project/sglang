//! Bounded SSE observation while a paired prefill request is still pending.
use axum::http::StatusCode;
use bytes::{Bytes, BytesMut};
use serde_json::{json, Value};

// This is a handshake/read-ahead bound, not a generation length limit. Once
// prefill completes, only one incomplete SSE event needs to be retained.
const MAX_PENDING_BYTES: usize = 1024 * 1024;

/// The public rid survives a retry; its worker state must not collide
/// with the cancelled attempt, which can still be awaiting a scheduler ACK.
#[derive(Clone)]
pub(super) struct RetryRequestIdentity {
    pub mappings: Vec<(String, String)>,
}

impl RetryRequestIdentity {
    fn replacement_inputs(request: &Value) -> Option<(Vec<String>, char)> {
        let rid = request.get("rid")?;
        let clients = if let Some(client) = rid.as_str() {
            vec![client.to_owned()]
        } else {
            rid.as_array()?
                .iter()
                .map(|v| v.as_str().map(str::to_owned))
                .collect::<Option<Vec<_>>>()?
        };
        // A fast Decode rejection can win before Prefill validates the batch.
        // Retrying must never turn invalid caller duplicates into valid IDs.
        if clients.is_empty()
            || clients.iter().any(String::is_empty)
            || clients
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != clients.len()
        {
            return None;
        }
        // Scheduler aborts use starts_with(rid). Avoid the original caller
        // IDs as prefixes, and use equal-length UUIDs so cancelling one batch
        // member cannot accidentally cancel another (e.g. index 1 vs 10).
        let marker = b"spq_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz-~"
            .iter()
            .copied()
            .find(|byte| clients.iter().all(|c| c.as_bytes()[0] != *byte))?;
        Some((clients, char::from(marker)))
    }

    pub fn can_replace(request: &Value) -> bool {
        Self::replacement_inputs(request).is_some()
    }

    pub fn replace(request: &mut Value) -> Option<Self> {
        let (clients, marker) = Self::replacement_inputs(request)?;
        let mappings: Vec<_> = clients
            .into_iter()
            .map(|client| (client, format!("{marker}mg-retry-{}", uuid::Uuid::new_v4())))
            .collect();
        let rid = request.get_mut("rid").unwrap();
        *rid = if rid.is_array() {
            Value::Array(
                mappings
                    .iter()
                    .map(|(_, worker)| Value::String(worker.clone()))
                    .collect(),
            )
        } else {
            Value::String(mappings[0].1.clone())
        };
        Some(Self { mappings })
    }

    pub fn restore_json(&self, value: &mut Value) -> bool {
        if let Some(items) = value.as_array_mut() {
            return items
                .iter_mut()
                .fold(false, |changed, item| self.restore_json(item) || changed);
        }
        let mut changed = false;
        // Native /generate and OpenAI chat/completion responses expose the
        // worker rid in these fields. Tool IDs and generated text are untouched.
        for path in ["/meta_info/id", "/id"] {
            if let Some(id) = value.pointer_mut(path) {
                if let Some((client, worker)) = self.mappings.iter().find(|(_, worker)| {
                    id.as_str().is_some_and(|value| {
                        value == worker
                            || value.strip_prefix(worker).is_some_and(valid_batch_suffix)
                    })
                }) {
                    // SRT expands a scalar batch rid into rid_0, rid_1, ... .
                    let suffix = &id.as_str().unwrap()[worker.len()..];
                    *id = Value::String(format!("{client}{suffix}"));
                    changed = true;
                }
            }
        }
        changed
    }

    pub fn restore_frame(&self, frame: Bytes) -> Bytes {
        // SseFrames has validated and assembled one complete UTF-8 event.
        // Rewrite only public response identity fields, never generated text or
        // error messages containing the same string.
        let Ok(text) = std::str::from_utf8(&frame) else {
            return frame;
        };
        let data = text
            .lines()
            .filter_map(|line| {
                line.strip_prefix("data:")
                    .map(|v| v.strip_prefix(' ').unwrap_or(v))
            })
            .collect::<Vec<_>>()
            .join("\n");
        let Ok(mut value) = serde_json::from_str::<Value>(&data) else {
            return frame; // DONE or a heartbeat.
        };
        if !self.restore_json(&mut value) {
            return frame;
        }
        let mut output = String::new();
        let mut written = false;
        for line in text.lines() {
            if line.starts_with("data:") {
                if !written {
                    output.push_str(&format!("data: {value}\n"));
                    written = true;
                }
            } else {
                output.push_str(line);
                output.push('\n');
            }
        }
        Bytes::from(output)
    }
}

fn valid_batch_suffix(suffix: &str) -> bool {
    suffix.strip_prefix('_').is_some_and(|index| {
        !index.is_empty()
            && (index == "0" || !index.starts_with('0'))
            && index.bytes().all(|b| b.is_ascii_digit())
    })
}

#[derive(Debug)]
pub(super) struct StreamFailure {
    pub status: StatusCode,
    pub message: String,
    pub worker_outcome: Option<bool>,
}

impl StreamFailure {
    pub fn protocol(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            message: message.into(),
            worker_outcome: Some(false),
        }
    }

    pub fn capacity(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            message: message.into(),
            worker_outcome: None,
        }
    }

    pub fn body(&self, responses_api: bool) -> Bytes {
        let error = json!({"error": {"message": self.message,
            "type": "decode_stream_error", "code": self.status.as_u16()}});
        if responses_api {
            let error =
                json!({"type": "error", "code": "decode_stream_error", "message": self.message});
            Bytes::from(format!("event: error\ndata: {error}\n\n"))
        } else {
            Bytes::from(format!("data: {error}\n\ndata: [DONE]\n\n"))
        }
    }
}

pub(super) enum Terminal {
    Done,
    Failed(StreamFailure),
}

fn failure(value: &Value) -> Option<StreamFailure> {
    if let Some(values) = value.as_array() {
        return values.iter().find_map(failure);
    }
    let abort = value.pointer("/meta_info/finish_reason");
    let reason = if abort.and_then(|v| v.get("type")).and_then(Value::as_str) == Some("abort") {
        let abort = abort?;
        // FINISH_ABORT without an error status is a user/session cancellation,
        // matching SGLang's native and OpenAI adapters. Do not invent a 502.
        if abort.get("status_code").is_none_or(Value::is_null) {
            return None;
        }
        abort
    } else if let Some(error) = value.get("error").filter(|v| !v.is_null()) {
        error
    } else if matches!(
        value.get("type").and_then(Value::as_str),
        Some("error" | "response.failed")
    ) {
        value.pointer("/response/error").unwrap_or(value)
    } else {
        return None;
    };
    let status = reason
        .get("status_code")
        .or_else(|| reason.get("code"))
        .and_then(Value::as_u64)
        .and_then(|n| u16::try_from(n).ok())
        .and_then(|n| StatusCode::from_u16(n).ok())
        .filter(|s| s.is_client_error() || s.is_server_error())
        .unwrap_or(StatusCode::BAD_GATEWAY);
    Some(StreamFailure {
        status,
        worker_outcome: Some(status.is_client_error()),
        message: reason
            .get("message")
            .and_then(Value::as_str)
            .or_else(|| reason.as_str())
            .unwrap_or("Decode stream aborted")
            .to_string(),
    })
}

fn terminal(frame: &[u8]) -> Result<Option<Terminal>, StreamFailure> {
    let text = std::str::from_utf8(frame)
        .map_err(|_| StreamFailure::protocol("Decode SSE contains invalid UTF-8"))?;
    let mut data = Vec::new();
    let mut event = None;
    for line in text.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if let Some(value) = line.strip_prefix("data:") {
            data.push(value.strip_prefix(' ').unwrap_or(value));
        } else if let Some(value) = line.strip_prefix("event:") {
            event = Some(value.trim());
        }
    }
    if data.is_empty() {
        return Ok(None);
    } // comment/heartbeat only
    let data = data.join("\n");
    if data.trim() == "[DONE]" {
        return Ok(Some(Terminal::Done));
    }
    if data.is_empty() {
        return Ok(None);
    }
    let value: Value = serde_json::from_str(&data)
        .map_err(|_| StreamFailure::protocol("Decode SSE contains invalid JSON"))?;
    if let Some(error) = failure(&value) {
        return Ok(Some(Terminal::Failed(error)));
    }
    if event == Some("error") {
        return Ok(Some(Terminal::Failed(StreamFailure::protocol(
            value
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("Decode error event"),
        ))));
    }
    Ok(matches!(
        value.get("type").and_then(Value::as_str),
        Some("response.completed" | "response.incomplete")
    )
    .then_some(Terminal::Done))
}

#[derive(Default)]
pub(super) struct SseFrames {
    pending: BytesMut,
    line_start: usize,
}

impl SseFrames {
    pub fn has_incomplete_event(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<(Bytes, Option<Terminal>)>, StreamFailure> {
        let mut frames = Vec::new();
        // Append at most one line at a time so a large transport chunk containing
        // many small events does not violate the per-event memory bound.
        for part in chunk.split_inclusive(|b| *b == b'\n') {
            if self.pending.len().saturating_add(part.len()) > MAX_PENDING_BYTES {
                return Err(StreamFailure::capacity("Decode SSE event exceeds 1 MiB"));
            }
            self.pending.extend_from_slice(part);
            if self.pending.last() != Some(&b'\n') {
                continue;
            }
            let end = self.pending.len();
            let line = &self.pending[self.line_start..end - 1];
            if line.is_empty() || line == b"\r" {
                let frame = self.pending.split().freeze();
                let state = terminal(&frame)?;
                let done = state.is_some();
                frames.push((frame, state));
                self.line_start = 0;
                if done {
                    break;
                }
            } else {
                self.line_start = end;
            }
        }
        Ok(frames)
    }
}

#[derive(Default)]
pub(super) struct DecodeReadAhead {
    pub payload: BytesMut,
    frames: SseFrames,
    pub finished: bool,
}

impl DecodeReadAhead {
    pub async fn read(&mut self, response: &mut reqwest::Response) -> Result<(), StreamFailure> {
        let chunk = response
            .chunk()
            .await
            .map_err(|e| StreamFailure::protocol(format!("Decode stream read failed: {e}")))?
            .ok_or_else(|| {
                StreamFailure::protocol("Decode stream ended before a terminal event")
            })?;
        if self.payload.len().saturating_add(chunk.len()) > MAX_PENDING_BYTES {
            return Err(StreamFailure::capacity(
                "Decode read-ahead exceeds 1 MiB while waiting for prefill",
            ));
        }
        for (_, state) in self.frames.push(&chunk)? {
            match state {
                Some(Terminal::Failed(error)) => return Err(error),
                Some(Terminal::Done) => self.finished = true,
                None => {}
            }
        }
        self.payload.extend_from_slice(&chunk);
        Ok(())
    }
}

/// Error headers are already decisive. Bound their optional diagnostic body so
/// a peer cannot make a known rejection wait for the full request timeout.
pub(super) async fn read_error_body(mut response: reqwest::Response) -> Result<Bytes, String> {
    let read = async {
        let mut body = BytesMut::new();
        while let Some(chunk) = response.chunk().await.map_err(|e| e.to_string())? {
            if body.len().saturating_add(chunk.len()) > 64 * 1024 {
                return Err("Worker error body exceeds 64 KiB".to_string());
            }
            body.extend_from_slice(&chunk);
        }
        Ok(body.freeze())
    };
    tokio::time::timeout(std::time::Duration::from_secs(1), read)
        .await
        .map_err(|_| "Worker error body timed out after 1 second".to_string())?
}
