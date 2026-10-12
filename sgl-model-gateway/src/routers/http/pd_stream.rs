//! Bounded SSE observation while a paired prefill request is still pending.
use axum::http::StatusCode;
use bytes::{Bytes, BytesMut};
use serde_json::{json, Value};

// This is a handshake/read-ahead bound, not a generation length limit. Once
// prefill completes, only one incomplete SSE event needs to be retained.
const MAX_PENDING_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
pub(super) struct StreamFailure {
    pub status: StatusCode,
    pub message: String,
}

impl StreamFailure {
    pub fn protocol(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            message: message.into(),
        }
    }

    pub fn body(&self) -> Bytes {
        let error = json!({"error": {"message": self.message,
            "type": "decode_stream_error", "code": self.status.as_u16()}});
        Bytes::from(format!("data: {error}\n\ndata: [DONE]\n\n"))
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
        abort?
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
        message: reason
            .get("message")
            .and_then(Value::as_str)
            .or_else(|| reason.as_str())
            .unwrap_or("Decode stream aborted")
            .to_string(),
    })
}

fn terminal(frame: &[u8]) -> Option<Terminal> {
    // SSE allows CRLF, comments, named events, and multiple data lines. TCP
    // chunks have no relationship to SSE event boundaries.
    let text = std::str::from_utf8(frame).ok()?;
    let data = text
        .split('\n')
        .filter_map(|line| {
            let line = line.strip_suffix('\r').unwrap_or(line);
            line.strip_prefix("data:")
                .map(|s| s.strip_prefix(' ').unwrap_or(s))
        })
        .collect::<Vec<_>>()
        .join("\n");
    if data.trim() == "[DONE]" {
        return Some(Terminal::Done);
    }
    let value: Value = serde_json::from_str(&data).ok()?;
    if let Some(error) = failure(&value) {
        return Some(Terminal::Failed(error));
    }
    (value.get("type").and_then(Value::as_str) == Some("response.completed"))
        .then_some(Terminal::Done)
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
                return Err(StreamFailure::protocol("Decode SSE event exceeds 1 MiB"));
            }
            self.pending.extend_from_slice(part);
            if self.pending.last() != Some(&b'\n') {
                continue;
            }
            let end = self.pending.len();
            let line = &self.pending[self.line_start..end - 1];
            if line.is_empty() || line == b"\r" {
                let frame = self.pending.split().freeze();
                let state = terminal(&frame);
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
    pub chunks: Vec<Bytes>,
    bytes: usize,
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
        if self.bytes.saturating_add(chunk.len()) > MAX_PENDING_BYTES {
            return Err(StreamFailure::protocol(
                "Decode read-ahead exceeds 1 MiB while waiting for prefill",
            ));
        }
        self.bytes += chunk.len();
        for (_, state) in self.frames.push(&chunk)? {
            match state {
                Some(Terminal::Failed(error)) => return Err(error),
                Some(Terminal::Done) => self.finished = true,
                None => {}
            }
        }
        self.chunks.push(chunk);
        Ok(())
    }
}
