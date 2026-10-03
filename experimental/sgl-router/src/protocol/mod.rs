// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Client protocols served by converting to and from chat completions.

pub mod responses;

use std::task::Poll;

use axum::body::Body;
use bytes::Bytes;
use futures::StreamExt;

/// Chat SSE → client-protocol SSE converter.
pub trait SseTransducer: Send + 'static {
    fn feed(&mut self, chunk: &[u8]) -> Vec<u8>;
    fn finish(&mut self) -> Vec<u8>;
    fn fail(&mut self, message: &str) -> Vec<u8>;
    fn is_terminal(&self) -> bool;
}

/// Wrap a chat SSE body in `conv`. After a terminal event the upstream is
/// drained, so the SSE pump sees a normal end, not a client disconnect.
pub fn transduce_body<T: SseTransducer>(body: Body, mut conv: T) -> Body {
    let mut inner = body.into_data_stream();
    // `inner` must not be polled after it ends.
    let mut inner_done = false;
    let out = futures::stream::poll_fn(move |cx| loop {
        if inner_done {
            return Poll::Ready(None);
        }
        if conv.is_terminal() {
            return match inner.poll_next_unpin(cx) {
                Poll::Ready(Some(_)) => continue,
                Poll::Ready(None) => {
                    inner_done = true;
                    Poll::Ready(None)
                }
                Poll::Pending => Poll::Pending,
            };
        }
        let bytes = match inner.poll_next_unpin(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Some(Ok(chunk))) => conv.feed(&chunk),
            Poll::Ready(Some(Err(e))) => {
                tracing::warn!(error = %e, "protocol stream: upstream body error");
                inner_done = true;
                let out = conv.fail("upstream stream interrupted");
                return Poll::Ready(Some(Ok::<_, std::io::Error>(Bytes::from(out))));
            }
            Poll::Ready(None) => {
                inner_done = true;
                let out = conv.finish();
                if out.is_empty() {
                    return Poll::Ready(None);
                }
                return Poll::Ready(Some(Ok(Bytes::from(out))));
            }
        };
        if !bytes.is_empty() {
            return Poll::Ready(Some(Ok(Bytes::from(bytes))));
        }
    });
    Body::from_stream(out)
}

/// Longer lines are dropped.
const MAX_LINE_BYTES: usize = 16 << 20;

/// Reassembles SSE lines across arbitrary chunk boundaries.
#[derive(Default)]
pub(crate) struct LineBuffer {
    line: Vec<u8>,
}

impl LineBuffer {
    pub(crate) fn push(&mut self, chunk: &[u8], mut f: impl FnMut(&[u8])) {
        let mut rest = chunk;
        while let Some(nl) = rest.iter().position(|&b| b == b'\n') {
            if self.line.len() + nl <= MAX_LINE_BYTES {
                self.line.extend_from_slice(&rest[..nl]);
                let line = std::mem::take(&mut self.line);
                f(line.strip_suffix(b"\r").unwrap_or(&line));
            } else {
                self.line.clear();
            }
            rest = &rest[nl + 1..];
        }
        if self.line.len() + rest.len() <= MAX_LINE_BYTES {
            self.line.extend_from_slice(rest);
        } else {
            self.line.clear();
        }
    }

    pub(crate) fn flush(&mut self, mut f: impl FnMut(&[u8])) {
        if !self.line.is_empty() {
            let line = std::mem::take(&mut self.line);
            f(line.strip_suffix(b"\r").unwrap_or(&line));
        }
    }
}

/// JSON payload of a `data:` line (`None` for anything else and `[DONE]`).
pub(crate) fn data_payload(line: &[u8]) -> Option<serde_json::Value> {
    let payload = line.strip_prefix(b"data:")?.trim_ascii_start();
    if payload.is_empty() || payload == b"[DONE]" {
        return None;
    }
    match serde_json::from_slice(payload) {
        Ok(v) => Some(v),
        Err(_) => {
            tracing::debug!("protocol stream: skipping non-JSON upstream data line");
            None
        }
    }
}

pub(crate) fn write_event(out: &mut Vec<u8>, event: &str, data: &serde_json::Value) {
    out.extend_from_slice(b"event: ");
    out.extend_from_slice(event.as_bytes());
    out.extend_from_slice(b"\ndata: ");
    serde_json::to_writer(&mut *out, data).expect("serialize event");
    out.extend_from_slice(b"\n\n");
}
