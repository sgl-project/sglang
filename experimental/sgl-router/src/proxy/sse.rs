// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SSE passthrough — bridges a reqwest `bytes_stream()` into an axum Body.

use std::panic::AssertUnwindSafe;

use axum::body::Body;
use bytes::Bytes;
use futures::{FutureExt, StreamExt};
use tokio_stream::wrappers::ReceiverStream;

/// Bridge a byte stream into an axum Body that streams chunks unchanged.
///
/// Spawns one tokio task per stream so the handler can return immediately.
/// Uses a **bounded** 64-slot channel so `tx.send().await` naturally
/// backpressures the upstream read when the client (axum Body consumer) falls
/// behind — an unbounded channel would buffer hundreds of MB for a slow client
/// receiving a long completion.
///
/// # Backpressure note
/// The channel bound of 64 absorbs short bursts while still limiting
/// worst-case outstanding bytes to 64 × chunk_size (typically a few MB).
///
/// # Client disconnect
/// When the axum Body is dropped the receiver is closed; `tx.send()` then
/// returns `Err`, which breaks the loop — no upstream bytes are read after the
/// client disconnects.
///
/// # Panic safety
/// The pump future is wrapped in `AssertUnwindSafe(..).catch_unwind()`. If the
/// upstream stream panics, we surface a loud `io::Error` to the client instead
/// of letting tokio silently swallow the panic and EOF the body — the worst
/// failure class (truncated output that looks like success).
pub fn bytes_stream_to_body<S, E>(stream: S) -> Body
where
    S: futures::Stream<Item = Result<Bytes, E>> + Send + Unpin + 'static,
    E: std::fmt::Display + Send + Sync + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    tokio::spawn(async move {
        let tx_for_panic = tx.clone();
        let pump = AssertUnwindSafe(async move {
            let mut s = stream;
            while let Some(chunk) = s.next().await {
                let item: Result<Bytes, std::io::Error> = chunk.map_err(|e| {
                    let msg = e.to_string();
                    tracing::warn!(error = %msg, "upstream SSE stream errored mid-flight");
                    std::io::Error::other(msg)
                });
                let is_err_chunk = item.is_err();
                if tx.send(item).await.is_err() {
                    // Receiver dropped. If we were about to ship an upstream
                    // error there's nothing left to report; otherwise this is
                    // a clean client-side disconnect — log at debug since it's
                    // not a router-side fault.
                    if !is_err_chunk {
                        tracing::debug!("SSE client disconnected mid-stream");
                    }
                    break;
                }
                if is_err_chunk {
                    // Surfaced upstream error to client; stop reading.
                    break;
                }
            }
        });
        if let Err(panic_payload) = pump.catch_unwind().await {
            let msg = panic_payload
                .downcast_ref::<&'static str>()
                .map(|s| (*s).to_string())
                .or_else(|| panic_payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            tracing::error!(error = %msg, "SSE pump task panicked");
            let _ = tx_for_panic
                .send(Err(std::io::Error::other(format!(
                    "SSE pump panicked: {msg}"
                ))))
                .await;
        }
    });
    Body::from_stream(ReceiverStream::new(rx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures::stream;
    use http_body_util::BodyExt;

    #[tokio::test]
    async fn passes_through_a_simple_byte_stream() {
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(b"hello ")),
            Ok(Bytes::from_static(b"world")),
        ];
        let s = stream::iter(chunks);
        let body = bytes_stream_to_body(s);
        let bytes = body.collect().await.unwrap().to_bytes();
        assert_eq!(&bytes[..], b"hello world");
    }

    #[tokio::test]
    async fn upstream_error_surfaces_to_consumer() {
        let chunks: Vec<Result<Bytes, std::io::Error>> = vec![
            Ok(Bytes::from_static(b"ok-chunk")),
            Err(std::io::Error::other("upstream blew up mid-stream")),
        ];
        let s = stream::iter(chunks);
        let body = bytes_stream_to_body(s);
        // Collecting a body that terminates with an error must return Err.
        let result = body.collect().await;
        assert!(
            result.is_err(),
            "expected body collect to surface upstream error, got Ok"
        );
    }

    /// A stream that yields one Ok chunk on the first poll, then panics on the
    /// second poll. Used to exercise the pump's panic-catch path.
    struct PanicOnSecondPoll {
        polls: usize,
    }

    impl futures::Stream for PanicOnSecondPoll {
        type Item = Result<Bytes, std::io::Error>;

        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            self.polls += 1;
            match self.polls {
                1 => std::task::Poll::Ready(Some(Ok(Bytes::from_static(b"first-chunk")))),
                _ => panic!("synthetic pump panic from stream poll"),
            }
        }
    }

    #[tokio::test]
    async fn bytes_stream_to_body_propagates_pump_panic() {
        // The pump task panics mid-stream. The client must see a loud Err,
        // NOT a silently-truncated success.
        let s = PanicOnSecondPoll { polls: 0 };
        let body = bytes_stream_to_body(s);
        let result = body.collect().await;
        assert!(
            result.is_err(),
            "expected body collect to surface pump panic as Err, got Ok (silent truncation)"
        );
        let err = result.err().unwrap();
        let msg = format!("{err}");
        assert!(
            msg.contains("pump panicked") || msg.contains("SSE pump panicked"),
            "expected error message to mention pump panic, got: {msg}"
        );
    }
}
