// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SSE passthrough — bridges a reqwest `bytes_stream()` into an axum Body.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

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
/// upstream stream panics, we surface a loud `io::Error` to the client; without
/// this, the body would EOF cleanly and clients couldn't distinguish that from
/// success — the worst failure class (truncated output that looks complete).
///
/// # Stream guards
/// When `stream_guards` is `Some`, the value is **moved into the spawned task**
/// and held for the entire body lifetime.  It is dropped only when the SSE
/// pump finishes (stream exhausted, client disconnects, or upstream errors).
/// The opaque `Box<dyn Send + 'static>` accepts any drop-only payload — most
/// commonly a tuple of [`crate::workers::LoadGuard`] and
/// [`crate::policies::active_load::ActiveLoadGuard`]. The proxy does not
/// inspect the value; it relies entirely on `Drop` semantics, so callers can
/// pack arbitrary cleanup state in. Pass `None` for callers that manage the
/// guard externally (e.g. non-streaming paths where the handler itself is the
/// guard scope).
///
/// # Completion hook
/// When `on_complete` is `Some`, the closure runs exactly once when the
/// pump task finishes. The bool argument is `true` on clean stream end
/// (including a clean client disconnect after at least the headers
/// landed cleanly), `false` on upstream stream error or pump panic.
/// `forward_streaming_to` passes a closure that records the worker's
/// circuit-breaker outcome — without this hook, a worker that returns
/// 2xx headers and then drops the stream mid-flight would stay credited
/// as healthy.
///
/// # First-byte hook
/// When `on_first_byte` is `Some`, the closure runs exactly once, the moment
/// the first `Ok` chunk is read from the upstream stream — i.e. time to first
/// token. It does NOT fire if the stream ends or errors before any `Ok` chunk
/// arrives. `forward_streaming_to` passes a closure that records
/// `sgl_router_ttft_seconds` for successful streaming responses.
pub fn bytes_stream_to_body<S, E>(
    stream: S,
    stream_guards: Option<Box<dyn Send + 'static>>,
    on_complete: Option<Box<dyn FnOnce(bool) + Send + 'static>>,
    on_first_byte: Option<Box<dyn FnOnce() + Send + 'static>>,
) -> Body
where
    S: futures::Stream<Item = Result<Bytes, E>> + Send + Unpin + 'static,
    E: std::fmt::Display + Send + Sync + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    tokio::spawn(async move {
        let tx_for_panic = tx.clone();
        // Capture the pump's outcome so we can report it through `on_complete`
        // AFTER `pump.catch_unwind()` settles. The closure inside owns
        // `outcome_setter`; the outer scope reads `outcome_holder` once.
        let outcome_holder = Arc::new(parking_lot::Mutex::new(true));
        let outcome_setter = Arc::clone(&outcome_holder);
        let pump = AssertUnwindSafe(async move {
            // Hold the guards for the task's lifetime — dropped when this
            // block exits (stream done or client disconnect).  Leading
            // underscore suppresses the "unused variable" lint while
            // keeping intent explicit.
            let _hold = stream_guards;
            let mut on_first_byte = on_first_byte;
            let mut s = stream;
            while let Some(chunk) = s.next().await {
                let item: Result<Bytes, std::io::Error> = chunk.map_err(|e| {
                    let msg = e.to_string();
                    tracing::warn!(error = %msg, "upstream SSE stream errored mid-flight");
                    std::io::Error::other(msg)
                });
                let is_err_chunk = item.is_err();
                // Fire the time-to-first-token hook on the first successful
                // chunk from upstream. `take()` makes it fire at most once;
                // an error-first stream never produced a token, so it's left
                // unfired (and dropped on task end).
                if !is_err_chunk {
                    if let Some(hook) = on_first_byte.take() {
                        hook();
                    }
                }
                if is_err_chunk {
                    *outcome_setter.lock() = false;
                }
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
        let pump_result = pump.catch_unwind().await;
        let panicked = pump_result.is_err();
        if let Err(panic_payload) = pump_result {
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
        if let Some(hook) = on_complete {
            let ok = !panicked && *outcome_holder.lock();
            hook(ok);
        }
    });
    Body::from_stream(ReceiverStream::new(rx))
}

/// Wrap a byte stream with an idle (between-chunks) timeout. If the upstream
/// delivers no chunk for `idle`, the wrapped stream yields one `io::Error` and
/// ends — so a hung upstream (sends 200 headers then stalls without sending
/// data or closing, e.g. a half-open TCP after a worker restart) can no longer
/// block the SSE pump forever. Without this the pump's `stream_guards` (the
/// per-worker admission slot + active-load entry) never drop and the router's
/// in-flight counter leaks, eventually pinning every worker at its cap and
/// shedding all traffic while the engines sit idle.
///
/// This is an *idle* timeout, not a total one: each delivered chunk resets it,
/// so a slow-but-progressing long generation is unaffected — only a true stall
/// trips it.
pub fn idle_timeout_stream<S, E>(
    stream: S,
    idle: std::time::Duration,
) -> futures::stream::BoxStream<'static, Result<Bytes, std::io::Error>>
where
    S: futures::Stream<Item = Result<Bytes, E>> + Send + Unpin + 'static,
    E: std::fmt::Display + Send + 'static,
{
    futures::stream::unfold((stream, false), move |(mut s, ended)| async move {
        if ended {
            return None;
        }
        match tokio::time::timeout(idle, s.next()).await {
            // Chunk arrived in time: pass it through, keep going.
            Ok(Some(Ok(chunk))) => Some((Ok(chunk), (s, false))),
            // Upstream error: surface it and stop (the pump breaks on errors).
            Ok(Some(Err(e))) => Some((
                std::io::Result::Err(std::io::Error::other(e.to_string())),
                (s, true),
            )),
            // Upstream ended cleanly.
            Ok(None) => None,
            // Idle timeout: surface a terminal error so the pump exits and drops
            // its guards instead of blocking on `s.next()` forever.
            Err(_elapsed) => Some((
                std::io::Result::Err(std::io::Error::other(format!(
                    "upstream stream idle for {idle:?}; aborting to release in-flight slot"
                ))),
                (s, true),
            )),
        }
    })
    .boxed()
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
        let body = bytes_stream_to_body(s, None, None, None);
        let bytes = body.collect().await.unwrap().to_bytes();
        assert_eq!(&bytes[..], b"hello world");
    }

    #[tokio::test]
    async fn on_first_byte_fires_once_on_first_ok_chunk() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let fired = Arc::new(AtomicUsize::new(0));
        let fired_c = Arc::clone(&fired);
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(b"a")),
            Ok(Bytes::from_static(b"b")),
        ];
        let s = stream::iter(chunks);
        let body = bytes_stream_to_body(
            s,
            None,
            None,
            Some(Box::new(move || {
                fired_c.fetch_add(1, Ordering::SeqCst);
            })),
        );
        let _ = body.collect().await.unwrap();
        assert_eq!(
            fired.load(Ordering::SeqCst),
            1,
            "first-byte hook must fire exactly once across the whole stream",
        );
    }

    #[tokio::test]
    async fn on_first_byte_not_fired_when_stream_errors_first() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let fired = Arc::new(AtomicUsize::new(0));
        let fired_c = Arc::clone(&fired);
        let chunks: Vec<Result<Bytes, std::io::Error>> = vec![Err(std::io::Error::other(
            "upstream failed before any token",
        ))];
        let s = stream::iter(chunks);
        let body = bytes_stream_to_body(
            s,
            None,
            None,
            Some(Box::new(move || {
                fired_c.fetch_add(1, Ordering::SeqCst);
            })),
        );
        let _ = body.collect().await;
        assert_eq!(
            fired.load(Ordering::SeqCst),
            0,
            "first-byte hook must not fire when no Ok chunk is ever produced",
        );
    }

    #[tokio::test]
    async fn upstream_error_surfaces_to_consumer() {
        let chunks: Vec<Result<Bytes, std::io::Error>> = vec![
            Ok(Bytes::from_static(b"ok-chunk")),
            Err(std::io::Error::other("upstream blew up mid-stream")),
        ];
        let s = stream::iter(chunks);
        let body = bytes_stream_to_body(s, None, None, None);
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

    /// A stream that yields one Ok chunk, then panics with a non-string
    /// payload (`i32`). Used to exercise the `<non-string panic payload>`
    /// fallback in the downcast ladder — the existing
    /// `PanicOnSecondPoll` test only covers the `&'static str` arm.
    struct PanicAnyOnSecondPoll {
        polls: usize,
    }

    impl futures::Stream for PanicAnyOnSecondPoll {
        type Item = Result<Bytes, std::io::Error>;

        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            self.polls += 1;
            match self.polls {
                1 => std::task::Poll::Ready(Some(Ok(Bytes::from_static(b"first-chunk")))),
                _ => std::panic::panic_any(42_i32),
            }
        }
    }

    #[tokio::test]
    async fn bytes_stream_to_body_handles_non_string_panic_payload() {
        // `panic_any(42_i32)` skips the formatter entirely — neither the
        // `&'static str` nor the `String` downcast arms match, so the
        // catch_unwind handler must fall through to the
        // `"<non-string panic payload>"` literal. If a refactor deletes
        // that arm, the closure unwrap-or-elses would panic itself or
        // produce an empty message, which this test catches.
        let s = PanicAnyOnSecondPoll { polls: 0 };
        let body = bytes_stream_to_body(s, None, None, None);
        let result = body.collect().await;
        assert!(
            result.is_err(),
            "expected body collect to surface non-string panic as Err, got Ok"
        );
        let err = result.err().unwrap();
        let msg = format!("{err}");
        assert!(
            msg.contains("<non-string panic payload>"),
            "expected fallback message for non-string panic payload, got: {msg}"
        );
        assert!(
            msg.contains("SSE pump panicked"),
            "expected wrapper message to remain, got: {msg}"
        );
    }

    #[tokio::test]
    async fn bytes_stream_to_body_propagates_pump_panic() {
        // The pump task panics mid-stream. The client must see a loud Err,
        // NOT a silently-truncated success.
        let s = PanicOnSecondPoll { polls: 0 };
        let body = bytes_stream_to_body(s, None, None, None);
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

    /// Regression guard for the backpressure-via-disconnect invariant.
    ///
    /// The doc on `bytes_stream_to_body` claims "when the axum Body is dropped
    /// the receiver is closed; `tx.send()` then returns `Err`, which breaks the
    /// loop — no upstream bytes are read after the client disconnects." This
    /// test pins that contract: a refactor that swaps the `if tx.send().await.
    /// is_err() { break; }` for `let _ = tx.send().await;` would silently
    /// regress (leaked upstream reads on every client cancel, visible only as
    /// ops-side memory growth).
    #[tokio::test]
    async fn bytes_stream_to_body_breaks_on_client_disconnect() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        // A stream that yields N Ok chunks readily, counting polls via a shared
        // atomic. After we read 1 chunk and drop the body, the pump must hit
        // tx.send-err and break — not drain all 1000 chunks.
        struct CountingStream {
            polls: Arc<AtomicUsize>,
            yielded: usize,
            max: usize,
        }

        impl futures::Stream for CountingStream {
            type Item = Result<Bytes, std::io::Error>;

            fn poll_next(
                mut self: std::pin::Pin<&mut Self>,
                _cx: &mut std::task::Context<'_>,
            ) -> std::task::Poll<Option<Self::Item>> {
                self.polls.fetch_add(1, Ordering::SeqCst);
                if self.yielded >= self.max {
                    return std::task::Poll::Ready(None);
                }
                self.yielded += 1;
                std::task::Poll::Ready(Some(Ok(Bytes::from_static(b"chunk"))))
            }
        }

        let polls = Arc::new(AtomicUsize::new(0));
        let stream = CountingStream {
            polls: polls.clone(),
            yielded: 0,
            max: 1000, // way more than we'll let it consume
        };
        let body = bytes_stream_to_body(stream, None, None, None);

        // Read exactly one frame, then drop the body to simulate client disconnect.
        let mut data_stream = body.into_data_stream();
        let first = data_stream.next().await;
        assert!(first.is_some(), "expected at least one chunk before drop");
        drop(data_stream);

        // Give the pump generous time to make additional polls if its break is
        // broken. Healthy code: pump fills the 64-slot channel, then on the
        // next iteration tx.send().await detects receiver-drop and breaks.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let final_polls = polls.load(Ordering::SeqCst);
        assert!(
            final_polls <= 70,
            "pump kept polling upstream after client disconnect: {final_polls} polls (expected <=70, channel bound + slack)"
        );
        // And: the pump must NOT have drained all 1000 chunks.
        assert!(
            final_polls < 1000,
            "pump drained the entire upstream after client disconnect ({final_polls} polls); the break-on-tx.send-err path is dead"
        );
    }

    /// A stalled upstream — sends headers, then never delivers a byte and never
    /// closes (a half-open TCP after a worker restart) — must not pin the SSE
    /// pump forever. The pump must exit and DROP its `stream_guards` (the
    /// per-worker admission slot + active-load entry). Without an idle timeout
    /// the pump blocks on `s.next()` indefinitely and the guard leaks,
    /// eventually pinning every worker at its cap (false shedding while engines
    /// sit idle).
    #[tokio::test]
    async fn stalled_upstream_releases_stream_guards() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicBool::new(false));
        let guard: Box<dyn Send + 'static> = Box::new(DropFlag(Arc::clone(&dropped)));

        // Upstream that never yields a byte and never ends, wrapped in the idle
        // timeout so the pump aborts and releases the guard.
        let stalled = idle_timeout_stream(
            stream::pending::<Result<Bytes, std::io::Error>>(),
            std::time::Duration::from_millis(50),
        );
        let body = bytes_stream_to_body(stalled, Some(guard), None, None);
        tokio::spawn(async move {
            let _ = body.collect().await;
        });

        for _ in 0..100 {
            if dropped.load(Ordering::SeqCst) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(
            dropped.load(Ordering::SeqCst),
            "stalled upstream must release stream_guards via the idle timeout",
        );
    }
}
