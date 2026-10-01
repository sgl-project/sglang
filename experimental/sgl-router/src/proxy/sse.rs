// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SSE passthrough — bridges a reqwest `bytes_stream()` into an axum Body.

use std::panic::AssertUnwindSafe;
use std::time::Duration;

use axum::body::Body;
use bytes::Bytes;
use futures::{FutureExt, StreamExt};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

/// Why the SSE pump stopped, independently of any SSE error event it observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamEndReason {
    Completed,
    UpstreamError,
    IdleTimeout,
    /// The router's stale-request deadline expired, regardless of worker health.
    Expired,
    ClientDisconnect,
    PumpPanicked,
}

/// How the SSE pump ended, reported to the `on_complete` hook.
#[derive(Debug, Clone, Copy)]
pub struct StreamEnd {
    pub reason: StreamEndReason,
    /// An SSE error event (`data: {"error"...}`) rode the stream.
    pub saw_error_event: bool,
}

/// A `data:` line whose payload's first JSON key is `error` — tolerant of
/// SSE-legal framing variants (no space after `data:`, whitespace after `{`),
/// so the match is anchored to the spec rather than one serializer's bytes.
fn is_error_event_line(line: &[u8]) -> bool {
    line.strip_prefix(b"data:")
        .map(|p| p.trim_ascii_start())
        .and_then(|p| p.strip_prefix(b"{"))
        .map(|p| p.trim_ascii_start())
        .is_some_and(|p| p.starts_with(b"\"error\""))
}

/// Line-start bytes that suffice to decide `is_error_event_line`.
const LINE_PROBE: usize = 32;

/// Finds error events emitted after an SSE response commits a 200.
/// Line-anchored, so lookalike text inside event payloads cannot match.
#[derive(Default)]
struct ErrorEventScanner {
    line_start: Vec<u8>,
}

impl ErrorEventScanner {
    fn feed(&mut self, chunk: &[u8]) -> bool {
        let mut hit = false;
        for (i, segment) in chunk.split(|&b| b == b'\n').enumerate() {
            if i > 0 {
                hit |= is_error_event_line(&self.line_start);
                self.line_start.clear();
            }
            let room = LINE_PROBE - self.line_start.len();
            self.line_start
                .extend_from_slice(&segment[..segment.len().min(room)]);
        }
        hit
    }
}

/// Bounds on a streaming response beyond what the upstream stream itself provides.
#[derive(Debug, Clone, Default)]
pub struct StreamLimits {
    /// Maximum silence between upstream chunks. `None` waits indefinitely.
    pub idle_timeout: Option<Duration>,
    /// Fires when the stale-request janitor expires the request.
    pub expiration: Option<CancellationToken>,
}

/// Bridge a byte stream into an axum Body that streams chunks unchanged.
///
/// One tokio task pumps upstream chunks through a bounded 64-slot channel so a
/// slow client backpressures the upstream read. The pump stops as soon as the
/// client disconnects, even while upstream is silent, when `limits.idle_timeout`
/// elapses between chunks, or when `limits.expiration` fires.
///
/// The terminal result travels on a separate channel and is chained after the
/// data, so a full queue cannot block cleanup or turn a failed stream into a
/// clean EOF. A pump panic is reported the same way.
///
/// `guards` is held until the pump finishes; `on_first_byte` runs on the first
/// `Ok` chunk; `on_complete` runs exactly once with the final [`StreamEnd`].
pub fn bytes_stream_to_body<S, E>(
    mut stream: S,
    guards: Option<Box<dyn Send + 'static>>,
    on_complete: Option<Box<dyn FnOnce(StreamEnd) + Send + 'static>>,
    mut on_first_byte: Option<Box<dyn FnOnce() + Send + 'static>>,
    limits: StreamLimits,
) -> Body
where
    S: futures::Stream<Item = Result<Bytes, E>> + Send + Unpin + 'static,
    E: std::fmt::Display + Send + Sync + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel::<Bytes>(64);
    let (terminal_tx, terminal_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let mut end = StreamEnd {
            reason: StreamEndReason::Completed,
            saw_error_event: false,
        };
        let mut scanner = ErrorEventScanner::default();
        let idle = limits.idle_timeout.unwrap_or(Duration::MAX);
        let expired = async {
            match limits.expiration {
                Some(token) => token.cancelled().await,
                None => std::future::pending().await,
            }
        };
        // Disconnect and expiration race the whole forwarding loop, so they
        // fire while `send` waits on a full queue as well as while upstream is silent.
        let pump = async {
            tokio::select! {
                biased;
                _ = tx.closed() => {
                    end.reason = StreamEndReason::ClientDisconnect;
                    Ok(())
                }
                _ = expired => {
                    end.reason = StreamEndReason::Expired;
                    Err(std::io::Error::other("SSE stream exceeded stale_request_timeout"))
                }
                result = async {
                    loop {
                        let bytes = match tokio::time::timeout(idle, stream.next()).await {
                            Ok(None) => return (StreamEndReason::Completed, Ok(())),
                            Ok(Some(Ok(bytes))) => bytes,
                            Ok(Some(Err(e))) => return (
                                StreamEndReason::UpstreamError,
                                Err(std::io::Error::other(e.to_string())),
                            ),
                            Err(_) => return (
                                StreamEndReason::IdleTimeout,
                                Err(std::io::Error::other("SSE upstream idle timeout")),
                            ),
                        };
                        if let Some(hook) = on_first_byte.take() {
                            hook();
                        }
                        if !end.saw_error_event {
                            end.saw_error_event = scanner.feed(&bytes);
                        }
                        if tx.send(bytes).await.is_err() {
                            // Receiver gone; the `tx.closed()` arm reports the disconnect.
                            std::future::pending::<()>().await;
                        }
                    }
                } => {
                    end.reason = result.0;
                    result.1
                }
            }
        };
        let result = match AssertUnwindSafe(pump).catch_unwind().await {
            Ok(result) => result,
            Err(payload) => {
                end.reason = StreamEndReason::PumpPanicked;
                let message = payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("<non-string panic payload>");
                Err(std::io::Error::other(format!(
                    "SSE pump panicked: {message}"
                )))
            }
        };
        if let Some(hook) = on_complete {
            hook(end);
        }
        drop(guards);
        let _ = terminal_tx.send(result);
    });
    let terminal = futures::stream::once(terminal_rx).filter_map(|result| {
        futures::future::ready(match result {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(Err(error)),
            Err(_) => Some(Err(std::io::Error::other("SSE pump cancelled"))),
        })
    });
    Body::from_stream(ReceiverStream::new(rx).map(Ok).chain(terminal))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures::stream;
    use http_body_util::BodyExt;

    fn limited_body<S>(
        stream: S,
        limits: StreamLimits,
    ) -> (Body, tokio::sync::oneshot::Receiver<StreamEnd>)
    where
        S: futures::Stream<Item = Result<Bytes, std::io::Error>> + Send + Unpin + 'static,
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let body = bytes_stream_to_body(
            stream,
            None,
            Some(Box::new(move |end| {
                let _ = tx.send(end);
            })),
            None,
            limits,
        );
        (body, rx)
    }

    #[tokio::test(start_paused = true)]
    async fn idle_disconnect_releases_guards_without_waiting_for_upstream() {
        struct Release(Option<tokio::sync::oneshot::Sender<()>>);
        impl Drop for Release {
            fn drop(&mut self) {
                let _ = self.0.take().unwrap().send(());
            }
        }
        let (tx, rx) = tokio::sync::oneshot::channel();
        let body = bytes_stream_to_body(
            stream::pending::<Result<Bytes, std::io::Error>>(),
            Some(Box::new(Release(Some(tx)))),
            None,
            None,
            StreamLimits::default(),
        );
        tokio::task::yield_now().await;
        drop(body);
        tokio::time::timeout(Duration::from_millis(1), rx)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn idle_timeout_is_a_visible_upstream_failure() {
        let (body, end) = limited_body(
            stream::pending(),
            StreamLimits {
                idle_timeout: Some(Duration::from_secs(1)),
                expiration: None,
            },
        );
        assert!(body
            .collect()
            .await
            .unwrap_err()
            .to_string()
            .contains("idle timeout"));
        let end = end.await.unwrap();
        assert_eq!(end.reason, StreamEndReason::IdleTimeout);
    }

    #[tokio::test(start_paused = true)]
    async fn expiration_releases_guards_while_queue_is_full() {
        struct Release(Option<tokio::sync::oneshot::Sender<()>>);
        impl Drop for Release {
            fn drop(&mut self) {
                let _ = self.0.take().unwrap().send(());
            }
        }
        let token = CancellationToken::new();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let (end_tx, end_rx) = tokio::sync::oneshot::channel();
        let body = bytes_stream_to_body(
            stream::repeat_with(|| Ok::<_, std::io::Error>(Bytes::from_static(b"chunk"))),
            Some(Box::new(Release(Some(release_tx)))),
            Some(Box::new(move |end| {
                let _ = end_tx.send(end);
            })),
            None,
            StreamLimits {
                idle_timeout: None,
                expiration: Some(token.clone()),
            },
        );
        tokio::task::yield_now().await;
        token.cancel();
        // Cleanup must finish before the client frees any channel capacity.
        tokio::time::timeout(Duration::from_millis(1), release_rx)
            .await
            .expect("expiration must release guards while the queue remains full")
            .unwrap();
        assert_eq!(end_rx.await.unwrap().reason, StreamEndReason::Expired);
        // The queue is full, so the failure must ride the terminal channel.
        assert!(body
            .collect()
            .await
            .unwrap_err()
            .to_string()
            .contains("stale_request_timeout"));
    }
    #[tokio::test]
    async fn passes_through_a_simple_byte_stream() {
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(b"hello ")),
            Ok(Bytes::from_static(b"world")),
        ];
        let s = stream::iter(chunks);
        let body = bytes_stream_to_body(s, None, None, None, StreamLimits::default());
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
            StreamLimits::default(),
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
            StreamLimits::default(),
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
        let body = bytes_stream_to_body(s, None, None, None, StreamLimits::default());
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
        let (body, end) = limited_body(s, StreamLimits::default());
        let result = body.collect().await;
        assert_eq!(end.await.unwrap().reason, StreamEndReason::PumpPanicked);
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
        let body = bytes_stream_to_body(s, None, None, None, StreamLimits::default());
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
        let body = bytes_stream_to_body(stream, None, None, None, StreamLimits::default());

        // Read exactly one frame, then drop the body to simulate client disconnect.
        let mut data_stream = body.into_data_stream();
        let first = data_stream.next().await;
        assert!(first.is_some(), "expected at least one chunk before drop");
        drop(data_stream);

        // Give the pump generous time to make additional polls if its break is
        // broken. Healthy code: pump fills the 64-slot channel, then on the
        // next iteration tx.send().await detects receiver-drop and breaks.
        tokio::time::sleep(Duration::from_millis(200)).await;
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

    #[test]
    fn error_event_scanner_detects_engine_error_event() {
        let mut scanner = ErrorEventScanner::default();
        assert!(!scanner.feed(b"data: {\"choices\": [{\"delta\": {\"content\": \"hi\"}}]}\n\n"));
        assert!(
            scanner.feed(b"data: {\"error\": {\"message\": \"queue is full\", \"code\": 503}}\n\n")
        );
    }

    #[test]
    fn error_event_scanner_detects_error_split_across_chunks() {
        let mut scanner = ErrorEventScanner::default();
        assert!(!scanner.feed(b"data: {\"err"));
        assert!(scanner.feed(b"or\": {\"code\": 503}}\n\n"));
    }

    #[test]
    fn error_event_scanner_ignores_error_text_inside_content() {
        let mut scanner = ErrorEventScanner::default();
        assert!(!scanner.feed(
            b"data: {\"choices\": [{\"delta\": {\"content\": \"data: {\\\"error\\\" is how it looks\"}}]}\n\n",
        ));
        assert!(!scanner.feed(b"data: [DONE]\n\n"));
    }

    #[test]
    fn error_event_scanner_accepts_sse_framing_variants() {
        for event in [
            &b"data:{\"error\": {\"code\": 503}}\n\n"[..],
            b"data: { \"error\": {\"code\": 503}}\n\n",
            b"data:  {\"error\": \"queue full\"}\n\n",
        ] {
            assert!(
                ErrorEventScanner::default().feed(event),
                "missed variant: {}",
                String::from_utf8_lossy(event)
            );
        }
    }

    #[test]
    fn error_event_scanner_bounds_line_buffer() {
        let mut scanner = ErrorEventScanner::default();
        let big = vec![b'x'; 1 << 20];
        assert!(!scanner.feed(&big));
        assert_eq!(scanner.line_start.len(), LINE_PROBE);
    }

    fn body_with_completion(
        chunks: Vec<Result<Bytes, std::io::Error>>,
    ) -> (Body, tokio::sync::oneshot::Receiver<StreamEnd>) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let body = bytes_stream_to_body(
            stream::iter(chunks),
            None,
            Some(Box::new(move |end| {
                let _ = tx.send(end);
            })),
            None,
            StreamLimits::default(),
        );
        (body, rx)
    }

    async fn stream_end(rx: tokio::sync::oneshot::Receiver<StreamEnd>) -> StreamEnd {
        rx.await.expect("completion hook dropped")
    }

    #[tokio::test]
    async fn completion_reports_error_event() {
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: {\"err")),
            Ok(Bytes::from_static(b"or\": {\"code\": 503}}\n\n")),
        ];
        let (body, completion) = body_with_completion(chunks);
        let _ = body.collect().await.unwrap();
        let end = stream_end(completion).await;
        assert_eq!(end.reason, StreamEndReason::Completed);
        assert!(end.saw_error_event);
    }

    #[tokio::test]
    async fn completion_reports_error_event_then_transport_error() {
        let chunks = vec![
            Ok(Bytes::from_static(
                b"data: {\"error\": {\"code\": 503}}\n\n",
            )),
            Err(std::io::Error::other("connection reset")),
        ];
        let (body, completion) = body_with_completion(chunks);
        let _ = body.collect().await;
        let end = stream_end(completion).await;
        assert_eq!(end.reason, StreamEndReason::UpstreamError);
        assert!(end.saw_error_event);
    }

    #[tokio::test]
    async fn completion_reports_upstream_error() {
        let chunks = vec![Err(std::io::Error::other("upstream failed"))];
        let (body, completion) = body_with_completion(chunks);
        let _ = body.collect().await;
        let end = stream_end(completion).await;
        assert_eq!(end.reason, StreamEndReason::UpstreamError);
        assert!(!end.saw_error_event);
    }

    #[tokio::test]
    async fn completion_reports_clean_end() {
        let chunks = vec![Ok::<Bytes, std::io::Error>(Bytes::from_static(
            b"data: [DONE]\n\n",
        ))];
        let (body, completion) = body_with_completion(chunks);
        let _ = body.collect().await.unwrap();
        let end = stream_end(completion).await;
        assert_eq!(end.reason, StreamEndReason::Completed);
        assert!(!end.saw_error_event);
    }

    #[tokio::test]
    async fn completion_reports_client_disconnect() {
        let chunks = std::iter::repeat_with(|| {
            Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: x\n\n"))
        })
        .take(1000)
        .collect();
        let (body, completion) = body_with_completion(chunks);
        let mut stream = body.into_data_stream();
        let _ = stream.next().await;
        drop(stream);
        let end = stream_end(completion).await;
        assert_eq!(end.reason, StreamEndReason::ClientDisconnect);
    }
}
