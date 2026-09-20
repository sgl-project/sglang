// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SSE passthrough — bridges a reqwest `bytes_stream()` into an axum Body.

use std::panic::AssertUnwindSafe;
use std::time::Duration;

use axum::body::Body;
use bytes::Bytes;
use futures::{FutureExt, StreamExt};
use tokio_stream::wrappers::ReceiverStream;

/// How the SSE pump ended, reported to the `on_complete` hook.
#[derive(Debug, Clone, Copy)]
pub struct StreamEnd {
    /// No upstream stream error and no pump panic.
    pub transport_ok: bool,
    /// An SSE error event (`data: {"error"...}`) rode the stream.
    pub saw_error_event: bool,
    /// The client disconnected or stopped consuming the response.
    pub client_disconnect: bool,
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

#[derive(Debug, Clone, Copy)]
pub struct StreamTimeouts {
    pub idle: Duration,
    pub send_stall: Duration,
    pub total: Duration,
}

impl Default for StreamTimeouts {
    fn default() -> Self {
        Self {
            idle: Duration::from_secs(180),
            send_stall: Duration::from_secs(180),
            total: Duration::from_secs(3600),
        }
    }
}

pub fn bytes_stream_to_body<S, E>(
    stream: S,
    guards: Option<Box<dyn Send + 'static>>,
    on_complete: Option<Box<dyn FnOnce(StreamEnd) + Send + 'static>>,
    on_first_byte: Option<Box<dyn FnOnce() + Send + 'static>>,
) -> Body
where
    S: futures::Stream<Item = Result<Bytes, E>> + Send + Unpin + 'static,
    E: std::fmt::Display + Send + Sync + 'static,
{
    bytes_stream_to_body_with_timeouts(
        stream,
        guards,
        on_complete,
        on_first_byte,
        StreamTimeouts::default(),
    )
}

/// Bounded forwarding with a separate terminal error, so a full queue cannot hide failure.
pub fn bytes_stream_to_body_with_timeouts<S, E>(
    mut stream: S,
    guards: Option<Box<dyn Send + 'static>>,
    on_complete: Option<Box<dyn FnOnce(StreamEnd) + Send + 'static>>,
    mut on_first_byte: Option<Box<dyn FnOnce() + Send + 'static>>,
    timeouts: StreamTimeouts,
) -> Body
where
    S: futures::Stream<Item = Result<Bytes, E>> + Send + Unpin + 'static,
    E: std::fmt::Display + Send + Sync + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    let (terminal_tx, terminal_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let mut end = StreamEnd {
            transport_ok: true,
            saw_error_event: false,
            client_disconnect: false,
        };
        let mut scanner = ErrorEventScanner::default();
        let deadline = tokio::time::Instant::now() + timeouts.total;
        let pump = async {
            loop {
                let chunk = tokio::select! {
                    biased;
                    _ = tx.closed() => { end.client_disconnect = true; return Ok(()); }
                    _ = tokio::time::sleep_until(deadline) => {
                        end.transport_ok = false;
                        return Err(std::io::Error::other("SSE total timeout"));
                    }
                    chunk = tokio::time::timeout(timeouts.idle, stream.next()) => chunk,
                };
                let bytes = match chunk {
                    Ok(None) => return Ok(()),
                    Ok(Some(Ok(bytes))) => bytes,
                    other => {
                        end.transport_ok = false;
                        let message = match other {
                            Ok(Some(Err(e))) => e.to_string(),
                            _ => "SSE upstream idle timeout".to_owned(),
                        };
                        return Err(std::io::Error::other(message));
                    }
                };
                if let Some(hook) = on_first_byte.take() {
                    hook();
                }
                if !end.saw_error_event {
                    end.saw_error_event = scanner.feed(&bytes);
                }
                let send_deadline = deadline.min(tokio::time::Instant::now() + timeouts.send_stall);
                match tokio::time::timeout_at(send_deadline, tx.send(Ok(bytes))).await {
                    Ok(Ok(())) => {}
                    Ok(Err(_)) => {
                        end.client_disconnect = true;
                        return Ok(());
                    }
                    Err(_) => {
                        // A slow consumer is not a worker fault.
                        end.client_disconnect = true;
                        return Err(std::io::Error::other("SSE downstream stalled"));
                    }
                }
            }
        };
        let result = match AssertUnwindSafe(pump).catch_unwind().await {
            Ok(result) => result,
            Err(payload) => {
                end.transport_ok = false;
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
    let terminal = futures::stream::once(async {
        match terminal_rx.await {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(Err(error)),
            Err(_) => Some(Err(std::io::Error::other("SSE pump cancelled"))),
        }
    })
    .filter_map(futures::future::ready);
    Body::from_stream(ReceiverStream::new(rx).chain(terminal))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures::stream;
    use http_body_util::BodyExt;

    fn timed_body<S>(
        stream: S,
        timeouts: StreamTimeouts,
    ) -> (Body, tokio::sync::oneshot::Receiver<StreamEnd>)
    where
        S: futures::Stream<Item = Result<Bytes, std::io::Error>> + Send + Unpin + 'static,
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let body = bytes_stream_to_body_with_timeouts(
            stream,
            None,
            Some(Box::new(move |end| {
                let _ = tx.send(end);
            })),
            None,
            timeouts,
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
        let (body, end) = timed_body(
            stream::pending(),
            StreamTimeouts {
                idle: Duration::from_secs(1),
                ..Default::default()
            },
        );
        assert!(body
            .collect()
            .await
            .unwrap_err()
            .to_string()
            .contains("idle timeout"));
        let end = end.await.unwrap();
        assert!(!end.transport_ok);
        assert!(!end.client_disconnect);
    }

    #[tokio::test(start_paused = true)]
    async fn full_queue_does_not_block_cleanup_or_hide_terminal_error() {
        let chunks = stream::repeat_with(|| Ok(Bytes::from_static(b"chunk")));
        let (body, end) = timed_body(
            chunks,
            StreamTimeouts {
                send_stall: Duration::from_secs(1),
                ..Default::default()
            },
        );
        let end = end.await.unwrap();
        assert!(end.client_disconnect);
        assert!(end.transport_ok);
        assert!(body
            .collect()
            .await
            .unwrap_err()
            .to_string()
            .contains("downstream stalled"));
    }

    #[tokio::test(start_paused = true)]
    async fn total_timeout_stops_a_stream_that_keeps_producing() {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(100)).await;
                if tx.send(Ok(Bytes::from_static(b"chunk"))).await.is_err() {
                    break;
                }
            }
        });
        let (body, end) = timed_body(
            ReceiverStream::new(rx),
            StreamTimeouts {
                idle: Duration::from_millis(200),
                total: Duration::from_secs(1),
                ..Default::default()
            },
        );
        assert!(body
            .collect()
            .await
            .unwrap_err()
            .to_string()
            .contains("total timeout"));
        assert!(!end.await.unwrap().transport_ok);
    }

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
        assert!(end.transport_ok);
        assert!(end.saw_error_event);
        assert!(!end.client_disconnect);
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
        assert!(!end.transport_ok);
        assert!(end.saw_error_event);
    }

    #[tokio::test]
    async fn completion_reports_upstream_error() {
        let chunks = vec![Err(std::io::Error::other("upstream failed"))];
        let (body, completion) = body_with_completion(chunks);
        let _ = body.collect().await;
        let end = stream_end(completion).await;
        assert!(!end.transport_ok);
        assert!(!end.saw_error_event);
        assert!(!end.client_disconnect);
    }

    #[tokio::test]
    async fn completion_reports_clean_end() {
        let chunks = vec![Ok::<Bytes, std::io::Error>(Bytes::from_static(
            b"data: [DONE]\n\n",
        ))];
        let (body, completion) = body_with_completion(chunks);
        let _ = body.collect().await.unwrap();
        let end = stream_end(completion).await;
        assert!(end.transport_ok);
        assert!(!end.saw_error_event);
        assert!(!end.client_disconnect);
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
        assert!(end.transport_ok);
        assert!(end.client_disconnect);
    }
}
