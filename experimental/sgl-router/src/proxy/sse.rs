// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SSE passthrough — bridges a reqwest `bytes_stream()` into an axum Body.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use axum::body::Body;
use bytes::Bytes;
use futures::{FutureExt, StreamExt};
use tokio_stream::wrappers::ReceiverStream;

/// How the SSE pump ended, reported to the `on_complete` hook. The fields
/// beyond `transport_ok` separate outcomes that are byte-level identical to
/// success: an in-band engine error under a committed 200, and a client that
/// walked away mid-generation.
#[derive(Debug, Clone, Copy)]
pub struct StreamEnd {
    /// No upstream stream error and no pump panic.
    pub transport_ok: bool,
    /// An in-band SSE error envelope (`data: {"error"...}`) rode the stream.
    pub saw_inband_error: bool,
    /// The client dropped the response body before upstream finished.
    pub client_disconnect: bool,
}

/// Carryover cap for the in-band scanner: a single longer line resets the
/// buffer — its detection is forfeited, but memory stays bounded.
const INBAND_SCAN_CARRYOVER_CAP: usize = 1 << 20; // 1 MiB

/// Incremental scanner for in-band SSE error envelopes — the shape sglang's
/// `create_streaming_error_response` emits after a committed 200. Complete
/// lines are scanned in place; only a trailing partial line is carried over.
/// The `{"error"` prefix check is parity-safe: inside a JSON string every
/// quote is escaped, so it cannot start a well-formed content payload.
#[derive(Default)]
struct InbandErrorScanner {
    carry: Vec<u8>,
    found: bool,
}

impl InbandErrorScanner {
    fn feed(&mut self, chunk: &[u8]) {
        if self.found {
            return;
        }
        let mut rest = chunk;
        // Finish the carried-over partial line first.
        if !self.carry.is_empty() {
            match rest.iter().position(|&b| b == b'\n') {
                Some(i) => {
                    self.carry.extend_from_slice(&rest[..i]);
                    if Self::line_is_inband_error(&self.carry) {
                        self.found = true;
                        self.carry = Vec::new();
                        return;
                    }
                    self.carry.clear();
                    rest = &rest[i + 1..];
                }
                None => {
                    self.carry.extend_from_slice(rest);
                    if self.carry.len() > INBAND_SCAN_CARRYOVER_CAP {
                        self.carry.clear();
                    }
                    return;
                }
            }
        }
        // Scan complete lines in place; keep only the trailing partial.
        while let Some(i) = rest.iter().position(|&b| b == b'\n') {
            if Self::line_is_inband_error(&rest[..i]) {
                self.found = true;
                return;
            }
            rest = &rest[i + 1..];
        }
        if rest.len() <= INBAND_SCAN_CARRYOVER_CAP {
            self.carry.extend_from_slice(rest);
        }
    }

    fn line_is_inband_error(line: &[u8]) -> bool {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let Some(mut payload) = line.strip_prefix(b"data:") else {
            return false;
        };
        while let Some(p) = payload.strip_prefix(b" ") {
            payload = p;
        }
        payload.starts_with(b"{\"error\"")
    }
}

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
/// When `on_complete` is `Some`, it runs exactly once when the pump task
/// finishes, receiving a [`StreamEnd`]. `forward_streaming_to` records the
/// worker's circuit-breaker outcome from `transport_ok` alone — an in-band
/// error is an application-level verdict, not a transport fault. The in-band
/// scan runs only when `on_complete` is installed.
///
/// # First-byte hook
/// When `on_first_byte` is `Some`, the closure runs exactly once, the moment
/// the first `Ok` chunk is read from the upstream stream — i.e. time to first
/// token. It does NOT fire if the stream ends or errors before any `Ok` chunk
/// arrives. `forward_streaming_to` passes a closure that records
/// `sgl_router_ttft_seconds` for successful streaming responses.
///
/// # Inter-chunk hook
/// When `on_inter_chunk` is `Some`, it runs once per non-empty `Ok` chunk
/// after the first with the gap (seconds) since the previous one — inter-token
/// latency as seen at the router. Gaps are between upstream ARRIVALS, so the
/// reading is engine pacing, not client drain speed, while the 64-slot
/// channel has room. Feeds `sgl_router_itl_seconds`.
pub fn bytes_stream_to_body<S, E>(
    stream: S,
    stream_guards: Option<Box<dyn Send + 'static>>,
    on_complete: Option<Box<dyn FnOnce(StreamEnd) + Send + 'static>>,
    on_first_byte: Option<Box<dyn FnOnce() + Send + 'static>>,
    on_inter_chunk: Option<Box<dyn Fn(f64) + Send + 'static>>,
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
        let outcome_holder = Arc::new(parking_lot::Mutex::new(StreamEnd {
            transport_ok: true,
            saw_inband_error: false,
            client_disconnect: false,
        }));
        let outcome_setter = Arc::clone(&outcome_holder);
        // The in-band scan exists solely to inform `on_complete`; skip the
        // per-chunk work entirely when nobody is listening.
        let mut scanner = on_complete.as_ref().map(|_| InbandErrorScanner::default());
        let pump = AssertUnwindSafe(async move {
            // Hold the guards for the task's lifetime — dropped when this
            // block exits (stream done or client disconnect).  Leading
            // underscore suppresses the "unused variable" lint while
            // keeping intent explicit.
            let _hold = stream_guards;
            let mut on_first_byte = on_first_byte;
            let mut prev_chunk_at: Option<std::time::Instant> = None;
            let mut s = stream;
            while let Some(chunk) = s.next().await {
                let item: Result<Bytes, std::io::Error> = chunk.map_err(|e| {
                    let msg = e.to_string();
                    tracing::warn!(error = %msg, "upstream SSE stream errored mid-flight");
                    std::io::Error::other(msg)
                });
                let is_err_chunk = item.is_err();
                match &item {
                    Ok(bytes) => {
                        // TTFT hook: at most once (`take()`); an error-first
                        // stream never produced a token, so it stays unfired.
                        if let Some(hook) = on_first_byte.take() {
                            hook();
                        }
                        // ITL: gap between non-empty Ok-chunk arrivals; the
                        // first chunk seeds the clock (its latency is TTFT).
                        if !bytes.is_empty() {
                            if let Some(hook) = on_inter_chunk.as_ref() {
                                let now = std::time::Instant::now();
                                if let Some(prev) = prev_chunk_at {
                                    hook(now.duration_since(prev).as_secs_f64());
                                }
                                prev_chunk_at = Some(now);
                            }
                        }
                        if let Some(scan) = scanner.as_mut() {
                            scan.feed(bytes);
                            if scan.found {
                                outcome_setter.lock().saw_inband_error = true;
                                // Verdict is sticky; stop scanning.
                                scanner = None;
                            }
                        }
                    }
                    Err(_) => outcome_setter.lock().transport_ok = false,
                }
                if tx.send(item).await.is_err() {
                    // Receiver dropped. If we were about to ship an upstream
                    // error there's nothing left to report; otherwise this is
                    // a clean client-side disconnect — log at debug since it's
                    // not a router-side fault.
                    if !is_err_chunk {
                        tracing::debug!("SSE client disconnected mid-stream");
                        outcome_setter.lock().client_disconnect = true;
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
            let mut end = *outcome_holder.lock();
            end.transport_ok = end.transport_ok && !panicked;
            hook(end);
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
        let body = bytes_stream_to_body(s, None, None, None, None);
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
            None,
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
            None,
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
        let body = bytes_stream_to_body(s, None, None, None, None);
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
        let body = bytes_stream_to_body(s, None, None, None, None);
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
        let body = bytes_stream_to_body(s, None, None, None, None);
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
        let body = bytes_stream_to_body(stream, None, None, None, None);

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

    #[test]
    fn inband_scanner_detects_engine_error_event() {
        // Exact shape sglang's create_streaming_error_response emits.
        let mut s = InbandErrorScanner::default();
        s.feed(b"data: {\"choices\": [{\"delta\": {\"content\": \"hi\"}}]}\n\n");
        assert!(!s.found);
        s.feed(b"data: {\"error\": {\"message\": \"queue is full\", \"code\": 503}}\n\n");
        assert!(s.found, "must detect a well-formed in-band error event");
    }

    #[test]
    fn inband_scanner_detects_error_split_across_chunks() {
        // Network chunking can split an SSE event anywhere — including inside
        // the `data: {"error"` prefix itself.
        let mut s = InbandErrorScanner::default();
        s.feed(b"data: {\"err");
        assert!(!s.found);
        s.feed(b"or\": {\"code\": 503}}\n\n");
        assert!(s.found, "must detect an error event split across chunks");
    }

    #[test]
    fn inband_scanner_tolerates_no_space_and_crlf() {
        // SSE permits `data:` with no space; proxies may normalize to CRLF.
        let mut s = InbandErrorScanner::default();
        s.feed(b"data:{\"error\": {\"code\": 500}}\r\n");
        assert!(s.found, "must handle data: without space and CRLF endings");
    }

    #[test]
    fn inband_scanner_ignores_error_text_inside_content() {
        // A model that TALKS about errors must not trip the scanner — escaped
        // quotes mean `{"error"` cannot start a content payload.
        let mut s = InbandErrorScanner::default();
        s.feed(
            b"data: {\"choices\": [{\"delta\": {\"content\": \"data: {\\\"error\\\" is how it looks\"}}]}\n\n",
        );
        assert!(
            !s.found,
            "escaped quotes in content must not false-positive"
        );
        s.feed(b"data: [DONE]\n\n");
        assert!(!s.found);
    }

    #[test]
    fn inband_scanner_bounds_carryover_on_pathological_line() {
        // A single line longer than the cap resets the buffer (detection of
        // that one line is forfeited by design), and scanning recovers after.
        let mut s = InbandErrorScanner::default();
        let big = vec![b'x'; INBAND_SCAN_CARRYOVER_CAP + 1024];
        s.feed(&big);
        assert!(s.carry.len() <= INBAND_SCAN_CARRYOVER_CAP);
        assert!(!s.found);
        s.feed(b"\ndata: {\"error\": {\"code\": 503}}\n");
        assert!(s.found, "scanner must recover after a pathological line");
    }

    /// Poll until the completion hook has fired (it runs on the spawned pump
    /// task, after the body is fully collected).
    async fn wait_for_stream_end(seen: &Arc<std::sync::Mutex<Option<StreamEnd>>>) -> StreamEnd {
        for _ in 0..200 {
            if let Some(end) = *seen.lock().unwrap() {
                return end;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        panic!("on_complete never fired");
    }

    #[tokio::test]
    async fn on_complete_reports_inband_error_with_clean_transport() {
        let seen = Arc::new(std::sync::Mutex::new(None));
        let seen_c = Arc::clone(&seen);
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(
                b"data: {\"choices\": [{\"delta\": {\"content\": \"partial\"}}]}\n\n",
            )),
            Ok(Bytes::from_static(
                b"data: {\"error\": {\"message\": \"aborted\", \"code\": 503}}\n\n",
            )),
            Ok(Bytes::from_static(b"data: [DONE]\n\n")),
        ];
        let body = bytes_stream_to_body(
            stream::iter(chunks),
            None,
            Some(Box::new(move |end| {
                *seen_c.lock().unwrap() = Some(end);
            })),
            None,
            None,
        );
        let _ = body.collect().await.unwrap();
        let end = wait_for_stream_end(&seen).await;
        assert!(end.transport_ok, "clean close: transport is fine");
        assert!(end.saw_inband_error, "in-band error must be reported");
        assert!(!end.client_disconnect);
    }

    #[tokio::test]
    async fn on_complete_reports_clean_success() {
        let seen = Arc::new(std::sync::Mutex::new(None));
        let seen_c = Arc::clone(&seen);
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(
                b"data: {\"choices\": [{\"delta\": {\"content\": \"hello\"}}]}\n\n",
            )),
            Ok(Bytes::from_static(b"data: [DONE]\n\n")),
        ];
        let body = bytes_stream_to_body(
            stream::iter(chunks),
            None,
            Some(Box::new(move |end| {
                *seen_c.lock().unwrap() = Some(end);
            })),
            None,
            None,
        );
        let _ = body.collect().await.unwrap();
        let end = wait_for_stream_end(&seen).await;
        assert!(end.transport_ok);
        assert!(!end.saw_inband_error);
        assert!(!end.client_disconnect);
    }

    #[tokio::test]
    async fn on_complete_reports_client_disconnect() {
        let seen = Arc::new(std::sync::Mutex::new(None));
        let seen_c = Arc::clone(&seen);
        // Enough chunks to outlast the 64-slot channel, so the pump is still
        // sending when the receiver is dropped.
        let chunks: Vec<Result<Bytes, std::io::Error>> =
            std::iter::repeat_with(|| Ok(Bytes::from_static(b"data: x\n\n")))
                .take(1000)
                .collect();
        let body = bytes_stream_to_body(
            stream::iter(chunks),
            None,
            Some(Box::new(move |end| {
                *seen_c.lock().unwrap() = Some(end);
            })),
            None,
            None,
        );
        let mut data_stream = body.into_data_stream();
        let _ = data_stream.next().await;
        drop(data_stream);
        let end = wait_for_stream_end(&seen).await;
        assert!(end.transport_ok, "a walk-away client is not a worker fault");
        assert!(end.client_disconnect, "disconnect must be reported");
    }

    #[tokio::test]
    async fn on_inter_chunk_reports_one_gap_per_nonempty_chunk_after_first() {
        let gaps: Arc<std::sync::Mutex<Vec<f64>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let gaps_c = Arc::clone(&gaps);
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: a\n\n")),
            Ok(Bytes::new()), // empty: no token, must not report or reset the clock
            Ok(Bytes::from_static(b"data: b\n\n")),
            Ok(Bytes::from_static(b"data: [DONE]\n\n")),
        ];
        let body = bytes_stream_to_body(
            stream::iter(chunks),
            None,
            None,
            None,
            Some(Box::new(move |gap| {
                gaps_c.lock().unwrap().push(gap);
            })),
        );
        let _ = body.collect().await.unwrap();
        // The hook fires from the spawned pump task; poll briefly.
        for _ in 0..200 {
            if gaps.lock().unwrap().len() == 2 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        let gaps = gaps.lock().unwrap();
        assert_eq!(
            gaps.len(),
            2,
            "3 non-empty chunks must report exactly 2 gaps; got {gaps:?}",
        );
        assert!(gaps.iter().all(|g| g.is_finite() && *g >= 0.0));
    }
}
