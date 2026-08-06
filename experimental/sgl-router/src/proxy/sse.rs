// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SSE passthrough — bridges a reqwest `bytes_stream()` into an axum Body.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use axum::body::Body;
use bytes::Bytes;
use futures::{FutureExt, StreamExt};
use tokio_stream::wrappers::ReceiverStream;

/// Whether a raw upstream SSE chunk carries an actual content token — used to
/// time `sgl_router_ttft_seconds` to the first GENERATED token rather than the
/// first byte off the wire.
///
/// SGLang's OpenAI-compatible stream opens with a role-prelude frame
/// (`{"choices":[{"delta":{"role":"assistant"}}]}`) and may interleave
/// empty-delta / finish-reason-only frames, the `data: [DONE]` sentinel, and
/// SSE comment / keepalive lines — none of which is a generated token. This
/// scans every `data:` line in the chunk and returns `true` on the first whose
/// JSON `choices[].delta` carries a non-empty `content` / `reasoning_content`
/// string OR a non-empty `tool_calls` array — Dynamo's "any generated token"
/// semantic. A tool-calling-first response emits `delta.tool_calls` with no
/// `content` / `reasoning_content`, so omitting it would record ZERO TTFT for
/// that whole (common) request class. Mirrors Dynamo's `adapter.rs` gate
/// ("Token-less chunks — SGLang's bootstrap handshake — don't count").
///
/// A single `bytes_stream()` chunk may pack several SSE events (TCP framing is
/// arbitrary) or split one mid-line. The scan is STATELESS and best-effort: it
/// never buffers across chunks — a non-JSON / partial `data:` payload simply
/// doesn't count (it never panics), and the next chunk gets a fresh look. So a
/// content frame split mid-JSON across two chunks does not fire on either half;
/// the hook fires on the next WHOLE content frame instead (TTFT clocks one
/// frame late — a sub-ms skew not worth cross-chunk buffering on the hot path).
/// If the only content frame is split and no later whole frame arrives, no
/// TTFT is recorded. This is the only place the router inspects SSE chunk
/// content; the bytes are still forwarded to the client unchanged.
fn sse_chunk_has_content_token(chunk: &[u8]) -> bool {
    // SSE events are newline-delimited; `data:` carries the JSON payload.
    for line in chunk.split(|&b| b == b'\n') {
        let line = trim_ascii_ws(line);
        let Some(payload) = line.strip_prefix(b"data:") else {
            continue; // comment (`:`), `event:`/`id:` fields, blank separators
        };
        let payload = trim_ascii_ws(payload);
        if payload == b"[DONE]" {
            continue;
        }
        let Ok(value) = serde_json::from_slice::<serde_json::Value>(payload) else {
            continue; // partial / non-JSON frame — wait for a later chunk
        };
        let has_token = value
            .get("choices")
            .and_then(|c| c.as_array())
            .is_some_and(|choices| {
                choices.iter().any(|choice| {
                    choice.get("delta").is_some_and(|delta| {
                        nonempty_str_field(delta, "content")
                            || nonempty_str_field(delta, "reasoning_content")
                            || delta
                                .get("tool_calls")
                                .and_then(|v| v.as_array())
                                .is_some_and(|a| !a.is_empty())
                    })
                })
            });
        if has_token {
            return true;
        }
    }
    false
}

/// Whether `value[key]` is a non-empty JSON string. Null / empty-string /
/// missing / non-string all return `false` — an empty `content` delta is not a
/// generated token.
fn nonempty_str_field(value: &serde_json::Value, key: &str) -> bool {
    value
        .get(key)
        .and_then(|v| v.as_str())
        .is_some_and(|s| !s.is_empty())
}

/// Trim leading/trailing ASCII whitespace (`u8::is_ascii_whitespace`) from a
/// byte slice. `[u8]::trim_ascii` is stable only on newer toolchains, so this
/// keeps the MSRV unconstrained.
fn trim_ascii_ws(mut s: &[u8]) -> &[u8] {
    while let [first, rest @ ..] = s {
        if first.is_ascii_whitespace() {
            s = rest;
        } else {
            break;
        }
    }
    while let [rest @ .., last] = s {
        if last.is_ascii_whitespace() {
            s = rest;
        } else {
            break;
        }
    }
    s
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
/// When `on_complete` is `Some`, the closure runs exactly once when the
/// pump task finishes. The bool argument is `true` on clean stream end
/// (including a clean client disconnect after at least the headers
/// landed cleanly), `false` on upstream stream error or pump panic.
/// `forward_streaming_to` passes a closure that records the worker's
/// circuit-breaker outcome — without this hook, a worker that returns
/// 2xx headers and then drops the stream mid-flight would stay credited
/// as healthy.
///
/// # First-token hook
/// When `on_first_token` is `Some`, the closure runs exactly once, the moment
/// the first upstream chunk carrying an actual content token is read — i.e.
/// time to first token. A chunk counts only when its parsed SSE `data:`
/// payload has a non-empty `content` (or `reasoning_content`) delta; the
/// role-prelude / empty-delta / `[DONE]` / keepalive frames SGLang emits
/// before the first generated token are skipped (see
/// [`sse_chunk_has_content_token`]). The hook does NOT fire if the stream ends
/// or errors before any content token arrives (client abort / drop before the
/// first token), so such requests record no TTFT sample.
/// `forward_streaming_to` passes a closure that records
/// `sgl_router_ttft_seconds` for successful streaming responses. This mirrors
/// Dynamo's `request_guard` (`observe_tokens` records TTFT only when
/// `new_tokens > 0`).
pub fn bytes_stream_to_body<S, E>(
    stream: S,
    stream_guards: Option<Box<dyn Send + 'static>>,
    on_complete: Option<Box<dyn FnOnce(bool) + Send + 'static>>,
    on_first_token: Option<Box<dyn FnOnce() + Send + 'static>>,
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
            let mut on_first_token = on_first_token;
            let mut s = stream;
            while let Some(chunk) = s.next().await {
                let item: Result<Bytes, std::io::Error> = chunk.map_err(|e| {
                    let msg = e.to_string();
                    tracing::warn!(error = %msg, "upstream SSE stream errored mid-flight");
                    std::io::Error::other(msg)
                });
                let is_err_chunk = item.is_err();
                // Fire the time-to-first-token hook on the first upstream chunk
                // that carries an actual content token — NOT merely the first
                // `Ok` chunk. SGLang's stream opens with a role-prelude frame
                // (and may emit empty-delta / keepalive frames) that carry no
                // generated token; timing TTFT to those would under-report by
                // one engine round-trip. `take()` makes it fire at most once;
                // a stream that errors or ends before any content frame leaves
                // it unfired (and dropped on task end) → no TTFT sample, which
                // is correct for a client abort / drop before the first token.
                if let Ok(bytes) = &item {
                    if on_first_token.is_some() && sse_chunk_has_content_token(bytes) {
                        if let Some(hook) = on_first_token.take() {
                            hook();
                        }
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
    async fn on_first_token_fires_once_on_first_content_chunk() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let fired = Arc::new(AtomicUsize::new(0));
        let fired_c = Arc::clone(&fired);
        // A real SGLang stream: a role-prelude frame carrying no content
        // token, then the first content frame, then a second content frame.
        // The hook must fire exactly once, timed to the FIRST content frame.
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
            )),
            Ok(Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
            )),
            Ok(Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
            )),
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
            "first-token hook must fire exactly once, on the first content frame",
        );
    }

    #[tokio::test]
    async fn on_first_token_not_fired_when_stream_errors_first() {
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
            "first-token hook must not fire when no Ok chunk is ever produced",
        );
    }

    #[tokio::test]
    async fn on_first_token_not_fired_when_no_content_chunk() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let fired = Arc::new(AtomicUsize::new(0));
        let fired_c = Arc::clone(&fired);
        // A stream that produces ONLY a role-prelude frame and then the
        // terminal `[DONE]` — no content token ever (client aborted / the
        // request dropped before the first generated token). TTFT must NOT
        // be recorded for such a request.
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            )),
            Ok(Bytes::from_static(b"data: [DONE]\n\n")),
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
            0,
            "first-token hook must not fire for a stream that carries no content token",
        );
    }

    #[test]
    fn sse_chunk_has_content_token_detects_content_delta() {
        // The first generated token: a non-empty `content` delta.
        assert!(sse_chunk_has_content_token(
            br#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#
        ));
        // `reasoning_content` (SGLang thinking models) also counts as a token.
        assert!(sse_chunk_has_content_token(
            br#"data: {"choices":[{"delta":{"reasoning_content":"Let me think"}}]}"#
        ));
        // A tool-calling-first response emits `delta.tool_calls` with NO
        // content/reasoning_content — it IS a generated token (Dynamo's "any
        // generated token" semantic). Without this the router would record
        // ZERO TTFT for the entire pure-tool-call request class.
        assert!(sse_chunk_has_content_token(
            br#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"get_weather"}}]}}]}"#
        ));
        // A chunk that bundles a role prelude and the first content token in
        // one TCP read still counts (scan all `data:` lines).
        assert!(sse_chunk_has_content_token(
            b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"
        ));
    }

    #[test]
    fn sse_chunk_has_content_token_skips_non_content_frames() {
        // Role-prelude frame — no token yet.
        assert!(!sse_chunk_has_content_token(
            br#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#
        ));
        // Empty-string content (some engines emit a leading empty delta).
        assert!(!sse_chunk_has_content_token(
            br#"data: {"choices":[{"delta":{"content":""}}]}"#
        ));
        // Null content.
        assert!(!sse_chunk_has_content_token(
            br#"data: {"choices":[{"delta":{"content":null}}]}"#
        ));
        // Terminal sentinel.
        assert!(!sse_chunk_has_content_token(b"data: [DONE]"));
        // SSE comment / keepalive line.
        assert!(!sse_chunk_has_content_token(b": keepalive"));
        // Finish-reason-only frame (no content delta).
        assert!(!sse_chunk_has_content_token(
            br#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#
        ));
        // Empty `tool_calls` array — not a generated token (some engines emit
        // an empty array on the prelude frame).
        assert!(!sse_chunk_has_content_token(
            br#"data: {"choices":[{"delta":{"tool_calls":[]}}]}"#
        ));
        // Non-JSON / partial frame must not panic and must not count.
        assert!(!sse_chunk_has_content_token(b"data: {not json"));
        assert!(!sse_chunk_has_content_token(b""));
    }

    /// End-to-end wiring: drive the pump with a REAL `MetricsRegistry`-backed
    /// hook and assert the rendered `sgl_router_ttft_seconds_count` series. This
    /// is the exact seam a sibling-gateway bug hid in — both the detector and
    /// the pump were individually green while the integrated path recorded the
    /// wrong count. A role-prelude + content stream must record EXACTLY ONE
    /// sample.
    #[tokio::test]
    async fn ttft_recorded_once_end_to_end_for_content_stream() {
        use crate::server::metrics::MetricsRegistry;

        let reg = MetricsRegistry::new();
        let reg_hook = Arc::clone(&reg);
        let start = std::time::Instant::now();
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            )),
            Ok(Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n",
            )),
            Ok(Bytes::from_static(b"data: [DONE]\n\n")),
        ];
        let s = stream::iter(chunks);
        let body = bytes_stream_to_body(
            s,
            None,
            None,
            Some(Box::new(move || {
                reg_hook.observe_ttft("m", start.elapsed().as_secs_f64());
            })),
        );
        let _ = body.collect().await.unwrap();
        let out = reg.render();
        assert!(
            out.contains(r#"sgl_router_ttft_seconds_count{model_id="m"} 1"#),
            "expected exactly one TTFT sample recorded end-to-end; got:\n{out}",
        );
    }

    /// End-to-end wiring: a stream that opens with a role prelude and then ends
    /// at `[DONE]` with no content token (client abort / drop before the first
    /// token) must record NO TTFT sample — there is no `_count` series for the
    /// model at all.
    #[tokio::test]
    async fn ttft_not_recorded_end_to_end_when_no_content() {
        use crate::server::metrics::MetricsRegistry;

        let reg = MetricsRegistry::new();
        let reg_hook = Arc::clone(&reg);
        let start = std::time::Instant::now();
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            )),
            Ok(Bytes::from_static(b"data: [DONE]\n\n")),
        ];
        let s = stream::iter(chunks);
        let body = bytes_stream_to_body(
            s,
            None,
            None,
            Some(Box::new(move || {
                reg_hook.observe_ttft("m", start.elapsed().as_secs_f64());
            })),
        );
        let _ = body.collect().await.unwrap();
        let out = reg.render();
        assert!(
            !out.contains(r#"sgl_router_ttft_seconds_count{model_id="m"}"#),
            "a content-less stream must record no TTFT series; got:\n{out}",
        );
    }

    /// Frame-split-across-chunks contract (item 3). `sse_chunk_has_content_token`
    /// is intentionally STATELESS — it never buffers across chunks (the sub-ms
    /// skew isn't worth hot-path buffering complexity). When the first content
    /// frame is split mid-JSON across two `bytes` chunks, the split halves are
    /// each non-JSON and do NOT fire the hook. This pins the "re-look the next
    /// chunk" contract so a future refactor can't silently make it never-fire:
    /// a LATER whole content frame still fires it (clocks one frame late).
    #[tokio::test]
    async fn split_first_content_frame_fires_on_next_whole_frame() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let fired = Arc::new(AtomicUsize::new(0));
        let fired_c = Arc::clone(&fired);
        // First content frame split mid-JSON across two chunks (neither half
        // parses), then a SECOND whole content frame arrives.
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"con",
            )),
            Ok(Bytes::from_static(b"tent\":\"Hi\"}}]}\n\n")),
            Ok(Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"content\":\" there\"}}]}\n\n",
            )),
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
            "split frame must not fire; the next whole content frame fires exactly once",
        );
    }

    /// Companion to the split-frame test: when the ONLY content frame is split
    /// across chunks and no later whole content frame arrives, the stateless
    /// detector never sees a parseable content frame, so the hook does NOT fire
    /// (0 samples). Documents the best-effort edge — acceptable because the
    /// alternative (cross-chunk buffering) costs hot-path complexity for a
    /// sub-ms timing gain on a rare framing split.
    #[tokio::test]
    async fn split_only_content_frame_does_not_fire() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let fired = Arc::new(AtomicUsize::new(0));
        let fired_c = Arc::clone(&fired);
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"con",
            )),
            Ok(Bytes::from_static(b"tent\":\"Hi\"}}]}\n\n")),
            Ok(Bytes::from_static(b"data: [DONE]\n\n")),
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
            0,
            "a single content frame split across chunks is best-effort: no whole frame, no fire",
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
}
