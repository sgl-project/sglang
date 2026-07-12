// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SSE passthrough — bridges a reqwest `bytes_stream()` into an axum Body.

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use bytes::Bytes;
use futures::{FutureExt, StreamExt};

use crate::proxy::AbortReason;

/// Sampling counter for the diagnostic `sse_pump_timing` log. ~1-in-`SAMPLE`
/// pump completions are logged with their first-byte / drain / exit timing and
/// exit reason, so we can see whether an admission slot lingers in the pump
/// (engine streaming slowly, or the task not being polled) without flooding.
static PUMP_LOG_COUNTER: AtomicU64 = AtomicU64::new(0);
const PUMP_LOG_SAMPLE: u64 = 64;

/// Max time the pump waits to hand a chunk to the client before giving up.
///
/// A client that reads the response headers, stops reading the body, but never
/// closes the connection lets the read-ahead buffer fill and parks the pump
/// acquiring read-ahead permits. The upstream idle timeout cannot rescue this:
/// it is only armed while the pump polls `s.next()`, not while it blocks on the
/// permit acquire. Without this bound the pump parks forever, its
/// `stream_guards` (the per-worker admission slot + active-load entry) never
/// drop, and the in-flight counter leaks until every worker is pinned at its cap
/// and all traffic is shed while the engines sit idle. The default mirrors the
/// upstream idle timeout: a client that accepts no bytes for that long is
/// treated as gone.
///
/// Default client-backpressure stall budget, used when a `Proxy` is built
/// without config (tests) and as the `bytes_stream_to_body` wrapper's value.
/// Prod overrides it per-request via `bytes_stream_to_body_with_stall`
/// (see `ProxyConfig::stream_send_stall_secs`). Must match
/// `config::default_stream_send_stall_secs`.
pub(crate) const STREAM_SEND_STALL: std::time::Duration = std::time::Duration::from_secs(180);

/// Per-stream read-ahead budget, in bytes.
///
/// The pump reads engine→client through a byte-bounded read-ahead buffer: it may
/// race ahead of a slow client by up to this many buffered bytes. This is what
/// decouples the engine-read from the client-write so the `stream_guards` (the
/// per-worker admission slot + active-load entry) release when the ENGINE
/// finishes — not when the client finishes draining. With a per-worker cap, that
/// difference is the whole game: if slots only freed at client-read speed, a fast
/// engine would be paced by slow clients and the router would forward far below
/// engine capacity while the engine sat idle.
///
/// Any completion whose total size is ≤ this cap reaches upstream-`None` at the
/// engine's pace and drops its slot immediately (engine-done release), while the
/// cap still bounds worst-case per-stream memory. A completion LARGER than this
/// cap whose client has stalled re-engages backpressure: the pump blocks
/// acquiring read-ahead permits (and, if the stall persists, trips
/// `STREAM_SEND_STALL`).
const STREAM_READAHEAD_MAX_BYTES: usize = 1 << 20; // 1 MiB

// `acquire_many` takes a u32 and the per-chunk charge is `min(len, MAX)`, so the
// cap must fit in u32 for the `as u32` cast in the pump to be lossless.
const _: () = assert!(STREAM_READAHEAD_MAX_BYTES <= u32::MAX as usize);

/// How the SSE pump ended, reported to the `on_complete` hook.
///
/// `transport_ok` is the old boolean: `true` on clean stream end (including a
/// clean client disconnect after the headers landed), `false` on upstream
/// stream error or pump panic. The two extra fields let callers distinguish
/// outcomes that are byte-level indistinguishable from success: an engine
/// that commits `200 OK`, then reports failure as a well-formed in-band
/// `data: {"error":...}` event and closes cleanly, and a client that walked
/// away mid-generation.
#[derive(Debug, Clone, Copy)]
pub struct StreamEnd {
    /// Byte-level stream health: no upstream error, no pump panic.
    pub transport_ok: bool,
    /// An in-band SSE error envelope (`data: {"error"...}`) was observed.
    pub saw_inband_error: bool,
    /// The client dropped the response body before upstream finished.
    pub client_disconnect: bool,
}

/// Carryover cap for the in-band error scanner (mirrors the gateway's
/// sseClassifyingBody bound). A single line longer than this without a
/// newline resets the buffer — detection of that one pathological line is
/// lost (the stream still passes through unchanged), but memory stays
/// bounded.
const INBAND_SCAN_CARRYOVER_CAP: usize = 1 << 20; // 1 MiB

/// Incremental, allocation-light scanner for in-band SSE error envelopes.
///
/// The engine commits `200 OK` at first byte; a failure after that point can
/// only be reported in-band, as `data: {"error": ...}\n\n` (see sglang's
/// `create_streaming_error_response`). This scanner watches the raw byte
/// stream for that shape: it splits on `\n`, strips an optional `\r`, matches
/// a `data:` prefix, skips leading spaces, and checks whether the payload
/// starts with `{"error"`. The prefix check is parity-safe: inside any JSON
/// string value a quote is escaped (`\"`), so the raw byte sequence
/// `{"error"` cannot occur at the start of a well-formed data payload unless
/// it really is an error envelope.
///
/// Complete lines within a chunk are scanned in place; only a trailing
/// partial line is copied into the carryover, so the common case (whole SSE
/// events per chunk) does no per-chunk allocation.
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
/// Engine→client chunks flow through an **unbounded** channel paired with a
/// byte-bounded read-ahead buffer (a [`tokio::sync::Semaphore`] seeded with
/// `STREAM_READAHEAD_MAX_BYTES` permits): the pump acquires
/// `min(chunk_len, STREAM_READAHEAD_MAX_BYTES)` permits before sending each
/// chunk, and the client-facing stream returns the same count as it yields each
/// chunk. Total buffered bytes therefore stay ≤ `STREAM_READAHEAD_MAX_BYTES`,
/// plus at most one in-flight chunk that individually exceeds the cap (an
/// oversized chunk is charged the full cap, never more, to avoid deadlock).
///
/// # Backpressure note
/// The read-ahead buffer lets the pump race ahead of a slow client by up to
/// `STREAM_READAHEAD_MAX_BYTES`. The point is the **engine-done release**: for
/// any completion that fits within the read-ahead cap, the pump reaches the
/// upstream `None` at the engine's pace and drops its `stream_guards`
/// immediately, regardless of how slowly the client reads — so a per-worker
/// admission slot frees as fast as the engine finishes, not as fast as clients
/// drain. A completion that EXCEEDS the cap with a non-draining client still
/// backpressures: the pump blocks acquiring permits until the client consumes
/// (or `STREAM_SEND_STALL` trips), bounding worst-case per-stream memory.
///
/// # Client disconnect
/// When the axum Body is dropped the receiver is closed; the next `tx.send()` on
/// the unbounded channel then returns `Err`, which breaks the loop. If the pump
/// is instead blocked acquiring read-ahead permits when the client disconnects,
/// a `tokio::select!` on `tx.closed()` wakes it so it breaks rather than holding
/// the guards — no upstream bytes are read after the client disconnects.
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
/// pump finishes (stream exhausted, client disconnects, the client stalls past
/// `STREAM_SEND_STALL` without draining, or upstream errors).
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
/// pump task finishes, receiving a [`StreamEnd`] describing how the stream
/// ended: `transport_ok` is `true` on clean stream end (including a clean
/// client disconnect, or a non-draining client aborted by
/// `STREAM_SEND_STALL`, after at least the headers landed cleanly), `false`
/// on upstream stream error or pump panic; `saw_inband_error` reports whether
/// a well-formed `data: {"error"...}` SSE event passed through; and
/// `client_disconnect` reports whether the client dropped the body early
/// (including the stall-abort case).
/// `forward_streaming_to` passes a closure that records the worker's
/// circuit-breaker outcome (from `transport_ok` only — an in-band error is
/// an application-level verdict, not a transport fault) — without this
/// hook, a worker that returns 2xx headers and then drops the stream
/// mid-flight would stay credited as healthy. The in-band scan only runs
/// when `on_complete` is `Some`; a hook-less pump skips it entirely.
///
/// # First-byte hook
/// When `on_first_byte` is `Some`, the closure runs exactly once, the moment
/// the first `Ok` chunk is read from the upstream stream — i.e. time to first
/// token. It does NOT fire if the stream ends or errors before any `Ok` chunk
/// arrives. `forward_streaming_to` passes a closure that records
/// `sgl_router_ttft_seconds` for successful streaming responses.
///
/// # Inter-chunk hook
/// When `on_inter_chunk` is `Some`, the closure runs once per non-empty `Ok`
/// chunk AFTER the first, receiving the gap (seconds) since the previous
/// non-empty `Ok` chunk arrived — i.e. inter-token latency as seen at the
/// router. Gaps are measured between upstream ARRIVALS: while the read-ahead
/// buffer has budget the pump polls at the engine's pace, so the reading is
/// engine pacing, not client drain speed. (A stream that exceeds the
/// read-ahead cap with a slow client blocks the pump on permits, and that
/// wait leaks into the next gap — unavoidable in a pull model, and rare with
/// the 1 MiB budget.) `forward_streaming_to` passes a closure that records
/// `sgl_router_itl_seconds` for successful streaming responses.
pub fn bytes_stream_to_body<S, E>(
    stream: S,
    stream_guards: Option<Box<dyn Send + 'static>>,
    on_complete: Option<Box<dyn FnOnce(StreamEnd) + Send + 'static>>,
    on_first_byte: Option<Box<dyn FnOnce() + Send + 'static>>,
    abort_reason: Option<Arc<AtomicU8>>,
    on_inter_chunk: Option<Box<dyn Fn(f64) + Send + 'static>>,
) -> Body
where
    S: futures::Stream<Item = Result<Bytes, E>> + Send + Unpin + 'static,
    E: std::fmt::Display + Send + Sync + 'static,
{
    bytes_stream_to_body_with_stall(
        stream,
        stream_guards,
        on_complete,
        on_first_byte,
        abort_reason,
        on_inter_chunk,
        STREAM_SEND_STALL,
    )
}

/// Same as [`bytes_stream_to_body`] but with a caller-supplied client-backpressure
/// stall budget (`send_stall`) instead of the default [`STREAM_SEND_STALL`]. The
/// production forward path passes `Proxy::stream_send_stall` (from
/// `ProxyConfig::stream_send_stall_secs`); the no-arg wrapper above is kept for
/// tests and callers that just want the default.
#[allow(clippy::too_many_arguments)]
pub fn bytes_stream_to_body_with_stall<S, E>(
    stream: S,
    stream_guards: Option<Box<dyn Send + 'static>>,
    on_complete: Option<Box<dyn FnOnce(StreamEnd) + Send + 'static>>,
    on_first_byte: Option<Box<dyn FnOnce() + Send + 'static>>,
    abort_reason: Option<Arc<AtomicU8>>,
    on_inter_chunk: Option<Box<dyn Fn(f64) + Send + 'static>>,
    send_stall: std::time::Duration,
) -> Body
where
    S: futures::Stream<Item = Result<Bytes, E>> + Send + Unpin + 'static,
    E: std::fmt::Display + Send + Sync + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();
    // Byte-bounded read-ahead budget shared between the pump (acquires permits
    // before sending) and the client-facing stream (returns them as it yields).
    // Total buffered bytes stay ≤ STREAM_READAHEAD_MAX_BYTES; see the type doc.
    let readahead = Arc::new(tokio::sync::Semaphore::new(STREAM_READAHEAD_MAX_BYTES));
    let readahead_pump = Arc::clone(&readahead);
    // Helper: stamp the shared abort-reason handle inside `AbortOnDrop` before
    // the pump's `_hold` (which owns the guard) drops. Cheap when the pump
    // wasn't given a handle (`abort_reason` is `None` for non-streaming callers
    // like tests), lock-free when it was.
    fn set_abort_reason(handle: &Option<Arc<AtomicU8>>, reason: AbortReason) {
        if let Some(h) = handle.as_ref() {
            h.store(reason as u8, Ordering::Relaxed);
        }
    }
    // Local scope guard: on drop, stamp `StreamPumpPanicked` into the shared
    // abort-reason handle UNLESS `defuse()` has been called. Every normal
    // pump exit path defuses (its `set_abort_reason` at the break site is the
    // narrow reason; the marker's default would just clobber it). Only a
    // panic — where control flow never reaches the defuse — leaves the marker
    // armed, so the guard's `Drop` fires with the specific `StreamPumpPanicked`
    // label instead of the constructor's `StreamClientGone` default. Declared
    // AFTER `_hold` in the pump body so LIFO drop order runs this marker
    // BEFORE `_hold` (which owns the `AbortOnDrop`) drops — i.e. the reason is
    // committed before the guard reads it.
    struct PanicReasonMarker {
        handle: Option<Arc<AtomicU8>>,
        defused: bool,
    }
    impl PanicReasonMarker {
        fn defuse(&mut self) {
            self.defused = true;
        }
    }
    impl Drop for PanicReasonMarker {
        fn drop(&mut self) {
            if !self.defused {
                if let Some(h) = self.handle.as_ref() {
                    h.store(AbortReason::StreamPumpPanicked as u8, Ordering::Relaxed);
                }
            }
        }
    }
    let abort_reason_for_pump = abort_reason.clone();
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
            // Declared AFTER `_hold` so its Drop runs BEFORE `_hold` (LIFO).
            // See `PanicReasonMarker`'s type-level doc for why this ordering
            // matters. Mutable so `defuse()` can flip its flag before we
            // exit through any of the non-panic paths below.
            let mut panic_marker = PanicReasonMarker {
                handle: abort_reason_for_pump.clone(),
                defused: false,
            };
            // Diagnostic timing for this stream's lifetime. `task_start` ≈ when
            // upstream headers landed (the task is spawned right after). These
            // localize whether the admission slot (held by `_hold`) lingers
            // because the engine streams slowly or the task isn't being polled.
            let task_start = Instant::now();
            let mut first_byte_at: Option<Instant> = None;
            let mut prev_chunk_at: Option<Instant> = None;
            let mut n_chunks: u64 = 0;
            let mut n_bytes: u64 = 0;
            let mut exit_reason = "upstream_end";
            let mut on_first_byte = on_first_byte;
            let mut s = stream;
            while let Some(chunk) = s.next().await {
                let item: Result<Bytes, std::io::Error> = chunk.map_err(|e| {
                    let msg = e.to_string();
                    tracing::warn!(error = %msg, "upstream SSE stream errored mid-flight");
                    std::io::Error::other(msg)
                });
                let is_err_chunk = item.is_err();
                // An empty Ok chunk carries no bytes, so forwarding it is a
                // client-visible no-op — but it charges zero read-ahead permits
                // (the byte budget can't bound it) and would still occupy a slot
                // on the unbounded channel. Skip it so an upstream emitting a
                // flood of empty chunks can't grow the buffer without bound, and
                // so the first-byte hook fires on the first chunk that actually
                // carries a token.
                if matches!(&item, Ok(b) if b.is_empty()) {
                    continue;
                }
                n_chunks += 1;
                if let Ok(b) = &item {
                    n_bytes += b.len() as u64;
                    if first_byte_at.is_none() {
                        first_byte_at = Some(Instant::now());
                    }
                }
                // Fire the time-to-first-token hook on the first successful
                // chunk from upstream. `take()` makes it fire at most once;
                // an error-first stream never produced a token, so it's left
                // unfired (and dropped on task end).
                if !is_err_chunk {
                    if let Some(hook) = on_first_byte.take() {
                        hook();
                    }
                    // Inter-token latency: gap between successive non-empty
                    // Ok-chunk ARRIVALS. The first chunk seeds the clock (its
                    // latency is TTFT, not ITL); every later chunk reports the
                    // gap since its predecessor.
                    if let Some(hook) = on_inter_chunk.as_ref() {
                        let now = Instant::now();
                        if let Some(prev) = prev_chunk_at {
                            hook(now.duration_since(prev).as_secs_f64());
                        }
                        prev_chunk_at = Some(now);
                    }
                    if let (Some(scan), Ok(bytes)) = (scanner.as_mut(), &item) {
                        scan.feed(bytes);
                        if scan.found {
                            outcome_setter.lock().saw_inband_error = true;
                        }
                    }
                }
                if is_err_chunk {
                    outcome_setter.lock().transport_ok = false;
                }
                // Reserve read-ahead budget before sending. We charge
                // `min(chunk_len, MAX)` permits so a single chunk larger than the
                // whole budget can still proceed (it would otherwise deadlock
                // asking for more permits than exist); the client-facing stream
                // returns exactly the same count, keeping the semaphore balanced.
                // An error chunk carries no payload to buffer, so it bypasses the
                // budget and is sent directly.
                let want = item
                    .as_ref()
                    .map(|b| b.len().min(STREAM_READAHEAD_MAX_BYTES))
                    .unwrap_or(0) as u32;
                if want > 0 {
                    // Acquire read-ahead permits. The pump may race ahead of a
                    // slow client up to STREAM_READAHEAD_MAX_BYTES; only when the
                    // buffer is full does this block. Two safety valves:
                    //  - `tx.closed()` resolves when the receiver (client Body) is
                    //    dropped, so a disconnect while blocked here wakes us
                    //    rather than pinning the guards.
                    //  - `STREAM_SEND_STALL` caps how long we wait on a client
                    //    that stays connected but never drains (so it never frees
                    //    permits); on elapse we abort and release the slot.
                    let acquired = tokio::select! {
                        biased;
                        _ = tx.closed() => {
                            // Client disconnected while we were blocked on the
                            // budget — clean client-side cancel, not a fault.
                            tracing::debug!("SSE client disconnected mid-stream");
                            outcome_setter.lock().client_disconnect = true;
                            exit_reason = "client_disconnect_blocked";
                            set_abort_reason(
                                &abort_reason_for_pump,
                                AbortReason::StreamClientGone,
                            );
                            break;
                        }
                        res = tokio::time::timeout(
                            send_stall,
                            readahead_pump.acquire_many(want),
                        ) => res,
                    };
                    match acquired {
                        Ok(Ok(permit)) => {
                            // Hand the bytes off to the client-facing stream,
                            // which returns these permits as it yields the chunk.
                            permit.forget();
                        }
                        Ok(Err(_)) => {
                            // The semaphore is never closed while the pump holds a
                            // clone, so this is unreachable. Log loudly rather than
                            // break silently: if a future change ever closes it,
                            // this surfaces the contract violation instead of
                            // masquerading as a clean completion.
                            tracing::error!(
                                "SSE read-ahead semaphore closed unexpectedly; aborting pump"
                            );
                            exit_reason = "semaphore_closed";
                            // Treated as a pump-side invariant violation, not a
                            // client-side event: the same bucket as
                            // `StreamPumpPanicked` so it stands out separately
                            // from routine disconnects in metrics/logs.
                            set_abort_reason(
                                &abort_reason_for_pump,
                                AbortReason::StreamPumpPanicked,
                            );
                            break;
                        }
                        Err(_elapsed) => {
                            // Client accepted no bytes for STREAM_SEND_STALL while
                            // the connection stayed open. Stop pumping so `_hold`
                            // drops and the slot is released. We leave `outcome` at
                            // its `true` default, so `on_complete` records a breaker
                            // SUCCESS — the worker was demonstrably delivering when
                            // the client stalled, so it is not at fault and must not
                            // be penalized. But the stream is being truncated
                            // mid-completion, so we inject a loud `io::Error` (as the
                            // panic path does) rather than letting the body EOF
                            // cleanly: a client that resumes reading after the stall
                            // must be able to tell its response was cut short, not
                            // mistake a short body for a complete one.
                            tracing::warn!(
                                stall = ?send_stall,
                                "SSE downstream not draining; aborting to release in-flight slot"
                            );
                            let _ = tx.send(Err(std::io::Error::other(
                                "SSE downstream stalled; stream aborted before completion",
                            )));
                            // Classified as a client disconnect for
                            // stream-outcome purposes: the truncation is
                            // client-side (stopped draining), not a worker
                            // fault, so `transport_ok` stays true.
                            outcome_setter.lock().client_disconnect = true;
                            exit_reason = "downstream_stall";
                            set_abort_reason(
                                &abort_reason_for_pump,
                                AbortReason::StreamDownstreamStall,
                            );
                            break;
                        }
                    }
                }
                // Send on the unbounded channel. This only fails if the receiver
                // (client Body) was dropped: a clean client-side disconnect (or
                // nothing left to report if we were shipping an upstream error).
                // The permits charged above are not returned on this break path,
                // but that is harmless: the pump is the only acquirer and it is
                // exiting, so nothing waits on the semaphore again — both Arc
                // clones drop with the channel and the whole per-stream semaphore
                // is freed. Permit balance only matters on the steady-state path,
                // where the client stream returns each charge as it yields a chunk.
                if tx.send(item).is_err() {
                    if !is_err_chunk {
                        tracing::debug!("SSE client disconnected mid-stream");
                        outcome_setter.lock().client_disconnect = true;
                    }
                    exit_reason = "client_gone";
                    set_abort_reason(&abort_reason_for_pump, AbortReason::StreamClientGone);
                    break;
                }
                if is_err_chunk {
                    // Surfaced upstream error to client; stop reading.
                    exit_reason = "upstream_error";
                    break;
                }
            }
            // Every non-panic exit reaches this line. Defuse the marker so its
            // Drop is a no-op: whatever narrow reason the specific exit path
            // set (or the constructor default, for the "upstream_end" /
            // "upstream_error" cases where `mark_terminal` flipped
            // `reached_end` and the abort won't fire at all) stays as-is.
            panic_marker.defuse();
            // Diagnostic: this stream's lifetime, sampled. `pump_exit_ms` is how
            // long the admission slot (held by `_hold`, dropped when this block
            // exits) was occupied by the pump; compare to the engine's own
            // per-request latency to spot slots lingering long after the engine
            // finished. `first_byte_ms` is headers→first token.
            if PUMP_LOG_COUNTER
                .fetch_add(1, Ordering::Relaxed)
                .is_multiple_of(PUMP_LOG_SAMPLE)
            {
                let first_byte_ms =
                    first_byte_at.map(|t| t.duration_since(task_start).as_millis() as u64);
                tracing::debug!(
                    reason = exit_reason,
                    chunks = n_chunks,
                    bytes = n_bytes,
                    first_byte_ms = ?first_byte_ms,
                    pump_exit_ms = task_start.elapsed().as_millis() as u64,
                    "sse_pump_timing",
                );
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
            // The `PanicReasonMarker` declared inside the pump already stamped
            // `StreamPumpPanicked` into the abort-reason handle during unwind
            // (its Drop runs BEFORE `_hold`'s, by LIFO drop order), so by the
            // time we reach here the abort has already fired with the correct
            // reason. Nothing to write from the outer scope; documenting the
            // ordering here so nobody re-adds a stamp that would come too late.
            let _ = tx_for_panic.send(Err(std::io::Error::other(format!(
                "SSE pump panicked: {msg}"
            ))));
        }
        if let Some(hook) = on_complete {
            let mut end = *outcome_holder.lock();
            end.transport_ok = end.transport_ok && !panicked;
            hook(end);
        }
    });
    // Client-facing stream: as each chunk is yielded to the client, return the
    // read-ahead permits the pump charged for it (`min(len, MAX)`), freeing the
    // pump to read further ahead. Error chunks carry no payload, so they
    // returned nothing to release.
    let body_stream = futures::stream::unfold((rx, readahead), |(mut rx, sem)| async move {
        let item = rx.recv().await?;
        if let Ok(bytes) = &item {
            let release = bytes.len().min(STREAM_READAHEAD_MAX_BYTES);
            if release > 0 {
                sem.add_permits(release);
            }
        }
        Some((item, (rx, sem)))
    });
    Body::from_stream(body_stream)
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

/// Wrap a stream so `reached_end` flips to `true` the instant the stream yields
/// its terminal item — a clean `None` (the engine finished generating) or a
/// terminal `Err` (the engine failed, or the idle timeout tripped). The SSE
/// pump only polls through to that terminal when it runs to completion; if the
/// client disconnects or stalls, the pump breaks early and never observes it, so
/// the flag stays `false`.
///
/// An [`AbortOnDrop`](crate::proxy::AbortOnDrop) packed into the pump's
/// `stream_guards` reads this flag when it drops: flag still `false` ⇒ the
/// stream was torn down before the engine was done ⇒ the client is gone and the
/// engine is told to abort. Keeping the signal in one shared flag avoids
/// coupling the pump's loop to the abort machinery (the pump never inspects its
/// opaque guards).
pub fn mark_terminal<S>(
    stream: S,
    reached_end: Arc<std::sync::atomic::AtomicBool>,
) -> futures::stream::BoxStream<'static, Result<Bytes, std::io::Error>>
where
    S: futures::Stream<Item = Result<Bytes, std::io::Error>> + Send + Unpin + 'static,
{
    futures::stream::unfold((stream, reached_end), |(mut s, flag)| async move {
        match s.next().await {
            Some(item) => {
                // A terminal `Err` is the last item the pump reads (it breaks on
                // error), so the engine is done/failed: mark end now, before the
                // guard can drop, so a real upstream failure is not mistaken for
                // a client disconnect.
                if item.is_err() {
                    flag.store(true, Ordering::SeqCst);
                }
                Some((item, (s, flag)))
            }
            None => {
                flag.store(true, Ordering::SeqCst);
                None
            }
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
        let body = bytes_stream_to_body(s, None, None, None, None, None);
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
            None,
        );
        let _ = body.collect().await;
        assert_eq!(
            fired.load(Ordering::SeqCst),
            0,
            "first-byte hook must not fire when no Ok chunk is ever produced",
        );
    }

    /// The inter-chunk hook reports one gap per non-empty Ok chunk AFTER the
    /// first: N chunks → N-1 gaps (the first chunk's latency is TTFT, not
    /// ITL). Empty chunks are skipped by the pump before the hook, so they
    /// contribute no gap and don't reset the clock.
    #[tokio::test]
    async fn on_inter_chunk_reports_one_gap_per_chunk_after_first() {
        use std::sync::Mutex;

        let gaps = Arc::new(Mutex::new(Vec::<f64>::new()));
        let gaps_c = Arc::clone(&gaps);
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(b"a")),
            Ok(Bytes::new()), // empty: skipped, no gap, no clock reset
            Ok(Bytes::from_static(b"b")),
            Ok(Bytes::from_static(b"c")),
        ];
        let s = stream::iter(chunks);
        let body = bytes_stream_to_body(
            s,
            None,
            None,
            None,
            None,
            Some(Box::new(move |gap| {
                gaps_c.lock().unwrap().push(gap);
            })),
        );
        let _ = body.collect().await.unwrap();
        let gaps = gaps.lock().unwrap();
        assert_eq!(
            gaps.len(),
            2,
            "3 non-empty chunks must produce exactly 2 inter-chunk gaps; got {gaps:?}",
        );
        assert!(
            gaps.iter().all(|g| g.is_finite() && *g >= 0.0),
            "gaps must be finite and non-negative; got {gaps:?}",
        );
    }

    /// A single-chunk stream has no inter-chunk gap, and an error chunk is
    /// not a token — neither may fire the hook.
    #[tokio::test]
    async fn on_inter_chunk_not_fired_for_single_chunk_or_error() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let fired = Arc::new(AtomicUsize::new(0));
        let fired_c = Arc::clone(&fired);
        let chunks: Vec<Result<Bytes, std::io::Error>> = vec![
            Ok(Bytes::from_static(b"only")),
            Err(std::io::Error::other("upstream died mid-stream")),
        ];
        let s = stream::iter(chunks);
        let body = bytes_stream_to_body(
            s,
            None,
            None,
            None,
            None,
            Some(Box::new(move |_gap| {
                fired_c.fetch_add(1, Ordering::SeqCst);
            })),
        );
        let _ = body.collect().await;
        assert_eq!(
            fired.load(Ordering::SeqCst),
            0,
            "one Ok chunk followed by an error chunk has no token-to-token gap",
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
        // the `data: {"error"` prefix itself. The carryover must reassemble it.
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
        // A model that TALKS about errors must not trip the scanner: inside a
        // JSON string every quote is escaped, so the raw `{"error"` byte
        // sequence cannot appear at the start of a content payload.
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
        // A single line longer than the cap must reset the buffer, not grow
        // without bound. Detection of that one line is forfeited by design.
        let mut s = InbandErrorScanner::default();
        let big = vec![b'x'; INBAND_SCAN_CARRYOVER_CAP + 1024];
        s.feed(&big);
        assert!(s.carry.len() <= INBAND_SCAN_CARRYOVER_CAP);
        assert!(!s.found);
        // Scanner still works on subsequent, well-formed lines.
        s.feed(b"\ndata: {\"error\": {\"code\": 503}}\n");
        assert!(s.found, "scanner must recover after a pathological line");
    }

    #[tokio::test]
    async fn on_complete_reports_inband_error() {
        use std::sync::Mutex as StdMutex;
        let seen: Arc<StdMutex<Option<StreamEnd>>> = Arc::new(StdMutex::new(None));
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
            None,
        );
        let _ = body.collect().await.unwrap();
        // The hook fires from the spawned pump task; wait for it briefly.
        for _ in 0..50 {
            if seen.lock().unwrap().is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let end = seen.lock().unwrap().expect("on_complete must fire");
        assert!(end.transport_ok, "clean close: transport is fine");
        assert!(end.saw_inband_error, "in-band error must be reported");
        assert!(!end.client_disconnect);
    }

    #[tokio::test]
    async fn on_complete_reports_clean_success() {
        use std::sync::Mutex as StdMutex;
        let seen: Arc<StdMutex<Option<StreamEnd>>> = Arc::new(StdMutex::new(None));
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
            None,
        );
        let _ = body.collect().await.unwrap();
        for _ in 0..50 {
            if seen.lock().unwrap().is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let end = seen.lock().unwrap().expect("on_complete must fire");
        assert!(end.transport_ok);
        assert!(!end.saw_inband_error);
        assert!(!end.client_disconnect);
    }

    #[tokio::test]
    async fn on_complete_reports_client_disconnect() {
        use std::sync::Mutex as StdMutex;
        let seen: Arc<StdMutex<Option<StreamEnd>>> = Arc::new(StdMutex::new(None));
        let seen_c = Arc::clone(&seen);
        // More chunks than the 64-slot channel so the pump is still sending
        // when the client walks away.
        let chunks: Vec<Result<Bytes, std::io::Error>> = (0..500)
            .map(|_| Ok(Bytes::from_static(b"data: {\"choices\": []}\n\n")))
            .collect();
        let body = bytes_stream_to_body(
            stream::iter(chunks),
            None,
            Some(Box::new(move |end| {
                *seen_c.lock().unwrap() = Some(end);
            })),
            None,
            None,
            None,
        );
        let mut data_stream = body.into_data_stream();
        let first = data_stream.next().await;
        assert!(first.is_some());
        drop(data_stream); // client disconnect
        for _ in 0..50 {
            if seen.lock().unwrap().is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let end = seen.lock().unwrap().expect("on_complete must fire");
        assert!(end.transport_ok, "disconnect is not a transport fault");
        assert!(end.client_disconnect, "client disconnect must be reported");
    }

    #[test]
    fn inband_scanner_detects_error_when_newline_opens_next_chunk() {
        // Chunk 1 carries the COMPLETE error line with no newline; chunk 2
        // begins with the `\n` at index 0 — the empty-extend path of the
        // carry branch, the boundary an off-by-one in `rest[i + 1..]` breaks.
        let mut s = InbandErrorScanner::default();
        s.feed(b"data: {\"error\": {\"code\": 503}}");
        assert!(!s.found, "no newline yet — line is incomplete");
        s.feed(b"\ndata: [DONE]\n");
        assert!(s.found, "line completed by a chunk-leading newline");
    }

    #[test]
    fn inband_scanner_detects_error_as_later_line_in_one_chunk() {
        // The in-place while-loop must advance PAST a non-error line and
        // match a later one within the same chunk.
        let mut s = InbandErrorScanner::default();
        s.feed(
            b"data: {\"choices\": []}\n\ndata: {\"error\": {\"code\": 500}}\n\ndata: [DONE]\n\n",
        );
        assert!(s.found, "error on a non-first line of a chunk must match");
    }

    #[tokio::test]
    async fn on_complete_reports_inband_error_and_transport_failure_together() {
        // The engine emits a well-formed in-band error, then the connection
        // breaks. BOTH flags must be reported — the chat handler's precedence
        // mapping (inband_error wins) depends on neither masking the other.
        use std::sync::Mutex as StdMutex;
        let seen: Arc<StdMutex<Option<StreamEnd>>> = Arc::new(StdMutex::new(None));
        let seen_c = Arc::clone(&seen);
        let chunks: Vec<Result<Bytes, std::io::Error>> = vec![
            Ok(Bytes::from_static(
                b"data: {\"error\": {\"message\": \"aborted\", \"code\": 503}}\n\n",
            )),
            Err(std::io::Error::other("connection reset")),
        ];
        let body = bytes_stream_to_body(
            stream::iter(chunks),
            None,
            Some(Box::new(move |end| {
                *seen_c.lock().unwrap() = Some(end);
            })),
            None,
            None,
            None,
        );
        let _ = body.collect().await;
        for _ in 0..50 {
            if seen.lock().unwrap().is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let end = seen.lock().unwrap().expect("on_complete must fire");
        assert!(end.saw_inband_error, "in-band error must be reported");
        assert!(!end.transport_ok, "transport failure must also be reported");
    }

    #[tokio::test]
    async fn on_complete_reports_upstream_error() {
        use std::sync::Mutex as StdMutex;
        let seen: Arc<StdMutex<Option<StreamEnd>>> = Arc::new(StdMutex::new(None));
        let seen_c = Arc::clone(&seen);
        let chunks: Vec<Result<Bytes, std::io::Error>> = vec![
            Ok(Bytes::from_static(b"data: {\"choices\": []}\n\n")),
            Err(std::io::Error::other("connection reset")),
        ];
        let body = bytes_stream_to_body(
            stream::iter(chunks),
            None,
            Some(Box::new(move |end| {
                *seen_c.lock().unwrap() = Some(end);
            })),
            None,
            None,
            None,
        );
        let _ = body.collect().await;
        for _ in 0..50 {
            if seen.lock().unwrap().is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let end = seen.lock().unwrap().expect("on_complete must fire");
        assert!(!end.transport_ok, "mid-stream error must fail transport_ok");
        assert!(!end.saw_inband_error);
    }

    #[tokio::test]
    async fn upstream_error_surfaces_to_consumer() {
        let chunks: Vec<Result<Bytes, std::io::Error>> = vec![
            Ok(Bytes::from_static(b"ok-chunk")),
            Err(std::io::Error::other("upstream blew up mid-stream")),
        ];
        let s = stream::iter(chunks);
        let body = bytes_stream_to_body(s, None, None, None, None, None);
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
        let body = bytes_stream_to_body(s, None, None, None, None, None);
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
        let body = bytes_stream_to_body(s, None, None, None, None, None);
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

    /// End-to-end verification that a panic inside the pump stamps
    /// `AbortReason::StreamPumpPanicked` into the shared reason atom BEFORE
    /// the pump task's `_hold` drops the guard. This is the whole point of
    /// the `PanicReasonMarker`: its Drop runs BEFORE `_hold`'s Drop (LIFO
    /// order of local variables), so the guard's own Drop reads the
    /// panic-tagged reason instead of the constructor default
    /// `StreamClientGone`.
    ///
    /// A regression here — someone reordering the marker vs. `_hold`
    /// declarations, or moving the panic-reason stamp to AFTER `catch_unwind`
    /// (which runs after `_hold` has dropped and is too late) — silently
    /// mislabels every panic-induced abort as `stream_client_gone`, sending
    /// operators looking at the wrong dashboards during an incident.
    #[tokio::test]
    async fn pump_stamps_stream_pump_panicked_on_panic() {
        use std::sync::atomic::AtomicU8;

        // Start with the streaming constructor default so we can prove the
        // panic path actively overwrites it (a no-op wouldn't).
        let reason = Arc::new(AtomicU8::new(AbortReason::StreamClientGone as u8));
        let s = PanicOnSecondPoll { polls: 0 };
        let body = bytes_stream_to_body(s, None, None, None, Some(Arc::clone(&reason)), None);
        // Drain the body so the pump task runs to completion (panics, gets
        // caught by `catch_unwind`, and the panic marker's Drop has fired).
        let _ = body.collect().await;
        assert_eq!(
            reason.load(Ordering::Relaxed),
            AbortReason::StreamPumpPanicked as u8,
            "PanicReasonMarker must have stamped StreamPumpPanicked before \
             the pump task's _hold dropped — got {}",
            reason.load(Ordering::Relaxed),
        );
    }

    /// Regression guard for the backpressure-via-disconnect invariant.
    ///
    /// The doc on `bytes_stream_to_body` claims "when the axum Body is dropped
    /// the receiver is closed; the next `tx.send()` returns `Err`, which breaks
    /// the loop — no upstream bytes are read after the client disconnects." With
    /// the byte-bounded read-ahead the pump can also be parked acquiring permits
    /// when the disconnect lands, so the break may instead come from the
    /// `tx.closed()` arm of the select. This test pins both: a refactor that
    /// drops the `tx.send().is_err()` break OR the `tx.closed()` select arm would
    /// silently regress (leaked upstream reads on every client cancel, visible
    /// only as ops-side memory growth).
    #[tokio::test]
    async fn bytes_stream_to_body_breaks_on_client_disconnect() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        // A stream that yields N Ok chunks readily, counting polls via a shared
        // atomic. After we read 1 chunk and drop the body, the pump must hit
        // tx.send-err / tx.closed() and break — not drain all 1000 chunks.
        //
        // Chunk size is sized to the read-ahead budget on purpose: each chunk is
        // 16 KiB, so 64 chunks (1 MiB) exactly fill STREAM_READAHEAD_MAX_BYTES.
        // The pump reads ~64 chunks ahead, then blocks acquiring read-ahead
        // permits for the 65th; the client never reads, so the only way it
        // un-blocks is the `tx.closed()` arm of the select firing on our
        // `drop(data_stream)`. That keeps the poll count tightly bounded
        // (deterministic, not scheduler-dependent) AND exercises the new
        // disconnect-while-blocked-on-permits path. Tiny chunks would instead
        // let the pump read hundreds of thousands of chunks ahead before the
        // budget bites, making the bound meaningless.
        const CHUNK_LEN: usize = 16 * 1024; // 64 chunks fill the 1 MiB budget
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
                std::task::Poll::Ready(Some(Ok(Bytes::from(vec![b'x'; CHUNK_LEN]))))
            }
        }

        let polls = Arc::new(AtomicUsize::new(0));
        let stream = CountingStream {
            polls: polls.clone(),
            yielded: 0,
            max: 1000, // way more than we'll let it consume
        };
        let body = bytes_stream_to_body(stream, None, None, None, None, None);

        // Read exactly one frame, then drop the body to simulate client disconnect.
        let mut data_stream = body.into_data_stream();
        let first = data_stream.next().await;
        assert!(first.is_some(), "expected at least one chunk before drop");
        drop(data_stream);

        // Give the pump generous time to make additional polls if its break is
        // broken. Healthy code: pump reads ~64 chunks ahead (1 MiB), blocks on
        // the permit acquire for the 65th, then `tx.closed()` fires on the drop
        // and it breaks.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let final_polls = polls.load(Ordering::SeqCst);
        assert!(
            final_polls <= 70,
            "pump kept polling upstream after client disconnect: {final_polls} polls (expected <=70, read-ahead budget + slack)"
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
        let body = bytes_stream_to_body(stalled, Some(guard), None, None, None, None);
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

    /// The engine-done release invariant: once the UPSTREAM stream ends (the
    /// engine finished generating), the pump must drop its `stream_guards` (the
    /// per-worker admission slot + active-load entry) at the engine's pace —
    /// NOT wait for a slow client to finish reading the body. A completion whose
    /// total size fits within the byte-bounded read-ahead buffer
    /// (`STREAM_READAHEAD_MAX_BYTES`) must reach upstream-`None` even if the
    /// client never reads a single byte. Without read-ahead the pump parks on a
    /// full bounded channel and the slot is held for the whole client-facing
    /// stream lifetime — starving a fast engine.
    #[tokio::test(start_paused = true)]
    async fn engine_done_releases_guards_before_client_drains() {
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

        // ~200 small chunks, total well under STREAM_READAHEAD_MAX_BYTES, then
        // the stream ends. The read-ahead buffer absorbs all of them so the pump
        // can drain the upstream to completion without the client reading.
        let chunks: Vec<Result<Bytes, std::io::Error>> = (0..200)
            .map(|_| Ok(Bytes::from_static(b"small-chunk")))
            .collect();
        let s = stream::iter(chunks);

        // Build the Body but DO NOT read it: the client never drains a byte.
        let _body = bytes_stream_to_body(s, Some(guard), None, None, None, None);

        // Let the pump run. It should race ahead of the (absent) client, reach
        // upstream-`None`, and drop the guards — all well within STREAM_SEND_STALL.
        for _ in 0..1000 {
            if dropped.load(Ordering::SeqCst) {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(
            dropped.load(Ordering::SeqCst),
            "engine-done must release stream_guards at the engine's pace: a \
             completion fitting in the read-ahead buffer reaches upstream-None \
             and drops the slot even when the client never reads",
        );
    }

    /// A stalled *downstream* — the client reads response headers, then stops
    /// reading the body but never closes the connection — must not pin the SSE
    /// pump forever. With the byte-bounded read-ahead, a completion LARGER than
    /// `STREAM_READAHEAD_MAX_BYTES` fills the buffer and the pump parks acquiring
    /// read-ahead permits; the upstream idle timeout CANNOT fire there (it is
    /// only armed while we poll `s.next()`), and the client never reads so it
    /// never frees permits. Without the send-stall timeout the pump blocks on the
    /// permit acquire indefinitely and `stream_guards` (the per-worker admission
    /// slot + active-load entry) leak — pinning every worker at its cap and
    /// shedding all traffic while the engines sit idle (the exact production
    /// failure observed: engine `num_running_reqs=0` while the router holds
    /// `inflight=cap`).
    #[tokio::test(start_paused = true)]
    async fn stalled_downstream_releases_stream_guards() {
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

        // Upstream yields more bytes than the read-ahead buffer holds without
        // ending: ~20 × 64 KiB = 1.25 MiB > STREAM_READAHEAD_MAX_BYTES (1 MiB).
        // The buffer fills, so the pump parks acquiring permits for a chunk the
        // (non-reading) client never makes room for — the path the send-stall
        // timeout must rescue. (A completion small enough to fit the buffer would
        // instead drain to upstream-`None` and release at engine-done; see
        // `engine_done_releases_guards_before_client_drains`.)
        let chunks: Vec<Result<Bytes, std::io::Error>> = (0..20)
            .map(|_| Ok(Bytes::from(vec![b'x'; 64 * 1024])))
            .collect();
        let s = stream::iter(chunks);

        // Hold the Body WITHOUT reading it: the receiver stays open (no clean
        // disconnect) but undrained, so the pump parks acquiring read-ahead
        // permits once the buffer is full.
        let _body = bytes_stream_to_body(s, Some(guard), None, None, None, None);

        // Advance past the send-stall timeout. The pump must give up on the
        // non-draining client and drop its guards.
        tokio::time::sleep(STREAM_SEND_STALL + std::time::Duration::from_secs(1)).await;
        assert!(
            dropped.load(Ordering::SeqCst),
            "a client that stops reading (without disconnecting) must not pin \
             stream_guards — the pump must time out the blocked permit acquire \
             and release the admission slot",
        );
    }

    /// End-to-end verification that a downstream stall (client stops draining
    /// for `STREAM_SEND_STALL` while the connection stays open) stamps
    /// `AbortReason::StreamDownstreamStall` into the shared reason atom before
    /// the pump exits. Without this stamp, the guard's Drop would fire with
    /// the constructor default `StreamClientGone`, which is a lie — the
    /// client didn't disconnect, it just stopped consuming — and would send
    /// operators looking at "client cancel" volume when the real signal is
    /// "consumer backpressure / slow client."
    ///
    /// Mirror of `stalled_downstream_releases_stream_guards` above, adding
    /// the reason-handle assertion the log/metric side depends on.
    #[tokio::test(start_paused = true)]
    async fn pump_stamps_stream_downstream_stall_on_send_stall() {
        use std::sync::atomic::AtomicU8;

        let reason = Arc::new(AtomicU8::new(AbortReason::StreamClientGone as u8));
        // Same shape as the sibling stall test: > read-ahead budget worth
        // of bytes to a client that never drains, so the pump parks
        // acquiring permits and the STREAM_SEND_STALL timeout trips.
        let chunks: Vec<Result<Bytes, std::io::Error>> = (0..20)
            .map(|_| Ok(Bytes::from(vec![b'x'; 64 * 1024])))
            .collect();
        let s = stream::iter(chunks);
        let _body = bytes_stream_to_body(s, None, None, None, Some(Arc::clone(&reason)), None);
        // Advance past send-stall so the pump gives up on the non-draining
        // client. `_body` is held to keep the receiver alive during the wait
        // — a Drop here would masquerade as a client_gone.
        tokio::time::sleep(STREAM_SEND_STALL + std::time::Duration::from_secs(1)).await;
        // Give the pump task a scheduling tick to finish its exit path
        // after the timeout fires.
        tokio::task::yield_now().await;
        assert_eq!(
            reason.load(Ordering::Relaxed),
            AbortReason::StreamDownstreamStall as u8,
            "downstream_stall path must stamp StreamDownstreamStall, not the \
             constructor default (a stalled client is not a gone client) — got {}",
            reason.load(Ordering::Relaxed),
        );
    }

    /// The send-stall abort must record a breaker SUCCESS, not a failure: a
    /// client that stops draining is not the worker's fault, and penalizing the
    /// worker for it would trip the breaker on a healthy engine. Pins
    /// `transport_ok = true` on the `STREAM_SEND_STALL` path (contrast the
    /// failure path on a genuine mid-stream upstream error, which sets it
    /// false), and `client_disconnect = true` so the stream-outcome metric
    /// classifies the truncation as client-side rather than `ok`.
    #[tokio::test(start_paused = true)]
    async fn stalled_downstream_records_breaker_success() {
        use std::sync::atomic::{AtomicU8, Ordering};
        use std::sync::Arc;

        // 0 = hook not yet called, 1 = transport_ok + client_disconnect,
        // 2 = any other verdict.
        let outcome = Arc::new(AtomicU8::new(0));
        let outcome_c = Arc::clone(&outcome);

        // > 1 MiB so the buffer fills and the pump parks on the permit acquire
        // (the send-stall path), with a client that never reads.
        let chunks: Vec<Result<Bytes, std::io::Error>> = (0..20)
            .map(|_| Ok(Bytes::from(vec![b'x'; 64 * 1024])))
            .collect();
        let s = stream::iter(chunks);
        let _body = bytes_stream_to_body(
            s,
            None,
            Some(Box::new(move |end: StreamEnd| {
                let verdict = if end.transport_ok && end.client_disconnect {
                    1
                } else {
                    2
                };
                outcome_c.store(verdict, Ordering::SeqCst);
            })),
            None,
            None,
            None,
        );

        // Advance past the send-stall timeout, then let the on_complete hook run.
        tokio::time::sleep(STREAM_SEND_STALL + std::time::Duration::from_secs(1)).await;
        for _ in 0..1000 {
            if outcome.load(Ordering::SeqCst) != 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(
            outcome.load(Ordering::SeqCst),
            1,
            "a client-stall abort must record breaker SUCCESS (the worker was \
             delivering; the client stalled) and classify as client_disconnect, \
             not penalize a healthy worker or report a clean `ok` stream",
        );
    }

    /// Engine-done release must also hold for a single chunk LARGER than the
    /// read-ahead budget. The pump charges `min(len, MAX)` permits (never more
    /// than the semaphore holds), so an oversized chunk proceeds instead of
    /// deadlocking on `acquire_many(len > MAX)`. This pins the acquire-side
    /// `.min(MAX)`: dropping it would ask for more permits than exist, park the
    /// pump forever, and leak the per-worker admission slot — the exact
    /// production false-shedding bug.
    #[tokio::test(start_paused = true)]
    async fn oversized_single_chunk_releases_guards() {
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

        // One chunk twice the read-ahead budget, then the stream ends. With a
        // non-reading client the pump charges only MAX permits (not 2×MAX), sends
        // the chunk onto the unbounded channel, reads upstream-None, and drops the
        // guards — even though the client never read a byte.
        let big = Bytes::from(vec![b'x'; 2 * STREAM_READAHEAD_MAX_BYTES]);
        let s = stream::iter(vec![Ok::<Bytes, std::io::Error>(big)]);
        let _body = bytes_stream_to_body(s, Some(guard), None, None, None, None);

        for _ in 0..1000 {
            if dropped.load(Ordering::SeqCst) {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(
            dropped.load(Ordering::SeqCst),
            "an oversized single chunk must still reach upstream-None and release \
             the slot — the acquire charges min(len, MAX), never more than the \
             semaphore holds, so it must not deadlock",
        );
    }

    /// An oversized chunk (> read-ahead budget) must stream through intact when
    /// the client reads — the `min(len, MAX)` charge/return on both sides keeps
    /// the semaphore balanced rather than corrupting or truncating the payload.
    #[tokio::test]
    async fn oversized_single_chunk_streams_through_intact() {
        let big = vec![b'x'; 2 * STREAM_READAHEAD_MAX_BYTES];
        let s = stream::iter(vec![Ok::<Bytes, std::io::Error>(Bytes::from(big.clone()))]);
        let body = bytes_stream_to_body(s, None, None, None, None, None);
        let bytes = body.collect().await.unwrap().to_bytes();
        assert_eq!(
            bytes.len(),
            big.len(),
            "oversized chunk must stream through without truncation",
        );
        assert_eq!(
            &bytes[..],
            &big[..],
            "oversized chunk payload must be intact"
        );
    }

    /// `mark_terminal` flips the flag the moment the stream is drained to its
    /// clean end — the engine-finished signal the abort guard reads to decide
    /// it must NOT abort.
    #[tokio::test]
    async fn mark_terminal_sets_flag_on_clean_end() {
        use std::sync::atomic::AtomicBool;
        let flag = Arc::new(AtomicBool::new(false));
        let s = stream::iter(vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(b"a")),
            Ok(Bytes::from_static(b"b")),
        ]);
        let mut wrapped = mark_terminal(s, Arc::clone(&flag));
        while wrapped.next().await.is_some() {}
        assert!(
            flag.load(Ordering::SeqCst),
            "reaching the clean upstream end must set reached_end (engine done → no abort)",
        );
    }

    /// The disconnect signal: a consumer that stops polling before the terminal
    /// item (exactly what the SSE pump does when the client disconnects) leaves
    /// the flag false — which is what arms the abort.
    #[tokio::test]
    async fn mark_terminal_flag_stays_false_when_consumer_stops_early() {
        use std::sync::atomic::AtomicBool;
        let flag = Arc::new(AtomicBool::new(false));
        let s = stream::iter(vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(b"a")),
            Ok(Bytes::from_static(b"b")),
            Ok(Bytes::from_static(b"c")),
        ]);
        let mut wrapped = mark_terminal(s, Arc::clone(&flag));
        // Read one item, then stop — the upstream still has chunks left and was
        // never driven to its terminal `None`.
        let _ = wrapped.next().await;
        drop(wrapped);
        assert!(
            !flag.load(Ordering::SeqCst),
            "stopping before the terminal item must leave reached_end false (client gone → abort fires)",
        );
    }

    /// A terminal upstream error means the engine is done/failed — the flag must
    /// be set as the error is yielded, so a real upstream failure is never
    /// mistaken for a client disconnect (which would send a pointless abort).
    #[tokio::test]
    async fn mark_terminal_sets_flag_on_terminal_error() {
        use std::sync::atomic::AtomicBool;
        let flag = Arc::new(AtomicBool::new(false));
        let s = stream::iter(vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(b"a")),
            Err(std::io::Error::other("boom")),
        ]);
        let mut wrapped = mark_terminal(s, Arc::clone(&flag));
        let _ok = wrapped.next().await; // "a"
        let _err = wrapped.next().await; // Err → flag set as it is yielded
        assert!(
            flag.load(Ordering::SeqCst),
            "a terminal error means the engine is done/failed — no abort",
        );
    }
}
