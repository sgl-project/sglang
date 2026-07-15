// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! HTTP proxy — forwards requests to the upstream SGLang worker.

pub mod sse;

use crate::health::circuit_breaker::CircuitBreaker;
use crate::server::error::ApiError;
use crate::server::header_utils::should_forward_request_header;
use crate::server::metrics::MetricsRegistry;
use crate::workers::WireProtocol;
use anyhow::Context;
use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response};
use bytes::Bytes;
use reqwest::{Client, Url};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// Total number of [`AbortReason`] variants. Referenced by
/// [`crate::server::metrics::MetricsRegistry`] to size the fixed-length
/// per-reason counter array — a compile-time invariant that adding a new
/// variant here without bumping this constant would break, which is exactly
/// the failure mode we want (versus a silent OOB or a HashMap re-alloc under
/// contention). Update alongside every new variant.
pub(crate) const ABORT_REASON_COUNT: usize = 8;

/// Why the router is telling an engine to stop generating a specific request.
///
/// Recorded via [`AbortOnDrop`] at the drop site and stamped into (a) the WARN
/// log line the drop emits and (b) the JSON body of the `/abort_request` POST
/// (as `router_reason`) so operators can attribute engine-side aborts to a
/// specific router-side trigger without having to correlate by `rid` alone.
/// Values are stored in an `AtomicU8` so the reason can be updated cross-thread
/// (the SSE pump is on a different tokio task than the handler owning the
/// guard) without borrowing gymnastics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum AbortReason {
    /// The drop happened without any code path narrowing the cause. Only
    /// observed when a call site failed to `set_reason` — treat as a bug in
    /// call-site coverage, not a legitimate reason.
    Unknown = 0,
    /// Unary/pre-headers guard: the handler future was dropped mid-await
    /// before any specific `ApiError` was assigned. In practice this is a
    /// client-side disconnect: axum cancelled the handler when the client's
    /// connection closed while the router was still awaiting the engine's
    /// buffered response.
    HandlerCancelled = 1,
    /// Unary/pre-headers guard: the stale-request janitor
    /// (`ActiveLoadGuard::cancel_token`) fired before the engine responded.
    /// The handler's `tokio::select!` returned `StaleRequestExpired` and the
    /// guard was left armed on purpose.
    StaleRequestExpired = 2,
    /// Unary/pre-headers guard: the router-side wait-for-response-headers
    /// budget (`request_timeout`) elapsed before the engine sent headers.
    /// Distinct from stale-timeout in that the router itself gave up on this
    /// worker, not the request as a whole.
    UpstreamTimeout = 3,
    /// Unary/pre-headers guard: connect / TCP / TLS / write error before a
    /// response was received. Covers reqwest's transport-layer failures,
    /// breaker-open rejections, and DNS/URL problems.
    TransportError = 4,
    /// Streaming guard: the SSE pump's `tx.send` failed or `tx.closed` fired
    /// while the pump had more upstream to deliver — i.e. axum/hyper dropped
    /// the response Body (client TCP close, HTTP/2 RST_STREAM, or a middle
    /// box severing the connection). The most common streaming abort cause.
    StreamClientGone = 5,
    /// Streaming guard: the pump waited the configured send-stall budget
    /// (`STREAM_SEND_STALL` by default) for the
    /// client to drain the read-ahead buffer and the client never made
    /// progress. Treated as "client is present but not consuming"; the abort
    /// releases the engine slot so a working consumer can be served.
    StreamDownstreamStall = 6,
    /// Streaming guard: the pump task itself panicked. Distinct from a
    /// clean client-gone because the panic implies a router-side bug, not a
    /// client problem — surface it so it doesn't get lumped with the
    /// normal disconnect volume.
    StreamPumpPanicked = 7,
}

impl AbortReason {
    /// Stable label for logs, metrics, and the outbound POST body. Keep in
    /// sync with the `AbortReason` variants: adding a variant without adding
    /// a label here surfaces a compile error, which is why this is a `match`
    /// and not e.g. a `Debug`-derived string.
    pub(crate) fn as_label(&self) -> &'static str {
        match self {
            AbortReason::Unknown => "unknown",
            AbortReason::HandlerCancelled => "handler_cancelled",
            AbortReason::StaleRequestExpired => "stale_request_expired",
            AbortReason::UpstreamTimeout => "upstream_timeout",
            AbortReason::TransportError => "transport_error",
            AbortReason::StreamClientGone => "stream_client_gone",
            AbortReason::StreamDownstreamStall => "stream_downstream_stall",
            AbortReason::StreamPumpPanicked => "stream_pump_panicked",
        }
    }

    /// Inverse of the `#[repr(u8)]` cast; used by [`crate::server::metrics`]
    /// to translate its per-reason array index back to a label at scrape
    /// time. `pub(crate)` (not `pub`) because the discriminant→variant map
    /// is a crate-private detail — external code goes through `as_label`.
    ///
    /// `0` is the legitimate `Unknown` discriminant; other out-of-range
    /// values are corruption (the atomic is only written from
    /// `AbortReason as u8`, so they should be unreachable). Split the two
    /// with a `debug_assert!` so a debug build panics loudly on
    /// discriminant drift, while release keeps the safe fallback.
    pub(crate) fn from_u8(v: u8) -> Self {
        match v {
            0 => AbortReason::Unknown,
            1 => AbortReason::HandlerCancelled,
            2 => AbortReason::StaleRequestExpired,
            3 => AbortReason::UpstreamTimeout,
            4 => AbortReason::TransportError,
            5 => AbortReason::StreamClientGone,
            6 => AbortReason::StreamDownstreamStall,
            7 => AbortReason::StreamPumpPanicked,
            other => {
                debug_assert!(
                    false,
                    "AbortReason::from_u8 got out-of-range value {other}; \
                     did you add a variant without bumping ABORT_REASON_COUNT?",
                );
                AbortReason::Unknown
            }
        }
    }

    /// Contiguous index in `0..ABORT_REASON_COUNT`. Used by the metrics
    /// registry to index its fixed-length per-reason counter array without a
    /// hash lookup or lock. Guaranteed in-range by the `#[repr(u8)]` +
    /// explicit-discriminant layout (0..=7 for the current 8 variants) and
    /// the `ABORT_REASON_COUNT` constant kept in sync above.
    pub(crate) fn as_index(&self) -> usize {
        *self as usize
    }
}

// Compile-time invariant: every `AbortReason` discriminant is < ABORT_REASON_COUNT.
// If you add a new variant without also bumping the constant above, this fails
// to build — which is the whole point. Runtime array-index panics on the hot
// path (metrics.rs `engine_aborts_total[reason.as_index()]`) would be
// significantly worse than a compile error here. Also asserts the "top
// discriminant is exactly COUNT-1" so a gap (someone assigning a scratch
// discriminant like `= 100`) also fails to build.
const _: () = {
    assert!((AbortReason::Unknown as u8) < ABORT_REASON_COUNT as u8);
    assert!((AbortReason::HandlerCancelled as u8) < ABORT_REASON_COUNT as u8);
    assert!((AbortReason::StaleRequestExpired as u8) < ABORT_REASON_COUNT as u8);
    assert!((AbortReason::UpstreamTimeout as u8) < ABORT_REASON_COUNT as u8);
    assert!((AbortReason::TransportError as u8) < ABORT_REASON_COUNT as u8);
    assert!((AbortReason::StreamClientGone as u8) < ABORT_REASON_COUNT as u8);
    assert!((AbortReason::StreamDownstreamStall as u8) < ABORT_REASON_COUNT as u8);
    assert!((AbortReason::StreamPumpPanicked as u8) < ABORT_REASON_COUNT as u8);
    assert!((AbortReason::StreamPumpPanicked as u8) == (ABORT_REASON_COUNT as u8) - 1);
};

/// Parse a worker URL emitted by discovery.  On failure, trip the worker's
/// circuit breaker so the malformed worker drops out of subsequent
/// `healthy_workers_for(...)` selection, then surface the error as
/// `ApiError::WorkerMisconfigured`.
fn parse_worker_url(worker_url: &str, breaker: &CircuitBreaker) -> Result<Url, ApiError> {
    Url::parse(worker_url).map_err(|e| {
        breaker.record_failure();
        ApiError::WorkerMisconfigured {
            worker: worker_url.to_string(),
            source: anyhow::Error::new(e).context("parse worker URL"),
        }
    })
}

/// How an upstream HTTP response status should affect the worker's circuit
/// breaker. Each variant maps to one `CircuitBreaker` call at the dispatch
/// sites (`forward_json_to` / `forward_streaming_to`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BreakerOutcome {
    /// Healthy completion → `record_success`: reset the failure streak and
    /// close the breaker.
    Success,
    /// A real fault (5xx other than backpressure) → `record_failure`: count
    /// toward opening.
    Failure,
    /// Backpressure (the worker is responsive but at capacity) →
    /// `record_backpressure`: never opens the breaker and, while Closed, leaves
    /// an in-progress failure streak intact — but still resolves a half-open
    /// probe so a recovered-but-busy worker isn't wedged shut.
    Neutral,
}

/// Classify an upstream status for circuit-breaker accounting.
///
/// A backpressure status — `503 Service Unavailable` or `429 Too Many
/// Requests` — is the worker signalling "responsive but at capacity", not a
/// fault. Counting it as a breaker failure is actively harmful: a saturated
/// worker trips the breaker on its own queue-full 503s, and with a single
/// worker the router then sheds *every* request for the whole cool-down —
/// including after the engine has drained and gone idle. So backpressure is
/// [`Neutral`](BreakerOutcome::Neutral) (see [`CircuitBreaker::record_backpressure`]
/// for its exact effect per breaker state). Genuine 5xx faults (500 / 502 /
/// 504 / …) still count as failures, and transport errors / timeouts /
/// mid-body drops are recorded as failures at the call sites.
///
/// Tradeoff: because 503 never opens the breaker, a worker stuck returning 503
/// indefinitely (a wedged engine, not transient load) is NOT detected here —
/// HTTP status alone can't distinguish "busy" from "broken-and-saying-503", and
/// counting it caused the worse fleet-wide false-shed above. Detecting a
/// chronically-backpressuring worker is left to higher-level signals.
fn breaker_outcome(status: reqwest::StatusCode) -> BreakerOutcome {
    use reqwest::StatusCode;
    match status {
        StatusCode::SERVICE_UNAVAILABLE | StatusCode::TOO_MANY_REQUESTS => BreakerOutcome::Neutral,
        s if s.is_server_error() => BreakerOutcome::Failure,
        _ => BreakerOutcome::Success,
    }
}

/// Idle (between-bytes) timeout for streaming upstream responses. A stream that
/// delivers no bytes for this long is treated as hung and aborted, releasing the
/// admission / active-load guards it holds. Distinct from `request_timeout` (the
/// total budget, which streaming deliberately skips so long generations can run):
/// this fires only on a *stall*, not on slow-but-progressing generation. Without
/// it, a half-open upstream (e.g. a worker killed mid-stream) pins the SSE pump
/// and leaks the per-worker in-flight slot forever.
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// How long an abort POST may take before we give up. The client is already
/// gone, so this only bounds how long the fire-and-forget task lingers; a slow
/// abort must never wedge a worker's connection pool.
const ABORT_TIMEOUT: Duration = Duration::from_secs(5);

/// Tell an engine to stop generating a request whose client has disconnected,
/// by `POST`ing `/abort_request {rid, abort_all:false, router_reason}`. The
/// engine cancels every in-flight request whose `rid` *starts with* this one,
/// which also covers the `n>1` parallel-sampling expansions
/// (`<rid>_0`, `<rid>_1`, …).
///
/// `reason` is a stable label from [`AbortReason::as_label`] identifying why
/// the router is aborting. It rides on the request body as `router_reason` so
/// the engine can log/count aborts by cause (SGLang currently ignores extra
/// fields, keeping this forward-compatible), and is also stamped into the
/// WARN log emitted by the [`AbortOnDrop`] drop site — that log is the
/// primary observability surface: one `sending /abort_request` line per real
/// abort, tagged with reason, so operators can quantify "how many of my
/// aborts were `stream_client_gone` vs `stale_request_expired`" without
/// having to parse pump-timing debug logs.
///
/// Best-effort by construction: the client is already gone, so there is no one
/// to surface an error to, and a missed abort wastes engine compute but is not
/// a correctness fault. Failures are logged, not propagated. Not circuit-breaker
/// gated — an abort is a courtesy to the engine, never counted against a worker.
async fn send_abort(client: &Client, abort_url: &str, rid: &str, reason: AbortReason) {
    let reason_label = reason.as_label();
    let body = serde_json::json!({
        "rid": rid,
        "abort_all": false,
        "router_reason": reason_label,
    });
    match client
        .post(abort_url)
        .json(&body)
        .timeout(ABORT_TIMEOUT)
        .send()
        .await
    {
        Ok(resp) => tracing::debug!(
            abort_url,
            rid,
            reason = reason_label,
            status = %resp.status(),
            "engine acknowledged /abort_request",
        ),
        Err(e) => tracing::warn!(
            abort_url,
            rid,
            reason = reason_label,
            error = %e,
            "failed to POST /abort_request to engine (best-effort; not retried)",
        ),
    }
}

/// Drop guard that aborts an in-flight engine request when the client goes
/// away. Spawns [`send_abort`] from its `Drop` when (and only when) it is still
/// "armed" at drop time.
///
/// Two arming modes, matching the two forwarding paths:
/// * **Unary** ([`for_unary`](Self::for_unary)): armed until [`disarm`](Self::disarm)
///   is called. The handler disarms it once a complete response is in hand, so a
///   drop without disarm means the handler future was cancelled (client
///   disconnect) or the stale-request janitor fired — both cases where the
///   engine may still be working and should be told to stop.
/// * **Streaming** ([`for_stream`](Self::for_stream)): never disarmed; instead it
///   reads a `reached_end` flag flipped true by [`sse::mark_terminal`] the moment
///   the upstream stream yields its terminal item. A drop with the flag still
///   false means the SSE pump was torn down before the engine finished — i.e. the
///   client disconnected or stalled mid-stream.
pub(crate) struct AbortOnDrop {
    client: Client,
    abort_url: String,
    rid: String,
    /// `None` for unary (decided solely by `armed`); `Some` for streaming, where
    /// the abort fires only if the engine had not reached its terminal item.
    reached_end: Option<Arc<AtomicBool>>,
    /// Encoded [`AbortReason`]. Stored via `Arc<AtomicU8>` so an update from a
    /// different task than the one that owns the guard (specifically, the SSE
    /// pump task writing into a streaming guard held inside `stream_guards`)
    /// is race-free and Send/Sync-friendly.
    ///
    /// Defaults are set in the constructors: unary → [`AbortReason::HandlerCancelled`]
    /// (the "you dropped me without telling me why" catch-all that also covers
    /// the common client-disconnect-during-await case); streaming →
    /// [`AbortReason::StreamClientGone`] (the majority streaming case). Call
    /// sites narrow those defaults via [`Self::set_reason`] or by writing to
    /// the [`Self::reason_handle`] before the guard drops.
    reason: Arc<AtomicU8>,
    /// Metrics sink used in `Drop` to bump `sgl_router_engine_aborts_total`
    /// with the per-reason label. `None` when the guard was constructed
    /// directly (unit tests) or the proxy never had metrics attached;
    /// the WARN log still fires, only the counter goes dark.
    metrics: Option<Arc<MetricsRegistry>>,
    armed: bool,
}

impl AbortOnDrop {
    fn for_unary(
        client: Client,
        abort_url: String,
        rid: String,
        metrics: Option<Arc<MetricsRegistry>>,
    ) -> Self {
        Self {
            client,
            abort_url,
            rid,
            reached_end: None,
            reason: Arc::new(AtomicU8::new(AbortReason::HandlerCancelled as u8)),
            metrics,
            armed: true,
        }
    }

    fn for_stream(
        client: Client,
        abort_url: String,
        rid: String,
        reached_end: Arc<AtomicBool>,
        metrics: Option<Arc<MetricsRegistry>>,
    ) -> Self {
        Self {
            client,
            abort_url,
            rid,
            reached_end: Some(reached_end),
            reason: Arc::new(AtomicU8::new(AbortReason::StreamClientGone as u8)),
            metrics,
            armed: true,
        }
    }

    /// Mark the request as completed so the guard does NOT abort on drop. Call
    /// once a full response has been received from the engine (unary path).
    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }

    /// Narrow the recorded [`AbortReason`] from the constructor default. Idempotent
    /// per drop — later writes overwrite earlier ones, which is intentional: the
    /// most-specific known reason at drop time wins.
    pub(crate) fn set_reason(&self, reason: AbortReason) {
        self.reason.store(reason as u8, Ordering::Relaxed);
    }

    /// Handle that lets a task NOT holding `&self` write the reason before drop.
    /// Used by the SSE pump: the pump task holds the guard inside its opaque
    /// `stream_guards`, but takes a separate `Arc<AtomicU8>` clone so it can
    /// tag the reason as it hits `break` on `client_gone` / `downstream_stall` /
    /// panic paths. Cheap: one `Arc::clone`, no allocation.
    pub(crate) fn reason_handle(&self) -> Arc<AtomicU8> {
        Arc::clone(&self.reason)
    }

    fn should_abort(&self) -> bool {
        self.armed
            && self
                .reached_end
                .as_ref()
                .is_none_or(|f| !f.load(Ordering::SeqCst))
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if !self.should_abort() {
            return;
        }
        let reason = AbortReason::from_u8(self.reason.load(Ordering::Relaxed));
        // One WARN line per real abort. Deliberately WARN, not DEBUG: an
        // abort means the engine is doing wasted work on our behalf, and
        // operators need to be able to grep this at scale. Paired with the
        // Prom counter bump below, this gives both a per-event trail (log)
        // and an aggregate signal (metric) with matching labels — cheap to
        // slice by reason without full log parsing.
        tracing::warn!(
            rid = %self.rid,
            reason = reason.as_label(),
            abort_url = %self.abort_url,
            "sending /abort_request to engine",
        );
        if let Some(m) = self.metrics.as_ref() {
            m.record_engine_abort(reason);
        }
        let client = self.client.clone();
        let abort_url = std::mem::take(&mut self.abort_url);
        let rid = std::mem::take(&mut self.rid);
        // `Drop` is sync; the POST is async and fire-and-forget. We need a
        // runtime handle to spawn it — present on every normal drop (the
        // handler / SSE pump run on the tokio runtime). It is absent only when
        // the runtime itself is tearing down, in which case the process is
        // exiting and there is no point chasing an abort.
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(async move { send_abort(&client, &abort_url, &rid, reason).await });
            }
            Err(_) => tracing::debug!(
                %abort_url,
                %rid,
                reason = reason.as_label(),
                "no tokio runtime at drop (shutdown); skipping engine abort",
            ),
        }
    }
}

#[derive(Debug)]
pub struct Proxy {
    /// HTTP/1.1 forwarding client. The always-safe default, and also the
    /// client used for side-channel admin traffic (e.g. `/flush_cache`):
    /// Granian's `auto` mode accepts HTTP/1.1 from every engine, so admin
    /// calls never need the h2c client.
    http1_client: Client,
    /// Cleartext-h2c forwarding client (HTTP/2 prior knowledge). Used per
    /// request for workers whose `/server_info` reported `--enable-http2` on a
    /// cleartext URL. Built up front alongside `http1_client` so a request
    /// only selects between two ready clients (no per-request build, no
    /// fleet-wide first-write-wins lock-in).
    h2c_client: Client,
    /// Wall-clock timeout applied to non-streaming upstream requests. Streaming
    /// requests deliberately do not use this (long generations are valid).
    pub request_timeout: Duration,
    /// Between-bytes idle timeout for the upstream→router streaming leg — how
    /// long the engine may go silent mid-stream before the pump aborts. A stall
    /// cap, not a total-time cap. Defaults to `STREAM_IDLE_TIMEOUT`; prod
    /// overrides it from config via [`Self::with_stream_timeouts`].
    pub stream_idle_timeout: Duration,
    /// Backpressure stall timeout for the router→client streaming leg — how long
    /// a connected-but-non-draining client may block the pump before it releases
    /// the per-worker slot. Defaults to `STREAM_SEND_STALL`; prod overrides it
    /// from config via [`Self::with_stream_timeouts`].
    pub stream_send_stall: Duration,
    /// Metrics sink for the drop-side of `AbortOnDrop`. Filled once at startup
    /// via [`Self::attach_metrics`] — matching the same pattern
    /// `ActiveLoadRegistry` and `PolicyRegistry` use — so tests that build a
    /// `Proxy` in isolation don't need a full metrics wiring, and prod flows
    /// get their per-reason `sgl_router_engine_aborts_total` counter bumped
    /// on every abort. `OnceLock`, not `Mutex<Option<_>>`, because the wire-in
    /// is a single-shot at app startup and the read path is on every abort
    /// drop — one atomic load beats a mutex acquire.
    metrics: OnceLock<Arc<MetricsRegistry>>,
}

/// Build a forwarding client for `protocol`, sharing pool/connect tuning
/// across protocols. The h2c variant pins HTTP/2 prior knowledge so it
/// speaks cleartext h2c to Granian engines (no ALPN on plaintext); the
/// HTTP/1.1 variant is the reqwest default and is safe against any engine.
fn build_client(protocol: WireProtocol) -> Result<Client, anyhow::Error> {
    let builder = Client::builder()
        .pool_max_idle_per_host(64)
        .connect_timeout(Duration::from_secs(5));
    match protocol {
        WireProtocol::Http1 => builder,
        WireProtocol::H2c => builder.http2_prior_knowledge(),
    }
    .build()
    .context("build reqwest client")
}

impl Proxy {
    /// Build a proxy. `request_timeout` is the per-request wall-clock budget for
    /// non-streaming forwards. Connect timeout is hard-coded to 5 s — even a
    /// streaming request fails fast at TCP setup if the worker is unreachable.
    ///
    /// Both forwarding clients (HTTP/1.1 and h2c) are built up front so the
    /// request hot path only *selects* between them by the worker's resolved
    /// [`WireProtocol`] — there is no per-request build and no fleet-wide
    /// lock-in.
    pub fn new(request_timeout: Duration) -> Result<Self, anyhow::Error> {
        Ok(Self {
            http1_client: build_client(WireProtocol::Http1)?,
            h2c_client: build_client(WireProtocol::H2c)?,
            request_timeout,
            // Defaults; prod overrides from config via `with_stream_timeouts`.
            stream_idle_timeout: STREAM_IDLE_TIMEOUT,
            stream_send_stall: sse::STREAM_SEND_STALL,
            metrics: OnceLock::new(),
        })
    }

    /// Override the streaming stall budgets (upstream idle / client backpressure)
    /// from configuration. Chained after [`Self::new`] at startup; tests that
    /// build a bare `Proxy` keep the `STREAM_IDLE_TIMEOUT` / `STREAM_SEND_STALL`
    /// defaults.
    pub fn with_stream_timeouts(
        mut self,
        stream_idle_timeout: Duration,
        stream_send_stall: Duration,
    ) -> Self {
        self.stream_idle_timeout = stream_idle_timeout;
        self.stream_send_stall = stream_send_stall;
        self
    }

    /// Wire a metrics registry into this proxy — every subsequent
    /// [`AbortOnDrop`] created via [`Self::abort_guard_for`] /
    /// [`Self::forward_streaming_to`] inherits it and bumps the per-reason
    /// `sgl_router_engine_aborts_total` counter on drop.
    ///
    /// Single-shot (**first attach wins**): backing storage is
    /// `OnceLock<Arc<MetricsRegistry>>`, deliberately not `Mutex<Option<_>>`
    /// like `ActiveLoadRegistry::attach_metrics` (which supports replace
    /// semantics). A `Proxy` outlives one metrics registry in practice —
    /// there is no hot-swap use case in prod — and the read path
    /// (`metrics_for_abort` called on every abort drop) benefits from being
    /// lock-free.
    ///
    /// The distinction is observable: if a caller wires the proxy twice
    /// with different registries, the SECOND registry is silently dropped
    /// and the FIRST keeps taking bumps. That's a wiring bug the operator
    /// needs to see — so the second call logs at WARN. The first call
    /// logs at INFO so a missing wire-in (proxy built, never attached) is
    /// grep-able at startup instead of showing up later as a flat metric.
    ///
    /// Not required for correctness — a proxy without a metrics sink still
    /// aborts and still WARN-logs the reason; only the aggregate counter goes
    /// dark. Tests that build a proxy in isolation deliberately skip this so
    /// they can assert on the log line alone.
    pub fn attach_metrics(&self, metrics: Arc<MetricsRegistry>) {
        if self.metrics.set(metrics).is_ok() {
            tracing::info!(
                "Proxy metrics attached; sgl_router_engine_aborts_total is now populated"
            );
        } else {
            tracing::warn!(
                "Proxy::attach_metrics called more than once; second registry ignored \
                 (first-attach-wins). sgl_router_engine_aborts_total continues to bump \
                 the first-attached registry, not this one — check AppContext wiring."
            );
        }
    }

    /// Clone the attached metrics registry, if any. Cheap: one `OnceLock::get`
    /// + one `Arc::clone` on the hot path.
    fn metrics_for_abort(&self) -> Option<Arc<MetricsRegistry>> {
        self.metrics.get().map(Arc::clone)
    }

    /// The forwarding client for `protocol`. Selected per request from the
    /// chosen worker's [`crate::workers::Worker::protocol`], so an h2c-capable
    /// worker forwards over h2c even if a sibling worker is still on HTTP/1.1.
    fn client_for(&self, protocol: WireProtocol) -> &Client {
        match protocol {
            WireProtocol::Http1 => &self.http1_client,
            WireProtocol::H2c => &self.h2c_client,
        }
    }

    /// The HTTP/1.1 forwarding client, for side-channel admin traffic
    /// (e.g. `/flush_cache`). Every engine accepts HTTP/1.1 — including
    /// h2c-capable Granian engines under `auto` mode — so admin fan-out needs
    /// no per-worker protocol selection.
    pub fn admin_client(&self) -> &Client {
        &self.http1_client
    }

    /// Build an [`AbortOnDrop`] guard for a **non-streaming** forward to
    /// `worker_url`. Hold it across the forward; disarm it once a complete
    /// response is in hand. If it instead drops while armed — the handler
    /// future was cancelled by a client disconnect, or the stale-request
    /// janitor fired — it `POST`s `/abort_request` so the engine stops
    /// generating a reply no one will read.
    ///
    /// `rid` must be the request id the router injected into the forwarded body
    /// (so the engine's request carries it). Returns `None` only if `worker_url`
    /// can't be parsed — in which case the forward itself fails the same way and
    /// the absent guard is moot. Streaming forwards get their guard internally
    /// via [`forward_streaming_to`](Self::forward_streaming_to)'s `abort_rid`.
    pub(crate) fn abort_guard_for(
        &self,
        worker_url: &str,
        protocol: WireProtocol,
        rid: &str,
    ) -> Option<AbortOnDrop> {
        let abort_url = Url::parse(worker_url).ok()?.join("/abort_request").ok()?;
        Some(AbortOnDrop::for_unary(
            self.client_for(protocol).clone(),
            abort_url.to_string(),
            rid.to_string(),
            self.metrics_for_abort(),
        ))
    }

    /// Classify a reqwest error into the right `ApiError` variant, given an
    /// explicit worker URL. Called from the breaker-gated `forward_*_to`
    /// methods, which carry per-request worker URLs (not a single proxy-level
    /// URL).
    ///
    /// Walks the full source chain to detect timeouts, because reqwest wraps
    /// hyper which wraps `std::io::Error` — a top-level `is_timeout()` check
    /// misses both the wrapped reqwest timeout and the `io::ErrorKind::TimedOut`
    /// cases.
    fn classify_reqwest_error_for(worker: Url, e: reqwest::Error, path: &str) -> ApiError {
        let source = anyhow::Error::new(e).context(format!("worker {worker}: post {path}"));
        let is_timeout = source.chain().any(|c| {
            c.downcast_ref::<reqwest::Error>()
                .is_some_and(|r| r.is_timeout())
        }) || source.chain().any(|c| {
            c.downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::TimedOut)
        });
        if is_timeout {
            ApiError::UpstreamTimeout { worker }
        } else {
            ApiError::UpstreamUnreachable { worker, source }
        }
    }

    /// Breaker-gated JSON POST: checks `breaker.allow()` first, records
    /// success/failure based on response status, and returns
    /// `ApiError::BreakerOpen` immediately when the breaker is Open.
    ///
    /// `worker_url` is the discovery-emitted worker URL string. It's parsed
    /// to [`reqwest::Url`] internally so we can use [`Url::join`] for clean
    /// path concatenation (no double-slash) and pass a typed URL to the
    /// split error variants (`UpstreamUnreachable` / `UpstreamTimeout` /
    /// `UpstreamStatus`).
    pub async fn forward_json_to(
        &self,
        worker_url: &str,
        protocol: WireProtocol,
        breaker: &CircuitBreaker,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Result<Response<Body>, ApiError> {
        if !breaker.allow() {
            return Err(ApiError::BreakerOpen {
                worker: worker_url.to_string(),
            });
        }
        let worker_url = parse_worker_url(worker_url, breaker)?;
        let url = worker_url.join(path).map_err(|e| {
            ApiError::Internal(anyhow::Error::new(e).context(format!("join worker path {path}")))
        })?;
        let mut req = self.client_for(protocol).post(url.clone()).body(body);
        for (k, v) in headers {
            if should_forward_request_header(k) {
                req = req.header(k, v);
            }
        }
        req = req
            .header("content-type", "application/json")
            .timeout(self.request_timeout);
        let resp = req.send().await.map_err(|e| {
            breaker.record_failure();
            Self::classify_reqwest_error_for(worker_url.clone(), e, path)
        })?;
        let status = resp.status();
        // Defer breaker recording until after the body completes — a
        // worker that returns 2xx headers and then drops mid-body is
        // still failing the request, and crediting it as healthy lets
        // a misbehaving worker stay eligible. For 5xx the early bail is
        // safe (no body to consume meaningfully), but we still wait
        // until after the read attempt to record exactly once.
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                // Walk the full source chain (`{:#}`) like the connect-error
                // handler in `classify_reqwest_error_for` — a mid-body drop's
                // real cause (incomplete message, connection reset) lives in the
                // wrapped source, not the outer reqwest error.
                let cause = anyhow::Error::new(e);
                tracing::warn!(
                    upstream = %url,
                    status = %status,
                    error = %format_args!("{cause:#}"),
                    "upstream dropped connection mid-body",
                );
                breaker.record_failure();
                return Err(ApiError::UpstreamStatus {
                    status,
                    worker: worker_url,
                });
            }
        };
        match breaker_outcome(status) {
            BreakerOutcome::Failure => breaker.record_failure(),
            BreakerOutcome::Success => breaker.record_success(),
            // Backpressure (503/429): the engine is healthy but busy. This never
            // opens the breaker and (in Closed) leaves the failure streak
            // intact, but it DOES resolve a half-open probe so a recovered
            // worker that answers a probe with 503 isn't wedged shut.
            BreakerOutcome::Neutral => breaker.record_backpressure(),
        }
        let mut out = Response::new(Body::from(bytes));
        *out.status_mut() = status;
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        );
        Ok(out)
    }

    /// Breaker-gated streaming POST: checks `breaker.allow()` first, records
    /// success/failure, and returns `ApiError::BreakerOpen` when Open.
    ///
    /// `stream_guards` — when `Some`, the value is threaded into the SSE
    /// pump task and held for the entire body lifetime (headers → last byte
    /// / client disconnect).  The proxy does not inspect the boxed value; it
    /// relies entirely on `Drop` semantics, so callers typically pack
    /// `(LoadGuard, ActiveLoadGuard)` here. This keeps both the per-worker
    /// `active_requests` counter and the per-request active-load entry alive
    /// for the full streaming lifetime — without which a long-running SSE
    /// response would under-report load.
    ///
    /// `abort_rid` — when `Some`, the request id the router injected into the
    /// forwarded body. On a successful stream this arms a client-disconnect
    /// abort: if the SSE pump is torn down before the engine finishes (client
    /// disconnect / non-draining stall), the engine is told to stop generating
    /// this rid. `None` disables it (e.g. callers that don't track a rid).
    ///
    /// `on_stream_end` — when `Some`, fires exactly once when the SSE pump
    /// finishes, with the pump's [`sse::StreamEnd`] verdict. Installed ONLY
    /// for 2xx upstream responses (a non-2xx error body draining is not a
    /// stream outcome worth classifying — the edge status metric already
    /// covers it). Callers use this to record
    /// `sgl_router_stream_outcome_total`, which is what distinguishes a real
    /// completed generation from an engine that committed `200 OK` and then
    /// reported failure via an in-band `data: {"error"...}` event — the two
    /// are identical to every headers-time metric.
    ///
    /// `on_inter_chunk` — when `Some`, fires once per upstream chunk after
    /// the first with the gap (seconds) since the previous chunk arrived.
    /// Installed ONLY for 2xx upstream responses (an error body's chunk
    /// pacing is not inter-token latency). Callers use this to record
    /// `sgl_router_itl_seconds`.
    // Each parameter is a distinct, required input to a single upstream
    // forward (target, protocol, breaker, path, headers, body, plus the
    // streaming-lifetime callbacks). Bundling them into a struct purely to
    // satisfy the arg-count heuristic would add indirection without clarity.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_streaming_to(
        &self,
        worker_url: &str,
        protocol: WireProtocol,
        breaker: &Arc<CircuitBreaker>,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
        stream_guards: Option<Box<dyn Send + 'static>>,
        on_first_byte: Option<Box<dyn FnOnce() + Send + 'static>>,
        abort_rid: Option<&str>,
        on_stream_end: Option<Box<dyn FnOnce(sse::StreamEnd) + Send + 'static>>,
        on_inter_chunk: Option<Box<dyn Fn(f64) + Send + 'static>>,
    ) -> Result<Response<Body>, ApiError> {
        if !breaker.allow() {
            return Err(ApiError::BreakerOpen {
                worker: worker_url.to_string(),
            });
        }
        let worker_url = parse_worker_url(worker_url, breaker)?;
        let url = worker_url.join(path).map_err(|e| {
            ApiError::Internal(anyhow::Error::new(e).context(format!("join worker path {path}")))
        })?;
        let mut req = self.client_for(protocol).post(url.clone()).body(body);
        for (k, v) in headers {
            if should_forward_request_header(k) {
                req = req.header(k, v);
            }
        }
        req = req
            .header("content-type", "application/json")
            .header("accept", "text/event-stream");
        // Diagnostic: count time blocked in upstream send (connect + write body +
        // await response headers). Scoped so it decrements the instant send
        // resolves, whether it returns headers or errors.
        //
        // Bounded by `request_timeout`, but via an EXTERNAL `tokio::time::timeout`
        // wrapping only this `.send()` future — not `req.timeout(self.request_timeout)`
        // (the builder method `forward_json_to` uses). Verified empirically: reqwest's
        // own per-request `.timeout()` covers the request's *entire* lifecycle,
        // including a manual `bytes_stream()` poll loop made well after `.send()`
        // already resolved — it is not a "wait for headers" timeout, it's "wait for
        // the whole exchange, however you choose to read the body." Setting it here
        // would silently re-impose `request_timeout` as a body-streaming cap, exactly
        // the regression `STREAM_IDLE_TIMEOUT` (below) and the "long generations are
        // valid" design this function documents elsewhere exist to avoid. An external
        // `tokio::time::timeout` has no such reach: once the wrapped future resolves,
        // the deadline is gone — the subsequent `bytes_stream()` consumption is
        // governed only by `STREAM_IDLE_TIMEOUT`, never by this.
        //
        // Without this, an engine that accepts the connection but never produces
        // headers (a wedged scheduler, not a connection failure) hangs the dispatch
        // forever — no breaker signal (it only fires on a completed response or
        // error), no client-visible failure, until some caller above the router
        // times out on its own and gives up with no explanation in this router's logs.
        let resp = {
            let _send_phase = crate::diag::PhaseGuard::in_send();
            match tokio::time::timeout(self.request_timeout, req.send()).await {
                Ok(Ok(resp)) => resp,
                Ok(Err(e)) => {
                    breaker.record_failure();
                    return Err(Self::classify_reqwest_error_for(
                        worker_url.clone(),
                        e,
                        path,
                    ));
                }
                Err(_elapsed) => {
                    breaker.record_failure();
                    return Err(ApiError::UpstreamTimeout {
                        worker: worker_url.clone(),
                    });
                }
            }
        };
        let status = resp.status();
        let upstream_ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/json")
            .to_string();
        let content_type = if status.is_success() {
            "text/event-stream".to_string()
        } else {
            upstream_ct
        };
        // Breaker recording is deferred to the pump's completion hook so
        // an upstream that returns 2xx headers and then drops mid-stream
        // is recorded as a failure. For a genuine 5xx fault we record_failure
        // up front and skip the pump hook (the body we surface is the
        // error response — its stream completing is not a worker win). For a
        // backpressure status (503/429) we record_backpressure up front and
        // skip the hook: a busy-but-healthy engine's queue-full responses can't
        // open the breaker, but a half-open probe answered with 503 is still
        // resolved rather than wedged (see `breaker_outcome` / `record_backpressure`).
        //
        // The breaker judges TRANSPORT health only: a clean in-band
        // `data: {"error"...}` event does not trip it (an application-level
        // verdict — visible via `on_stream_end` / the stream-outcome metric,
        // deliberately kept out of routing decisions for now).
        let stream_end_hook = if status.is_success() {
            on_stream_end
        } else {
            None
        };
        let on_complete: Option<Box<dyn FnOnce(sse::StreamEnd) + Send + 'static>> =
            match breaker_outcome(status) {
                BreakerOutcome::Failure => {
                    breaker.record_failure();
                    None
                }
                BreakerOutcome::Neutral => {
                    breaker.record_backpressure();
                    None
                }
                BreakerOutcome::Success => {
                    let breaker_for_hook = Arc::clone(breaker);
                    Some(Box::new(move |end: sse::StreamEnd| {
                        if end.transport_ok {
                            breaker_for_hook.record_success();
                        } else {
                            breaker_for_hook.record_failure();
                        }
                        if let Some(hook) = stream_end_hook {
                            hook(end);
                        }
                    }))
                }
            };
        // Only record TTFT for successful streams — a 4xx/5xx error body
        // streaming back is not a generated token, so drop the hook for
        // non-2xx responses.
        let first_byte_hook = if status.is_success() {
            on_first_byte
        } else {
            None
        };
        // Same gate for inter-token latency: an error body's chunk pacing is
        // not a token cadence.
        let inter_chunk_hook = if status.is_success() {
            on_inter_chunk
        } else {
            None
        };
        // Diagnostic: count this stream as an active SSE pump for its whole
        // lifetime by packing a pump-phase guard into the stream guards (created
        // post-headers, dropped when the pump task ends).
        let pump_phase = crate::diag::PhaseGuard::pump();
        // Client-disconnect abort: for a successful stream, wrap the upstream so
        // a `reached_end` flag flips true when the engine finishes (or fails),
        // and pack an `AbortOnDrop` reading that flag into the stream guards. If
        // the SSE pump tears down before the engine is done — client disconnect
        // or a non-draining stall — the guard drops with the flag still false and
        // tells the engine to stop. A non-2xx stream is the engine's own error
        // body (it isn't generating), so it is never abortable.
        let upstream: futures::stream::BoxStream<'static, Result<Bytes, std::io::Error>> =
            sse::idle_timeout_stream(resp.bytes_stream(), self.stream_idle_timeout);
        let (upstream, abort_guard, abort_reason_handle) = match abort_rid {
            Some(rid) if status.is_success() => match worker_url.join("/abort_request") {
                Ok(abort_url) => {
                    let reached_end = Arc::new(AtomicBool::new(false));
                    let guard = AbortOnDrop::for_stream(
                        self.client_for(protocol).clone(),
                        abort_url.to_string(),
                        rid.to_string(),
                        Arc::clone(&reached_end),
                        self.metrics_for_abort(),
                    );
                    // Extract the reason atom so the SSE pump can narrow it
                    // from the constructor default (`StreamClientGone`) to the
                    // specific cause (stall / panic) as it hits each break
                    // site. The atom is `Arc`-shared with the guard, so the
                    // pump's write is visible to the guard's `Drop`.
                    let reason_handle = guard.reason_handle();
                    (
                        sse::mark_terminal(upstream, reached_end),
                        Some(guard),
                        Some(reason_handle),
                    )
                }
                Err(_) => (upstream, None, None),
            },
            _ => (upstream, None, None),
        };
        let guards: Option<Box<dyn Send + 'static>> = match (stream_guards, abort_guard) {
            (Some(g), Some(a)) => Some(Box::new((g, pump_phase, a))),
            (Some(g), None) => Some(Box::new((g, pump_phase))),
            (None, Some(a)) => Some(Box::new((pump_phase, a))),
            (None, None) => Some(Box::new(pump_phase)),
        };
        let body = sse::bytes_stream_to_body_with_stall(
            upstream,
            guards,
            on_complete,
            first_byte_hook,
            abort_reason_handle,
            inter_chunk_hook,
            self.stream_send_stall,
        );
        let mut out = Response::new(body);
        *out.status_mut() = status;
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_str(&content_type)
                .unwrap_or_else(|_| HeaderValue::from_static("application/json")),
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
    use axum::extract::State;
    use axum::routing::post;
    use axum::Json;
    use axum::Router;
    use reqwest::StatusCode;
    use serde_json::Value;
    use std::num::NonZeroU32;
    use std::sync::Mutex;
    use std::time::Duration;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    #[tokio::test]
    async fn new_returns_result_not_panic() {
        let p = Proxy::new(Duration::from_secs(5)).unwrap();
        assert_eq!(p.request_timeout, Duration::from_secs(5));
        // Bare `new` keeps the streaming-stall defaults.
        assert_eq!(p.stream_idle_timeout, STREAM_IDLE_TIMEOUT);
        assert_eq!(p.stream_send_stall, sse::STREAM_SEND_STALL);
    }

    #[tokio::test]
    async fn with_stream_timeouts_overrides_both_legs() {
        let p = Proxy::new(Duration::from_secs(5))
            .unwrap()
            .with_stream_timeouts(Duration::from_secs(90), Duration::from_secs(45));
        assert_eq!(p.stream_idle_timeout, Duration::from_secs(90));
        assert_eq!(p.stream_send_stall, Duration::from_secs(45));
        // Untouched by the builder.
        assert_eq!(p.request_timeout, Duration::from_secs(5));
    }

    #[test]
    fn breaker_outcome_treats_backpressure_as_neutral() {
        // Backpressure: healthy but busy — must not touch the breaker.
        assert_eq!(
            breaker_outcome(StatusCode::SERVICE_UNAVAILABLE),
            BreakerOutcome::Neutral,
        );
        assert_eq!(
            breaker_outcome(StatusCode::TOO_MANY_REQUESTS),
            BreakerOutcome::Neutral,
        );
        // Genuine faults: still failures.
        for s in [
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
            StatusCode::GATEWAY_TIMEOUT,
        ] {
            assert_eq!(breaker_outcome(s), BreakerOutcome::Failure, "{s}");
        }
        // Non-5xx (incl. 4xx client errors): treated as success.
        for s in [
            StatusCode::OK,
            StatusCode::BAD_REQUEST,
            StatusCode::NOT_FOUND,
        ] {
            assert_eq!(breaker_outcome(s), BreakerOutcome::Success, "{s}");
        }
    }

    /// A fake upstream that answers every POST with a fixed status + tiny body.
    async fn spawn_status_worker(status: u16) -> (String, oneshot::Sender<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let code = StatusCode::from_u16(status).unwrap();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move || async move { (code, "{\"error\":\"x\"}") }),
        );
        let (tx, rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = rx.await;
                })
                .await;
        });
        (format!("http://127.0.0.1:{port}"), tx)
    }

    /// `spawn_status_worker`, plus an `/abort_request` route that records
    /// every POSTed body — so a test can assert a non-2xx response never
    /// triggers an abort, rather than just hoping a stray POST went nowhere.
    async fn spawn_status_worker_with_abort_capture(
        status: u16,
    ) -> (String, AbortLog, oneshot::Sender<()>) {
        let log: AbortLog = Arc::new(Mutex::new(Vec::new()));
        let code = StatusCode::from_u16(status).unwrap();
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(move || async move { (code, "{\"error\":\"x\"}") }),
            )
            .route("/abort_request", post(abort_request_handler))
            .with_state(log.clone());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = rx.await;
                })
                .await;
        });
        (format!("http://127.0.0.1:{port}"), log, tx)
    }

    /// The bug this fixes: a saturated engine returning its own queue-full 503s
    /// must NOT trip the router's circuit breaker. Dispatch well past the
    /// default threshold (3) and assert the breaker stays Closed and admitting.
    #[tokio::test]
    async fn engine_503_does_not_trip_breaker() {
        let (url, _shutdown) = spawn_status_worker(503).await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        let breaker = CircuitBreaker::new(); // default threshold = 3
        let headers = HeaderMap::new();

        for i in 0..6 {
            let resp = proxy
                .forward_json_to(
                    &url,
                    WireProtocol::Http1,
                    &breaker,
                    "/v1/chat/completions",
                    &headers,
                    Bytes::from_static(b"{}"),
                )
                .await
                .expect("dispatch should reach the worker (breaker must stay closed)");
            assert_eq!(
                resp.status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "iter {i}: client must still see the engine's 503",
            );
            assert_eq!(
                breaker.snapshot().state_code,
                0,
                "iter {i}: 503 backpressure must leave the breaker Closed",
            );
        }
        assert!(
            breaker.would_allow(),
            "breaker must keep admitting after a burst of engine 503s",
        );
    }

    /// Contrast / regression guard: a genuine 5xx fault (500) MUST still trip the
    /// breaker after the threshold, so the backpressure carve-out didn't disable
    /// fault detection.
    #[tokio::test]
    async fn engine_500_still_trips_breaker() {
        let (url, _shutdown) = spawn_status_worker(500).await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        let breaker = CircuitBreaker::new(); // default threshold = 3
        let headers = HeaderMap::new();

        for _ in 0..3 {
            let _ = proxy
                .forward_json_to(
                    &url,
                    WireProtocol::Http1,
                    &breaker,
                    "/v1/chat/completions",
                    &headers,
                    Bytes::from_static(b"{}"),
                )
                .await;
        }
        assert_eq!(
            breaker.snapshot().state_code,
            1,
            "three 500s must open the breaker (fault detection still works)",
        );
    }

    /// End-to-end wedge guard: a breaker that opened on real faults, then has
    /// its half-open probe answered with a 503, must RECOVER — not stay shut
    /// out forever. Exercises the `Neutral => record_backpressure` wiring in
    /// `forward_json_to` through the half-open path.
    #[tokio::test]
    async fn engine_503_recovers_a_half_open_breaker() {
        let (url, _shutdown) = spawn_status_worker(503).await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        // threshold=1 so one prior fault opens it; tiny cooldown so the probe
        // is admitted almost immediately.
        let breaker = CircuitBreaker::with_config(CircuitBreakerConfig {
            threshold: NonZeroU32::new(1).unwrap(),
            cool_down: Duration::from_millis(50),
        });
        let headers = HeaderMap::new();

        // Simulate a prior genuine fault (e.g. a 500 / timeout) that tripped it.
        breaker.record_failure();
        assert_eq!(breaker.snapshot().state_code, 1, "breaker should be Open");

        // Let the cooldown elapse so the next dispatch claims the half-open probe.
        tokio::time::sleep(Duration::from_millis(80)).await;

        let resp = proxy
            .forward_json_to(
                &url,
                WireProtocol::Http1,
                &breaker,
                "/v1/chat/completions",
                &headers,
                Bytes::from_static(b"{}"),
            )
            .await
            .expect("the half-open probe must be admitted and reach the worker");
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            breaker.snapshot().state_code,
            0,
            "a 503 answer to the probe must close the breaker, not wedge it half-open",
        );
        assert!(
            breaker.would_allow(),
            "worker must admit traffic again after recovering from the probe",
        );
    }

    /// A worker's own response is forwarded with its status VERBATIM and carries
    /// NO `x-router-error-code` header. The absence of that header is exactly how
    /// a gateway tells "the engine said this" from "the router said this" — a
    /// worker 2xx, 4xx, and 5xx (incl. a complete 500, distinct from a synthesized
    /// 502 mid-body drop) all pass straight through, unannotated.
    #[tokio::test]
    async fn forwarded_worker_response_is_verbatim_with_no_router_error_code() {
        for status in [200u16, 400, 500, 503] {
            let (url, _shutdown) = spawn_status_worker(status).await;
            let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
            let breaker = CircuitBreaker::new();
            let resp = proxy
                .forward_json_to(
                    &url,
                    WireProtocol::Http1,
                    &breaker,
                    "/v1/chat/completions",
                    &HeaderMap::new(),
                    Bytes::from_static(b"{}"),
                )
                .await
                .expect("dispatch should reach the worker");
            assert_eq!(
                resp.status().as_u16(),
                status,
                "worker status {status} must be forwarded verbatim",
            );
            assert!(
                resp.headers().get("x-router-error-code").is_none(),
                "a forwarded worker response must NOT carry x-router-error-code (status {status})",
            );
            assert!(
                resp.headers().get("x-router-upstream-status").is_none(),
                "a forwarded worker response must NOT carry x-router-upstream-status (status {status})",
            );
        }
    }

    /// Streaming path parity: the engine's 503 on the streaming arm must also
    /// leave the breaker untouched (no up-front failure, no completion hook).
    #[tokio::test]
    async fn engine_503_does_not_trip_breaker_streaming() {
        use http_body_util::BodyExt;

        let (url, _shutdown) = spawn_status_worker(503).await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        let breaker = Arc::new(CircuitBreaker::new());
        let headers = HeaderMap::new();

        for i in 0..6 {
            let resp = proxy
                .forward_streaming_to(
                    &url,
                    WireProtocol::Http1,
                    &breaker,
                    "/v1/chat/completions",
                    &headers,
                    Bytes::from_static(b"{}"),
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .expect("streaming dispatch should reach the worker");
            assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE, "iter {i}");
            // Drain the body so the pump task runs to completion (would fire any
            // completion hook). For a 503 there is none, but draining proves it.
            let _ = resp.into_body().collect().await;
            assert_eq!(
                breaker.snapshot().state_code,
                0,
                "iter {i}: streaming 503 must leave the breaker Closed",
            );
        }
    }

    /// Both clients are built up front and `client_for` selects between them
    /// purely by the per-request protocol — no shared cell, so one worker's
    /// protocol never influences another's. The on-the-wire behavior of each
    /// client is covered by tests/proxy/h2c_forward.rs; here we only pin that
    /// the selector returns distinct clients per protocol.
    #[tokio::test]
    async fn client_for_selects_distinct_clients_per_protocol() {
        let p = Proxy::new(Duration::from_secs(5)).unwrap();
        let h1 = p.client_for(WireProtocol::Http1);
        let h2 = p.client_for(WireProtocol::H2c);
        assert!(
            !std::ptr::eq(h1, h2),
            "h2c and HTTP/1.1 requests must use different clients",
        );
        // The selector is stable per protocol and matches the admin client for
        // HTTP/1.1.
        assert!(std::ptr::eq(
            p.client_for(WireProtocol::Http1),
            p.admin_client()
        ));
    }

    // ---- AbortOnDrop / send_abort -----------------------------------------

    /// Every `/abort_request` POST appends its parsed JSON body here, so tests
    /// can assert on `rid` / `abort_all` without racing a single "last body"
    /// slot.
    type AbortLog = Arc<Mutex<Vec<Value>>>;

    async fn abort_request_handler(
        State(log): State<AbortLog>,
        Json(body): Json<Value>,
    ) -> StatusCode {
        log.lock().unwrap().push(body);
        StatusCode::OK
    }

    /// A fake upstream whose `/v1/chat/completions` sleeps `delay` before
    /// answering 200 (simulating an engine still generating a unary
    /// response), and whose `/abort_request` records every POSTed body.
    async fn spawn_hanging_worker_with_abort_capture(
        delay: Duration,
    ) -> (String, AbortLog, oneshot::Sender<()>) {
        let log: AbortLog = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(move || async move {
                    tokio::time::sleep(delay).await;
                    (StatusCode::OK, "{}")
                }),
            )
            .route("/abort_request", post(abort_request_handler))
            .with_state(log.clone());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = rx.await;
                })
                .await;
        });
        (format!("http://127.0.0.1:{port}"), log, tx)
    }

    /// A fake upstream whose `/v1/chat/completions` streams `chunks` with
    /// `delay` between each (an SSE body, like the real engine), and whose
    /// `/abort_request` records every POSTed body. Mirrors
    /// `tests/proxy/common/mock_worker.rs::start_slow_stream`, duplicated here
    /// because that helper lives in the separate `tests/` integration-test
    /// crate and isn't visible to this lib-internal `#[cfg(test)]` module.
    async fn spawn_streaming_worker_with_abort_capture(
        chunks: Vec<&'static str>,
        delay: Duration,
    ) -> (String, AbortLog, oneshot::Sender<()>) {
        let log: AbortLog = Arc::new(Mutex::new(Vec::new()));

        async fn stream_chat(chunks: Vec<&'static str>, delay: Duration) -> Response<Body> {
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(4);
            tokio::spawn(async move {
                for c in chunks {
                    tokio::time::sleep(delay).await;
                    if tx.send(Ok(Bytes::from(c))).await.is_err() {
                        break;
                    }
                }
            });
            let body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
            let mut r = Response::new(body);
            *r.status_mut() = StatusCode::OK;
            r.headers_mut().insert(
                HeaderName::from_static("content-type"),
                HeaderValue::from_static("text/event-stream"),
            );
            r
        }

        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(move || stream_chat(chunks.clone(), delay)),
            )
            .route("/abort_request", post(abort_request_handler))
            .with_state(log.clone());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = rx.await;
                })
                .await;
        });
        (format!("http://127.0.0.1:{port}"), log, tx)
    }

    /// Same shape as [`spawn_streaming_worker_with_abort_capture`], except
    /// `/abort_request` always answers `500` (while still recording the
    /// POSTed body). Used to prove a failing abort never reaches the circuit
    /// breaker — `send_abort` has no breaker access by construction, so this
    /// pins that invariant against a future refactor that might add one.
    async fn spawn_streaming_worker_with_failing_abort(
        chunks: Vec<&'static str>,
        delay: Duration,
    ) -> (String, AbortLog, oneshot::Sender<()>) {
        let log: AbortLog = Arc::new(Mutex::new(Vec::new()));

        async fn stream_chat(chunks: Vec<&'static str>, delay: Duration) -> Response<Body> {
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(4);
            tokio::spawn(async move {
                for c in chunks {
                    tokio::time::sleep(delay).await;
                    if tx.send(Ok(Bytes::from(c))).await.is_err() {
                        break;
                    }
                }
            });
            let body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
            let mut r = Response::new(body);
            *r.status_mut() = StatusCode::OK;
            r.headers_mut().insert(
                HeaderName::from_static("content-type"),
                HeaderValue::from_static("text/event-stream"),
            );
            r
        }

        async fn failing_abort_handler(
            State(log): State<AbortLog>,
            Json(body): Json<Value>,
        ) -> StatusCode {
            log.lock().unwrap().push(body);
            StatusCode::INTERNAL_SERVER_ERROR
        }

        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(move || stream_chat(chunks.clone(), delay)),
            )
            .route("/abort_request", post(failing_abort_handler))
            .with_state(log.clone());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = rx.await;
                })
                .await;
        });
        (format!("http://127.0.0.1:{port}"), log, tx)
    }

    /// Poll `log` until it has at least one entry or `timeout` elapses.
    async fn wait_for_abort(log: &AbortLog, timeout: Duration) {
        let deadline = std::time::Instant::now() + timeout;
        while log.lock().unwrap().is_empty() && std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Unary guard: dropped while still armed (no `disarm()` call) must POST
    /// exactly one abort — the handler-cancelled-by-client-disconnect case.
    #[tokio::test]
    async fn abort_on_drop_unary_fires_when_dropped_armed() {
        let (url, abort_log, _shutdown) =
            spawn_hanging_worker_with_abort_capture(Duration::from_secs(10)).await;
        let client = Client::new();
        {
            let _guard = AbortOnDrop::for_unary(
                client,
                format!("{url}/abort_request"),
                "test-rid-1".into(),
                None,
            );
        }
        wait_for_abort(&abort_log, Duration::from_secs(2)).await;
        let log = abort_log.lock().unwrap();
        assert_eq!(
            log.len(),
            1,
            "an armed guard dropped without disarm must POST exactly one abort"
        );
        assert_eq!(log[0]["rid"], "test-rid-1");
        assert_eq!(log[0]["abort_all"], false);
        // No call site refined the reason before drop, so the constructor
        // default (unary → `HandlerCancelled`, the catch-all covering
        // "handler future dropped mid-await") must be what the engine sees.
        assert_eq!(log[0]["router_reason"], "handler_cancelled");
    }

    /// Unary guard: `disarm()` before drop (a complete response was received)
    /// must suppress the abort entirely.
    #[tokio::test]
    async fn abort_on_drop_unary_does_not_fire_when_disarmed() {
        let (url, abort_log, _shutdown) =
            spawn_hanging_worker_with_abort_capture(Duration::from_secs(10)).await;
        let client = Client::new();
        {
            let mut guard = AbortOnDrop::for_unary(
                client,
                format!("{url}/abort_request"),
                "test-rid-2".into(),
                None,
            );
            guard.disarm();
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            abort_log.lock().unwrap().is_empty(),
            "a disarmed guard must never abort"
        );
    }

    /// Streaming guard: dropped with `reached_end` still false (the SSE pump
    /// was torn down before the engine's terminal item) must abort.
    #[tokio::test]
    async fn abort_on_drop_stream_fires_when_reached_end_false() {
        let (url, abort_log, _shutdown) =
            spawn_hanging_worker_with_abort_capture(Duration::from_secs(10)).await;
        let client = Client::new();
        let reached_end = Arc::new(AtomicBool::new(false));
        {
            let _guard = AbortOnDrop::for_stream(
                client,
                format!("{url}/abort_request"),
                "test-rid-3".into(),
                Arc::clone(&reached_end),
                None,
            );
        }
        wait_for_abort(&abort_log, Duration::from_secs(2)).await;
        let log = abort_log.lock().unwrap();
        assert_eq!(
            log.len(),
            1,
            "reached_end=false at drop (client gone before engine finished) must abort"
        );
        // Streaming guards default to `StreamClientGone` — the majority
        // real-world cause when the pump exits without setting a narrower
        // reason (e.g., a test that drops the guard directly without ever
        // running the pump loop).
        assert_eq!(log[0]["router_reason"], "stream_client_gone");
    }

    /// Streaming guard: dropped with `reached_end` already true (the engine
    /// reached its clean terminal item, or a terminal error) must NOT abort —
    /// this is the normal-completion path and must never pollute engine abort
    /// metrics.
    #[tokio::test]
    async fn abort_on_drop_stream_does_not_fire_when_reached_end_true() {
        let (url, abort_log, _shutdown) =
            spawn_hanging_worker_with_abort_capture(Duration::from_secs(10)).await;
        let client = Client::new();
        let reached_end = Arc::new(AtomicBool::new(true));
        {
            let _guard = AbortOnDrop::for_stream(
                client,
                format!("{url}/abort_request"),
                "test-rid-4".into(),
                Arc::clone(&reached_end),
                None,
            );
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            abort_log.lock().unwrap().is_empty(),
            "reached_end=true at drop (normal completion) must NOT abort"
        );
    }

    /// `send_abort` posts `{rid, abort_all:false, router_reason}` to the given URL.
    /// The `router_reason` field lets operators break down engine-side aborts
    /// by the router-side trigger without correlating by rid alone.
    #[tokio::test]
    async fn send_abort_posts_rid_and_abort_all_false() {
        let (url, abort_log, _shutdown) =
            spawn_hanging_worker_with_abort_capture(Duration::from_secs(10)).await;
        let client = Client::new();
        send_abort(
            &client,
            &format!("{url}/abort_request"),
            "direct-rid",
            AbortReason::StreamClientGone,
        )
        .await;
        let log = abort_log.lock().unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0]["rid"], "direct-rid");
        assert_eq!(log[0]["abort_all"], false);
        assert_eq!(log[0]["router_reason"], "stream_client_gone");
    }

    /// `send_abort` is best-effort: an unreachable abort URL (connection
    /// refused) must be swallowed — logged, not panicked or propagated.
    #[tokio::test]
    async fn send_abort_swallows_unreachable_url_without_panicking() {
        let client = Client::new();
        // Port 1 is privileged / never listened on in CI sandboxes — refused
        // promptly. No assertion beyond "this does not panic or hang".
        send_abort(
            &client,
            "http://127.0.0.1:1/abort_request",
            "rid-x",
            AbortReason::HandlerCancelled,
        )
        .await;
    }

    /// The abort guard's constructor default is refined by `set_reason` at the
    /// call site; the final `router_reason` on the wire must match the last
    /// value stored, not the default. Exercises the `Arc<AtomicU8>` handoff.
    #[tokio::test]
    async fn set_reason_narrows_the_default_before_drop() {
        let (url, abort_log, _shutdown) =
            spawn_hanging_worker_with_abort_capture(Duration::from_secs(10)).await;
        let client = Client::new();
        {
            let guard = AbortOnDrop::for_unary(
                client,
                format!("{url}/abort_request"),
                "narrowed-rid".into(),
                None,
            );
            // Default is `HandlerCancelled`; narrow to `UpstreamTimeout`.
            guard.set_reason(AbortReason::UpstreamTimeout);
            // guard drops at end of scope while armed
        }
        wait_for_abort(&abort_log, Duration::from_secs(2)).await;
        let log = abort_log.lock().unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0]["router_reason"], "upstream_timeout");
    }

    /// A guard whose `reason_handle` is written from a different task than
    /// the one that owns the guard (the streaming SSE pump pattern) still
    /// sees the narrowed reason on Drop.
    #[tokio::test]
    async fn reason_handle_writes_are_visible_to_drop() {
        let (url, abort_log, _shutdown) =
            spawn_hanging_worker_with_abort_capture(Duration::from_secs(10)).await;
        let client = Client::new();
        let reached_end = Arc::new(AtomicBool::new(false));
        let guard = AbortOnDrop::for_stream(
            client,
            format!("{url}/abort_request"),
            "cross-task-rid".into(),
            Arc::clone(&reached_end),
            None,
        );
        let handle = guard.reason_handle();
        // Simulate the pump's write from a different task.
        tokio::spawn(async move {
            handle.store(AbortReason::StreamDownstreamStall as u8, Ordering::Relaxed);
        })
        .await
        .unwrap();
        drop(guard);
        wait_for_abort(&abort_log, Duration::from_secs(2)).await;
        let log = abort_log.lock().unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0]["router_reason"], "stream_downstream_stall");
    }

    /// `abort_guard_for` returns `None` for a worker URL that can't be parsed
    /// — matching the moot forward failure (the dispatch itself would fail the
    /// same way, so there is nothing to abort).
    #[tokio::test]
    async fn abort_guard_for_returns_none_for_unparsable_url() {
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        let guard = proxy.abort_guard_for("not a valid url", WireProtocol::Http1, "rid");
        assert!(
            guard.is_none(),
            "an unparsable worker URL must yield no guard"
        );
    }

    /// `abort_guard_for` joins `worker_url` + `/abort_request` correctly: a
    /// guard built through it (not constructed directly) must still reach the
    /// right endpoint when dropped armed.
    #[tokio::test]
    async fn abort_guard_for_builds_a_working_guard() {
        let (url, abort_log, _shutdown) =
            spawn_hanging_worker_with_abort_capture(Duration::from_secs(10)).await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        {
            let _guard = proxy
                .abort_guard_for(&url, WireProtocol::Http1, "rid-via-proxy")
                .expect("a well-formed worker URL must yield a guard");
        }
        wait_for_abort(&abort_log, Duration::from_secs(2)).await;
        let log = abort_log.lock().unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0]["rid"], "rid-via-proxy");
    }

    // ---- forward_streaming_to abort wiring --------------------------------

    /// End-to-end through `forward_streaming_to`: a client that disconnects
    /// before the upstream stream reaches its terminal item must trigger an
    /// abort POST carrying the `abort_rid` the caller supplied.
    #[tokio::test]
    async fn forward_streaming_to_aborts_on_client_disconnect() {
        let (url, abort_log, _shutdown) = spawn_streaming_worker_with_abort_capture(
            vec!["data: a\n\n", "data: b\n\n", "data: c\n\n"],
            Duration::from_millis(50),
        )
        .await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        let breaker = Arc::new(CircuitBreaker::new());
        let resp = proxy
            .forward_streaming_to(
                &url,
                WireProtocol::Http1,
                &breaker,
                "/v1/chat/completions",
                &HeaderMap::new(),
                Bytes::from_static(b"{}"),
                None,
                None,
                Some("stream-rid-1"),
                None,
                None,
            )
            .await
            .expect("streaming dispatch should reach the worker");

        // Read exactly one chunk, then drop the body — simulating the client
        // going away before the engine finishes streaming.
        let mut data_stream = resp.into_body().into_data_stream();
        use futures::StreamExt;
        let first = data_stream.next().await;
        assert!(first.is_some(), "expected at least one chunk before drop");
        drop(data_stream);

        wait_for_abort(&abort_log, Duration::from_secs(2)).await;
        let log = abort_log.lock().unwrap();
        assert_eq!(
            log.len(),
            1,
            "client disconnect mid-stream must trigger exactly one abort"
        );
        assert_eq!(log[0]["rid"], "stream-rid-1");
        assert_eq!(log[0]["abort_all"], false);
    }

    /// Contrast: a stream drained to its normal completion must NEVER abort —
    /// the engine finished on its own, so telling it to stop would be a
    /// spurious abort that pollutes engine abort metrics.
    #[tokio::test]
    async fn forward_streaming_to_does_not_abort_on_normal_completion() {
        use http_body_util::BodyExt;

        let (url, abort_log, _shutdown) = spawn_streaming_worker_with_abort_capture(
            vec!["data: a\n\n", "data: b\n\n"],
            Duration::from_millis(10),
        )
        .await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        let breaker = Arc::new(CircuitBreaker::new());
        let resp = proxy
            .forward_streaming_to(
                &url,
                WireProtocol::Http1,
                &breaker,
                "/v1/chat/completions",
                &HeaderMap::new(),
                Bytes::from_static(b"{}"),
                None,
                None,
                Some("stream-rid-2"),
                None,
                None,
            )
            .await
            .expect("streaming dispatch should reach the worker");

        // Drain to completion — the normal-completion path.
        let _ = resp.into_body().collect().await;
        // Give the pump's mark_terminal + guard-drop a moment to run.
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            abort_log.lock().unwrap().is_empty(),
            "a stream drained to completion must never trigger an abort"
        );
    }

    /// A non-2xx upstream response is the engine's own error body (it isn't
    /// generating), so it must never be abortable even if the client
    /// disconnects while reading it.
    #[tokio::test]
    async fn forward_streaming_to_does_not_abort_non_2xx_response() {
        let (url, abort_log, _shutdown) = spawn_status_worker_with_abort_capture(503).await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        let breaker = Arc::new(CircuitBreaker::new());
        let resp = proxy
            .forward_streaming_to(
                &url,
                WireProtocol::Http1,
                &breaker,
                "/v1/chat/completions",
                &HeaderMap::new(),
                Bytes::from_static(b"{}"),
                None,
                None,
                Some("stream-rid-3"),
                None,
                None,
            )
            .await
            .expect("streaming dispatch should reach the worker");
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        // Drop the body without draining it — the same "client disconnect"
        // shape as the positive abort test — to prove the `status.is_success()`
        // gate (not just an absent client disconnect) is what suppresses the
        // abort here.
        use futures::StreamExt;
        let mut data_stream = resp.into_body().into_data_stream();
        let _ = data_stream.next().await;
        drop(data_stream);

        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            abort_log.lock().unwrap().is_empty(),
            "a non-2xx response must never trigger an abort, even on client disconnect"
        );
    }

    /// A worker that accepts the connection but never produces a response at
    /// all (a wedged scheduler, not a connection failure) must time out
    /// within `request_timeout`, return `UpstreamTimeout`, and trip the
    /// breaker — not hang the dispatch forever. This is the headers-await gap:
    /// before this guard existed, nothing router-side bounded this wait.
    #[tokio::test]
    async fn forward_streaming_to_times_out_when_headers_never_arrive() {
        // Worker "responds" after 10s; the test's Proxy has a 150ms timeout,
        // so the dispatch must fail long before the worker would ever answer.
        let (url, _abort_log, _shutdown) =
            spawn_hanging_worker_with_abort_capture(Duration::from_secs(10)).await;
        let proxy = Proxy::new(Duration::from_millis(150)).unwrap();
        // threshold=1: a single headers-await timeout must be enough to trip
        // it, same as any other dispatch failure.
        let breaker = Arc::new(CircuitBreaker::with_config(CircuitBreakerConfig {
            threshold: NonZeroU32::new(1).unwrap(),
            cool_down: Duration::from_secs(30),
        }));

        let start = std::time::Instant::now();
        let result = proxy
            .forward_streaming_to(
                &url,
                WireProtocol::Http1,
                &breaker,
                "/v1/chat/completions",
                &HeaderMap::new(),
                Bytes::from_static(b"{}"),
                None,
                None,
                Some("headers-timeout-rid"),
                None,
                None,
            )
            .await;

        assert!(
            start.elapsed() < Duration::from_secs(2),
            "a wedged engine must time out promptly (≈150ms), not hang — took {:?}",
            start.elapsed()
        );
        match result {
            Err(ApiError::UpstreamTimeout { .. }) => {}
            other => panic!("expected UpstreamTimeout, got {other:?}"),
        }
        assert_eq!(
            breaker.snapshot().state_code,
            1,
            "a headers-await timeout must count as a breaker failure, same as any other dispatch failure"
        );
    }

    /// `forward_json_to`'s pre-existing timeout mechanism (`req.timeout(self.request_timeout)`,
    /// the builder method — appropriate there since non-streaming has no
    /// later body-stream phase for it to over-reach into) had no test of its
    /// own anywhere in this file before this commit, despite this same
    /// commit explaining in detail why that mechanism would be wrong for
    /// `forward_streaming_to`. Mirrors
    /// `forward_streaming_to_times_out_when_headers_never_arrive` for the
    /// unary path, closing that gap: a never-responding worker must still
    /// fail fast with `UpstreamTimeout` and trip the breaker.
    #[tokio::test]
    async fn forward_json_to_times_out_when_response_never_arrives() {
        let (url, _abort_log, _shutdown) =
            spawn_hanging_worker_with_abort_capture(Duration::from_secs(10)).await;
        let proxy = Proxy::new(Duration::from_millis(150)).unwrap();
        let breaker = Arc::new(CircuitBreaker::with_config(CircuitBreakerConfig {
            threshold: NonZeroU32::new(1).unwrap(),
            cool_down: Duration::from_secs(30),
        }));

        let start = std::time::Instant::now();
        let result = proxy
            .forward_json_to(
                &url,
                WireProtocol::Http1,
                &breaker,
                "/v1/chat/completions",
                &HeaderMap::new(),
                Bytes::from_static(b"{}"),
            )
            .await;

        assert!(
            start.elapsed() < Duration::from_secs(2),
            "a wedged engine must time out promptly (≈150ms), not hang — took {:?}",
            start.elapsed()
        );
        match result {
            Err(ApiError::UpstreamTimeout { .. }) => {}
            other => panic!("expected UpstreamTimeout, got {other:?}"),
        }
        assert_eq!(
            breaker.snapshot().state_code,
            1,
            "a response timeout must count as a breaker failure, same as the streaming arm"
        );
    }

    /// The headers-await timeout must NOT cap the body-streaming phase that
    /// follows. Verified empirically (outside this crate) that reqwest's own
    /// per-request `.timeout()` covers the WHOLE exchange — including a
    /// manual `bytes_stream()` poll loop made well after `.send()` already
    /// resolved — which is exactly why the fix uses an external
    /// `tokio::time::timeout` around only `.send()` instead. This test pins
    /// that a stream running longer than `request_timeout`, entirely via
    /// legitimate slow-but-progressing chunks (never idle long enough to
    /// trip `STREAM_IDLE_TIMEOUT` either), completes intact.
    #[tokio::test]
    async fn forward_streaming_to_body_duration_is_not_capped_by_headers_timeout() {
        let (url, _abort_log, _shutdown) = spawn_streaming_worker_with_abort_capture(
            vec!["data: a\n\n", "data: b\n\n", "data: c\n\n", "data: d\n\n"],
            Duration::from_millis(150),
        )
        .await;
        // 4 chunks * 150ms = 600ms total body time, well past this 200ms
        // headers-timeout — but headers arrive immediately, so only the
        // (unbounded-by-this-guard) body phase is what runs long.
        let proxy = Proxy::new(Duration::from_millis(200)).unwrap();
        let breaker = Arc::new(CircuitBreaker::new());

        let resp = proxy
            .forward_streaming_to(
                &url,
                WireProtocol::Http1,
                &breaker,
                "/v1/chat/completions",
                &HeaderMap::new(),
                Bytes::from_static(b"{}"),
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .expect("headers arrive well within the timeout");
        assert_eq!(resp.status(), StatusCode::OK);

        use http_body_util::BodyExt;
        let body = resp
            .into_body()
            .collect()
            .await
            .expect("the full slow stream must complete, not be cut off at ~200ms")
            .to_bytes();
        let body_str = String::from_utf8_lossy(&body);
        for chunk in ["data: a", "data: b", "data: c", "data: d"] {
            assert!(
                body_str.contains(chunk),
                "expected all 4 slow chunks (600ms total) past the 200ms headers-timeout; got: {body_str}"
            );
        }
    }

    /// `send_abort` must never affect the worker's circuit breaker — an
    /// abort is a courtesy to the engine, never counted against a worker (see
    /// `send_abort`'s doc comment). `send_abort` has no breaker parameter at
    /// all today, so this can't currently fail; this is a regression guard
    /// against a future refactor that routes the abort POST through a
    /// breaker-gated path (a change that would itself require adding a
    /// breaker parameter to `send_abort`'s signature — the more durable
    /// protection here is that shape change being visible at review time).
    ///
    /// Threshold is 1 so a leaked `record_failure()` would open the breaker
    /// off a single failure. Note this is NOT a fully race-proof guard: each
    /// iteration's own chat request succeeds and disconnects from a healthy
    /// stream, which records a breaker SUCCESS synchronously, inline in the
    /// SSE pump's completion hook (`bytes_stream_to_body`'s `on_complete`) —
    /// the moment `AbortOnDrop` drops and spawns `send_abort` onto the
    /// runtime. A hypothetical regressed `send_abort` recording a failure
    /// only does so later, after a real async network round-trip, so the
    /// synchronous success is very likely to win that race and mask a
    /// transient open before either `would_allow()` below or the next
    /// iteration's `breaker.allow()` check observes it. Treat this test as a
    /// best-effort behavioral pin, not a deterministic regression detector —
    /// the API-shape argument above is what actually carries the guarantee.
    #[tokio::test]
    async fn send_abort_failure_does_not_trip_circuit_breaker() {
        let (url, abort_log, _shutdown) = spawn_streaming_worker_with_failing_abort(
            vec!["data: a\n\n", "data: b\n\n", "data: c\n\n"],
            Duration::from_millis(50),
        )
        .await;
        let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
        let breaker = Arc::new(CircuitBreaker::with_config(CircuitBreakerConfig {
            threshold: NonZeroU32::new(1).unwrap(),
            cool_down: Duration::from_secs(30),
        }));

        for i in 0..3 {
            let resp = proxy
                .forward_streaming_to(
                    &url,
                    WireProtocol::Http1,
                    &breaker,
                    "/v1/chat/completions",
                    &HeaderMap::new(),
                    Bytes::from_static(b"{}"),
                    None,
                    None,
                    Some(&format!("breaker-test-rid-{i}")),
                    None,
                    None,
                )
                .await
                .unwrap_or_else(|e| {
                    panic!(
                        "iter {i}: dispatch must reach the worker (breaker must stay closed): {e}"
                    )
                });
            assert_eq!(resp.status(), StatusCode::OK, "iter {i}");

            use futures::StreamExt;
            let mut data_stream = resp.into_body().into_data_stream();
            assert!(
                data_stream.next().await.is_some(),
                "iter {i}: expected at least one chunk before drop"
            );
            drop(data_stream);
        }

        wait_for_abort_count(&abort_log, 3, Duration::from_secs(2)).await;
        assert_eq!(
            abort_log.lock().unwrap().len(),
            3,
            "all 3 disconnects must have attempted an abort, even though each one 500s"
        );
        assert!(
            breaker.would_allow(),
            "3 failed (500) /abort_request POSTs must NOT trip the breaker"
        );
    }

    /// Poll `log` until it has at least `count` entries or `timeout` elapses.
    async fn wait_for_abort_count(log: &AbortLog, count: usize, timeout: Duration) {
        let deadline = std::time::Instant::now() + timeout;
        while log.lock().unwrap().len() < count && std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}
