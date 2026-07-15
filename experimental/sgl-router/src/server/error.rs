// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::header_utils::SERVER_TIMING;
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use thiserror::Error;

pub const X_ROUTER_ERROR_CODE: HeaderName = HeaderName::from_static("x-router-error-code");

/// Carries the worker's *own* HTTP status when the router had to synthesize a
/// status of its own over a worker that did respond (today: a mid-body drop,
/// where headers arrived but the body did not). Lets a gateway / operator
/// recover what the engine actually said instead of seeing only the router's
/// synthesized 502. Absent on every other response: a forwarded worker response
/// already carries the worker's status in the status line, and a router-only
/// condition (admission shed, no workers, …) has no upstream status to report.
pub const X_ROUTER_UPSTREAM_STATUS: HeaderName =
    HeaderName::from_static("x-router-upstream-status");

/// Coarse failure class for a router-originated error. The router's HTTP status
/// is a pure function of the class, so two conditions that mean the same thing
/// — e.g. a per-request timeout and a stale-deadline cancel — can never drift to
/// different status codes. The *precise* condition travels in
/// `x-router-error-code` (see [`ApiError::error_code`]): a gateway in front
/// converts on that header (the authoritative signal), while the status stays a
/// self-sufficient HTTP-honest default for a direct caller. The class never
/// contradicts the precise code — it only generalizes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ErrorClass {
    /// 400 — request rejected at ingress as malformed / invalid.
    BadRequest,
    /// 404 — requested model / resource not found.
    NotFound,
    /// 502 — a selected worker failed to return a usable response (unreachable,
    /// or started a response then dropped the body).
    Upstream,
    /// 503 — the router had no worker to dispatch to, or declined to.
    NoTarget,
    /// 504 — the router gave up waiting (any timeout / deadline).
    Timeout,
    /// 500 — internal router fault.
    Internal,
}

impl ErrorClass {
    fn status(self) -> StatusCode {
        match self {
            ErrorClass::BadRequest => StatusCode::BAD_REQUEST,
            ErrorClass::NotFound => StatusCode::NOT_FOUND,
            ErrorClass::Upstream => StatusCode::BAD_GATEWAY,
            ErrorClass::NoTarget => StatusCode::SERVICE_UNAVAILABLE,
            ErrorClass::Timeout => StatusCode::GATEWAY_TIMEOUT,
            ErrorClass::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

/// Where in the request lifecycle an [`ApiError`] originated — see
/// [`ApiError::stage`] for the per-variant mapping and rationale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    Ingress,
    Queue,
    Dispatch,
}

impl Stage {
    /// The pre-composed `Server-Timing` entry for this stage. A `HeaderValue`
    /// (not a `&'static str`) so the call site (`into_response`) is a plain
    /// `append`, with no runtime formatting and no fallible parse — the three
    /// values are fixed at compile time.
    fn server_timing(self) -> HeaderValue {
        match self {
            Stage::Ingress => HeaderValue::from_static("router.stage;desc=ingress"),
            Stage::Queue => HeaderValue::from_static("router.stage;desc=queue"),
            Stage::Dispatch => HeaderValue::from_static("router.stage;desc=dispatch"),
        }
    }
}

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("bad request: {0}")]
    BadRequest(String),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// Could not reach the upstream worker (connect refused, DNS, TLS, request
    /// build error). `source` captures the full anyhow chain for server-side
    /// logging; clients see a generic message.
    ///
    /// `worker` is the typed `reqwest::Url` so we don't re-stringify a value
    /// that is already a `Url` at the construction site. Rendering goes
    /// through `Display`, which produces the same canonical form as
    /// `Url::as_str()`.
    #[error("upstream unreachable: worker {worker}")]
    UpstreamUnreachable {
        worker: reqwest::Url,
        #[source]
        source: anyhow::Error,
    },

    /// The worker started a response (status + headers received) but failed
    /// to deliver the full body — mid-body socket drop, framing error, etc.
    /// Distinct from `UpstreamUnreachable` (no reply at all) and from a
    /// well-formed non-2xx (which `Proxy` forwards verbatim with the worker's
    /// own body).
    ///
    /// `worker` names the specific worker that started but failed to complete
    /// the response, surfaced on the wire via `Server-Timing: engine.worker`
    /// so a fronting gateway can attribute the mid-body drop to a specific
    /// downstream worker in a multi-worker pool.
    #[error("upstream returned status {status} from worker {worker}")]
    UpstreamStatus {
        status: StatusCode,
        worker: reqwest::Url,
    },

    /// Wall-clock timeout exceeded while waiting for the upstream worker's
    /// response (per-request `request_timeout`).
    ///
    /// `worker` is the typed `reqwest::Url` for the same reason as
    /// `UpstreamUnreachable`.
    #[error("upstream timed out: worker {worker}")]
    UpstreamTimeout { worker: reqwest::Url },

    /// No healthy worker is available for `model`: either none were ever
    /// registered, or every candidate's circuit breaker is open.  Clients
    /// should retry; operators should check discovery + worker health.
    #[error("no healthy workers for model {model}")]
    NoHealthyWorkers { model: String },

    /// PD-mode deployment whose prefill pool has zero healthy workers.
    /// Distinct from `NoHealthyWorkers` because the decode pool may
    /// still be healthy — the failure is pool-specific, and surfacing
    /// the distinct code lets operators alert on prefill-fleet outages
    /// independently of full-model outages.
    #[error("no prefill workers available for model {model}")]
    NoPrefillWorkersAvailable { model: String },

    /// PD-mode deployment whose decode pool has zero healthy workers.
    /// Mirror of [`Self::NoPrefillWorkersAvailable`].
    #[error("no decode workers available for model {model}")]
    NoDecodeWorkersAvailable { model: String },

    /// A request whose lifetime exceeded `stale_request_timeout` — the
    /// active-load janitor force-expired the in-flight bookkeeping
    /// AND fired the per-request cancellation token, which the chat
    /// handler `select!`-races against the upstream fetch.  When the
    /// token wins, the handler returns this variant → HTTP 504 →
    /// client sees `stale_request_expired`.
    ///
    /// Classed as [`ErrorClass::Timeout`] (→ 504), the same class as
    /// `UpstreamTimeout`: from the client's perspective both are a router-side
    /// gateway timeout. Here the upstream worker is still potentially fine — the
    /// router gave up because the per-request budget elapsed. The shared class
    /// is what keeps the two timeouts on the same status; they stay tellable
    /// apart only by `x-router-error-code`.
    /// `worker` names the worker that held the stalled in-flight request when
    /// the janitor fired, surfaced on the wire via `Server-Timing:
    /// engine.worker` so a fronting gateway can attribute the stale-cancel to
    /// a specific downstream worker.
    #[error("stale request expired for model {model} on worker {worker}")]
    StaleRequestExpired {
        model: String,
        worker: reqwest::Url,
    },

    /// A single dispatch attempt exceeded the per-attempt response deadline
    /// (`retry.attempt_deadline_ms`) before the worker produced a response — a
    /// slow/wedged worker. Distinct from `StaleRequestExpired` (whole-request
    /// budget) and `UpstreamTimeout` (the proxy's per-request timeout): this is
    /// the shorter, retry-scoped deadline whose purpose is to fail over off a
    /// slow worker. Retryable (see [`Self::is_retryable_upstream`]); classed as
    /// [`ErrorClass::Timeout`].
    #[error("attempt deadline exceeded for model {model}")]
    AttemptTimeout { model: String },

    /// The per-model policy returned `None` despite the candidate set
    /// being non-empty.  Almost always a router bug or an unsupported
    /// policy state; surfaced as 503 (not 500) so retry-on-failure clients
    /// can drain through a rotation rather than fail-fast on internal_error.
    #[error("policy selected no worker for model {model}")]
    PolicySelectionFailed { model: String },

    /// The worker's circuit breaker was open at the moment of dispatch.
    /// Surfaced post-policy-selection (race with `healthy_workers_for`);
    /// the next selection will skip this worker.
    #[error("worker circuit breaker open: {worker}")]
    BreakerOpen { worker: String },

    /// The worker URL emitted by discovery failed to parse.  Always a
    /// config / discovery-backend bug, not a transient infra issue — but
    /// from the client's perspective the worker is unreachable, so 503.
    /// The forwarder trips the circuit breaker before returning so the
    /// malformed worker drops out of subsequent selection.
    #[error("worker misconfigured: {worker}")]
    WorkerMisconfigured {
        worker: String,
        #[source]
        source: anyhow::Error,
    },

    /// Every eligible worker is at its in-flight cap and the admission wait
    /// queue is full, so the router sheds this request instead of piling more
    /// onto saturated workers. Surfaced as 503 (retryable) so clients / the
    /// upstream load balancer back off rather than fail-fast.
    #[error("service overloaded for model {model}")]
    ServiceOverloaded { model: String },

    #[error("internal: {0}")]
    Internal(#[from] anyhow::Error),
}

impl ApiError {
    /// The failure class — the sole determinant of the HTTP status. Grouping by
    /// class is what makes the status non-divergent: every timeout is
    /// [`ErrorClass::Timeout`], so a per-request timeout and a stale-deadline
    /// cancel are guaranteed the same status, and no future variant can quietly
    /// pick a different one.
    fn class(&self) -> ErrorClass {
        match self {
            ApiError::BadRequest(_) => ErrorClass::BadRequest,
            ApiError::ModelNotFound(_) => ErrorClass::NotFound,
            ApiError::UpstreamUnreachable { .. } => ErrorClass::Upstream,
            ApiError::UpstreamStatus { .. } => ErrorClass::Upstream,
            ApiError::UpstreamTimeout { .. } => ErrorClass::Timeout,
            ApiError::NoHealthyWorkers { .. } => ErrorClass::NoTarget,
            ApiError::NoPrefillWorkersAvailable { .. } => ErrorClass::NoTarget,
            ApiError::NoDecodeWorkersAvailable { .. } => ErrorClass::NoTarget,
            ApiError::StaleRequestExpired { .. } => ErrorClass::Timeout,
            ApiError::AttemptTimeout { .. } => ErrorClass::Timeout,
            ApiError::PolicySelectionFailed { .. } => ErrorClass::NoTarget,
            ApiError::BreakerOpen { .. } => ErrorClass::NoTarget,
            ApiError::WorkerMisconfigured { .. } => ErrorClass::NoTarget,
            ApiError::ServiceOverloaded { .. } => ErrorClass::NoTarget,
            ApiError::Internal(_) => ErrorClass::Internal,
        }
    }

    /// Stable, machine-readable `x-router-error-code` — the authoritative signal
    /// a gateway converts on. Distinct per condition even when two conditions
    /// share a class (and thus a status): `upstream_timeout` and
    /// `stale_request_expired` both map to 504 but stay tellable apart here.
    fn error_code(&self) -> &'static str {
        match self {
            ApiError::BadRequest(_) => "bad_request",
            ApiError::ModelNotFound(_) => "model_not_found",
            ApiError::UpstreamUnreachable { .. } => "upstream_unreachable",
            // Mid-body drop: headers (incl. a status) arrived, then the body
            // didn't. We surface a 502 but echo the worker's status in
            // `x-router-upstream-status` (see `into_response`).
            ApiError::UpstreamStatus { .. } => "upstream_body_incomplete",
            ApiError::UpstreamTimeout { .. } => "upstream_timeout",
            ApiError::NoHealthyWorkers { .. } => "no_healthy_workers",
            ApiError::NoPrefillWorkersAvailable { .. } => "no_prefill_workers_available",
            ApiError::NoDecodeWorkersAvailable { .. } => "no_decode_workers_available",
            ApiError::StaleRequestExpired { .. } => "stale_request_expired",
            ApiError::AttemptTimeout { .. } => "attempt_timeout",
            ApiError::PolicySelectionFailed { .. } => "policy_selection_failed",
            ApiError::BreakerOpen { .. } => "breaker_open",
            ApiError::WorkerMisconfigured { .. } => "worker_misconfigured",
            ApiError::ServiceOverloaded { .. } => "service_overloaded",
            ApiError::Internal(_) => "internal_error",
        }
    }

    /// The HTTP status this error maps to — same value the client receives via
    /// `into_response`. Exposed so the access log records the real status
    /// (e.g. 502/503/504) instead of a sentinel.
    pub fn status_code(&self) -> StatusCode {
        self.class().status()
    }

    /// Coarse attribution of WHERE in the request lifecycle an error
    /// originated — a wire contract read by a downstream hop (see
    /// [`Stage::server_timing`]'s `Server-Timing: router.stage;desc=<stage>`)
    /// to tell a router-owned stall apart from an engine-owned one without a
    /// join back to the engine's own timing data. Three values:
    ///
    /// * [`Ingress`](Stage::Ingress) — died before worker selection ever
    ///   started ([`BadRequest`](ApiError::BadRequest),
    ///   [`ModelNotFound`](ApiError::ModelNotFound)), or the catch-all
    ///   [`Internal`](ApiError::Internal): that variant is in practice
    ///   raisable at any point (including well after selection — see e.g. the
    ///   worker-URL joins in the forwarders), but its stage is never knowable
    ///   from the variant alone, so it defaults here. `stage=ingress` paired
    ///   with `internal_error` reads as "unattributed", not "pre-admission".
    /// * [`Queue`](Stage::Queue) — the router owns the stall: either it died
    ///   during admission / worker selection itself
    ///   ([`NoHealthyWorkers`](ApiError::NoHealthyWorkers),
    ///   [`NoPrefillWorkersAvailable`](ApiError::NoPrefillWorkersAvailable),
    ///   [`NoDecodeWorkersAvailable`](ApiError::NoDecodeWorkersAvailable),
    ///   [`PolicySelectionFailed`](ApiError::PolicySelectionFailed),
    ///   [`BreakerOpen`](ApiError::BreakerOpen),
    ///   [`ServiceOverloaded`](ApiError::ServiceOverloaded)), or it surfaces
    ///   after selection but no byte ever reached a worker and the fault is
    ///   router/discovery-owned, not the engine's
    ///   ([`WorkerMisconfigured`](ApiError::WorkerMisconfigured) — the
    ///   forwarder trips the breaker before returning, so this is a
    ///   discovery-config defect, not an engine stall).
    /// * [`Dispatch`](Stage::Dispatch) — died waiting on (or talking to) an
    ///   already-selected worker; the engine owns the stall
    ///   ([`UpstreamUnreachable`](ApiError::UpstreamUnreachable),
    ///   [`UpstreamStatus`](ApiError::UpstreamStatus),
    ///   [`UpstreamTimeout`](ApiError::UpstreamTimeout),
    ///   [`AttemptTimeout`](ApiError::AttemptTimeout),
    ///   [`StaleRequestExpired`](ApiError::StaleRequestExpired) — the
    ///   stale-cancel token only exists once admission has already granted a
    ///   slot, so this one is `dispatch` by construction even though the
    ///   janitor is a router-side actor).
    ///
    /// Exhaustive (wildcard-free) for the same reason as `error_code` and
    /// `class`: a future variant is forced to pick a stage rather than
    /// silently inheriting one via a catch-all arm.
    fn stage(&self) -> Stage {
        match self {
            ApiError::BadRequest(_) | ApiError::ModelNotFound(_) | ApiError::Internal(_) => {
                Stage::Ingress
            }
            ApiError::NoHealthyWorkers { .. }
            | ApiError::NoPrefillWorkersAvailable { .. }
            | ApiError::NoDecodeWorkersAvailable { .. }
            | ApiError::PolicySelectionFailed { .. }
            | ApiError::BreakerOpen { .. }
            | ApiError::WorkerMisconfigured { .. }
            | ApiError::ServiceOverloaded { .. } => Stage::Queue,
            ApiError::UpstreamUnreachable { .. }
            | ApiError::UpstreamStatus { .. }
            | ApiError::UpstreamTimeout { .. }
            | ApiError::AttemptTimeout { .. }
            | ApiError::StaleRequestExpired { .. } => Stage::Dispatch,
        }
    }

    /// Whether this is a transient *dispatch* failure the router may recover by
    /// re-dispatching the SAME request to a DIFFERENT worker — the failover the
    /// plain-mode retry loop performs (see [`crate::config::RetryConfig`]).
    ///
    /// Retryable exactly when the selected worker never returned a usable
    /// response AND no bytes reached the client, so a re-dispatch neither
    /// double-serves the client nor is known to double-execute on the engine:
    ///
    /// * [`UpstreamUnreachable`](Self::UpstreamUnreachable) — connect refused /
    ///   DNS / TLS: the request never landed on a working engine.
    /// * [`UpstreamTimeout`](Self::UpstreamTimeout) — no response within the
    ///   per-request budget; the failed attempt's abort guard tells that engine
    ///   to stop before we try elsewhere.
    /// * [`BreakerOpen`](Self::BreakerOpen) — the worker's breaker was open at
    ///   dispatch (a race with `healthy_workers_for`); a different worker is
    ///   almost certainly eligible.
    /// * [`WorkerMisconfigured`](Self::WorkerMisconfigured) — the worker URL
    ///   failed to parse; its breaker was tripped, so reselection skips it.
    ///
    /// Deliberately NOT retryable:
    /// * [`UpstreamStatus`](Self::UpstreamStatus) — produced only by the
    ///   buffered (non-streaming) forward when the worker returned headers and
    ///   then dropped mid-body. Nothing reached the client, but the engine
    ///   received and by then has likely fully executed the request —
    ///   re-dispatch would double-execute work an abort can no longer stop.
    ///   (Streaming mid-body drops never reach this predicate: they occur
    ///   after `Ok`, inside the SSE pump.)
    /// * The `NoTarget` selection errors ([`NoHealthyWorkers`](Self::NoHealthyWorkers),
    ///   [`PolicySelectionFailed`](Self::PolicySelectionFailed),
    ///   [`ServiceOverloaded`](Self::ServiceOverloaded), the PD-pool variants) —
    ///   these come from admission, not from a worker; there is nothing better
    ///   to retry onto in the same instant.
    /// * [`StaleRequestExpired`](Self::StaleRequestExpired) — the request budget
    ///   is already spent; retrying would only exceed it further.
    /// * [`BadRequest`](Self::BadRequest) / [`ModelNotFound`](Self::ModelNotFound)
    ///   / [`Internal`](Self::Internal) — terminal; a retry cannot change the
    ///   outcome.
    ///
    /// Exhaustive (wildcard-free) so a future variant is forced to decide its
    /// retryability rather than silently defaulting to "not retryable".
    pub fn is_retryable_upstream(&self) -> bool {
        match self {
            ApiError::UpstreamUnreachable { .. }
            | ApiError::UpstreamTimeout { .. }
            | ApiError::AttemptTimeout { .. }
            | ApiError::BreakerOpen { .. }
            | ApiError::WorkerMisconfigured { .. } => true,
            ApiError::UpstreamStatus { .. }
            | ApiError::BadRequest(_)
            | ApiError::ModelNotFound(_)
            | ApiError::NoHealthyWorkers { .. }
            | ApiError::NoPrefillWorkersAvailable { .. }
            | ApiError::NoDecodeWorkersAvailable { .. }
            | ApiError::StaleRequestExpired { .. }
            | ApiError::PolicySelectionFailed { .. }
            | ApiError::ServiceOverloaded { .. }
            | ApiError::Internal(_) => false,
        }
    }

    /// The worker's *own* status to echo in `x-router-upstream-status`, for the
    /// case where the router synthesized its own status over a worker that did
    /// respond. Today only the mid-body-drop (`UpstreamStatus`) carries one. This
    /// is an exhaustive, wildcard-free match (not an `if let` at the call site) so
    /// a future "synthesized over a responding worker" variant is forced to decide
    /// whether it echoes a status, rather than silently inheriting `None`.
    fn upstream_status(&self) -> Option<StatusCode> {
        match self {
            ApiError::UpstreamStatus { status, .. } => Some(*status),
            ApiError::BadRequest(_)
            | ApiError::ModelNotFound(_)
            | ApiError::UpstreamUnreachable { .. }
            | ApiError::UpstreamTimeout { .. }
            | ApiError::NoHealthyWorkers { .. }
            | ApiError::NoPrefillWorkersAvailable { .. }
            | ApiError::NoDecodeWorkersAvailable { .. }
            | ApiError::StaleRequestExpired { .. }
            | ApiError::AttemptTimeout { .. }
            | ApiError::PolicySelectionFailed { .. }
            | ApiError::BreakerOpen { .. }
            | ApiError::WorkerMisconfigured { .. }
            | ApiError::ServiceOverloaded { .. }
            | ApiError::Internal(_) => None,
        }
    }
}

#[derive(Serialize)]
struct ErrorEnvelope<'a> {
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    #[serde(rename = "type")]
    typ: &'static str,
    code: &'a str,
    message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.class().status();
        let code = self.error_code();
        let typ = match status.as_u16() {
            400..=499 => "invalid_request_error",
            _ => "server_error",
        };
        // Pick a client-facing message that NEVER leaks worker URLs or raw
        // source chains; full structured details are logged server-side.
        let message = match &self {
            ApiError::Internal(e) => {
                // `{:#}` prints the anyhow chain (top error + sources) — `?e`
                // would only show the outermost message.
                tracing::error!("internal error serving request: {e:#}");
                "internal error".to_string()
            }
            ApiError::UpstreamUnreachable { worker, source } => {
                tracing::warn!(
                    upstream = %worker,
                    error = %format_args!("{source:#}"),
                    "upstream worker unreachable",
                );
                "upstream unavailable".to_string()
            }
            ApiError::UpstreamStatus { status, worker } => {
                tracing::warn!(
                    upstream = %worker,
                    upstream_status = %status,
                    "upstream returned an error status",
                );
                "upstream returned an error status".to_string()
            }
            ApiError::UpstreamTimeout { worker } => {
                tracing::warn!(upstream = %worker, "upstream request timed out");
                "upstream request timed out".to_string()
            }
            ApiError::NoHealthyWorkers { model } => {
                tracing::warn!(model = %model, reason = "no_healthy_workers", "service unavailable");
                "no healthy workers for the requested model".to_string()
            }
            ApiError::NoPrefillWorkersAvailable { model } => {
                tracing::warn!(
                    model = %model,
                    reason = "no_prefill_workers_available",
                    "service unavailable",
                );
                "no prefill workers available for the requested model".to_string()
            }
            ApiError::NoDecodeWorkersAvailable { model } => {
                tracing::warn!(
                    model = %model,
                    reason = "no_decode_workers_available",
                    "service unavailable",
                );
                "no decode workers available for the requested model".to_string()
            }
            ApiError::StaleRequestExpired { model, worker } => {
                tracing::warn!(
                    model = %model,
                    upstream = %worker,
                    reason = "stale_request_expired",
                    "stale-request janitor expired in-flight request",
                );
                "request expired before completion".to_string()
            }
            ApiError::AttemptTimeout { model } => {
                tracing::warn!(
                    model = %model,
                    reason = "attempt_timeout",
                    "dispatch attempt exceeded the per-attempt response deadline",
                );
                "upstream did not respond within the per-attempt deadline".to_string()
            }
            ApiError::PolicySelectionFailed { model } => {
                tracing::warn!(model = %model, reason = "policy_selection_failed", "service unavailable");
                "service unavailable".to_string()
            }
            ApiError::BreakerOpen { worker } => {
                tracing::warn!(upstream = %worker, reason = "breaker_open", "service unavailable");
                "service unavailable".to_string()
            }
            ApiError::WorkerMisconfigured { worker, source } => {
                tracing::error!(
                    upstream = %worker,
                    error = %format_args!("{source:#}"),
                    "worker URL emitted by discovery is malformed",
                );
                "service unavailable".to_string()
            }
            ApiError::ServiceOverloaded { model } => {
                tracing::warn!(model = %model, reason = "service_overloaded", "shedding request: all workers at capacity and wait queue full");
                "service overloaded".to_string()
            }
            ApiError::BadRequest(_) | ApiError::ModelNotFound(_) => self.to_string(),
        };
        let mut resp = (
            status,
            Json(ErrorEnvelope {
                error: ErrorBody { typ, code, message },
            }),
        )
            .into_response();
        resp.headers_mut()
            .insert(X_ROUTER_ERROR_CODE, HeaderValue::from_static(code));
        // Preserve the worker's real status when we synthesized our own (today,
        // only the mid-body-drop case: the worker sent a status, then dropped the
        // body, so we report a 502 but don't throw away what it said).
        if let Some(upstream) = self.upstream_status() {
            resp.headers_mut().insert(
                X_ROUTER_UPSTREAM_STATUS,
                HeaderValue::from(upstream.as_u16()),
            );
        }
        // `router.stage` on `Server-Timing` — see `stage()`'s doc comment for the
        // wire contract. Every `ApiError` reaches this `into_response`, so EVERY
        // router-generated error response gets stamped; a well-formed non-2xx
        // response the proxy forwards with the worker's own status and body
        // never passes through here at all, so it is never stamped either — a
        // worker can't spoof this header even if it tried. The discriminator is
        // ONE-directional though: presence of `router.stage` ⇒ the router
        // synthesized the response; absence proves nothing on its own (a
        // response that bypasses `ApiError` entirely — a panic-caught 500, an
        // extractor-rejection 400, an unmatched-route 404 — is also unstamped).
        // `append`, not `insert` — `Server-Timing` is list-valued, so this must
        // add without clobbering a value a prior layer may have set (mirrors
        // the `router.ttfb` stamping in the chat handler).
        resp.headers_mut()
            .append(SERVER_TIMING, self.stage().server_timing());
        // `engine.worker` on `Server-Timing` — sibling of `router.stage` for the
        // dispatch-stage variants that name a specific downstream worker. Emit
        // it as its own repeated `Server-Timing` line (same discipline as
        // `router.pod` / `router.stage`) so a consumer that joins `Header.Values`
        // and matches segment names can attribute an engine-owned stall to the
        // specific worker in a multi-worker pool. Router-only; the gateway-side
        // reader is staged in radixark PR #911.
        if let Some(hv) = self.engine_worker_server_timing() {
            resp.headers_mut().append(SERVER_TIMING, hv);
        }
        resp
    }
}

impl ApiError {
    /// The pre-composed `Server-Timing` entry naming the downstream worker
    /// this error is attributed to, or `None` if this variant doesn't name a
    /// worker (ingress-stage validation, queue-stage no-target-available,
    /// stage-`Dispatch` variants that fired before a specific worker was
    /// selected). Called from `into_response()` — see the wire-contract
    /// comment there.
    ///
    /// `desc` is the worker's `reqwest::Url` rendered by `Display`, which
    /// produces the canonical `scheme://host:port` string. Falls back to a
    /// static `unknown` entry rather than dropping the header entirely if the
    /// rendered URL somehow isn't a valid `HeaderValue` (URLs always are, but
    /// mirroring the `pod_server_timing()` fallback keeps the invariant that
    /// dispatch-stage errors ALWAYS carry an `engine.worker` line).
    fn engine_worker_server_timing(&self) -> Option<HeaderValue> {
        let worker = match self {
            ApiError::UpstreamTimeout { worker } => worker,
            ApiError::UpstreamUnreachable { worker, .. } => worker,
            ApiError::UpstreamStatus { worker, .. } => worker,
            ApiError::StaleRequestExpired { worker, .. } => worker,
            _ => return None,
        };
        Some(
            HeaderValue::from_str(&format!("engine.worker;desc={worker}")).unwrap_or_else(
                |_| HeaderValue::from_static("engine.worker;desc=unknown"),
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use http_body_util::BodyExt;
    use serde::Deserialize;

    fn collect_body(resp: Response) -> String {
        let bytes = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async { BodyExt::collect(resp.into_body()).await.unwrap().to_bytes() });
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Pin the exact JSON envelope shape that clients see. Renaming any of
    /// these fields (or removing one) breaks every downstream consumer
    /// silently, so we deserialize into a fixed struct rather than
    /// regex-matching the rendered JSON.
    #[derive(Deserialize)]
    struct ErrEnv {
        error: ErrField,
    }

    #[derive(Deserialize)]
    struct ErrField {
        #[serde(rename = "type")]
        typ: String,
        code: String,
        message: String,
    }

    fn parse_envelope(resp: Response) -> (StatusCode, Option<String>, ErrEnv) {
        let status = resp.status();
        let code_header = resp
            .headers()
            .get("x-router-error-code")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let body_str = collect_body(resp);
        let env: ErrEnv = serde_json::from_str(&body_str)
            .unwrap_or_else(|e| panic!("envelope did not match expected shape: {e}: {body_str}"));
        (status, code_header, env)
    }

    #[test]
    fn is_retryable_upstream_covers_transient_dispatch_failures_only() {
        let worker = reqwest::Url::parse("http://w:1/").unwrap();
        // Retryable: no usable response, nothing sent to the client.
        assert!(ApiError::UpstreamUnreachable {
            worker: worker.clone(),
            source: anyhow::anyhow!("connect refused"),
        }
        .is_retryable_upstream());
        assert!(ApiError::UpstreamTimeout {
            worker: worker.clone()
        }
        .is_retryable_upstream());
        assert!(ApiError::BreakerOpen {
            worker: "http://w:1".into()
        }
        .is_retryable_upstream());
        assert!(ApiError::WorkerMisconfigured {
            worker: "http://w:1".into(),
            source: anyhow::anyhow!("bad url"),
        }
        .is_retryable_upstream());

        // Not retryable: bytes may have gone out, terminal, or nothing to retry onto.
        assert!(!ApiError::UpstreamStatus {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            worker: reqwest::Url::parse("http://test-worker/").unwrap(),
        }
        .is_retryable_upstream());
        assert!(!ApiError::ServiceOverloaded { model: "m".into() }.is_retryable_upstream());
        assert!(!ApiError::NoHealthyWorkers { model: "m".into() }.is_retryable_upstream());
        assert!(!ApiError::StaleRequestExpired { model: "m".into(), worker: reqwest::Url::parse("http://test-worker/").unwrap() }.is_retryable_upstream());
        assert!(!ApiError::BadRequest("x".into()).is_retryable_upstream());
        assert!(!ApiError::ModelNotFound("x".into()).is_retryable_upstream());
        assert!(!ApiError::Internal(anyhow::anyhow!("boom")).is_retryable_upstream());
    }

    #[test]
    fn upstream_unreachable_envelope_has_code_and_no_leak() {
        let worker_str = "http://10.0.0.42:30000/";
        let worker = reqwest::Url::parse(worker_str).unwrap();
        let secret = "TLS_HANDSHAKE_FAILED at /etc/secret_ca.pem";
        let err = ApiError::UpstreamUnreachable {
            worker: worker.clone(),
            source: anyhow::anyhow!("{secret}"),
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            resp.headers()
                .get("x-router-error-code")
                .and_then(|v| v.to_str().ok()),
            Some("upstream_unreachable"),
        );
        let body = collect_body(resp);
        assert!(body.contains("\"code\":\"upstream_unreachable\""), "{body}");
        assert!(body.contains("\"type\":\"server_error\""), "{body}");
        assert!(
            !body.contains(worker_str) && !body.contains(secret),
            "client body must NOT leak worker URL or reqwest source chain; got: {body}",
        );
    }

    #[test]
    fn upstream_body_incomplete_preserves_worker_status_in_header() {
        // Mid-body drop: the worker sent a status (here 500) then dropped the
        // body. The router synthesizes its own 502, but the worker's real status
        // is preserved in `x-router-upstream-status` rather than silently lost.
        let err = ApiError::UpstreamStatus {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            worker: reqwest::Url::parse("http://test-worker/").unwrap(),
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            resp.headers()
                .get("x-router-error-code")
                .and_then(|v| v.to_str().ok()),
            Some("upstream_body_incomplete"),
        );
        assert_eq!(
            resp.headers()
                .get("x-router-upstream-status")
                .and_then(|v| v.to_str().ok()),
            Some("500"),
            "the worker's real status must be preserved, not discarded",
        );
        let body = collect_body(resp);
        assert!(
            body.contains("\"code\":\"upstream_body_incomplete\""),
            "{body}"
        );
    }

    #[test]
    fn upstream_timeout_envelope_has_code_and_no_leak() {
        let worker_str = "http://10.0.0.42:30000/";
        let worker = reqwest::Url::parse(worker_str).unwrap();
        let err = ApiError::UpstreamTimeout {
            worker: worker.clone(),
        };
        let resp = err.into_response();
        // A timeout is a gateway timeout (504), not a bad gateway (502): same
        // class — and same status — as the stale-deadline cancel, so the two
        // can't drift apart.
        assert_eq!(resp.status(), StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(
            resp.headers()
                .get("x-router-error-code")
                .and_then(|v| v.to_str().ok()),
            Some("upstream_timeout"),
        );
        let body = collect_body(resp);
        assert!(body.contains("\"code\":\"upstream_timeout\""), "{body}");
        assert!(
            !body.contains(worker_str),
            "client body must NOT leak worker URL; got: {body}",
        );
    }

    #[test]
    fn bad_request_envelope_has_expected_shape() {
        let msg = "invalid_request: body must be an object";
        let err = ApiError::BadRequest(msg.into());
        let resp = err.into_response();
        let (status, code_header, env) = parse_envelope(resp);

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(code_header.as_deref(), Some("bad_request"));
        assert_eq!(env.error.code, "bad_request");
        assert_eq!(env.error.typ, "invalid_request_error");
        assert!(
            !env.error.message.is_empty(),
            "message must not be empty: {:?}",
            env.error.message,
        );
        assert_ne!(env.error.code, "internal_error");
        assert_ne!(env.error.code, "model_not_found");
    }

    #[test]
    fn service_overloaded_maps_to_503_with_code() {
        let err = ApiError::ServiceOverloaded { model: "m".into() };
        let resp = err.into_response();
        let (status, code_header, env) = parse_envelope(resp);

        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(code_header.as_deref(), Some("service_overloaded"));
        assert_eq!(env.error.code, "service_overloaded");
        assert_eq!(env.error.typ, "server_error");
    }

    #[test]
    fn model_not_found_envelope_has_expected_shape() {
        let err = ApiError::ModelNotFound("ghost-7b".into());
        let resp = err.into_response();
        let (status, code_header, env) = parse_envelope(resp);

        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(code_header.as_deref(), Some("model_not_found"));
        assert_eq!(env.error.code, "model_not_found");
        assert_eq!(env.error.typ, "invalid_request_error");
        assert!(
            !env.error.message.is_empty(),
            "message must not be empty: {:?}",
            env.error.message,
        );
        assert_ne!(env.error.code, "internal_error");
        assert_ne!(env.error.code, "bad_request");
    }

    #[test]
    fn internal_error_response_sanitizes_anyhow_chain() {
        let secret_msg = "internal /opt/secret/credential.json missing";
        let err = ApiError::Internal(anyhow::anyhow!("{secret_msg}"));
        let resp = err.into_response();
        let body_str = collect_body(resp);
        // Generic to client:
        assert!(
            body_str.contains("\"code\":\"internal_error\""),
            "body: {body_str}"
        );
        assert!(
            body_str.contains("\"type\":\"server_error\""),
            "body: {body_str}"
        );
        // No leak of the original anyhow message:
        assert!(
            !body_str.contains(secret_msg),
            "ApiError::Internal must not leak anyhow chain to client; got: {body_str}"
        );
    }

    /// The four client-facing signals of the router status-code contract:
    /// the HTTP status, `x-router-error-code`, `x-router-upstream-status`, and
    /// the `router.stage` entry on `Server-Timing`. `Server-Timing` is
    /// list-valued (appended, not inserted), but every router-originated
    /// error appends exactly one entry, so a single `get` (first value) is
    /// enough here — a genuine multi-value case would need `get_all`.
    fn signals(err: ApiError) -> (StatusCode, Option<String>, Option<String>, Option<String>) {
        let resp = err.into_response();
        let status = resp.status();
        let header = |name: &str| {
            resp.headers()
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned)
        };
        (
            status,
            header("x-router-error-code"),
            header("x-router-upstream-status"),
            header("server-timing"),
        )
    }

    /// One table-test row: label, error, expected status, expected
    /// x-router-error-code, expected x-router-upstream-status, expected
    /// router.stage.
    type StageCase = (
        &'static str,
        ApiError,
        StatusCode,
        &'static str,
        Option<&'static str>,
        &'static str,
    );

    /// Every router-*originated* condition maps to a fixed (status, error-code)
    /// pair, and ONLY the mid-body-drop case echoes the worker's real status in
    /// `x-router-upstream-status`. This pins the router half of the status-code
    /// contract:
    ///   * same condition class → same status — both timeouts are 504, so the
    ///     per-request timeout and the stale-deadline cancel can never diverge
    ///     to 502-vs-504 the way they used to;
    ///   * a worker's real status is preserved, never silently rewritten.
    #[test]
    fn router_originated_scenarios_match_status_and_headers() {
        let worker = reqwest::Url::parse("http://host:1/").unwrap();
        let cases: Vec<StageCase> = vec![
            (
                "mid-body drop preserves worker status",
                ApiError::UpstreamStatus {
                    status: StatusCode::OK,
                    worker: worker.clone(),
                },
                StatusCode::BAD_GATEWAY,
                "upstream_body_incomplete",
                Some("200"),
                "dispatch",
            ),
            (
                "unreachable",
                ApiError::UpstreamUnreachable {
                    worker: worker.clone(),
                    source: anyhow::anyhow!("connect refused"),
                },
                StatusCode::BAD_GATEWAY,
                "upstream_unreachable",
                None,
                "dispatch",
            ),
            (
                "request timeout",
                ApiError::UpstreamTimeout {
                    worker: worker.clone(),
                },
                StatusCode::GATEWAY_TIMEOUT,
                "upstream_timeout",
                None,
                "dispatch",
            ),
            (
                "attempt deadline exceeded",
                ApiError::AttemptTimeout { model: "m".into() },
                StatusCode::GATEWAY_TIMEOUT,
                "attempt_timeout",
                None,
                "dispatch",
            ),
            (
                "stale-deadline cancel",
                ApiError::StaleRequestExpired { model: "m".into(), worker: worker.clone() },
                StatusCode::GATEWAY_TIMEOUT,
                "stale_request_expired",
                None,
                "dispatch",
            ),
            (
                "admission shed",
                ApiError::ServiceOverloaded { model: "m".into() },
                StatusCode::SERVICE_UNAVAILABLE,
                "service_overloaded",
                None,
                "queue",
            ),
            (
                "no healthy workers",
                ApiError::NoHealthyWorkers { model: "m".into() },
                StatusCode::SERVICE_UNAVAILABLE,
                "no_healthy_workers",
                None,
                "queue",
            ),
            (
                "bad request",
                ApiError::BadRequest("bad".into()),
                StatusCode::BAD_REQUEST,
                "bad_request",
                None,
                "ingress",
            ),
            (
                "model not found",
                ApiError::ModelNotFound("ghost".into()),
                StatusCode::NOT_FOUND,
                "model_not_found",
                None,
                "ingress",
            ),
            (
                "no prefill workers",
                ApiError::NoPrefillWorkersAvailable { model: "m".into() },
                StatusCode::SERVICE_UNAVAILABLE,
                "no_prefill_workers_available",
                None,
                "queue",
            ),
            (
                "no decode workers",
                ApiError::NoDecodeWorkersAvailable { model: "m".into() },
                StatusCode::SERVICE_UNAVAILABLE,
                "no_decode_workers_available",
                None,
                "queue",
            ),
            (
                "policy selection failed",
                ApiError::PolicySelectionFailed { model: "m".into() },
                StatusCode::SERVICE_UNAVAILABLE,
                "policy_selection_failed",
                None,
                "queue",
            ),
            (
                "breaker open",
                ApiError::BreakerOpen {
                    worker: "http://w:1".into(),
                },
                StatusCode::SERVICE_UNAVAILABLE,
                "breaker_open",
                None,
                "queue",
            ),
            (
                "worker misconfigured",
                ApiError::WorkerMisconfigured {
                    worker: "http://w:1".into(),
                    source: anyhow::anyhow!("unparsable url"),
                },
                StatusCode::SERVICE_UNAVAILABLE,
                "worker_misconfigured",
                None,
                "queue",
            ),
            (
                "internal",
                ApiError::Internal(anyhow::anyhow!("boom")),
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                None,
                "ingress",
            ),
        ];
        for (label, err, want_status, want_code, want_upstream, want_stage) in cases {
            let (status, code, upstream, server_timing) = signals(err);
            assert_eq!(status, want_status, "status mismatch for `{label}`");
            assert_eq!(
                code.as_deref(),
                Some(want_code),
                "x-router-error-code mismatch for `{label}`",
            );
            assert_eq!(
                upstream.as_deref(),
                want_upstream,
                "x-router-upstream-status mismatch for `{label}`",
            );
            assert_eq!(
                server_timing.as_deref(),
                Some(format!("router.stage;desc={want_stage}").as_str()),
                "router.stage mismatch for `{label}`",
            );
        }
    }

    /// Dispatch-stage errors that name a specific downstream worker must
    /// emit BOTH `router.stage;desc=dispatch` and
    /// `engine.worker;desc=<worker_url>` as SEPARATE repeated `Server-Timing`
    /// lines (not combined into one comma-joined value). The
    /// `HeaderMap::get_all` view returns each `append` as its own entry so a
    /// consumer that joins `Header.Values` and matches segment names reads
    /// both without one clobbering the other — same discipline as
    /// `router.pod` (`app.rs`).
    #[test]
    fn dispatch_stage_error_carries_engine_worker_and_router_stage_together() {
        use axum::response::IntoResponse;
        let worker = reqwest::Url::parse("http://10.4.2.7:30000/").unwrap();
        let err = ApiError::UpstreamTimeout {
            worker: worker.clone(),
        };
        let resp = err.into_response();
        let values: Vec<String> = resp
            .headers()
            .get_all("server-timing")
            .iter()
            .filter_map(|v| v.to_str().ok().map(str::to_owned))
            .collect();
        assert!(
            values
                .iter()
                .any(|v| v == "router.stage;desc=dispatch"),
            "router.stage;desc=dispatch missing from Server-Timing values: {values:?}",
        );
        assert!(
            values
                .iter()
                .any(|v| v == "engine.worker;desc=http://10.4.2.7:30000/"),
            "engine.worker;desc=<worker> missing from Server-Timing values: {values:?}",
        );
    }

    /// Non-dispatch errors — ingress-stage validation, queue-stage
    /// no-workers, etc. — don't name a specific worker, so they emit
    /// `router.stage` but must NOT emit `engine.worker`. The `_ => None` arm
    /// in `engine_worker_server_timing` enforces this via exhaustive match;
    /// this test locks it in behaviorally so a future variant that
    /// legitimately shouldn't carry a worker doesn't accidentally start
    /// emitting an empty `engine.worker;desc=` entry.
    #[test]
    fn non_dispatch_stage_error_omits_engine_worker() {
        use axum::response::IntoResponse;
        let err = ApiError::BadRequest("malformed prompt".into());
        let resp = err.into_response();
        let has_engine_worker = resp
            .headers()
            .get_all("server-timing")
            .iter()
            .filter_map(|v| v.to_str().ok())
            .any(|v| v.starts_with("engine.worker;"));
        assert!(
            !has_engine_worker,
            "engine.worker unexpectedly present on a non-dispatch-stage error",
        );
    }
}
