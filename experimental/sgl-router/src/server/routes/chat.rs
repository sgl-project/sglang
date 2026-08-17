// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::{RetryConfig, DEFAULT_RETRY_ITL_REL_FACTOR};
use crate::discovery::{ModelId, WorkerMode};
use crate::policies::registry::{PdPoolResolver, PdResolveError};
use crate::policies::{request_tokens_for, RequestTokens, SelectionContext};
use crate::proxy::sse::{StreamCapture, StreamEnd};
use crate::proxy::AbortReason;
use crate::server::app::{RequestPhase, RequestPhaseCell};
use crate::server::app_context::AppContext;
use crate::server::cache_sim_extend::{self, ReplySource};
use crate::server::error::ApiError;
use crate::server::header_utils::SERVER_TIMING;
use crate::server::metrics::{
    MetricsRegistry, RequestLogContext, StaleRequestOutcome, StreamOutcome, WorkerModeLabel,
};

/// Parse a discovery-emitted worker URL into `reqwest::Url` for embedding in
/// dispatch-stage error variants (`StaleRequestExpired` / `UpstreamStatus` /
/// ...) so the `Server-Timing: engine.worker` stamp on the error response can
/// name a canonical URL. Discovery-emitted URLs are always valid at this
/// point (a malformed URL trips the breaker upstream in `parse_worker_url`
/// and never reaches here), but we soft-fail to an `http://unknown/`
/// placeholder rather than panic — the `Server-Timing` fallback in
/// `engine_worker_server_timing()` renders the placeholder as
/// `engine.worker;desc=http://unknown/` which is still parseable.
fn worker_url_for_error(url: &str) -> reqwest::Url {
    url.parse::<reqwest::Url>()
        .unwrap_or_else(|_| reqwest::Url::parse("http://unknown/").expect("static URL parses"))
}

/// Narrow the pre-drop abort reason on the pre-headers / unary guard from
/// its constructor default (`HandlerCancelled`, the catch-all for "handler
/// future dropped mid-await") to the specific `ApiError` variant that caused
/// the fetch to resolve as `Err`. Called at the one site where we know both
/// the guard and the settled `Result` — right after the `tokio::select!` in
/// the plain-mode streaming and plain-mode unary arms.
///
/// `Ok(_)` means the engine responded (any status): responsibility passes
/// to either the streaming pump's internal guard (2xx) or nothing at all
/// (non-2xx, engine's own error body). Nothing to record here.
fn abort_reason_from_api_error(err: &ApiError) -> AbortReason {
    match err {
        ApiError::UpstreamTimeout { .. } => AbortReason::UpstreamTimeout,
        // Deliberately NOT folded into UpstreamTimeout: the abort label is the
        // only fleet-wide signal that separates "our budget ran out" from "the
        // network path broke", and those need opposite remedies.
        ApiError::UpstreamSocketTimeout { .. } => AbortReason::UpstreamSocketTimeout,
        // The per-attempt deadline elapsed on a slow/wedged worker — a
        // timeout in the same family as UpstreamTimeout (the router gave up
        // on THIS worker), so it shares that label rather than falling
        // through to the transport-error catch-all.
        ApiError::AttemptTimeout { .. } => AbortReason::UpstreamTimeout,
        ApiError::StaleRequestExpired { .. } => AbortReason::StaleRequestExpired,
        // Everything else is a router-side transport / configuration failure.
        // The abort still fires (the engine may have started work), but its
        // origin is on our side, not a client cancel or timeout.
        _ => AbortReason::TransportError,
    }
}
use crate::workers::Worker;
use axum::body::Body;
use axum::extract::{Extension, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response};
use axum::response::IntoResponse;
use bytes::Bytes;
use serde::de::IgnoredAny;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Sampling counter for the diagnostic `phase_*` timing logs below. Logs roughly
/// 1-in-`PHASE_LOG_SAMPLE` requests so a steady flood doesn't drown the access
/// log while still yielding a representative latency-phase breakdown.
static PHASE_LOG_COUNTER: AtomicU64 = AtomicU64::new(0);
const PHASE_LOG_SAMPLE: u64 = 64;

/// Observability header carrying the decode-pool URL selected via host
/// affinity for a PD-disaggregated request. The router fans the
/// bootstrap-injected request body to BOTH the prefill and the decode
/// worker concurrently; this header lets the prefill log the chosen
/// peer, and is mirrored onto the response so sidecars / tests can
/// observe affinity without sniffing the proxy hop. The `x-sgl-`
/// prefix matches `x-sgl-router-error-code` so router-emitted metadata
/// stays grouped.
const X_SGL_DECODE_URL: HeaderName = HeaderName::from_static("x-sgl-decode-url");

/// Coarse char-count → token-count divisor used to estimate prefill load
/// from the request body when no real tokenizer count is available. Four
/// bytes per token is the standard SGLang upstream estimate; it
/// overcounts ASCII and undercounts CJK but stays within an order of
/// magnitude of the real token count, which is plenty for load
/// scoring. The active-load counters' role is relative ordering across
/// workers — not absolute accuracy — so the estimate is fit for
/// purpose.
const CHARS_PER_TOKEN_ESTIMATE: usize = 4;

/// Per-route body-size cap on `/v1/chat/completions`. 100 MiB accommodates a
/// long text context AND multimodal requests, whose base64-encoded image or
/// audio payloads dwarf any text body — a handful of high-resolution images
/// alone runs to several MiB — while still bounding the heap a hostile client
/// can force the router to allocate before forwarding. The cap is wired in
/// `crate::server::app::build_router` as a route-level `DefaultBodyLimit`
/// layer; axum's `Bytes` extractor enforces it and returns 413
/// PAYLOAD_TOO_LARGE before this handler runs.
pub const MAX_CHAT_BODY_BYTES: usize = 100 << 20;

/// Minimal probe over the request body — we only need the `stream` field,
/// the `model` field, and a client-supplied `rid` to decide between buffered
/// vs SSE forwarding, select a worker, and know whether to reuse the
/// client's abort-by-rid identifier. Deserializing into this struct (vs
/// `serde_json::Value`) does two things:
///
/// 1. Avoids the per-field heap allocation of `Value` for a multi-MiB body.
/// 2. Pins the contract: the body MUST be a JSON object. Degenerate
///    shapes (`null`, `[]`, `"hi"`) fail at this step rather than being
///    silently forwarded with `stream=false`.
///
/// All other fields are ignored — the worker is authoritative for the
/// full request schema.
///
/// `rid` is probed here — not via `request_value` — because `request_value`
/// is only populated when `want_tokens` is true (a tokenization-offload
/// decision unrelated to whether the client passed a `rid`). Gating the
/// client-rid reuse on `want_tokens` would silently drop it for any
/// model/policy that doesn't need ingress tokenization, undermining the
/// "reuse, don't override" contract on the common path.
#[derive(Debug, Deserialize)]
struct RequestProbe {
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    rid: Option<String>,
    /// Probed as raw `Value`s, not `u64`: a mistyped value (float, string,
    /// negative) must not fail the probe's deserialize — the engine is
    /// authoritative for schema errors and returns the better message.
    /// `null` deserializes to `None` (same as absent), matching the
    /// engine's treatment of an explicit `"max_tokens": null`.
    #[serde(default)]
    max_tokens: Option<serde_json::Value>,
    #[serde(default)]
    max_completion_tokens: Option<serde_json::Value>,
}

/// RAII guard that records `sgl_router_request_duration_seconds` when
/// dropped. For streaming requests the handler returns at response-headers
/// time, so recording end-to-end latency at the dispatch site would capture
/// only time-to-headers (≈ TTFT). Instead this guard is packed into the SSE
/// pump's `stream_guards`, so it drops — and records — when the stream
/// completes (or the client disconnects), yielding true end-to-end latency.
/// Non-streaming requests record at the dispatch site directly (the body is
/// already buffered there) and do not use this guard.
struct RecordDurationOnDrop {
    metrics: Arc<MetricsRegistry>,
    model: String,
    start: std::time::Instant,
}

impl Drop for RecordDurationOnDrop {
    fn drop(&mut self) {
        self.metrics
            .observe_request_duration(&self.model, self.start.elapsed().as_secs_f64());
    }
}

/// POST /v1/chat/completions handler. Thin delegator to
/// [`chat_completions_inner`]. Per-request logging and `requests_total` /
/// `responses_total` counting happen once, centrally, in the outermost
/// `access_log_and_record` middleware (see [`crate::server::app`]) — including
/// the early `?` short-circuits this returns as `Err` (a body-validation 400,
/// an admission 503 shed, model-not-found), which the middleware records as a
/// pre-routing rejection. Routed requests carry a
/// [`RequestLogContext`](crate::server::metrics::RequestLogContext) so the
/// middleware can attach their per-worker labels.
pub(crate) async fn chat_completions(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    // `Option<...>` so handler-only tests that exercise this function without
    // the `access_log_and_record` middleware (which is what inserts the
    // extension) keep working — see `RequestPhaseCell`'s doc comment in
    // `crate::server::app`.
    phase: Option<Extension<Arc<RequestPhaseCell>>>,
    // Body-consuming extractor: MUST stay last. Every extractor before this
    // one must implement `FromRequestParts` (borrows the request head only);
    // `Bytes` is the one `FromRequest` extractor allowed per handler because
    // it consumes the request. Axum enforces the ordering at compile time —
    // putting another extractor after `Bytes` fails to compile, it does not
    // silently receive an empty body.
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    chat_completions_inner(ctx, headers, phase.map(|Extension(p)| p), body).await
}

/// Parse model from body, select a healthy worker via the per-model policy, then
/// proxy the request. If the request opts into streaming (`stream: true`), we
/// pipe SSE bytes back; otherwise buffer. This function does not emit the
/// per-request access-log line and does not count `requests_total` /
/// `responses_total`: it attaches a [`RequestLogContext`] to routed responses and
/// returns early errors via `?`, leaving all access logging and request/response
/// counting to the outermost `access_log_and_record` middleware (see
/// [`crate::server::app`]). It still records auxiliary metrics (TTFT, request
/// duration, stale-request, ingress-tokenize errors) and emits diagnostic logs.
async fn chat_completions_inner(
    ctx: Arc<AppContext>,
    headers: HeaderMap,
    phase: Option<Arc<RequestPhaseCell>>,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let start = std::time::Instant::now();
    let probe = parse_probe(&body)?;
    let streaming = probe.stream.unwrap_or(false);
    let model_str = probe
        .model
        .as_deref()
        .ok_or_else(|| ApiError::BadRequest("missing `model` field".into()))?
        .to_owned();
    let model_id = ModelId(model_str.clone());

    // Enforce the per-model output-token contract (`--max-output-tokens`)
    // before admission: an explicit ask above the cap is a client error
    // (mirroring the engine's own validation), and rejecting here costs no
    // queue slot and no engine round-trip. When the request set no output
    // budget at all, remember the cap for injection into the forwarded body
    // below — otherwise an unbounded request generates until EOS or the
    // engine's full context window fills.
    let inject_max_tokens = output_budget_action(ctx.config.model.max_output_tokens, &probe)?;

    // PD pool isolation: for PD-mode deployments, prefill traffic
    // selects from the prefill pool only. Plain-mode deployments fall
    // through to the full candidate set. Partial-failure errors
    // (`no_prefill_workers_available`) are surfaced as 503 with a
    // distinct error code so operators can alert independently.
    let resolver = PdPoolResolver::new(Arc::clone(&ctx.registry));
    let workers = resolver
        .prefill_candidates(&model_id)
        .map_err(|e| match e {
            PdResolveError::NoHealthyWorkers => ApiError::NoHealthyWorkers {
                model: model_str.clone(),
            },
            PdResolveError::NoPrefillWorkersAvailable => ApiError::NoPrefillWorkersAvailable {
                model: model_str.clone(),
            },
            PdResolveError::NoDecodeWorkersAvailable => ApiError::NoDecodeWorkersAvailable {
                model: model_str.clone(),
            },
        })?;

    let policy = ctx
        .policies
        .get(&model_id)
        .ok_or_else(|| ApiError::ModelNotFound(model_str.clone()))?;

    // Tokenize once at ingress whenever it can pay off — decoupled from the
    // routing policy, because forwarding `input_ids` is a property of the
    // MODEL (does it have a chat encoder so the router can produce
    // engine-equivalent tokens?), not of how we pick the worker. Two gates:
    //
    //   * `has_chat_encoder` → a chat request on this model yields
    //     engine-equivalent ids we can forward as `input_ids` so the engine
    //     skips re-tokenizing. This enables the offload for EVERY policy —
    //     sticky and round-robin included — not just cache-aware.
    //   * `needs_request_tokens()` → the cache-aware policy ALSO wants the
    //     raw-prompt path tokenized for tree matching even on a model with no
    //     chat encoder (`/v1/completions` / `text`), which the first gate
    //     alone wouldn't trigger.
    //
    // When neither holds, `parse_probe`'s minimal probe is enough, so we keep
    // avoiding the full `serde_json::Value` allocation over a (up to 1 MiB)
    // body. When parsed, this single value is reused for the routing
    // tokenization and the outgoing-body injection below (and PD bootstrap
    // injection). `parse_probe` already validated the object shape.
    let at_pre_tokenize = start.elapsed();
    let want_tokens = ctx.tokenizers.has_chat_encoder(&model_str) || policy.needs_request_tokens();
    let request_value: Option<serde_json::Value> = if want_tokens {
        Some(serde_json::from_slice(&body).map_err(|_| {
            ApiError::BadRequest("invalid request: body must be a JSON object".into())
        })?)
    } else {
        None
    };

    // The ids feed both the routing decision (cache-aware consumes them; other
    // policies ignore them) and — when engine-equivalent — the engine itself,
    // forwarded as `input_ids` so it skips re-tokenizing the same prompt. The
    // ingress owns the tokenize via the shared registry, so the choice of
    // policy never changes whether we tokenize.
    let request_tokens = request_value
        .as_ref()
        .and_then(|v| request_tokens_for(&ctx.tokenizers, &model_id, v));
    let at_post_tokenize = start.elapsed();
    // Recorded here, adjacent to the two markers, so it is evident nothing
    // downstream is inside the span — widening it past admission or dispatch
    // would make the metric a duplicate of the `router.ttfb` Server-Timing
    // header. Unsampled, unlike the diagnostic log below. Gated on
    // `want_tokens`: a request the ingress never tries to tokenize would
    // otherwise contribute a meaningless ~0 and drag every quantile down.
    if want_tokens {
        ctx.metrics.observe_tokenize(
            &model_str,
            at_post_tokenize
                .saturating_sub(at_pre_tokenize)
                .as_secs_f64(),
        );
    }

    // The request id, derived HERE rather than at dispatch because both tees
    // need it and the ingress one fires now. It is the join key that lets the
    // oracle pair a request's prompt record with its response record; without
    // it the two are unattributable and output tokens have nothing to attach
    // to. Same value is reused for the engine-facing `rid` below — one
    // derivation, so the two can never disagree.
    let derived_request_id = derive_request_id(probe.rid.as_deref(), &headers);

    // Arm the response-completion extend tee (see `cache_sim_extend`): when the
    // response finishes, the assistant reply's rendered turn is appended to
    // this request's token sequence and teed insert-only, so the NEXT round's
    // prompt — which re-sends this response as history — measures as the cache
    // hit a real engine's KV cache would serve. Two gates: the ingress
    // tokenization must have succeeded (if this request couldn't tokenize, the
    // next round's probe can't either, so seeding is pointless), and the
    // extension must be matchable at all (`extension_can_match`: DSV4 thinking
    // mode without tools re-renders history divergently, so its extensions
    // could only ever be dead blocks — mirroring the engine's own miss).
    let extend_tee_armed = (ctx.cache_sim_tee.is_some() || ctx.s3_export_sink.is_some())
        && request_tokens.is_some()
        && request_value
            .as_ref()
            .is_some_and(cache_sim_extend::extension_can_match);
    // The ingress ids the extension appends to, enabling the O(output)
    // incremental encode at response time. ONLY the chat-encoder
    // (engine-equivalent) tokenization qualifies: the raw-prompt fallback
    // renders differently, so concatenating a chat-rendered suffix onto it
    // would produce garbage — those requests take the full-re-encode fallback
    // instead (`prompt = None`). Bundled with the request's resolved
    // `RenderOpts` (the same resolution `request_tokens_for` used) so the
    // reply's turn suffix renders in the same thinking/effort mode as the
    // prompt. Cost when armed: one Vec clone (4 B/token), held until the
    // response completes.
    let extend_prompt: Option<cache_sim_extend::IngressPrompt> = if extend_tee_armed {
        request_tokens
            .as_ref()
            .filter(|t| t.engine_equivalent)
            .zip(request_value.as_ref())
            .map(|(t, v)| cache_sim_extend::IngressPrompt {
                ids: t.ids.clone(),
                opts: crate::tokenizer::dsv4::resolve_render_opts(v),
            })
    } else {
        None
    };
    // Diagnostic: ingress-tokenize cost, sampled. Fires for EVERY request that
    // reaches here — including those about to be shed at admission below — so a
    // shed request's pre-admission time (the latency the access log shows on a
    // 503) is attributable to tokenize vs. the rest.
    if PHASE_LOG_COUNTER
        .fetch_add(1, Ordering::Relaxed)
        .is_multiple_of(PHASE_LOG_SAMPLE)
    {
        tracing::debug!(
            tokenize_ms = at_post_tokenize.saturating_sub(at_pre_tokenize).as_millis() as u64,
            pre_admit_total_ms = at_post_tokenize.as_millis() as u64,
            want_tokens,
            model = %model_str,
            "phase_pre_admit",
        );
    }

    // Sticky-session routing key. When the sticky policy is configured,
    // read the routing key from the operator-chosen header into the
    // selection context; the policy pins it to a worker. Other policies
    // leave `routing_key` `None` and ignore it.
    //
    // Held as an owned `String` (not a `&str` borrowed from `headers`) so the
    // `selection_ctx` that carries it does not keep `headers` borrowed: the
    // handler moves `headers` below (decode-hint injection), and the retry loop
    // re-reads `selection_ctx` after that move, so a borrow of `headers` here
    // would outlive the move.
    let routing_key_owned: Option<String> = ctx
        .config
        .model
        .sticky
        .as_ref()
        .and_then(|s| headers.get(s.header_name.as_str()))
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let selection_ctx =
        SelectionContext::with_routing_key(&model_id, Some(&body), routing_key_owned.as_deref())
            .with_request_tokens(request_tokens.as_ref().map(|t| t.ids.as_slice()));
    // Admission gate: pick a worker and claim an in-flight slot, parking until
    // one frees if every candidate is at its cap. A pass-through (immediate
    // dispatch, unconditional guard) when no per-worker cap is configured.
    // Yields 503 `service_overloaded` when the wait queue is full.
    // `worker` / `guard` are `mut` so the plain-mode retry loop below can
    // reselect a different worker and claim a fresh slot on a retryable
    // dispatch failure. The PD branch never reassigns them.
    //
    // Advance the phase cell right before the parking `.await` and right
    // after it resolves, so a caller hang-up (which cancels this future —
    // see `RequestPhaseCell` in `crate::server::app`) is attributed to
    // `queue` while parked in the admission wait vs `dispatch` once a worker
    // has been selected.
    if let Some(p) = &phase {
        p.set(RequestPhase::Queue);
    }
    let (mut worker, mut guard) = ctx
        .admission
        .acquire(&workers, policy.as_ref(), &selection_ctx, &model_str)
        .await?;
    // Best-effort tee of the ingress-computed ids to the theoretical cache-sim.
    // Fire-and-forget — never blocks or fails the request. No-op unless
    // `--cache-sim-url` is set.
    //
    // Deliberately AFTER admission, not next to the tokenization it reuses. A
    // shed request never reaches an engine, so counting it would both inflate
    // the oracle's denominator with traffic that was never served and leave an
    // ingest record whose extend can never arrive — indistinguishable
    // downstream from a response that generated nothing. The bias would grow
    // exactly when the fleet is shedding, i.e. anti-correlated with health.
    let tee_attr = tee_attribution(&headers);
    if let (Some(tee), Some(t)) = (ctx.cache_sim_tee.as_ref(), request_tokens.as_ref()) {
        tee.offer(&model_str, &t.ids, &derived_request_id, tee_attr.clone());
    }
    if let (Some(sink), Some(t)) = (ctx.s3_export_sink.as_ref(), request_tokens.as_ref()) {
        sink.offer_ingest(
            &model_str,
            &t.ids,
            &derived_request_id,
            tee_attr.slug.as_deref(),
        );
    }
    if let Some(p) = &phase {
        p.set(RequestPhase::Dispatch);
    }
    let at_post_admit = start.elapsed();
    // Diagnostic: count this request as holding a slot inside the synchronous
    // handler (post-acquire → response returned). Drops when the function
    // returns (headers time for streaming), so HANDLER_INFLIGHT reflects slots
    // stuck before the SSE pump takes over.
    let _handler_phase = crate::diag::PhaseGuard::handler();

    // PD-mode decoder affinity. When the selected prefill worker is
    // part of a PD-disagg deployment, also resolve the matching decode
    // peer (same host where possible, falling back to min-load via
    // `select_decode_with_affinity`). Both workers receive the SAME
    // request body — augmented with the three flat `bootstrap_*`
    // fields below — so the SGLang engine can match incoming KV
    // transfers via `bootstrap_room`.
    //
    // Plain-mode workers skip the decode resolution entirely (no
    // decode peer to find). PD-mode requests that fail to resolve a
    // decode peer (`NoDecodeWorkersAvailable`) bubble up as 503 so
    // operators can alert on prefill-vs-decode pool imbalance.
    let decode_peer: Option<Arc<Worker>> = if worker.mode() == WorkerMode::Prefill {
        Some(
            resolver
                .decode_with_affinity(&model_id, &worker.url)
                .map_err(|e| match e {
                    PdResolveError::NoHealthyWorkers => ApiError::NoHealthyWorkers {
                        model: model_str.clone(),
                    },
                    PdResolveError::NoDecodeWorkersAvailable => {
                        ApiError::NoDecodeWorkersAvailable {
                            model: model_str.clone(),
                        }
                    }
                    PdResolveError::NoPrefillWorkersAvailable => {
                        ApiError::NoPrefillWorkersAvailable {
                            model: model_str.clone(),
                        }
                    }
                })?,
        )
    } else {
        None
    };
    let decode_hint_url: Option<String> = decode_peer.as_ref().map(|d| d.url.clone());
    let mut request_headers = headers;
    if let Some(url) = &decode_hint_url {
        match HeaderValue::from_str(url) {
            Ok(v) => {
                request_headers.insert(X_SGL_DECODE_URL, v);
            }
            Err(e) => {
                // Discovery emits URLs the proxy has already used; a
                // header-value parse failure here means the URL
                // contains a control character (e.g. CR / LF) — drop
                // the header but keep the request: bootstrap injection
                // below carries the host/port the engine actually
                // needs; the header is purely observability.
                tracing::warn!(
                    decode_url = %url,
                    error = %e,
                    "decode worker URL rejected by header parser; sending request without decode hint",
                );
            }
        }
    }
    let headers = request_headers;

    // Per-worker `active_requests` guard. The `ActiveLoadGuard` below
    // sits beside this one: both track in-flight load, but the
    // ActiveLoadGuard entry is per-request (with timeout-based janitor)
    // while the worker-scoped counter is what the cache-aware policy
    // reads. Both must drop at the same time — when the response stream
    // ends, the client disconnects, or the handler returns an error. In
    // PD mode the pair moves into the spawned prefill task so prefill
    // load is tracked for the full duration of the KV transfer; in plain
    // mode the pair stays in this handler. Decode-load contribution is
    // 0 here: the active-load registry's decode axis is reserved for a
    // future decode-side scheduler — current decode selection is
    // host-affinity only.
    // `guard` (this request's per-worker in-flight slot) was claimed by the
    // admission gate above; it is held until the dispatch guards below drop.
    // Use the exact token count from the ingress tokenization when available;
    // fall back to the byte-count heuristic for load-only policies that don't
    // tokenize. The exact count makes the cache-aware load-imbalance fast-path
    // accurate rather than off by the char/token ratio.
    let prefill_load = request_tokens
        .as_ref()
        .map(|t| t.ids.len().max(1))
        .unwrap_or_else(|| estimate_prefill_tokens(&body));
    // `mut` for the same reason as `worker`/`guard`: a plain-mode retry
    // re-registers active load on the newly-selected worker.
    let mut active_guard =
        ctx.active_load
            .register(worker.id.clone(), worker.url.clone(), prefill_load, 0);
    // Snapshot the stale-request cancel token BEFORE moving the guard
    // into the spawned prefill task / streaming pump / response future.
    // The token is cheap to clone (it's an `Arc<...>` internally) and
    // the chat handler races the client-facing fetch against
    // `token.cancelled()` to surface a 504 `stale_request_expired` if
    // the janitor expires the request mid-flight. `mut` so a retry can
    // re-snapshot the token of the new attempt's active-load entry.
    let mut stale_token = active_guard.cancel_token().clone();

    // Snapshot the labels we need for metrics BEFORE moving the worker
    // / model_str values into the per-branch fetch futures. `mut` so a
    // plain-mode retry updates them to the worker that actually served, so
    // the access log / `worker_requests_total` reflect the final worker.
    let mut metrics_worker_url = worker.url.clone();
    let mut metrics_mode = mode_label(worker.mode());
    let metrics_model = model_str.clone();

    // Builds the time-to-first-token hook the SSE pump fires when the first
    // upstream chunk lands. Installed only on the streaming arms below —
    // non-streaming "first token" equals total latency, already captured by
    // `sgl_router_request_duration_seconds`. The proxy drops the hook for
    // non-2xx responses so error bodies don't pollute TTFT.
    let make_ttft_hook = || -> Box<dyn FnOnce() + Send + 'static> {
        let metrics = Arc::clone(&ctx.metrics);
        let model = metrics_model.clone();
        let started = start;
        Box::new(move || {
            metrics.observe_ttft(&model, started.elapsed().as_secs_f64());
        })
    };

    // Builds the inter-token-latency hook the SSE pump fires with the gap
    // between successive upstream chunks. Installed only on the streaming
    // arms below, and only armed by the proxy for 2xx responses — the same
    // gate as TTFT.
    // Takes the serving worker's URL so the same per-chunk gap feeds both the
    // per-model `sgl_router_itl_seconds` histogram AND the per-worker `ItlTable`
    // that the retry load gate reads. Called fresh per streaming arm / retry
    // attempt with the current worker.
    let make_itl_hook = |worker_url: &str| -> Box<dyn Fn(f64) + Send + 'static> {
        let metrics = Arc::clone(&ctx.metrics);
        let model = metrics_model.clone();
        let itl = Arc::clone(&ctx.itl);
        let url = worker_url.to_owned();
        Box::new(move |gap_seconds: f64| {
            metrics.observe_itl(&model, gap_seconds);
            itl.record(&url, gap_seconds * 1000.0, std::time::Instant::now());
        })
    };

    // Builds the end-to-end-latency guard for streaming requests. Shared (via
    // `Arc`) across the plain-mode retry attempts and packed into each
    // attempt's `stream_guards`, so it records exactly once — when the SSE
    // pump finishes (stream end or client disconnect), not at
    // response-headers time, and not once per retry attempt. Non-streaming
    // records at the dispatch site instead (see below). The PD arm packs its
    // own instance directly (PD is single-attempt).
    let make_duration_guard = || RecordDurationOnDrop {
        metrics: Arc::clone(&ctx.metrics),
        model: metrics_model.clone(),
        start,
    };

    // Builds the stream-end hook the SSE pump fires when a 2xx streaming
    // response finishes, recording `sgl_router_stream_outcome_total` — the
    // end-of-stream truth that every headers-time counter is blind to. An
    // in-band error wins over the other classifications: it's the specific
    // signal (engine said no under a committed 200) even if the transport
    // also broke or the client bailed afterwards.
    let make_stream_end_hook = |url: &str| -> Box<dyn FnOnce(StreamEnd) + Send + 'static> {
        let metrics = Arc::clone(&ctx.metrics);
        let model = metrics_model.clone();
        let worker_url = url.to_string();
        Box::new(move |end: StreamEnd| {
            let outcome = if end.saw_inband_error {
                StreamOutcome::InbandError
            } else if !end.transport_ok {
                StreamOutcome::UpstreamError
            } else if end.client_disconnect {
                StreamOutcome::ClientDisconnect
            } else {
                StreamOutcome::Ok
            };
            metrics.record_stream_outcome(&worker_url, &model, outcome);
        })
    };

    // Builds the raw-bytes capture the SSE pump feeds for a streaming
    // response, so the completed generation can be teed to the cache-sim's
    // insert-only extension path (see `cache_sim_extend`). `None` (no capture,
    // no buffering) unless the extend tee is armed. Owns its own clones so the
    // per-arm `model_str` moves below don't conflict; `Bytes` clones are
    // refcount bumps. Callable once per dispatch attempt — a failed attempt's
    // capture is discarded by the pump (unclean end) and never fires.
    let make_extend_capture = {
        let armed = extend_tee_armed;
        let ctx = Arc::clone(&ctx);
        let model = model_str.clone();
        let request_body = body.clone();
        let prompt = extend_prompt.clone();
        let request_id = derived_request_id.clone();
        let slug = tee_attr.slug.clone();
        move || -> Option<StreamCapture> {
            if !armed {
                return None;
            }
            // Reserve a global capture slot so total concurrent captures — and
            // thus aggregate capture memory (≤ N × MAX_EXTEND_CAPTURE_BYTES) —
            // stay bounded under any load. Budget exhausted ⇒ don't capture this
            // stream (skip its extend tee, counted as `capture_capped`); the
            // capture is observational, so shedding it never affects serving.
            //
            // When cache-sim is on, draw from the cache-sim tee's pool and record
            // `capture_capped` there. Also record `dropped_capture_capped` on the
            // S3 sink so both consumers reflect the drop.
            // When cache-sim is off but the S3 sink is on, draw from the sink's
            // own permit pool so S3-only mode is equally memory-bounded.
            let permit = if let Some(tee) = ctx.cache_sim_tee.as_ref() {
                let p = tee.try_acquire_capture_permit();
                if p.is_none() {
                    ctx.metrics.record_cache_sim_tee("capture_capped");
                    if ctx.s3_export_sink.is_some() {
                        ctx.metrics.record_s3_export("dropped_capture_capped");
                    }
                    return None;
                }
                p
            } else if let Some(sink) = ctx.s3_export_sink.as_ref() {
                let p = sink.try_acquire_capture_permit();
                if p.is_none() {
                    ctx.metrics.record_s3_export("dropped_capture_capped");
                    return None;
                }
                p
            } else {
                return None;
            };
            let ctx = Arc::clone(&ctx);
            let model = model.clone();
            let request_body = request_body.clone();
            let prompt = prompt.clone();
            let request_id = request_id.clone();
            let slug = slug.clone();
            Some(StreamCapture {
                max_bytes: cache_sim_extend::MAX_EXTEND_CAPTURE_BYTES,
                _permit: permit,
                on_done: Box::new(move |buf| {
                    cache_sim_extend::spawn_extend_tee(
                        ctx,
                        model,
                        request_id,
                        request_body,
                        prompt,
                        ReplySource::Sse(buf),
                        slug,
                    );
                }),
            })
        }
    };

    // Forward the router-computed tokens to the engine as `input_ids` so it
    // skips re-tokenizing the same prompt — but only when the offload is
    // enabled (`--disable-input-ids-offload` gates it centrally) AND the ids
    // are engine-equivalent (chat-encoder path) AND the request contains
    // nothing the router's encoder didn't replicate (the per-encoder
    // predicate, selected via the ids' stamped parity in
    // `select_forward_input_ids`). Otherwise omit them and the engine
    // tokenizes from `messages` as usual — a transparent, always-correct
    // fallback (`messages` are always retained in the forwarded body).
    // `input_ids_to_forward` (the selected ids) is named distinct from
    // `ModelConfig::forward_input_ids` (the gate) — both names are live in
    // this scope.
    let input_ids_to_forward: Option<&[u32]> = select_forward_input_ids(
        ctx.config.model.forward_input_ids,
        request_tokens.as_ref(),
        request_value.as_ref(),
    );

    // Surface a broken offload: when the encoder SHOULD have produced
    // engine-equivalent ids but didn't, the chat request silently fell back to
    // engine-side tokenization. Count only that case (see
    // `ingress_tokenize_offload_failed`); successful forwards and expected
    // omissions are not problems. A dsv4 render error is a CLIENT error the
    // engine rejects identically — never broken-offload signal.
    if ingress_tokenize_offload_failed(
        ctx.tokenizers.has_chat_encoder(&model_str),
        request_value.as_ref(),
        request_tokens.as_ref(),
    ) && !dsv4_render_rejects_request(&ctx.tokenizers, &model_str, request_value.as_ref())
    {
        ctx.metrics.record_ingress_tokenize_error(&metrics_model);
    }

    // PD-disagg bootstrap fields (prefill worker address + a per-request
    // room). Present only when a decode peer was resolved.
    let bootstrap = decode_peer.as_ref().map(|_| BootstrapFields {
        host: worker.bootstrap_host().to_string(),
        port: worker.bootstrap_port(),
        room: generate_room_id(),
    });
    let bootstrap_room = bootstrap.as_ref().map(|b| b.room);

    // Request id used to abort the engine if the client disconnects mid-flight.
    // Scoped to plain (non-PD) mode: PD deliberately detaches its prefill so it
    // outlives the client (KV-transfer correctness), and aborting only the
    // decode half mid-transfer is a riskier change left out of scope here — so
    // PD requests get no rid injection and no abort, preserving today's
    // behavior exactly.
    //
    // Reuse a client-supplied string `rid` (so an external abort-by-rid keeps
    // working and we don't override intent); otherwise mint one and inject it
    // into the forwarded body so the engine adopts it. SGLang keeps a provided
    // `rid` and only generates one when it is absent, and it aborts every rid
    // that *starts with* the one we send — covering `n>1` expansions.
    //
    // Sourced from `probe.rid`, NOT `request_value` — `request_value` is only
    // parsed when `want_tokens` is true, which has nothing to do with whether
    // the client passed a `rid`. Gating on it silently dropped client rids for
    // any model/policy that doesn't need ingress tokenization.
    let client_rid: Option<String> = probe.rid;
    // Reuses `derived_request_id` (computed before the ingress tee) rather
    // than re-deriving: a second derivation would mint a fresh UUID on the
    // no-header path, so the id the engine sees and the id the tee sent would
    // silently differ for exactly the traffic that bypasses the gateway.
    let request_id: Option<String> = if decode_peer.is_none() {
        Some(derived_request_id.clone())
    } else {
        None
    };
    // Inject only a router-minted rid; a client-supplied one is already in the
    // body, and PD mode (`request_id == None`) is never injected.
    // Same empty filter `derive_request_id` applies. Without it, a client
    // sending `"rid": ""` falls to the `_ => None` arm: nothing is injected, so
    // the engine mints its own id while the tee and the disconnect-abort guards
    // use `router-<uuid>` — three legs, three different ids, for exactly the
    // input the filter was added to handle. (The abort also silently no-ops,
    // which is strictly better than what it did before the filter existed:
    // sending `""`, which SGLang prefix-matches against EVERY in-flight rid.)
    let rid_to_inject: Option<&str> = match (
        request_id.as_deref(),
        client_rid.as_deref().filter(|s| !s.is_empty()),
    ) {
        (Some(rid), None) => Some(rid),
        _ => None,
    };

    // Build the body forwarded to the engine(s) exactly once — injecting the
    // `rid`, `input_ids`, bootstrap fields, and/or default `max_tokens`, or
    // forwarding the original bytes untouched when none apply.
    let outgoing_body = build_outgoing_body(
        &body,
        request_value,
        input_ids_to_forward,
        bootstrap.as_ref(),
        rid_to_inject,
        inject_max_tokens,
    )?;
    let at_post_build = start.elapsed();

    let result = if let Some(decode_worker) = decode_peer {
        // PD-disagg dispatch (Pattern B — spawn prefill, await decode).
        //
        // SGLang's HTTP-mode disagg-prefill requires three flat
        // top-level fields on the request body: `bootstrap_host`,
        // `bootstrap_port` (the prefill worker's bootstrap-server
        // address) and `bootstrap_room` (a per-request 63-bit u64 ID
        // used by both sides to pair up the KV transfer). We inject
        // these here and fan the same modified body to both the
        // prefill and decode workers concurrently.
        //
        // **Why spawn-and-forget for prefill instead of
        // `tokio::join!`?** All three peer SGLang-HTTP-PD routers
        // (Dynamo / llm-d / aibrix) converged on this shape: the
        // prefill request must outlive the client connection because
        // tying prefill to the client future opens a cancel-race
        // window where the engine's NIXL RPC teardown can leak KV
        // block refs (NVBugs 5969206 in Dynamo). The detached task
        // also keeps the LoadGuard + ActiveLoadGuard alive for the full
        // prefill duration — KV transfer can run for tens of seconds
        // even when the client gave up.
        //
        // No watchdog for fail-fast on prefill 5xx: llm-d / aibrix both
        // ship without one. On prefill failure the client experiences
        // the SGLang decode-side bootstrap_room timeout (~30–60 s by
        // default) instead of an immediate 502. A follow-up can wire a
        // `tokio::sync::watch` channel if telemetry shows it matters.
        //
        // **Scope of the "detached" guarantee.** The spawn protects
        // against client disconnect — the handler future being dropped
        // does NOT cancel the prefill HTTP request. It does NOT protect
        // against router shutdown: when `AppContext` tears down, the
        // tokio runtime cancels all unfinished tasks including this
        // one. A future follow-up could thread a `TaskTracker` /
        // `JoinSet` through `AppContext` for graceful shutdown drain;
        // the current implementation ships without one (matching SMG's
        // shutdown behaviour).
        let bootstrap_room = bootstrap_room.expect("PD dispatch implies a resolved bootstrap room");

        let prefill_url = worker.url.clone();
        let prefill_protocol = worker.protocol();
        let prefill_breaker = Arc::clone(&worker.breaker);
        let prefill_headers = headers.clone();
        let prefill_body = outgoing_body.clone();
        let prefill_proxy = Arc::clone(&ctx.proxy);
        let prefill_holds = (guard, active_guard);
        tokio::spawn(async move {
            // The tuple binding extends both guards' lifetime to the
            // end of this async block, which lasts until the prefill
            // HTTP request returns (success / error / engine-side
            // bootstrap_room timeout). The result is logged and
            // swallowed — no channel back to the client. See the big
            // comment above for the rationale.
            let _hold = prefill_holds;
            match prefill_proxy
                .forward_json_to(
                    &prefill_url,
                    prefill_protocol,
                    &prefill_breaker,
                    "/v1/chat/completions",
                    &prefill_headers,
                    prefill_body,
                )
                .await
            {
                Ok(_) => tracing::debug!(
                    prefill_url = %prefill_url,
                    bootstrap_room,
                    "prefill side completed",
                ),
                Err(e) => tracing::warn!(
                    prefill_url = %prefill_url,
                    bootstrap_room,
                    error = %e,
                    "prefill request failed; decode will time out on bootstrap_room",
                ),
            }
        });

        // Synchronously await the decode worker. Its response is what
        // the client sees. The decode side gets its own LoadGuard so
        // per-worker `active_requests` reflects decode-pool load for
        // cache-aware-zmq decisions on the decode side.
        let decode_guard = decode_worker.load_guard();
        if streaming {
            let stream_guards: Box<dyn Send + 'static> =
                Box::new((decode_guard, make_duration_guard()));
            let fetch = ctx.proxy.forward_streaming_to(
                &decode_worker.url,
                decode_worker.protocol(),
                &decode_worker.breaker,
                "/v1/chat/completions",
                &headers,
                outgoing_body,
                Some(stream_guards),
                Some(make_ttft_hook()),
                // `None` in PD mode (request_id is None): decode abort is out
                // of scope here — see the request_id comment above.
                request_id.as_deref(),
                Some(make_stream_end_hook(&decode_worker.url)),
                Some(make_itl_hook(&decode_worker.url)),
                make_extend_capture(),
            );
            tokio::select! {
                biased;
                r = fetch => r,
                _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str, worker: worker_url_for_error(&decode_worker.url) }),
            }
        } else {
            let _decode_hold = decode_guard;
            let fetch = ctx.proxy.forward_json_to(
                &decode_worker.url,
                decode_worker.protocol(),
                &decode_worker.breaker,
                "/v1/chat/completions",
                &headers,
                outgoing_body,
            );
            tokio::select! {
                biased;
                r = fetch => r,
                _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str, worker: worker_url_for_error(&decode_worker.url) }),
            }
        }
    } else {
        // Plain mode. Single-retry failover: on a transient dispatch failure
        // that occurs before any bytes reach the client, re-dispatch the SAME
        // request ONCE, to a DIFFERENT worker whose in-flight load is below the
        // admission cap. This is the router-side failover the circuit breaker
        // alone can't provide — the breaker ejects a bad worker from *future*
        // selections, but only retry recovers the in-flight request that hit it.
        //
        // Retry is opt-in (`retry.enabled`, default false ⇒ exactly one
        // attempt). The retryable failures (see `ApiError::is_retryable_upstream`)
        // all return `Err` from the forward BEFORE the response headers
        // (streaming) or full body (non-streaming) arrive, so nothing has been
        // streamed to the client and the request is safe to re-dispatch. A
        // worker that returns a well-formed non-2xx (e.g. an engine 503) is
        // `Ok(response)` here, not `Err` — the proxy forwards it verbatim, so it
        // reaches the client and is never retried.
        //
        // The re-dispatch uses the non-parking, load-gated
        // `AdmissionQueue::try_acquire`: it lands only on a worker with a free
        // slot (never onto one already at capacity) and never waits for a slot —
        // if the rest of the fleet is saturated the retry is skipped and the
        // original error surfaces. Dropping a failed attempt's guards releases
        // its admission slot and fires its abort guard (a no-op for a worker
        // that never received the request), telling that engine to stop before
        // we try elsewhere. The retry re-registers active load on the new
        // worker, which re-snapshots the stale-request token — so a retried
        // request gets a fresh stale budget and can live up to roughly
        // (first-attempt elapsed + stale timeout) before the janitor fires.
        //
        // Exactly-once end-to-end duration for streaming requests, shared
        // across attempts: each attempt's `stream_guards` box holds a clone of
        // this Arc, and `RecordDurationOnDrop` fires when the LAST clone drops —
        // the successful attempt's SSE-pump end (the pump outlives both this
        // scope and any failed attempt's box), or this scope's handle when
        // every attempt failed. A failed attempt's box dropping early is never
        // last, so a retried request records ONE duration sample, not one per
        // attempt. Streaming-only: the non-streaming arm records once at the
        // dispatch site below.
        let stream_duration = streaming.then(|| Arc::new(make_duration_guard()));
        let mut retried = false;
        loop {
            // Wall-clock start of THIS dispatch attempt. Read by the retry TTFT
            // gate below (if the attempt fails) to decide whether a re-dispatch is
            // still worth it. The deadline is NOT a timer here — it never
            // interrupts a running attempt; the attempt completes or fails on its
            // own, bounded only by `request_timeout` / the stale budget.
            let attempt_start = std::time::Instant::now();
            let attempt_result = if streaming {
                // Plain mode, streaming. Both guards ride the SSE pump until
                // the body completes — see the matching comment in the
                // non-streaming arm.
                let stream_guards: Box<dyn Send + 'static> = Box::new((
                    guard,
                    active_guard,
                    stream_duration.as_ref().map(Arc::clone),
                ));
                // Pre-headers abort guard: `forward_streaming_to` only constructs its
                // own (reached_end-tracking) guard AFTER a response is received, so
                // it can't protect the window before that — if the stale-request
                // janitor fires while `fetch` is still awaiting headers, `fetch` (and
                // the not-yet-existing internal guard) is dropped with no abort sent,
                // even though the engine may already be working on the request. This
                // guard covers exactly that window: armed until `fetch` resolves to
                // any received response (disarmed below), at which point the
                // internal guard (for a 2xx) or nothing (non-2xx, same as today)
                // takes over — same disarm-on-`Ok` pattern as the non-streaming arm.
                let mut pre_headers_abort_guard = request_id.as_deref().and_then(|rid| {
                    ctx.proxy
                        .abort_guard_for(&worker.url, worker.protocol(), rid)
                });
                let fetch = ctx.proxy.forward_streaming_to(
                    &worker.url,
                    worker.protocol(),
                    &worker.breaker,
                    "/v1/chat/completions",
                    &headers,
                    // Cloned so the body survives for a possible re-dispatch;
                    // `Bytes` clone is a cheap refcount bump.
                    outgoing_body.clone(),
                    Some(stream_guards),
                    Some(make_ttft_hook()),
                    // Abort the engine if the client disconnects before the engine
                    // finishes streaming. The SSE pump (which owns the guard) fires
                    // it; here we only supply the rid the engine knows this request by.
                    request_id.as_deref(),
                    Some(make_stream_end_hook(&metrics_worker_url)),
                    Some(make_itl_hook(&worker.url)),
                    make_extend_capture(),
                );
                // Bias `fetch` over the cancellation branch: a successful
                // response that completes in the same poll as the token firing
                // MUST win (returning 504 for a request that already has
                // headers is a correctness regression). The cancellation
                // branch only matters when fetch is still pending — at that
                // point biasing the order is a wash.
                let r = tokio::select! {
                    biased;
                    r = fetch => r,
                    _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str.clone(), worker: worker_url_for_error(&worker.url) }),
                };
                // A received response (any status) means responsibility has passed
                // to `forward_streaming_to`'s own guard (or nothing, for non-2xx) —
                // disarm so this one doesn't also fire. Left armed only when `fetch`
                // never resolved (stale-timeout) or a transport-level dispatch error
                // occurred before any response.
                match &r {
                    Ok(_) => {
                        if let Some(g) = pre_headers_abort_guard.as_mut() {
                            g.disarm();
                        }
                    }
                    Err(err) => {
                        // Narrow the abort reason before the guard drops so the
                        // WARN log + POST body identify the *specific* trigger
                        // (stale timeout / upstream timeout / transport) rather
                        // than the constructor default `HandlerCancelled`.
                        if let Some(g) = pre_headers_abort_guard.as_ref() {
                            g.set_reason(abort_reason_from_api_error(err));
                        }
                    }
                }
                r
            } else {
                // Plain mode, non-streaming. The handler awaits the full
                // buffered response, so both guards live correctly in this
                // scope. The tuple binding exists only to extend the guards'
                // lifetime to the end of the attempt — the `forward_json_to`
                // future does not need them (it does not return until the
                // body is buffered).
                let _holds = (guard, active_guard);
                // Abort-on-disconnect: armed for the whole forward, disarmed once a
                // complete response is in hand. If the client disconnects first the
                // handler future is dropped mid-await and this guard, still armed,
                // tells the engine to stop. A stale-request timeout (the cancel arm
                // below) also leaves it armed — we've given up, so the engine should
                // too. `None` only in PD mode / on a worker-URL parse failure.
                let mut abort_guard = request_id.as_deref().and_then(|rid| {
                    ctx.proxy
                        .abort_guard_for(&worker.url, worker.protocol(), rid)
                });
                let fetch = ctx.proxy.forward_json_to(
                    &worker.url,
                    worker.protocol(),
                    &worker.breaker,
                    "/v1/chat/completions",
                    &headers,
                    outgoing_body.clone(),
                );
                // Same `biased` order as the streaming arm.
                let r = tokio::select! {
                    biased;
                    r = fetch => r,
                    _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str.clone(), worker: worker_url_for_error(&worker.url) }),
                };
                // A complete response (any status) means the engine is done with this
                // request — don't abort it. Only an early drop (client disconnect) or
                // stale-timeout leaves the guard armed.
                match &r {
                    Ok(_) => {
                        if let Some(g) = abort_guard.as_mut() {
                            g.disarm();
                        }
                    }
                    Err(err) => {
                        // Same reason-narrowing as the streaming arm above — see there
                        // for the rationale. Handler-cancellation (client disconnect
                        // during the buffered await) still hits the constructor
                        // default because that path returns no `Err`, it just drops.
                        if let Some(g) = abort_guard.as_ref() {
                            g.set_reason(abort_reason_from_api_error(err));
                        }
                    }
                }
                r
            };

            match attempt_result {
                Ok(resp) => break Ok(resp),
                Err(e) => {
                    // At most one retry: a request that already used its single
                    // re-dispatch and still failed is the un-recovered tail.
                    // Checked BEFORE the retryability test so the second failure
                    // counts as exhausted even when it is itself non-retryable
                    // (e.g. the stale deadline firing mid-retry) — otherwise
                    // retries_total − retries_exhausted over-reports recoveries.
                    if retried {
                        ctx.metrics.record_retries_exhausted(&metrics_model);
                        break Err(e);
                    }
                    // Retry only transient dispatch failures, and only while
                    // retry is enabled. A non-retryable error (bad request,
                    // mid-body drop, stale-deadline cancel, …) surfaces as-is.
                    if !ctx.config.retry.enabled || !e.is_retryable_upstream() {
                        break Err(e);
                    }
                    // Retry TTFT gate (`retry.attempt_deadline_ms`): the failed
                    // attempt already ran this long. If it spent AT LEAST the
                    // configured budget before failing, a re-dispatch would add a
                    // full fresh generation onto a healthy worker for a request
                    // that has already blown its time budget — so don't retry;
                    // surface the original failure. This is a gate on the retry
                    // decision, never a cap on the attempt itself. Unset ⇒ no time
                    // gate (retry regardless of how long the attempt took).
                    if let Some(deadline_ms) = ctx.config.retry.attempt_deadline_ms {
                        let elapsed = attempt_start.elapsed();
                        if elapsed >= std::time::Duration::from_millis(deadline_ms) {
                            tracing::debug!(
                                model = %model_str,
                                failed_worker = %worker.url,
                                error = %e,
                                elapsed_ms = elapsed.as_millis() as u64,
                                deadline_ms,
                                "retry skipped: first attempt reached the retry deadline before failing (TTFT gate)",
                            );
                            ctx.metrics.record_retries_exhausted(&metrics_model);
                            break Err(e);
                        }
                    }
                    // Fail over to a DIFFERENT worker (exclude the one we just
                    // tried). With only one retry, excluding the current worker
                    // by URL is enough.
                    let remaining: Vec<Arc<Worker>> = workers
                        .iter()
                        .filter(|w| w.url != worker.url)
                        .cloned()
                        .collect();
                    if remaining.is_empty() {
                        // No other worker to fail over to.
                        tracing::debug!(
                            model = %model_str,
                            failed_worker = %worker.url,
                            error = %e,
                            "retry skipped: no other worker to fail over to",
                        );
                        ctx.metrics.record_retries_exhausted(&metrics_model);
                        break Err(e);
                    }
                    // ITL load gate: drop candidates whose router-observed decode
                    // latency is over the ceiling (opt-in via
                    // `retry.max_target_itl_ms`) so a re-dispatch never lands on a
                    // decode-congested worker. A worker with no fresh ITL sample is
                    // eligible (missing data must not block failover); with the gate
                    // unconfigured every worker passes, leaving the count-based path.
                    let now = std::time::Instant::now();
                    let source_itl = ctx.itl.get_fresh(&worker.url, now);
                    let remaining: Vec<Arc<Worker>> = remaining
                        .into_iter()
                        .filter(|w| {
                            itl_target_eligible(
                                &ctx.config.retry,
                                ctx.itl.get_fresh(&w.url, now),
                                source_itl,
                            )
                        })
                        .collect();
                    if remaining.is_empty() {
                        // Every other worker is ITL-hot.
                        tracing::debug!(
                            model = %model_str,
                            failed_worker = %worker.url,
                            error = %e,
                            "retry skipped: every other worker is ITL-hot (above the retry ITL ceiling)",
                        );
                        ctx.metrics.record_retries_exhausted(&metrics_model);
                        break Err(e);
                    }
                    // Load gate: claim a slot on a different worker ONLY if one
                    // is below its in-flight cap — never onto a full worker, and
                    // never by waiting for a slot. `try_acquire` returns `None`
                    // when the whole remaining fleet is saturated; in that case
                    // (and when the policy declines) we skip the retry and
                    // surface the ORIGINAL upstream error — the request did
                    // reach a worker — not an admission error.
                    match ctx.admission.try_acquire(
                        &remaining,
                        policy.as_ref(),
                        &selection_ctx,
                        &model_str,
                    ) {
                        Ok(Some((next_worker, next_guard))) => {
                            retried = true;
                            ctx.metrics.record_retry(&metrics_model);
                            tracing::info!(
                                model = %model_str,
                                from = %worker.url,
                                to = %next_worker.url,
                                error = %e,
                                "retry: re-dispatching to a different not-full worker after transient dispatch failure",
                            );
                            // Update the per-attempt state: metrics labels (so
                            // the access log reflects the final worker), a fresh
                            // active-load registration + its stale token.
                            // `guard`/`active_guard` were consumed by the attempt
                            // above; reassign them for the retry pass.
                            metrics_worker_url = next_worker.url.clone();
                            metrics_mode = mode_label(next_worker.mode());
                            active_guard = ctx.active_load.register(
                                next_worker.id.clone(),
                                next_worker.url.clone(),
                                prefill_load,
                                0,
                            );
                            stale_token = active_guard.cancel_token().clone();
                            guard = next_guard;
                            worker = next_worker;
                            continue;
                        }
                        Ok(None) => {
                            // Every other worker is at capacity: don't retry.
                            // Expected under load — debug, not warn.
                            tracing::debug!(
                                model = %model_str,
                                failed_worker = %worker.url,
                                error = %e,
                                "retry skipped: every other worker is at its in-flight cap",
                            );
                            ctx.metrics.record_retries_exhausted(&metrics_model);
                            break Err(e);
                        }
                        Err(sel_err) => {
                            // The policy declined a NON-EMPTY set of under-cap
                            // workers — `try_acquire` only errors after the
                            // all-full check, so this is abnormal (a policy bug
                            // or a cache-aware/sticky edge case), not ordinary
                            // saturation: warn where saturation only debugs.
                            tracing::warn!(
                                model = %model_str,
                                failed_worker = %worker.url,
                                error = %e,
                                selection_error = %sel_err,
                                "retry skipped: policy declined the remaining candidates",
                            );
                            ctx.metrics.record_retries_exhausted(&metrics_model);
                            break Err(e);
                        }
                    }
                }
            }
        }
    };

    // Diagnostic: phase breakdown up to response headers, for ADMITTED requests
    // (those that got a slot). `dispatch_to_headers_ms` is connect + send-body +
    // wait-for-upstream-headers; for streaming this is the whole synchronous cost
    // before the SSE pump takes over. The pump's own first-byte/drain/exit timing
    // is logged separately as `sse_pump_timing`.
    let at_post_dispatch = start.elapsed();
    if PHASE_LOG_COUNTER
        .fetch_add(1, Ordering::Relaxed)
        .is_multiple_of(PHASE_LOG_SAMPLE)
    {
        let (handler_inflight, in_send, pump_inflight) = crate::diag::snapshot();
        tracing::debug!(
            tokenize_ms = at_post_tokenize.saturating_sub(at_pre_tokenize).as_millis() as u64,
            admit_ms = at_post_admit.saturating_sub(at_post_tokenize).as_millis() as u64,
            build_ms = at_post_build.saturating_sub(at_post_admit).as_millis() as u64,
            dispatch_to_headers_ms = at_post_dispatch.saturating_sub(at_post_build).as_millis() as u64,
            to_headers_total_ms = at_post_dispatch.as_millis() as u64,
            // process-wide phase gauges: where do held admission slots sit?
            g_handler_inflight = handler_inflight,
            g_in_send = in_send,
            g_pump_inflight = pump_inflight,
            streaming,
            worker = %metrics_worker_url,
            model = %metrics_model,
            "phase_dispatch",
        );
    }

    // The stale-request janitor fired and we observed it user-side (a 504).
    // Record the global `expired` count; the per-request `cancelled` outcome and
    // the access-log line are emitted centrally by the `access_log_and_record`
    // middleware, derived from the final HTTP status.
    if matches!(&result, Err(ApiError::StaleRequestExpired { .. })) {
        ctx.metrics
            .record_stale_request(StaleRequestOutcome::Expired);
    }

    // End-to-end latency for non-streaming requests: the body is already
    // buffered here, so `start.elapsed()` is the true total. Streaming records
    // at stream completion via the `RecordDurationOnDrop` guard packed into
    // `stream_guards` (so it isn't just time-to-headers).
    if !streaming {
        ctx.metrics
            .observe_request_duration(&metrics_model, start.elapsed().as_secs_f64());
    }

    // Routing context for the outermost middleware: it records
    // `worker_requests_total{worker_url,model_id,mode,outcome}` and the access-log line
    // for this request. Attaching it here (rather than recording directly) keeps
    // all request accounting at one site that also covers pre-routing
    // rejections, so the by-outcome view reflects ALL ingress.
    let log_ctx = RequestLogContext {
        worker_url: metrics_worker_url,
        model_id: metrics_model,
        mode: metrics_mode,
    };

    // Mirror the upstream `x-sgl-decode-url` hint onto the response so
    // external tests / sidecars can observe PD decode affinity without
    // sniffing the proxy hop. The request-side header was set above for
    // the prefill worker; copying it here makes the affinity observable
    // end-to-end. Plain-mode requests skip this (no decode peer was
    // resolved). A malformed URL was already rejected at the
    // request-side parse — we only reach this branch when the URL was
    // header-valid, so the second parse is safe.
    let mut response = match (result, decode_hint_url) {
        (Ok(mut response), Some(url)) => {
            match HeaderValue::from_str(&url) {
                Ok(v) => {
                    response.headers_mut().insert(X_SGL_DECODE_URL, v);
                }
                Err(e) => {
                    // Already-validated upstream; defensive log only.
                    tracing::warn!(
                        decode_url = %url,
                        error = %e,
                        "decode worker URL rejected by header parser on response; omitting response-side hint",
                    );
                }
            }
            response
        }
        (Ok(response), None) => response,
        // Post-dispatch error (a worker was selected). Materialize it so it can
        // be tagged with the routing context for the middleware, instead of
        // returning `Err` — early `?` short-circuits return `Err` and the
        // middleware records those as pre-routing rejections (empty worker_url).
        (Err(e), _) => e.into_response(),
    };
    // Response-completion tee, non-streaming half (streaming rides the SSE
    // pump's `StreamCapture` instead): `forward_json_to` stashes the buffered
    // body on the response as a `BufferedResponseBody` extension (a refcount
    // bump on the same buffer the `Body` serves), so the tee reads it without
    // consuming or rebuilding the response — the client's response object is
    // untouched by this block.
    if extend_tee_armed && !streaming && response.status().is_success() {
        if let Some(crate::proxy::BufferedResponseBody(bytes)) = response
            .extensions()
            .get::<crate::proxy::BufferedResponseBody>(
        ) {
            if bytes.len() <= cache_sim_extend::MAX_EXTEND_CAPTURE_BYTES {
                cache_sim_extend::spawn_extend_tee(
                    Arc::clone(&ctx),
                    log_ctx.model_id.clone(),
                    derived_request_id.clone(),
                    body.clone(),
                    extend_prompt,
                    ReplySource::Json(bytes.clone()),
                    tee_attr.slug.clone(),
                );
            }
        }
    }
    // Router TTFT observability — streaming responses that reached a worker and
    // got 2xx headers. Gated on `streaming` because on a non-streaming response
    // "time to first byte" is just total latency (there is no first token to be
    // early for); on `is_success()` to exclude forwarded engine errors. Two
    // artifacts, both from timestamps already taken above:
    //   - `sgl_router_ttft_overhead_seconds` = `at_post_build` = the router's
    //     pre-dispatch, self-attributable share of TTFT (tokenize + admission
    //     wait + build), a sub-term of `ttft`, retry-independent. Recorded here
    //     at headers time, so its sample set is a SUPERSET of `ttft_seconds`
    //     (which additionally needs the first chunk to arrive from the SSE
    //     pump); the two diverge only for a 2xx stream that yields no first
    //     byte.
    //   - `Server-Timing: router.ttfb;dur=<ms>` = `at_post_dispatch` = the full
    //     ingress → upstream-response-headers span. Carries the whole span,
    //     not the pre-dispatch subset, so a downstream hop that also measures
    //     the engine's own generation latency can separate router-incurred
    //     time from engine-incurred time on a per-request basis (something
    //     aggregate histograms can't do per-request). It is a response
    //     header, so it must be readable at header-flush time — hence
    //     to-headers rather than to-first-token (the first token has not
    //     arrived yet). `error.rs` appends `router.stage;desc=<stage>` to the
    //     same `Server-Timing` header on router-generated ERROR responses
    //     (see [`crate::server::error::ApiError::stage`]) — the two never
    //     collide since this site only fires on a 2xx.
    if streaming && response.status().is_success() {
        ctx.metrics
            .observe_ttft_overhead(&log_ctx.model_id, at_post_build.as_secs_f64());
        let ttfb_ms = at_post_dispatch.as_secs_f64() * 1000.0;
        match HeaderValue::from_str(&format!("router.ttfb;dur={ttfb_ms:.1}")) {
            // `append`, not `insert`: `Server-Timing` is a list-valued header,
            // so we add our metric without clobbering any the upstream set.
            Ok(v) => {
                response.headers_mut().append(SERVER_TIMING, v);
            }
            // Fully controlled ASCII value — unreachable in practice; log
            // rather than silently drop so a future format change is visible.
            Err(e) => {
                tracing::warn!(error = %e, "Server-Timing router.ttfb header rejected by parser; omitting");
            }
        }
    }

    // Tag the routed response so the middleware records its per-worker labels
    // and logs it with the worker/model it was dispatched to.
    response.extensions_mut().insert(log_ctx);
    Ok(response)
}

/// Whether a retry target passes the ITL load gate. The gate is opt-in: it
/// engages ONLY when a ceiling (`max_target_itl_ms`) is set — with no ceiling
/// every worker passes and retry behaves exactly as the count-based path. When
/// engaged, a target with a known ITL must be at or below the ceiling AND (when
/// the failed worker's ITL is also known) at or below it times the relative
/// factor. A target with unknown ITL always passes — missing data must never
/// block a failover.
fn itl_target_eligible(
    retry: &RetryConfig,
    target_itl: Option<f64>,
    source_itl: Option<f64>,
) -> bool {
    let Some(ceiling) = retry.max_target_itl_ms else {
        return true;
    };
    let Some(t) = target_itl else {
        return true;
    };
    if t > ceiling as f64 {
        return false;
    }
    if let Some(s) = source_itl {
        let factor = retry.itl_rel_factor.unwrap_or(DEFAULT_RETRY_ITL_REL_FACTOR);
        if t > s * factor as f64 {
            return false;
        }
    }
    true
}

/// Map a worker's [`WorkerMode`] to the metrics [`WorkerModeLabel`]. Used at
/// the initial dispatch and again after a plain-mode retry reselects a worker,
/// so both sites agree on the mapping.
fn mode_label(mode: WorkerMode) -> WorkerModeLabel {
    match mode {
        WorkerMode::Prefill => WorkerModeLabel::Prefill,
        WorkerMode::Decode => WorkerModeLabel::Decode,
        WorkerMode::Plain => WorkerModeLabel::Plain,
    }
}

/// Estimate prefill-token count from the raw request body for use as
/// the active-load `prefill_load` counter. Returns 1 at minimum so
/// a registered request always shows up as "load > 0" — under-counting
/// to zero would hide the request from the cache-aware policy's
/// load-imbalance fast-path.
///
/// This is a coarse approximation: we count the body length in bytes
/// and divide by [`CHARS_PER_TOKEN_ESTIMATE`]. A future improvement is
/// to thread the tokenizer's actual token count through (the
/// cache-aware-zmq policy already tokenizes the prompt for tree
/// matching — that count could be reused here).
fn estimate_prefill_tokens(body: &Bytes) -> usize {
    (body.len() / CHARS_PER_TOKEN_ESTIMATE).max(1)
}

/// Mint a fresh `bootstrap_room` for a PD-disagg request.
///
/// SGLang's disagg-prefill stores the room as a signed `i64` internally
/// (see `python/sglang/srt/disaggregation/utils.py` — `bootstrap_room`
/// metadata buffer is allocated as `torch.int64`). Generating in
/// `[0, i64::MAX]` keeps the value safely positive when reinterpreted
/// signed. Mirrors SMG's `pd_types::generate_room_id`, Dynamo's
/// `rand::random_range(0..=i64::MAX.cast_unsigned())`, and SGLang's
/// own Python-side `random.randint(0, 2**63 - 1)`.
fn generate_room_id() -> u64 {
    rand::random::<u64>() & (i64::MAX as u64)
}

/// PD-disagg bootstrap fields injected into the body forwarded to both the
/// prefill and decode workers. SGLang's HTTP disagg-prefill validator
/// requires all three as flat top-level fields:
///
/// * `host` → `bootstrap_host` — the prefill worker's hostname; decode
///   connects here for the KV transfer.
/// * `port` → `bootstrap_port` — the prefill worker's bootstrap-server port
///   (`null` when the worker is misconfigured; the engine rejects with a
///   clear error). Emitted as JSON `null`, not omitted — SGLang's validator
///   distinguishes missing from null.
/// * `room` → `bootstrap_room` — a per-request 63-bit `u64` identifying this
///   request on both prefill and decode sides.
struct BootstrapFields {
    host: String,
    port: Option<u16>,
    room: u64,
}

/// Build the body forwarded to the engine, injecting (when present) the
/// precomputed `input_ids` and/or the PD `bootstrap_*` fields into the
/// already-parsed request object and serializing once. When neither is
/// needed, returns the original bytes unchanged (no re-serialize).
///
/// `input_ids`: the router-computed prompt tokens. When set, the engine skips
/// its own chat-template tokenization; `messages` are retained in the body so
/// the engine still derives stop tokens / tool-call constraint and the OpenAI
/// response shape. The caller sets this only when the tokens are
/// engine-equivalent and the per-encoder forwarding predicate held.
///
/// `value` is the already-parsed request body when one is on hand (the
/// cache-aware path parses once at ingress); it is consumed so the mutation
/// reuses that parse. It is `None` only for a load-only policy in PD mode — a
/// path that never parses at ingress — so the bootstrap injection re-parses
/// the bytes here (matching the pre-refactor behavior). The body shape was
/// validated by `parse_probe`; the non-object arm defends against a TOCTOU
/// regression rather than panicking.
fn build_outgoing_body(
    body: &Bytes,
    value: Option<serde_json::Value>,
    input_ids: Option<&[u32]>,
    bootstrap: Option<&BootstrapFields>,
    rid: Option<&str>,
    max_tokens: Option<u64>,
) -> Result<Bytes, ApiError> {
    if input_ids.is_none() && bootstrap.is_none() && rid.is_none() && max_tokens.is_none() {
        // Nothing to inject — forward the original bytes (cheap Arc clone).
        return Ok(body.clone());
    }
    let parsed = match value {
        Some(v) => v,
        // Load-only + PD: the ingress skipped the parse, so re-parse for the
        // bootstrap injection (input_ids is never set on this path).
        None => serde_json::from_slice(body).map_err(|_| {
            ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
        })?,
    };
    let mut obj = match parsed {
        serde_json::Value::Object(map) => map,
        _ => {
            return Err(ApiError::BadRequest(
                "invalid request: body must be a JSON object".to_string(),
            ))
        }
    };
    if let Some(rid) = rid {
        // The engine adopts a provided `rid` verbatim (only minting one when
        // absent), so this is what the router later aborts by if the client
        // disconnects. Caller passes `Some` only for a router-minted rid.
        obj.insert(
            "rid".to_string(),
            serde_json::Value::String(rid.to_string()),
        );
    }
    if let Some(cap) = max_tokens {
        // Caller passes `Some` only when the request set neither
        // `max_tokens` nor `max_completion_tokens` (decided at probe time by
        // `output_budget_action`), so this never overrides a client value —
        // it defaults the output budget to the per-model cap so an
        // unbounded request can't run to the full context window.
        obj.insert(
            "max_tokens".to_string(),
            serde_json::Value::Number(cap.into()),
        );
    }
    if let Some(ids) = input_ids {
        obj.insert(
            "input_ids".to_string(),
            serde_json::Value::Array(
                ids.iter()
                    .map(|&i| serde_json::Value::Number(i.into()))
                    .collect(),
            ),
        );
    }
    if let Some(b) = bootstrap {
        obj.insert(
            "bootstrap_host".to_string(),
            serde_json::Value::String(b.host.clone()),
        );
        obj.insert(
            "bootstrap_port".to_string(),
            match b.port {
                Some(p) => serde_json::Value::Number(p.into()),
                None => serde_json::Value::Null,
            },
        );
        obj.insert(
            "bootstrap_room".to_string(),
            serde_json::Value::Number(b.room.into()),
        );
    }
    let bytes = serde_json::to_vec(&obj).map_err(|e| {
        ApiError::Internal(anyhow::Error::new(e).context("re-serialize injected request body"))
    })?;
    Ok(Bytes::from(bytes))
}

/// Select the ids to forward to the engine as `input_ids`, if any.
///
/// `Some` only when ALL of:
///   * `offload_enabled` — the `--disable-input-ids-offload` kill switch is
///     off. This is the central gate: with the offload disabled the engine
///     always re-tokenizes from `messages`, while ingress tokenization still
///     ran (routing and the cache-sim tees don't change).
///   * the ids exist and are engine-equivalent (chat-encoder path);
///   * the predicate matching the stamped [`crate::tokenizer::ForwardParity`]
///     of the ENCODER THAT PRODUCED the ids holds:
///     [`input_ids_safe_to_forward_dsv4`] for the built-in dsv4 encoder, the
///     conservative [`input_ids_safe_to_forward`] for everything else. The
///     stamp travels with the tokens, so ids can never be gated by the wrong
///     encoder's predicate.
///
/// `request_value` is `Some` exactly when the ingress parsed the body for
/// tokenization (`want_tokens` — regardless of whether that tokenization
/// succeeded), so the predicate always has a parsed body to inspect inside
/// the match.
fn select_forward_input_ids<'a>(
    offload_enabled: bool,
    request_tokens: Option<&'a RequestTokens>,
    request_value: Option<&serde_json::Value>,
) -> Option<&'a [u32]> {
    match (offload_enabled, request_tokens, request_value) {
        (true, Some(t), Some(v)) if t.engine_equivalent => match t.parity {
            crate::tokenizer::ForwardParity::Dsv4Full if input_ids_safe_to_forward_dsv4(v) => {
                Some(t.ids.as_slice())
            }
            crate::tokenizer::ForwardParity::Conservative if input_ids_safe_to_forward(v) => {
                Some(t.ids.as_slice())
            }
            _ => None,
        },
        _ => None,
    }
}

/// The dsv4-encoder forwarding predicate. The built-in encoder mirrors the
/// engine's FULL dsv4 request handling — tools and tool results, per-request
/// thinking / reasoning-effort resolution, `task` attachment, legacy
/// `functions` and per-request `chat_template` (both ignored engine-side on
/// the dsv4 path), and the trailing-assistant surgery behind
/// `continue_final_message` (see `dsv4::render_request`) — so those are all
/// forwardable. What remains withheld is exactly what the encoder cannot
/// mirror faithfully:
///
///   * content parts that are not `type: "text"` — the dsv4 engine path
///     stringifies OpenAI parts-content and silently discards non-text parts,
///     which the router replicates, but the surrounding mm-item plumbing on
///     the engine side is unverified, so anything carrying a non-text part
///     stays engine-tokenized.
///   * message-level `tools` — the one DECLARED extra that actually renders
///     (after that message's content, and it flips `drop_thinking` off).
///     `wo_eos` / `mask` / `content_blocks` / `response_format` are NOT
///     withheld: undeclared on both message models, which use pydantic's
///     default `extra="ignore"`, so they never reach the encoder.
///   * message-level `tools` — a declared protocol field the engine renders
///     after that message's content, independently of the request-level
///     tools the (mirrored) system-turn rendering uses.
///   * any role outside the engine's dsv4 render set (`system`, `user`,
///     `developer`, `assistant`, `tool`) — `latest_reminder` renders a marker
///     the encoder does not emit, and unknown roles are engine errors that
///     forwarding would silently convert into served prompts.
///   * a `continue_final_message` value that isn't pydantic-coercible — the
///     engine would 422 the request; the router must not guess false and
///     forward surgery-mismatched ids for it.
///   * a history tool call whose `arguments` is not a JSON object (inlined, or
///     a string that parses to one) — `encode_arguments_to_dsml` raises before
///     a prompt exists, so forwarding would serve a completion for a request
///     the engine rejects. BOTH accepted spellings stay forwardable; the
///     encoder renders them identically, as the engine does.
///
/// Every other field either feeds `dsv4::render_request` (mirrored,
/// including case-normalized roles) or is ignored by the engine's dsv4 path
/// entirely (identical ids either way).
///
/// NOTE (deploy): three router-side settings must match the engine's, because
/// nothing in the request lets the predicate detect a mismatch — it forwards
/// wrong-mode ids undetectably:
///   * `SGLANG_ROUTER_DSV4_DEFAULT_THINKING` /
///     `SGLANG_ROUTER_DSV4_REASONING_EFFORT` vs the engine's
///     `SGLANG_DEFAULT_THINKING` / `SGLANG_DSV4_REASONING_EFFORT`
///     (`--default-chat-template-kwargs`) — a PLAIN request (no mode fields)
///     carries no field to key on;
///   * `SGLANG_ROUTER_DSV4_REASONING_EFFORT_PROFILE` vs the profile the engine
///     resolves from the CHECKPOINT (`chat_encoding.resolve_dsv4_reasoning_
///     effort_profile`, defaulting to `preview` when detection fails). The
///     router defaults to `official`; a preview-era checkpoint needs the env
///     set to `preview`, and a `reasoning_effort` of `high`/`max` carries no
///     signal distinguishing the two profiles.
fn input_ids_safe_to_forward_dsv4(value: &serde_json::Value) -> bool {
    if let Some(v) = value.get("continue_final_message").filter(|v| !v.is_null()) {
        if crate::tokenizer::openai_bool(v).is_none() {
            return false;
        }
    }
    let Some(msgs) = value.get("messages").and_then(|m| m.as_array()) else {
        return true;
    };
    for m in msgs {
        // Only message-level `tools` is a DECLARED protocol field, so only it
        // can reach the encoder. `wo_eos` / `mask` / `content_blocks` /
        // `response_format` are undeclared on both message models, which use
        // pydantic's default `extra="ignore"`, so they are stripped before
        // `model_dump()` — withholding them bought nothing and cost coverage.
        if m.get("tools").is_some_and(|v| !v.is_null()) {
            return false;
        }
        // The engine's dsv4 render set. Compared case-insensitively to match
        // `render_request`'s lowercasing; note only the six GENERIC roles are
        // case-normalized protocol-side (`user` is a bare Literal), so a
        // case-varied `user` is a 422 rather than a role this ever sees.
        let role_ok = m
            .get("role")
            .and_then(|r| r.as_str())
            .map(|r| r.to_ascii_lowercase())
            .is_none_or(|r| {
                matches!(
                    r.as_str(),
                    "system" | "user" | "developer" | "assistant" | "tool"
                )
            });
        if !role_ok {
            return false;
        }
        if let Some(parts) = m.get("content").and_then(|c| c.as_array()) {
            let has_non_text_part = parts
                .iter()
                .any(|p| p.get("type").and_then(|t| t.as_str()) != Some("text"));
            if has_non_text_part {
                return false;
            }
        }
        // History tool calls: the engine requires `arguments` to BE a JSON
        // object — inlined, or a string it can `json.loads` into one. Anything
        // else raises (`JSONDecodeError`, or its `must be a JSON object`
        // ValueError) before a prompt exists, so forwarding would serve a
        // completion for a request the engine rejects.
        if let Some(calls) = m.get("tool_calls").and_then(|t| t.as_array()) {
            let all_object_args = calls.iter().all(|c| {
                match c.get("function").and_then(|f| f.get("arguments")) {
                    Some(serde_json::Value::String(s)) => {
                        serde_json::from_str::<serde_json::Value>(s).is_ok_and(|v| v.is_object())
                    }
                    Some(serde_json::Value::Object(_)) => true,
                    // Absent / null / scalar: `arguments` is Optional, so this
                    // validates, then fails the engine's object check.
                    _ => false,
                }
            });
            if !all_object_args {
                return false;
            }
        }
    }
    true
}

/// Whether the router's `input_ids` may be forwarded for this request — the
/// predicate for NON-dsv4 chat encoders (today: the generic Jinja path). The
/// dsv4 encoder has its own, far less conservative predicate
/// [`input_ids_safe_to_forward_dsv4`]; selection between them happens in
/// [`select_forward_input_ids`] via the ids' stamped parity.
///
/// We forward only when the engine, fed `input_ids`, would have produced the
/// SAME prompt the router tokenized. When `input_ids` is present the engine
/// uses it verbatim and ignores everything that would otherwise steer its
/// `messages`-side tokenization (only stop tokens / tool-call constraint are
/// still taken from `messages`). So any request field that changes that
/// tokenization but which the router's chat encoder does not replicate makes
/// the forwarded ids wrong. This predicate is conservative by construction —
/// any such signal returns `false` and the engine tokenizes from `messages`
/// (always correct).
///
/// Replicated-and-safe: plain text `messages` with a string `content`.
/// Not replicated → omit:
///   * `tools` / `functions` — the generic Jinja encoder renders only
///     `messages`; its ids would omit the tool schemas the engine's template
///     injects into the prompt.
///   * multimodal (array) `content` — a text tokenizer can't represent images.
///   * `chat_template` — an OpenAI-compatible per-request template override
///     (e.g. vLLM); the router renders with the model's default template, so a
///     custom one would diverge.
///   * `chat_template_kwargs` (carries `enable_thinking`/`thinking`),
///     `reasoning` / `reasoning_effort`, `task` — thinking/mode toggles the
///     Jinja encoder renders in the model's default mode only.
///   * `continue_final_message: true`, or a trailing `assistant` message — the
///     engine rewrites/strips the final assistant turn; the Jinja encoder
///     renders it verbatim.
///
/// One parity assumption the predicate can't check: the router renders
/// specials via the template, matching the engine on tokenizers that auto-add
/// them (the common case); a model whose engine does not would diverge by a
/// leading special.
fn input_ids_safe_to_forward(value: &serde_json::Value) -> bool {
    if request_has_tools(value) || request_is_multimodal(value) {
        return false;
    }
    // Fields that steer the engine's template tokenization but which the
    // router's encoder does not thread through.
    for key in [
        "chat_template",
        "chat_template_kwargs",
        "reasoning",
        "reasoning_effort",
        "task",
    ] {
        if value.get(key).is_some_and(|v| !v.is_null()) {
            return false;
        }
    }
    if value
        .get("continue_final_message")
        .filter(|v| !v.is_null())
        .is_some_and(|v| crate::tokenizer::openai_bool(v) != Some(false))
    {
        return false;
    }
    !last_message_is_assistant(value)
}

/// Whether the ingress tokenization offload was expected to fire but failed —
/// the condition behind `sgl_router_ingress_tokenize_errors_total`.
///
/// True only when ALL of:
///   * the model has a chat encoder (`has_chat_encoder`), so a chat request
///     on it SHOULD have produced engine-equivalent ids;
///   * the request is a chat request (`messages` array present);
///   * the tokens are absent OR not engine-equivalent — i.e. `encode_chat`
///     render/encode failed and the request silently fell back to engine-side
///     tokenization.
///
/// Non-chat-encoder / non-`messages` requests never expected the offload, so
/// they are not failures. Requests withheld by the safe-predicates (see
/// `input_ids_safe_to_forward_dsv4` / `input_ids_safe_to_forward`) still get
/// engine-equivalent ids (`encode_chat` succeeded; the predicate withholds
/// forwarding for other reasons), so they too are expected omissions, not
/// failures.
fn ingress_tokenize_offload_failed(
    has_chat_encoder: bool,
    request_value: Option<&serde_json::Value>,
    request_tokens: Option<&RequestTokens>,
) -> bool {
    if !has_chat_encoder {
        return false;
    }
    let chat_request =
        request_value.is_some_and(|v| v.get("messages").is_some_and(|m| m.is_array()));
    if !chat_request {
        return false;
    }
    !request_tokens.is_some_and(|t| t.engine_equivalent)
}

/// Whether the dsv4 render pipeline rejects this request outright — the
/// cases [`crate::tokenizer::dsv4::render_request`] errors on (invalid task,
/// task without any user/developer message). Such requests are rejected by
/// the ENGINE identically, so the ingress-encode fallback here is not
/// offload breakage and must not count in
/// `sgl_router_ingress_tokenize_errors_total`. Only invoked on the failure
/// path, where the re-render cost is irrelevant.
fn dsv4_render_rejects_request(
    tokenizers: &crate::tokenizer::TokenizerRegistry,
    model_id: &str,
    value: Option<&serde_json::Value>,
) -> bool {
    if tokenizers.forward_parity(model_id) != crate::tokenizer::ForwardParity::Dsv4Full {
        return false;
    }
    let Some(v) = value else { return false };
    let Some(messages) = v.get("messages").filter(|m| m.is_array()) else {
        return false;
    };
    let opts = crate::tokenizer::dsv4::resolve_render_opts(v);
    let parts = crate::tokenizer::dsv4::RequestParts {
        task: v.get("task").and_then(|t| t.as_str()),
        continue_final_message: v
            .get("continue_final_message")
            .and_then(crate::tokenizer::openai_bool)
            == Some(true),
    };
    crate::tokenizer::dsv4::render_request(messages, v.get("tools"), opts, parts).is_err()
}

/// Whether the final chat message has `role: "assistant"` (a prefix /
/// continuation turn the engine's template path special-cases).
fn last_message_is_assistant(value: &serde_json::Value) -> bool {
    value
        .get("messages")
        .and_then(|m| m.as_array())
        .and_then(|msgs| msgs.last())
        .and_then(|m| m.get("role"))
        .and_then(|r| r.as_str())
        == Some("assistant")
}

/// Whether the request carries tool / function definitions. The router's chat
/// encoder renders only `messages`, so its `input_ids` would omit the tool
/// schemas the engine's template injects into the prompt — the caller must let
/// the engine tokenize these itself.
fn request_has_tools(value: &serde_json::Value) -> bool {
    let nonempty = |key: &str| {
        value.get(key).is_some_and(|v| match v {
            serde_json::Value::Array(a) => !a.is_empty(),
            serde_json::Value::Null => false,
            _ => true,
        })
    };
    nonempty("tools") || nonempty("functions")
}

/// Whether any message carries non-string (array / multimodal) content. A text
/// tokenizer cannot represent image content, so the router's `input_ids` would
/// drop it — the caller must let the engine handle these requests.
fn request_is_multimodal(value: &serde_json::Value) -> bool {
    value
        .get("messages")
        .and_then(|m| m.as_array())
        .is_some_and(|msgs| {
            msgs.iter()
                .any(|m| matches!(m.get("content"), Some(serde_json::Value::Array(_))))
        })
}

/// Resolve the per-model output-token contract (`--max-output-tokens`)
/// against what the request asked for.
///
/// Returns the value to inject as `max_tokens` into the forwarded body
/// (`Some(cap)` exactly when the cap is configured and the request set no
/// effective output budget), or a 400 when the request explicitly asked
/// for more than the cap.
///
/// The effective value mirrors the engine's resolution (protocol.py:
/// `max_completion_tokens or max_tokens` — Python `or`, where an explicit
/// numeric `0` is falsy): `max_completion_tokens` wins when present and
/// non-zero, otherwise `max_tokens`. Diverging here would open a bypass —
/// e.g. a legal `max_completion_tokens: 0` shadowing an over-cap
/// `max_tokens` that the engine would then actually use.
///
/// Values are read through [`lax_number`] — not `as_u64` — for the same
/// reason: the engine's pydantic lax mode coerces integral floats
/// (`999999.0`) and numeric strings (`"999999"`) to ints, so a stricter
/// read here could be slipped past. Everything `lax_number` can't read
/// (non-numeric string, bool, …) neither rejects nor injects: it forwards
/// untouched so the engine — authoritative for the request schema —
/// produces its own 4xx with the better message.
fn output_budget_action(
    cap: Option<std::num::NonZeroU64>,
    probe: &RequestProbe,
) -> Result<Option<u64>, ApiError> {
    let Some(cap) = cap else {
        return Ok(None);
    };
    let requested = probe
        .max_completion_tokens
        .as_ref()
        .filter(|v| lax_number(v) != Some(0.0))
        .or(probe.max_tokens.as_ref());
    match requested {
        None => Ok(Some(cap.get())),
        Some(v) => {
            if let Some(f) = lax_number(v) {
                if f > cap.get() as f64 {
                    return Err(ApiError::BadRequest(format!(
                        "max_tokens is too large: {v}. This model supports at most \
                         {cap} completion tokens."
                    )));
                }
            }
            Ok(None)
        }
    }
}

/// Read a JSON value the way the engine's pydantic lax mode reads an int
/// field: numbers directly, numeric strings by parsing. Returns `None` for
/// anything pydantic would reject outright (non-numeric string, bool,
/// array, …). Non-integral floats parse here but fail pydantic's int
/// coercion — harmless for the cap check: an over-cap `1e9` gets our 400
/// instead of reaching the engine, an under-cap `1.5` forwards and gets
/// the engine's 4xx.
fn lax_number(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// Derive this request's id: reuse a client-supplied `rid` (so an external
/// abort-by-rid keeps working and we do not override intent), else mint one
/// from the gateway/access-log `x-request-id` — read the same way as the
/// global access-log middleware (`server/app.rs`) — else a UUID.
///
/// Deriving from `x-request-id` rather than an unrelated UUID is what keeps
/// the router/gateway logs (keyed on that header), the engine logs (keyed on
/// the rid we forward), and the cache-sim oracle's per-request records (keyed
/// on the id the tee sends) all pointing at the same request. A fresh UUID
/// here would leave three disjoint identifiers for one request.
///
/// One exception: in PD-disagg mode no rid is forwarded (see `request_id`'s
/// `decode_peer` gate), so the engine leg of that correlation does not exist
/// and the engine mints its own. The gateway and oracle legs still line up.
///
/// SGLang keeps a provided `rid` and only generates one when absent, and it
/// aborts every rid that *starts with* the one we send — covering `n>1`
/// expansions.
///
/// Called ONCE per request, before the ingress tee, and the result reused for
/// the engine-facing rid: two call sites would each mint their own UUID on the
/// no-header path, and the tee's id would then not match the engine's.
fn derive_request_id(client_rid: Option<&str>, headers: &HeaderMap) -> String {
    // Empty is rejected at both sources, the way the sticky-routing key in this
    // same handler already does. An empty `rid` or a bare `x-request-id:`
    // header would otherwise collapse every affected request onto the single
    // key "" / "router-", turning any downstream join into a cross product.
    if let Some(rid) = client_rid.filter(|s| !s.is_empty()) {
        return rid.to_owned();
    }
    match headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
    {
        Some(id) => format!("router-{id}"),
        None => format!("router-{}", uuid::Uuid::new_v4().simple()),
    }
}

/// Read the upstream's per-request attribution headers off the incoming request
/// so the ingress tee can re-forward them verbatim to the cache-sim (an upstream
/// gateway stamps them; the cache-sim receiver reads them back into each record).
/// Pure pass-through — the router mints nothing here, it only relays what the
/// upstream resolved. Empty headers are treated as absent (same contract as
/// `derive_request_id`), so a blank never masquerades as a real
/// endpoint/key/slug/correlation.
fn tee_attribution(headers: &HeaderMap) -> crate::server::cache_sim_tee::Attribution {
    let get = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    };
    crate::server::cache_sim_tee::Attribution {
        correlation_id: get("x-radixark-correlation-id"),
        endpoint_id: get("x-radixark-endpoint-id"),
        key_id: get("x-radixark-key-id"),
        slug: get("x-radixark-endpoint-slug"),
    }
}

fn parse_probe(body: &Bytes) -> Result<RequestProbe, ApiError> {
    // We deliberately do NOT echo the serde error into the client-visible
    // message — that risks leaking field-level detail and is also of little
    // help to a real client (which already has its own JSON validator).
    // Server-side, the full error is logged with `tracing::debug!` for
    // operator triage.
    //
    // Two-step deserialize:
    //   1. `Map<String, IgnoredAny>` *anchors* the shape to a JSON object.
    //      This rejects `null` / `[]` / `"hi"` (all valid JSON but not
    //      request shape) without walking the full value into a
    //      `serde_json::Value` per field.
    //   2. `RequestProbe` (struct of `Option<bool>` + `Option<String>`)
    //      lifts out only the fields we care about — `stream` and `model`.
    //      Other fields are ignored; the worker is authoritative for the
    //      rest of the schema.
    let _: HashMap<String, IgnoredAny> = serde_json::from_slice(body).map_err(|e| {
        tracing::debug!(error = %e, "chat-completions body rejected as non-object JSON");
        ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
    })?;
    let probe: RequestProbe = serde_json::from_slice(body).map_err(|e| {
        tracing::debug!(error = %e, "chat-completions request-probe deserialize failed");
        ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
    })?;
    Ok(probe)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use axum::http::StatusCode;
    use reqwest::Url;

    fn retry_cfg(max_target_itl_ms: Option<u64>, itl_rel_factor: f32) -> RetryConfig {
        RetryConfig {
            enabled: true,
            max_target_itl_ms,
            itl_rel_factor: Some(itl_rel_factor),
            attempt_deadline_ms: None,
        }
    }

    #[test]
    fn itl_gate_off_accepts_everything() {
        // No ceiling → gate disabled → any target passes, even a very slow one.
        let cfg = retry_cfg(None, 1.0);
        assert!(itl_target_eligible(&cfg, Some(9999.0), Some(10.0)));
        assert!(itl_target_eligible(&cfg, None, None));
    }

    #[test]
    fn itl_ceiling_excludes_hot_targets_but_not_unknown() {
        let cfg = retry_cfg(Some(100), 1.0);
        assert!(
            !itl_target_eligible(&cfg, Some(150.0), None),
            "over ceiling → excluded"
        );
        assert!(
            itl_target_eligible(&cfg, Some(80.0), None),
            "under ceiling → eligible"
        );
        // Unknown target ITL must NOT be excluded — missing data can't block failover.
        assert!(itl_target_eligible(&cfg, None, Some(10.0)));
    }

    #[test]
    fn itl_relative_gate_requires_target_no_worse_than_source() {
        // Ceiling high so the absolute check never bites — isolates the relative
        // gate (which only engages because a ceiling is set).
        let cfg = retry_cfg(Some(1000), 1.0);
        assert!(
            !itl_target_eligible(&cfg, Some(120.0), Some(100.0)),
            "slower than source → excluded"
        );
        assert!(
            itl_target_eligible(&cfg, Some(90.0), Some(100.0)),
            "faster than source → eligible"
        );
        // Relative gate only applies when BOTH are known.
        assert!(itl_target_eligible(&cfg, Some(120.0), None));
    }

    #[test]
    fn itl_relative_factor_loosens_the_bound() {
        let cfg = retry_cfg(Some(1000), 2.0);
        assert!(itl_target_eligible(&cfg, Some(180.0), Some(100.0)));
        assert!(!itl_target_eligible(&cfg, Some(210.0), Some(100.0)));
    }

    /// Every `ApiError` variant must map deterministically to exactly one
    /// `AbortReason`. This test pins the mapping so:
    ///   * A future PR adding a new `ApiError` variant is forced to think
    ///     about which abort reason it belongs to (the wildcard `_ =>
    ///     TransportError` in `abort_reason_from_api_error` catches it, but
    ///     silently — this test freezes the intended default).
    ///   * A typo swap (e.g. `UpstreamTimeout → StaleRequestExpired`) fails
    ///     loudly here even though both are timeout-family and would
    ///     otherwise pass a code review.
    ///
    /// Kept as one function with a table of cases so adding a new
    /// `ApiError` variant fails one assertion instead of one whole test —
    /// the diff shows the exact new row that needed a decision.
    #[test]
    fn abort_reason_from_api_error_covers_every_variant() {
        // sentinel worker URL for the variants that carry one
        let worker_url = Url::parse("http://w:1/").unwrap();
        let cases: Vec<(ApiError, AbortReason)> = vec![
            // Explicitly-mapped narrow reasons — these are the informative
            // labels the whole change exists to surface.
            (
                ApiError::UpstreamTimeout {
                    worker: worker_url.clone(),
                },
                AbortReason::UpstreamTimeout,
            ),
            (
                ApiError::UpstreamSocketTimeout {
                    worker: worker_url.clone(),
                    source: anyhow!("kernel ETIMEDOUT"),
                },
                AbortReason::UpstreamSocketTimeout,
            ),
            (
                ApiError::StaleRequestExpired {
                    model: "m".into(),
                    worker: Url::parse("http://test-worker/").unwrap(),
                },
                AbortReason::StaleRequestExpired,
            ),
            (
                ApiError::AttemptTimeout { model: "m".into() },
                AbortReason::UpstreamTimeout,
            ),
            // Every other variant falls through to `TransportError`. Each
            // row is a decision — do not silently accept a default without
            // considering whether a distinct label would be more useful.
            (
                ApiError::BadRequest("x".into()),
                AbortReason::TransportError,
            ),
            (
                ApiError::ModelNotFound("m".into()),
                AbortReason::TransportError,
            ),
            (
                ApiError::UpstreamUnreachable {
                    worker: worker_url.clone(),
                    source: anyhow!("unreachable"),
                },
                AbortReason::TransportError,
            ),
            (
                ApiError::UpstreamStatus {
                    status: StatusCode::BAD_GATEWAY,
                    worker: worker_url.clone(),
                },
                AbortReason::TransportError,
            ),
            (
                ApiError::NoHealthyWorkers { model: "m".into() },
                AbortReason::TransportError,
            ),
            (
                ApiError::NoPrefillWorkersAvailable { model: "m".into() },
                AbortReason::TransportError,
            ),
            (
                ApiError::NoDecodeWorkersAvailable { model: "m".into() },
                AbortReason::TransportError,
            ),
            (
                ApiError::PolicySelectionFailed { model: "m".into() },
                AbortReason::TransportError,
            ),
            (
                ApiError::BreakerOpen {
                    worker: "http://w".into(),
                },
                AbortReason::TransportError,
            ),
            (
                ApiError::WorkerMisconfigured {
                    worker: "http://w".into(),
                    source: anyhow!("bad url"),
                },
                AbortReason::TransportError,
            ),
            (
                ApiError::ServiceOverloaded { model: "m".into() },
                AbortReason::TransportError,
            ),
            (
                ApiError::Internal(anyhow!("boom")),
                AbortReason::TransportError,
            ),
        ];
        for (err, expected) in cases.iter() {
            let got = abort_reason_from_api_error(err);
            assert_eq!(
                got, *expected,
                "abort_reason_from_api_error({err}) — expected {:?}, got {:?}. \
                 If you changed the mapping, update the expected value; \
                 if you added a new ApiError variant, add a row here to pin \
                 which AbortReason it maps to.",
                expected, got,
            );
        }
        // Coverage check. The net is `variant_tag`'s wildcard-free match, NOT
        // the count below: adding an `ApiError` variant fails to COMPILE until
        // someone adds an arm there, and whoever does is then looking straight
        // at the table that needs a row.
        //
        // The previous net was `assert_eq!(cases.len(), <hand-written const>)`,
        // which a forgotten update leaves stale on BOTH sides — it passed
        // silently when `UpstreamSocketTimeout` was added, the exact case it
        // was written to catch. Deduping tags keeps the count honest even if a
        // row is duplicated instead of added.
        let mut tags: Vec<u8> = cases.iter().map(|(e, _)| variant_tag(e)).collect();
        tags.sort_unstable();
        tags.dedup();
        assert_eq!(
            tags.len(),
            cases.len(),
            "the table has duplicate rows for the same ApiError variant",
        );
        assert_eq!(
            tags.len(),
            APIERROR_VARIANTS,
            "abort_reason_from_api_error test table covers {} distinct ApiError \
             variants; the enum has {} — add the missing row(s).",
            tags.len(),
            APIERROR_VARIANTS,
        );
    }

    /// Number of `ApiError` variants. Kept honest by [`variant_tag`], whose
    /// exhaustive match cannot compile if the enum grows.
    const APIERROR_VARIANTS: usize = 16;

    /// Wildcard-free variant tag, existing purely so the COMPILER enforces
    /// coverage of `abort_reason_from_api_error`'s test table. `_ =>` here
    /// would defeat the entire point — `abort_reason_from_api_error` itself
    /// ends in a catch-all, so this match is the only thing that forces a new
    /// variant to be consciously classified.
    fn variant_tag(e: &ApiError) -> u8 {
        match e {
            ApiError::BadRequest(_) => 0,
            ApiError::ModelNotFound(_) => 1,
            ApiError::UpstreamUnreachable { .. } => 2,
            ApiError::UpstreamStatus { .. } => 3,
            ApiError::UpstreamTimeout { .. } => 4,
            ApiError::UpstreamSocketTimeout { .. } => 5,
            ApiError::NoHealthyWorkers { .. } => 6,
            ApiError::NoPrefillWorkersAvailable { .. } => 7,
            ApiError::NoDecodeWorkersAvailable { .. } => 8,
            ApiError::StaleRequestExpired { .. } => 9,
            ApiError::AttemptTimeout { .. } => 10,
            ApiError::PolicySelectionFailed { .. } => 11,
            ApiError::BreakerOpen { .. } => 12,
            ApiError::WorkerMisconfigured { .. } => 13,
            ApiError::ServiceOverloaded { .. } => 14,
            ApiError::Internal(_) => 15,
        }
    }

    /// `generate_room_id` MUST return values in `[0, i64::MAX]`. The
    /// SGLang prefill stores `bootstrap_room` as `torch.int64`; a u64
    /// with the top bit set would wrap negative on the engine side.
    /// Sample many times to defend against future refactors of the
    /// mask (e.g. someone "simplifying" to plain `rand::random::<u64>()`).
    #[test]
    fn generate_room_id_stays_in_63_bit_range() {
        for _ in 0..10_000 {
            let r = generate_room_id();
            assert!(
                r <= i64::MAX as u64,
                "generate_room_id() returned {r} > i64::MAX; would wrap negative as torch.int64",
            );
        }
    }

    /// When the prefill worker has no `bootstrap_port` configured
    /// (a misconfiguration the engine will reject loudly), the
    /// injected field MUST be JSON `null` — not omitted, not 0.
    /// SGLang's validator distinguishes "missing field" from
    /// "null field" in some code paths.
    #[test]
    fn build_outgoing_body_emits_null_for_missing_port() {
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let bootstrap = BootstrapFields {
            host: "host".into(),
            port: None,
            room: 42,
        };
        let injected =
            build_outgoing_body(&body, Some(value), None, Some(&bootstrap), None, None).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&injected).unwrap();
        assert_eq!(parsed.get("bootstrap_port"), Some(&serde_json::Value::Null));
        assert_eq!(
            parsed.get("bootstrap_host"),
            Some(&serde_json::Value::String("host".into()))
        );
        assert_eq!(
            parsed.get("bootstrap_room"),
            Some(&serde_json::Value::Number(42.into()))
        );
    }

    /// The central kill switch: with the offload disabled, engine-equivalent
    /// ids for a perfectly forwardable plain-text chat are still withheld, so
    /// the engine always re-tokenizes from `messages`.
    #[test]
    fn select_forward_input_ids_disabled_gate_withholds_ids() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        let tokens = RequestTokens {
            ids: vec![1, 2, 3],
            engine_equivalent: true,
            parity: crate::tokenizer::ForwardParity::Conservative,
        };
        assert_eq!(
            select_forward_input_ids(false, Some(&tokens), Some(&value)),
            None
        );
    }

    /// With the offload enabled, plain-text chat with engine-equivalent ids
    /// forwards them.
    #[test]
    fn select_forward_input_ids_enabled_forwards_engine_equivalent_ids() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        let tokens = RequestTokens {
            ids: vec![1, 2, 3],
            engine_equivalent: true,
            parity: crate::tokenizer::ForwardParity::Conservative,
        };
        assert_eq!(
            select_forward_input_ids(true, Some(&tokens), Some(&value)),
            Some(&[1, 2, 3][..])
        );
    }

    /// Non-engine-equivalent ids (the raw-prompt path — a model with no chat
    /// encoder, or a chat-encode failure that fell through to raw text) are
    /// withheld even with the offload enabled and a plain body — the
    /// `engine_equivalent` conjunct, so this gate can never forward ids the
    /// engine wouldn't have produced itself.
    #[test]
    fn select_forward_input_ids_withholds_non_engine_equivalent_ids() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        let tokens = RequestTokens {
            ids: vec![1, 2, 3],
            engine_equivalent: false,
            parity: crate::tokenizer::ForwardParity::Conservative,
        };
        assert_eq!(
            select_forward_input_ids(true, Some(&tokens), Some(&value)),
            None
        );
    }

    /// Enabling the offload does NOT bypass the per-request safety predicate:
    /// unreplicated signals (tools, here) still withhold the ids when the ids
    /// came from the conservative (Jinja) encoder path.
    #[test]
    fn select_forward_input_ids_enabled_still_respects_safe_predicate() {
        let value = serde_json::json!({
            "messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f"}}]
        });
        let tokens = RequestTokens {
            ids: vec![1, 2, 3],
            engine_equivalent: true,
            parity: crate::tokenizer::ForwardParity::Conservative,
        };
        assert_eq!(
            select_forward_input_ids(true, Some(&tokens), Some(&value)),
            None
        );
    }

    /// The stamped parity dispatches the predicate: the SAME tool-bearing
    /// body is withheld for Conservative-stamped ids (Jinja renders no tool
    /// schemas) but forwarded for Dsv4Full-stamped ids (the dsv4 encoder
    /// mirrors tool rendering).
    #[test]
    fn select_forward_input_ids_dispatches_predicate_by_stamped_parity() {
        let value = serde_json::json!({
            "messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f"}}]
        });
        let conservative = RequestTokens {
            ids: vec![1, 2, 3],
            engine_equivalent: true,
            parity: crate::tokenizer::ForwardParity::Conservative,
        };
        assert_eq!(
            select_forward_input_ids(true, Some(&conservative), Some(&value)),
            None
        );
        let dsv4 = RequestTokens {
            ids: vec![1, 2, 3],
            engine_equivalent: true,
            parity: crate::tokenizer::ForwardParity::Dsv4Full,
        };
        assert_eq!(
            select_forward_input_ids(true, Some(&dsv4), Some(&value)),
            Some(&[1, 2, 3][..])
        );
    }

    /// The dsv4 predicate allows exactly the classes the dsv4 encoder mirrors
    /// (or the engine ignores) — tools, thinking overrides, reasoning_effort,
    /// task, trailing-assistant/continue-final-message, functions, per-request
    /// chat_template — so a regression back toward the conservative set would
    /// silently inert the offload for them.
    #[test]
    fn input_ids_safe_to_forward_dsv4_allows_mirrored_classes() {
        for body in [
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],
                               "tools":[{"type":"function","function":{"name":"f"}}]}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],
                               "functions":[{"name":"f"}]}),
            // BOTH `arguments` spellings the engine accepts and renders
            // identically (JSON string, and the inlined object its type
            // permits) stay forwardable.
            serde_json::json!({"messages":[{"role":"assistant","tool_calls":[
                 {"function":{"name":"f","arguments":"{\"a\": 1}"}}]}]}),
            serde_json::json!({"messages":[{"role":"assistant","tool_calls":[
                 {"function":{"name":"f","arguments":{"a":1}}}]}]}),
            // Now mirrored / provably inert, so all of these forward:
            // the `reasoning` OBJECT (both alias spellings and `enabled`),
            // and the undeclared message keys pydantic strips.
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],
                               "reasoning":{"effort":"high"}}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],
                               "reasoning":{"enabled":true,"effort":"max"}}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],
                               "reasoning":{"enabled":false}}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"reasoning":false}),
            serde_json::json!({"messages":[{"role":"user","content":"hi","wo_eos":true}]}),
            serde_json::json!({"messages":[{"role":"user","content":"hi","mask":true}]}),
            serde_json::json!({"messages":[{"role":"system","content":"s",
                               "response_format":{"type":"json_object"}}]}),
            // Tool-level `defer_loading` is now mirrored, so it stays allowed.
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],
                               "tools":[{"type":"function","defer_loading":true,
                                         "function":{"name":"f"}}]}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],
                               "chat_template_kwargs":{"thinking":true}}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],
                               "reasoning_effort":"high"}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"task":"query"}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"},
                                           {"role":"assistant","content":"partial"}],
                               "continue_final_message":true}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"},
                                           {"role":"assistant","content":"partial"}]}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],
                               "chat_template":"custom"}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],
                               "continue_final_message":"true"}), // pydantic-coercible form
            // Numeric content and a case-varied role are both 422s at the
            // protocol boundary (pydantic v2 does not coerce int->str, and
            // only the six GENERIC roles have a case-normalizer — `user` is a
            // bare Literal). They are listed as ALLOWED because the predicate
            // need not withhold what can never reach the engine; the ids are
            // simply never built for a request the engine rejects.
            serde_json::json!({"messages":[{"role":"user","content":42}]}),
            serde_json::json!({"messages":[{"role":"User","content":"hi"}]}),
        ] {
            assert!(
                input_ids_safe_to_forward_dsv4(&body),
                "dsv4 predicate must allow: {body}"
            );
        }
    }

    /// …and blocks exactly what the encoder cannot mirror: non-text content
    /// parts (or parts with no type), the engine-internal message keys
    /// (`wo_eos`, `mask`, `content_blocks`), message-level `tools`, unmirrored
    /// roles, the `reasoning` object, and an uncoercible
    /// `continue_final_message`.
    #[test]
    fn input_ids_safe_to_forward_dsv4_blocks_unmirrored_classes() {
        for body in [
            serde_json::json!({"messages":[{"role":"user",
                 "content":[{"type":"image_url","image_url":"x"}]}]}),
            serde_json::json!({"messages":[{"role":"user",
                 "content":[{"text":"no type key"}]}]}),
            // `arguments` shapes the engine rejects before a prompt exists:
            // unparsable, parsing to a non-object, a scalar, and absent.
            serde_json::json!({"messages":[{"role":"assistant","tool_calls":[
                 {"function":{"name":"f","arguments":"not json"}}]}]}),
            serde_json::json!({"messages":[{"role":"assistant","tool_calls":[
                 {"function":{"name":"f","arguments":"[1, 2]"}}]}]}),
            serde_json::json!({"messages":[{"role":"assistant","tool_calls":[
                 {"function":{"name":"f","arguments":5}}]}]}),
            serde_json::json!({"messages":[{"role":"assistant","tool_calls":[
                 {"function":{"name":"f"}}]}]}),
            serde_json::json!({"messages":[{"role":"system","content":"hi",
                 "tools":[{"type":"function","function":{"name":"f"}}]}]}),
            serde_json::json!({"messages":[{"role":"latest_reminder","content":"hi"}]}),
            serde_json::json!({"messages":[{"role":"function","content":"hi"}]}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],
                 "continue_final_message":"maybe"}),
        ] {
            assert!(
                !input_ids_safe_to_forward_dsv4(&body),
                "dsv4 predicate must block: {body}"
            );
        }
        // null/absent forms are tolerated (treated as not-present).
        for body in [
            serde_json::json!({"messages":[{"role":"user","content":"hi","wo_eos":null}]}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],
                 "continue_final_message":null}),
        ] {
            assert!(
                input_ids_safe_to_forward_dsv4(&body),
                "null forms must not block: {body}"
            );
        }
    }

    /// `input_ids` are injected and `messages` retained (the engine still
    /// needs them for stop tokens / tool-call constraint / response shape).
    #[test]
    fn build_outgoing_body_injects_input_ids_and_keeps_messages() {
        let body =
            Bytes::from_static(br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let ids = [1u32, 2, 3];
        let out = build_outgoing_body(&body, Some(value), Some(&ids), None, None, None).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(parsed.get("input_ids"), Some(&serde_json::json!([1, 2, 3])));
        assert!(
            parsed.get("messages").is_some(),
            "messages must be retained alongside input_ids"
        );
    }

    /// With nothing to inject, the original bytes are forwarded unchanged
    /// (no re-serialize) — the transparent no-op fallback.
    #[test]
    fn build_outgoing_body_no_injection_returns_original_bytes() {
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let out = build_outgoing_body(&body, Some(value), None, None, None, None).unwrap();
        assert_eq!(
            out, body,
            "no injection must forward the original bytes unchanged"
        );
    }

    /// A `max_tokens` default is injected as a top-level number — including
    /// on the path where nothing else needs injection (the early-return must
    /// not swallow it).
    #[test]
    fn build_outgoing_body_injects_default_max_tokens() {
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let out = build_outgoing_body(&body, None, None, None, None, Some(131072)).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(parsed.get("max_tokens"), Some(&serde_json::json!(131072)));
        assert!(parsed.get("messages").is_some());
    }

    fn probe_of(body: &str) -> RequestProbe {
        parse_probe(&Bytes::copy_from_slice(body.as_bytes())).unwrap()
    }

    fn cap(n: u64) -> Option<std::num::NonZeroU64> {
        Some(std::num::NonZeroU64::new(n).unwrap())
    }

    /// No cap configured → never rejects, never injects, regardless of what
    /// the request asked for.
    #[test]
    fn output_budget_no_cap_is_passthrough() {
        let p = probe_of(r#"{"model":"x","max_tokens":999999999}"#);
        assert_eq!(output_budget_action(None, &p).unwrap(), None);
    }

    /// Neither field set → inject the cap.
    #[test]
    fn output_budget_injects_cap_when_unset() {
        let p = probe_of(r#"{"model":"x"}"#);
        assert_eq!(output_budget_action(cap(131072), &p).unwrap(), Some(131072));
        // An explicit `null` is treated the same as absent (engine parity).
        let p = probe_of(r#"{"model":"x","max_tokens":null}"#);
        assert_eq!(output_budget_action(cap(131072), &p).unwrap(), Some(131072));
    }

    /// A legal explicit ask (≤ cap, boundary included) forwards untouched —
    /// no rejection, no injection.
    #[test]
    fn output_budget_legal_explicit_value_forwards_untouched() {
        let p = probe_of(r#"{"model":"x","max_tokens":131072}"#);
        assert_eq!(output_budget_action(cap(131072), &p).unwrap(), None);
        let p = probe_of(r#"{"model":"x","max_completion_tokens":42}"#);
        assert_eq!(output_budget_action(cap(131072), &p).unwrap(), None);
    }

    /// An explicit ask above the cap is rejected with a 400 naming both the
    /// asked-for value and the cap.
    #[test]
    fn output_budget_rejects_over_cap() {
        let p = probe_of(r#"{"model":"x","max_tokens":131073}"#);
        let err = output_budget_action(cap(131072), &p).unwrap_err();
        assert!(matches!(&err, ApiError::BadRequest(m)
            if m.contains("131073") && m.contains("131072")));
    }

    /// `max_completion_tokens` wins over the deprecated `max_tokens` when
    /// both are present — same precedence as the engine.
    #[test]
    fn output_budget_max_completion_tokens_takes_precedence() {
        // Over-cap max_tokens is ignored because max_completion_tokens is legal.
        let p = probe_of(r#"{"model":"x","max_tokens":999999,"max_completion_tokens":100}"#);
        assert_eq!(output_budget_action(cap(131072), &p).unwrap(), None);
        // And the reverse: over-cap max_completion_tokens rejects even when
        // max_tokens is legal.
        let p = probe_of(r#"{"model":"x","max_tokens":100,"max_completion_tokens":999999}"#);
        assert!(output_budget_action(cap(131072), &p).is_err());
    }

    /// Engine parity for Python-`or` falsiness: a numeric-zero
    /// `max_completion_tokens` (0 or 0.0) falls through to `max_tokens`, so
    /// it must not shadow an over-cap `max_tokens` — and standing alone it
    /// leaves the budget unset (→ inject).
    #[test]
    fn output_budget_zero_mct_falls_through_to_max_tokens() {
        for body in [
            r#"{"model":"x","max_completion_tokens":0,"max_tokens":999999}"#,
            r#"{"model":"x","max_completion_tokens":0.0,"max_tokens":999999}"#,
        ] {
            let p = probe_of(body);
            assert!(
                output_budget_action(cap(131072), &p).is_err(),
                "body {body} must reject via the max_tokens fallthrough"
            );
        }
        let p = probe_of(r#"{"model":"x","max_completion_tokens":0}"#);
        assert_eq!(output_budget_action(cap(131072), &p).unwrap(), Some(131072));
    }

    /// Values the engine's pydantic lax mode would coerce to an over-cap int
    /// — integral floats and numeric strings — are rejected, not forwarded.
    #[test]
    fn output_budget_rejects_laxly_coercible_over_cap_values() {
        for body in [
            r#"{"model":"x","max_tokens":999999.0}"#,
            r#"{"model":"x","max_tokens":"999999"}"#,
        ] {
            let p = probe_of(body);
            assert!(
                output_budget_action(cap(131072), &p).is_err(),
                "body {body} must reject"
            );
        }
    }

    /// A mistyped value that can't reach an over-cap int at the engine
    /// (non-numeric string, under-cap float, negative) neither rejects nor
    /// injects — it forwards untouched for the engine's own validation.
    #[test]
    fn output_budget_mistyped_value_forwards_untouched() {
        for body in [
            r#"{"model":"x","max_tokens":"large"}"#,
            r#"{"model":"x","max_tokens":1.5}"#,
            r#"{"model":"x","max_tokens":-5}"#,
        ] {
            let p = probe_of(body);
            assert_eq!(
                output_budget_action(cap(131072), &p).unwrap(),
                None,
                "body {body} must forward untouched"
            );
        }
    }

    /// A router-minted `rid` is injected as a top-level string so the engine
    /// adopts it (and the router can later abort by it). `messages` are
    /// untouched.
    #[test]
    fn build_outgoing_body_injects_rid() {
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let out = build_outgoing_body(&body, Some(value), None, None, Some("router-abc123"), None)
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(
            parsed.get("rid").and_then(|r| r.as_str()),
            Some("router-abc123"),
            "the router-minted rid must be injected as a top-level string",
        );
        assert!(
            parsed.get("messages").is_some(),
            "messages must be retained alongside the injected rid",
        );
    }

    /// PD + forwarding: both `input_ids` and the bootstrap fields land in one
    /// serialized body.
    #[test]
    fn build_outgoing_body_injects_both_input_ids_and_bootstrap() {
        let body =
            Bytes::from_static(br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let ids = [7u32, 8];
        let bootstrap = BootstrapFields {
            host: "h".into(),
            port: Some(9),
            room: 5,
        };
        let out = build_outgoing_body(&body, Some(value), Some(&ids), Some(&bootstrap), None, None)
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(parsed.get("input_ids"), Some(&serde_json::json!([7, 8])));
        assert_eq!(
            parsed.get("bootstrap_room"),
            Some(&serde_json::Value::Number(5.into()))
        );
        assert_eq!(
            parsed.get("bootstrap_port"),
            Some(&serde_json::Value::Number(9.into()))
        );
    }

    /// Tool / function requests are detected so the caller omits `input_ids`
    /// (the router's encoder doesn't render tools).
    #[test]
    fn request_has_tools_detects_tools_and_functions() {
        assert!(request_has_tools(
            &serde_json::json!({"tools":[{"type":"function"}]})
        ));
        assert!(request_has_tools(
            &serde_json::json!({"functions":[{"name":"f"}]})
        ));
        assert!(!request_has_tools(&serde_json::json!({"tools":[]})));
        assert!(!request_has_tools(&serde_json::json!({"messages":[]})));
    }

    /// Array (multimodal) message content is detected so the caller omits
    /// `input_ids` (a text tokenizer can't represent image content).
    #[test]
    fn request_is_multimodal_detects_array_content() {
        assert!(request_is_multimodal(&serde_json::json!({
            "messages":[{"role":"user","content":[{"type":"image_url","image_url":"x"}]}]
        })));
        assert!(!request_is_multimodal(&serde_json::json!({
            "messages":[{"role":"user","content":"hello"}]
        })));
    }

    /// Plain text chat with nothing unreplicated → input_ids may be forwarded.
    #[test]
    fn input_ids_safe_to_forward_allows_plain_text_chat() {
        assert!(input_ids_safe_to_forward(&serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}]
        })));
    }

    /// Every field the engine honors on the `messages` path but which the
    /// router's encoder does not replicate must block forwarding — otherwise
    /// the engine uses the router's ids verbatim and silently runs a different
    /// prompt than the request asked for.
    #[test]
    fn input_ids_safe_to_forward_blocks_unreplicated_signals() {
        let blockers = [
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function"}]}),
            serde_json::json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":"x"}]}]}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"chat_template":"{{ custom }}"}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":true}}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"task":"generate"}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"continue_final_message":true}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"partial"}]}),
        ];
        for b in blockers {
            assert!(
                !input_ids_safe_to_forward(&b),
                "must NOT forward input_ids for: {b}"
            );
        }
    }

    /// Null / false-valued fields do not block (absent ≡ null ≡ default).
    #[test]
    fn input_ids_safe_to_forward_ignores_null_and_false_fields() {
        assert!(input_ids_safe_to_forward(&serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template": null,
            "reasoning_effort": null,
            "chat_template_kwargs": null,
            "continue_final_message": false
        })));
    }

    /// Load-only + PD: `build_outgoing_body` is handed `None` for the value
    /// (the ingress skipped the parse for a load-only policy) and re-parses the
    /// bytes to inject the bootstrap fields. `input_ids` is never set here.
    #[test]
    fn build_outgoing_body_reparses_when_value_absent() {
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let bootstrap = BootstrapFields {
            host: "h".into(),
            port: Some(1),
            room: 2,
        };
        let out = build_outgoing_body(&body, None, None, Some(&bootstrap), None, None).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(
            parsed.get("bootstrap_room"),
            Some(&serde_json::Value::Number(2.into()))
        );
        assert!(parsed.get("input_ids").is_none());
    }

    /// A chat request on a chat-encoder model that yields engine-equivalent
    /// ids (encode succeeded) is NOT a failure — the offload worked.
    #[test]
    fn offload_failed_false_when_tokens_engine_equivalent() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        let tokens = RequestTokens {
            ids: vec![1, 2, 3],
            engine_equivalent: true,
            parity: crate::tokenizer::ForwardParity::Conservative,
        };
        assert!(!ingress_tokenize_offload_failed(
            true,
            Some(&value),
            Some(&tokens)
        ));
    }

    /// A chat request on a chat-encoder model whose tokenization yielded NO
    /// tokens (encode_chat returned None → request_tokens None) IS a failure:
    /// the encoder should have fired but didn't.
    #[test]
    fn offload_failed_true_when_chat_encoder_request_has_no_tokens() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        assert!(ingress_tokenize_offload_failed(true, Some(&value), None));
    }

    /// Encode produced ids but NOT via the chat encoder (raw fallback,
    /// `engine_equivalent = false`) on a chat-encoder model + chat request →
    /// the chat-encode render/encode failed and fell through to the raw path.
    #[test]
    fn offload_failed_true_when_tokens_not_engine_equivalent() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        let tokens = RequestTokens {
            ids: vec![1, 2, 3],
            engine_equivalent: false,
            parity: crate::tokenizer::ForwardParity::Conservative,
        };
        assert!(ingress_tokenize_offload_failed(
            true,
            Some(&value),
            Some(&tokens)
        ));
    }

    /// Non-chat-encoder models never expected the offload → not a failure even
    /// with no tokens.
    #[test]
    fn offload_failed_false_without_chat_encoder() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        assert!(!ingress_tokenize_offload_failed(false, Some(&value), None));
    }

    /// A non-chat (no `messages`) request on a chat-encoder model — e.g.
    /// `/v1/completions` `prompt` — never expected the chat-encode offload, so
    /// the absence of engine-equivalent ids is not a failure.
    #[test]
    fn offload_failed_false_for_non_messages_request() {
        let value = serde_json::json!({"prompt":"hi"});
        assert!(!ingress_tokenize_offload_failed(true, Some(&value), None));
    }

    /// The id must be derived EXACTLY ONCE per request.
    ///
    /// A second `derive_request_id(...)` at dispatch mints a fresh UUID on the
    /// no-client-rid, no-`x-request-id` path, so the id the tee sent and the id
    /// the engine logged diverge — silently, for exactly the direct-to-router
    /// traffic where correlation is already hardest. Mutation testing showed
    /// that regression surviving the entire 773-test suite, because the
    /// behavioral path needs a live worker to drive.
    ///
    /// This is a structural guard, not a behavioral one: it counts call sites
    /// in the source. Crude, but it encodes the invariant the doc comment
    /// states, costs nothing, and fails loudly on the exact reintroduction.
    #[test]
    fn request_id_is_derived_exactly_once_in_the_handler() {
        // Count real call sites only: production code (everything before the
        // test module), with comment lines stripped so prose mentioning the
        // function does not inflate the count.
        let src = include_str!("chat.rs");
        let production = src.split("#[cfg(test)]").next().unwrap();
        let hits = production
            .lines()
            .map(str::trim_start)
            .filter(|l| !l.starts_with("//"))
            .map(|l| l.matches("derive_request_id(").count())
            .sum::<usize>();
        // One definition + exactly one call.
        assert_eq!(
            hits, 2,
            "expected the definition plus exactly ONE call to derive_request_id \
             in production code; found {hits} occurrences. Reuse the existing \
             `derived_request_id` binding instead of deriving again — a second \
             derivation hands one request two identities, silently, on the \
             no-x-request-id path."
        );
    }

    #[test]
    fn derive_request_id_rejects_empty_ids_that_would_collapse_the_key() {
        // An empty client rid or a bare `x-request-id:` header would put every
        // affected request under one key, turning a downstream join into a
        // cross product.
        let mut h = HeaderMap::new();
        h.insert("x-request-id", "".parse().unwrap());
        let id = derive_request_id(Some(""), &h);
        assert!(id.starts_with("router-"), "got {id}");
        assert!(
            id.len() > "router-".len(),
            "must not be the bare prefix: {id}"
        );
    }

    #[test]
    fn derive_request_id_prefers_a_client_supplied_rid() {
        let mut h = HeaderMap::new();
        h.insert("x-request-id", "gw-abc".parse().unwrap());
        assert_eq!(derive_request_id(Some("client-rid"), &h), "client-rid");
    }

    #[test]
    fn derive_request_id_falls_back_to_the_gateway_header() {
        // Sharing the gateway's id is what lets a cache-sim record, a router
        // log line, and an engine log line be recognized as the same request.
        let mut h = HeaderMap::new();
        h.insert("x-request-id", "gw-abc".parse().unwrap());
        assert_eq!(derive_request_id(None, &h), "router-gw-abc");
    }

    #[test]
    fn derive_request_id_mints_a_uuid_with_no_client_rid_and_no_header() {
        let h = HeaderMap::new();
        let a = derive_request_id(None, &h);
        assert!(a.starts_with("router-"));
        // Distinct per call — which is exactly WHY the handler derives once
        // and reuses the value: deriving separately for the tee and for the
        // engine would hand one request two different identities, and the
        // oracle's ingest and extend records would never pair.
        assert_ne!(a, derive_request_id(None, &h));
    }

    #[test]
    fn parse_probe_reads_stream_bool_from_object() {
        let b = Bytes::from_static(br#"{"stream": true, "model": "tiny"}"#);
        assert_eq!(parse_probe(&b).unwrap().stream, Some(true));
        let b = Bytes::from_static(br#"{"stream": false, "model": "tiny"}"#);
        assert_eq!(parse_probe(&b).unwrap().stream, Some(false));
    }

    #[test]
    fn parse_probe_defaults_when_stream_absent() {
        // Existing happy-path contract: well-formed object missing `stream`
        // must default to None (caller picks false). The minimal `RequestProbe`
        // (Option<bool> + #[serde(default)]) must NOT break this.
        let b = Bytes::from_static(br#"{"model": "tiny", "messages": []}"#);
        let p = parse_probe(&b).unwrap();
        assert_eq!(p.stream, None);
        assert_eq!(p.model.as_deref(), Some("tiny"));
    }

    #[test]
    fn parse_probe_rejects_non_object_shapes() {
        // Pin the contract: degenerate JSON (valid JSON but wrong shape)
        // must be rejected, not silently forwarded with `stream=false`.
        for bad in [&b"null"[..], &b"[]"[..], &b"\"hi\""[..], &b"42"[..]] {
            let b = Bytes::copy_from_slice(bad);
            let err = parse_probe(&b).unwrap_err();
            match err {
                ApiError::BadRequest(_) => {}
                other => panic!("expected BadRequest for {bad:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn parse_probe_rejects_malformed_json() {
        let b = Bytes::from_static(b"{not json}");
        let err = parse_probe(&b).unwrap_err();
        assert!(matches!(err, ApiError::BadRequest(_)));
    }

    #[test]
    fn parse_probe_handles_nested_messages_with_stream_true() {
        // Well-formed object with nested arrays/objects (real chat-completions
        // payloads carry `messages: [{role, content: [{type, text}]}]`). The
        // two-step deserialize must not balk on this — only the top-level
        // object shape and the `stream`/`model` fields matter.
        let b = Bytes::from_static(
            br#"{
              "model": "x",
              "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
              "stream": true
            }"#,
        );
        assert_eq!(parse_probe(&b).unwrap().stream, Some(true));
    }

    #[test]
    fn parse_probe_handles_nested_messages_with_stream_false() {
        let b = Bytes::from_static(
            br#"{
              "model": "x",
              "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
              "stream": false
            }"#,
        );
        assert_eq!(parse_probe(&b).unwrap().stream, Some(false));
    }

    #[test]
    fn parse_probe_handles_duplicate_stream_keys() {
        // RFC 8259 says "names within an object SHOULD be unique" but a
        // parser MAY accept duplicates. Step 1 (HashMap) silently
        // last-wins, but step 2 deserializes into the typed `RequestProbe`
        // struct, and `serde_json`'s `#[derive(Deserialize)]` REJECTS
        // duplicate fields with a `duplicate field` error.
        //
        // We map that to `BadRequest` (same path as other malformed input).
        // Pinning "reject" rather than "last-wins" is intentional —
        // ambiguous bodies should fail loudly at the edge, not silently
        // route based on which copy serde happened to see last.
        let b = Bytes::from_static(br#"{"stream": true, "stream": false}"#);
        let err = parse_probe(&b).unwrap_err();
        match err {
            ApiError::BadRequest(_) => {}
            other => panic!("expected BadRequest on duplicate `stream` key, got {other:?}"),
        }
    }

    #[test]
    fn parse_probe_bad_request_message_does_not_leak_serde_detail() {
        // Info-leak guard: the client-visible message must be a fixed
        // string, not the serde error (which can contain line/column
        // detail or hint at field shape).
        let b = Bytes::from_static(br#"{"stream": "not-a-bool"}"#);
        let err = parse_probe(&b).unwrap_err();
        match err {
            ApiError::BadRequest(msg) => assert_eq!(
                msg, "invalid request: body must be a JSON object",
                "client-visible message must be fixed; got: {msg}"
            ),
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    /// A request rejected BEFORE routing — here a body that is valid JSON but not
    /// an object, which `parse_probe` 400s before any worker is selected — must
    /// still be (a) logged by the outermost access-log middleware and (b) counted
    /// in `requests_total`. Both happen in `access_log_and_record`, NOT the
    /// handler (which returns the 400 via `?`), so this drives the request
    /// through `build_router` to exercise that middleware. Before request
    /// accounting moved to the middleware, pre-routing rejections were invisible
    /// to `requests_total` (so `sum by (outcome)` undercounted) and absent from
    /// the access log.
    #[tokio::test]
    async fn pre_routing_400_is_logged_and_counted() {
        use std::sync::Mutex;
        use tower::ServiceExt;
        use tracing_subscriber::fmt::MakeWriter;

        #[derive(Clone)]
        struct VecWriter(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for VecWriter {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        impl<'a> MakeWriter<'a> for VecWriter {
            type Writer = VecWriter;
            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(VecWriter(buf.clone()))
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);

        let ctx = Arc::new(AppContext::stub());
        let app = crate::server::app::build_router(ctx.clone());
        // Valid JSON but not an object → `parse_probe` rejects with 400 via `?`
        // before any worker is selected.
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from("[]"))
            .unwrap();
        let res = app.oneshot(req).await.unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);

        let logs = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        assert!(
            logs.contains("http_request"),
            "every request must be logged by the middleware; captured:\n{logs}"
        );
        assert!(
            logs.contains("path=/v1/chat/completions") && logs.contains("status=400"),
            "access log must record the path and 400 status; captured:\n{logs}"
        );

        let metrics = ctx.metrics.render();
        assert!(
            metrics
                .lines()
                .any(|l| l.starts_with("sgl_router_worker_requests_total")
                    && l.contains(r#"outcome="error""#)),
            "a pre-routing 400 must be counted in worker_requests_total{{outcome=error}}; got:\n{metrics}"
        );
    }
}
