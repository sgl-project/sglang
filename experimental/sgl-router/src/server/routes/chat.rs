// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelId, WorkerMode};
use crate::policies::registry::{PdPoolResolver, PdResolveError};
use crate::policies::{request_tokens_for, RequestTokens, SelectionContext};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::{
    MetricsRegistry, RequestLogContext, StaleRequestOutcome, WorkerModeLabel,
};
use crate::workers::Worker;
use axum::body::Body;
use axum::extract::State;
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

/// Per-route body-size cap on `/v1/chat/completions`. 5 MiB accommodates a
/// long context — a ~1 M-token context tokenized as JSON fits under this —
/// while preventing a hostile client from forcing the router to
/// heap-allocate hundreds of MiB before forwarding. The cap is wired in
/// `crate::server::app::build_router` as a route-level `DefaultBodyLimit`
/// layer; axum's `Bytes` extractor enforces it and returns 413
/// PAYLOAD_TOO_LARGE before this handler runs.
pub const MAX_CHAT_BODY_BYTES: usize = 5 << 20;

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
pub async fn chat_completions(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    chat_completions_inner(ctx, headers, body).await
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
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let start = std::time::Instant::now();
    let probe = parse_probe(&body)?;
    let streaming = probe.stream.unwrap_or(false);
    let model_str = probe
        .model
        .ok_or_else(|| ApiError::BadRequest("missing `model` field".into()))?;
    let model_id = ModelId(model_str.clone());

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
    // Diagnostic: ingress-tokenize cost, sampled. Fires for EVERY request that
    // reaches here — including those about to be shed at admission below — so a
    // shed request's pre-admission time (the latency the access log shows on a
    // 503) is attributable to tokenize vs. the rest.
    if PHASE_LOG_COUNTER.fetch_add(1, Ordering::Relaxed) % PHASE_LOG_SAMPLE == 0 {
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
    let routing_key = ctx
        .config
        .model
        .sticky
        .as_ref()
        .and_then(|s| headers.get(s.header_name.as_str()))
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty());
    let selection_ctx = SelectionContext::with_routing_key(&model_id, Some(&body), routing_key)
        .with_request_tokens(request_tokens.as_ref().map(|t| t.ids.as_slice()));
    // Admission gate: pick a worker and claim an in-flight slot, parking until
    // one frees if every candidate is at its cap. A pass-through (immediate
    // dispatch, unconditional guard) when no per-worker cap is configured.
    // Yields 503 `service_overloaded` when the wait queue is full.
    let (worker, guard) = ctx
        .admission
        .acquire(&workers, policy.as_ref(), &selection_ctx, &model_str)
        .await?;
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
    let active_guard =
        ctx.active_load
            .register(worker.id.clone(), worker.url.clone(), prefill_load, 0);
    // Snapshot the stale-request cancel token BEFORE moving the guard
    // into the spawned prefill task / streaming pump / response future.
    // The token is cheap to clone (it's an `Arc<...>` internally) and
    // the chat handler races the client-facing fetch against
    // `token.cancelled()` to surface a 504 `stale_request_expired` if
    // the janitor expires the request mid-flight.
    let stale_token = active_guard.cancel_token().clone();

    // Snapshot the labels we need for metrics BEFORE moving the worker
    // / model_str values into the per-branch fetch futures.
    let metrics_worker_url = worker.url.clone();
    let metrics_mode = match worker.mode() {
        WorkerMode::Prefill => WorkerModeLabel::Prefill,
        WorkerMode::Decode => WorkerModeLabel::Decode,
        WorkerMode::Plain => WorkerModeLabel::Plain,
    };
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

    // Builds the end-to-end-latency guard for streaming requests. Packed into
    // `stream_guards` so it records when the SSE pump finishes (stream end or
    // client disconnect), not at response-headers time. Non-streaming records
    // at the dispatch site instead (see below).
    let make_duration_guard = || RecordDurationOnDrop {
        metrics: Arc::clone(&ctx.metrics),
        model: metrics_model.clone(),
        start,
    };

    // Forward the router-computed tokens to the engine as `input_ids` so it
    // skips re-tokenizing the same prompt — but only when they are
    // engine-equivalent (chat-encoder path) AND the request contains nothing
    // the router's encoder didn't replicate (see `input_ids_safe_to_forward`).
    // Otherwise omit them and the engine tokenizes from `messages` as usual —
    // a transparent, always-correct fallback (`messages` are always retained
    // in the forwarded body). `forward_input_ids` is `Some` only when
    // `request_value` is `Some` (a model the ingress tokenized for), so the
    // predicate always has a parsed body to inspect.
    let forward_input_ids: Option<&[u32]> = match (request_tokens.as_ref(), request_value.as_ref())
    {
        (Some(t), Some(v)) if t.engine_equivalent && input_ids_safe_to_forward(v) => {
            Some(t.ids.as_slice())
        }
        _ => None,
    };

    // Surface a broken offload: when the encoder SHOULD have produced
    // engine-equivalent ids but didn't, the chat request silently fell back to
    // engine-side tokenization. Count only that case (see
    // `ingress_tokenize_offload_failed`); successful forwards and expected
    // omissions are not problems.
    if ingress_tokenize_offload_failed(
        ctx.tokenizers.has_chat_encoder(&model_str),
        request_value.as_ref(),
        request_tokens.as_ref(),
    ) {
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
    let request_id: Option<String> = if decode_peer.is_none() {
        Some(match &client_rid {
            Some(rid) => rid.clone(),
            // No client-supplied rid: derive the minted one from the
            // gateway/access-log `x-request-id` (already read the same way by
            // the global access-log middleware, server/app.rs) instead of an
            // unrelated random UUID. Without this, the router/gateway logs
            // (keyed on x-request-id) and the engine logs (keyed on this rid)
            // use two disjoint identifiers for the same request, making it
            // impossible to correlate a log line found in one against the
            // other. Falls back to a UUID only when the header itself is
            // absent (e.g. direct-to-router traffic bypassing the gateway).
            None => {
                let xrid = headers.get("x-request-id").and_then(|v| v.to_str().ok());
                match xrid {
                    Some(id) => format!("router-{id}"),
                    None => format!("router-{}", uuid::Uuid::new_v4().simple()),
                }
            }
        })
    } else {
        None
    };
    // Inject only a router-minted rid; a client-supplied one is already in the
    // body, and PD mode (`request_id == None`) is never injected.
    let rid_to_inject: Option<&str> = match (request_id.as_deref(), client_rid.as_deref()) {
        (Some(rid), None) => Some(rid),
        _ => None,
    };

    // Build the body forwarded to the engine(s) exactly once — injecting the
    // `rid`, `input_ids`, and/or bootstrap fields, or forwarding the original
    // bytes untouched when none apply.
    let outgoing_body = build_outgoing_body(
        &body,
        request_value,
        forward_input_ids,
        bootstrap.as_ref(),
        rid_to_inject,
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
            );
            tokio::select! {
                biased;
                r = fetch => r,
                _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
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
                _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
            }
        }
    } else if streaming {
        // Plain mode, streaming. Both guards ride the SSE pump until
        // the body completes — see the matching comment in the
        // non-streaming arm.
        let stream_guards: Box<dyn Send + 'static> =
            Box::new((guard, active_guard, make_duration_guard()));
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
            outgoing_body,
            Some(stream_guards),
            Some(make_ttft_hook()),
            // Abort the engine if the client disconnects before the engine
            // finishes streaming. The SSE pump (which owns the guard) fires
            // it; here we only supply the rid the engine knows this request by.
            request_id.as_deref(),
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
            _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
        };
        // A received response (any status) means responsibility has passed
        // to `forward_streaming_to`'s own guard (or nothing, for non-2xx) —
        // disarm so this one doesn't also fire. Left armed only when `fetch`
        // never resolved (stale-timeout) or a transport-level dispatch error
        // occurred before any response.
        if r.is_ok() {
            if let Some(g) = pre_headers_abort_guard.as_mut() {
                g.disarm();
            }
        }
        r
    } else {
        // Plain mode, non-streaming. The handler awaits the full
        // buffered response, so both guards live correctly in this
        // scope. The tuple binding exists only to extend the guards'
        // lifetime to the end of the function — the `forward_json_to`
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
            outgoing_body,
        );
        // Same `biased` order as the streaming arm.
        let r = tokio::select! {
            biased;
            r = fetch => r,
            _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
        };
        // A complete response (any status) means the engine is done with this
        // request — don't abort it. Only an early drop (client disconnect) or
        // stale-timeout leaves the guard armed.
        if r.is_ok() {
            if let Some(g) = abort_guard.as_mut() {
                g.disarm();
            }
        }
        r
    };

    // Diagnostic: phase breakdown up to response headers, for ADMITTED requests
    // (those that got a slot). `dispatch_to_headers_ms` is connect + send-body +
    // wait-for-upstream-headers; for streaming this is the whole synchronous cost
    // before the SSE pump takes over. The pump's own first-byte/drain/exit timing
    // is logged separately as `sse_pump_timing`.
    let at_post_dispatch = start.elapsed();
    if PHASE_LOG_COUNTER.fetch_add(1, Ordering::Relaxed) % PHASE_LOG_SAMPLE == 0 {
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
    // Tag the routed response so the middleware records its per-worker labels
    // and logs it with the worker/model it was dispatched to.
    response.extensions_mut().insert(log_ctx);
    Ok(response)
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
/// engine-equivalent and `input_ids_safe_to_forward` held.
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
) -> Result<Bytes, ApiError> {
    if input_ids.is_none() && bootstrap.is_none() && rid.is_none() {
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

/// Whether the router's `input_ids` may be forwarded for this request.
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
///   * `tools` / `functions` — the encoder doesn't render tool schemas.
///   * multimodal (array) `content` — a text tokenizer can't represent images.
///   * `chat_template` — an OpenAI-compatible per-request template override
///     (e.g. vLLM); the router renders with the model's default template, so a
///     custom one would diverge. (SGLang ignores it today, but block it so the
///     offload stays correct across engines / future versions.)
///   * `chat_template_kwargs` (carries `enable_thinking`/`thinking`),
///     `reasoning` / `reasoning_effort`, `task` — thinking/mode toggles the
///     encoder renders in the engine's default mode only.
///   * `continue_final_message: true`, or a trailing `assistant` message — the
///     engine rewrites/strips the final assistant turn; the encoder renders it
///     verbatim.
///
/// NOTE: the router's chat encoder renders in the engine's default
/// (non-thinking) mode. Current sglang derives thinking from the request
/// (`chat_template_kwargs`), which this guard already omits, so a plain request
/// the router rendered matches the engine. The only way to diverge is an engine
/// build that applies a non-default thinking mode the router can't observe from
/// the request — the same router↔engine tokenization-parity assumption that
/// cache-aware routing already depends on. The same assumption covers
/// `add_special_tokens`: the router renders specials via the chat template, which
/// matches the engine on tokenizers that auto-add them (the common case); a
/// tokenizer that does not would diverge by a leading special, again undetectable
/// from the request.
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
        .and_then(|v| v.as_bool())
        == Some(true)
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
/// they are not failures. A tools / multimodal / thinking request on a
/// chat-encoder model still gets engine-equivalent ids (`encode_chat`
/// succeeded; the safe-predicate withholds forwarding for other reasons), so it
/// is an expected omission, not a failure.
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
            build_outgoing_body(&body, Some(value), None, Some(&bootstrap), None).unwrap();
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

    /// `input_ids` are injected and `messages` retained (the engine still
    /// needs them for stop tokens / tool-call constraint / response shape).
    #[test]
    fn build_outgoing_body_injects_input_ids_and_keeps_messages() {
        let body =
            Bytes::from_static(br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let ids = [1u32, 2, 3];
        let out = build_outgoing_body(&body, Some(value), Some(&ids), None, None).unwrap();
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
        let out = build_outgoing_body(&body, Some(value), None, None, None).unwrap();
        assert_eq!(
            out, body,
            "no injection must forward the original bytes unchanged"
        );
    }

    /// A router-minted `rid` is injected as a top-level string so the engine
    /// adopts it (and the router can later abort by it). `messages` are
    /// untouched.
    #[test]
    fn build_outgoing_body_injects_rid() {
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let out =
            build_outgoing_body(&body, Some(value), None, None, Some("router-abc123")).unwrap();
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
        let out =
            build_outgoing_body(&body, Some(value), Some(&ids), Some(&bootstrap), None).unwrap();
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
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"reasoning":{"enabled":true}}),
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
        let out = build_outgoing_body(&body, None, None, Some(&bootstrap), None).unwrap();
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
        assert_eq!(res.status(), axum::http::StatusCode::BAD_REQUEST);

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
