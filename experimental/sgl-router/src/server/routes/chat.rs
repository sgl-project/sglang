// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

mod forward;
mod preparation;
mod reorg;

use crate::config::{SessionAffinityMode, DEFAULT_MIN_LOAD_CHOICES};
use crate::discovery::{ModelId, WorkerMode};
use crate::policies::registry::{PdPoolResolver, PdResolveError};
use crate::policies::selection::{
    select_decode_peer, select_prefill_worker, DecodeSelectionInputs, PrefillSelectionInputs,
};
use crate::policies::{ExternalPrefixSignal, Policy};
use crate::server::app_context::{AppContext, ChatRouting};
use crate::server::error::ApiError;
use crate::server::metrics::PolicySelectionFailureReason;
use crate::state::kv_events::{compute_block_hashes, compute_block_hashes_bigram};
use crate::state::load_monitor::engine_reported_load::EngineReportedLoadSnapshot;
use crate::workers::Worker;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, Response};
use bytes::Bytes;
use forward::{forward_chat_request, SelectedWorkers};
use preparation::{parse_routing_fields, PreparedChatRequest};
use std::sync::Arc;
use std::time::Instant;

const X_SGL_TTFT_SLO_MS: HeaderName = HeaderName::from_static("x-sgl-ttft-slo-ms");
const X_SGL_TPS_SLO: HeaderName = HeaderName::from_static("x-sgl-tps-slo");

/// Maximum buffered request body, including base64 multimodal inputs (32 MiB).
/// Enforced by the `DefaultBodyLimit` layer in app.rs, which returns 413.
pub const MAX_CHAT_BODY_BYTES: usize = 32 << 20;

/// Validate, select workers, and forward a chat-completions request.
pub async fn chat_completions(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    match &ctx.chat_routing {
        ChatRouting::Legacy => chat_completions_legacy(&ctx, headers, body).await,
        ChatRouting::Reorg(resolvers) => {
            reorg::chat_completions(&ctx, resolvers, headers, body).await
        }
    }
}

async fn chat_completions_legacy(
    ctx: &AppContext,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let start = Instant::now();
    let mut fields = parse_routing_fields(&body)?;
    let model = ModelId(
        fields
            .model
            .take()
            .ok_or_else(|| ApiError::BadRequest("missing `model` field".into()))?,
    );

    // Find healthy workers: the prefill pool in PD mode, otherwise the plain pool.
    let resolver = PdPoolResolver::new(Arc::clone(&ctx.registry));
    let candidates = resolver
        .prefill_candidates(&model)
        .map_err(|error| pool_error(error, &model))?;
    let policy = ctx
        .policies
        .get(&model)
        .ok_or_else(|| ApiError::ModelNotFound(model.0.clone()))?;

    let request =
        PreparedChatRequest::prepare(ctx, model, fields, body, policy.needs_request_tokens())?;

    // Pick a plain worker, or a prefill worker followed by a decode peer in PD mode.
    let workers = select_workers(
        ctx,
        &request,
        &headers,
        policy.as_ref(),
        &candidates,
        &resolver,
    )
    .await?;

    // PD sends to both workers and returns the decode response.
    forward_chat_request(ctx, request, workers, headers, start).await
}

fn pool_error(error: PdResolveError, model: &ModelId) -> ApiError {
    let model = model.0.clone();
    match error {
        PdResolveError::NoHealthyWorkers => ApiError::NoHealthyWorkers { model },
        PdResolveError::NoPrefillWorkersAvailable => ApiError::NoPrefillWorkersAvailable { model },
        PdResolveError::NoDecodeWorkersAvailable => ApiError::NoDecodeWorkersAvailable { model },
    }
}

async fn select_workers(
    ctx: &AppContext,
    request: &PreparedChatRequest,
    headers: &HeaderMap,
    policy: &dyn Policy,
    candidates: &[Arc<Worker>],
    resolver: &PdPoolResolver,
) -> Result<SelectedWorkers, ApiError> {
    // Find cached prompt prefixes and capture engine load info.
    let routing_context = RoutingContext {
        prefix_matches: lookup_prefix_matches(ctx, request).await?,
        load_snapshot: capture_load_snapshot(ctx, policy, candidates),
        ..RoutingContext::from_headers(ctx, headers)?
    };

    let prefill = pick_prefill_worker(ctx, request, policy, candidates, &routing_context)?;
    let decode = pick_decode_worker(ctx, request, &prefill, resolver, &routing_context)?;
    Ok(SelectedWorkers {
        prefill,
        decode,
        track_dispatch_timestamps: policy.needs_dispatch_timestamps(),
    })
}

fn capture_load_snapshot(
    ctx: &AppContext,
    policy: &dyn Policy,
    candidates: &[Arc<Worker>],
) -> Option<EngineReportedLoadSnapshot> {
    let needed = policy.needs_load_snapshot()
        || candidates
            .iter()
            .any(|worker| worker.mode() == WorkerMode::Prefill);
    needed.then(|| ctx.engine_reported_load.capture_snapshot(Instant::now()))
}

struct RoutingContext<'a> {
    prefix_matches: Option<ExternalPrefixSignal>,
    load_snapshot: Option<EngineReportedLoadSnapshot>,
    ttft_slo_ms: Option<u64>,
    tps_slo: Option<f64>,
    routing_key: Option<&'a str>,
    session_id: Option<&'a str>,
}

impl<'a> RoutingContext<'a> {
    /// Header-derived selection inputs; prefix and load fields start empty.
    fn from_headers(ctx: &AppContext, headers: &'a HeaderMap) -> Result<Self, ApiError> {
        // Buckets group workers by token limits and service targets; disabled means one pool per role.
        let (ttft_slo_ms, tps_slo) = if ctx.bucket_selector.is_enabled() {
            (
                parse_optional_positive_u64_header(headers, &X_SGL_TTFT_SLO_MS, "TTFT SLO")?,
                parse_optional_positive_f64_header(headers, &X_SGL_TPS_SLO, "TPS SLO")?,
            )
        } else {
            (None, None)
        };
        let routing_key = ctx
            .config
            .model
            .sticky
            .as_ref()
            .and_then(|config| nonempty_header(headers, &config.header_name));
        let session_id = ctx
            .config
            .model
            .affinity
            .as_ref()
            .and_then(|config| nonempty_header(headers, &config.session_id_header));
        Ok(Self {
            prefix_matches: None,
            load_snapshot: None,
            ttft_slo_ms,
            tps_slo,
            routing_key,
            session_id,
        })
    }
}

fn nonempty_header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
}

fn pick_prefill_worker(
    ctx: &AppContext,
    request: &PreparedChatRequest,
    policy: &dyn Policy,
    candidates: &[Arc<Worker>],
    routing: &RoutingContext<'_>,
) -> Result<Arc<Worker>, ApiError> {
    let affinity = ctx.config.model.affinity.as_ref();
    select_prefill_worker(&PrefillSelectionInputs {
        policy,
        policy_kind: ctx.config.model.policy,
        bucket_selector: ctx.bucket_selector.as_ref(),
        metrics: ctx.metrics.as_ref(),
        model_id: &request.model,
        body: Some(&request.body),
        routing_key: routing.routing_key,
        session_id: routing.session_id,
        request_input_tokens: request.input_token_count as u64,
        request_tokens: request.tokens.as_ref().map(|tokens| tokens.ids.as_slice()),
        external_prefix: routing.prefix_matches.as_ref(),
        load_snapshot: routing.load_snapshot.as_ref(),
        workers: candidates,
        ttft_slo_ms: routing.ttft_slo_ms,
        tps_slo: routing.tps_slo,
        session_affinity_mode: affinity
            .map(|config| config.session_affinity_mode)
            .unwrap_or(SessionAffinityMode::Bucket),
        worker_queue_limit: affinity.and_then(|config| config.worker_queue_limit),
        saturation_queue_floor: affinity.and_then(|config| config.saturation_queue_floor),
        min_load_choices: affinity
            .map(|config| config.min_load_choices)
            .unwrap_or(DEFAULT_MIN_LOAD_CHOICES),
    })
    .map_err(|reason| policy_selection_failed(ctx, &request.model.0, reason))
}

fn pick_decode_worker(
    ctx: &AppContext,
    request: &PreparedChatRequest,
    prefill: &Worker,
    resolver: &PdPoolResolver,
    routing: &RoutingContext<'_>,
) -> Result<Option<Arc<Worker>>, ApiError> {
    if prefill.mode() != WorkerMode::Prefill {
        return Ok(None);
    }
    let candidates = resolver
        .decode_candidates(&request.model)
        .map_err(|error| pool_error(error, &request.model))?;
    let decode = select_decode_peer(&DecodeSelectionInputs {
        decode_policy_kind: ctx.config.model.decode_policy,
        bucket_selector: ctx.bucket_selector.as_ref(),
        model_id: &request.model,
        prefill_url: &prefill.url,
        decode_workers: &candidates,
        request_input_tokens: request.input_token_count as u64,
        requested_max_output_tokens: request.max_output_tokens,
        ttft_slo_ms: routing.ttft_slo_ms,
        tps_slo: routing.tps_slo,
        load_snapshot: routing.load_snapshot.as_ref(),
    })
    .ok_or_else(|| ApiError::NoDecodeWorkersAvailable {
        model: request.model.0.clone(),
    })?;
    Ok(Some(decode))
}

/// Ask which workers already hold a KV prefix for this prompt; not a worker pick.
async fn lookup_prefix_matches(
    ctx: &AppContext,
    request: &PreparedChatRequest,
) -> Result<Option<ExternalPrefixSignal>, ApiError> {
    let signal = match (
        ctx.prefix_index.as_ref(),
        request.tokens.as_ref(),
        ctx.block_size_oracle.get(),
    ) {
        // Remote indexer: hash tokens into blocks and match against the KV index.
        (Some(index), Some(tokens), Some(block_size)) => {
            let hashes = if ctx.block_size_oracle.is_bigram() {
                compute_block_hashes_bigram(&tokens.ids, block_size as usize)
            } else {
                compute_block_hashes(&tokens.ids, block_size as usize)
            };
            let query_blocks = hashes.len();
            let outcome = if hashes.is_empty() {
                sgl_kv_indexer::PrefixOutcome::Empty
            } else {
                resolve_prefix_query(index.match_prefix(hashes).await, &request.model.0)?
            };
            Some(ExternalPrefixSignal {
                outcome,
                query_blocks,
            })
        }
        // Without usable indexer inputs, try the in-process radix tree.
        _ => ctx
            .radix_tree_prefix_provider
            .as_ref()
            .zip(request.tokens.as_ref())
            .and_then(|(provider, tokens)| provider.match_request_tokens(&tokens.ids)),
    };
    Ok(signal)
}

    // Prefer exact ingress tokens; otherwise use the conservative estimate.
    let prefill_load = request_tokens
        .as_ref()
        .map(|tokens| tokens.ids.len().max(1))
        .unwrap_or_else(|| estimate_prefill_tokens(&body));
    let request_input_tokens = prefill_load as u64;
    let needs_load_snapshot = policy.needs_load_snapshot()
        || workers
            .iter()
            .any(|worker| worker.mode() == WorkerMode::Prefill);
    let load_snapshot =
        needs_load_snapshot.then(|| ctx.engine_load.capture_snapshot(std::time::Instant::now()));
    let needs_dispatch_timestamps = policy.needs_dispatch_timestamps();
    let (ttft_slo_ms, tps_slo) = if ctx.bucket_selector.is_enabled() {
        (
            parse_optional_positive_u64_header(&headers, &X_SGL_TTFT_SLO_MS, "TTFT SLO")?,
            parse_optional_positive_f64_header(&headers, &X_SGL_TPS_SLO, "TPS SLO")?,
        )
    } else {
        (None, None)
    };

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
    let session_id = ctx
        .config
        .model
        .affinity
        .as_ref()
        .and_then(|config| headers.get(config.session_id_header.as_str()))
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty());
    // `select_prefill_worker` reduces this to `Bucket` when Bucket
    // partitioning is off.
    let session_affinity_mode = ctx
        .config
        .model
        .affinity
        .as_ref()
        .map(|config| config.session_affinity_mode)
        .unwrap_or(SessionAffinityMode::Bucket);
    // Each Bucket retry rebuilds the proposal and reruns Admission/Guard.
    let worker = select_prefill_worker(&PrefillSelectionInputs {
        policy: policy.as_ref(),
        policy_kind: ctx.config.model.policy,
        bucket_selector: ctx.bucket_selector.as_ref(),
        metrics: ctx.metrics.as_ref(),
        model_id: &model_id,
        body: Some(&body),
        routing_key,
        session_id,
        request_input_tokens,
        request_tokens: request_tokens.as_ref().map(|tokens| tokens.ids.as_slice()),
        external_prefix: external_prefix.as_ref(),
        load_snapshot: load_snapshot.as_ref(),
        workers: &workers,
        ttft_slo_ms,
        tps_slo,
        session_affinity_mode,
    })
    .map_err(|reason| policy_selection_failed(&ctx, &model_str, reason))?;

    // Decode selection starts after Final P.
    //
    // Plain-mode workers skip the decode resolution entirely (no
    // decode peer to find). PD-mode requests that fail to resolve a
    // decode peer (`NoDecodeWorkersAvailable`) bubble up as 503 so
    // operators can alert on prefill-vs-decode pool imbalance.
    let decode_peer: Option<Arc<Worker>> = if worker.mode() == WorkerMode::Prefill {
        let decode_workers = resolver.decode_candidates(&model_id).map_err(|e| match e {
            PdResolveError::NoHealthyWorkers => ApiError::NoHealthyWorkers {
                model: model_str.clone(),
            },
            PdResolveError::NoDecodeWorkersAvailable => ApiError::NoDecodeWorkersAvailable {
                model: model_str.clone(),
            },
            PdResolveError::NoPrefillWorkersAvailable => ApiError::NoPrefillWorkersAvailable {
                model: model_str.clone(),
            },
        })?;
        Some(
            select_decode_peer(&DecodeSelectionInputs {
                decode_policy_kind: ctx.config.model.decode_policy,
                bucket_selector: ctx.bucket_selector.as_ref(),
                model_id: &model_id,
                prefill_url: &worker.url,
                decode_workers: &decode_workers,
                request_input_tokens,
                requested_max_output_tokens,
                ttft_slo_ms,
                tps_slo,
                load_snapshot: load_snapshot.as_ref(),
                metrics: Some(ctx.metrics.as_ref()),
            })
            .ok_or_else(|| ApiError::NoDecodeWorkersAvailable {
                model: model_str.clone(),
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
    // mode the pair stays in this handler. Decode load is tracked on Final D.
    let guard = if needs_dispatch_timestamps {
        worker.timestamped_load_guard()
    } else {
        worker.load_guard()
    };
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

    // Classifies a 2xx stream after its headers are committed. Takes the
    // streaming worker's URL (Final D in PD mode).
    let make_stream_end_hook = |worker_url: String| -> Box<dyn FnOnce(StreamEnd) + Send + 'static> {
        let metrics = Arc::clone(&ctx.metrics);
        let model = metrics_model.clone();
        Box::new(move |end| {
            metrics.record_stream_outcome(&worker_url, &model, classify_stream_end(end));
        })
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

    // Build the body forwarded to the engine(s) exactly once — injecting the
    // `input_ids` and/or bootstrap fields, or forwarding the original bytes
    // untouched when neither applies.
    let outgoing_body =
        build_outgoing_body(&body, request_value, forward_input_ids, bootstrap.as_ref())?;

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
        let prefill_breaker = Arc::clone(&worker.breaker);
        let prefill_headers = headers.clone();
        let prefill_body = outgoing_body.clone();
        let prefill_proxy = Arc::clone(&ctx.proxy);
        let prefill_holds: (LoadGuard, _) = (guard, active_guard);
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
        // per-worker `active_requests` reflects load on Final D.
        let decode_guard = decode_worker.load_guard();
        let decode_active_guard =
            ctx.active_load
                .register(decode_worker.id.clone(), decode_worker.url.clone(), 0, 1);
        if streaming {
            let stream_guards: Box<dyn Send + 'static> =
                Box::new((decode_guard, decode_active_guard, make_duration_guard()));
            let fetch = ctx.proxy.forward_streaming_to(
                &decode_worker.url,
                &decode_worker.breaker,
                "/v1/chat/completions",
                &headers,
                outgoing_body,
                Some(stream_guards),
                Some(make_ttft_hook()),
                Some(make_stream_end_hook(decode_worker.url.clone())),
            );
            tokio::select! {
                biased;
                r = fetch => r,
                _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
            }
        } else {
            let _decode_hold = (decode_guard, decode_active_guard);
            let fetch = ctx.proxy.forward_json_to(
                &decode_worker.url,
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
        let fetch = ctx.proxy.forward_streaming_to(
            &worker.url,
            &worker.breaker,
            "/v1/chat/completions",
            &headers,
            outgoing_body,
            Some(stream_guards),
            Some(make_ttft_hook()),
            Some(make_stream_end_hook(worker.url.clone())),
        );
        // Bias `fetch` over the cancellation branch: a successful
        // response that completes in the same poll as the token firing
        // MUST win (returning 504 for a request that already has
        // headers is a correctness regression). The cancellation
        // branch only matters when fetch is still pending — at that
        // point biasing the order is a wash.
        tokio::select! {
            biased;
            r = fetch => r,
            _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
        }
    } else {
        // Plain mode, non-streaming. The handler awaits the full
        // buffered response, so both guards live correctly in this
        // scope. The tuple binding exists only to extend the guards'
        // lifetime to the end of the function — the `forward_json_to`
        // future does not need them (it does not return until the
        // body is buffered).
        let _holds: (LoadGuard, _) = (guard, active_guard);
        let fetch = ctx.proxy.forward_json_to(
            &worker.url,
            &worker.breaker,
            "/v1/chat/completions",
            &headers,
            outgoing_body,
        );
        // Same `biased` order as the streaming arm.
        tokio::select! {
            biased;
            r = fetch => r,
            _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
        }
    };

    // Record the dispatch outcome AFTER we know whether the upstream
    // accepted the request. A 504 from the stale-request branch counts as
    // `cancelled` — semantically distinct from upstream errors that bubble
    // through as `error`. The metric is per-worker so convergence tests
    // can scrape `/metrics` and assert that ≥N requests landed on a
    // single prefill worker.
    let outcome = match &result {
        Ok(_) => RequestOutcome::Success,
        Err(ApiError::StaleRequestExpired { .. }) => {
            // The janitor fired the stale-cancel and we observed it
            // user-side; record both the per-request `cancelled` outcome
            // AND the global `expired` count. The two views are useful for
            // different alerts: per-worker request_total{cancelled} flags a
            // worker that's hanging, while stale_requests_total{expired}
            // tracks the global health of the janitor.
            ctx.metrics
                .record_stale_request(StaleRequestOutcome::Expired);
            RequestOutcome::Cancelled
        }
        Err(_) => RequestOutcome::Error,
    };
    ctx.metrics
        .record_policy_selection_failure(ctx.config.model.policy, reason);
    tracing::warn!(
        policy = %ctx.config.model.policy,
        reason = reason.as_str(),
        model,
        "prefill policy selection failed"
    );
    ApiError::PolicySelectionFailed {
        model: model.to_owned(),
    }
}

fn resolve_prefix_query(
    result: Result<sgl_kv_indexer::PrefixOutcome, sgl_kv_indexer::PrefixIndexError>,
    model: &str,
) -> Result<sgl_kv_indexer::PrefixOutcome, ApiError> {
    use sgl_kv_indexer::PrefixIndexError;
    match result {
        Ok(outcome) => Ok(outcome),
        Err(
            error @ (PrefixIndexError::Overloaded
            | PrefixIndexError::Timeout
            | PrefixIndexError::Unreachable),
        ) => {
            tracing::warn!(%model, error = %error, "KV Indexer unavailable; falling back to min-load routing");
            Ok(sgl_kv_indexer::PrefixOutcome::Empty)
        }
        Err(error @ PrefixIndexError::QueryTooLarge) => {
            tracing::warn!(%model, error = %error, "prompt exceeds the KV Indexer query size limit; falling back to min-load routing");
            Ok(sgl_kv_indexer::PrefixOutcome::Empty)
        }
        Err(error) => {
            tracing::warn!(%model, error = %error, "KV Indexer rejected the query");
            Err(ApiError::PolicySelectionFailed {
                model: model.to_string(),
            })
        }
    }
}

fn parse_optional_positive_u64_header(
    headers: &HeaderMap,
    name: &HeaderName,
    label: &str,
) -> Result<Option<u64>, ApiError> {
    let Some(value) = headers.get(name) else {
        return Ok(None);
    };
    let raw = value
        .to_str()
        .map_err(|_| ApiError::BadRequest(format!("{label} header must be ASCII")))?;
    let parsed = raw
        .parse::<u64>()
        .map_err(|_| ApiError::BadRequest(format!("{label} header must be a positive integer")))?;
    if parsed == 0 {
        return Err(ApiError::BadRequest(format!(
            "{label} header must be a positive integer"
        )));
    }
    Ok(Some(parsed))
}

fn parse_optional_positive_f64_header(
    headers: &HeaderMap,
    name: &HeaderName,
    label: &str,
) -> Result<Option<f64>, ApiError> {
    let Some(value) = headers.get(name) else {
        return Ok(None);
    };
    let raw = value
        .to_str()
        .map_err(|_| ApiError::BadRequest(format!("{label} header must be ASCII")))?;
    let parsed = raw
        .parse::<f64>()
        .map_err(|_| ApiError::BadRequest(format!("{label} header must be a positive number")))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(ApiError::BadRequest(format!(
            "{label} header must be a finite positive number"
        )));
    }
    Ok(Some(parsed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_indexer_degrades_to_empty_prefix_signal() {
        for error in [
            sgl_kv_indexer::PrefixIndexError::Overloaded,
            sgl_kv_indexer::PrefixIndexError::Timeout,
            sgl_kv_indexer::PrefixIndexError::Unreachable,
            sgl_kv_indexer::PrefixIndexError::QueryTooLarge,
        ] {
            assert_eq!(
                resolve_prefix_query(Err(error.clone()), "tiny").unwrap(),
                sgl_kv_indexer::PrefixOutcome::Empty,
                "{error} should degrade"
            );
        }
    }

    #[test]
    fn rejected_indexer_query_still_fails_selection() {
        assert!(matches!(
            resolve_prefix_query(
                Err(sgl_kv_indexer::PrefixIndexError::Rejected(
                    sgl_kv_indexer::RpcCode::InvalidArgument
                )),
                "tiny"
            ),
            Err(ApiError::PolicySelectionFailed { .. })
        ));
    }
}
