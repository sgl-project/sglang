// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

mod forward;
mod preparation;
mod reorg;

use crate::buckets_reorg::BucketResolver;
use crate::config::{SessionAffinityMode, DEFAULT_MIN_LOAD_CHOICES};
use crate::discovery::{ModelId, WorkerId, WorkerMode};
use crate::policies::registry::{PdPoolResolver, PdResolveError};
use crate::policies::selection::{
    select_decode_peer, select_prefill_worker, DecodeSelectionInputs, PrefillSelectionInputs,
};
use crate::policies::{Policy, PrefixLookupResult};
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
use forward::{forward_request, RequestDurationGuard, SelectedWorkers};
use preparation::{
    parse_embedding_request, parse_routing_fields, PreparedRequest, CLASSIFY_PATH, EMBEDDINGS_PATH,
};
use std::collections::HashMap;
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
    let start = Instant::now();
    let mut fields = parse_routing_fields(&body)?;
    let model = ModelId(
        fields
            .model
            .take()
            .ok_or_else(|| ApiError::BadRequest("missing `model` field".into()))?,
    );
    let routing = ModelRouting::lookup(&ctx, &model)?;
    let request = PreparedRequest::chat(
        &ctx,
        model,
        fields,
        body,
        routing.needs_request_tokens(&ctx),
    )?;
    routing.dispatch(&ctx, request, headers, start).await
}

/// SGLang's native `/generate`: same request and response schema as the engine.
/// The body names no model, so it goes to the one this router serves.
pub async fn generate(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let start = Instant::now();
    let model = ModelId(ctx.config.model.id.clone());
    let routing = ModelRouting::lookup(&ctx, &model)?;
    let request = PreparedRequest::generate(&ctx, model, body)?;
    routing.dispatch(&ctx, request, headers, start).await
}

/// OpenAI `/v1/embeddings`, forwarded to the engine's with the same request and response.
pub async fn embeddings(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    embedding_input(ctx, EMBEDDINGS_PATH, headers, body).await
}

/// SGLang's `/v1/classify`, which takes the same `input` as embeddings.
pub async fn classify(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    embedding_input(ctx, CLASSIFY_PATH, headers, body).await
}

async fn embedding_input(
    ctx: Arc<AppContext>,
    path: &'static str,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let start = Instant::now();
    let (model, value) = parse_embedding_request(&body)?;
    let routing = ModelRouting::lookup(&ctx, &model)?;
    require_plain_workers(&ctx, &model, path)?;
    let request = PreparedRequest::embeddings(&ctx, path, model, body, value)?;
    routing.dispatch(&ctx, request, headers, start).await
}

/// SGLang's `/v1/rerank`, forwarded as sent to the model this router serves.
pub async fn rerank(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let start = Instant::now();
    let model = ModelId(ctx.config.model.id.clone());
    let routing = ModelRouting::lookup(&ctx, &model)?;
    require_plain_workers(&ctx, &model, "/v1/rerank")?;
    let request = PreparedRequest::rerank(model, body)?;
    routing.dispatch(&ctx, request, headers, start).await
}

/// Prefill and decode engines serve generation only.
fn require_plain_workers(ctx: &AppContext, model: &ModelId, path: &str) -> Result<(), ApiError> {
    let registered = ctx.registry.workers_for(model);
    if registered.iter().any(|w| w.mode() != WorkerMode::Plain) {
        return Err(ApiError::BadRequest(format!(
            "{path} is not served by prefill-decode workers"
        )));
    }
    Ok(())
}

/// A model's routing state, resolved before the request is prepared.
enum ModelRouting<'a> {
    Legacy(Arc<dyn Policy>),
    Reorg(&'a BucketResolver),
}

impl<'a> ModelRouting<'a> {
    fn lookup(ctx: &'a AppContext, model: &ModelId) -> Result<Self, ApiError> {
        let routing = match &ctx.chat_routing {
            ChatRouting::Legacy => ctx.policies.get(model).map(Self::Legacy),
            ChatRouting::Reorg(resolvers) => resolvers.get(model).map(Self::Reorg),
        };
        routing.ok_or_else(|| ApiError::ModelNotFound(model.0.clone()))
    }

    fn needs_request_tokens(&self, ctx: &AppContext) -> bool {
        match self {
            Self::Legacy(policy) => policy.needs_request_tokens() || ctx.config.model.dp_aware,
            // The same predicate gates `--no-tokenizer` at startup,
            // so a load-only bucket skips the body parse.
            Self::Reorg(resolver) => resolver.needs_request_tokens(),
        }
    }

    /// Select workers for `request` and forward it to them.
    async fn dispatch(
        &self,
        ctx: &AppContext,
        mut request: PreparedRequest,
        headers: HeaderMap,
        start: Instant,
    ) -> Result<Response<Body>, ApiError> {
        let workers = self.select_workers(ctx, &request, &headers, &[]).await?;
        let duration = RequestDurationGuard::new(ctx, &request.model, start);
        forward_request(ctx, &mut request, workers, headers, start, &duration).await
    }

    /// Pick a plain worker, or a prefill worker followed by a decode peer in PD mode,
    /// never one in `excluded`.
    async fn select_workers(
        &self,
        ctx: &AppContext,
        request: &PreparedRequest,
        headers: &HeaderMap,
        excluded: &[WorkerId],
    ) -> Result<SelectedWorkers, ApiError> {
        match self {
            Self::Legacy(policy) => {
                // Find healthy workers: the prefill pool in PD mode, otherwise the plain pool.
                let resolver = PdPoolResolver::new(Arc::clone(&ctx.registry)).excluding(excluded);
                let candidates = resolver
                    .prefill_candidates(&request.model)
                    .map_err(|error| pool_error(error, &request.model))?;
                select_workers(
                    ctx,
                    request,
                    headers,
                    policy.as_ref(),
                    &candidates,
                    &resolver,
                )
                .await
            }
            Self::Reorg(resolver) => {
                reorg::select_workers(ctx, resolver, request, headers, excluded).await
            }
        }
    }
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
    request: &PreparedRequest,
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

    let candidates = prefills_with_decode(ctx, request, candidates, resolver, &routing_context);
    let prefill = pick_prefill_worker(ctx, request, policy, &candidates, &routing_context)?;
    let decode = pick_decode_worker(ctx, request, &prefill, resolver, &routing_context, true)?;
    record_prefill_route(ctx, routing_context.prefix_matches.as_ref(), &prefill.url);
    Ok(SelectedWorkers {
        prefill,
        decode,
        track_dispatch_timestamps: policy.needs_dispatch_timestamps(),
    })
}

/// Credit the chosen prefill with the prompt's prefix until KV events confirm
/// it. Called only once the whole selection succeeded, so a request that is
/// never dispatched credits nobody.
fn record_prefill_route(ctx: &AppContext, signal: Option<&PrefixLookupResult>, prefill_url: &str) {
    if let (Some(provider), Some(signal)) = (&ctx.radix_tree_prefix_provider, signal) {
        provider.record_route(signal, prefill_url);
    }
}

/// Keep prefills whose version group has a decode that fits this request, so a
/// full group falls back to another. Decode selection has no side effects, while
/// a discarded prefill pick would already have bound affinity.
fn prefills_with_decode(
    ctx: &AppContext,
    request: &PreparedRequest,
    candidates: &[Arc<Worker>],
    resolver: &PdPoolResolver,
    routing: &RoutingContext<'_>,
) -> Vec<Arc<Worker>> {
    let first = candidates.first().map(|p| p.version_group());
    if candidates.iter().all(|p| Some(p.version_group()) == first) {
        return candidates.to_vec();
    }
    // Try strict admission across groups before relaxing capacity in any group.
    for allow_capacity_fallback in [false, true] {
        let mut fits = HashMap::new();
        let kept: Vec<_> = candidates
            .iter()
            .filter(|p| {
                *fits.entry(p.version_group()).or_insert_with(|| {
                    pick_decode_worker(ctx, request, p, resolver, routing, allow_capacity_fallback)
                        .is_ok()
                })
            })
            .cloned()
            .collect();
        if !kept.is_empty() {
            return kept;
        }
    }
    // With no group fitting, keep them all so the pick reports the real error.
    candidates.to_vec()
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
    prefix_matches: Option<PrefixLookupResult>,
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
    request: &PreparedRequest,
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
        request_sequence_tokens: request.sequence_token_count as u64,
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
    request: &PreparedRequest,
    prefill: &Worker,
    resolver: &PdPoolResolver,
    routing: &RoutingContext<'_>,
    allow_capacity_fallback: bool,
) -> Result<Option<Arc<Worker>>, ApiError> {
    if prefill.mode() != WorkerMode::Prefill {
        return Ok(None);
    }
    // Only a decode worker in the prefill's version group can receive its KV.
    let candidates = resolver
        .decode_peers(&request.model, prefill)
        .map_err(|error| pool_error(error, &request.model))?;
    let decode = select_decode_peer(&DecodeSelectionInputs {
        decode_policy_kind: ctx.config.model.decode_policy,
        allow_capacity_fallback,
        bucket_selector: ctx.bucket_selector.as_ref(),
        model_id: &request.model,
        prefill_url: &prefill.url,
        decode_workers: &candidates,
        request_input_tokens: request.input_token_count as u64,
        request_sequence_tokens: request.sequence_token_count as u64,
        requested_max_output_tokens: request.output_tokens,
        expected_peak_sequence_tokens: request.expected_peak_sequence_tokens,
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
    request: &PreparedRequest,
) -> Result<Option<PrefixLookupResult>, ApiError> {
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
            Some(PrefixLookupResult {
                outcome,
                query_blocks,
                block_hashes: None,
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

fn policy_selection_failed(
    ctx: &AppContext,
    model: &str,
    reason: PolicySelectionFailureReason,
) -> ApiError {
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
