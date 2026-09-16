// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::pool_error;
use super::request::ChatRequest;
use crate::config::SessionAffinityMode;
use crate::discovery::WorkerMode;
use crate::policies::kv_events::{compute_block_hashes, compute_block_hashes_bigram};
use crate::policies::registry::PdPoolResolver;
use crate::policies::selection::{
    select_decode_peer, select_prefill_worker, DecodeSelectionInputs, PrefillSelectionInputs,
};
use crate::policies::{ExternalPrefixSignal, Policy};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::PolicySelectionFailureReason;
use crate::workers::Worker;
use axum::http::{HeaderMap, HeaderName};
use std::sync::Arc;

const X_SGL_TTFT_SLO_MS: HeaderName = HeaderName::from_static("x-sgl-ttft-slo-ms");
const X_SGL_TPS_SLO: HeaderName = HeaderName::from_static("x-sgl-tps-slo");

pub(super) struct SelectedWorkers {
    pub(super) prefill: Arc<Worker>,
    pub(super) decode: Option<Arc<Worker>>,
    pub(super) timestamped: bool,
}

pub(super) async fn select_workers(
    ctx: &AppContext,
    request: &ChatRequest,
    headers: &HeaderMap,
    policy: &dyn Policy,
    candidates: &[Arc<Worker>],
    resolver: &PdPoolResolver,
) -> Result<SelectedWorkers, ApiError> {
    let external_prefix = external_prefix(ctx, request).await?;
    let request_input_tokens = request.prefill_load as u64;
    let needs_load_snapshot = policy.needs_load_snapshot()
        || candidates
            .iter()
            .any(|worker| worker.mode() == WorkerMode::Prefill);
    let load_snapshot =
        needs_load_snapshot.then(|| ctx.engine_load.capture_snapshot(std::time::Instant::now()));
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
        .and_then(|s| headers.get(s.header_name.as_str()))
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty());
    let affinity = ctx.config.model.affinity.as_ref();
    let session_id = affinity
        .and_then(|config| headers.get(config.session_id_header.as_str()))
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty());
    let session_affinity_mode = affinity
        .map(|config| config.session_affinity_mode)
        .unwrap_or(SessionAffinityMode::Bucket);
    let worker_queue_limit = affinity.and_then(|config| config.worker_queue_limit);
    let saturation_queue_floor = affinity.and_then(|config| config.saturation_queue_floor);
    let worker = select_prefill_worker(&PrefillSelectionInputs {
        policy,
        policy_kind: ctx.config.model.policy,
        bucket_selector: ctx.bucket_selector.as_ref(),
        metrics: ctx.metrics.as_ref(),
        model_id: &request.model,
        body: Some(&request.body),
        routing_key,
        session_id,
        request_input_tokens,
        request_tokens: request.tokens.as_ref().map(|tokens| tokens.ids.as_slice()),
        external_prefix: external_prefix.as_ref(),
        load_snapshot: load_snapshot.as_ref(),
        workers: candidates,
        ttft_slo_ms,
        tps_slo,
        session_affinity_mode,
        worker_queue_limit,
        saturation_queue_floor,
    })
    .map_err(|reason| policy_selection_failed(ctx, &request.model.0, reason))?;

    let decode_peer: Option<Arc<Worker>> = if worker.mode() == WorkerMode::Prefill {
        let decode_workers = resolver
            .decode_candidates(&request.model)
            .map_err(|error| pool_error(error, &request.model))?;
        Some(
            select_decode_peer(&DecodeSelectionInputs {
                decode_policy_kind: ctx.config.model.decode_policy,
                bucket_selector: ctx.bucket_selector.as_ref(),
                model_id: &request.model,
                prefill_url: &worker.url,
                decode_workers: &decode_workers,
                request_input_tokens,
                requested_max_output_tokens: request.max_output_tokens,
                ttft_slo_ms,
                tps_slo,
                load_snapshot: load_snapshot.as_ref(),
            })
            .ok_or_else(|| ApiError::NoDecodeWorkersAvailable {
                model: request.model.0.clone(),
            })?,
        )
    } else {
        None
    };
    Ok(SelectedWorkers {
        prefill: worker,
        decode: decode_peer,
        timestamped: policy.needs_dispatch_timestamps(),
    })
}

async fn external_prefix(
    ctx: &AppContext,
    request: &ChatRequest,
) -> Result<Option<ExternalPrefixSignal>, ApiError> {
    let signal = match (
        ctx.prefix_index.as_ref(),
        request.tokens.as_ref(),
        ctx.block_size_oracle.get(),
    ) {
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
mod tests;
