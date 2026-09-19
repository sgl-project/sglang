// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

mod forward;
mod preparation;

use crate::buckets_reorg::SelectionRequest;
use crate::discovery::ModelId;
use crate::policies::registry::{PdPoolResolver, PdPools, PdResolveError};
use crate::policies_reorg::{PickError, PickRequest, Stage};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::PolicySelectionFailureReason;
use crate::state::{LoadView, PrefixMemo};
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
    let start = Instant::now();
    let mut fields = parse_routing_fields(&body)?;
    let model = ModelId(
        fields
            .model
            .take()
            .ok_or_else(|| ApiError::BadRequest("missing `model` field".into()))?,
    );
    if model.0 != ctx.config.model.id {
        return Err(ApiError::ModelNotFound(model.0));
    }
    let stage = first_stage(&ctx, &model)?;
    let request = PreparedChatRequest::prepare(
        &ctx,
        model,
        fields,
        body,
        ctx.buckets.pools.needs_request_tokens,
    )?;
    let workers = select_workers(&ctx, &request, &headers, stage).await?;
    // PD sends to both workers and returns the decode response.
    forward_chat_request(&ctx, request, workers, headers, start).await
}

/// Plain for a plain deployment, prefill for PD; an empty stage pool is an error.
fn first_stage(ctx: &AppContext, model: &ModelId) -> Result<Stage, ApiError> {
    let pools = PdPoolResolver::new(Arc::clone(&ctx.registry))
        .resolve(model)
        .map_err(|error| pool_error(error, model))?;
    match pools {
        PdPools::Plain { .. } => Ok(Stage::Plain),
        PdPools::Pd { prefill, .. } if prefill.is_empty() => {
            Err(ApiError::NoPrefillWorkersAvailable {
                model: model.0.clone(),
            })
        }
        PdPools::Pd { .. } => Ok(Stage::Prefill),
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

/// One pick per stage: a plain or prefill engine, then a decode peer in PD mode.
async fn select_workers(
    ctx: &AppContext,
    request: &PreparedChatRequest,
    headers: &HeaderMap,
    stage: Stage,
) -> Result<SelectedWorkers, ApiError> {
    let routing = RoutingContext::from_headers(ctx, headers)?;
    let load = LoadView::new(&ctx.engine_load);
    let prefix = PrefixMemo::new();
    let mut pick = PickRequest::new(
        &request.model,
        stage,
        request.input_token_count as u64,
        &load,
    );
    pick.token_ids = request.tokens.as_ref().map(|tokens| tokens.ids.as_slice());
    pick.prefix = Some(&prefix);
    pick.session_key = routing.session_id;
    pick.routing_key = routing.routing_key;
    let selection = SelectionRequest {
        pick,
        ttft_ms: routing.ttft_slo_ms,
        tokens_per_second: routing.tps_slo,
    };
    let prefill = ctx
        .buckets
        .pick(&selection)
        .await
        .map_err(|error| select_error(ctx, &request.model, error))?;
    ctx.metrics
        .record_policy_decision(&ctx.config.model.policy.to_string(), prefill.reason);
    let decode = match stage {
        Stage::Prefill => {
            let decode = SelectionRequest {
                pick: PickRequest {
                    stage: Stage::Decode,
                    expected_peak_tokens: request
                        .max_output_tokens
                        .map(|out| pick.input_tokens.saturating_add(out)),
                    ..pick
                },
                ..selection
            };
            let pick = ctx.buckets.pick(&decode).await.map_err(|_| {
                ApiError::NoDecodeWorkersAvailable {
                    model: request.model.0.clone(),
                }
            })?;
            Some(pick.engine)
        }
        _ => None,
    };
    Ok(SelectedWorkers {
        prefill: prefill.engine,
        decode,
        track_dispatch_timestamps: ctx.buckets.pools.needs_dispatch_timestamps,
    })
}

fn select_error(ctx: &AppContext, model: &ModelId, error: PickError) -> ApiError {
    let reason = match error {
        PickError::NoCandidates => PolicySelectionFailureReason::ProposalEmpty,
        PickError::NoAdmissibleEngine(_) | PickError::AdmissionRejected(_) => {
            PolicySelectionFailureReason::PrefillAdmissionExhausted
        }
        PickError::InvalidSignal(_) | PickError::OutsideCandidates(_) => {
            tracing::warn!(model = %model.0, %error, "prefill policy selection failed");
            return ApiError::PolicySelectionFailed {
                model: model.0.clone(),
            };
        }
    };
    policy_selection_failed(ctx, &model.0, reason)
}

struct RoutingContext<'a> {
    ttft_slo_ms: Option<u64>,
    tps_slo: Option<f64>,
    routing_key: Option<&'a str>,
    session_id: Option<&'a str>,
}

impl<'a> RoutingContext<'a> {
    /// Header-derived selection inputs; SLO headers only matter with buckets.
    fn from_headers(ctx: &AppContext, headers: &'a HeaderMap) -> Result<Self, ApiError> {
        let (ttft_slo_ms, tps_slo) = if ctx.config.model.bucket_config.is_some() {
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
