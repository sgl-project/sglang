// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::forward::{forward_chat_request, SelectedWorkers};
use super::nonempty_header;
use super::preparation::{parse_routing_fields, PreparedChatRequest};
use crate::buckets_reorg::{Bucket, BucketGroups, BucketResolver, EngineGroup};
use crate::discovery::ModelId;
use crate::policies_reorg::{PickError, PickRequest, Stage};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::state::LoadView;
use crate::workers::Worker;
use axum::body::Body;
use axum::http::{HeaderMap, Response};
use bytes::Bytes;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Bucket-first implementation selected by `AppContext::chat_routing`.
pub(super) async fn chat_completions(
    ctx: &AppContext,
    resolvers: &HashMap<ModelId, BucketResolver>,
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
    let resolver = resolvers
        .get(&model)
        .ok_or_else(|| ApiError::ModelNotFound(model.0.clone()))?;
    // Length-based routing needs tokenization even for load-only group policies.
    let request = PreparedChatRequest::prepare(ctx, model, fields, body, true)?;
    let input_tokens = request.input_token_count as u64;
    let expected_peak_tokens = request
        .max_output_tokens
        .map(|output| {
            input_tokens.checked_add(output).ok_or_else(|| {
                ApiError::BadRequest("input and output token counts overflow".into())
            })
        })
        .transpose()?;

    let buckets = resolver
        .resolve(input_tokens, expected_peak_tokens)
        .map_err(|error| selection_error(error, &request.model, None))?;
    if buckets.is_empty() {
        return Err(selection_error(
            PickError::NoMatchingBucket,
            &request.model,
            None,
        ));
    }
    let mut rejections: Option<Vec<_>> = None;
    let mut missing_stage = None;
    for bucket in buckets {
        match pick_bucket(ctx, &request, &headers, bucket, expected_peak_tokens).await {
            Ok(workers) => {
                // Dispatch only after this bucket supplies the entire plain or PD selection.
                return forward_chat_request(ctx, request, workers, headers, start).await;
            }
            Err((stage, error)) => {
                tracing::debug!(bucket = %bucket.id, ?stage, %error, "bucket selection failed");
                match error {
                    PickError::NoCandidates => missing_stage = Some(stage),
                    PickError::NoAdmissibleEngine(reasons) => {
                        rejections.get_or_insert_with(Vec::new).extend(reasons);
                    }
                    PickError::AdmissionRejected(reason) => {
                        rejections.get_or_insert_with(Vec::new).push(reason);
                    }
                    error => return Err(selection_error(error, &request.model, Some(stage))),
                }
            }
        }
    }
    // Preserve admission exhaustion even if a later bucket has no candidates.
    let error = match rejections {
        Some(reasons) => PickError::NoAdmissibleEngine(reasons),
        None => PickError::NoCandidates,
    };
    Err(selection_error(error, &request.model, missing_stage))
}

async fn pick_bucket(
    ctx: &AppContext,
    request: &PreparedChatRequest,
    headers: &HeaderMap,
    bucket: &Bucket,
    expected_peak_tokens: Option<u64>,
) -> Result<SelectedWorkers, (Stage, PickError)> {
    let (prefill, decode) = match &bucket.groups {
        BucketGroups::Plain(group) => (
            pick_engine(
                ctx,
                request,
                headers,
                bucket,
                group,
                Stage::Plain,
                expected_peak_tokens,
            )
            .await?,
            None,
        ),
        BucketGroups::Pd { prefill, decode } => {
            let prefill = pick_engine(
                ctx,
                request,
                headers,
                bucket,
                prefill,
                Stage::Prefill,
                expected_peak_tokens,
            )
            .await?;
            let decode = pick_engine(
                ctx,
                request,
                headers,
                bucket,
                decode,
                Stage::Decode,
                expected_peak_tokens,
            )
            .await?;
            (prefill, Some(decode))
        }
    };
    Ok(SelectedWorkers {
        prefill,
        decode,
        track_dispatch_timestamps: false,
    })
}

async fn pick_engine(
    ctx: &AppContext,
    request: &PreparedChatRequest,
    headers: &HeaderMap,
    bucket: &Bucket,
    group: &EngineGroup,
    stage: Stage,
    expected_peak_tokens: Option<u64>,
) -> Result<Arc<Worker>, (Stage, PickError)> {
    // One lazy report snapshot per stage, including its policy fallback/admission.
    let load = LoadView::new(&ctx.engine_load);
    let pick_request = PickRequest {
        bucket: &bucket.id,
        expected_peak_tokens,
        token_ids: request.tokens.as_ref().map(|tokens| tokens.ids.as_slice()),
        session_key: ctx
            .config
            .model
            .affinity
            .as_ref()
            .and_then(|config| nonempty_header(headers, &config.session_id_header)),
        routing_key: ctx
            .config
            .model
            .sticky
            .as_ref()
            .and_then(|config| nonempty_header(headers, &config.header_name)),
        ..PickRequest::new(
            &request.model,
            stage,
            request.input_token_count as u64,
            &load,
        )
    };
    group
        .pick(&ctx.registry, &pick_request)
        .await
        .map(|pick| pick.engine)
        .map_err(|error| (stage, error))
}

fn selection_error(error: PickError, model: &ModelId, stage: Option<Stage>) -> ApiError {
    tracing::warn!(%model, ?stage, %error, "reorg selection failed");
    match error {
        PickError::NoMatchingBucket => {
            ApiError::BadRequest("no bucket supports the requested token length".into())
        }
        PickError::NoCandidates => match stage {
            Some(Stage::Prefill) => ApiError::NoPrefillWorkersAvailable {
                model: model.0.clone(),
            },
            Some(Stage::Decode) => ApiError::NoDecodeWorkersAvailable {
                model: model.0.clone(),
            },
            _ => ApiError::NoHealthyWorkers {
                model: model.0.clone(),
            },
        },
        PickError::NoAdmissibleEngine(_) | PickError::AdmissionRejected(_) => {
            ApiError::PolicySelectionFailed {
                model: model.0.clone(),
            }
        }
        error => ApiError::Internal(error.into()),
    }
}
