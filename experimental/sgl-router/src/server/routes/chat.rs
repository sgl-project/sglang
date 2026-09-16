// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

mod dispatch;
mod request;
mod routing;

use crate::discovery::ModelId;
use crate::policies::registry::{PdPoolResolver, PdResolveError};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Response};
use bytes::Bytes;
use request::{parse_probe, ChatRequest};
use std::sync::Arc;
use std::time::Instant;

/// Maximum buffered request body, including base64 multimodal inputs (32 MiB).
pub const MAX_CHAT_BODY_BYTES: usize = 32 << 20;

/// Validate, select workers, and forward a chat-completions request.
pub async fn chat_completions(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let start = Instant::now();
    let mut probe = parse_probe(&body)?;
    let model = ModelId(
        probe
            .model
            .take()
            .ok_or_else(|| ApiError::BadRequest("missing `model` field".into()))?,
    );
    let resolver = PdPoolResolver::new(Arc::clone(&ctx.registry));
    let candidates = resolver
        .prefill_candidates(&model)
        .map_err(|error| pool_error(error, &model))?;
    let policy = ctx
        .policies
        .get(&model)
        .ok_or_else(|| ApiError::ModelNotFound(model.0.clone()))?;

    // Resolve the model before enforcing its sampling contract.
    let request = ChatRequest::prepare(&ctx, model, probe, body, policy.needs_request_tokens())?;
    let workers = routing::select_workers(
        &ctx,
        &request,
        &headers,
        policy.as_ref(),
        &candidates,
        &resolver,
    )
    .await?;
    dispatch::forward(&ctx, request, workers, headers, start).await
}

fn pool_error(error: PdResolveError, model: &ModelId) -> ApiError {
    let model = model.0.clone();
    match error {
        PdResolveError::NoHealthyWorkers => ApiError::NoHealthyWorkers { model },
        PdResolveError::NoPrefillWorkersAvailable => ApiError::NoPrefillWorkersAvailable { model },
        PdResolveError::NoDecodeWorkersAvailable => ApiError::NoDecodeWorkersAvailable { model },
    }
}
