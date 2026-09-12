// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `POST /generate` — the sglang-native generation surface.
//!
//! The body is the sglang `GenerateReqInput` shape (`text` or `input_ids`,
//! `sampling_params`, `rid`, …) PLUS a top-level `model` key naming the
//! endpoint: the shared gateway exposes exactly this surface and rewrites
//! `body.model` to the canonical HF id before forwarding. Everything the
//! handler does — routing, admission, the per-model sampling controls,
//! PD bootstrap injection, abort-by-rid — is shared with the chat handler
//! via [`chat::chat_completions_inner`]; only the surface-varying reads and
//! writes differ (see [`crate::server::routes::surface`]).

use super::chat;
use crate::server::app::RequestPhaseCell;
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use axum::body::Body;
use axum::extract::{Extension, State};
use axum::http::{HeaderMap, Response};
use bytes::Bytes;
use std::sync::Arc;

/// POST /generate handler. Thin delegator to
/// [`chat::chat_completions_inner`] with [`Surface::Generate`] — see
/// [`chat::chat_completions`] for the logging/counting contract (all of it
/// happens in the outermost `access_log_and_record` middleware).
pub(crate) async fn generate(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    phase: Option<Extension<Arc<RequestPhaseCell>>>,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    chat::chat_completions_inner(
        ctx,
        headers,
        phase.map(|Extension(p)| p),
        body,
        super::surface::Surface::Generate,
    )
    .await
}
