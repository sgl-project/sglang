// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::server::app_context::AppContext;
use crate::server::routes::chat::MAX_CHAT_BODY_BYTES;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use axum::Router;
use std::sync::Arc;

pub fn build_router(ctx: Arc<AppContext>) -> Router {
    // Per-route body-size cap on /v1/chat/completions. Wired at the route
    // level (not the app level) so /v1/tokenize and /v1/detokenize can pick
    // their own limits later without coupling. axum's `Bytes` extractor
    // enforces the limit and returns 413 PAYLOAD_TOO_LARGE before the
    // handler runs.
    Router::new()
        .route("/healthz", get(crate::server::routes::health::healthz))
        .route("/readyz", get(crate::server::routes::health::readyz))
        .route(
            "/v1/models",
            get(crate::server::routes::models::list_models),
        )
        .route(
            "/v1/tokenize",
            post(crate::server::routes::tokenize::tokenize),
        )
        .route(
            "/v1/detokenize",
            post(crate::server::routes::tokenize::detokenize),
        )
        .route(
            "/v1/chat/completions",
            post(crate::server::routes::chat::chat_completions)
                .layer(DefaultBodyLimit::max(MAX_CHAT_BODY_BYTES)),
        )
        .with_state(ctx)
}
