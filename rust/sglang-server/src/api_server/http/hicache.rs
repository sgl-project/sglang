//! HiCache routes with dedicated state shared with the gRPC adapter.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use sglang_api_types::api::v1 as api;

use super::auth::DeclaredRoute;
use crate::api_server::handlers::{HiCacheError, HiCacheManager};

pub(super) fn routes(manager: Arc<HiCacheManager>) -> Router {
    Router::new()
        .declared_route(
            "/hicache/storage-backend",
            get(status).put(attach).delete(detach),
        )
        .declared_route("/hicache/storage-backend/clear", post(clear))
        .with_state(manager)
}

async fn status(State(manager): State<Arc<HiCacheManager>>) -> Response {
    Json(manager.status().await).into_response()
}

async fn attach(
    State(manager): State<Arc<HiCacheManager>>,
    body: Result<Json<api::AttachHiCacheStorageRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(error) => return (StatusCode::UNPROCESSABLE_ENTITY, error.body_text()).into_response(),
    };
    if body.hicache_storage_backend.is_none() {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            "missing field `hicache_storage_backend`",
        )
            .into_response();
    }
    storage_response(manager.attach(body).await)
}

async fn detach(State(manager): State<Arc<HiCacheManager>>) -> Response {
    storage_response(manager.detach().await)
}

async fn clear(State(manager): State<Arc<HiCacheManager>>) -> Response {
    storage_response(manager.clear().await.map_err(HiCacheError::from))
}

fn storage_response(result: Result<api::HiCacheStorageResponse, HiCacheError>) -> Response {
    match result {
        Ok(output) => {
            let status = if output.success {
                StatusCode::OK
            } else {
                StatusCode::BAD_REQUEST
            };
            (status, Json(output)).into_response()
        }
        Err(HiCacheError::AdminKeyMissing) => super::auth::admin_key_missing(),
        Err(HiCacheError::Core(error)) => (error.http_status(), error.to_string()).into_response(),
    }
}
