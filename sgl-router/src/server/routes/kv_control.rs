//! Authenticated Router-level lifecycle/selector control, separate from Engine's
//! same-named endpoint, which accepts resolved physical-owner commands.
use crate::server::app_context::AppContext;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};
use std::sync::Arc;
async fn call(ctx: Arc<AppContext>, headers: HeaderMap, mut body: Value, event: bool) -> Response {
    let Some(runtime) = &ctx.orchestration else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"success":false,"error":{"code":"disabled"}})),
        )
            .into_response();
    };
    if !runtime.authorized(&headers) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"success":false,"error":{"code":"unauthorized"}})),
        )
            .into_response();
    }
    if event {
        body["kind"] = json!("session_event");
    }
    let result = runtime.api(body).await;
    let status = if result["success"] == true {
        StatusCode::OK
    } else {
        match result["error"]["code"].as_str() {
            Some("capacity") => StatusCode::TOO_MANY_REQUESTS,
            Some("unknown") => StatusCode::SERVICE_UNAVAILABLE,
            _ => StatusCode::BAD_REQUEST,
        }
    };
    (status, Json(result)).into_response()
}
pub async fn control(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    call(ctx, headers, body, false).await
}
pub async fn events(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    call(ctx, headers, body, true).await
}

pub async fn capabilities(State(ctx): State<Arc<AppContext>>, headers: HeaderMap) -> Response {
    call(ctx, headers, json!({"kind":"capabilities"}), false).await
}
