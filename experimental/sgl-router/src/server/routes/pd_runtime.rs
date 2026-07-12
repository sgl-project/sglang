// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! PD runtime-role router control endpoints.

use crate::discovery::{WorkerId, WorkerMode};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::workers::Worker;
use axum::extract::State;
use axum::http::header::AUTHORIZATION;
use axum::http::HeaderMap;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize)]
pub struct RouterWorkerDrainReq {
    pub worker_id: String,
    pub draining: bool,
}

#[derive(Debug, Deserialize)]
pub struct RouterWorkerRoleReq {
    pub worker_id: String,
    pub role: WorkerMode,
    pub bootstrap_port: Option<u16>,
    #[serde(default)]
    pub draining: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RouterWorkerStatus {
    pub worker_id: String,
    pub url: String,
    pub role: WorkerMode,
    pub draining: bool,
    pub active_load: usize,
    pub bootstrap_port: Option<u16>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RouterWorkerControlResponse {
    pub success: bool,
    pub message: String,
    pub worker: Option<RouterWorkerStatus>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RouterWorkersResponse {
    pub workers: Vec<RouterWorkerStatus>,
}

fn snapshot_worker(w: &Arc<Worker>) -> RouterWorkerStatus {
    RouterWorkerStatus {
        worker_id: w.id.0.clone(),
        url: w.url.clone(),
        role: w.mode(),
        draining: w.is_draining(),
        active_load: w.active_load(),
        bootstrap_port: w.bootstrap_port(),
    }
}

fn get_worker(ctx: &AppContext, worker_id: &str) -> Result<Arc<Worker>, ApiError> {
    ctx.registry
        .get(&WorkerId(worker_id.to_string()))
        .ok_or_else(|| ApiError::BadRequest(format!("unknown worker_id: {worker_id}")))
}

fn require_admin(headers: &HeaderMap, ctx: &AppContext) -> Result<(), ApiError> {
    let key = ctx
        .config
        .server
        .pd_flip_router_admin_api_key
        .as_ref()
        .ok_or(ApiError::Unauthorized)?;
    let mut values = headers.get_all(AUTHORIZATION).iter();
    let actual = values.next().ok_or(ApiError::Unauthorized)?;
    if values.next().is_some() {
        return Err(ApiError::Unauthorized);
    }
    let actual = actual.as_bytes();
    if actual.len() < 8
        || !actual[..6].eq_ignore_ascii_case(b"Bearer")
        || actual[6] != b' '
        || !constant_time_eq(&actual[7..], key.expose().as_bytes())
    {
        return Err(ApiError::Unauthorized);
    }
    Ok(())
}

fn constant_time_eq(actual: &[u8], expected: &[u8]) -> bool {
    let max_len = actual.len().max(expected.len());
    let mut difference = actual.len() ^ expected.len();
    for index in 0..max_len {
        let left = actual.get(index).copied().unwrap_or(0);
        let right = expected.get(index).copied().unwrap_or(0);
        difference |= usize::from(left ^ right);
    }
    difference == 0
}

pub async fn list_workers(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
) -> Result<Json<RouterWorkersResponse>, ApiError> {
    require_admin(&headers, &ctx)?;
    let mut workers: Vec<RouterWorkerStatus> = ctx
        .registry
        .all()
        .into_iter()
        .map(|w| snapshot_worker(&w))
        .collect();
    workers.sort_by(|a, b| a.worker_id.cmp(&b.worker_id));
    Ok(Json(RouterWorkersResponse { workers }))
}

pub async fn set_worker_drain(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    Json(req): Json<RouterWorkerDrainReq>,
) -> Result<Json<RouterWorkerControlResponse>, ApiError> {
    require_admin(&headers, &ctx)?;
    let worker = get_worker(&ctx, &req.worker_id)?;
    worker.set_draining(req.draining);
    let snapshot = snapshot_worker(&worker);
    Ok(Json(RouterWorkerControlResponse {
        success: true,
        message: if req.draining {
            format!("worker {} is draining", req.worker_id)
        } else {
            format!("worker {} is admitting", req.worker_id)
        },
        worker: Some(snapshot),
    }))
}

pub async fn set_worker_role(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    Json(req): Json<RouterWorkerRoleReq>,
) -> Result<Json<RouterWorkerControlResponse>, ApiError> {
    require_admin(&headers, &ctx)?;
    let worker = get_worker(&ctx, &req.worker_id)?;
    worker.set_runtime_role(req.role, req.bootstrap_port);
    if let Some(draining) = req.draining {
        worker.set_draining(draining);
    }
    let snapshot = snapshot_worker(&worker);
    Ok(Json(RouterWorkerControlResponse {
        success: true,
        message: format!("worker {} role set to {:?}", req.worker_id, req.role),
        worker: Some(snapshot),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerSpec};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn app_with_worker(admin_key: Option<&str>) -> (axum::Router, Arc<AppContext>) {
        let mut ctx = AppContext::stub();
        ctx.config.server.pd_flip_router_admin_api_key =
            admin_key.map(crate::config::SecretString::new);
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId("w1".into()),
                url: "http://127.0.0.1:30000".into(),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("stub-model".into())],
                bootstrap_port: None,
            })
            .expect("test worker accepted");
        let ctx = Arc::new(ctx);
        (crate::server::app::build_router(Arc::clone(&ctx)), ctx)
    }

    #[tokio::test]
    async fn drain_endpoint_marks_worker_draining() {
        let (app, _) = app_with_worker(Some("test-secret"));
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/pd_flip/router/worker/drain")
                    .header("authorization", "Bearer test-secret")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"worker_id":"w1","draining":true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let body: RouterWorkerControlResponse = serde_json::from_slice(&bytes).unwrap();
        let worker = body.worker.unwrap();
        assert!(worker.draining);
        assert_eq!(worker.role, WorkerMode::Decode);
    }

    #[tokio::test]
    async fn role_endpoint_updates_role_and_bootstrap_port() {
        let (app, _) = app_with_worker(Some("test-secret"));
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/pd_flip/router/worker/role")
                    .header("authorization", "Bearer test-secret")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"worker_id":"w1","role":"prefill","bootstrap_port":8997,"draining":false}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let body: RouterWorkerControlResponse = serde_json::from_slice(&bytes).unwrap();
        let worker = body.worker.unwrap();
        assert_eq!(worker.role, WorkerMode::Prefill);
        assert_eq!(worker.bootstrap_port, Some(8997));
        assert!(!worker.draining);
    }

    #[tokio::test]
    async fn missing_admin_key_is_unauthorized() {
        let (app, _) = app_with_worker(None);
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/pd_flip/router/workers")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn missing_bearer_is_unauthorized() {
        let (app, _) = app_with_worker(Some("test-secret"));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/pd_flip/router/workers")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn wrong_bearer_is_unauthorized_without_mutation() {
        let (app, ctx) = app_with_worker(Some("test-secret"));
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/pd_flip/router/worker/drain")
                    .header("authorization", "Bearer wrong")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"worker_id":"w1","draining":true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
        let worker = get_worker(&ctx, "w1").unwrap();
        assert!(!worker.is_draining(), "unauthorized request mutated worker");
    }

    #[tokio::test]
    async fn correct_bearer_authorizes_controls() {
        let (app, _) = app_with_worker(Some("test-secret"));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/pd_flip/router/workers")
                    .header("authorization", "Bearer test-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn bearer_scheme_is_ascii_case_insensitive() {
        let (app, _) = app_with_worker(Some("test-secret"));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/pd_flip/router/workers")
                    .header("authorization", "bEaReR test-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn duplicate_authorization_headers_are_rejected() {
        let (app, _) = app_with_worker(Some("test-secret"));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/pd_flip/router/workers")
                    .header("authorization", "Bearer test-secret")
                    .header("authorization", "Bearer test-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn malformed_or_whitespace_bearers_are_rejected() {
        for value in [
            "Basic test-secret",
            "Bearer  test-secret",
            "Bearer\ttest-secret",
            "Bearer test-secret extra",
        ] {
            let (app, _) = app_with_worker(Some("test-secret"));
            let res = app
                .oneshot(
                    Request::builder()
                        .uri("/pd_flip/router/workers")
                        .header("authorization", value)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(res.status(), StatusCode::UNAUTHORIZED, "value={value:?}");
        }
    }
}
