// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! PD runtime-role router control endpoints.

use crate::discovery::{WorkerId, WorkerMode};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::workers::Worker;
use axum::extract::State;
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

pub async fn list_workers(State(ctx): State<Arc<AppContext>>) -> Json<RouterWorkersResponse> {
    let mut workers: Vec<RouterWorkerStatus> = ctx
        .registry
        .all()
        .into_iter()
        .map(|w| snapshot_worker(&w))
        .collect();
    workers.sort_by(|a, b| a.worker_id.cmp(&b.worker_id));
    Json(RouterWorkersResponse { workers })
}

pub async fn set_worker_drain(
    State(ctx): State<Arc<AppContext>>,
    Json(req): Json<RouterWorkerDrainReq>,
) -> Result<Json<RouterWorkerControlResponse>, ApiError> {
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
    Json(req): Json<RouterWorkerRoleReq>,
) -> Result<Json<RouterWorkerControlResponse>, ApiError> {
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

    fn app_with_worker() -> axum::Router {
        let ctx = AppContext::stub();
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId("w1".into()),
                url: "http://127.0.0.1:30000".into(),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("stub-model".into())],
                bootstrap_port: None,
            })
            .expect("test worker accepted");
        crate::server::app::build_router(Arc::new(ctx))
    }

    #[tokio::test]
    async fn drain_endpoint_marks_worker_draining() {
        let app = app_with_worker();
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/pd_flip/router/worker/drain")
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
        let app = app_with_worker();
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/pd_flip/router/worker/role")
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
}
