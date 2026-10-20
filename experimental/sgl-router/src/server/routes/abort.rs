// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Request-abort endpoint.

use crate::server::app_context::AppContext;
use crate::workers::worker::Worker;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{self, StreamExt};
use reqwest::Client;
use serde::Serialize;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

/// Bound the number of simultaneous abort calls for large worker fleets.
const MAX_CONCURRENT_ABORTS: usize = 32;

#[derive(Serialize)]
struct FailedWorker {
    worker: String,
    error: String,
}

#[derive(Serialize)]
struct AbortFailure {
    error: String,
    failed: Vec<FailedWorker>,
    total_workers: usize,
}

/// `POST /abort_request` — forward an engine abort request to every registered
/// worker.
///
/// A request id is not permanently mapped to one worker in the router, and a
/// disaggregated request can occupy both prefill and decode workers. Broadcasting
/// is therefore necessary for both single-request aborts and `abort_all`.
/// Workers that do not own the request treat the call as a no-op.
///
/// The JSON body is forwarded without imposing a router-specific schema so the
/// endpoint stays compatible with all fields accepted by the engine's
/// `AbortReq`. The caller's `Authorization` header is forwarded as well.
///
/// Returns an empty `200 OK`, matching the engine endpoint, when every worker
/// succeeds (or no workers are registered). If any worker cannot be reached or
/// returns a non-success status, returns `502 BAD_GATEWAY` with a per-worker
/// failure breakdown.
pub async fn abort_request(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let workers = ctx.registry.all();
    let total_workers = workers.len();

    if workers.is_empty() {
        tracing::warn!("abort_request called but no workers are registered");
        return StatusCode::OK.into_response();
    }

    let failed = fan_out_abort(
        &workers,
        ctx.proxy.admin_client(),
        ctx.proxy.request_timeout,
        headers.get(header::AUTHORIZATION),
        &body,
    )
    .await;

    if failed.is_empty() {
        tracing::info!(total_workers, "abort_request: all workers acknowledged");
        StatusCode::OK.into_response()
    } else {
        tracing::warn!(
            total_workers,
            failed = failed.len(),
            "abort_request: some workers failed to acknowledge",
        );
        (
            StatusCode::BAD_GATEWAY,
            Json(AbortFailure {
                error: format!(
                    "abort request failed on {} of {} workers",
                    failed.len(),
                    total_workers
                ),
                failed,
                total_workers,
            }),
        )
            .into_response()
    }
}

async fn fan_out_abort(
    workers: &[Arc<Worker>],
    client: &Client,
    timeout: Duration,
    auth: Option<&HeaderValue>,
    body: &Value,
) -> Vec<FailedWorker> {
    let urls: Vec<String> = workers.iter().map(|worker| worker.url.clone()).collect();

    let outcomes = stream::iter(urls)
        .map(|url| {
            let mut request = client
                .post(format!("{}/abort_request", url.trim_end_matches('/')))
                .timeout(timeout)
                .json(body);
            if let Some(auth) = auth {
                request = request.header(header::AUTHORIZATION, auth);
            }
            async move { (url, request.send().await) }
        })
        .buffer_unordered(MAX_CONCURRENT_ABORTS)
        .collect::<Vec<_>>()
        .await;

    outcomes
        .into_iter()
        .filter_map(|(url, result)| match result {
            Ok(response) if response.status().is_success() => None,
            Ok(response) => Some(FailedWorker {
                worker: url,
                error: format!("HTTP {}", response.status()),
            }),
            Err(error) => Some(FailedWorker {
                worker: url,
                error: format!("{:#}", anyhow::Error::new(error)),
            }),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use axum::body::Body;
    use axum::http::Request;
    use axum::routing::post;
    use axum::Router;
    use http_body_util::BodyExt;
    use tokio::net::TcpListener;
    use tokio::sync::{mpsc, oneshot};
    use tower::ServiceExt;

    async fn spawn_fake_worker(
        status: StatusCode,
    ) -> (
        String,
        mpsc::UnboundedReceiver<(HeaderMap, Value)>,
        oneshot::Sender<()>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        let app = Router::new().route(
            "/abort_request",
            post(move |headers: HeaderMap, Json(body): Json<Value>| {
                let request_tx = request_tx.clone();
                async move {
                    request_tx.send((headers, body)).unwrap();
                    status
                }
            }),
        );
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await;
        });
        (url, request_rx, shutdown_tx)
    }

    fn ctx_with_workers(workers: &[(&str, WorkerMode)]) -> Arc<AppContext> {
        let ctx = AppContext::stub();
        for (index, (url, mode)) in workers.iter().enumerate() {
            ctx.registry
                .add(WorkerSpec {
                    id: WorkerId(format!("w-{index}")),
                    url: (*url).to_string(),
                    mode: *mode,
                    model_ids: vec![ModelId("stub-model".into())],
                    bootstrap_port: (*mode == WorkerMode::Prefill).then_some(8998),
                    ..Default::default()
                })
                .expect("worker accepted");
        }
        Arc::new(ctx)
    }

    async fn post_abort(ctx: Arc<AppContext>, body: Value) -> Response {
        crate::server::app::build_router(ctx)
            .oneshot(
                Request::post("/abort_request")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn broadcasts_body_to_plain_prefill_and_decode_workers() {
        let (plain_url, mut plain_rx, _s1) = spawn_fake_worker(StatusCode::OK).await;
        let (prefill_url, mut prefill_rx, _s2) = spawn_fake_worker(StatusCode::OK).await;
        let (decode_url, mut decode_rx, _s3) = spawn_fake_worker(StatusCode::OK).await;
        let body = serde_json::json!({
            "rid": "request-1",
            "abort_all": false,
            "abort_message": "client disconnected"
        });
        let response = post_abort(
            ctx_with_workers(&[
                (&plain_url, WorkerMode::Plain),
                (&prefill_url, WorkerMode::Prefill),
                (&decode_url, WorkerMode::Decode),
            ]),
            body.clone(),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert!(response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .is_empty());
        assert_eq!(plain_rx.recv().await.unwrap().1, body);
        assert_eq!(prefill_rx.recv().await.unwrap().1, body);
        assert_eq!(decode_rx.recv().await.unwrap().1, body);
    }

    #[tokio::test]
    async fn forwards_caller_authorization() {
        let (url, mut requests, _shutdown) = spawn_fake_worker(StatusCode::OK).await;
        let app = crate::server::app::build_router(ctx_with_workers(&[(&url, WorkerMode::Plain)]));
        let response = app
            .oneshot(
                Request::post("/abort_request")
                    .header(header::CONTENT_TYPE, "application/json")
                    .header(header::AUTHORIZATION, "Bearer secret")
                    .body(Body::from(r#"{"rid":"request-1"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let headers = requests.recv().await.unwrap().0;
        assert_eq!(headers.get(header::AUTHORIZATION).unwrap(), "Bearer secret");
    }

    #[tokio::test]
    async fn partial_failure_returns_502_with_breakdown() {
        let (ok_url, _ok_rx, _s1) = spawn_fake_worker(StatusCode::OK).await;
        let (failed_url, _failed_rx, _s2) =
            spawn_fake_worker(StatusCode::INTERNAL_SERVER_ERROR).await;
        let response = post_abort(
            ctx_with_workers(&[
                (&ok_url, WorkerMode::Plain),
                (&failed_url, WorkerMode::Plain),
            ]),
            serde_json::json!({"rid": "request-1"}),
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["total_workers"], 2);
        assert_eq!(body["failed"].as_array().unwrap().len(), 1);
        assert_eq!(body["failed"][0]["worker"], failed_url);
        assert!(body["failed"][0]["error"].as_str().unwrap().contains("500"));
    }

    #[tokio::test]
    async fn empty_registry_is_a_successful_no_op() {
        let response = post_abort(
            Arc::new(AppContext::stub()),
            serde_json::json!({"abort_all": true}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .is_empty());
    }
}
