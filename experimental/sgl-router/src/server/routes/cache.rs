// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cache-management admin endpoints.

use crate::server::app_context::AppContext;
use crate::workers::worker::Worker;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{self, StreamExt};
use reqwest::Client;
use serde::Serialize;
use std::sync::Arc;
use std::time::Duration;

/// Cap on concurrent in-flight `/flush_cache` requests. Bounds how many
/// flushes are issued at once when a large fleet is flushed; the rest queue
/// and run as slots free up.
const MAX_CONCURRENT_FLUSH: usize = 32;

/// One worker that failed to flush, with a human-readable reason.
#[derive(Serialize)]
pub struct FailedWorker {
    pub worker: String,
    pub error: String,
}

/// Per-worker breakdown of a `/flush_cache` fan-out. `total_workers` is the
/// registry size snapshotted at call time; every registered worker is
/// attempted, so `successful.len() + failed.len() == total_workers`.
/// `message` is a human/log summary — the HTTP status is authoritative.
#[derive(Serialize)]
pub struct FlushCacheResult {
    pub successful: Vec<String>,
    pub failed: Vec<FailedWorker>,
    pub total_workers: usize,
    pub message: String,
}

impl FlushCacheResult {
    /// Build a result from a completed fan-out, deriving `message` from the
    /// outcome counts so the count/message coherence lives in one place
    /// rather than at each call site.
    fn from_outcomes(
        total_workers: usize,
        successful: Vec<String>,
        failed: Vec<FailedWorker>,
    ) -> Self {
        let message = if total_workers == 0 {
            "No workers registered; nothing to flush".to_string()
        } else if failed.is_empty() {
            format!(
                "Successfully flushed cache on all {} workers",
                total_workers
            )
        } else {
            format!(
                "Cache flush: {} succeeded, {} failed",
                successful.len(),
                failed.len()
            )
        };
        Self {
            successful,
            failed,
            total_workers,
            message,
        }
    }
}

/// `POST /flush_cache` — fan SGLang's `/flush_cache` admin call out to every
/// registered worker and report a per-worker breakdown.
///
/// Targets the whole fleet (plain, prefill, and decode workers all hold KV
/// cache), not just one model's pool. Deliberately **bypasses the circuit
/// breaker**: an operator flushing caches wants every worker hit — including
/// ones whose breaker is open — and recording breaker success/failure for an
/// out-of-band admin call would skew the state the request router uses to
/// pick workers.
///
/// Status: `200 OK` when every worker flushed successfully (or the fleet is
/// empty); `502 BAD_GATEWAY` when at least one worker failed. The JSON body
/// always carries the full breakdown so a partial failure is actionable.
pub async fn flush_cache(State(ctx): State<Arc<AppContext>>) -> Response {
    let workers = ctx.registry.all();
    let total_workers = workers.len();

    if workers.is_empty() {
        // A flush against an empty fleet is a no-op, but it usually means a
        // discovery/config problem (the router knows of no workers), so warn
        // rather than stay silent.
        tracing::warn!("flush_cache called but no workers are registered");
        return (
            StatusCode::OK,
            Json(FlushCacheResult::from_outcomes(0, Vec::new(), Vec::new())),
        )
            .into_response();
    }

    let (successful, failed) =
        fan_out_flush(&workers, &ctx.proxy.client, ctx.proxy.request_timeout).await;

    // Partial failure is an operational event an operator needs to see at the
    // common production log level — match the rest of the router, which warns
    // on upstream failures.
    if failed.is_empty() {
        tracing::info!(total_workers, "flush_cache: all workers flushed");
    } else {
        tracing::warn!(
            total_workers,
            succeeded = successful.len(),
            failed = failed.len(),
            "flush_cache: some workers failed to flush",
        );
    }

    let status = if failed.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::BAD_GATEWAY
    };

    (
        status,
        Json(FlushCacheResult::from_outcomes(
            total_workers,
            successful,
            failed,
        )),
    )
        .into_response()
}

/// POST `/flush_cache` to each worker concurrently (bounded by
/// [`MAX_CONCURRENT_FLUSH`]) and partition the outcomes into
/// (successful URLs, failed workers). A non-2xx status or a transport
/// error both count as failures.
async fn fan_out_flush(
    workers: &[Arc<Worker>],
    client: &Client,
    timeout: Duration,
) -> (Vec<String>, Vec<FailedWorker>) {
    // Snapshot the URLs into owned Strings up front so the per-worker stream
    // does not borrow the `workers` slice across the await points.
    let urls: Vec<String> = workers.iter().map(|w| w.url.clone()).collect();

    let outcomes = stream::iter(urls)
        .map(|url| {
            let client = client.clone();
            async move {
                let flush_url = format!("{}/flush_cache", url.trim_end_matches('/'));
                let result = client.post(&flush_url).timeout(timeout).send().await;
                (url, result)
            }
        })
        .buffer_unordered(MAX_CONCURRENT_FLUSH)
        .collect::<Vec<_>>()
        .await;

    let mut successful = Vec::new();
    let mut failed = Vec::new();
    for (url, result) in outcomes {
        match result {
            Ok(resp) if resp.status().is_success() => successful.push(url),
            Ok(resp) => failed.push(FailedWorker {
                worker: url,
                error: format!("HTTP {}", resp.status()),
            }),
            // Render the full source chain (`{:#}`), not just reqwest's outer
            // message, so a connect-refused / DNS / TLS / timeout cause is
            // visible in the per-worker error rather than collapsed away.
            Err(e) => failed.push(FailedWorker {
                worker: url,
                error: format!("{:#}", anyhow::Error::new(e)),
            }),
        }
    }
    (successful, failed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::server::app_context::AppContext;
    use axum::body::Body;
    use axum::http::Request;
    use axum::routing::post;
    use axum::Router;
    use http_body_util::BodyExt;
    use serde_json::Value;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tower::ServiceExt;

    /// Spawn a fake worker that answers `POST /flush_cache` with `status`.
    /// Returns its base URL and a shutdown handle (drop or send to stop).
    async fn spawn_fake_flush_worker(status: StatusCode) -> (String, oneshot::Sender<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = Router::new().route("/flush_cache", post(move || async move { status }));
        let (tx, rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = rx.await;
                })
                .await;
        });
        (format!("http://127.0.0.1:{port}"), tx)
    }

    /// Reserve a port then drop the listener so a connect attempt fails fast
    /// with ConnectionRefused (no waiting on the connect timeout).
    fn unused_port() -> u16 {
        use std::net::TcpListener;
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    }

    fn ctx_with_workers(urls: &[&str]) -> Arc<AppContext> {
        let ctx = AppContext::stub();
        for (i, url) in urls.iter().enumerate() {
            ctx.registry
                .add(WorkerSpec {
                    id: WorkerId(format!("w-{i}")),
                    url: (*url).to_string(),
                    mode: WorkerMode::Plain,
                    model_ids: vec![ModelId("stub-model".into())],
                    bootstrap_port: None,
                    min_priority: None,
                })
                .expect("worker accepted");
        }
        Arc::new(ctx)
    }

    async fn post_flush(ctx: Arc<AppContext>) -> (StatusCode, Value) {
        let app = crate::server::app::build_router(ctx);
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/flush_cache")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        (status, body)
    }

    #[tokio::test]
    async fn all_workers_succeed_returns_200() {
        let (u1, _s1) = spawn_fake_flush_worker(StatusCode::OK).await;
        let (u2, _s2) = spawn_fake_flush_worker(StatusCode::OK).await;
        let (status, body) = post_flush(ctx_with_workers(&[&u1, &u2])).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["total_workers"], 2);
        assert_eq!(body["successful"].as_array().unwrap().len(), 2);
        assert!(body["failed"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn partial_failure_returns_502_with_breakdown() {
        let (ok_url, _s1) = spawn_fake_flush_worker(StatusCode::OK).await;
        let (err_url, _s2) = spawn_fake_flush_worker(StatusCode::INTERNAL_SERVER_ERROR).await;
        let (status, body) = post_flush(ctx_with_workers(&[&ok_url, &err_url])).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(body["total_workers"], 2);
        assert_eq!(
            body["successful"].as_array().unwrap(),
            &vec![Value::String(ok_url.clone())]
        );
        let failed = body["failed"].as_array().unwrap();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0]["worker"], err_url);
        assert!(failed[0]["error"].as_str().unwrap().contains("500"));
    }

    #[tokio::test]
    async fn empty_registry_returns_200_with_zero_workers() {
        let (status, body) = post_flush(Arc::new(AppContext::stub())).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["total_workers"], 0);
        assert!(body["successful"].as_array().unwrap().is_empty());
        assert!(body["failed"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn unreachable_worker_is_reported_failed() {
        let url = format!("http://127.0.0.1:{}", unused_port());
        let (status, body) = post_flush(ctx_with_workers(&[&url])).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        let failed = body["failed"].as_array().unwrap();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0]["worker"], url);
    }

    /// A non-5xx, non-2xx status (e.g. 404) is still a failure and still
    /// drives the top-level 502, with the status echoed in the error.
    #[tokio::test]
    async fn non_5xx_error_status_is_reported_failed() {
        let (ok_url, _s1) = spawn_fake_flush_worker(StatusCode::OK).await;
        let (nf_url, _s2) = spawn_fake_flush_worker(StatusCode::NOT_FOUND).await;
        let (status, body) = post_flush(ctx_with_workers(&[&ok_url, &nf_url])).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        let failed = body["failed"].as_array().unwrap();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0]["worker"], nf_url);
        assert!(failed[0]["error"].as_str().unwrap().contains("404"));
    }

    /// A worker URL with a trailing slash must still resolve to
    /// `<url>/flush_cache` (not `<url>//flush_cache`). Guards the
    /// `trim_end_matches('/')` in `fan_out_flush` against a regression that
    /// would 404 every slash-suffixed worker.
    #[tokio::test]
    async fn worker_url_with_trailing_slash_is_flushed() {
        let (base, _s) = spawn_fake_flush_worker(StatusCode::OK).await;
        let url = format!("{base}/");
        let (status, body) = post_flush(ctx_with_workers(&[&url])).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["successful"].as_array().unwrap(),
            &vec![Value::String(url.clone())]
        );
        assert!(body["failed"].as_array().unwrap().is_empty());
    }

    /// The fan-out targets the whole fleet, not one model's pool: prefill
    /// and decode workers (which also hold KV cache) must both be flushed.
    /// Asserted through the handler — `registry::all()` returning mixed modes
    /// is necessary but not sufficient if a mode filter ever slips into the
    /// handler path.
    #[tokio::test]
    async fn flushes_prefill_and_decode_workers() {
        let (p_url, _s1) = spawn_fake_flush_worker(StatusCode::OK).await;
        let (d_url, _s2) = spawn_fake_flush_worker(StatusCode::OK).await;
        let ctx = AppContext::stub();
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId("p".into()),
                url: p_url.clone(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId("stub-model".into())],
                bootstrap_port: Some(8998),
                min_priority: None,
            })
            .expect("prefill accepted");
        ctx.registry
            .add(WorkerSpec {
                id: WorkerId("d".into()),
                url: d_url.clone(),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId("stub-model".into())],
                bootstrap_port: None,
                min_priority: None,
            })
            .expect("decode accepted");

        let (status, body) = post_flush(Arc::new(ctx)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["total_workers"], 2);
        let mut succeeded: Vec<&str> = body["successful"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        succeeded.sort_unstable();
        let mut expected = [p_url.as_str(), d_url.as_str()];
        expected.sort_unstable();
        assert_eq!(succeeded, expected);
    }
}
