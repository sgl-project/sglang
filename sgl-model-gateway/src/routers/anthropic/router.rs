//! Standalone proxy functions for the Anthropic API.
//!
//! Both `POST /v1/messages` and `POST /v1/messages/count_tokens` share the
//! same core logic:
//!   1. Extract `model` from the request for worker selection.
//!   2. Forward the raw request body unchanged to the backend.
//!   3. Return the backend response unchanged.
//!
//! No protocol conversion is performed — SGLang natively supports the
//! Anthropic API (PR #18630).

use std::{sync::Arc, time::Instant};

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::{
    app_context::AppContext,
    core::{is_retryable_status, RetryExecutor, RuntimeType, Worker},
    observability::metrics::{bool_to_static_str, metrics_labels, Metrics},
    routers::header_utils::{apply_provider_headers, extract_auth_header},
};

use super::protocol::{AnthropicCountTokensRequest, AnthropicMessagesRequest};

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Proxy `POST /v1/messages` to the backend.
pub async fn proxy(
    ctx: &AppContext,
    headers: &HeaderMap,
    routing: &AnthropicMessagesRequest,
    body: Bytes,
) -> Response {
    proxy_raw(
        ctx,
        headers,
        &routing.model,
        "/v1/messages",
        routing.stream,
        metrics_labels::ENDPOINT_MESSAGES,
        body,
    )
    .await
}

/// Proxy `POST /v1/messages/count_tokens` to the backend.
pub async fn proxy_count_tokens(
    ctx: &AppContext,
    headers: &HeaderMap,
    routing: &AnthropicCountTokensRequest,
    body: Bytes,
) -> Response {
    proxy_raw(
        ctx,
        headers,
        &routing.model,
        "/v1/messages/count_tokens",
        false, // count_tokens never streams
        metrics_labels::ENDPOINT_COUNT_TOKENS,
        body,
    )
    .await
}

// ---------------------------------------------------------------------------
// Shared implementation
// ---------------------------------------------------------------------------

/// Core proxy: select a worker, forward `body` to `{worker_url}{path}`,
/// return the response unchanged (streaming or non-streaming).
async fn proxy_raw(
    ctx: &AppContext,
    headers: &HeaderMap,
    model: &str,
    path: &str,
    streaming: bool,
    metrics_endpoint: &'static str,
    body: Bytes,
) -> Response {
    let start = Instant::now();

    Metrics::record_router_request(
        metrics_labels::ROUTER_ANTHROPIC,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model,
        metrics_endpoint,
        bool_to_static_str(streaming),
    );

    let auth_header = extract_auth_header(Some(headers), &None);

    let worker = match select_worker(ctx, model, auth_header.as_ref()).await {
        Ok(w) => w,
        Err(resp) => {
            Metrics::record_router_error(
                metrics_labels::ROUTER_ANTHROPIC,
                metrics_labels::BACKEND_EXTERNAL,
                metrics_labels::CONNECTION_HTTP,
                model,
                metrics_endpoint,
                metrics_labels::ERROR_NO_WORKERS,
            );
            return resp;
        }
    };

    let url = format!("{}{}", worker.url(), path);
    let client = ctx.client.clone();
    let body = Arc::new(body);
    let worker_api_key = Arc::new(worker.api_key().clone());
    let auth_header_arc = Arc::new(auth_header);
    let headers_arc = Arc::new(Some(headers.clone()));
    let retry_config = ctx.router_config.effective_retry_config();

    let response = RetryExecutor::execute_response_with_retry(
        &retry_config,
        |_attempt| {
            let client = client.clone();
            let url = url.clone();
            let body = Arc::clone(&body);
            let worker = Arc::clone(&worker);
            let worker_api_key = Arc::clone(&worker_api_key);
            let auth_header = Arc::clone(&auth_header_arc);
            let headers = Arc::clone(&headers_arc);

            async move {
                let mut req = client
                    .post(&url)
                    .header(CONTENT_TYPE, "application/json")
                    .body((*body).clone());

                let auth = extract_auth_header((*headers).as_ref(), &worker_api_key);
                req = apply_provider_headers(req, &url, auth.as_ref());

                if streaming {
                    req = req.header("Accept", "text/event-stream");
                }

                let resp = match req.send().await {
                    Ok(r) => r,
                    Err(e) => {
                        worker.circuit_breaker().record_failure();
                        return (
                            StatusCode::SERVICE_UNAVAILABLE,
                            format!("Failed to contact upstream: {}", e),
                        )
                            .into_response();
                    }
                };

                let status = StatusCode::from_u16(resp.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                if !status.is_success() {
                    worker.circuit_breaker().record_failure();
                }

                if !streaming {
                    let content_type = resp.headers().get(CONTENT_TYPE).cloned();
                    match resp.bytes().await {
                        Ok(bytes) => {
                            if status.is_success() {
                                worker.circuit_breaker().record_success();
                            }
                            let mut response = Response::new(Body::from(bytes));
                            *response.status_mut() = status;
                            if let Some(ct) = content_type {
                                response.headers_mut().insert(CONTENT_TYPE, ct);
                            }
                            response
                        }
                        Err(e) => {
                            worker.circuit_breaker().record_failure();
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                format!("Failed to read response: {}", e),
                            )
                                .into_response()
                        }
                    }
                } else {
                    if status.is_success() {
                        worker.circuit_breaker().record_success();
                    }
                    let stream = resp.bytes_stream();
                    let (tx, rx) = mpsc::unbounded_channel();
                    tokio::spawn(async move {
                        let mut s = stream;
                        while let Some(chunk) = s.next().await {
                            match chunk {
                                Ok(bytes) => {
                                    if tx.send(Ok(bytes)).is_err() {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    let _ = tx.send(Err(format!("Stream error: {}", e)));
                                    break;
                                }
                            }
                        }
                    });
                    let mut response =
                        Response::new(Body::from_stream(UnboundedReceiverStream::new(rx)));
                    *response.status_mut() = status;
                    response.headers_mut().insert(
                        CONTENT_TYPE,
                        HeaderValue::from_static("text/event-stream"),
                    );
                    response
                }
            }
        },
        |res, _attempt| is_retryable_status(res.status()),
        |_delay, _attempt| {
            Metrics::record_worker_retry(
                metrics_labels::BACKEND_EXTERNAL,
                metrics_endpoint,
            );
        },
        || {
            Metrics::record_worker_retries_exhausted(
                metrics_labels::BACKEND_EXTERNAL,
                metrics_endpoint,
            );
        },
    )
    .await;

    if response.status().is_success() {
        Metrics::record_router_duration(
            metrics_labels::ROUTER_ANTHROPIC,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model,
            metrics_endpoint,
            start.elapsed(),
        );
    } else {
        Metrics::record_router_error(
            metrics_labels::ROUTER_ANTHROPIC,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model,
            metrics_endpoint,
            metrics_labels::ERROR_BACKEND,
        );
    }

    response
}

// ---------------------------------------------------------------------------
// Worker selection helpers
// ---------------------------------------------------------------------------

/// Select the least-loaded healthy external worker for `model`.
/// Refreshes model lists once on miss before giving up.
async fn select_worker(
    ctx: &AppContext,
    model: &str,
    auth_header: Option<&axum::http::HeaderValue>,
) -> Result<Arc<dyn Worker>, Response> {
    if let Some(worker) = find_best_worker(&ctx.worker_registry, model) {
        return Ok(worker);
    }

    tracing::debug!(
        "No worker found for model '{}', refreshing external worker models",
        model
    );
    refresh_external_models(ctx, auth_header).await;

    find_best_worker(&ctx.worker_registry, model).ok_or_else(|| {
        let any_supports = ctx
            .worker_registry
            .get_workers_filtered(None, None, None, Some(RuntimeType::External), true)
            .into_iter()
            .any(|w| w.supports_model(model));

        if any_supports {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("All workers for model '{}' are temporarily unavailable", model),
            )
                .into_response()
        } else if ctx
            .worker_registry
            .get_workers_filtered(None, None, None, Some(RuntimeType::External), true)
            .is_empty()
        {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "type": "error",
                    "error": {
                        "type": "service_unavailable",
                        "message": "No backend workers available"
                    }
                })),
            )
                .into_response()
        } else {
            (
                StatusCode::NOT_FOUND,
                Json(json!({
                    "type": "error",
                    "error": {
                        "type": "not_found_error",
                        "message": format!("No worker available for model '{}'", model)
                    }
                })),
            )
                .into_response()
        }
    })
}

fn find_best_worker(
    registry: &crate::core::WorkerRegistry,
    model: &str,
) -> Option<Arc<dyn Worker>> {
    registry
        .get_workers_filtered(None, None, None, Some(RuntimeType::External), true)
        .into_iter()
        .filter(|w| w.supports_model(model) && w.circuit_breaker().can_execute())
        .min_by_key(|w| w.load())
}

async fn refresh_external_models(ctx: &AppContext, auth_header: Option<&axum::http::HeaderValue>) {
    let workers =
        ctx.worker_registry
            .get_workers_filtered(None, None, None, Some(RuntimeType::External), true);

    if workers.is_empty() {
        return;
    }

    futures_util::future::join_all(
        workers
            .iter()
            .map(|w| refresh_worker_models(ctx, w, auth_header)),
    )
    .await;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::core::{
        circuit_breaker::CircuitBreakerConfig, BasicWorkerBuilder, ModelCard, RuntimeType,
        WorkerRegistry, WorkerType,
    };

    use super::find_best_worker;

    fn make_registry() -> Arc<WorkerRegistry> {
        Arc::new(WorkerRegistry::new())
    }

    /// Register an External worker that supports `models` into `registry`.
    fn add_external_worker(
        registry: &WorkerRegistry,
        url: &str,
        models: &[&str],
    ) -> Arc<dyn crate::core::Worker> {
        let cards: Vec<ModelCard> = models.iter().map(|m| ModelCard::new(*m)).collect();
        let w: Arc<dyn crate::core::Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .runtime_type(RuntimeType::External)
                .models(cards)
                .build(),
        );
        registry.register(Arc::clone(&w));
        w
    }

    // ── find_best_worker ──────────────────────────────────────────────────

    #[test]
    fn no_workers_returns_none() {
        let registry = make_registry();
        assert!(find_best_worker(&registry, "claude-3-5-sonnet-20241022").is_none());
    }

    #[test]
    fn worker_does_not_support_model_returns_none() {
        let registry = make_registry();
        add_external_worker(&registry, "http://w1:8080", &["gpt-4"]);

        assert!(find_best_worker(&registry, "claude-3-5-sonnet-20241022").is_none());
    }

    #[test]
    fn matching_worker_is_returned() {
        let registry = make_registry();
        add_external_worker(
            &registry,
            "http://w1:8080",
            &["claude-3-5-sonnet-20241022"],
        );

        let worker = find_best_worker(&registry, "claude-3-5-sonnet-20241022");
        assert!(worker.is_some());
        assert_eq!(worker.unwrap().url(), "http://w1:8080");
    }

    #[test]
    fn open_circuit_breaker_skips_worker() {
        let registry = make_registry();
        // failure_threshold = 1 so a single failure opens the circuit.
        let cb_config = CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        };
        let w: Arc<dyn crate::core::Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8080")
                .worker_type(WorkerType::Regular)
                .runtime_type(RuntimeType::External)
                .models(vec![ModelCard::new("claude-3-5-sonnet-20241022")])
                .circuit_breaker_config(cb_config)
                .build(),
        );
        registry.register(Arc::clone(&w));
        w.circuit_breaker().record_failure(); // opens the circuit

        assert!(find_best_worker(&registry, "claude-3-5-sonnet-20241022").is_none());
    }

    #[test]
    fn non_external_worker_is_ignored() {
        // A "Regular" (internal gRPC/HTTP) worker must never be selected for
        // Anthropic proxy routing, which is only for External workers.
        let registry = make_registry();
        let w: Arc<dyn crate::core::Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8080")
                .worker_type(WorkerType::Regular)
                .runtime_type(RuntimeType::Regular) // NOT External
                .models(vec![ModelCard::new("claude-3-5-sonnet-20241022")])
                .build(),
        );
        registry.register(w);

        assert!(find_best_worker(&registry, "claude-3-5-sonnet-20241022").is_none());
    }

    #[test]
    fn least_loaded_worker_is_preferred() {
        let registry = make_registry();
        let model = "claude-3-5-sonnet-20241022";

        let w_busy = add_external_worker(&registry, "http://w1:8080", &[model]);
        let w_idle = add_external_worker(&registry, "http://w2:8080", &[model]);

        // Artificially raise load on w_busy.
        w_busy.increment_load();
        w_busy.increment_load();

        let selected = find_best_worker(&registry, model).unwrap();
        assert_eq!(selected.url(), w_idle.url(), "should pick the idle worker");
    }

    #[test]
    fn multiple_models_per_worker_matched_correctly() {
        let registry = make_registry();
        add_external_worker(
            &registry,
            "http://w1:8080",
            &["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        );

        assert!(find_best_worker(&registry, "claude-3-5-sonnet-20241022").is_some());
        assert!(find_best_worker(&registry, "claude-3-haiku-20240307").is_some());
        assert!(find_best_worker(&registry, "gpt-4").is_none());
    }

    #[test]
    fn healthy_worker_chosen_over_circuit_broken_worker() {
        let registry = make_registry();
        let model = "claude-3-5-sonnet-20241022";

        let cb_open = CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        };
        // Worker that will have its circuit broken.
        let w_broken: Arc<dyn crate::core::Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8080")
                .worker_type(WorkerType::Regular)
                .runtime_type(RuntimeType::External)
                .models(vec![ModelCard::new(model)])
                .circuit_breaker_config(cb_open)
                .build(),
        );
        registry.register(Arc::clone(&w_broken));
        w_broken.circuit_breaker().record_failure();

        // Healthy worker.
        add_external_worker(&registry, "http://w2:8080", &[model]);

        let selected = find_best_worker(&registry, model).unwrap();
        assert_eq!(selected.url(), "http://w2:8080");
    }
}

async fn refresh_worker_models(
    ctx: &AppContext,
    worker: &Arc<dyn Worker>,
    auth_header: Option<&axum::http::HeaderValue>,
) {
    let url = format!("{}/v1/models", worker.url());
    let mut req = ctx.client.get(&url);
    if let Some(auth) = auth_header {
        req = apply_provider_headers(req, &url, Some(auth));
    }

    match req.send().await {
        Ok(response) if response.status().is_success() => {
            if let Ok(json) = response.json::<serde_json::Value>().await {
                if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
                    let cards: Vec<crate::core::ModelCard> = data
                        .iter()
                        .filter_map(|m| m.get("id").and_then(|id| id.as_str()))
                        .map(crate::core::ModelCard::new)
                        .collect();
                    if !cards.is_empty() {
                        tracing::info!(
                            "Model refresh: found {} models from {}",
                            cards.len(),
                            url
                        );
                        worker.set_models(cards);
                    }
                }
            }
        }
        Ok(r) => tracing::debug!("Model refresh returned {} from {}", r.status(), url),
        Err(e) => tracing::warn!("Failed to fetch models from {}: {}", url, e),
    }
}
