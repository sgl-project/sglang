// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::request::{generate_room_id, BootstrapFields, ChatRequest};
use super::routing::SelectedWorkers;
use crate::discovery::WorkerMode;
use crate::policies::active_load::ActiveLoadGuard;
use crate::proxy::sse::StreamEnd;
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::{
    classify_stream_end, MetricsRegistry, RequestOutcome, StaleRequestOutcome, WorkerModeLabel,
};
use crate::workers::{LoadGuard, Worker};
use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response};
use bytes::Bytes;
use std::sync::Arc;
use std::time::Instant;

const CHAT_PATH: &str = "/v1/chat/completions";
const X_SGL_DECODE_URL: HeaderName = HeaderName::from_static("x-sgl-decode-url");
type LoadGuards = (LoadGuard, ActiveLoadGuard);

pub(super) async fn forward(
    ctx: &AppContext,
    request: ChatRequest,
    workers: SelectedWorkers,
    mut headers: HeaderMap,
    start: Instant,
) -> Result<Response<Body>, ApiError> {
    let SelectedWorkers {
        prefill,
        decode,
        timestamped,
    } = workers;
    let decode_hint = decode.as_ref().and_then(|worker| decode_hint(&worker.url));
    if let Some(hint) = &decode_hint {
        headers.insert(X_SGL_DECODE_URL, hint.clone());
    }

    let guard = if timestamped {
        prefill.timestamped_load_guard()
    } else {
        prefill.load_guard()
    };
    let active_guard = ctx.active_load.register(
        prefill.id.clone(),
        prefill.url.clone(),
        request.prefill_load,
        0,
    );
    // PD requests keep using the prefill expiration token after dispatching decode.
    let stale_token = active_guard.cancel_token().clone();
    let metrics = DispatchMetrics::new(ctx, &request, &prefill, start);
    let bootstrap = decode.as_ref().map(|_| BootstrapFields {
        host: prefill.bootstrap_host().to_string(),
        port: prefill.bootstrap_port(),
        room: generate_room_id(),
    });
    let body = request.into_outgoing_body(ctx, bootstrap.as_ref())?;
    let guards = (guard, active_guard);

    let (response_worker, response_guards) = if let Some(decode) = decode {
        let room = bootstrap
            .expect("PD dispatch requires bootstrap fields")
            .room;
        spawn_prefill(ctx, prefill, headers.clone(), body.clone(), guards, room);
        let guards = (
            decode.load_guard(),
            ctx.active_load
                .register(decode.id.clone(), decode.url.clone(), 0, 1),
        );
        (decode, guards)
    } else {
        (prefill, guards)
    };

    let fetch = forward_response(
        ctx,
        &response_worker,
        &headers,
        body,
        response_guards,
        &metrics,
    );
    // A completed fetch wins if expiration fires in the same poll.
    let result = tokio::select! {
        biased;
        result = fetch => result,
        _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired {
            model: metrics.model.clone(),
        }),
    };
    metrics.record_result(&headers, &result);
    result.map(|mut response| {
        if let Some(hint) = decode_hint {
            response.headers_mut().insert(X_SGL_DECODE_URL, hint);
        }
        response
    })
}

fn decode_hint(url: &str) -> Option<HeaderValue> {
    HeaderValue::from_str(url)
        .map_err(|error| {
            tracing::warn!(
                decode_url = %url,
                %error,
                "decode worker URL rejected by header parser; sending request without decode hint",
            );
        })
        .ok()
}

fn spawn_prefill(
    ctx: &AppContext,
    worker: Arc<Worker>,
    headers: HeaderMap,
    body: Bytes,
    guards: LoadGuards,
    bootstrap_room: u64,
) {
    let proxy = Arc::clone(&ctx.proxy);
    // Detach prefill so client cancellation cannot interrupt the KV transfer.
    tokio::spawn(async move {
        let _holds = guards;
        match proxy
            .forward_json_to(
                &worker.url,
                worker.protocol(),
                &worker.breaker,
                CHAT_PATH,
                &headers,
                body,
            )
            .await
        {
            Ok(_) => tracing::debug!(
                prefill_url = %worker.url, bootstrap_room, "prefill side completed",
            ),
            Err(error) => tracing::warn!(
                prefill_url = %worker.url,
                bootstrap_room,
                %error,
                "prefill request failed; decode will time out on bootstrap_room",
            ),
        }
    });
}

async fn forward_response(
    ctx: &AppContext,
    worker: &Worker,
    headers: &HeaderMap,
    body: Bytes,
    guards: LoadGuards,
    metrics: &DispatchMetrics,
) -> Result<Response<Body>, ApiError> {
    if metrics.streaming {
        // Load and duration guards live until the SSE pump ends, not just until headers arrive.
        let stream_guards: Box<dyn Send + 'static> = Box::new((guards, metrics.duration_guard()));
        ctx.proxy
            .forward_streaming_to(
                &worker.url,
                worker.protocol(),
                &worker.breaker,
                CHAT_PATH,
                headers,
                body,
                Some(stream_guards),
                Some(metrics.ttft_hook()),
                Some(metrics.stream_end_hook(worker.url.clone())),
            )
            .await
    } else {
        let _holds = guards;
        ctx.proxy
            .forward_json_to(
                &worker.url,
                worker.protocol(),
                &worker.breaker,
                CHAT_PATH,
                headers,
                body,
            )
            .await
    }
}

struct DispatchMetrics {
    registry: Arc<MetricsRegistry>,
    model: String,
    worker_url: String,
    mode: WorkerModeLabel,
    streaming: bool,
    start: Instant,
}

impl DispatchMetrics {
    fn new(ctx: &AppContext, request: &ChatRequest, worker: &Worker, start: Instant) -> Self {
        Self {
            registry: Arc::clone(&ctx.metrics),
            model: request.model.0.clone(),
            worker_url: worker.url.clone(),
            mode: match worker.mode() {
                WorkerMode::Prefill => WorkerModeLabel::Prefill,
                WorkerMode::Decode => WorkerModeLabel::Decode,
                WorkerMode::Plain => WorkerModeLabel::Plain,
            },
            streaming: request.streaming,
            start,
        }
    }

    fn ttft_hook(&self) -> Box<dyn FnOnce() + Send + 'static> {
        let metrics = Arc::clone(&self.registry);
        let model = self.model.clone();
        let start = self.start;
        Box::new(move || metrics.observe_ttft(&model, start.elapsed().as_secs_f64()))
    }

    fn duration_guard(&self) -> RecordDurationOnDrop {
        RecordDurationOnDrop {
            metrics: Arc::clone(&self.registry),
            model: self.model.clone(),
            start: self.start,
        }
    }

    fn stream_end_hook(&self, worker_url: String) -> Box<dyn FnOnce(StreamEnd) + Send + 'static> {
        let metrics = Arc::clone(&self.registry);
        let model = self.model.clone();
        Box::new(move |end| {
            metrics.record_stream_outcome(&worker_url, &model, classify_stream_end(end));
        })
    }

    fn record_result(&self, headers: &HeaderMap, result: &Result<Response<Body>, ApiError>) {
        let outcome = match result {
            Ok(_) => RequestOutcome::Success,
            Err(ApiError::StaleRequestExpired { .. }) => {
                self.registry
                    .record_stale_request(StaleRequestOutcome::Expired);
                RequestOutcome::Cancelled
            }
            Err(_) => RequestOutcome::Error,
        };
        self.registry
            .record_worker_request(&self.worker_url, &self.model, self.mode, outcome);
        let elapsed = self.start.elapsed();
        if !self.streaming {
            self.registry
                .observe_request_duration(&self.model, elapsed.as_secs_f64());
        }
        let request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("-");
        let http_status = match result {
            Ok(response) => response.status().as_u16(),
            Err(error) => error.status_code().as_u16(),
        };
        let outcome = match outcome {
            RequestOutcome::Success => "success",
            RequestOutcome::Error => "error",
            RequestOutcome::Cancelled => "cancelled",
        };
        tracing::info!(
            request_id,
            method = "POST",
            path = CHAT_PATH,
            model = %self.model,
            worker = %self.worker_url,
            outcome,
            http_status,
            stream = self.streaming,
            latency_ms = elapsed.as_millis() as u64,
            "chat_completions",
        );
    }
}

struct RecordDurationOnDrop {
    metrics: Arc<MetricsRegistry>,
    model: String,
    start: Instant,
}

impl Drop for RecordDurationOnDrop {
    fn drop(&mut self) {
        self.metrics
            .observe_request_duration(&self.model, self.start.elapsed().as_secs_f64());
    }
}
