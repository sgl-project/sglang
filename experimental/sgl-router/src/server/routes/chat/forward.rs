// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Plain and PD chat forwarding, including load tracking and streaming metrics.

use super::preparation::{generate_room_id, BootstrapFields, PreparedChatRequest};
use crate::discovery::WorkerMode;
use crate::proxy::sse::StreamEnd;
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::{
    classify_stream_end, MetricsRegistry, RequestOutcome, StaleRequestOutcome, WorkerModeLabel,
};
use crate::state::load_monitor::router_inflight_load::RouterInflightLoadGuard;
use crate::workers::{LoadGuard, Worker};
use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response};
use bytes::Bytes;
use std::sync::Arc;
use std::time::Instant;

const CHAT_PATH: &str = "/v1/chat/completions";
// Expose the selected decode worker to both PD workers and the client.
const X_SGL_DECODE_URL: HeaderName = HeaderName::from_static("x-sgl-decode-url");
type LoadGuards = (LoadGuard, RouterInflightLoadGuard);

/// A plain worker, or a prefill worker paired with a decode worker for PD.
pub(super) struct SelectedWorkers {
    pub(super) prefill: Arc<Worker>,
    pub(super) decode: Option<Arc<Worker>>,
    pub(super) track_dispatch_timestamps: bool,
}

pub(super) async fn forward_chat_request(
    ctx: &AppContext,
    request: PreparedChatRequest,
    workers: SelectedWorkers,
    mut headers: HeaderMap,
    request_started_at: Instant,
) -> Result<Response<Body>, ApiError> {
    let SelectedWorkers {
        prefill,
        decode,
        track_dispatch_timestamps,
    } = workers;
    let decode_url_header = decode
        .as_ref()
        .and_then(|worker| parse_decode_url_header(&worker.url));
    if let Some(hint) = &decode_url_header {
        headers.insert(X_SGL_DECODE_URL, hint.clone());
    }

    // Track worker occupancy and the prompt's contribution to active load.
    let worker_load_guard = if track_dispatch_timestamps {
        prefill.timestamped_load_guard()
    } else {
        prefill.load_guard()
    };
    let active_request_guard = ctx.router_inflight_load.register(
        prefill.id.clone(),
        prefill.url.clone(),
        request.input_token_count,
        0,
    );
    // PD requests keep using the prefill expiration token after dispatching decode.
    let expiration_token = active_request_guard.cancel_token().clone();
    let metrics = DispatchMetrics::new(ctx, &request, &prefill, request_started_at);
    // Both PD workers receive the same bootstrap room to coordinate KV transfer.
    let pd = decode.map(|decode| {
        let bootstrap = BootstrapFields {
            host: prefill.bootstrap_host().to_string(),
            port: prefill.bootstrap_port(),
            room: generate_room_id(),
        };
        (decode, bootstrap)
    });
    let body = request.into_outgoing_body(ctx, pd.as_ref().map(|(_, bootstrap)| bootstrap))?;
    let prefill_load_guards = (worker_load_guard, active_request_guard);

    // In PD mode, prefill runs independently and decode supplies the client response.
    let (response_worker, response_load_guards) = if let Some((decode, bootstrap)) = pd {
        spawn_prefill_request(
            ctx,
            prefill,
            headers.clone(),
            body.clone(),
            prefill_load_guards,
            bootstrap.room,
        );
        let decode_load_guards = (
            decode.load_guard(),
            ctx.router_inflight_load
                .register(decode.id.clone(), decode.url.clone(), 0, 1),
        );
        (decode, decode_load_guards)
    } else {
        (prefill, prefill_load_guards)
    };

    let response_future = forward_to_response_worker(
        ctx,
        &response_worker,
        &headers,
        body,
        response_load_guards,
        &metrics,
    );
    // A ready response wins if request expiration fires in the same poll.
    let result = tokio::select! {
        biased;
        result = response_future => result,
        _ = expiration_token.cancelled() => Err(ApiError::StaleRequestExpired {
            model: metrics.model.clone(),
        }),
    };
    metrics.record_dispatch_result(&headers, &result);
    result.map(|mut response| {
        if let Some(hint) = decode_url_header {
            response.headers_mut().insert(X_SGL_DECODE_URL, hint);
        }
        response
    })
}

fn parse_decode_url_header(decode_url: &str) -> Option<HeaderValue> {
    HeaderValue::from_str(decode_url)
        .map_err(|error| {
            tracing::warn!(
                decode_url = %decode_url,
                %error,
                "decode worker URL rejected by header parser; sending request without decode hint",
            );
        })
        .ok()
}

fn spawn_prefill_request(
    ctx: &AppContext,
    prefill_worker: Arc<Worker>,
    headers: HeaderMap,
    body: Bytes,
    load_guards: LoadGuards,
    bootstrap_room: u64,
) {
    let proxy = Arc::clone(&ctx.proxy);
    // Let prefill finish KV transfer after client cancellation; router shutdown still cancels it.
    tokio::spawn(async move {
        let _load_guards = load_guards;
        match proxy
            .forward_json_to(
                &prefill_worker.url,
                prefill_worker.protocol(),
                &prefill_worker.breaker,
                CHAT_PATH,
                &headers,
                body,
            )
            .await
        {
            Ok(_) => tracing::debug!(
                prefill_url = %prefill_worker.url, bootstrap_room, "prefill side completed",
            ),
            // Prefill failures surface to the client through decode's bootstrap timeout.
            Err(error) => tracing::warn!(
                prefill_url = %prefill_worker.url,
                bootstrap_room,
                %error,
                "prefill request failed; decode will time out on bootstrap_room",
            ),
        }
    });
}

async fn forward_to_response_worker(
    ctx: &AppContext,
    worker: &Worker,
    headers: &HeaderMap,
    body: Bytes,
    load_guards: LoadGuards,
    metrics: &DispatchMetrics,
) -> Result<Response<Body>, ApiError> {
    if metrics.streaming {
        // Load and duration guards live until the SSE pump ends, not just until headers arrive.
        let stream_guards: Box<dyn Send + 'static> =
            Box::new((load_guards, metrics.stream_duration_guard()));
        ctx.proxy
            .forward_streaming_to(
                &worker.url,
                worker.protocol(),
                &worker.breaker,
                CHAT_PATH,
                headers,
                body,
                Some(stream_guards),
                Some(metrics.first_byte_callback()),
                Some(metrics.stream_end_callback(worker.url.clone())),
            )
            .await
    } else {
        // JSON forwarding reads the full response body before releasing load guards.
        let _load_guards = load_guards;
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
    request_started_at: Instant,
}

impl DispatchMetrics {
    fn new(
        ctx: &AppContext,
        request: &PreparedChatRequest,
        selected_worker: &Worker,
        request_started_at: Instant,
    ) -> Self {
        // Request outcomes use the plain/prefill worker; stream outcomes use the response worker.
        Self {
            registry: Arc::clone(&ctx.metrics),
            model: request.model.0.clone(),
            worker_url: selected_worker.url.clone(),
            mode: match selected_worker.mode() {
                WorkerMode::Prefill => WorkerModeLabel::Prefill,
                WorkerMode::Decode => WorkerModeLabel::Decode,
                WorkerMode::Plain => WorkerModeLabel::Plain,
            },
            streaming: request.streaming,
            request_started_at,
        }
    }

    // TTFT uses the first successful stream chunk, measured from request arrival.
    fn first_byte_callback(&self) -> Box<dyn FnOnce() + Send + 'static> {
        let metrics = Arc::clone(&self.registry);
        let model = self.model.clone();
        let request_started_at = self.request_started_at;
        Box::new(move || metrics.observe_ttft(&model, request_started_at.elapsed().as_secs_f64()))
    }

    fn stream_duration_guard(&self) -> StreamDurationGuard {
        StreamDurationGuard {
            metrics: Arc::clone(&self.registry),
            model: self.model.clone(),
            request_started_at: self.request_started_at,
        }
    }

    fn stream_end_callback(
        &self,
        response_worker_url: String,
    ) -> Box<dyn FnOnce(StreamEnd) + Send + 'static> {
        let metrics = Arc::clone(&self.registry);
        let model = self.model.clone();
        Box::new(move |end| {
            metrics.record_stream_outcome(&response_worker_url, &model, classify_stream_end(end));
        })
    }

    // Streaming success here means response setup succeeded; the callback records stream completion.
    fn record_dispatch_result(
        &self,
        headers: &HeaderMap,
        result: &Result<Response<Body>, ApiError>,
    ) {
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
        let elapsed = self.request_started_at.elapsed();
        if !self.streaming {
            self.registry
                .observe_request_duration(&self.model, elapsed.as_secs_f64());
        }
        let request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("-");
        // HTTP status is counted into responses_total by the app.rs middleware, not here.
        let http_status = match result {
            Ok(response) => response.status().as_u16(),
            Err(error) => error.status_code().as_u16(),
        };
        tracing::info!(
            request_id = %request_id,
            method = "POST",
            path = CHAT_PATH,
            model = %self.model,
            worker = %self.worker_url,
            outcome = outcome.as_str(),
            http_status,
            stream = self.streaming,
            latency_ms = elapsed.as_millis() as u64,
            "chat_completions",
        );
    }
}

/// Record total request duration when streaming ends or setup fails.
struct StreamDurationGuard {
    metrics: Arc<MetricsRegistry>,
    model: String,
    request_started_at: Instant,
}

impl Drop for StreamDurationGuard {
    fn drop(&mut self) {
        self.metrics
            .observe_request_duration(&self.model, self.request_started_at.elapsed().as_secs_f64());
    }
}
