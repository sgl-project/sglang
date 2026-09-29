// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Plain and PD chat forwarding, including load tracking and streaming metrics.

use super::preparation::{generate_room_id, BootstrapFields, PreparedChatRequest};
use crate::discovery::WorkerMode;
use crate::proxy::sse::{self, StreamEnd, StreamEndReason};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::{
    classify_stream_end, outcome_from_status, MetricsRegistry, RequestLogContext, RequestOutcome,
    StaleRequestOutcome, WorkerModeLabel,
};
use crate::state::load_monitor::router_inflight_load::RouterInflightLoadGuard;
use crate::workers::{LoadGuard, Worker};
use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response};
use axum::response::IntoResponse;
use bytes::Bytes;
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

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
    // Attribute the outcome to the worker supplying the client-visible response.
    let metrics = DispatchMetrics::new(
        ctx,
        &request,
        decode.as_deref().unwrap_or(&prefill),
        request_started_at,
    );
    // Both PD workers receive the same bootstrap room to coordinate KV transfer.
    let pd = decode.map(|decode| {
        let bootstrap = BootstrapFields {
            host: prefill.bootstrap_host().to_string(),
            port: prefill.bootstrap_port(),
            room: generate_room_id(),
        };
        (decode, bootstrap)
    });
    let engine_rid = request.engine_rid();
    let body = request.into_outgoing_body(
        ctx,
        pd.as_ref().map(|(_, bootstrap)| bootstrap),
        engine_rid.as_deref(),
    )?;
    let prefill_load_guards = (worker_load_guard, active_request_guard);

    // In PD mode, prefill runs independently and decode supplies the client response.
    let stream_abort = CancellationToken::new();
    let (response_worker, response_load_guards, prefill_task) =
        if let Some((decode, bootstrap)) = pd {
            let task = spawn_prefill_request(
                ctx,
                &metrics,
                Arc::clone(&prefill),
                headers.clone(),
                body.clone(),
                prefill_load_guards,
                bootstrap.room,
                stream_abort.clone(),
            );
            let decode_load_guards = (
                decode.load_guard(),
                ctx.router_inflight_load
                    .register(decode.id.clone(), decode.url.clone(), 0, 1),
            );
            (decode, decode_load_guards, Some((task, prefill)))
        } else {
            (prefill, prefill_load_guards, None)
        };

    // In PD mode, prefill can finish before decode. Watch the registration
    // held by the response so expiration remains live for its full lifetime.
    let expiration_token = response_load_guards.1.cancel_token().clone();
    let response = forward_to_response_worker(
        ctx,
        &response_worker,
        &headers,
        body,
        engine_rid.as_deref(),
        response_load_guards,
        &metrics,
        expiration_token.clone(),
        stream_abort,
    );
    let dispatch = async {
        match prefill_task {
            Some((task, prefill)) => forward_pd(task, prefill, response).await,
            None => (response.await, None),
        }
    };
    // A ready response wins if request expiration fires in the same poll.
    let (result, blamed_prefill) = tokio::select! {
        biased;
        dispatch = dispatch => dispatch,
        _ = expiration_token.cancelled() => {
            let model = metrics.model.clone();
            (Err(ApiError::StaleRequestExpired { model }), None)
        }
    };
    let log_context =
        metrics.record_dispatch_result(&result, engine_rid, blamed_prefill.as_deref());
    // Materialize dispatch errors here so the access log retains the selected worker.
    let mut response = match result {
        Ok(mut response) => {
            if let Some(hint) = decode_url_header {
                response.headers_mut().insert(X_SGL_DECODE_URL, hint);
            }
            response
        }
        Err(error) => error.into_response(),
    };
    response.extensions_mut().insert(log_context);
    Ok(response)
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

/// A prefill's client-visible failure; `None` means it succeeded.
type PrefillFailure = Option<Result<Response<Body>, ApiError>>;

/// Runs prefill to completion even if the client disconnects. A failure also
/// aborts decode's stream until its first token, which proves KV transfer completed.
#[allow(clippy::too_many_arguments)]
fn spawn_prefill_request(
    ctx: &AppContext,
    metrics: &DispatchMetrics,
    prefill_worker: Arc<Worker>,
    headers: HeaderMap,
    body: Bytes,
    load_guards: LoadGuards,
    bootstrap_room: u64,
    stream_abort: CancellationToken,
) -> tokio::task::JoinHandle<PrefillFailure> {
    let proxy = Arc::clone(&ctx.proxy);
    let (registry, model) = (Arc::clone(&metrics.registry), metrics.model.clone());
    tokio::spawn(async move {
        let _load_guards = load_guards;
        let result = proxy
            .forward_json_to(
                &prefill_worker.url,
                prefill_worker.protocol(),
                &prefill_worker.breaker,
                CHAT_PATH,
                &headers,
                body,
                None,
            )
            .await;
        let failure = prefill_failure(result).await;
        let prefill_url = &prefill_worker.url;
        let outcome = failure
            .as_ref()
            .map_or(RequestOutcome::Success, dispatch_outcome);
        registry.record_worker_request(prefill_url, &model, WorkerModeLabel::Prefill, outcome);
        match &failure {
            None => tracing::debug!(%prefill_url, bootstrap_room, "prefill side completed"),
            Some(Ok(response)) => tracing::debug!(
                %prefill_url, bootstrap_room, status = %response.status(),
                "prefill rejected the request",
            ),
            Some(Err(error)) => {
                tracing::warn!(%prefill_url, bootstrap_room, %error, "prefill failed")
            }
        }
        if failure.is_some() {
            stream_abort.cancel();
        }
        failure
    })
}

/// Client errors and backpressure pass through; only a prefill fault becomes
/// `prefill_failed`.
async fn prefill_failure(result: Result<Response<Body>, ApiError>) -> PrefillFailure {
    let status = match result {
        // A streaming prefill reports a late failure as a 200 carrying an SSE error event.
        Ok(response) if response.status().is_success() => {
            let status = response.status();
            match axum::body::to_bytes(response.into_body(), usize::MAX).await {
                Ok(body) if !sse::has_error_event(&body) => return None,
                _ => status,
            }
        }
        Ok(response)
            if matches!(
                outcome_from_status(response.status().as_u16()),
                RequestOutcome::Error
            ) =>
        {
            response.status()
        }
        result => return Some(result),
    };
    Some(Err(ApiError::PrefillFailed {
        status: Some(status),
    }))
}

/// Returns decode's response as soon as decode answers, unless prefill fails
/// first; then prefill's failure is returned along with the blamed prefill.
async fn forward_pd(
    task: tokio::task::JoinHandle<PrefillFailure>,
    prefill: Arc<Worker>,
    decode: impl std::future::Future<Output = Result<Response<Body>, ApiError>>,
) -> (Result<Response<Body>, ApiError>, Option<Arc<Worker>>) {
    let mut decode = std::pin::pin!(decode);
    tokio::select! {
        biased;
        failure = task => match failure {
            Ok(None) => (decode.await, None),
            Ok(Some(failure)) => (failure, Some(prefill)),
            Err(_) => (Err(ApiError::PrefillFailed { status: None }), Some(prefill)),
        },
        // Dropping the handle leaves prefill running.
        response = &mut decode => (response, None),
    }
}

#[allow(clippy::too_many_arguments)]
async fn forward_to_response_worker(
    ctx: &AppContext,
    worker: &Worker,
    headers: &HeaderMap,
    body: Bytes,
    engine_rid: Option<&str>,
    load_guards: LoadGuards,
    metrics: &DispatchMetrics,
    expiration: CancellationToken,
    stream_abort: CancellationToken,
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
                engine_rid,
                Some(stream_guards),
                Some(metrics.first_byte_callback()),
                Some(metrics.stream_end_callback(worker.url.clone())),
                Some(expiration),
                Some(stream_abort),
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
                engine_rid,
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
        response_worker: &Worker,
        request_started_at: Instant,
    ) -> Self {
        Self {
            registry: Arc::clone(&ctx.metrics),
            model: request.model.0.clone(),
            worker_url: response_worker.url.clone(),
            mode: match response_worker.mode() {
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
            if end.reason == StreamEndReason::Expired {
                metrics.record_stale_request(StaleRequestOutcome::Expired);
            }
            metrics.record_stream_outcome(&response_worker_url, &model, classify_stream_end(end));
        })
    }

    /// Logs a blamed prefill's failure against it; its task already recorded the outcome.
    fn record_dispatch_result(
        &self,
        result: &Result<Response<Body>, ApiError>,
        engine_rid: Option<String>,
        blamed_prefill: Option<&Worker>,
    ) -> RequestLogContext {
        if let Err(ApiError::StaleRequestExpired { .. }) = result {
            self.registry
                .record_stale_request(StaleRequestOutcome::Expired);
        }
        let outcome = dispatch_outcome(result);
        let worker_url = match blamed_prefill {
            Some(prefill) => &prefill.url,
            None => {
                self.registry.record_worker_request(
                    &self.worker_url,
                    &self.model,
                    self.mode,
                    outcome,
                );
                &self.worker_url
            }
        };
        if !self.streaming {
            self.registry.observe_request_duration(
                &self.model,
                self.request_started_at.elapsed().as_secs_f64(),
            );
        }
        // The app middleware emits the access log and edge counters exactly once.
        RequestLogContext {
            worker_url: worker_url.clone(),
            model_id: self.model.clone(),
            streaming: self.streaming,
            outcome,
            engine_rid,
        }
    }
}

// HTTP status determines the outcome; router cancellations and dispatch failures stay distinct.
fn dispatch_outcome(result: &Result<Response<Body>, ApiError>) -> RequestOutcome {
    match result {
        Ok(response) => outcome_from_status(response.status().as_u16()),
        Err(ApiError::StaleRequestExpired { .. }) => RequestOutcome::Cancelled,
        // These 503s come from the router, not worker backpressure.
        Err(ApiError::BreakerOpen { .. } | ApiError::WorkerMisconfigured { .. }) => {
            RequestOutcome::Error
        }
        Err(error) => outcome_from_status(error.status_code().as_u16()),
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
