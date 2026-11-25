// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Plain and PD forwarding, including load tracking and streaming metrics.

use super::preparation::PreparedRequest;
use crate::discovery::{ModelId, WorkerId, WorkerMode};
use crate::proxy::grpc::{Additions, GrpcResponse};
use crate::proxy::sse::{StreamEnd, StreamEndReason};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::{
    classify_stream_end, outcome_from_status, MediaCounts, MetricsRegistry, RequestLogContext,
    RequestOutcome, StaleRequestOutcome, WorkerModeLabel,
};
use crate::state::load_monitor::router_inflight_load::RouterInflightLoadGuard;
use crate::workers::{DecodeSlot, DpRankGuard, LoadGuard, Worker};
use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response};
use axum::response::IntoResponse;
use bytes::Bytes;
use sglang_api_types::api::v1::GenerateRequest;
use std::future::Future;
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::routing::dispatch::dispatch;
use crate::routing::dispatch::SelectedWorkers;

// Expose the selected decode worker to both PD workers and the client.
pub(crate) const X_SGL_DECODE_URL: HeaderName = HeaderName::from_static("x-sgl-decode-url");
// SGLang's DP controller dispatches to this rank; it outranks `routed_dp_rank` in the body.
pub(crate) const X_DATA_PARALLEL_RANK: HeaderName = HeaderName::from_static("x-data-parallel-rank");
pub(crate) type LoadGuards = (LoadGuard, RouterInflightLoadGuard, Option<DpRankGuard>);

/// One dispatch attempt's client-ready response.
pub(super) struct HttpDispatched {
    pub(super) response: Response<Body>,
    /// The workers to avoid on a retry, set only when nothing beyond a
    /// failure status reached the client, so another worker may serve it.
    pub(super) retry_excluding: Vec<WorkerId>,
}

pub(super) async fn forward_request(
    ctx: &AppContext,
    request: &mut PreparedRequest,
    workers: SelectedWorkers,
    headers: HeaderMap,
    request_started_at: Instant,
    duration: &Arc<RequestDurationGuard>,
) -> Result<HttpDispatched, ApiError> {
    let dispatched = dispatch(
        Http,
        ctx,
        request,
        workers,
        headers,
        request_started_at,
        duration,
    )
    .await?;
    let mut response = match dispatched.result {
        Ok(mut response) => {
            if let Some(hint) = dispatched.decode_url_header {
                response.headers_mut().insert(X_SGL_DECODE_URL, hint);
            }
            response
        }
        Err(error) => error.into_response(),
    };
    response.extensions_mut().insert(dispatched.log_context);
    Ok(HttpDispatched {
        response,
        retry_excluding: dispatched.retry_excluding,
    })
}

/// [`forward_request`] for a gRPC client: the engines are called with `rpc`.
/// The log context is `None` when the request failed before dispatch.
pub(super) async fn forward_request_grpc(
    rpc: GenerateRequest,
    ctx: &AppContext,
    request: &mut PreparedRequest,
    routing: &super::ModelRouting<'_>,
    headers: HeaderMap,
    request_started_at: Instant,
) -> (Result<GrpcResponse, ApiError>, Option<RequestLogContext>) {
    let transport = Grpc(Arc::new(rpc));
    let mut excluded = Vec::new();
    let mut failed = None;
    let mut duration = None;
    for attempt in 0..ctx.config.proxy.max_attempts.get() {
        if attempt > 0 {
            let deadline = ctx.router_inflight_load.stale_request_timeout();
            let Some(remaining) = deadline.checked_sub(request_started_at.elapsed()) else {
                break;
            };
            tokio::time::sleep(ctx.config.proxy.backoff(attempt).min(remaining)).await;
            if request_started_at.elapsed() >= deadline {
                break;
            }
        }
        let workers = match routing
            .select_workers(ctx, request, &headers, &excluded)
            .await
        {
            Ok(workers) => workers,
            Err(error) => return failed.unwrap_or((Err(error), None)),
        };
        if failed.is_some() {
            ctx.metrics.record_retry(&request.model.0);
        }
        let duration = duration.get_or_insert_with(|| {
            RequestDurationGuard::new(ctx, &request.model, request_started_at)
        });
        let dispatched = match dispatch(
            transport.clone(),
            ctx,
            request,
            workers,
            headers.clone(),
            request_started_at,
            duration,
        )
        .await
        {
            Ok(dispatched) => dispatched,
            Err(error) => return (Err(error), None),
        };
        let result = (dispatched.result, Some(dispatched.log_context));
        if dispatched.retry_excluding.is_empty() {
            return result;
        }
        excluded.extend(dispatched.retry_excluding);
        failed = Some(result);
    }
    failed.expect("at least one attempt")
}

pub(crate) struct Dispatched<R> {
    pub(crate) result: Result<R, ApiError>,
    pub(crate) log_context: RequestLogContext,
    pub(crate) decode_url_header: Option<HeaderValue>,
    pub(crate) retry_excluding: Vec<WorkerId>,
}

fn release_on_first_byte(
    slot: Option<DecodeSlot>,
    first_byte: Box<dyn FnOnce() + Send + 'static>,
) -> Box<dyn FnOnce() + Send + 'static> {
    match slot {
        Some(slot) => Box::new(move || {
            slot.release();
            first_byte();
        }),
        None => first_byte,
    }
}

/// How the selected workers are reached: HTTP clients over HTTP, gRPC clients over gRPC.
pub(crate) trait Transport: Clone + Send + Sync + 'static {
    type Response: Send + 'static;

    fn status(response: &Self::Response) -> u16;
    /// The client's response.
    fn respond(
        request: &mut PreparedRequest,
        response: Self::Response,
        streaming: bool,
    ) -> impl Future<Output = Self::Response> + Send;

    /// Send to the worker whose response the client receives.
    #[allow(clippy::too_many_arguments)]
    fn send(
        self,
        ctx: &AppContext,
        worker: &Worker,
        path: &'static str,
        leg: Leg,
        engine_rid: Option<&str>,
        load_guards: LoadGuards,
        metrics: &DispatchMetrics,
        expiration: CancellationToken,
        stream_abort: CancellationToken,
        decode_slot: Option<DecodeSlot>,
    ) -> impl Future<Output = Result<Self::Response, ApiError>> + Send;
}

/// One worker's request: the HTTP body, and the same additions as typed fields.
pub(crate) struct Leg {
    pub(crate) headers: HeaderMap,
    pub(crate) body: Bytes,
    pub(crate) additions: Additions,
}

#[derive(Clone, Copy)]
pub(crate) struct Http;

impl Transport for Http {
    type Response = Response<Body>;
    async fn respond(
        request: &mut PreparedRequest,
        response: Self::Response,
        streaming: bool,
    ) -> Self::Response {
        if request.responder.is_some() {
            super::openai::respond(&mut request.responder, response, streaming).await
        } else {
            response
        }
    }

    fn status(response: &Response<Body>) -> u16 {
        response.status().as_u16()
    }

    #[allow(clippy::too_many_arguments)]
    async fn send(
        self,
        ctx: &AppContext,
        worker: &Worker,
        path: &'static str,
        leg: Leg,
        engine_rid: Option<&str>,
        load_guards: LoadGuards,
        metrics: &DispatchMetrics,
        expiration: CancellationToken,
        stream_abort: CancellationToken,
        decode_slot: Option<DecodeSlot>,
    ) -> Result<Response<Body>, ApiError> {
        forward_to_response_worker(
            ctx,
            worker,
            path,
            &leg.headers,
            leg.body,
            engine_rid,
            load_guards,
            metrics,
            expiration,
            stream_abort,
            decode_slot,
        )
        .await
    }
}

/// Forwards the shared Generate request to the selected engines.
#[derive(Clone)]
pub(crate) struct Grpc(pub(crate) Arc<GenerateRequest>);

impl Transport for Grpc {
    type Response = GrpcResponse;
    async fn respond(
        _request: &mut PreparedRequest,
        response: Self::Response,
        _streaming: bool,
    ) -> Self::Response {
        response
    }

    fn status(response: &Self::Response) -> u16 {
        response.status.as_u16()
    }

    async fn send(
        self,
        ctx: &AppContext,
        worker: &Worker,
        _path: &'static str,
        leg: Leg,
        engine_rid: Option<&str>,
        load_guards: LoadGuards,
        metrics: &DispatchMetrics,
        expiration: CancellationToken,
        stream_abort: CancellationToken,
        decode_slot: Option<DecodeSlot>,
    ) -> Result<Self::Response, ApiError> {
        let (guards, on_first_byte, on_stream_end): (Box<dyn Send + 'static>, _, _) =
            if metrics.streaming {
                (
                    Box::new((load_guards, metrics.stream_duration_guard())),
                    Some(release_on_first_byte(
                        decode_slot,
                        metrics.first_byte_callback(),
                    )),
                    Some(metrics.stream_end_callback(worker.url.clone())),
                )
            } else {
                (Box::new(load_guards), None, None)
            };
        ctx.proxy
            .forward_grpc(
                worker,
                &self.0,
                &leg.headers,
                &leg.additions,
                engine_rid,
                metrics.streaming,
                Some(guards),
                on_first_byte,
                on_stream_end,
                Some(expiration),
                Some(stream_abort),
            )
            .await
    }
}

#[allow(clippy::too_many_arguments)]
async fn forward_to_response_worker(
    ctx: &AppContext,
    worker: &Worker,
    path: &str,
    headers: &HeaderMap,
    body: Bytes,
    engine_rid: Option<&str>,
    load_guards: LoadGuards,
    metrics: &DispatchMetrics,
    expiration: CancellationToken,
    stream_abort: CancellationToken,
    decode_slot: Option<DecodeSlot>,
) -> Result<Response<Body>, ApiError> {
    if metrics.streaming {
        // Load and duration guards live until the SSE pump ends, not just until headers arrive.
        let stream_guards: Box<dyn Send + 'static> =
            Box::new((load_guards, metrics.stream_duration_guard()));
        let first_byte = release_on_first_byte(decode_slot, metrics.first_byte_callback());
        ctx.proxy
            .forward_streaming_to(
                &worker.url,
                worker.protocol(),
                &worker.breaker,
                path,
                headers,
                body,
                engine_rid,
                Some(stream_guards),
                Some(first_byte),
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
                path,
                headers,
                body,
                engine_rid,
            )
            .await
    }
}

pub(crate) struct DispatchMetrics {
    pub(crate) registry: Arc<MetricsRegistry>,
    pub(crate) model: String,
    worker_url: String,
    mode: WorkerModeLabel,
    pub(crate) streaming: bool,
    media: MediaCounts,
    request_started_at: Instant,
    duration: Arc<RequestDurationGuard>,
}

impl DispatchMetrics {
    pub(crate) fn new(
        ctx: &AppContext,
        request: &PreparedRequest,
        response_worker: &Worker,
        request_started_at: Instant,
        duration: &Arc<RequestDurationGuard>,
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
            media: request.media,
            request_started_at,
            duration: Arc::clone(duration),
        }
    }

    // TTFT uses the first successful stream chunk, measured from request arrival.
    fn first_byte_callback(&self) -> Box<dyn FnOnce() + Send + 'static> {
        let metrics = Arc::clone(&self.registry);
        let model = self.model.clone();
        let request_started_at = self.request_started_at;
        Box::new(move || metrics.observe_ttft(&model, request_started_at.elapsed().as_secs_f64()))
    }

    fn stream_duration_guard(&self) -> Arc<RequestDurationGuard> {
        Arc::clone(&self.duration)
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
    pub(crate) fn record_dispatch_result<T: Transport>(
        &self,
        result: &Result<T::Response, ApiError>,
        engine_rid: Option<String>,
        blame: Option<&Blame>,
    ) -> RequestLogContext {
        if let Err(ApiError::StaleRequestExpired { .. }) = result {
            self.registry
                .record_stale_request(StaleRequestOutcome::Expired);
        }
        let outcome = dispatch_outcome::<T>(result);
        let worker_url = match blame {
            // The prefill task books its own outcome, so recording `outcome`
            // here would double-count it against prefill. Decode still has to
            // be accounted for: its dispatch reached the worker and was then
            // abandoned, and leaving it out would silently shrink decode's
            // dispatch counts during exactly the prefill incident an operator
            // is reading them to understand.
            Some(blame) => {
                if blame.decode_dispatched {
                    self.registry.record_worker_request(
                        &self.worker_url,
                        &self.model,
                        self.mode,
                        RequestOutcome::Cancelled,
                    );
                }
                &blame.prefill.url
            }
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
        // The app middleware emits the access log and edge counters exactly once.
        RequestLogContext {
            worker_url: worker_url.clone(),
            model_id: self.model.clone(),
            streaming: self.streaming,
            outcome,
            engine_rid,
            media: self.media,
        }
    }
}

// HTTP status determines the outcome; router cancellations and dispatch failures stay distinct.
pub(crate) fn dispatch_outcome<T: Transport>(
    result: &Result<T::Response, ApiError>,
) -> RequestOutcome {
    match result {
        Ok(response) => outcome_from_status(T::status(response)),
        Err(ApiError::StaleRequestExpired { .. }) => RequestOutcome::Cancelled,
        // These 503s come from the router, not worker backpressure.
        Err(ApiError::BreakerOpen { .. } | ApiError::WorkerMisconfigured { .. }) => {
            RequestOutcome::Error
        }
        Err(error) => outcome_from_status(error.status_code().as_u16()),
    }
}

/// Records a dispatched request's total duration once its last holder drops:
/// the handler for JSON, the SSE pump once a stream ends, so every attempt
/// of a request shares one observation.
pub(crate) struct RequestDurationGuard {
    metrics: Arc<MetricsRegistry>,
    model: String,
    request_started_at: Instant,
}

impl RequestDurationGuard {
    pub(super) fn new(ctx: &AppContext, model: &ModelId, request_started_at: Instant) -> Arc<Self> {
        Arc::new(Self {
            metrics: Arc::clone(&ctx.metrics),
            model: model.0.clone(),
            request_started_at,
        })
    }
}

impl Drop for RequestDurationGuard {
    fn drop(&mut self) {
        self.metrics
            .observe_request_duration(&self.model, self.request_started_at.elapsed().as_secs_f64());
    }
}

/// Prefill's failure ended the request, so the client-visible outcome belongs to
/// `prefill` rather than to the decode worker the dispatch had selected.
pub(crate) struct Blame {
    pub(crate) prefill: Arc<Worker>,
    /// Whether decode had already reached the wire when it was dropped. False
    /// only if prefill failed before decode's future was ever polled, which is
    /// a pre-dispatch drop and must stay uncounted.
    pub(crate) decode_dispatched: bool,
}

pub(crate) type DispatchResult<R> = (Result<R, ApiError>, Option<Blame>, Option<Bytes>);

pub(crate) async fn response_before_expiry<R>(
    model: &str,
    expiration: CancellationToken,
    pending: impl Future<Output = DispatchResult<R>>,
) -> DispatchResult<R> {
    tokio::select! {
        biased;
        response = pending => response,
        _ = expiration.cancelled() => {
            (Err(ApiError::StaleRequestExpired { model: model.to_owned() }), None, None)
        }
    }
}
