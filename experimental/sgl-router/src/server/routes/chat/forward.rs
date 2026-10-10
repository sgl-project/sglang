// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Plain and PD forwarding, including load tracking and streaming metrics.

use super::nonempty_header;
use super::preparation::{
    append_fields, generate_room_id, generate_room_id_for_rank, BootstrapFields, PreparedRequest,
};
use crate::discovery::{ModelId, WorkerId, WorkerMode};
use crate::policies::dp_rank::select_dp_rank;
use crate::proxy::sse::{self, StreamEnd, StreamEndReason};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::{
    classify_stream_end, outcome_from_status, MetricsRegistry, RequestLogContext, RequestOutcome,
    StaleRequestOutcome, WorkerModeLabel,
};
use crate::state::load_monitor::router_inflight_load::RouterInflightLoadGuard;
use crate::workers::{DpRankGuard, LoadGuard, Worker};
use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response};
use axum::response::IntoResponse;
use bytes::Bytes;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

// Expose the selected decode worker to both PD workers and the client.
const X_SGL_DECODE_URL: HeaderName = HeaderName::from_static("x-sgl-decode-url");
// SGLang's DP controller dispatches to this rank; it outranks `routed_dp_rank` in the body.
const X_DATA_PARALLEL_RANK: HeaderName = HeaderName::from_static("x-data-parallel-rank");
type LoadGuards = (LoadGuard, RouterInflightLoadGuard, Option<DpRankGuard>);

/// A plain worker, or a prefill worker paired with a decode worker for PD.
pub(super) struct SelectedWorkers {
    pub(super) prefill: Arc<Worker>,
    pub(super) decode: Option<Arc<Worker>>,
    pub(super) track_dispatch_timestamps: bool,
}

/// One dispatch attempt's client-ready response.
pub(super) struct Dispatched {
    pub(super) response: Response<Body>,
    /// The workers to avoid on a retry, set only when nothing beyond a
    /// failure status reached the client, so another worker may serve it.
    pub(super) retry_excluding: Vec<WorkerId>,
}

/// PD sends to both workers and returns the decode response.
pub(super) async fn forward_request(
    ctx: &AppContext,
    request: &mut PreparedRequest,
    workers: SelectedWorkers,
    mut headers: HeaderMap,
    request_started_at: Instant,
    duration: &Arc<RequestDurationGuard>,
) -> Result<Dispatched, ApiError> {
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
    // Only the router chooses DP ranks; a client-supplied rank is never forwarded.
    headers.remove(X_DATA_PARALLEL_RANK);
    let dp_aware = ctx.config.model.dp_aware && request.accepts_dp_rank();
    // The engine gives fan-out item i the room `room + i`, so in PD decode looks
    // for each item on a different prefill rank; one pinned rank would break that.
    let unpin_prefill = dp_aware && decode.is_some() && request.fans_out;
    let prefill_rank = (dp_aware && !unpin_prefill)
        .then(|| prompt_dp_rank(ctx, request, &headers, &prefill))
        .flatten();
    let decode_rank = decode
        .as_deref()
        .filter(|_| dp_aware)
        .and_then(|decode| select_dp_rank(decode, None, &[], ctx.config.model.dp_rank_policy));

    // Track worker occupancy and the prompt's contribution to active load.
    // A retry keeps the request's stale deadline rather than starting a new one.
    let in_flight = request_started_at.elapsed();
    let worker_load_guard = if track_dispatch_timestamps {
        prefill.timestamped_load_guard()
    } else {
        prefill.load_guard()
    };
    let active_request_guard = ctx.router_inflight_load.register_aged(
        prefill.id.clone(),
        prefill.url.clone(),
        request.input_token_count,
        0,
        in_flight,
    );
    // Attribute the outcome to the worker supplying the client-visible response.
    let metrics = DispatchMetrics::new(
        ctx,
        request,
        decode.as_deref().unwrap_or(&prefill),
        request_started_at,
        duration,
    );
    let pd_prefill = decode.is_some().then(|| prefill.id.clone());
    // Both PD workers receive the same bootstrap room to coordinate KV transfer.
    let pd = decode.map(|decode| {
        let bootstrap = BootstrapFields {
            host: prefill.bootstrap_host().to_string(),
            port: prefill.bootstrap_port(),
            room: match prefill_rank {
                Some(rank) => generate_room_id_for_rank(rank, prefill.dp_ranks()),
                None => generate_room_id(),
            },
        };
        (decode, bootstrap)
    });
    let path = request.path;
    let engine_rid = request.engine_rid();
    let body = request.outgoing_body(
        ctx,
        pd.as_ref().map(|(_, bootstrap)| bootstrap),
        engine_rid.as_deref(),
    )?;
    let (prefill_headers, prefill_body) =
        with_dp_rank(dp_aware, headers.clone(), &body, prefill_rank);
    let prefill_load_guards = (
        worker_load_guard,
        active_request_guard,
        prefill_rank.map(|rank| prefill.dp_rank_guard(rank)),
    );

    // In PD mode, prefill runs independently and decode supplies the client response.
    let stream_abort = CancellationToken::new();
    let (response_worker, response_headers, response_body, response_load_guards, prefill_task) =
        if let Some((decode, bootstrap)) = pd {
            let task = spawn_prefill_request(
                ctx,
                &metrics,
                Arc::clone(&prefill),
                path,
                prefill_headers,
                prefill_body,
                prefill_load_guards,
                bootstrap.room,
                stream_abort.clone(),
            );
            let decode_load_guards = (
                decode.load_guard(),
                ctx.router_inflight_load.register_aged(
                    decode.id.clone(),
                    decode.url.clone(),
                    0,
                    1,
                    in_flight,
                ),
                decode_rank.map(|rank| decode.dp_rank_guard(rank)),
            );
            let (decode_headers, decode_body) = with_dp_rank(dp_aware, headers, &body, decode_rank);
            (
                decode,
                decode_headers,
                decode_body,
                decode_load_guards,
                Some((task, prefill)),
            )
        } else {
            (
                prefill,
                prefill_headers,
                prefill_body,
                prefill_load_guards,
                None,
            )
        };

    // In PD mode, prefill can finish before decode. Watch the registration
    // held by the response so expiration remains live for its full lifetime.
    let expiration_token = response_load_guards.1.cancel_token().clone();
    let response = forward_to_response_worker(
        ctx,
        &response_worker,
        path,
        &response_headers,
        response_body,
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
    // A 2xx stream is already the client's; any other success or client error is final too.
    let retryable = matches!(
        dispatch_outcome(&result),
        RequestOutcome::Error | RequestOutcome::Backpressure
    );
    let retry_excluding = match (&blamed_prefill, pd_prefill) {
        _ if !retryable => Vec::new(),
        // The other PD side may still hold a caller's rid, which an engine refuses twice.
        (_, Some(prefill)) if request.caller_set_rid => vec![prefill, response_worker.id.clone()],
        (Some(blame), _) => vec![blame.prefill.id.clone()],
        (None, _) => vec![response_worker.id.clone()],
    };
    // Converted first, so metrics and the access log see the client's status.
    let result = match result {
        Ok(response) if request.responder.is_some() => {
            Ok(super::openai::respond(&mut request.responder, response, metrics.streaming).await)
        }
        result => result,
    };
    let log_context = metrics.record_dispatch_result(&result, engine_rid, blamed_prefill.as_ref());
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
    Ok(Dispatched {
        response,
        retry_excluding,
    })
}

/// Rank for the worker that computes the prompt; decode gets its KV from
/// prefill, so it is placed by load alone.
fn prompt_dp_rank(
    ctx: &AppContext,
    request: &PreparedRequest,
    headers: &HeaderMap,
    worker: &Worker,
) -> Option<u32> {
    if worker.dp_ranks() <= 1 {
        return None;
    }
    let model = &ctx.config.model;
    let sticky = model.sticky.as_ref().map(|c| c.header_name.as_str());
    let session = model
        .affinity
        .as_ref()
        .map(|c| c.session_id_header.as_str());
    let key = [sticky, session]
        .into_iter()
        .flatten()
        .find_map(|name| nonempty_header(headers, name));
    let prefix_depths = match (key, &ctx.dp_rank_prefix_provider, &request.tokens) {
        (None, Some(provider), Some(tokens)) => provider.rank_depths(&tokens.ids, &worker.url),
        _ => Vec::new(),
    };
    select_dp_rank(worker, key, &prefix_depths, model.dp_rank_policy)
}

/// Under `--dp-aware` the router owns the rank: the header pins it for chat, and
/// the body, which `/generate` reads instead, carries it or null to unpin.
fn with_dp_rank(
    dp_aware: bool,
    mut headers: HeaderMap,
    body: &Bytes,
    rank: Option<u32>,
) -> (HeaderMap, Bytes) {
    if !dp_aware {
        return (headers, body.clone());
    }
    if let Some(rank) = rank {
        headers.insert(X_DATA_PARALLEL_RANK, HeaderValue::from(rank));
    }
    let rank = rank.map_or_else(|| "null".to_owned(), |rank| rank.to_string());
    let fields = [
        ("routed_dp_rank", rank),
        ("data_parallel_rank", "null".into()),
    ];
    let body = append_fields(body, &fields).unwrap_or_else(|| body.clone());
    (headers, body)
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
    path: &'static str,
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
                path,
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

/// Prefill's failure ended the request, so the client-visible outcome belongs to
/// `prefill` rather than to the decode worker the dispatch had selected.
struct Blame {
    prefill: Arc<Worker>,
    /// Whether decode had already reached the wire when it was dropped. False
    /// only if prefill failed before decode's future was ever polled, which is
    /// a pre-dispatch drop and must stay uncounted.
    decode_dispatched: bool,
}

/// Returns decode's response as soon as decode answers, unless prefill fails
/// first; then prefill's failure is returned along with the blamed prefill.
async fn forward_pd(
    task: tokio::task::JoinHandle<PrefillFailure>,
    prefill: Arc<Worker>,
    decode: impl std::future::Future<Output = Result<Response<Body>, ApiError>>,
) -> (Result<Response<Body>, ApiError>, Option<Blame>) {
    // Set on decode's first poll, which is where `forward_to_response_worker`
    // begins and the request reaches the worker. Read from the same task, so
    // `Relaxed` needs no ordering beyond what the select already gives.
    let dispatched = AtomicBool::new(false);
    let decode = async {
        dispatched.store(true, Ordering::Relaxed);
        decode.await
    };
    let mut decode = std::pin::pin!(decode);
    let blame = |prefill| {
        Some(Blame {
            prefill,
            decode_dispatched: dispatched.load(Ordering::Relaxed),
        })
    };
    tokio::select! {
        biased;
        failure = task => match failure {
            Ok(None) => (decode.await, None),
            Ok(Some(failure)) => (failure, blame(prefill)),
            Err(_) => (Err(ApiError::PrefillFailed { status: None }), blame(prefill)),
        },
        // Dropping the handle leaves prefill running.
        response = &mut decode => (response, None),
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
) -> Result<Response<Body>, ApiError> {
    if metrics.streaming {
        // Load and duration guards live until the SSE pump ends, not just until headers arrive.
        let stream_guards: Box<dyn Send + 'static> =
            Box::new((load_guards, Arc::clone(&metrics.duration)));
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
                path,
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
    duration: Arc<RequestDurationGuard>,
}

impl DispatchMetrics {
    fn new(
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
        blame: Option<&Blame>,
    ) -> RequestLogContext {
        if let Err(ApiError::StaleRequestExpired { .. }) = result {
            self.registry
                .record_stale_request(StaleRequestOutcome::Expired);
        }
        let outcome = dispatch_outcome(result);
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

/// Records a dispatched request's total duration once its last holder drops:
/// the handler for JSON, the SSE pump once a stream ends, so every attempt
/// of a request shares one observation.
pub(super) struct RequestDurationGuard {
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
