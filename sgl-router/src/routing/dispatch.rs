// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::{
    future::Future,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

use axum::body::Body;
use axum::http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, Response, StatusCode};
use bytes::Bytes;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::config::DpRankPolicy;
use crate::policies::dp_rank::{expected_rank_tokens, select_dp_rank};
use crate::proxy::grpc::{Additions, BootstrapFields};
use crate::proxy::sse::{self, ErrorEventScanner, PumpItem};
use crate::proxy::Proxy;
use crate::routing::selection::nonempty_header;
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::{
    outcome_from_status, RequestOutcome, StaleRequestOutcome, WorkerModeLabel,
};
use crate::server::routes::chat::{
    forward::{
        dispatch_outcome, response_before_expiry, Blame, DispatchMetrics, Dispatched, Grpc, Http,
        Leg, LoadGuards, RequestDurationGuard, Transport, X_DATA_PARALLEL_RANK, X_SGL_DECODE_URL,
    },
    preparation::{append_fields, generate_room_id, generate_room_id_for_rank, PreparedRequest},
};
use crate::state::kv_events::prefix_provider::RankDepths;
use crate::workers::{DecodeSlot, DpRankGuard, Worker};

/// A plain worker, or a prefill worker paired with a decode worker for PD.
pub(crate) struct SelectedWorkers {
    pub(crate) prefill: Arc<Worker>,
    pub(crate) decode: Option<Arc<Worker>>,
    pub(crate) track_dispatch_timestamps: bool,
}

pub(crate) async fn dispatch<T: PrefillTransport>(
    transport: T,
    ctx: &AppContext,
    request: &mut PreparedRequest,
    workers: SelectedWorkers,
    mut headers: HeaderMap,
    request_started_at: Instant,
    duration: &Arc<RequestDurationGuard>,
) -> Result<Dispatched<T::Response>, ApiError> {
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

    // Track worker occupancy and the prompt's contribution to active load.
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
        request_started_at.elapsed(),
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
    let additions = Additions {
        rid: engine_rid.clone(),
        bootstrap: pd.as_ref().map(|(_, bootstrap)| bootstrap.clone()),
        sampling_defaults: request.sampling_defaults(),
        routed_dp_rank: None,
        disagg_prefill_dp_rank: None,
    };
    let body = request.outgoing_body(
        &ctx.metrics,
        pd.as_ref().map(|(_, bootstrap)| bootstrap),
        engine_rid.as_deref(),
    )?;
    let prefill_leg = Leg::new(
        dp_aware,
        headers.clone(),
        &body,
        &additions,
        prefill_rank,
        None,
    );
    let prefill_load_guards = (
        worker_load_guard,
        active_request_guard,
        prefill_rank.map(|rank| rank_guard(&prefill, rank, ctx.config.model.dp_rank_policy)),
    );

    // In PD mode, prefill runs independently and decode supplies the client response.
    let stream_abort = CancellationToken::new();
    let decode_slot = (pd.is_some() && ctx.decode_gate.is_enabled() && engine_rid.is_some())
        .then(DecodeSlot::default);
    let mut unanswered = ReleaseOnDrop(decode_slot.clone());
    let (response_worker, response_leg, response_load_guards, prefill_task) =
        if let Some((decode, bootstrap)) = pd {
            let task = spawn_prefill_request(
                transport.clone(),
                ctx,
                &metrics,
                Arc::clone(&prefill),
                path,
                prefill_leg,
                prefill_load_guards,
                bootstrap.room,
                stream_abort.clone(),
                decode_slot.clone(),
            );
            // Counted from arrival, so decode selection sees requests waiting here.
            let decode_load = decode.load_guard();
            if let Some(slot) = &decode_slot {
                let mut abort = ctx.proxy.abort_on_drop(
                    &prefill.url,
                    prefill.protocol(),
                    &headers,
                    engine_rid.as_deref(),
                );
                admit_decode(
                    ctx,
                    request,
                    &prefill,
                    prefill_rank,
                    request_started_at,
                    slot,
                    &stream_abort,
                )
                .await?;
                abort.disarm();
            }
            let decode_policy = ctx.config.model.decode_dp_rank_policy();
            let decode_rank = dp_aware
                .then(|| {
                    let rank_tokens = rank_tokens(ctx, &decode, decode_policy);
                    select_dp_rank(&decode, None, &[], None, &rank_tokens, decode_policy)
                })
                .flatten();
            let decode_load_guards = (
                decode_load,
                ctx.router_inflight_load.register_aged(
                    decode.id.clone(),
                    decode.url.clone(),
                    0,
                    1,
                    request_started_at.elapsed(),
                ),
                decode_rank.map(|rank| rank_guard(&decode, rank, decode_policy)),
            );
            let decode_leg = Leg::new(
                dp_aware,
                headers,
                &body,
                &additions,
                decode_rank,
                prefill_rank,
            );
            (
                decode,
                decode_leg,
                decode_load_guards,
                Some((task, prefill)),
            )
        } else {
            (prefill, prefill_leg, prefill_load_guards, None)
        };

    // In PD mode, prefill can finish before decode. Watch the registration
    // held by the response so expiration remains live for its full lifetime.
    let expiration_token = response_load_guards.1.cancel_token().clone();
    let response = transport.send(
        ctx,
        &response_worker,
        path,
        response_leg,
        engine_rid.as_deref(),
        response_load_guards,
        &metrics,
        expiration_token.clone(),
        stream_abort,
        decode_slot,
    );
    // A logprob merge needs prefill's body, so it waits for prefill too.
    let wait_for_prefill = ctx.config.proxy.stream_after_prefill || request.merges_input_logprobs;
    let dispatch = async {
        match prefill_task {
            Some((task, prefill)) => forward_pd(task, prefill, response, wait_for_prefill).await,
            None => (response.await, None, None),
        }
    };
    let (result, blamed_prefill, prefill_body) =
        response_before_expiry(&metrics.model, expiration_token, dispatch).await;
    if matches!(dispatch_outcome::<T>(&result), RequestOutcome::Success) {
        unanswered.0 = None;
    }
    // A 2xx stream is already the client's; any other success or client error is final too.
    let retryable = matches!(
        dispatch_outcome::<T>(&result),
        RequestOutcome::Error | RequestOutcome::Backpressure
    );
    let retry_excluding = match (&blamed_prefill, pd_prefill) {
        _ if !retryable => Vec::new(),
        // The other PD side may still hold a caller's rid, which an engine refuses twice.
        (_, Some(prefill)) if request.caller_set_rid => vec![prefill, response_worker.id.clone()],
        (Some(blame), _) => vec![blame.prefill.id.clone()],
        (None, _) => vec![response_worker.id.clone()],
    };
    let result = match result {
        Ok(response) => {
            Ok(T::respond_with_prefill(request, response, metrics.streaming, prefill_body).await)
        }
        result => result,
    };
    let log_context =
        metrics.record_dispatch_result::<T>(&result, engine_rid, blamed_prefill.as_ref());
    Ok(Dispatched {
        result,
        log_context,
        decode_url_header,
        retry_excluding,
    })
}

async fn admit_decode(
    ctx: &AppContext,
    request: &PreparedRequest,
    prefill: &Worker,
    rank: Option<u32>,
    request_started_at: Instant,
    slot: &DecodeSlot,
    prefill_failed: &CancellationToken,
) -> Result<(), ApiError> {
    let budget = ctx
        .router_inflight_load
        .stale_request_timeout()
        .saturating_sub(request_started_at.elapsed());
    let admit = ctx.decode_gate.admit(&prefill.id, rank, slot);
    tokio::select! {
        admitted = tokio::time::timeout(budget, admit) => admitted.map_err(|_| {
            ctx.metrics
                .record_stale_request(StaleRequestOutcome::Expired);
            ApiError::StaleRequestExpired {
                model: request.model.0.clone(),
            }
        }),
        () = prefill_failed.cancelled() => Ok(()),
    }
}

struct ReleaseOnDrop(Option<DecodeSlot>);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        if let Some(slot) = &self.0 {
            slot.release();
        }
    }
}

impl Leg {
    fn new(
        dp_aware: bool,
        headers: HeaderMap,
        body: &Bytes,
        additions: &Additions,
        rank: Option<u32>,
        prefill_rank: Option<u32>,
    ) -> Self {
        let (headers, body) = with_dp_rank(dp_aware, headers, body, rank, prefill_rank);
        let routed_dp_rank = dp_aware.then_some(rank);
        let additions = Additions {
            routed_dp_rank,
            disagg_prefill_dp_rank: prefill_rank,
            ..additions.clone()
        };
        Self {
            headers,
            body,
            additions,
        }
    }
}

pub(crate) trait PrefillTransport: Transport {
    /// Run a PD prefill to completion and judge it.
    fn prefill(
        self,
        proxy: Arc<Proxy>,
        worker: Arc<Worker>,
        path: &'static str,
        leg: Leg,
    ) -> impl Future<Output = PrefillOutcome<Self::Response>> + Send;
    fn respond_with_prefill(
        request: &mut PreparedRequest,
        response: Self::Response,
        streaming: bool,
        _prefill_body: Option<Bytes>,
    ) -> impl Future<Output = Self::Response> + Send {
        Self::respond(request, response, streaming)
    }
}

impl PrefillTransport for Http {
    async fn respond_with_prefill(
        request: &mut PreparedRequest,
        response: Self::Response,
        streaming: bool,
        prefill_body: Option<Bytes>,
    ) -> Self::Response {
        if request.responder.is_none() {
            if let Some(prefill_body) = prefill_body.filter(|_| request.merges_input_logprobs) {
                return merge_input_logprobs(&prefill_body, response).await;
            }
        }
        Self::respond(request, response, streaming).await
    }

    async fn prefill(
        self,
        proxy: Arc<Proxy>,
        worker: Arc<Worker>,
        path: &'static str,
        leg: Leg,
    ) -> PrefillOutcome {
        let result = proxy
            .forward_json_to(
                &worker.url,
                worker.protocol(),
                &worker.breaker,
                path,
                &leg.headers,
                leg.body,
                None,
            )
            .await;
        prefill_outcome(result).await
    }
}
impl PrefillTransport for Grpc {
    async fn prefill(
        self,
        proxy: Arc<Proxy>,
        worker: Arc<Worker>,
        _path: &'static str,
        leg: Leg,
    ) -> PrefillOutcome<Self::Response> {
        let (headers, additions) = (&leg.headers, &leg.additions);
        let result = proxy
            .forward_grpc(
                &worker, &self.0, headers, additions, None, false, None, None, None, None, None,
            )
            .await;
        let status = match result {
            Ok(response) if response.status.is_success() => {
                let replies: Vec<_> = response.replies.collect().await;
                let mut scanner = ErrorEventScanner::default();
                let clean = replies.iter().all(|reply| {
                    reply
                        .as_ref()
                        .is_ok_and(|reply| !reply.is_error_event(&mut scanner))
                });
                if clean {
                    return PrefillOutcome::Completed(None);
                }
                response.status
            }
            Ok(response)
                if matches!(
                    outcome_from_status(response.status.as_u16()),
                    RequestOutcome::Error
                ) =>
            {
                response.status
            }
            result => return PrefillOutcome::Failed(result),
        };
        PrefillOutcome::Failed(Err(ApiError::PrefillFailed {
            status: Some(status),
        }))
    }
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
    let placement = match (key, &ctx.dp_rank_prefix_provider, &request.tokens) {
        (None, Some(provider), Some(tokens)) => {
            provider.rank_depths(&tokens.ids, &request.cache_namespace, &worker.url)
        }
        _ => RankDepths::default(),
    };
    let rank_tokens = rank_tokens(ctx, worker, model.dp_rank_policy);
    let rank = select_dp_rank(
        worker,
        key,
        &placement.depths,
        placement.preferred,
        &rank_tokens,
        model.dp_rank_policy,
    );
    if let (Some(rank), Some(provider), Some(tokens)) =
        (rank, &ctx.dp_rank_prefix_provider, &request.tokens)
    {
        provider.record_rank(&tokens.ids, &request.cache_namespace, &worker.url, rank);
    }
    rank
}

/// KV tokens per rank of `worker` when `policy` is `least_tokens`: what each
/// rank is expected to hold, from the engine's fresh readings and this
/// router's dispatches since them.
/// Empty under any other policy, and while any rank lacks a fresh reading,
/// which makes the selector fall back to its in-flight count.
fn rank_tokens(ctx: &AppContext, worker: &Worker, policy: DpRankPolicy) -> Vec<(u32, u64)> {
    if policy != DpRankPolicy::LeastTokens || worker.dp_ranks() <= 1 {
        return Vec::new();
    }
    ctx.engine_reported_load
        .fresh_rank_loads(&worker.url, Instant::now())
        .map_or_else(Vec::new, |loads| expected_rank_tokens(worker, &loads))
}

/// Holds the rank slot; stamped under `least_tokens` so later selections can
/// count it against the engine's last reading.
fn rank_guard(worker: &Worker, rank: u32, policy: DpRankPolicy) -> DpRankGuard {
    if policy == DpRankPolicy::LeastTokens {
        worker.dp_rank_timestamped_guard(rank)
    } else {
        worker.dp_rank_guard(rank)
    }
}

/// Under `--dp-aware` the router owns the rank: the header pins it for chat, and
/// the body's `routed_dp_rank`, which `/generate` reads instead, carries it or
/// null to unpin.
/// Decode's body also names the pinned prefill rank, so decode need not look it up.
fn with_dp_rank(
    dp_aware: bool,
    mut headers: HeaderMap,
    body: &Bytes,
    rank: Option<u32>,
    prefill_rank: Option<u32>,
) -> (HeaderMap, Bytes) {
    if !dp_aware {
        return (headers, body.clone());
    }
    if let Some(rank) = rank {
        headers.insert(X_DATA_PARALLEL_RANK, HeaderValue::from(rank));
    }
    let rank = rank.map_or_else(|| "null".to_owned(), |rank| rank.to_string());
    let mut fields = vec![("routed_dp_rank", rank)];
    fields.extend(prefill_rank.map(|rank| ("disagg_prefill_dp_rank", rank.to_string())));
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

/// How a prefill ended.
#[derive(Debug)]
pub(crate) enum PrefillOutcome<R = Response<Body>> {
    /// Succeeded, with its HTTP body.
    Completed(Option<Bytes>),
    /// Its client-visible failure.
    Failed(Result<R, ApiError>),
}

/// Runs prefill to completion even if the client disconnects. A failure also
/// aborts decode's stream until its first token, which proves KV transfer completed.
#[allow(clippy::too_many_arguments)]
fn spawn_prefill_request<T: PrefillTransport>(
    transport: T,
    ctx: &AppContext,
    metrics: &DispatchMetrics,
    prefill_worker: Arc<Worker>,
    path: &'static str,
    leg: Leg,
    load_guards: LoadGuards,
    bootstrap_room: u64,
    stream_abort: CancellationToken,
    slot: Option<DecodeSlot>,
) -> tokio::task::JoinHandle<PrefillOutcome<T::Response>> {
    let proxy = Arc::clone(&ctx.proxy);
    let (registry, model) = (Arc::clone(&metrics.registry), metrics.model.clone());
    tokio::spawn(async move {
        let _load_guards = load_guards;
        let prefill = transport
            .prefill(proxy, Arc::clone(&prefill_worker), path, leg)
            .await;
        if let Some(slot) = &slot {
            slot.release();
        }
        let failure = match &prefill {
            PrefillOutcome::Completed(_) => None,
            PrefillOutcome::Failed(failure) => Some(failure),
        };
        let prefill_url = &prefill_worker.url;
        let outcome = failure.map_or(RequestOutcome::Success, dispatch_outcome::<T>);
        registry.record_worker_request(prefill_url, &model, WorkerModeLabel::Prefill, outcome);
        match failure {
            None => tracing::debug!(%prefill_url, bootstrap_room, "prefill side completed"),
            Some(Ok(response)) => tracing::debug!(
                %prefill_url, bootstrap_room, status = T::status(response),
                "prefill rejected the request",
            ),
            Some(Err(error)) => {
                tracing::warn!(%prefill_url, bootstrap_room, %error, "prefill failed")
            }
        }
        if failure.is_some() {
            stream_abort.cancel();
        }
        prefill
    })
}

/// Client errors and backpressure pass through; only a prefill fault becomes
/// `prefill_failed`.
async fn prefill_outcome(result: Result<Response<Body>, ApiError>) -> PrefillOutcome {
    let status = match result {
        // A streaming prefill reports a late failure as a 200 carrying an SSE
        // error event; one with a 4xx code is the engine refusing the request.
        Ok(response) if response.status().is_success() => {
            let status = response.status();
            match axum::body::to_bytes(response.into_body(), usize::MAX).await {
                Ok(body) if !sse::has_error_event(&body) => {
                    return PrefillOutcome::Completed(Some(body))
                }
                Ok(body) => match sse::client_error_event(&body) {
                    Some((status, error)) => {
                        let mut response = Response::new(Body::from(error));
                        *response.status_mut() = status;
                        let json = HeaderValue::from_static("application/json");
                        response.headers_mut().insert(CONTENT_TYPE, json);
                        return PrefillOutcome::Failed(Ok(response));
                    }
                    None => status,
                },
                Err(_) => status,
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
        result => return PrefillOutcome::Failed(result),
    };
    PrefillOutcome::Failed(Err(ApiError::PrefillFailed {
        status: Some(status),
    }))
}

/// Returns decode's response as soon as decode answers, or with
/// `wait_for_prefill` once prefill also completes, unless prefill fails
/// first; then prefill's failure is returned along with the blamed prefill.
/// Prefill's body comes along when prefill completed before the return.
async fn forward_pd<R>(
    mut task: tokio::task::JoinHandle<PrefillOutcome<R>>,
    prefill: Arc<Worker>,
    decode: impl Future<Output = Result<R, ApiError>>,
    wait_for_prefill: bool,
) -> (Result<R, ApiError>, Option<Blame>, Option<Bytes>) {
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
    let mut held = None;
    loop {
        tokio::select! {
            biased;
            outcome = &mut task => return match outcome {
                Ok(PrefillOutcome::Completed(body)) => match held {
                    Some(response) => (response, None, body),
                    None => (decode.await, None, body),
                },
                Ok(PrefillOutcome::Failed(failure)) => (failure, blame(prefill), None),
                Err(_) => (Err(ApiError::PrefillFailed { status: None }), blame(prefill), None),
            },
            // Dropping the handle leaves prefill running.
            response = &mut decode, if held.is_none() => {
                if !wait_for_prefill {
                    return (response, None, None);
                }
                held = Some(response);
            }
        }
    }
}

/// Decode's response with prefill's `meta_info.input_token_logprobs` before
/// its own. A failure status, or a body without both lists, passes through;
/// an unreadable decode body is a 502.
async fn merge_input_logprobs(prefill_body: &[u8], response: Response<Body>) -> Response<Body> {
    if !response.status().is_success() {
        return response;
    }
    let (parts, body) = response.into_parts();
    let Ok(decode_body) = axum::body::to_bytes(body, usize::MAX).await else {
        let mut response = Response::new(Body::empty());
        *response.status_mut() = StatusCode::BAD_GATEWAY;
        return response;
    };
    let body =
        prepend_prefill_logprobs(prefill_body, &decode_body).map_or(decode_body, Bytes::from);
    Response::from_parts(parts, Body::from(body))
}

fn prepend_prefill_logprobs(prefill_body: &[u8], decode_body: &[u8]) -> Option<Vec<u8>> {
    const LOGPROBS: &str = "/meta_info/input_token_logprobs";
    let prefill: serde_json::Value = serde_json::from_slice(prefill_body).ok()?;
    let mut decode: serde_json::Value = serde_json::from_slice(decode_body).ok()?;
    let prefill_logprobs = prefill.pointer(LOGPROBS)?.as_array()?;
    let decode_logprobs = decode.pointer_mut(LOGPROBS)?.as_array_mut()?;
    decode_logprobs.splice(0..0, prefill_logprobs.iter().cloned());
    serde_json::to_vec(&decode).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::WorkerMode;
    use axum::http::StatusCode;

    #[tokio::test]
    async fn stream_after_prefill_waits_for_prefill() {
        let prefill = Arc::new(Worker::new(crate::discovery::WorkerSpec {
            mode: WorkerMode::Prefill,
            ..Default::default()
        }));
        for wait in [false, true] {
            let done = Arc::new(AtomicBool::new(false));
            let finished = Arc::clone(&done);
            let task = tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                finished.store(true, Ordering::SeqCst);
                PrefillOutcome::Completed(Some(Bytes::from_static(b"p")))
            });
            let decode = async { Ok(Response::new(Body::empty())) };
            let (response, blame, body) =
                forward_pd(task, Arc::clone(&prefill), decode, wait).await;
            assert!(response.is_ok() && blame.is_none());
            assert_eq!(done.load(Ordering::SeqCst), wait);
            assert_eq!(body, wait.then(|| Bytes::from_static(b"p")));
        }
    }

    fn sse(events: &[&str]) -> Result<Response<Body>, ApiError> {
        let body: String = events.iter().map(|e| format!("data: {e}\n\n")).collect();
        Ok(Response::new(Body::from(body)))
    }

    async fn body_json(response: Response<Body>) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX);
        serde_json::from_slice(&bytes.await.unwrap()).unwrap()
    }

    fn assert_prefill_failed(outcome: PrefillOutcome) {
        match outcome {
            PrefillOutcome::Failed(Err(ApiError::PrefillFailed { status })) => {
                assert_eq!(status, Some(StatusCode::OK));
            }
            other => panic!("expected prefill_failed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn in_band_client_error_passes_through() {
        // As sglang's OpenAI entrypoints stream a 4xx abort (`create_streaming_error_response`).
        let error = r#"{"error": {"object": "error", "message": "Invalid grammar", "type": "BAD_REQUEST", "param": null, "code": 400}}"#;
        let PrefillOutcome::Failed(Ok(response)) = prefill_outcome(sse(&[error, "[DONE]"])).await
        else {
            panic!("expected the engine's response");
        };
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(response.headers()[CONTENT_TYPE], "application/json");
        let expected: serde_json::Value = serde_json::from_str(error).unwrap();
        assert_eq!(body_json(response).await, expected);

        // As native `/generate` streams it (compact orjson): an abort finish reason with a status code.
        let native = r#"{"text":"","meta_info":{"finish_reason":{"type":"abort","message":"Invalid grammar","status_code":422}}}"#;
        let PrefillOutcome::Failed(Ok(response)) = prefill_outcome(sse(&[native, "[DONE]"])).await
        else {
            panic!("expected the engine's response");
        };
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            body_json(response).await,
            serde_json::json!({"error": {"message": "Invalid grammar", "code": 422}}),
        );
    }

    #[tokio::test]
    async fn in_band_server_error_stays_prefill_failed() {
        let error = r#"{"error": {"object": "error", "message": "boom", "type": "INTERNAL_SERVER_ERROR", "param": null, "code": 500}}"#;
        assert_prefill_failed(prefill_outcome(sse(&[error, "[DONE]"])).await);
        let native = r#"{"text":"","meta_info":{"finish_reason":{"type":"abort","message":"boom","status_code":503}}}"#;
        assert_prefill_failed(prefill_outcome(sse(&[native])).await);
    }

    #[tokio::test]
    async fn unparseable_in_band_error_stays_prefill_failed() {
        for event in [
            r#"{"error": {"message": "truncated", "code": 40"#,
            r#"{"error": {"message": "no code"}}"#,
            r#"{"error": {"message": "string code", "code": "400"}}"#,
            r#"{"error": {"message": "not a status", "code": 4000}}"#,
            r#"{"error": "bare string"}"#,
        ] {
            assert_prefill_failed(prefill_outcome(sse(&[event])).await);
        }
    }

    #[tokio::test]
    async fn clean_prefill_stream_is_not_a_failure() {
        let chunk = r#"{"choices": [{"delta": {"content": "{\"error\": {\"code\": 400}}"}}]}"#;
        assert!(matches!(
            prefill_outcome(sse(&[chunk, "[DONE]"])).await,
            PrefillOutcome::Completed(_)
        ));
    }
}
