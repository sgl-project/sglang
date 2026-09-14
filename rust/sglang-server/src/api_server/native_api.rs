//! The native SGLang data-plane endpoints: `/generate` (submit a request, then
//! either fold decode frames to one unary JSON response or relay them as SSE
//! `data: {json}` … `[DONE]`, byte-compatible with Python
//! `http_server.generate_request`) and `/health` + `/health_generate` (which
//! round-trip a 1-token generate probe). Frame shaping (`meta_info`, logprob
//! tuples, cumulative vs incremental streams) lives here; submission and
//! request lifetime belong to the shared [`crate::frontend::FrontendHandle`].

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::app::AppState;
use super::frame::{
    OutputAccumulator, cumulative_frame_string, frame_value, stream_frame_string, tag_value,
};
use crate::frontend::{FrontendCall, FrontendError, HealthStatus};
use crate::message::ids::Rid;
use crate::message::request::{GenerateBody, GenerateRequest};
use crate::message::response::{ChunkEvent, ResponseItem};
use crate::utils::{
    environ,
    response::{error_response, error_value},
};
use axum::{
    Json, Router,
    extract::State,
    extract::rejection::JsonRejection,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};

/// API-local timing for one request.
///
/// Python records time-to-first-token on the first output batch and end-to-end
/// latency when that request finishes. Keep both measurements here even though
/// `/generate` currently exposes only `e2e_latency`; this avoids putting
/// API-only timestamps onto scheduler messages.
#[derive(Clone, Debug)]
struct RequestTiming {
    // TODO: Move request lifecycle timing into a dedicated tracing/metrics
    // module and align its design with Python's APIServerReqTimeStats.
    created_at: Instant,
    time_to_first_token: Option<Duration>,
    e2e_latency: Option<Duration>,
}

impl RequestTiming {
    fn new() -> Self {
        Self {
            created_at: Instant::now(),
            time_to_first_token: None,
            e2e_latency: None,
        }
    }

    fn observe_first_output(&mut self) {
        self.time_to_first_token
            .get_or_insert_with(|| self.created_at.elapsed());
    }

    fn finish(&mut self) {
        self.e2e_latency
            .get_or_insert_with(|| self.created_at.elapsed());
    }

    fn terminal_latencies(&self) -> Option<(Duration, Duration)> {
        Some((self.time_to_first_token?, self.e2e_latency?))
    }
}

/// The routes this module owns, mounted by `api_server::serve`.
pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/generate", post(generate))
        .merge(health_routes())
}

/// native api error response: unary → `code` plus the JSON `body`,
/// streaming → 200 with one SSE error frame + `[DONE]`.
pub(super) fn native_error(code: StatusCode, message: &str, stream: bool) -> Response {
    error_response(code, error_value(code.as_u16(), message), stream)
}

fn native_frontend_error(error: FrontendError, stream: bool) -> Response {
    match error {
        FrontendError::Unavailable => {
            native_error(StatusCode::SERVICE_UNAVAILABLE, &error.to_string(), stream)
        }
        FrontendError::InvalidArgument(_) => {
            native_error(StatusCode::BAD_REQUEST, &error.to_string(), stream)
        }
        _ => native_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &error.to_string(),
            stream,
        ),
    }
}

/// `/health` + `/health_generate`. Both env knobs are resolved ONCE here, at
/// router build (server startup) — changing them on a live process needs a
/// restart. The deep-probe handler is built once with
/// `SGLANG_HEALTH_CHECK_TIMEOUT` frozen in and serves `/health_generate`
/// always; `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` (default true, mirroring
/// Python) decides whether `/health` shares it or, after startup warmup, is a
/// plain 200 (routing the request proves the frontend is up).
fn health_routes() -> Router<Arc<AppState>> {
    let timeout = std::time::Duration::from_secs(
        environ::env_i64("SGLANG_HEALTH_CHECK_TIMEOUT", 20).max(0) as u64,
    );
    let probe = get(move |state: State<Arc<AppState>>| health_generate(state, timeout));
    let health = if environ::env_bool("SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION", true) {
        probe.clone()
    } else {
        get(health_without_generation)
    };
    Router::new()
        .route("/health", health)
        .route("/health_generate", probe)
}

async fn health_without_generation(State(state): State<Arc<AppState>>) -> Response {
    if state.frontend.is_ready() {
        StatusCode::OK.into_response()
    } else {
        StatusCode::SERVICE_UNAVAILABLE.into_response()
    }
}

/// `GET /health_generate` — deep health: confirm the scheduler → detok path is
/// producing output. 200 if the response heartbeat advances within `timeout`
/// (from `SGLANG_HEALTH_CHECK_TIMEOUT`, frozen at router build), else 503.
/// It also returns 503 until startup warmup completes. (`/health` uses the same
/// handler when its env gate is on.)
///
/// Fires a pre-tokenized 1-token probe (`input_ids = [0]`, skips the tokenizer) so
/// an idle pipeline produces a frame, then watches the *global*
/// frontend response-activity counter (not the probe's own rid) — so a busy
/// server passes immediately and a backlog never false-503s (the analogue of
/// Python's `last_receive_tstamp`).
async fn health_generate(
    State(state): State<Arc<AppState>>,
    timeout: std::time::Duration,
) -> Response {
    match state.frontend.probe_health(timeout).await {
        Ok(HealthStatus::Healthy) => StatusCode::OK.into_response(),
        Ok(HealthStatus::NotReady | HealthStatus::Stalled) => {
            StatusCode::SERVICE_UNAVAILABLE.into_response()
        }
        Err(FrontendError::Unavailable) => native_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "service unavailable",
            false,
        ),
        Err(error) => native_error(StatusCode::INTERNAL_SERVER_ERROR, &error.to_string(), false),
    }
}

/// `POST /generate` — the native generation endpoint. Splits the body
/// into per-request payloads (a scalar body → one, a list body → a batch) and
/// dispatches to the single or batch path; a malformed body is a 400 before
/// anything reaches the scheduler.
///
/// The body is extracted as a `Result` so a deserialization failure is answered
/// with **400** (Python's status for a bad request) carrying serde's field-level
/// message, instead of axum's default 422.
async fn generate(
    State(state): State<Arc<AppState>>,
    body: Result<Json<GenerateBody>, JsonRejection>,
) -> Response {
    let mut body = match body {
        Ok(Json(body)) => body,
        // A body that fails to parse has no readable `stream` flag, so this one
        // can only answer unary — as Python's does (FastAPI rejects before its
        // handler runs).
        Err(rejection) => {
            return native_error(StatusCode::BAD_REQUEST, &rejection.body_text(), false);
        }
    };
    let stream = body.stream;
    if let Some(preferred) = &state.server_args.preferred_sampling_params
        && let Err(error) = body.apply_preferred_sampling(&preferred.0)
    {
        return native_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &error.to_string(),
            stream,
        );
    }
    // Fan `text`/`input_ids`/`sampling_params` (scalar or list) into per-request
    // payloads. `is_batch` = list form → the response is a JSON array.
    let (payloads, is_batch) = match body.into_requests() {
        Ok(v) => v,
        // The error carries its own status (a bad batch is `Validation` → 400).
        Err(e) => {
            let code = StatusCode::from_u16(e.http_status()).unwrap_or(StatusCode::BAD_REQUEST);
            return native_error(code, &e.to_string(), stream);
        }
    };
    // Python starts APIServerReqTimeStats after request normalization and before
    // tokenization / multimodal preprocessing / scheduler dispatch. Start at the
    // equivalent boundary: into_requests() has normalized the body, while prefetch
    // and every downstream stage are still ahead of us.
    let timing = RequestTiming::new();
    // Shared frontend admission performs media I/O (URL downloads, file reads)
    // on this async runtime before any request reaches the MM worker pool.
    if !is_batch {
        // `into_requests` guarantees exactly one payload for a non-batch body.
        let payload = payloads
            .into_iter()
            .next()
            .expect("into_requests yields >=1 payload");
        generate_single(&state, payload, stream, timing).await
    } else {
        generate_batch(&state, payloads, stream, timing).await
    }
}

/// A single (non-batched) `/generate`: submit one request, then either stream its
/// SSE frames or fold to one unary response.
async fn generate_single(
    state: &AppState,
    req: GenerateRequest,
    stream: bool,
    timing: RequestTiming,
) -> Response {
    // `return_text_in_logprobs` is decoded on the detok shard into `*_txt`, so
    // `frame_value` just reads them — no tokenizer needed here.
    let mut call = match state.frontend.generate(req).await {
        Ok(call) => call,
        Err(error) => return native_frontend_error(error, stream),
    };
    let rid = call.rid().clone();
    // Cumulative frames (SGLang default) vs per-step deltas.
    let incremental = state.server_args.incremental_streaming_output;

    if stream {
        // A single request is a 1-element batch without the `index` field — reuse
        // the same stream so the frame/abort/truncation logic lives in one place.
        use futures::StreamExt;
        let s = generation_event_stream(vec![(call, timing)], incremental, false)
            .map(|data| Ok::<_, Infallible>(Event::default().data(data)));
        Sse::new(s).into_response()
    } else {
        // Unary: fold to the terminal and respond once. `FrontendCall` disarms
        // itself on a terminal item; truncation leaves it armed for drop cleanup.
        let (status, value) = drain_unary(&mut call, rid.client_facing(), timing).await;
        (status, Json(value)).into_response()
    }
}

/// Fold a unary request to its terminal HTTP result. Shared by single + batch.
async fn drain_unary(
    call: &mut FrontendCall,
    rid_str: &str,
    mut timing: RequestTiming,
) -> (StatusCode, serde_json::Value) {
    let mut acc = OutputAccumulator::default();
    while let Some(item) = call.recv().await {
        match item {
            ResponseItem::Frame(out) => {
                timing.observe_first_output();
                acc.fold(&out);
            }
            ResponseItem::Done(out) => {
                timing.observe_first_output();
                timing.finish();
                acc.fold(&out);
                let final_out = acc.into_output();
                // A validation abort carries its own HTTP status + diagnostic.
                if let Some((code, message)) = final_out
                    .finish_reason
                    .as_ref()
                    .and_then(|f| f.abort_status())
                {
                    let status =
                        StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    return (status, error_value(code, message));
                }
                let mut value = frame_value(&final_out, rid_str);
                add_e2e_latency(&mut value, &timing);
                return (StatusCode::OK, value);
            }
            ResponseItem::Error(e) => {
                timing.finish();
                let code = e.http_status();
                let status =
                    StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                return (status, error_value(code, &e.to_string()));
            }
            ResponseItem::Control(_) | ResponseItem::Data(_) => continue, // never on `/generate`
        }
    }
    // Sender dropped without a terminal item: the shard dropped this request (a
    // truncation — a client disconnect would have dropped the handler future).
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        error_value(500, "response truncated before completion"),
    )
}

/// Batch `/generate`: submit all sub-requests first (scheduler runs them together),
/// then either (unary) drain them concurrently into a request-ordered JSON array,
/// or (streaming) multiplex their streams into one SSE response, each frame carrying
/// its `index`.
/// Each submitted call owns its own cancellation. A failed unary item is its
/// own `{ "error": … }` entry; the batch response is 200.
async fn generate_batch(
    state: &AppState,
    requests: Vec<GenerateRequest>,
    stream: bool,
    timing: RequestTiming,
) -> Response {
    // No cross-item rid collision to worry about: `into_requests` rejected duplicate
    // rids within this batch, and `Rid::from_client` made each one unique against
    // every other in-flight request.
    let calls = match state.frontend.generate_batch(requests).await {
        Ok(calls) => calls
            .into_iter()
            .map(|call| (call, timing.clone()))
            .collect(),
        Err(error) => return native_frontend_error(error, stream),
    };

    if stream {
        // Multiplex the N streams (mirrors the Python `_handle_batch_request` path);
        // Calls move into the stream so a disconnect aborts what's unfinished.
        use futures::StreamExt;
        let incremental = state.server_args.incremental_streaming_output;
        let s = generation_event_stream(calls, incremental, true)
            .map(|data| Ok::<_, Infallible>(Event::default().data(data)));
        Sse::new(s).into_response()
    } else {
        // Unary: poll every item concurrently, as Python's gather does. `join_all`
        // preserves input order for the final JSON array, while each drain observes
        // its own terminal output promptly (important for per-item e2e_latency).
        let drained = futures::future::join_all(calls.into_iter().map(
            |(mut call, request_timing)| async move {
                let client_rid = call.rid().client_facing().to_owned();
                let (_status, value) = drain_unary(&mut call, &client_rid, request_timing).await;
                value
            },
        ))
        .await;
        (StatusCode::OK, Json(serde_json::Value::Array(drained))).into_response()
    }
}

/// Await the next item from `call`, then drain whatever queued behind it (so the caller
/// can coalesce a backlog, as Python's `state.out_list` does), handing the call
/// back for `FuturesUnordered` to re-poll. Empty result = channel closed.
async fn recv_indexed(
    index: usize,
    mut call: FrontendCall,
) -> (usize, FrontendCall, Vec<ResponseItem>) {
    let mut items = Vec::new();
    match call.recv().await {
        Some(item) => items.push(item),
        None => return (index, call, items), // closed
    }
    while let Ok(item) = call.try_recv() {
        items.push(item);
    }
    (index, call, items)
}

/// Multiplex calls into SSE `data` strings + a final `[DONE]`; `with_index` tags
/// each frame (batch only), `incremental` = delta vs cumulative. Each call
/// aborts itself if the stream is dropped while it is unfinished.
fn generation_event_stream(
    calls: Vec<(FrontendCall, RequestTiming)>,
    incremental: bool,
    with_index: bool,
) -> impl futures::Stream<Item = String> {
    async_stream::stream! {
        use futures::StreamExt;

        let n = calls.len();
        let rid_strs: Vec<Rid> = calls
            .iter()
            .map(|(call, _)| call.rid().clone())
            .collect();
        let mut timings: Vec<RequestTiming> = calls
            .iter()
            .map(|(_, timing)| timing.clone())
            .collect();
        let mut accs: Vec<OutputAccumulator> =
            (0..n).map(|_| OutputAccumulator::default()).collect();

        // Batch position, tagged onto every frame (a single request omits it).
        let idx = |i: usize| with_index.then_some(i);

        // Poll all receivers concurrently; re-arm a receiver's future after each
        // non-terminal frame so its stream keeps flowing.
        let mut futs = futures::stream::FuturesUnordered::new();
        for (i, (call, _)) in calls.into_iter().enumerate() {
            futs.push(recv_indexed(i, call));
        }

        while let Some((i, call, items)) = futs.next().await {
            if items.is_empty() {
                // Channel closed with no terminal → truncation for this item;
                // leave its rid armed so the scheduler work is aborted.
                yield tag_value(error_value(500, "response truncated before completion"), idx(i));
                continue;
            }

            // Cumulative frames supersede one another, so a drained backlog collapses
            // to its last (Python's `out_list[-1]`); deltas can't be dropped.
            let mut coalesced = false; // a cumulative frame is pending
            let mut terminal = None;   // (finish_reason) of a `Done` in this batch
            let mut failed = None;     // an `Error` in this batch

            for item in items {
                match item {
                    ResponseItem::Frame(out) => {
                        timings[i].observe_first_output();
                        accs[i].fold(&out);
                        if incremental {
                            yield stream_frame_string(out, &accs[i], true, rid_strs[i].client_facing(), idx(i));
                        } else {
                            coalesced = true;
                        }
                    }
                    ResponseItem::Done(out) => {
                        timings[i].observe_first_output();
                        timings[i].finish();
                        accs[i].fold(&out);
                        terminal = Some(out);
                    }
                    ResponseItem::Error(e) => {
                        timings[i].finish();
                        failed = Some(e);
                    }
                    ResponseItem::Control(_) | ResponseItem::Data(_) => {} // never on /generate
                }
            }

            if let Some(e) = failed {
                yield tag_value(error_value(e.http_status(), &e.to_string()), idx(i));
            } else if let Some(out) = terminal {
                // A validation abort → an error object, not a frame. The final frame
                // carries the full cumulative state, so any coalesced ones are moot.
                yield match out.finish_reason.as_ref().and_then(|f| f.abort_status()) {
                    Some((code, message)) => tag_value(error_value(code, message), idx(i)),
                    None => terminal_stream_frame_string(
                        out,
                        &accs[i],
                        incremental,
                        rid_strs[i].client_facing(),
                        idx(i),
                        &timings[i],
                    ),
                };
            } else {
                if coalesced {
                    yield cumulative_frame_string(&accs[i], rid_strs[i].client_facing(), idx(i));
                }
                futs.push(recv_indexed(i, call)); // keep this item flowing
            }
        }
        yield "[DONE]".to_string();
    }
}

/// Python's `e2e_latency` is `finished_time - created_time`, in seconds, and is
/// attached only when the request finishes. The Rust native API owns the same
/// lifecycle boundary, so it adds the value while handling the terminal egress
/// item rather than putting API-only timing onto every scheduler `ChunkEvent`.
fn add_e2e_latency(value: &mut serde_json::Value, timing: &RequestTiming) {
    let (time_to_first_token, e2e_latency) = timing
        .terminal_latencies()
        .expect("a successful terminal output has complete request timing");
    debug_assert!(time_to_first_token <= e2e_latency);
    value["meta_info"]["e2e_latency"] = serde_json::json!(e2e_latency.as_secs_f64());
}

/// Render a terminal streaming frame. Intermediate cumulative frames keep the
/// memoized fast path; the one terminal frame uses the Value path so it can carry
/// the request-local `e2e_latency`, exactly as Python does.
fn terminal_stream_frame_string(
    out: ChunkEvent,
    acc: &OutputAccumulator,
    incremental: bool,
    rid_str: &str,
    index: Option<usize>,
    timing: &RequestTiming,
) -> String {
    let mut value = super::frame::stream_frame_value(out, acc, incremental, rid_str);
    add_e2e_latency(&mut value, timing);
    tag_value(value, index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::request::RequestKind;
    use crate::message::response::ChunkEvent;
    use crate::tokenizer_manager::wiring::TmEvent;
    use crate::utils::error::Error;
    use axum::body::Body;
    use axum::http::Request;
    use futures::StreamExt;
    use std::time::Duration;
    use tokio::sync::mpsc;
    use tower::ServiceExt;

    fn frontend(startup_ready: bool) -> crate::frontend::FrontendHandle {
        crate::frontend::FrontendHandle::new(
            flume::unbounded().0,
            flume::unbounded().0,
            crate::frontend::FrontendConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready,
                is_disaggregation: false,
                mm_limits: Default::default(),
            },
        )
    }

    fn frame(rid: u64, text: &str) -> ResponseItem {
        ResponseItem::Frame(ChunkEvent {
            rid: Rid::from(rid.to_string()),
            text: text.into(),
            completion_tokens: 1,
            ..Default::default()
        })
    }
    fn done(rid: u64, text: &str) -> ResponseItem {
        ResponseItem::Done(ChunkEvent {
            rid: Rid::from(rid.to_string()),
            text: text.into(),
            completion_tokens: 1,
            // Parsed from the wire map Python emits, not a hand-built enum.
            finish_reason: Some(
                serde_json::from_value(serde_json::json!({"type": "length", "length": 1}))
                    .expect("finish reason must parse"),
            ),
            ..Default::default()
        })
    }
    fn parse(s: &str) -> serde_json::Value {
        serde_json::from_str(s).expect("frame is JSON")
    }

    fn timed_call(rid: u64, rx: mpsc::Receiver<ResponseItem>) -> (FrontendCall, RequestTiming) {
        timed_call_with_abort(rid, rx, flume::unbounded().0)
    }

    fn timed_call_with_abort(
        rid: u64,
        rx: mpsc::Receiver<ResponseItem>,
        abort_tx: flume::Sender<crate::tokenizer_manager::wiring::AbortSource>,
    ) -> (FrontendCall, RequestTiming) {
        (
            FrontendCall::from_test_generation_parts(Rid::from(rid.to_string()), rx, abort_tx),
            RequestTiming {
                created_at: Instant::now() - Duration::from_millis(10),
                time_to_first_token: None,
                e2e_latency: None,
            },
        )
    }

    #[tokio::test]
    async fn health_is_unavailable_before_startup_warmup_finishes() {
        let state = Arc::new(AppState {
            frontend: frontend(false),
            server_args: Arc::new(crate::message::config::ServerArgs::default()),
            chat_formatter: None,
        });

        let response = health_generate(State(state), Duration::ZERO).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn health_closed_intake_preserves_native_503_body() {
        let state = Arc::new(AppState {
            // The helper deliberately retains no intake receiver, so this
            // ready probe reaches the operational submission-failure path.
            frontend: frontend(true),
            server_args: Arc::new(crate::message::config::ServerArgs::default()),
            chat_formatter: None,
        });

        let response = health_generate(State(state), Duration::ZERO).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["message"], "service unavailable");
        assert_eq!(value["error"]["code"], 503);
    }

    #[tokio::test]
    async fn unary_http_generate_round_trips_through_frontend_handle() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        let frontend = crate::frontend::FrontendHandle::new(
            intake_tx,
            abort_tx,
            crate::frontend::FrontendConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Default::default(),
            },
        );
        let state = Arc::new(AppState {
            frontend,
            server_args: Arc::new(crate::message::config::ServerArgs::default()),
            chat_formatter: None,
        });

        let responder = tokio::spawn(async move {
            let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap()
            else {
                panic!("HTTP generation must enter through frontend intake");
            };
            assert!(admission.try_accept());
            let RequestKind::Generate(generate) = &request.kind else {
                panic!("/generate must submit a generation request");
            };
            assert_eq!(generate.input_ids.as_deref(), Some([1, 2].as_slice()));
            let rid = request.rid.clone();
            request
                .sink
                .try_send(ResponseItem::Done(ChunkEvent {
                    rid,
                    text: "ok".into(),
                    prompt_tokens: 2,
                    completion_tokens: 1,
                    finish_reason: serde_json::from_value(serde_json::json!({
                        "type": "length",
                        "length": 1
                    }))
                    .unwrap(),
                    ..Default::default()
                }))
                .unwrap();
        });

        let request = Request::builder()
            .method("POST")
            .uri("/generate")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"rid":"client-rid","input_ids":[1,2],"sampling_params":{"max_new_tokens":1}}"#,
            ))
            .unwrap();
        let response = routes().with_state(state).oneshot(request).await.unwrap();
        responder.await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["text"], "ok");
        assert_eq!(value["meta_info"]["id"], "client-rid");
        assert_eq!(value["meta_info"]["prompt_tokens"], 2);
        assert_eq!(value["meta_info"]["completion_tokens"], 1);
        assert!(abort_rx.try_recv().is_err());
    }

    #[test]
    fn request_timing_records_ttft_once_and_e2e_on_finish() {
        let mut timing = RequestTiming {
            created_at: Instant::now() - Duration::from_millis(10),
            time_to_first_token: None,
            e2e_latency: None,
        };
        assert!(timing.terminal_latencies().is_none());

        timing.observe_first_output();
        let time_to_first_token = timing.time_to_first_token.unwrap();
        timing.observe_first_output();
        assert_eq!(
            timing.time_to_first_token,
            Some(time_to_first_token),
            "later output must not overwrite TTFT"
        );

        timing.finish();
        let (recorded_ttft, e2e_latency) = timing.terminal_latencies().unwrap();
        assert_eq!(recorded_ttft, time_to_first_token);
        assert!(e2e_latency >= recorded_ttft);

        timing.finish();
        assert_eq!(
            timing.terminal_latencies(),
            Some((recorded_ttft, e2e_latency)),
            "later terminal handling must not overwrite E2E latency"
        );
    }

    /// The native unary response uses the same names and meanings as Python's
    /// TokenizerManager metadata, and adds e2e_latency only on the terminal
    /// result. The timer is seconds from normalized-request acceptance through
    /// terminal-output handling.
    #[tokio::test]
    async fn unary_terminal_meta_info_matches_python_semantics() {
        let (tx, rx) = mpsc::channel(2);
        tx.send(ResponseItem::Done(ChunkEvent {
            rid: "internal-rid".into(),
            text: "ok".into(),
            token_ids: vec![7, 8],
            prompt_tokens: 5,
            completion_tokens: 2,
            finish_reason: serde_json::from_value(serde_json::json!({
                "type": "length",
                "length": 2
            }))
            .expect("finish reason must parse"),
            ..Default::default()
        }))
        .await
        .unwrap();
        let mut call = FrontendCall::from_test_generation_parts(
            "internal-rid".into(),
            rx,
            flume::unbounded().0,
        );

        let timing = RequestTiming {
            created_at: Instant::now() - Duration::from_millis(20),
            time_to_first_token: None,
            e2e_latency: None,
        };
        let (status, value) = drain_unary(&mut call, "client-rid", timing).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(value["meta_info"]["id"], "client-rid");
        assert_eq!(value["meta_info"]["prompt_tokens"], 5);
        assert_eq!(value["meta_info"]["completion_tokens"], 2);
        assert_eq!(
            value["meta_info"]["finish_reason"],
            serde_json::json!({"type": "length", "length": 2})
        );
        assert!(
            value["meta_info"]["e2e_latency"].as_f64().unwrap() >= 0.020,
            "latency is expressed in seconds from request creation"
        );
        assert!(
            value["meta_info"].get("ttft").is_none()
                && value["meta_info"].get("time_to_first_token").is_none(),
            "TTFT is recorded internally but is not part of this PR's API"
        );
    }

    /// Two sub-requests' frames interleave into one stream, each tagged with its
    /// batch `index`; text accumulates per item; `[DONE]` comes only after both
    /// terminate, then the stream ends.
    #[tokio::test]
    async fn interleaves_indexes_and_accumulates() {
        let (tx0, rx0) = mpsc::channel(8);
        let (tx1, rx1) = mpsc::channel(8);
        let calls = vec![timed_call(10, rx0), timed_call(11, rx1)];
        let stream = generation_event_stream(calls, false, true);
        futures::pin_mut!(stream);

        // Drive deterministically: exactly one channel has data before each poll.
        tx0.send(frame(10, "a")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 0);
        assert_eq!(v["text"], "a");

        tx1.send(frame(11, "b")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 1);
        assert_eq!(v["text"], "b");

        tx0.send(done(10, "!")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 0);
        assert_eq!(v["text"], "a!", "cumulative per item");
        assert_eq!(v["meta_info"]["finish_reason"]["type"], "length");
        assert!(v["meta_info"]["e2e_latency"].as_f64().unwrap() >= 0.010);

        tx1.send(done(11, "?")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 1);
        assert_eq!(v["text"], "b?");
        assert!(v["meta_info"]["e2e_latency"].as_f64().unwrap() >= 0.010);

        assert_eq!(stream.next().await.unwrap(), "[DONE]");
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn dropping_stream_aborts_only_unfinished_calls() {
        use crate::tokenizer_manager::wiring::AbortSource;

        let (tx0, rx0) = mpsc::channel(8);
        let (tx1, rx1) = mpsc::channel(8);
        let (abort_tx, abort_rx) = flume::unbounded();
        let calls = vec![
            timed_call_with_abort(10, rx0, abort_tx.clone()),
            timed_call_with_abort(11, rx1, abort_tx),
        ];
        let mut stream = Box::pin(generation_event_stream(calls, false, true));

        tx0.send(done(10, "done")).await.unwrap();
        assert_eq!(parse(&stream.next().await.unwrap())["index"], 0);
        tx1.send(frame(11, "live")).await.unwrap();
        assert_eq!(parse(&stream.next().await.unwrap())["index"], 1);

        drop(stream);
        assert!(matches!(
            abort_rx.recv().unwrap(),
            AbortSource::Guard(rid) if rid.as_str() == "11"
        ));
        assert!(abort_rx.try_recv().is_err());
    }

    /// A per-item error is surfaced with its `index` and doesn't end the batch;
    /// `[DONE]` still waits for the other item.
    #[tokio::test]
    async fn per_item_error_carries_index() {
        let (tx0, rx0) = mpsc::channel(8);
        let (tx1, rx1) = mpsc::channel(8);
        let calls = vec![timed_call(10, rx0), timed_call(11, rx1)];
        let stream = generation_event_stream(calls, false, true);
        futures::pin_mut!(stream);

        tx0.send(ResponseItem::Error(Error::Validation("bad".into())))
            .await
            .unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 0);
        assert_eq!(v["error"]["code"], 400);

        tx1.send(done(11, "ok")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 1);

        assert_eq!(stream.next().await.unwrap(), "[DONE]");
    }

    /// `incremental=true`: each frame carries this step's **delta** text/output_ids,
    /// but `meta_info.completion_tokens` stays cumulative (matching Python).
    #[tokio::test]
    async fn incremental_emits_deltas_with_cumulative_count() {
        let (tx, rx) = mpsc::channel(8);
        let calls = vec![timed_call(10, rx)];
        let stream = generation_event_stream(calls, true, true);
        futures::pin_mut!(stream);

        tx.send(frame(10, "Hello")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "Hello");
        assert_eq!(v["meta_info"]["completion_tokens"], 1);
        assert!(v["meta_info"].get("e2e_latency").is_none());

        tx.send(frame(10, " world")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], " world", "delta, not cumulative 'Hello world'");
        assert_eq!(
            v["meta_info"]["completion_tokens"], 2,
            "count stays cumulative"
        );
        assert!(v["meta_info"].get("e2e_latency").is_none());

        tx.send(done(10, "!")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "!");
        assert_eq!(v["meta_info"]["completion_tokens"], 3);
        assert_eq!(v["meta_info"]["finish_reason"]["type"], "length");
        assert!(v["meta_info"]["e2e_latency"].as_f64().unwrap() >= 0.010);

        assert_eq!(stream.next().await.unwrap(), "[DONE]");
    }

    /// The single-request shape (`with_index=false`, one receiver) omits the
    /// `index` field entirely, and still terminates with `[DONE]`.
    #[tokio::test]
    async fn single_shape_omits_index() {
        let (tx, rx) = mpsc::channel(8);
        let calls = vec![timed_call(10, rx)];
        let stream = generation_event_stream(calls, false, false);
        futures::pin_mut!(stream);

        tx.send(done(10, "hi")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "hi");
        assert!(v.get("index").is_none(), "single response has no index");
        assert!(v["meta_info"]["e2e_latency"].as_f64().unwrap() >= 0.010);

        assert_eq!(stream.next().await.unwrap(), "[DONE]");
    }

    /// A backlog of cumulative chunks collapses to a single frame carrying the latest
    /// state — each cumulative frame supersedes the last, so emitting the intermediate
    /// ones ships the full O(T) payload again for nothing. Mirrors the Python waiter's
    /// `out = out_list[-1]`. This is the whole point of draining in `recv_indexed`.
    #[tokio::test]
    async fn cumulative_backlog_coalesces_to_latest() {
        let (tx, rx) = mpsc::channel(8);
        let calls = vec![timed_call(10, rx)];
        let stream = generation_event_stream(calls, false, false);
        futures::pin_mut!(stream);

        // Three chunks queued before the stream is ever polled (a client falling behind).
        tx.send(frame(10, "a")).await.unwrap();
        tx.send(frame(10, "b")).await.unwrap();
        tx.send(frame(10, "c")).await.unwrap();

        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "abc", "one frame, full cumulative text");
        assert_eq!(v["meta_info"]["completion_tokens"], 3, "no tokens lost");

        // The terminal frame still carries everything, and only then does [DONE] land.
        tx.send(done(10, "!")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "abc!");
        assert_eq!(v["meta_info"]["finish_reason"]["type"], "length");
        assert_eq!(stream.next().await.unwrap(), "[DONE]");
    }

    /// Incremental frames are *deltas*, so a backlog must emit every one — dropping
    /// any would silently lose tokens. Only the cumulative protocol may coalesce.
    #[tokio::test]
    async fn incremental_backlog_emits_every_delta() {
        let (tx, rx) = mpsc::channel(8);
        let calls = vec![timed_call(10, rx)];
        let stream = generation_event_stream(calls, true, false);
        futures::pin_mut!(stream);

        tx.send(frame(10, "a")).await.unwrap();
        tx.send(frame(10, "b")).await.unwrap();
        tx.send(frame(10, "c")).await.unwrap();

        for (n, expect) in [(1, "a"), (2, "b"), (3, "c")] {
            let v = parse(&stream.next().await.unwrap());
            assert_eq!(v["text"], expect, "delta {n} must not be dropped");
            assert_eq!(
                v["meta_info"]["completion_tokens"], n,
                "count stays cumulative"
            );
        }
    }
}
