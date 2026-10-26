//! The native SGLang data-plane endpoints: `/generate` (submit a request, then
//! either fold decode frames to one unary JSON response or relay them as SSE
//! `data: {json}` … `[DONE]`, byte-compatible with Python
//! `http_server.generate_request`) and `/health` + `/health_generate` (which
//! round-trip a 1-token generate probe). Frame shaping (`meta_info`, logprob
//! tuples, cumulative vs incremental streams) lives here; submission and
//! request lifetime belong to the shared [`crate::api_server::core::CoreHandle`].

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::app::AppState;
use super::generate_hook::{GenerateFrameHook, GenerateFrameInfo, GenerateFrameKeys};
use super::request_headers::apply_header_overrides;
use crate::api_server::core::{
    CoreCall, CoreError, CoreEvent, CoreOutput, HealthStatus, recv_indexed,
};
#[cfg(test)]
use crate::message::ids::Rid;
#[cfg(test)]
use crate::message::request::GenerationPolicy;
use crate::message::request::{GenerateRequest, into_requests};
use crate::message::wire::{ClientMaxNewTokens, merge_preferred_sampling};
use crate::utils::{
    environ,
    response::{error_response, error_value},
};
use axum::{
    Json, Router,
    extract::State,
    extract::rejection::JsonRejection,
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use sglang_api_types::api::v1 as api;

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

    /// First output to terminal output, once the request finished.
    fn decode_latency(&self) -> Option<Duration> {
        let (time_to_first_token, e2e_latency) = self.terminal_latencies()?;
        Some(e2e_latency.saturating_sub(time_to_first_token))
    }
}

/// API-local state for one request: its timing, and the annotator a model
/// package's [`GenerateHook`](super::generate_hook::GenerateHook) started for
/// it.
struct RequestState {
    timing: RequestTiming,
    /// The scheduler's decode latency, sent for a finished non-streaming
    /// request: this frontend sees that request's first output only up to
    /// `SGLANG_FORCE_STREAM_INTERVAL` tokens late, not when its first token
    /// was sampled.
    scheduler_decode_latency: Option<Duration>,
    hook: Option<Box<dyn GenerateFrameHook>>,
}

impl RequestState {
    fn new(state: &AppState, request: &GenerateRequest, timing: RequestTiming) -> Self {
        Self {
            timing,
            scheduler_decode_latency: None,
            hook: state
                .generate_hook
                .as_ref()
                .map(|hook| hook.start(&request.sampling_params)),
        }
    }

    fn finish(&mut self, terminal: &CoreOutput) {
        self.timing.observe_first_output();
        self.timing.finish();
        self.scheduler_decode_latency = terminal.scheduler_decode_latency();
    }

    /// From the first token to the terminal output, as Python's
    /// TokenizerManager measures it: the scheduler's measure when it sent one,
    /// else this frontend's first and terminal outputs.
    fn decode_latency(&self) -> Option<Duration> {
        self.scheduler_decode_latency
            .or_else(|| self.timing.decode_latency())
    }

    /// One frame as the client receives it: the schema's, plus the hook's keys.
    fn frame(&mut self, frame: api::GenerateResponse) -> NativeFrame {
        let finished = frame
            .meta_info
            .as_ref()
            .is_some_and(|meta| meta.finish_reason.is_some());
        let info = GenerateFrameInfo {
            finished,
            decode_latency: if finished {
                self.decode_latency()
            } else {
                None
            },
        };
        let Some(hook) = self.hook.as_mut() else {
            return NativeFrame { frame, keys: None };
        };
        let mut keys = GenerateFrameKeys::default();
        hook.annotate(&info, &mut keys);
        NativeFrame {
            frame,
            keys: Some(Box::new(keys)),
        }
    }
}

/// One native `/generate` frame: the `api.v1` frame, plus the keys a model
/// package's hook added after the schema's own.
#[derive(Debug)]
struct NativeFrame {
    frame: api::GenerateResponse,
    keys: Option<Box<GenerateFrameKeys>>,
}

impl serde::Serialize for NativeFrame {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap; // codespell:ignore ser

        /// A schema object's keys, then the added ones.
        #[derive(serde::Serialize)]
        struct WithKeys<'a, T> {
            #[serde(flatten)]
            schema: &'a T,
            #[serde(flatten)]
            added: &'a serde_json::Map<String, serde_json::Value>,
        }

        let Some(keys) = self.keys.as_deref() else {
            return self.frame.serialize(serializer);
        };
        // `GenerateResponse`'s generated serializer, key for key, with the
        // added keys in `meta_info` and after the frame's own. Destructured
        // whole so a field the schema gains fails to compile here until it is
        // written.
        let api::GenerateResponse {
            text,
            meta_info,
            output_ids,
            index,
            prompt_token_ids,
        } = &self.frame;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("text", text)?;
        if let Some(meta_info) = meta_info {
            let meta_info = WithKeys {
                schema: meta_info,
                added: &keys.meta_info,
            };
            map.serialize_entry("meta_info", &meta_info)?;
        }
        if let Some(output_ids) = output_ids {
            map.serialize_entry("output_ids", output_ids)?;
        }
        if let Some(index) = index {
            map.serialize_entry("index", index)?;
        }
        if let Some(prompt_token_ids) = prompt_token_ids {
            map.serialize_entry("prompt_token_ids", prompt_token_ids)?;
        }
        for (key, value) in &keys.frame {
            map.serialize_entry(key, value)?;
        }
        map.end()
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

fn native_core_error(error: CoreError, stream: bool) -> Response {
    native_error(error.http_status(), &error.to_string(), stream)
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
    let health = if environ::env_bool("SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION", true) {
        get(move |state: State<Arc<AppState>>| health_generate(state, timeout, false))
    } else {
        get(health_without_generation)
    };
    let probe = get(move |state: State<Arc<AppState>>| health_generate(state, timeout, true));
    Router::new()
        .route("/health", health)
        .route("/health_generate", probe)
}

async fn health_without_generation(State(state): State<Arc<AppState>>) -> Response {
    if state.core.is_ready() {
        StatusCode::OK.into_response()
    } else {
        StatusCode::SERVICE_UNAVAILABLE.into_response()
    }
}

/// `GET /health_generate` — deep health: confirm the scheduler → detok path is
/// producing output. 200 if the response heartbeat advances within `timeout`
/// (from `SGLANG_HEALTH_CHECK_TIMEOUT`, frozen at router build), else 503.
/// It also returns 503 until startup warmup completes. (`/health` uses the same
/// handler when its env gate is on, without `metadata_headers`.) A healthy
/// `/health_generate` carries the configured `health_generate_headers`.
///
/// Fires a pre-tokenized 1-token probe (`input_ids = [0]`, skips the tokenizer) so
/// an idle pipeline produces a frame, then watches the *global*
/// core response-activity counter (not the probe's own rid) — so a busy
/// server passes immediately and a backlog never false-503s (the analogue of
/// Python's `last_receive_tstamp`).
async fn health_generate(
    State(state): State<Arc<AppState>>,
    timeout: std::time::Duration,
    metadata_headers: bool,
) -> Response {
    match state.core.probe_health(timeout).await {
        Ok(HealthStatus::Healthy) => {
            let mut response = StatusCode::OK.into_response();
            if metadata_headers {
                // Validated at startup (`ServerArgs::validate`).
                for (name, value) in &state.server_args.health_generate_headers {
                    if let (Ok(name), Ok(value)) = (
                        HeaderName::from_bytes(name.as_bytes()),
                        HeaderValue::from_str(value),
                    ) {
                        response.headers_mut().insert(name, value);
                    }
                }
            }
            response
        }
        Ok(HealthStatus::NotReady | HealthStatus::Stalled) => {
            StatusCode::SERVICE_UNAVAILABLE.into_response()
        }
        Err(CoreError::Unavailable) => native_error(
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
    headers: HeaderMap,
    body: Result<Json<serde_json::Value>, JsonRejection>,
) -> Response {
    let mut body = match body {
        Ok(Json(body)) => body,
        // A body that fails to parse has no readable `stream` flag, so this one
        // can only answer unary -- as Python's does (FastAPI rejects before its
        // handler runs).
        Err(rejection) => {
            return native_error(StatusCode::BAD_REQUEST, &rejection.body_text(), false);
        }
    };
    if state.server_args.enable_request_header_overrides
        && let Err(error) = apply_header_overrides(&mut body, &headers)
    {
        return native_error(StatusCode::BAD_REQUEST, &error, false);
    }
    // Read before the fills below: only a `max_new_tokens` the client sent is
    // checked against the context window.
    let client_max_new_tokens = ClientMaxNewTokens::of_json(&body);
    // Launch-time preferred sampling params fill in beneath the request's own
    // keys before decoding, so an explicit request value (even a default or
    // null) wins: Python TokenizerManager's precedence.
    if let Some(preferred) = &state.server_args.preferred_sampling_params
        && let Err(error) = merge_preferred_sampling(&mut body, &preferred.0)
    {
        return native_error(StatusCode::INTERNAL_SERVER_ERROR, &error, false);
    }
    // Beneath both, the launch-time default for an omitted `max_new_tokens`.
    let default_max_new_tokens = serde_json::json!({
        "max_new_tokens": state.server_args.default_max_new_tokens,
    });
    if let Err(error) = merge_preferred_sampling(&mut body, &default_max_new_tokens) {
        return native_error(StatusCode::INTERNAL_SERVER_ERROR, &error, false);
    }
    // The schema (sglang_api_types) is the contract: decode errors carry
    // serde's field-level text, as Python's 400s do.
    let body: api::GenerateRequest = match serde_json::from_value(body) {
        Ok(body) => body,
        Err(error) => return native_error(StatusCode::BAD_REQUEST, &error.to_string(), false),
    };
    let stream = body.stream.unwrap_or(false);
    // Fan `text`/`input_ids`/`sampling_params` (scalar or list) into per-request
    // payloads. `is_batch` = list form -> the response is a JSON array.
    let (mut payloads, is_batch) = match into_requests(body, &state.generation_policy) {
        Ok(v) => v,
        // Request normalization only returns validation failures.
        Err(e) => {
            return native_error(StatusCode::BAD_REQUEST, &e.to_string(), stream);
        }
    };
    client_max_new_tokens.mark(&mut payloads);
    // Python starts APIServerReqTimeStats after request normalization and before
    // tokenization / multimodal preprocessing / scheduler dispatch. Start at the
    // equivalent boundary: into_requests() has normalized the body, while prefetch
    // and every downstream stage are still ahead of us.
    let timing = RequestTiming::new();
    // Shared core admission performs media I/O (URL downloads, file reads)
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
    let request = RequestState::new(state, &req, timing);
    // `return_text_in_logprobs` is decoded on the detok shard into `*_txt`, so
    // the frame builder just reads them — no tokenizer needed here.
    let mut call = match state.core.generate(req).await {
        Ok(call) => call,
        Err(error) => return native_core_error(error, stream),
    };
    let rid = call.public_id().to_owned();
    // Cumulative frames (SGLang default) vs per-step deltas.
    let incremental = state.server_args.incremental_streaming_output;

    if stream {
        // A single request is a 1-element batch without the `index` field — reuse
        // the same stream so the frame/abort/truncation logic lives in one place.
        use futures::StreamExt;
        let s = generation_event_stream(vec![(call, request)], incremental, false)
            .map(|data| Ok::<_, Infallible>(Event::default().data(data)));
        Sse::new(s).into_response()
    } else {
        // Unary: fold to the terminal and respond once. `CoreCall` disarms
        // itself on a terminal item; truncation leaves it armed for drop cleanup.
        let (status, item) = drain_unary(&mut call, &rid, request).await;
        (status, Json(item)).into_response()
    }
}

/// One unary `/generate` result: the frame, or the item's own error body (a
/// failed batch item is its own `{ "error": ... }` entry in the array).
#[derive(Debug, serde::Serialize)]
#[serde(untagged)]
enum GenerateItem {
    // Boxed: a frame is ~300 bytes against the error's ~40, and a batch
    // array holds one per item.
    Frame(Box<NativeFrame>),
    Error(api::GenerateStreamError),
}

/// Fold a unary request to its terminal HTTP result. Shared by single + batch.
async fn drain_unary(
    call: &mut CoreCall,
    rid_str: &str,
    mut request: RequestState,
) -> (StatusCode, GenerateItem) {
    let mut accumulated = CoreOutput::default();
    while let Some(item) = call.recv().await {
        match item {
            CoreEvent::Delta(out) => {
                request.timing.observe_first_output();
                accumulated.append_delta(&out);
            }
            CoreEvent::Finished(out) => {
                request.finish(&out);
                accumulated.append_delta(&out);
                let e2e_latency = Some(e2e_latency(&request.timing));
                let frame = request.frame(accumulated.frame(rid_str, None, e2e_latency));
                return (StatusCode::OK, GenerateItem::Frame(Box::new(frame)));
            }
            CoreEvent::Failed(error) => {
                request.timing.finish();
                return (
                    error.http_status(),
                    GenerateItem::Error(error.stream_error(None)),
                );
            }
        }
    }
    // Defensive fallback: CoreCall normally turns premature runtime
    // termination into a terminal Failed event before exposing stream end.
    let truncated = CoreError::ResponseTruncated;
    (
        truncated.http_status(),
        GenerateItem::Error(truncated.stream_error(None)),
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
    let request_states: Vec<RequestState> = requests
        .iter()
        .map(|request| RequestState::new(state, request, timing.clone()))
        .collect();
    // No cross-item rid collision to worry about: `into_requests` rejected duplicate
    // rids within this batch, and `Rid::from_client` made each one unique against
    // every other in-flight request.
    let calls = match state.core.generate_batch(requests).await {
        Ok(calls) => calls.into_iter().zip(request_states).collect(),
        Err(error) => return native_core_error(error, stream),
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
        let drained =
            futures::future::join_all(calls.into_iter().map(|(mut call, request)| async move {
                let client_rid = call.public_id().to_owned();
                let (_status, item) = drain_unary(&mut call, &client_rid, request).await;
                item
            }))
            .await;
        (StatusCode::OK, Json(drained)).into_response()
    }
}

/// Multiplex calls into SSE `data` strings + a final `[DONE]`; `with_index` tags
/// each frame (batch only), `incremental` = delta vs cumulative. Each call
/// aborts itself if the stream is dropped while it is unfinished.
fn generation_event_stream(
    calls: Vec<(CoreCall, RequestState)>,
    incremental: bool,
    with_index: bool,
) -> impl futures::Stream<Item = String> {
    async_stream::stream! {
        use futures::StreamExt;

        let n = calls.len();
        let rid_strs: Vec<String> = calls
            .iter()
            .map(|(call, _)| call.public_id().to_owned())
            .collect();
        let mut requests: Vec<RequestState> = Vec::with_capacity(n);
        let mut accumulated: Vec<CoreOutput> = (0..n).map(|_| CoreOutput::default()).collect();

        // Batch position, tagged onto every frame (a single request omits it).
        let idx = |i: usize| with_index.then_some(i as u32);

        // Poll all receivers concurrently; re-arm a receiver's future after each
        // non-terminal frame so its stream keeps flowing.
        let mut futs = futures::stream::FuturesUnordered::new();
        for (i, (call, request)) in calls.into_iter().enumerate() {
            requests.push(request);
            futs.push(recv_indexed(i, call));
        }

        while let Some((i, call, items)) = futs.next().await {
            if items.is_empty() {
                // Defensive fallback: a premature runtime close is normally a
                // Failed event produced by CoreCall. Keep this item's call
                // armed if its semantic stream somehow ends unexpectedly.
                yield json_string(&CoreError::ResponseTruncated.stream_error(idx(i)));
                continue;
            }

            // Cumulative frames supersede one another, so a drained backlog collapses
            // to its last (Python's `out_list[-1]`); deltas can't be dropped.
            let mut coalesced = false; // a cumulative frame is pending
            let mut terminal = None;   // (finish_reason) of a `Done` in this batch
            let mut failed = None;     // an `Error` in this batch

            for item in items {
                match item {
                    CoreEvent::Delta(out) => {
                        requests[i].timing.observe_first_output();
                        accumulated[i].append_delta(&out);
                        if incremental {
                            let frame = delta_frame(out, &accumulated[i], &rid_strs[i], idx(i), None);
                            yield json_string(&requests[i].frame(frame));
                        } else {
                            coalesced = true;
                        }
                    }
                    CoreEvent::Finished(out) => {
                        requests[i].finish(&out);
                        accumulated[i].append_delta(&out);
                        terminal = Some(out);
                    }
                    CoreEvent::Failed(error) => {
                        requests[i].timing.finish();
                        failed = Some(error);
                    }
                }
            }

            if let Some(error) = failed {
                yield json_string(&error.stream_error(idx(i)));
            } else if let Some(out) = terminal {
                // The final frame carries the full cumulative state, so any
                // coalesced non-terminal frames are moot.
                let e2e_latency = Some(e2e_latency(&requests[i].timing));
                // The terminal frame is the accumulated output's last use, so
                // the cumulative form moves it out instead of cloning.
                let frame = if incremental {
                    delta_frame(out, &accumulated[i], &rid_strs[i], idx(i), e2e_latency)
                } else {
                    std::mem::take(&mut accumulated[i]).frame(&rid_strs[i], idx(i), e2e_latency)
                };
                yield json_string(&requests[i].frame(frame));
            } else {
                if coalesced {
                    // An intermediate cumulative frame has to clone: the accumulated
                    // output stays behind for the next one.
                    let frame = accumulated[i].clone().frame(&rid_strs[i], idx(i), None);
                    yield json_string(&requests[i].frame(frame));
                }
                futs.push(recv_indexed(i, call)); // keep this item flowing
            }
        }
        yield "[DONE]".to_string();
    }
}

/// Python's `e2e_latency` is `finished_time - created_time`, in seconds, and is
/// attached only when the request finishes. The Rust native API owns the same
/// lifecycle boundary, so it sets the value while handling the terminal egress
/// item rather than putting API-only timing onto every shared core output.
fn e2e_latency(timing: &RequestTiming) -> f64 {
    let (time_to_first_token, e2e_latency) = timing
        .terminal_latencies()
        .expect("a successful terminal output has complete request timing");
    debug_assert!(time_to_first_token <= e2e_latency);
    e2e_latency.as_secs_f64()
}

/// One incremental streaming frame: this step's delta with the cumulative
/// token count in `meta_info` (matching Python). Only the terminal frame
/// carries `e2e_latency`.
fn delta_frame(
    mut delta: CoreOutput,
    accumulated: &CoreOutput,
    rid_str: &str,
    index: Option<u32>,
    e2e_latency: Option<f64>,
) -> api::GenerateResponse {
    delta.completion_tokens = accumulated.completion_tokens;
    delta.frame(rid_str, index, e2e_latency)
}

/// The SSE `data` text of one `api.v1` item (a frame or an error).
fn json_string<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("api.v1 types serialize to JSON")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::request::RequestKind;
    use crate::message::response::{ChunkEvent, ResponseItem};
    use crate::tokenizer_manager::wiring::TmEvent;
    use crate::utils::error::Error;
    use axum::body::Body;
    use axum::http::Request;
    use futures::StreamExt;
    use std::time::Duration;
    use tokio::sync::mpsc;
    use tower::ServiceExt;

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

    fn timed_call(rid: u64, rx: mpsc::Receiver<ResponseItem>) -> (CoreCall, RequestState) {
        timed_call_with_abort(rid, rx, flume::unbounded().0)
    }

    fn timed_call_with_abort(
        rid: u64,
        rx: mpsc::Receiver<ResponseItem>,
        abort_tx: flume::Sender<crate::tokenizer_manager::wiring::AbortSource>,
    ) -> (CoreCall, RequestState) {
        (
            CoreCall::from_test_generation_parts(Rid::from(rid.to_string()), rx, abort_tx),
            RequestState {
                timing: RequestTiming {
                    created_at: Instant::now() - Duration::from_millis(10),
                    time_to_first_token: None,
                    e2e_latency: None,
                },
                scheduler_decode_latency: None,
                hook: None,
            },
        )
    }

    #[tokio::test]
    async fn health_is_unavailable_before_startup_warmup_finishes() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        let state = Arc::new(AppState {
            core: crate::api_server::core::CoreHandle::new(
                intake_tx,
                abort_tx,
                crate::api_server::core::CoreConfig {
                    response_capacity: 8,
                    response_activity: Default::default(),
                    startup_ready: false,
                    is_disaggregation: false,
                    mm_limits: Default::default(),
                    metadata: crate::api_server::core::CoreMetadata::default(),
                    metrics: None,
                },
            ),
            server_args: Arc::new(crate::message::config::ServerArgs::default()),
            generation_policy: GenerationPolicy::default(),
            chat_formatter: None,
            generate_hook: None,
        });

        let response = health_generate(State(state), Duration::ZERO, true).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(intake_rx.try_recv().is_err());
        assert!(abort_rx.try_recv().is_err());
    }

    /// A healthy `/health_generate` carries the configured metadata headers;
    /// `/health` sharing the probe does not.
    #[tokio::test]
    async fn healthy_probe_carries_configured_headers() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let activity: crate::tokenizer_manager::from_scheduler::ActivityCounter =
            Default::default();
        let server_args = crate::message::config::ServerArgs {
            health_generate_headers: [
                ("x-custom-header".to_string(), "v".to_string()),
                ("x-rust-server".to_string(), "health".to_string()),
            ]
            .into(),
            ..Default::default()
        };
        let state = Arc::new(AppState {
            core: crate::api_server::core::CoreHandle::new(
                intake_tx,
                abort_tx,
                crate::api_server::core::CoreConfig {
                    response_capacity: 8,
                    response_activity: activity.clone(),
                    startup_ready: true,
                    is_disaggregation: false,
                    mm_limits: Default::default(),
                    metadata: crate::api_server::core::CoreMetadata::default(),
                    metrics: None,
                },
            ),
            generation_policy: crate::message::request::GenerationPolicy::from(&server_args),
            server_args: Arc::new(server_args),
            chat_formatter: None,
            generate_hook: None,
        });
        // Each probe that reaches intake counts as scheduler output.
        tokio::spawn(async move {
            while intake_rx.recv_async().await.is_ok() {
                activity.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        });

        let response = health_generate(State(state.clone()), Duration::from_secs(5), true).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-custom-header"], "v");
        assert_eq!(response.headers()["x-rust-server"], "health");

        let response = health_generate(State(state), Duration::from_secs(5), false).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers().get("x-custom-header").is_none());
    }

    #[tokio::test]
    async fn unary_http_generate_round_trips_through_frontend_handle() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        let core = crate::api_server::core::CoreHandle::new(
            intake_tx,
            abort_tx,
            crate::api_server::core::CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Default::default(),
                metadata: crate::api_server::core::CoreMetadata::default(),
                metrics: None,
            },
        );
        let state = Arc::new(AppState {
            core,
            server_args: Arc::new(crate::message::config::ServerArgs::default()),
            generation_policy: GenerationPolicy::default(),
            chat_formatter: None,
            generate_hook: None,
        });

        let responder = tokio::spawn(async move {
            let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap()
            else {
                panic!("HTTP generation must enter through core intake");
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

    /// Each item `body` submits under `server_args`: its `max_new_tokens`, and
    /// whether that value is marked defaulted.
    async fn submitted_max_new_tokens(
        server_args: crate::message::config::ServerArgs,
        body: serde_json::Value,
        items: usize,
    ) -> Vec<(Option<i64>, bool)> {
        let (intake_tx, intake_rx) = flume::unbounded();
        let core = crate::api_server::core::CoreHandle::new(
            intake_tx,
            flume::unbounded().0,
            crate::api_server::core::CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Default::default(),
                metadata: crate::api_server::core::CoreMetadata::default(),
                metrics: None,
            },
        );
        let state = Arc::new(AppState {
            core,
            generation_policy: GenerationPolicy::from(&server_args),
            server_args: Arc::new(server_args),
            chat_formatter: None,
            generate_hook: None,
        });
        let responder = tokio::spawn(async move {
            let mut submitted = Vec::new();
            for _ in 0..items {
                let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap()
                else {
                    panic!("expected an intake");
                };
                assert!(admission.try_accept());
                let RequestKind::Generate(generate) = &request.kind else {
                    panic!("/generate must submit a generation request");
                };
                submitted.push((
                    generate.sampling_params.max_new_tokens,
                    generate.max_new_tokens_defaulted,
                ));
                request.sink.try_send(done(0, "ok")).unwrap();
            }
            submitted
        });
        let request = Request::builder()
            .method("POST")
            .uri("/generate")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = routes().with_state(state).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        responder.await.unwrap()
    }

    /// An item whose `sampling_params` name no `max_new_tokens` is marked
    /// defaulted and keeps the value the fills gave it (the preferred one, else
    /// the launch default); one that names it, even as null, keeps the client's. A
    /// list is marked object by object, and one object marks every item.
    #[tokio::test]
    async fn generate_marks_each_max_new_tokens_the_client_omits() {
        use crate::message::config::{PreferredSamplingParams, ServerArgs};
        let args = |preferred: Option<serde_json::Value>| ServerArgs {
            preferred_sampling_params: preferred.map(PreferredSamplingParams),
            ..Default::default()
        };
        let preferred = Some(serde_json::json!({"temperature": 1.0, "max_new_tokens": 4096}));
        let cases = [
            (
                args(preferred.clone()),
                serde_json::json!({"input_ids": [1]}),
                vec![(Some(4096), true)],
            ),
            (
                args(preferred.clone()),
                serde_json::json!({"input_ids": [1], "sampling_params": {"top_p": 0.5}}),
                vec![(Some(4096), true)],
            ),
            (
                args(None),
                serde_json::json!({"input_ids": [1]}),
                vec![(Some(128), true)],
            ),
            (
                args(preferred.clone()),
                serde_json::json!({"input_ids": [1], "sampling_params": {"max_new_tokens": 8}}),
                vec![(Some(8), false)],
            ),
            (
                args(preferred.clone()),
                serde_json::json!({"input_ids": [1], "sampling_params": {"max_new_tokens": null}}),
                vec![(None, false)],
            ),
            (
                args(preferred.clone()),
                serde_json::json!({
                    "input_ids": [[1], [2]],
                    "sampling_params": [{}, {"max_new_tokens": 8}],
                }),
                vec![(Some(4096), true), (Some(8), false)],
            ),
            (
                args(preferred),
                serde_json::json!({
                    "input_ids": [[1], [2]],
                    "sampling_params": {"max_new_tokens": 8},
                }),
                vec![(Some(8), false), (Some(8), false)],
            ),
        ];
        for (server_args, body, expected) in cases {
            let submitted =
                submitted_max_new_tokens(server_args, body.clone(), expected.len()).await;
            assert_eq!(submitted, expected, "{body}");
        }
    }

    /// An omitted `max_new_tokens` takes the launch default, beneath the
    /// request's own value and the preferred sampling params; a `None`
    /// default is no limit. Whichever fill supplies it, it is marked defaulted.
    #[tokio::test]
    async fn omitted_max_new_tokens_takes_the_launch_default() {
        use crate::message::config::{PreferredSamplingParams, ServerArgs};
        let args = |default_max_new_tokens, preferred: Option<serde_json::Value>| ServerArgs {
            default_max_new_tokens,
            preferred_sampling_params: preferred.map(PreferredSamplingParams),
            ..Default::default()
        };
        let omitted = serde_json::json!({"input_ids": [1, 2]});
        let explicit = serde_json::json!({
            "input_ids": [1, 2],
            "sampling_params": {"max_new_tokens": 3},
        });
        let preferred = Some(serde_json::json!({"max_new_tokens": 7}));
        for (server_args, body, expected) in [
            (args(Some(4096), None), &omitted, (Some(4096), true)),
            (args(None, None), &omitted, (None, true)),
            (args(Some(4096), preferred), &omitted, (Some(7), true)),
            (args(Some(4096), None), &explicit, (Some(3), false)),
        ] {
            assert_eq!(
                submitted_max_new_tokens(server_args, body.clone(), 1).await,
                vec![expected],
                "{body}"
            );
        }
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

        assert!(timing.decode_latency().is_none(), "unfinished");
        timing.finish();
        let (recorded_ttft, e2e_latency) = timing.terminal_latencies().unwrap();
        assert_eq!(recorded_ttft, time_to_first_token);
        assert!(e2e_latency >= recorded_ttft);
        assert_eq!(timing.decode_latency(), Some(e2e_latency - recorded_ttft));

        timing.finish();
        assert_eq!(
            timing.terminal_latencies(),
            Some((recorded_ttft, e2e_latency)),
            "later terminal handling must not overwrite E2E latency"
        );
    }

    /// A model package's hook: the request's `top_p` on every frame, a
    /// top-level key on its first frame, and the decode latency on its last.
    #[derive(Debug)]
    struct TopPHook;

    struct TopPFrames {
        top_p: f64,
        first: bool,
    }

    impl crate::GenerateHook for TopPHook {
        fn start(&self, sampling_params: &crate::SamplingParams) -> Box<dyn GenerateFrameHook> {
            Box::new(TopPFrames {
                top_p: sampling_params.top_p,
                first: true,
            })
        }
    }

    impl GenerateFrameHook for TopPFrames {
        fn annotate(&mut self, frame: &GenerateFrameInfo, keys: &mut GenerateFrameKeys) {
            assert_eq!(frame.finished, frame.decode_latency.is_some());
            keys.meta_info
                .insert("top_p".into(), serde_json::json!(self.top_p));
            if let Some(latency) = frame.decode_latency {
                keys.meta_info
                    .insert("decode_s".into(), serde_json::json!(latency.as_secs_f64()));
            }
            if std::mem::take(&mut self.first) {
                keys.frame.insert("first".into(), serde_json::json!(true));
            }
        }
    }

    /// A hook's keys reach every native frame of every batch item — unary,
    /// cumulative and incremental streams — each item's hook started from its
    /// own sampling params, with the scheduler's decode latency for a unary
    /// item and this frontend's for a streamed one; without a hook the frames
    /// are the schema's.
    #[tokio::test]
    async fn generate_hook_keys_reach_every_frame() {
        for (stream, incremental, hook) in [
            (false, false, true),
            (true, false, true),
            (true, true, true),
            (true, true, false),
        ] {
            let (intake_tx, intake_rx) = flume::unbounded();
            let core = crate::api_server::core::CoreHandle::new(
                intake_tx,
                flume::unbounded().0,
                crate::api_server::core::CoreConfig {
                    response_capacity: 8,
                    response_activity: Default::default(),
                    startup_ready: true,
                    is_disaggregation: false,
                    mm_limits: Default::default(),
                    metadata: crate::api_server::core::CoreMetadata::default(),
                    metrics: None,
                },
            );
            let state = Arc::new(AppState {
                core,
                server_args: Arc::new(crate::message::config::ServerArgs {
                    incremental_streaming_output: incremental,
                    ..Default::default()
                }),
                generation_policy: GenerationPolicy::default(),
                chat_formatter: None,
                generate_hook: hook.then(|| Arc::new(TopPHook) as Arc<dyn crate::GenerateHook>),
            });
            // Both items' delta, then both terminals: an incremental stream
            // emits two frames per item.
            let responder = tokio::spawn(async move {
                let mut sinks = Vec::new();
                for _ in 0..2 {
                    let TmEvent::Intake { request, admission } =
                        intake_rx.recv_async().await.unwrap()
                    else {
                        panic!("expected a generation intake");
                    };
                    assert!(admission.try_accept());
                    sinks.push((request.rid.clone(), request.sink.clone()));
                }
                for (rid, sink) in &sinks {
                    let ResponseItem::Frame(mut chunk) = frame(0, "a") else {
                        unreachable!()
                    };
                    chunk.rid = rid.clone();
                    sink.try_send(ResponseItem::Frame(chunk)).unwrap();
                }
                for (rid, sink) in &sinks {
                    let ResponseItem::Done(mut chunk) = done(0, "b") else {
                        unreachable!()
                    };
                    chunk.rid = rid.clone();
                    // The scheduler times non-streaming requests only.
                    chunk.extras = (!stream).then(|| {
                        Box::new(crate::message::response::ChunkExtras {
                            decode_latency: Some(0.25),
                            ..Default::default()
                        })
                    });
                    sink.try_send(ResponseItem::Done(chunk)).unwrap();
                }
            });
            let body = serde_json::json!({
                "text": ["x", "y"],
                "stream": stream,
                "sampling_params": [{"top_p": 0.5}, {"top_p": 0.25}],
            });
            let request = Request::builder()
                .method("POST")
                .uri("/generate")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap();
            let response = routes().with_state(state).oneshot(request).await.unwrap();
            let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
                .await
                .unwrap();
            responder.await.unwrap();
            let body = std::str::from_utf8(&body).unwrap();
            let frames: Vec<serde_json::Value> = if stream {
                body.lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter(|data| *data != "[DONE]")
                    .map(parse)
                    .collect()
            } else {
                parse(body).as_array().unwrap().clone()
            };

            let case = format!("stream={stream} incremental={incremental} hook={hook}");
            assert_eq!(frames.len(), if incremental { 4 } else { 2 }, "{case}");
            let mut seen = [0usize; 2];
            for (position, frame) in frames.iter().enumerate() {
                // A unary batch is an array in request order, without `index`.
                let index = frame["index"].as_u64().map_or(position, |i| i as usize);
                let meta = &frame["meta_info"];
                let terminal = !meta["finish_reason"].is_null();
                if !hook {
                    assert!(meta.get("top_p").is_none() && frame.get("first").is_none());
                    continue;
                }
                assert_eq!(meta["top_p"], [0.5, 0.25][index], "{case}");
                assert_eq!(frame.get("first").is_some(), seen[index] == 0, "{case}");
                assert_eq!(meta.get("decode_s").is_some(), terminal, "{case}");
                if terminal && !stream {
                    assert_eq!(meta["decode_s"], 0.25, "{case}");
                }
                seen[index] += 1;
            }
        }
    }

    /// The hook-aware serializer writes the generated one's keys, in its
    /// order, before the hook's.
    #[test]
    fn native_frame_keeps_the_schema_keys_first() {
        let schema = api::GenerateResponse {
            text: "ok".into(),
            meta_info: Some(api::GenerateMetaInfo {
                id: "r".into(),
                prompt_tokens: 2,
                ..Default::default()
            }),
            output_ids: Some(api::TokenIds { ids: vec![7] }),
            index: Some(1),
            prompt_token_ids: None,
        };
        let expected = json_string(&schema);
        let empty = NativeFrame {
            frame: schema.clone(),
            keys: Some(Box::default()),
        };
        assert_eq!(json_string(&empty), expected);

        let mut keys = GenerateFrameKeys::default();
        keys.meta_info.insert("m".into(), serde_json::json!(1));
        keys.frame.insert("f".into(), serde_json::json!({"k": "v"}));
        let added = NativeFrame {
            frame: schema,
            keys: Some(Box::new(keys)),
        };
        let mut with_meta: serde_json::Value = parse(&expected);
        with_meta["meta_info"]["m"] = serde_json::json!(1);
        with_meta["f"] = serde_json::json!({"k": "v"});
        assert_eq!(json_string(&added), with_meta.to_string());
    }

    /// A recording hook's annotator: every frame it was handed, in order, each
    /// frame marked with its position as `meta_info.hook_frame`.
    #[derive(Clone, Default)]
    struct Recorded(Arc<std::sync::Mutex<Vec<GenerateFrameInfo>>>);

    impl Recorded {
        fn hook(&self) -> Option<Box<dyn GenerateFrameHook>> {
            Some(Box::new(self.clone()))
        }

        fn frames(&self) -> Vec<GenerateFrameInfo> {
            self.0.lock().unwrap().clone()
        }
    }

    impl GenerateFrameHook for Recorded {
        fn annotate(&mut self, frame: &GenerateFrameInfo, keys: &mut GenerateFrameKeys) {
            let mut frames = self.0.lock().unwrap();
            keys.meta_info
                .insert("hook_frame".into(), serde_json::json!(frames.len()));
            frames.push(*frame);
        }
    }

    /// A cumulative stream calls the hook once per frame it emits, in order:
    /// each intermediate frame (one for a coalesced backlog) unfinished and
    /// without a decode latency, then the terminal one with this frontend's
    /// decode latency (a streamed request carries no scheduler measure). Each
    /// frame carries its own call's keys beside its cumulative text.
    #[tokio::test]
    async fn generate_hook_annotates_intermediate_cumulative_frames() {
        let recorded = Recorded::default();
        let (tx, rx) = mpsc::channel(8);
        let (call, mut request) = timed_call(10, rx);
        request.hook = recorded.hook();
        let stream = generation_event_stream(vec![(call, request)], false, false);
        futures::pin_mut!(stream);

        tx.send(frame(10, "a")).await.unwrap();
        let first = parse(&stream.next().await.unwrap());
        tx.send(frame(10, "b")).await.unwrap();
        tx.send(frame(10, "c")).await.unwrap();
        let backlog = parse(&stream.next().await.unwrap());
        tx.send(done(10, "!")).await.unwrap();
        let terminal = parse(&stream.next().await.unwrap());
        assert_eq!(stream.next().await.unwrap(), "[DONE]");

        for (position, (frame, text)) in [(&first, "a"), (&backlog, "abc"), (&terminal, "abc!")]
            .into_iter()
            .enumerate()
        {
            assert_eq!(frame["text"], text);
            assert_eq!(frame["meta_info"]["hook_frame"], position, "{text}");
        }
        let frames = recorded.frames();
        let finished: Vec<bool> = frames.iter().map(|frame| frame.finished).collect();
        assert_eq!(finished, [false, false, true]);
        assert!(
            frames[..2]
                .iter()
                .all(|frame| frame.decode_latency.is_none())
        );
        let decode_latency = frames[2].decode_latency.expect("terminal").as_secs_f64();
        let e2e_latency = terminal["meta_info"]["e2e_latency"].as_f64().unwrap();
        assert!(
            decode_latency <= e2e_latency,
            "{decode_latency} > {e2e_latency}"
        );
    }

    /// The scheduler's decode latency, as Python's `push_generation` frames it
    /// (`testdata/frame_metadata.json`), reaches the hook on a unary request's
    /// terminal frame, also from behind logprob columns, in place of this
    /// frontend's own measure (zero here: the terminal is the first output).
    #[tokio::test]
    async fn scheduler_decode_latency_reaches_the_hook() {
        for (name, rid) in [("batch", "done"), ("logprob_batch", "logprobs")] {
            let (case, events) = crate::api_server::core::frame_metadata_case(name);
            let mut event = events
                .into_iter()
                .find(|event| event.rid.as_str() == rid)
                .unwrap();
            event.rid = Rid::from("10".to_owned());
            let expected = case["decode_latencies"][rid]
                .as_f64()
                .map(Duration::from_secs_f64);
            assert!(expected.is_some(), "{name}: the fixture times {rid}");

            let recorded = Recorded::default();
            let (tx, rx) = mpsc::channel(1);
            let (mut call, mut request) = timed_call(10, rx);
            request.hook = recorded.hook();
            tx.send(ResponseItem::Done(event)).await.unwrap();
            let (status, item) = drain_unary(&mut call, "client-rid", request).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(
                serde_json::to_value(&item).unwrap()["meta_info"]["hook_frame"],
                0
            );

            let frames = recorded.frames();
            assert_eq!(frames.len(), 1, "{name}");
            assert!(frames[0].finished, "{name}");
            assert_eq!(frames[0].decode_latency, expected, "{name}");
        }
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
        let mut call =
            CoreCall::from_test_generation_parts("internal-rid".into(), rx, flume::unbounded().0);

        let timing = RequestTiming {
            created_at: Instant::now() - Duration::from_millis(20),
            time_to_first_token: None,
            e2e_latency: None,
        };
        let request = RequestState {
            timing,
            scheduler_decode_latency: None,
            hook: None,
        };
        let (status, item) = drain_unary(&mut call, "client-rid", request).await;
        let value = serde_json::to_value(&item).unwrap();
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

    #[tokio::test]
    async fn scheduler_rejection_preserves_legacy_http_status() {
        let (tx, rx) = mpsc::channel(8);
        let calls = vec![timed_call(10, rx)];
        let stream = generation_event_stream(calls, false, false);
        futures::pin_mut!(stream);

        tx.send(ResponseItem::Done(ChunkEvent {
            finish_reason: serde_json::from_value(serde_json::json!({
                "type": "abort",
                "message": "media decode failed",
                "status_code": 432,
                "err_type": "BadRequestError"
            }))
            .unwrap(),
            ..Default::default()
        }))
        .await
        .unwrap();

        let failure = parse(&stream.next().await.unwrap());
        assert_eq!(failure["error"]["code"], 432);
        assert_eq!(failure["error"]["message"], "media decode failed");
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

    /// A runtime close must not overtake output that was already queued. The
    /// cumulative adapter first exposes the produced text, then reports the
    /// truncation on the next event, preserving the Step 2 HTTP behavior.
    #[tokio::test]
    async fn cumulative_output_precedes_truncation_error() {
        let (tx, rx) = mpsc::channel(8);
        let calls = vec![timed_call(10, rx)];
        let stream = generation_event_stream(calls, false, false);
        futures::pin_mut!(stream);

        tx.send(frame(10, "already-produced")).await.unwrap();
        drop(tx);

        let output = parse(&stream.next().await.unwrap());
        assert_eq!(output["text"], "already-produced");

        let failure = parse(&stream.next().await.unwrap());
        assert_eq!(failure["error"]["code"], 500);
        assert_eq!(
            failure["error"]["message"],
            "response truncated before completion"
        );
        assert_eq!(stream.next().await.unwrap(), "[DONE]");
        assert!(stream.next().await.is_none());
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
