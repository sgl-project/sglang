//! The native SGLang data-plane endpoints: `/generate` (submit a request, then
//! either fold decode frames to one unary JSON response or relay them as SSE
//! `data: {json}` … `[DONE]`, byte-compatible with Python
//! `http_server.generate_request`) and `/health` + `/health_generate` (which
//! round-trip a 1-token generate probe). Frame shaping (`meta_info`, logprob
//! tuples, cumulative vs incremental streams) lives here; submission and
//! request lifetime belong to the shared [`crate::api_server::handlers::CoreHandle`].

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::app::AppState;
use super::auth::DeclaredRoute;
use super::generate_hook::{
    GenerateFrameHook, GenerateFrameInfo, GenerateFrameKeys, GenerateHook, GenerateTerminal,
};
use super::pre_generate::HeldGeneration;
use super::request_headers::apply_header_overrides;
use crate::api_server::core::{
    CoreError, CoreEvent, CoreOutput, HealthStatus, STREAM_INVALID_REQUEST_ERROR,
};
use crate::api_server::handlers::{CoreCall, PendingGeneration, recv_indexed};
use crate::api_server::log::{RawBody, ReceivedAs};
use crate::message::config::ServerArgs;
#[cfg(test)]
use crate::message::ids::Rid;
use crate::message::request::{GenerateRequest, GenerationPolicy, into_requests};
use crate::message::response::{CustomizedInfo, RequestStats};
use crate::message::wire::{ClientMaxNewTokens, merge_preferred_sampling};
use crate::utils::{
    environ,
    response::{error_response, error_value, sse_error_response, sse_response},
};
use axum::{
    Extension, Json, Router,
    extract::State,
    extract::rejection::JsonRejection,
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response, sse::Event},
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
pub(in crate::api_server) struct RequestTiming {
    // TODO: Move request lifecycle timing into a dedicated tracing/metrics
    // module and align its design with Python's APIServerReqTimeStats.
    created_at: Instant,
    time_to_first_token: Option<Duration>,
    e2e_latency: Option<Duration>,
}

impl RequestTiming {
    pub(in crate::api_server) fn new() -> Self {
        Self::starting_at(Instant::now())
    }

    fn starting_at(created_at: Instant) -> Self {
        Self {
            created_at,
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

    /// The terminal output arrived; when no output came before it, it is
    /// also the first, at the same instant: Python's TokenizerManager reuses
    /// the first output's arrival time as the finish time, so a request that
    /// finishes in its first output has no decode interval.
    fn observe_terminal_output(&mut self) {
        let now = self.created_at.elapsed();
        self.time_to_first_token.get_or_insert(now);
        self.e2e_latency.get_or_insert(now);
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

/// API-local state for one request: its timing, its `customized_info` as its
/// frames report it, and the annotator a model package's [`GenerateHook`]
/// started for it.
pub(in crate::api_server) struct RequestState {
    timing: RequestTiming,
    /// The scheduler's decode latency, sent for a finished non-streaming
    /// request: this frontend sees that request's first output only up to
    /// `SGLANG_FORCE_STREAM_INTERVAL` tokens late, not when its first token
    /// was sampled.
    scheduler_decode_latency: Option<Duration>,
    /// Metrics are on, so the terminal frame reports the decode throughput
    /// after the scheduler's request stats, as Python's TokenizerManager does.
    report_decode_throughput: bool,
    /// The output token logprobs the request has returned so far, for the
    /// hook's terminal view.
    output_token_logprobs: usize,
    customized_info: CustomizedInfoState,
    hook: Option<RequestHook>,
}

/// A request's annotator, with the hook that started it.
struct RequestHook {
    hook: Arc<dyn GenerateHook>,
    frames: Box<dyn GenerateFrameHook>,
}

/// What decoding and annotating a `/generate` body reads of a listener.
pub(in crate::api_server) struct GenerateContext<'a> {
    pub(in crate::api_server) server_args: &'a ServerArgs,
    pub(in crate::api_server) generation_policy: &'a GenerationPolicy,
    pub(in crate::api_server) generate_hook: Option<&'a Arc<dyn GenerateHook>>,
}

impl AppState {
    pub(super) fn generate_context(&self) -> GenerateContext<'_> {
        GenerateContext {
            server_args: &self.server_args,
            generation_policy: &self.generation_policy,
            generate_hook: self.generate_hook.as_ref(),
        }
    }
}

impl RequestState {
    pub(in crate::api_server) fn new(
        state: &GenerateContext<'_>,
        request: &GenerateRequest,
        headers: &HeaderMap,
        client_sampling_params: Option<&serde_json::Map<String, serde_json::Value>>,
        timing: RequestTiming,
    ) -> Self {
        Self {
            timing,
            scheduler_decode_latency: None,
            report_decode_throughput: state.server_args.metrics.is_some(),
            output_token_logprobs: 0,
            customized_info: CustomizedInfoState::default(),
            hook: state.generate_hook.map(|hook| RequestHook {
                hook: hook.clone(),
                frames: hook.start(request, headers, client_sampling_params),
            }),
        }
    }

    /// Note one of the request's outputs as it arrives, taking its
    /// `customized_info`.
    fn observe(&mut self, output: &mut CoreOutput) {
        self.timing.observe_first_output();
        self.customized_info.observe(output.take_customized_info());
        self.count_output_token_logprobs(output);
    }

    fn finish(&mut self, terminal: &mut CoreOutput) {
        self.customized_info
            .observe(terminal.take_customized_info());
        self.timing.observe_terminal_output();
        self.scheduler_decode_latency = terminal.scheduler_decode_latency();
        self.count_output_token_logprobs(terminal);
    }

    fn count_output_token_logprobs(&mut self, delta: &CoreOutput) {
        if let Some(extras) = delta.extras.as_deref() {
            self.output_token_logprobs += extras.out_lp_val.len();
        }
    }

    fn fail(&mut self, error: &CoreError) {
        self.timing.finish();
        if let Some(RequestHook { frames, .. }) = &mut self.hook {
            frames.failed(error.http_status(), &error.to_string());
        }
    }

    /// From the first token to the terminal output, as Python's
    /// TokenizerManager measures it: the scheduler's measure when it sent one,
    /// else this frontend's first and terminal outputs.
    fn decode_latency(&self) -> Option<Duration> {
        self.scheduler_decode_latency
            .or_else(|| self.timing.decode_latency())
    }

    /// One frame as the client receives it, from `output`: the request's
    /// cumulative output, or one step's delta in an incremental stream. The
    /// schema's keys, then a finished request's stats, then the
    /// `customized_info` keys it reports, then the hook's keys.
    fn frame(
        &mut self,
        mut output: CoreOutput,
        mode: FrameMode,
        id: &str,
        index: Option<u32>,
        e2e_latency: Option<f64>,
    ) -> NativeFrame {
        let finished = output.finish_reason.is_some();
        let decode_latency = if finished {
            self.decode_latency()
        } else {
            None
        };
        let mut request_stats = output.take_request_stats();
        if self.report_decode_throughput
            && let Some(RequestStats(stats)) = request_stats.as_mut()
            && let Some(throughput) = decode_throughput(decode_latency, output.completion_tokens)
        {
            stats.insert("decode_throughput".into(), throughput.into());
        }
        let frame = output.frame(id, index, e2e_latency);
        let mut keys = None;
        if let Some(RequestHook { frames, .. }) = &mut self.hook {
            let new_values = self.customized_info.new_values(mode);
            let info = GenerateFrameInfo {
                finished,
                decode_latency,
                customized_info: &new_values,
            };
            let mut added = GenerateFrameKeys::default();
            frames.annotate(&info, &mut added);
            keys = Some(Box::new(added));
        }
        let hook = self.hook.as_ref().map(|request| &request.hook);
        let hook_keys = keys.as_deref().map(|keys| &keys.meta_info);
        let stats_keys = request_stats.as_ref().map(|RequestStats(stats)| stats);
        let customized_info = self.customized_info.report(mode, |key| {
            if hook.is_some_and(|hook| hook.consumes_customized_info_key(key)) {
                return false;
            }
            let repeated = SCHEMA_META_INFO_KEYS.contains(&key)
                || stats_keys.is_some_and(|stats| stats.contains_key(key))
                || hook_keys.is_some_and(|keys| keys.contains_key(key));
            if repeated && !REPEATED_KEY_WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    key,
                    "a customized_info key repeats one meta_info already has; frames keep the other"
                );
            }
            !repeated
        });
        if finished
            && let (Some(RequestHook { frames, .. }), Some(meta_info)) =
                (&mut self.hook, frame.meta_info.as_ref())
        {
            frames.finished(&GenerateTerminal {
                meta_info,
                request_stats: request_stats.as_ref().map(|RequestStats(stats)| stats),
                decode_latency,
                output_token_logprobs: self.output_token_logprobs,
            });
        }
        NativeFrame {
            frame,
            request_stats,
            customized_info,
            keys,
        }
    }
}

/// Python's `get_decode_throughput`: one over the time per output token after
/// the first, for a request with more than one and a positive decode latency.
fn decode_throughput(decode_latency: Option<Duration>, completion_tokens: u64) -> Option<f64> {
    let decode_latency = decode_latency?.as_secs_f64();
    (completion_tokens > 1 && decode_latency > 0.0)
        .then(|| 1.0 / (decode_latency / (completion_tokens - 1) as f64))
}

/// The keys `api::GenerateMetaInfo` writes. A `customized_info` key of the
/// same name gives way (Python would overwrite them with it).
const SCHEMA_META_INFO_KEYS: &[&str] = &[
    "id",
    "prompt_tokens",
    "completion_tokens",
    "finish_reason",
    "output_token_logprobs",
    "input_token_logprobs",
    "output_top_logprobs",
    "input_top_logprobs",
    "output_token_ids_logprobs",
    "input_token_ids_logprobs",
    "hidden_states",
    "e2e_latency",
    "cached_tokens",
    "reasoning_tokens",
    "output_token_logprobs_length",
    "weight_version",
    "weight_versions",
    "image_tokens",
    "audio_tokens",
    "video_tokens",
    "cached_tokens_details",
    "dp_rank",
];

/// A repeated key has been logged once.
static REPEATED_KEY_WARNED: AtomicBool = AtomicBool::new(false);

/// How a frame reports `customized_info`, following Python's TokenizerManager.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrameMode {
    /// A unary response's one frame: the keys its terminal output carried,
    /// with all their values.
    Unary,
    /// A cumulative stream's frame: every key the request has surfaced so far,
    /// with all its values, whether or not the frame's latest output carried
    /// it.
    Cumulative,
    /// An incremental stream's frame: the keys its latest output carried, with
    /// that output's values.
    Incremental,
}

/// A request's `customized_info` as its frames report it, which is how
/// Python's TokenizerManager reports it. A key surfaces once an output carries
/// it and the frame accepts it (not one a hook consumes); after that a
/// cumulative frame re-reports it, while an incremental frame or a unary
/// response reports only the keys its latest output carried. A hook sees each
/// value once, on the first frame that shows its key after it arrives.
#[derive(Debug, Default)]
struct CustomizedInfoState(Vec<KeyState>);

#[derive(Debug)]
struct KeyState {
    key: Arc<str>,
    /// The values since the last frame that showed the key.
    new: Vec<serde_json::Value>,
    /// The ones earlier frames reported, for a cumulative frame. Kept only
    /// for a key the frames report (not one the hook consumes); shared, so a
    /// frame does not copy them.
    reported: Arc<Vec<serde_json::Value>>,
    /// The latest output carried the key.
    latest: bool,
}

impl CustomizedInfoState {
    fn observe(&mut self, info: CustomizedInfo) {
        for key in &mut self.0 {
            key.latest = false;
        }
        for (key, mut values) in info {
            match self.0.iter_mut().find(|state| state.key == key) {
                Some(state) => {
                    state.new.append(&mut values);
                    state.latest = true;
                }
                None => self.0.push(KeyState {
                    key,
                    new: values,
                    reported: Arc::default(),
                    latest: true,
                }),
            }
        }
    }

    /// The keys the next frame shows a hook, with the values new to it: in a
    /// cumulative stream every key so far, otherwise the latest output's.
    fn new_values(&self, mode: FrameMode) -> Vec<(&str, &[serde_json::Value])> {
        self.0
            .iter()
            .filter(|state| mode == FrameMode::Cumulative || state.latest)
            .map(|state| (&*state.key, state.new.as_slice()))
            .collect()
    }

    /// The frame's report: each key `report` accepts, of every key so far in
    /// a cumulative frame or of the latest output's otherwise, with its values
    /// so far or, in an incremental frame, the new ones. The new values are
    /// then shown.
    fn report(
        &mut self,
        mode: FrameMode,
        mut report: impl FnMut(&str) -> bool,
    ) -> ReportedCustomizedInfo {
        let mut reported = Vec::new();
        for state in self
            .0
            .iter_mut()
            .filter(|state| mode == FrameMode::Cumulative || state.latest)
        {
            let new = std::mem::take(&mut state.new);
            if !report(&state.key) {
                continue;
            }
            let values = if mode == FrameMode::Incremental {
                Arc::new(new)
            } else {
                if state.reported.is_empty() {
                    state.reported = Arc::new(new);
                } else {
                    Arc::make_mut(&mut state.reported).extend(new);
                }
                state.reported.clone()
            };
            reported.push((state.key.clone(), values));
        }
        ReportedCustomizedInfo(reported)
    }
}

/// The `customized_info` keys one frame's `meta_info` reports, with their
/// values.
#[derive(Debug, Default)]
struct ReportedCustomizedInfo(Vec<(Arc<str>, Arc<Vec<serde_json::Value>>)>);

impl serde::Serialize for ReportedCustomizedInfo {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.iter().map(|(key, values)| (&**key, &**values)))
    }
}

/// One native `/generate` frame: the `api.v1` frame, then in its `meta_info`
/// a finished request's stats, the request's `customized_info` keys and the
/// keys a model package's hook added.
#[derive(Debug)]
pub(in crate::api_server) struct NativeFrame {
    frame: api::GenerateResponse,
    request_stats: Option<RequestStats>,
    customized_info: ReportedCustomizedInfo,
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
            request_stats: Option<&'a RequestStats>,
            #[serde(flatten)]
            customized_info: &'a ReportedCustomizedInfo,
            #[serde(flatten)]
            added: &'a serde_json::Map<String, serde_json::Value>,
        }

        if self.request_stats.is_none() && self.keys.is_none() && self.customized_info.0.is_empty()
        {
            return self.frame.serialize(serializer);
        }
        let no_keys = GenerateFrameKeys::default();
        let keys = self.keys.as_deref().unwrap_or(&no_keys);
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
                request_stats: self.request_stats.as_ref(),
                customized_info: &self.customized_info,
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
        // Python takes PUT as well as POST, with the same handler.
        .declared_route("/generate", post(generate).put(generate))
        .merge(health_routes())
}

/// native api error response: unary → `code` plus the JSON `body`,
/// streaming → 200 with one SSE error frame + `[DONE]`. A message-only error
/// is a request the handler refused before it reached the core, which Python
/// raises as a `ValueError`, so its stream frame is `invalid_request_error`
/// and not retryable (see [`CoreError::stream_error`]).
pub(super) fn native_error(code: StatusCode, message: &str, stream: bool) -> Response {
    if !stream {
        return error_response(code, error_value(code.as_u16(), message), false);
    }
    let body = api::ErrorBody {
        message: message.to_owned(),
        code: u32::from(code.as_u16()),
        r#type: Some(STREAM_INVALID_REQUEST_ERROR.to_owned()),
        retryable: Some(false),
    };
    sse_error_response(serde_json::json!({ "error": body }))
}

fn native_core_error(error: CoreError, stream: bool) -> Response {
    if !stream {
        return native_error(error.http_status(), &error.to_string(), false);
    }
    sse_error_response(
        serde_json::to_value(error.stream_error(None)).expect("an error item is JSON"),
    )
}

/// `/health` + `/health_generate`. Both env knobs are resolved ONCE here, at
/// router build (server startup) — changing them on a live process needs a
/// restart. The deep-probe handler is built once with
/// `SGLANG_HEALTH_CHECK_TIMEOUT` frozen in and serves `/health_generate`
/// always; `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` (default true, mirroring
/// Python) decides whether `/health` shares it or is a plain 200 once startup
/// warmup finishes and a multimodal model's MM processor is installed (routing
/// the request proves the frontend is up).
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
        .declared_route("/health", health)
        .declared_route("/health_generate", probe)
}

async fn health_without_generation(State(state): State<Arc<AppState>>) -> Response {
    if state.core.is_serving() && !state.core.is_draining() {
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
        Ok(HealthStatus::NotReady | HealthStatus::Stalled | HealthStatus::Draining) => {
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
    start: Option<Extension<crate::RequestStart>>,
    held: Option<Extension<HeldGeneration>>,
    headers: HeaderMap,
    request: axum::extract::Request,
) -> Response {
    let (mut body, raw_body) = match json_body(&state, request).await {
        Ok(body) => body,
        // A body that fails to parse has no readable `stream` flag, so this one
        // can only answer unary -- as Python's does (FastAPI rejects before its
        // handler runs).
        Err(rejection) => {
            return native_error(StatusCode::BAD_REQUEST, &rejection.body_text(), false);
        }
    };
    let overrides = if state.server_args.enable_request_header_overrides {
        match apply_header_overrides(&mut body, &headers) {
            Ok(applied) => Some(applied),
            Err(error) => return native_error(StatusCode::BAD_REQUEST, &error, false),
        }
    } else {
        None
    };
    // What the client sent as `sampling_params` (an object, or one per batch
    // item), for a package's hook, before anything fills in beneath it.
    let client_sampling_params = state
        .generate_hook
        .as_ref()
        .and_then(|_| body.get("sampling_params").cloned());
    // The request log and crash dump show the body as the client sent it:
    // the defaults merged below go into a copy, as Python does.
    let raw_body = RawBody {
        bytes: raw_body,
        overrides,
    };
    let received = ReceivedAs::Body {
        snapshot: state.core.snapshot_body(&body, Some(raw_body), &headers),
        started_at: start.map(|Extension(crate::RequestStart(at))| at),
    };
    let (payloads, is_batch) = match generate_requests(&state.generate_context(), body) {
        Ok(requests) => requests,
        Err((code, message, stream)) => return native_error(code, &message, stream),
    };
    let stream = payloads.first().is_some_and(|request| request.stream);
    // Python starts APIServerReqTimeStats after request normalization and before
    // tokenization / multimodal preprocessing / scheduler dispatch. Start at the
    // equivalent boundary: into_requests() has normalized the body, while prefetch
    // and every downstream stage are still ahead of us. An HTTP layer that did
    // some of that work before this handler ran set where it started.
    let timing = RequestTiming::starting_at(received.created_at());
    // Shared core admission performs media I/O (URL downloads, file reads)
    // on this async runtime before any request reaches the MM worker pool.
    if !is_batch {
        // `into_requests` guarantees exactly one payload for a non-batch body.
        let mut payload = payloads
            .into_iter()
            .next()
            .expect("into_requests yields >=1 payload");
        let held = held
            .and_then(|Extension(held)| held.take())
            .map(|held| held.claim(&mut payload));
        let client_sampling_params = item_sampling_params(client_sampling_params.as_ref(), 0);
        let context = state.generate_context();
        let request =
            RequestState::new(&context, &payload, &headers, client_sampling_params, timing);
        generate_single(&state, payload, held, stream, request, received).await
    } else {
        let requests = payloads
            .iter()
            .enumerate()
            .map(|(item, payload)| {
                let client_sampling_params =
                    item_sampling_params(client_sampling_params.as_ref(), item);
                RequestState::new(
                    &state.generate_context(),
                    payload,
                    &headers,
                    client_sampling_params,
                    timing.clone(),
                )
            })
            .collect();
        generate_batch(&state, payloads, stream, requests, received).await
    }
}

/// Whether Python reads `sampling_logprobs_mode` as set: any value but `null`,
/// or for a batch of `n` prompts a list naming a mode for one of them (Python
/// reads no entry past them; an empty list is set, being no per-prompt list).
fn sampling_logprobs_mode_set(mode: &serde_json::Value, batch: Option<usize>) -> bool {
    match (mode, batch) {
        (serde_json::Value::Null, _) => false,
        (serde_json::Value::Array(modes), Some(n)) => {
            modes.is_empty() || modes.iter().take(n).any(|mode| !mode.is_null())
        }
        _ => true,
    }
}

/// Batch item `item`'s share of the `sampling_params` the client sent: the
/// object itself, or the item's entry of a list.
fn item_sampling_params(
    sent: Option<&serde_json::Value>,
    item: usize,
) -> Option<&serde_json::Map<String, serde_json::Value>> {
    match sent? {
        serde_json::Value::Array(items) => items.get(item)?.as_object(),
        sent => sent.as_object(),
    }
}

/// The body exactly as axum's `Json<Value>` extracts it (the same content-type
/// check, body limit and rejections), with its raw bytes, which a crash dump
/// keeps as they came rather than serializing the body again.
async fn json_body<S: Send + Sync>(
    state: &S,
    request: axum::extract::Request,
) -> Result<(serde_json::Value, bytes::Bytes), JsonRejection> {
    use axum::extract::FromRequest;

    if !json_content_type(request.headers()) {
        return Err(axum::extract::rejection::MissingJsonContentType::default().into());
    }
    let raw = bytes::Bytes::from_request(request, state).await?;
    let Json(body) = Json::<serde_json::Value>::from_bytes(&raw)?;
    Ok((body, raw))
}

/// axum's `Json` content-type rule: `application/json` or `application/*+json`.
fn json_content_type(headers: &HeaderMap) -> bool {
    headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<mime::Mime>().ok())
        .is_some_and(|mime| {
            mime.type_() == "application"
                && (mime.subtype() == "json" || mime.suffix().is_some_and(|name| name == "json"))
        })
}

/// Decode a `/generate` body under the launch-time defaults into its requests and
/// whether it is a batch; an error carries the body's `stream` flag once known.
pub(in crate::api_server) fn generate_requests(
    state: &GenerateContext<'_>,
    mut body: serde_json::Value,
) -> Result<(Vec<GenerateRequest>, bool), (StatusCode, String, bool)> {
    // Read before the fills below: only a `max_new_tokens` the client sent is
    // checked against the context window.
    let client_max_new_tokens = ClientMaxNewTokens::of_json(&body);
    // Launch-time preferred sampling params fill in beneath the request's own
    // keys before decoding, so an explicit request value (even a default or
    // null) wins: Python TokenizerManager's precedence.
    if let Some(preferred) = &state.server_args.preferred_sampling_params
        && let Err(error) = merge_preferred_sampling(&mut body, &preferred.0)
    {
        return Err((StatusCode::INTERNAL_SERVER_ERROR, error, false));
    }
    // Beneath both, the launch-time default for an omitted `max_new_tokens`.
    let default_max_new_tokens = serde_json::json!({
        "max_new_tokens": state.server_args.default_max_new_tokens,
    });
    if let Err(error) = merge_preferred_sampling(&mut body, &default_max_new_tokens) {
        return Err((StatusCode::INTERNAL_SERVER_ERROR, error, false));
    }
    // Outside the schema, so decoding drops it; checked below.
    let sampling_logprobs_mode = body.get("sampling_logprobs_mode").cloned();
    // The schema (sglang_api_types) is the contract: decode errors carry
    // serde's field-level text, as Python's 400s do.
    let body: api::GenerateRequest = match serde_json::from_value(body) {
        Ok(body) => body,
        Err(error) => return Err((StatusCode::BAD_REQUEST, error.to_string(), false)),
    };
    let stream = body.stream.unwrap_or(false);
    // Fan `text`/`input_ids`/`sampling_params` (scalar or list) into per-request
    // payloads. `is_batch` = list form -> the response is a JSON array.
    // Request normalization only returns validation failures.
    let (mut requests, is_batch) = into_requests(body, state.generation_policy).map_err(|e| {
        tracing::warn!(error = %e, "rejected request");
        (StatusCode::BAD_REQUEST, e.to_string(), stream)
    })?;
    // Python refuses a mode without `return_sampling_mask=true`, and
    // `into_requests` refused the mask.
    let batch = is_batch.then_some(requests.len());
    if sampling_logprobs_mode.is_some_and(|mode| sampling_logprobs_mode_set(&mode, batch)) {
        let message = "sampling_logprobs_mode can only be set when return_sampling_mask=true.";
        tracing::warn!(error = %message, "rejected request");
        return Err((StatusCode::BAD_REQUEST, message.into(), stream));
    }
    client_max_new_tokens.mark(&mut requests);
    Ok((requests, is_batch))
}

/// A single (non-batched) `/generate`: submit one request, then either stream its
/// SSE frames or fold to one unary response.
async fn generate_single(
    state: &AppState,
    req: GenerateRequest,
    held: Option<PendingGeneration>,
    stream: bool,
    request: RequestState,
    received: ReceivedAs,
) -> Response {
    // `return_text_in_logprobs` is decoded on the detok shard into `*_txt`, so
    // the frame builder just reads them — no tokenizer needed here.
    let call = match held {
        Some(held) => state.core.generate_pending(req, received, held).await,
        None => state.core.generate(req, received).await,
    };
    let mut call = match call {
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
        sse_response(s)
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
pub(in crate::api_server) enum GenerateItem {
    // Boxed: a frame is ~300 bytes against the error's ~40, and a batch
    // array holds one per item.
    Frame(Box<NativeFrame>),
    Error(api::GenerateStreamError),
}

/// Fold a unary request to its terminal HTTP result. Shared by single + batch.
pub(super) async fn drain_unary(
    call: &mut CoreCall,
    rid_str: &str,
    request: RequestState,
) -> (StatusCode, GenerateItem) {
    match drain_unary_frame(call, rid_str, request).await {
        Ok(frame) => (StatusCode::OK, GenerateItem::Frame(Box::new(frame))),
        Err(error) => (
            error.http_status(),
            GenerateItem::Error(error.item_error(None)),
        ),
    }
}

/// [`drain_unary`]'s terminal frame, or the error that failed the call.
pub(in crate::api_server) async fn drain_unary_frame(
    call: &mut CoreCall,
    rid_str: &str,
    mut request: RequestState,
) -> Result<NativeFrame, CoreError> {
    let mut accumulated = CoreOutput::default();
    while let Some(item) = call.recv().await {
        match item {
            CoreEvent::Delta(mut out) => {
                request.observe(&mut out);
                accumulated.append_delta(&out);
            }
            CoreEvent::Finished(mut out) => {
                request.finish(&mut out);
                accumulated.append_delta(&out);
                let e2e_latency = Some(e2e_latency(&request.timing));
                return Ok(request.frame(
                    accumulated,
                    FrameMode::Unary,
                    rid_str,
                    None,
                    e2e_latency,
                ));
            }
            CoreEvent::Failed(error) => {
                request.fail(&error);
                return Err(error);
            }
        }
    }
    // Defensive fallback: CoreCall normally turns premature runtime
    // termination into a terminal Failed event before exposing stream end.
    Err(CoreError::ResponseTruncated)
}

/// Batch `/generate`: submit all sub-requests first (scheduler runs them together),
/// then either (unary) drain them concurrently into a request-ordered JSON array,
/// or (streaming) multiplex their streams into one SSE response, each frame carrying
/// its `index`.
/// Each submitted call owns its own cancellation. A failed unary item is its
/// own `{ "error": … }` entry; the batch response is 200. This differs from
/// Python on purpose: there, one item failing validation fails the whole batch
/// with a single 400.
async fn generate_batch(
    state: &AppState,
    requests: Vec<GenerateRequest>,
    stream: bool,
    request_states: Vec<RequestState>,
    received: ReceivedAs,
) -> Response {
    // No cross-item rid collision to worry about: `into_requests` rejected duplicate
    // rids within this batch, and `Rid::from_client` made each one unique against
    // every other in-flight request.
    let calls = match state.core.generate_batch(requests, received).await {
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
        sse_response(s)
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
pub(super) fn generation_event_stream(
    calls: Vec<(CoreCall, RequestState)>,
    incremental: bool,
    with_index: bool,
) -> impl futures::Stream<Item = String> {
    generation_event_stream_with(calls, incremental, with_index, |error, index| {
        json_string(&error.stream_error(index))
    })
}

/// [`generation_event_stream`], writing a failed call's frame with `error_frame`.
pub(in crate::api_server) fn generation_event_stream_with(
    calls: Vec<(CoreCall, RequestState)>,
    incremental: bool,
    with_index: bool,
    error_frame: impl Fn(&CoreError, Option<u32>) -> String,
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

            for item in items.into_iter().map(stream_event) {
                match item {
                    CoreEvent::Delta(mut out) => {
                        requests[i].observe(&mut out);
                        accumulated[i].append_delta(&out);
                        if incremental {
                            let out = delta_output(out, &accumulated[i]);
                            yield json_string(&requests[i].frame(out, FrameMode::Incremental, &rid_strs[i], idx(i), None));
                        } else {
                            coalesced = true;
                        }
                    }
                    CoreEvent::Finished(mut out) => {
                        requests[i].finish(&mut out);
                        accumulated[i].append_delta(&out);
                        terminal = Some(out);
                    }
                    CoreEvent::Failed(error) => {
                        requests[i].fail(&error);
                        failed = Some(error);
                    }
                }
            }

            if let Some(error) = failed {
                yield error_frame(&error, idx(i));
            } else if let Some(out) = terminal {
                // The final frame carries the full cumulative state, so any
                // coalesced non-terminal frames are moot.
                let e2e_latency = Some(e2e_latency(&requests[i].timing));
                // The terminal frame is the accumulated output's last use, so
                // the cumulative form moves it out instead of cloning.
                let (out, mode) = if incremental {
                    (delta_output(out, &accumulated[i]), FrameMode::Incremental)
                } else {
                    (std::mem::take(&mut accumulated[i]), FrameMode::Cumulative)
                };
                yield json_string(&requests[i].frame(out, mode, &rid_strs[i], idx(i), e2e_latency));
            } else {
                if coalesced {
                    // An intermediate cumulative frame has to clone: the accumulated
                    // output stays behind for the next one.
                    let out = accumulated[i].clone();
                    yield json_string(&requests[i].frame(out, FrameMode::Cumulative, &rid_strs[i], idx(i), None));
                }
                futs.push(recv_indexed(i, call)); // keep this item flowing
            }
        }
        yield "[DONE]".to_string();
    }
}

/// A stream's view of `event`: a scheduler abort that carried an error status
/// ends a native stream as Python's server ends it, on a normal terminal frame
/// whose `meta_info.finish_reason` is the abort, status code and message
/// included. A unary response reports the abort's status instead.
fn stream_event(event: CoreEvent) -> CoreEvent {
    match event {
        CoreEvent::Failed(error) => match error.into_rejected_output() {
            Ok(output) => CoreEvent::Finished(output),
            Err(error) => CoreEvent::Failed(error),
        },
        event => event,
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

/// One incremental streaming frame's output: this step's delta with the
/// cumulative counts in `meta_info` (see [`CoreOutput::with_cumulative_counts`]).
fn delta_output(delta: CoreOutput, accumulated: &CoreOutput) -> CoreOutput {
    delta.with_cumulative_counts(accumulated)
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
                report_decode_throughput: false,
                output_token_logprobs: 0,
                customized_info: CustomizedInfoState::default(),
                hook: None,
            },
        )
    }

    #[tokio::test]
    async fn health_is_unavailable_before_startup_warmup_finishes() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        let state = Arc::new(AppState {
            core: crate::api_server::handlers::CoreHandle::new(
                intake_tx,
                abort_tx,
                crate::api_server::handlers::CoreConfig {
                    response_capacity: 8,
                    response_activity: Default::default(),
                    startup_ready: false,
                    is_disaggregation: false,
                    mm_limits: Default::default(),
                    multimodal: false,
                    peer_endpoints: Vec::new(),
                    metadata: crate::api_server::handlers::CoreMetadata::default(),
                    metrics: None,
                    request_log: None,
                    generate_validator: None,
                    abort_hook: None,
                },
            ),
            server_args: Arc::new(crate::message::config::ServerArgs::default()),
            generation_policy: GenerationPolicy::default(),
            chat_formatter: None,
            generate_hook: None,
            openai: Default::default(),
        });

        let response = health_generate(State(state), Duration::ZERO, true).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(intake_rx.try_recv().is_err());
        assert!(abort_rx.try_recv().is_err());
    }

    /// The generation-free `/health` of a multimodal model answers 503 until
    /// its MM processor is installed, even with startup warmup skipped, since
    /// a generation sent earlier would wait in admission.
    #[tokio::test]
    async fn generation_free_health_waits_for_the_mm_processor() {
        struct NoMedia;
        impl crate::multi_modality::worker::MmProcessor for NoMedia {
            fn process(
                &self,
                _input_ids: crate::message::types::TokenIds,
                _mm: crate::message::request::MmData,
            ) -> Result<
                crate::multi_modality::worker::MmProcessOutput,
                crate::multi_modality::worker::MediaError,
            > {
                unreachable!("no request carries media")
            }
        }
        let state = Arc::new(AppState {
            core: crate::api_server::handlers::CoreHandle::new(
                flume::unbounded().0,
                flume::unbounded().0,
                crate::api_server::handlers::CoreConfig {
                    response_capacity: 8,
                    response_activity: Default::default(),
                    startup_ready: true,
                    is_disaggregation: false,
                    mm_limits: Default::default(),
                    multimodal: true,
                    peer_endpoints: Vec::new(),
                    metadata: crate::api_server::handlers::CoreMetadata::default(),
                    metrics: None,
                    request_log: None,
                    generate_validator: None,
                    abort_hook: None,
                },
            ),
            server_args: Arc::new(crate::message::config::ServerArgs::default()),
            generation_policy: GenerationPolicy::default(),
            chat_formatter: None,
            generate_hook: None,
            openai: Default::default(),
        });

        let response = health_without_generation(State(state.clone())).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(state.core.set_mm_processor(Arc::new(NoMedia)));
        let response = health_without_generation(State(state)).await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    /// `/health` answers 503 while Python runs `--warmups` through the
    /// listener, though startup warmup already finished, so the router sends
    /// no traffic before the warmups compiled what they exist to compile.
    #[tokio::test]
    async fn health_waits_for_custom_warmups() {
        let state = Arc::new(AppState {
            core: crate::api_server::handlers::CoreHandle::new(
                flume::unbounded().0,
                flume::unbounded().0,
                crate::api_server::handlers::CoreConfig {
                    response_capacity: 8,
                    response_activity: Default::default(),
                    startup_ready: true,
                    is_disaggregation: false,
                    mm_limits: Default::default(),
                    multimodal: false,
                    peer_endpoints: Vec::new(),
                    metadata: crate::api_server::handlers::CoreMetadata::default(),
                    metrics: None,
                    request_log: None,
                    generate_validator: None,
                    abort_hook: None,
                },
            ),
            server_args: Arc::new(crate::message::config::ServerArgs::default()),
            generation_policy: GenerationPolicy::default(),
            chat_formatter: None,
            generate_hook: None,
            openai: Default::default(),
        });
        state.core.hold_for_custom_warmups();

        let response = health_without_generation(State(state.clone())).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        state.core.finish_custom_warmups();
        let response = health_without_generation(State(state)).await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    /// Each item `body` submits under `server_args`: its `max_new_tokens`, and
    /// whether that value is marked defaulted.
    async fn submitted_max_new_tokens(
        server_args: crate::message::config::ServerArgs,
        body: serde_json::Value,
        items: usize,
    ) -> Vec<(Option<i64>, bool)> {
        let (intake_tx, intake_rx) = flume::unbounded();
        let core = crate::api_server::handlers::CoreHandle::new(
            intake_tx,
            flume::unbounded().0,
            crate::api_server::handlers::CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Default::default(),
                multimodal: false,
                peer_endpoints: Vec::new(),
                metadata: crate::api_server::handlers::CoreMetadata::default(),
                metrics: None,
                request_log: None,
                generate_validator: None,
                abort_hook: None,
            },
        );
        let state = Arc::new(AppState {
            core,
            generation_policy: GenerationPolicy::from(&server_args),
            server_args: Arc::new(server_args),
            chat_formatter: None,
            generate_hook: None,
            openai: Default::default(),
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

    /// A body `/generate` refuses before the core sees it is answered with the
    /// bare message, as Python's `ValueError` text, in the shape the client
    /// asked for (a 400, or one in-stream error frame), and logged with the
    /// reason, as Python's `[http_server] Error:` line is.
    #[tokio::test]
    async fn refused_bodies_answer_and_log_the_bare_reason() {
        let core = crate::api_server::handlers::CoreHandle::new(
            flume::unbounded().0,
            flume::unbounded().0,
            crate::api_server::handlers::CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Default::default(),
                multimodal: false,
                metadata: crate::api_server::handlers::CoreMetadata::default(),
                metrics: None,
                peer_endpoints: Vec::new(),
                request_log: None,
                generate_validator: None,
                abort_hook: None,
            },
        );
        let server_args = crate::message::config::ServerArgs::default();
        let state = Arc::new(AppState {
            core,
            generation_policy: GenerationPolicy::from(&server_args),
            server_args: Arc::new(server_args),
            chat_formatter: None,
            generate_hook: None,
            openai: Default::default(),
        });
        let app = routes().with_state(state.clone());
        for (body, message) in [
            (
                serde_json::json!({"text": "hi", "input_ids": [1]}),
                "provide either `text` or `input_ids`, not both",
            ),
            (
                serde_json::json!({"text": "hi", "custom_logit_processor": "cls"}),
                "custom_logit_processor is not supported by the Rust server",
            ),
            (
                serde_json::json!({"text": "hi", "sampling_logprobs_mode": "support"}),
                "sampling_logprobs_mode can only be set when return_sampling_mask=true.",
            ),
            (
                serde_json::json!({"text": "hi", "sampling_logprobs_mode": [null]}),
                "sampling_logprobs_mode can only be set when return_sampling_mask=true.",
            ),
            (
                serde_json::json!({"text": ["a", "b"], "sampling_logprobs_mode": [null, "support"]}),
                "sampling_logprobs_mode can only be set when return_sampling_mask=true.",
            ),
            (
                serde_json::json!({
                    "text": "hi",
                    "return_sampling_mask": true,
                    "sampling_logprobs_mode": "support",
                }),
                "return_sampling_mask is not supported by the Rust server",
            ),
        ] {
            for stream in [false, true] {
                let mut body = body.clone();
                body["stream"] = stream.into();
                let logs = crate::utils::logging::capture_logs();
                let request = Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap();
                let response = app.clone().oneshot(request).await.unwrap();
                let status = response.status();
                let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
                    .await
                    .unwrap();
                let text = std::str::from_utf8(&bytes).unwrap();
                let error = if stream {
                    assert_eq!(status, StatusCode::OK, "{body}");
                    let frame = text
                        .strip_prefix("data: ")
                        .and_then(|rest| rest.strip_suffix("\n\ndata: [DONE]\n\n"))
                        .unwrap_or_else(|| panic!("one error frame then [DONE]: {text}"));
                    parse(frame)
                } else {
                    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
                    parse(text)
                };
                assert_eq!(error["error"]["message"], message, "{body}");
                assert_eq!(error["error"]["code"], 400, "{body}");
                // Python's stream frame names the error's type and that a
                // retry won't help; its unary body has neither.
                if stream {
                    assert_eq!(error["error"]["type"], "invalid_request_error", "{body}");
                    assert_eq!(error["error"]["retryable"], false, "{body}");
                } else {
                    assert!(error["error"].get("type").is_none(), "{body}");
                    assert!(error["error"].get("retryable").is_none(), "{body}");
                }
                let rejections = logs.rejections();
                assert!(
                    matches!(rejections.as_slice(), [line] if line.contains("WARN") && line.contains(message)),
                    "{rejections:?}"
                );
            }
        }
        // A mode Python reads as unset passes.
        for body in [
            serde_json::json!({"text": "hi", "sampling_logprobs_mode": null}),
            serde_json::json!({"text": ["a", "b"], "sampling_logprobs_mode": [null, null, "support"]}),
        ] {
            assert!(
                generate_requests(&state.generate_context(), body.clone()).is_ok(),
                "{body}"
            );
        }
        // A body the schema refuses is refused as such first, unlogged, as
        // Python's contract check is.
        let logs = crate::utils::logging::capture_logs();
        let body = serde_json::json!({
            "text": "hi",
            "stream": true,
            "sampling_params": {"unknown": 1},
            "sampling_logprobs_mode": "support",
        });
        let (status, message, _) = generate_requests(&state.generate_context(), body).unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(message.contains("unknown"), "{message}");
        assert!(logs.rejections().is_empty());
    }

    /// `/generate` checks the DP rank a body names as Python does: a rank
    /// outside the deployment is a 400 with Python's message, logged once.
    #[tokio::test]
    async fn generate_refuses_a_routed_dp_rank_out_of_range() {
        let state = Arc::new(AppState {
            core: crate::api_server::handlers::CoreHandle::new(
                flume::unbounded().0,
                flume::unbounded().0,
                crate::api_server::handlers::CoreConfig {
                    response_capacity: 8,
                    response_activity: Default::default(),
                    startup_ready: true,
                    is_disaggregation: false,
                    mm_limits: Default::default(),
                    multimodal: false,
                    peer_endpoints: Vec::new(),
                    metadata: crate::api_server::handlers::CoreMetadata::default(),
                    metrics: None,
                    request_log: None,
                    generate_validator: None,
                    abort_hook: None,
                },
            ),
            generation_policy: GenerationPolicy::default(),
            server_args: Arc::new(crate::message::config::ServerArgs::default()),
            chat_formatter: None,
            generate_hook: None,
            openai: Default::default(),
        });
        let (body, rank) = (r#"{"input_ids": [1], "routed_dp_rank": 3}"#, 3);
        let logs = crate::utils::logging::capture_logs();
        let request = Request::builder()
            .method("POST")
            .uri("/generate")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let response = routes()
            .with_state(state.clone())
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{body}");
        let reply = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let message = format!("routed_dp_rank={rank} out of range [0, 1)");
        assert_eq!(
            parse(std::str::from_utf8(&reply).unwrap())["error"]["message"],
            message
        );
        let rejections = logs.rejections();
        assert!(
            matches!(rejections.as_slice(), [line] if line.contains("WARN") && line.contains(&message)),
            "{rejections:?}"
        );
    }

    /// What each item is submitted with for `max_new_tokens`, and whether it
    /// is marked defaulted. An item whose `sampling_params` name none takes
    /// the preferred sampling params' value, else the launch default (no
    /// limit when that is `None`), and is marked; one that names it, even as
    /// null, keeps the client's and is not. A list is marked object by
    /// object, and one object marks every item.
    #[tokio::test]
    async fn omitted_max_new_tokens_is_filled_and_marked_defaulted() {
        use crate::message::config::{PreferredSamplingParams, ServerArgs};
        let launch_default = ServerArgs::default().default_max_new_tokens;
        let args = |default_max_new_tokens, preferred: Option<serde_json::Value>| ServerArgs {
            default_max_new_tokens,
            preferred_sampling_params: preferred.map(PreferredSamplingParams),
            ..Default::default()
        };
        let preferred = Some(serde_json::json!({"temperature": 1.0, "max_new_tokens": 4096}));
        let preferred_args = || args(launch_default, preferred.clone());
        let omitted = serde_json::json!({"input_ids": [1, 2]});
        let cases = [
            // Filled by the preferred sampling params.
            (
                preferred_args(),
                serde_json::json!({"input_ids": [1]}),
                vec![(Some(4096), true)],
            ),
            (
                preferred_args(),
                serde_json::json!({"input_ids": [1], "sampling_params": {"top_p": 0.5}}),
                vec![(Some(4096), true)],
            ),
            (
                args(Some(4096), Some(serde_json::json!({"max_new_tokens": 7}))),
                omitted.clone(),
                vec![(Some(7), true)],
            ),
            // Filled by the launch default, which may be no limit.
            (
                args(launch_default, None),
                serde_json::json!({"input_ids": [1]}),
                vec![(Some(128), true)],
            ),
            (
                args(Some(4096), None),
                omitted.clone(),
                vec![(Some(4096), true)],
            ),
            (args(None, None), omitted, vec![(None, true)]),
            // Named by the client, even as null.
            (
                preferred_args(),
                serde_json::json!({"input_ids": [1], "sampling_params": {"max_new_tokens": 8}}),
                vec![(Some(8), false)],
            ),
            (
                preferred_args(),
                serde_json::json!({"input_ids": [1], "sampling_params": {"max_new_tokens": null}}),
                vec![(None, false)],
            ),
            (
                args(Some(4096), None),
                serde_json::json!({
                    "input_ids": [1, 2],
                    "sampling_params": {"max_new_tokens": 3},
                }),
                vec![(Some(3), false)],
            ),
            // A batch: object by object, or one object for every item.
            (
                preferred_args(),
                serde_json::json!({
                    "input_ids": [[1], [2]],
                    "sampling_params": [{}, {"max_new_tokens": 8}],
                }),
                vec![(Some(4096), true), (Some(8), false)],
            ),
            (
                preferred_args(),
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

        let mut first_is_terminal = RequestTiming::new();
        first_is_terminal.observe_terminal_output();
        assert_eq!(
            first_is_terminal.decode_latency(),
            Some(Duration::ZERO),
            "a request that finishes in its first output has no decode interval"
        );
    }

    /// A model package's hook: the request's `top_p` on every frame, a
    /// top-level key on its first frame, and the decode latency on its last.
    /// It logs what it saw of each request's start and end.
    #[derive(Clone, Debug, Default)]
    struct TopPHook {
        log: Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
    }

    struct TopPFrames {
        top_p: f64,
        first: bool,
        /// What the hook saw at the start: the rid, a header, the submitted
        /// `top_k` and the client's own sampling params.
        started: serde_json::Value,
        log: Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
    }

    impl TopPFrames {
        fn logging(log: &Arc<std::sync::Mutex<Vec<serde_json::Value>>>) -> Box<Self> {
            Box::new(Self {
                top_p: 0.0,
                first: true,
                started: serde_json::Value::Null,
                log: log.clone(),
            })
        }
    }

    impl crate::GenerateHook for TopPHook {
        fn start(
            &self,
            request: &GenerateRequest,
            headers: &HeaderMap,
            client_sampling_params: Option<&serde_json::Map<String, serde_json::Value>>,
        ) -> Box<dyn GenerateFrameHook> {
            let mut frames = TopPFrames::logging(&self.log);
            frames.top_p = request.sampling_params.top_p;
            frames.started = serde_json::json!({
                "rid": request.rid.client_facing(),
                "header": headers.get("x-test").map(|v| v.to_str().unwrap()),
                "top_k": request.sampling_params.top_k,
                "client": client_sampling_params,
            });
            frames
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

        fn finished(&mut self, terminal: &GenerateTerminal<'_>) {
            self.log.lock().unwrap().push(serde_json::json!({
                "started": self.started,
                "completion_tokens": terminal.meta_info.completion_tokens,
                "queue_time": terminal.request_stats.map(|stats| stats["queue_time"].clone()),
                "decode_latency": terminal.decode_latency.map(|latency| latency.as_secs_f64()),
                "output_token_logprobs": terminal.output_token_logprobs,
            }));
        }

        fn failed(&mut self, status: StatusCode, message: &str) {
            self.log.lock().unwrap().push(serde_json::json!({
                "started": self.started,
                "failed": [status.as_u16(), message],
            }));
        }
    }

    /// Consumes `probe`; each frame reports the new values it saw as `seen`.
    #[derive(Debug)]
    struct ProbeConsumer;

    struct ProbeFrames;

    impl crate::GenerateHook for ProbeConsumer {
        fn start(
            &self,
            _: &GenerateRequest,
            _: &HeaderMap,
            _: Option<&serde_json::Map<String, serde_json::Value>>,
        ) -> Box<dyn GenerateFrameHook> {
            Box::new(ProbeFrames)
        }

        fn consumes_customized_info_key(&self, key: &str) -> bool {
            key == "probe"
        }
    }

    impl GenerateFrameHook for ProbeFrames {
        fn annotate(&mut self, frame: &GenerateFrameInfo<'_>, keys: &mut GenerateFrameKeys) {
            let seen = frame
                .customized_info
                .iter()
                .map(|&(key, values)| (key.to_owned(), serde_json::json!(values)))
                .collect();
            keys.meta_info
                .insert("seen".into(), serde_json::Value::Object(seen));
        }
    }

    /// One request's native frames for its runtime outputs, fed one at a
    /// time: a stream's, or a unary response's one.
    async fn native_frames(
        outputs: &[ChunkEvent],
        mode: &str,
        hook: bool,
    ) -> Vec<serde_json::Value> {
        let (tx, rx) = mpsc::channel(outputs.len());
        let (mut call, mut request) = timed_call(1, rx);
        if hook {
            request.hook = Some(RequestHook {
                hook: Arc::new(ProbeConsumer),
                frames: Box::new(ProbeFrames),
            });
        }
        let items = outputs.iter().cloned().map(|output| {
            if output.finish_reason.is_some() {
                ResponseItem::Done(output)
            } else {
                ResponseItem::Frame(output)
            }
        });
        if mode == "non_streaming" {
            for item in items {
                tx.send(item).await.unwrap();
            }
            let (_, item) = drain_unary(&mut call, "rid", request).await;
            return vec![serde_json::to_value(&item).unwrap()];
        }
        let stream = generation_event_stream(vec![(call, request)], mode == "incremental", false);
        futures::pin_mut!(stream);
        let mut frames = Vec::new();
        for item in items {
            tx.send(item).await.unwrap();
            frames.push(parse(&stream.next().await.unwrap()));
        }
        frames
    }

    /// Native frames report the scheduler's `customized_info` as Python's
    /// TokenizerManager does (`testdata/customized_info.json`, built by
    /// `build_rust_customized_info_fixture` from Python's frames): a
    /// cumulative stream's running lists of every key so far, even on a frame
    /// whose output lacked it (a scheduler abort), an incremental stream's own
    /// values of its output's keys, a unary response's terminal keys with all
    /// their values, in the scheduler's key order. Where Python's incremental
    /// slicing loses values (`python_incremental`), each frame reports its
    /// own. A hook that consumes a key hides it on every frame, and sees
    /// exactly the fixture's `hook_inputs`: each value once.
    #[tokio::test]
    async fn customized_info_matches_python() {
        use crate::message::response::{for_each_chunk, frame_decode_batch_cols};

        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../../testdata/customized_info.json")).unwrap();
        let bytes = |hex: &serde_json::Value| -> Vec<u8> {
            let hex = hex.as_str().unwrap();
            (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                .collect()
        };
        let mut outputs = std::collections::HashMap::<String, Vec<ChunkEvent>>::new();
        for frame in fixture["frames"].as_array().unwrap() {
            let framed =
                frame_decode_batch_cols(&bytes(&frame["header"]), &[&bytes(&frame["data"])]);
            let decoded = for_each_chunk(&framed[1..], |output| {
                let rid = output.rid.as_str().to_owned();
                outputs.entry(rid).or_default().push(output);
            });
            assert!(decoded.ok);
        }
        let entries =
            |object: &serde_json::Value, keys: &[&str]| -> Vec<(String, serde_json::Value)> {
                object
                    .as_object()
                    .unwrap()
                    .iter()
                    .filter(|(key, _)| keys.contains(&key.as_str()))
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect()
            };

        const KEYS: &[&str] = &["probe", "aux", "score"];
        for (rid, modes) in fixture["meta_info"].as_object().unwrap() {
            for (mode, expected) in modes.as_object().unwrap() {
                let expected = expected.as_array().unwrap();
                let case = format!("{rid} {mode}");
                let frames = native_frames(&outputs[rid], mode, false).await;
                assert_eq!(frames.len(), expected.len(), "{case}");
                for (frame, expected) in frames.iter().zip(expected) {
                    assert_eq!(
                        entries(&frame["meta_info"], KEYS),
                        entries(expected, KEYS),
                        "{case}"
                    );
                }

                let hook_inputs = fixture["hook_inputs"][rid][mode].as_array().unwrap();
                let frames = native_frames(&outputs[rid], mode, true).await;
                assert_eq!(frames.len(), hook_inputs.len(), "{case}");
                for (i, ((frame, expected), inputs)) in
                    frames.iter().zip(expected).zip(hook_inputs).enumerate()
                {
                    let meta = &frame["meta_info"];
                    assert!(meta.get("probe").is_none(), "{case}");
                    assert_eq!(
                        entries(meta, &["aux", "score"]),
                        entries(expected, &["aux", "score"]),
                        "{case}"
                    );
                    let inputs: serde_json::Map<String, serde_json::Value> = inputs
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|pair| (pair[0].as_str().unwrap().to_owned(), pair[1].clone()))
                        .collect();
                    let seen = meta["seen"].as_object().unwrap();
                    assert_eq!(
                        seen.keys().collect::<Vec<_>>(),
                        inputs.keys().collect::<Vec<_>>(),
                        "{case} {i}"
                    );
                    assert_eq!(seen, &inputs, "{case} {i}");
                }
            }
        }
    }

    /// A runtime output carrying `customized_info` (key, values), terminal
    /// when `finished`.
    fn customized(rid: u64, info: &[(&str, serde_json::Value)], finished: bool) -> ResponseItem {
        let customized_info = crate::message::response::CustomizedInfo(
            info.iter()
                .map(|(key, values)| (Arc::from(*key), values.as_array().unwrap().clone()))
                .collect(),
        );
        let ResponseItem::Done(mut output) = done(rid, "t") else {
            unreachable!()
        };
        if !finished {
            output.finish_reason = None;
        }
        output.extras = (!customized_info.is_empty()).then(|| {
            Box::new(crate::message::response::ChunkExtras {
                customized_info,
                ..Default::default()
            })
        });
        if finished {
            ResponseItem::Done(output)
        } else {
            ResponseItem::Frame(output)
        }
    }

    /// A cumulative stream that coalesces outputs still reports every key
    /// surfaced so far when the last output lacked it, but never one the hook
    /// consumes; the hook sees every value once, on the frame after it
    /// arrives.
    #[tokio::test]
    async fn coalesced_frames_catch_up_on_customized_info() {
        use serde_json::json;

        let (tx, rx) = mpsc::channel(8);
        let (call, mut request) = timed_call(1, rx);
        request.hook = Some(RequestHook {
            hook: Arc::new(ProbeConsumer),
            frames: Box::new(ProbeFrames),
        });
        let stream = generation_event_stream(vec![(call, request)], false, false);
        futures::pin_mut!(stream);

        // Both before the stream polls, the second without the keys.
        let first = [("probe", json!([1])), ("aux", json!(["x"]))];
        tx.send(customized(1, &first, false)).await.unwrap();
        tx.send(customized(1, &[], false)).await.unwrap();
        let meta = parse(&stream.next().await.unwrap())["meta_info"].take();
        assert_eq!(meta["aux"], json!(["x"]));
        assert!(meta.get("probe").is_none());
        assert_eq!(meta["seen"], json!({"probe": [1], "aux": ["x"]}));

        // A frame whose output lacks the keys: the hook sees them, with no new
        // values.
        tx.send(customized(1, &[], false)).await.unwrap();
        let meta = parse(&stream.next().await.unwrap())["meta_info"].take();
        assert_eq!(meta["aux"], json!(["x"]));
        assert!(meta.get("probe").is_none());
        assert_eq!(meta["seen"], json!({"probe": [], "aux": []}));

        let last = [("probe", json!([2])), ("aux", json!(["y"]))];
        tx.send(customized(1, &last, true)).await.unwrap();
        let meta = parse(&stream.next().await.unwrap())["meta_info"].take();
        assert_eq!(meta["aux"], json!(["x", "y"]));
        assert!(meta.get("probe").is_none());
        assert_eq!(meta["seen"], json!({"probe": [2], "aux": ["y"]}));
    }

    /// A `customized_info` key that repeats a key the schema, the request's
    /// stats or the hook writes gives way, so `meta_info` never has a key
    /// twice; the hook still sees it. `SCHEMA_META_INFO_KEYS` names every key
    /// the schema writes.
    #[tokio::test]
    async fn repeated_customized_info_keys_give_way() {
        use serde_json::json;

        let every = api::GenerateMetaInfo {
            id: String::new(),
            prompt_tokens: 0,
            completion_tokens: 0,
            finish_reason: None,
            output_token_logprobs: Some(Default::default()),
            input_token_logprobs: Some(Default::default()),
            output_top_logprobs: vec![Default::default()],
            input_top_logprobs: vec![Default::default()],
            output_token_ids_logprobs: vec![Default::default()],
            input_token_ids_logprobs: vec![Default::default()],
            hidden_states: vec![Default::default()],
            e2e_latency: Some(0.0),
            cached_tokens: 0,
            reasoning_tokens: 0,
            output_token_logprobs_length: Some(0),
            weight_version: Some(String::new()),
            weight_versions: vec![Default::default()],
            image_tokens: Some(1),
            audio_tokens: Some(1),
            video_tokens: Some(1),
            cached_tokens_details: Some(Default::default()),
            dp_rank: Some(0),
        };
        let keys: Vec<String> = serde_json::to_value(&every)
            .unwrap()
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(keys, SCHEMA_META_INFO_KEYS);

        let (tx, rx) = mpsc::channel(1);
        let (mut call, mut request) = timed_call(1, rx);
        let submitted = GenerateRequest {
            sampling_params: serde_json::from_value(json!({})).unwrap(),
            ..Default::default()
        };
        let hook = TopPHook::default();
        request.hook = Some(RequestHook {
            frames: hook.start(&submitted, &HeaderMap::new(), None),
            hook: Arc::new(hook),
        });
        let info = [
            ("id", json!([1])),
            ("top_p", json!([2])),
            ("queue_time", json!([4])),
            ("kept", json!([3])),
        ];
        let ResponseItem::Done(mut terminal) = customized(1, &info, true) else {
            unreachable!()
        };
        let stats = serde_json::Map::from_iter([("queue_time".into(), json!(0.5))]);
        terminal.extras.as_mut().unwrap().request_stats = Some(RequestStats(stats));
        tx.send(ResponseItem::Done(terminal)).await.unwrap();
        let (_, item) = drain_unary(&mut call, "rid", request).await;
        let text = json_string(&item);
        assert_eq!(text.matches(r#""id":"#).count(), 1, "{text}");
        assert_eq!(text.matches(r#""top_p":"#).count(), 1, "{text}");
        assert_eq!(text.matches(r#""queue_time":"#).count(), 1, "{text}");
        let meta = &parse(&text)["meta_info"];
        assert_eq!(meta["id"], "rid");
        assert_eq!(meta["top_p"], 1.0);
        assert_eq!(meta["queue_time"], 0.5);
        assert_eq!(meta["kept"], json!([3]));
    }

    /// As Python's `get_time_per_output_token`, a request reports no decode
    /// throughput without a second token or without a positive decode
    /// latency (its first output was also its last).
    #[test]
    fn decode_throughput_needs_two_tokens_and_a_decode_latency() {
        let latency = Some(Duration::from_millis(250));
        assert_eq!(decode_throughput(latency, 3), Some(8.0));
        assert_eq!(decode_throughput(latency, 1), None);
        assert_eq!(decode_throughput(Some(Duration::ZERO), 3), None);
        assert_eq!(decode_throughput(None, 3), None);
    }

    /// The serializer writes the generated one's keys, in its order, then a
    /// finished request's stats, then the hook's.
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
            request_stats: Some(RequestStats::default()),
            customized_info: ReportedCustomizedInfo::default(),
            keys: Some(Box::default()),
        };
        assert_eq!(json_string(&empty), expected);

        let mut keys = GenerateFrameKeys::default();
        keys.meta_info.insert("m".into(), serde_json::json!(1));
        keys.frame.insert("f".into(), serde_json::json!({"k": "v"}));
        let mut stats = serde_json::Map::new();
        stats.insert("num_retractions".into(), serde_json::json!(0));
        stats.insert("queue_time".into(), serde_json::json!(0.5));
        let added = NativeFrame {
            frame: schema,
            request_stats: Some(RequestStats(stats)),
            customized_info: ReportedCustomizedInfo::default(),
            keys: Some(Box::new(keys)),
        };
        let mut with_meta: serde_json::Value = parse(&expected);
        with_meta["meta_info"]["num_retractions"] = serde_json::json!(0);
        with_meta["meta_info"]["queue_time"] = serde_json::json!(0.5);
        with_meta["meta_info"]["m"] = serde_json::json!(1);
        with_meta["f"] = serde_json::json!({"k": "v"});
        assert_eq!(json_string(&added), with_meta.to_string());
    }

    /// A recording hook's annotator: every frame it was handed (without its
    /// `customized_info`), in order, each frame marked with its position as
    /// `meta_info.hook_frame`.
    #[derive(Clone, Debug, Default)]
    struct Recorded(Arc<std::sync::Mutex<Vec<GenerateFrameInfo<'static>>>>);

    impl Recorded {
        fn hook(&self) -> Option<RequestHook> {
            Some(RequestHook {
                hook: Arc::new(self.clone()),
                frames: Box::new(self.clone()),
            })
        }

        fn frames(&self) -> Vec<GenerateFrameInfo<'static>> {
            self.0.lock().unwrap().clone()
        }
    }

    impl crate::GenerateHook for Recorded {
        fn start(
            &self,
            _: &GenerateRequest,
            _: &HeaderMap,
            _: Option<&serde_json::Map<String, serde_json::Value>>,
        ) -> Box<dyn GenerateFrameHook> {
            Box::new(self.clone())
        }
    }

    impl GenerateFrameHook for Recorded {
        fn annotate(&mut self, frame: &GenerateFrameInfo<'_>, keys: &mut GenerateFrameKeys) {
            let mut frames = self.0.lock().unwrap();
            keys.meta_info
                .insert("hook_frame".into(), serde_json::json!(frames.len()));
            frames.push(GenerateFrameInfo {
                finished: frame.finished,
                decode_latency: frame.decode_latency,
                customized_info: &[],
            });
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

    /// A core failure before the stream starts is one SSE frame with Python's
    /// `type` and `retryable` for its kind; unary, it's the bare native body.
    #[tokio::test]
    async fn core_error_frames_name_type_and_retryable() {
        for (error, code, error_type, retryable) in [
            (CoreError::Unavailable, 503, "invalid_request_error", false),
            (
                CoreError::Internal("bug".into()),
                500,
                "internal_error",
                true,
            ),
        ] {
            let message = error.to_string();
            let streamed = native_core_error(error.clone(), true);
            assert_eq!(streamed.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(streamed.into_body(), 64 * 1024)
                .await
                .unwrap();
            let text = std::str::from_utf8(&bytes).unwrap();
            let frame = text
                .strip_prefix("data: ")
                .and_then(|rest| rest.strip_suffix("\n\ndata: [DONE]\n\n"))
                .unwrap_or_else(|| panic!("one error frame then [DONE]: {text}"));
            assert_eq!(
                parse(frame),
                serde_json::json!({"error": {
                    "message": message, "code": code, "type": error_type, "retryable": retryable,
                }})
            );

            let unary = native_core_error(error, false);
            assert_eq!(unary.status().as_u16(), code);
            let bytes = axum::body::to_bytes(unary.into_body(), 64 * 1024)
                .await
                .unwrap();
            assert_eq!(
                parse(std::str::from_utf8(&bytes).unwrap()),
                serde_json::json!({"error": {"message": message, "code": code}})
            );
        }
    }

    /// The Rust server answers the failures Python's handler answered in
    /// `testdata/error_frames.json` (built by `build_rust_error_frames_fixture`
    /// in `test_rust_server_error_frames.py`): the same stream frame, and the
    /// same unary status and body, which Rust extends with `code` only.
    #[tokio::test]
    async fn error_frames_match_python_fixture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../../testdata/error_frames.json")).unwrap();
        for (name, case) in fixture.as_object().unwrap() {
            let message = case["message"].as_str().unwrap().to_owned();
            let error = match name.as_str() {
                "invalid_argument" => CoreError::InvalidArgument(message),
                "status_503" => CoreError::Overloaded(message),
                other => panic!("no Rust failure for fixture case {other}"),
            };

            let streamed = native_core_error(error.clone(), true);
            assert_eq!(
                streamed.status().as_u16(),
                case["stream"]["status"],
                "{name}"
            );
            let bytes = axum::body::to_bytes(streamed.into_body(), 64 * 1024)
                .await
                .unwrap();
            let text = std::str::from_utf8(&bytes).unwrap();
            let frame = text
                .strip_prefix("data: ")
                .and_then(|rest| rest.strip_suffix("\n\ndata: [DONE]\n\n"))
                .unwrap_or_else(|| panic!("one error frame then [DONE]: {text}"));
            assert_eq!(parse(frame), case["stream"]["frame"], "{name}");

            let unary = native_core_error(error, false);
            assert_eq!(unary.status().as_u16(), case["unary"]["status"], "{name}");
            let bytes = axum::body::to_bytes(unary.into_body(), 64 * 1024)
                .await
                .unwrap();
            let mut body = parse(std::str::from_utf8(&bytes).unwrap());
            assert_eq!(body["error"]["code"], case["unary"]["status"], "{name}");
            body["error"].as_object_mut().unwrap().remove("code");
            assert_eq!(body, case["unary"]["body"], "{name}");
        }
    }

    /// A client that falls behind finds a backlog. Cumulative frames each
    /// supersede the last, so the backlog collapses to one frame carrying the
    /// latest state (the Python waiter's `out = out_list[-1]`); incremental
    /// frames are deltas, so every one is emitted.
    #[tokio::test]
    async fn a_backlog_coalesces_cumulative_frames_and_keeps_every_delta() {
        for incremental in [false, true] {
            let (tx, rx) = mpsc::channel(8);
            let calls = vec![timed_call(10, rx)];
            let stream = generation_event_stream(calls, incremental, false);
            futures::pin_mut!(stream);

            // Three chunks queued before the stream is ever polled.
            for text in ["a", "b", "c"] {
                tx.send(frame(10, text)).await.unwrap();
            }
            if incremental {
                for (n, expect) in [(1, "a"), (2, "b"), (3, "c")] {
                    let v = parse(&stream.next().await.unwrap());
                    assert_eq!(v["text"], expect, "delta {n} must not be dropped");
                    assert_eq!(
                        v["meta_info"]["completion_tokens"], n,
                        "count stays cumulative"
                    );
                }
                continue;
            }
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

    /// `json_body` accepts and rejects exactly what `Json<Value>` does, with
    /// the same rejection text, and hands back the bytes it was sent.
    #[tokio::test]
    async fn json_body_matches_the_json_extractor() {
        use axum::extract::FromRequest;

        let request = |content_type: Option<&str>, body: &'static str| {
            let mut builder = http::Request::builder().method("POST").uri("/generate");
            if let Some(content_type) = content_type {
                builder = builder.header(http::header::CONTENT_TYPE, content_type);
            }
            builder.body(axum::body::Body::from(body)).unwrap()
        };
        let body = "{ \"text\": \"hi\",  \"rid\": \"r\" }";
        for (content_type, body) in [
            (Some("application/json"), body),
            (Some("application/json; charset=utf-8"), body),
            (Some("application/vnd.api+json"), body),
            (Some("text/plain"), body),
            (None, body),
            (Some("application/json"), "{\"text\": "),
            (Some("application/json"), "[1, 2"),
        ] {
            let ours = json_body(&(), request(content_type, body)).await;
            let axums =
                Json::<serde_json::Value>::from_request(request(content_type, body), &()).await;
            match (ours, axums) {
                (Ok((value, raw)), Ok(Json(expected))) => {
                    assert_eq!(value, expected);
                    assert_eq!(raw, body.as_bytes(), "the bytes as sent");
                }
                (Err(ours), Err(axums)) => {
                    assert_eq!(ours.status(), axums.status(), "{content_type:?} {body}");
                    assert_eq!(ours.body_text(), axums.body_text());
                }
                (ours, axums) => panic!(
                    "{content_type:?} {body}: ours ok={}, axum ok={}",
                    ours.is_ok(),
                    axums.is_ok()
                ),
            }
        }
    }
}
