//! API server (axum / tokio). I/O-bound; runs on its own pinned multi-thread
//! runtime. Designed so additional protocols (h2/h3/websocket/grpc) can mount
//! the same `AppState` later — only this module knows about HTTP.
//!
//! `/generate` opens a per-request egress channel, moves a `Request` into the
//! ingress pipeline, and then either awaits a single `Done` (unary) or relays
//! frames as Server-Sent Events (streaming), byte-compatible with the Python
//! `http_server.generate_request` (`data: {json}\n\n` … `data: [DONE]\n\n`).
//! `/server_info` reuses the same submit machinery for a single control result.
mod openai;

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use std::convert::Infallible;
use tokio::sync::mpsc;

use crate::fsm::RequestState;
use crate::ids::{RequestId, RequestIdGen};

use crate::message::{
    ControlRequest, EgressItem, EgressSink, GeneratePayload, GenerateRequest, GenerationOutput,
    Request, RequestKind,
};
use crate::runtime::ServerArgs;
use crate::runtime::channels::{Senders, TmEvent};
use crate::tokenizer_manager::ActivityCounter;

/// Shared state for every handler. Holds the submit machinery (`senders`,
/// `id_gen`, `egress_buf`) and the shared tokenizer. `Clone` is a set of
/// refcount bumps (each field is `Arc`-backed), so axum's per-request clone is
/// cheap.
#[derive(Clone)]
struct AppState {
    senders: Senders,
    id_gen: Arc<RequestIdGen>,
    egress_buf: usize,
    server_args: Arc<ServerArgs>,
    /// Egress heartbeat (bumped per drained ring frame). `/health_generate`
    /// watches it advance to confirm the scheduler → detok path is alive.
    egress_activity: ActivityCounter,
    /// `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` (default true). When true,
    /// `/health` also runs the generation round-trip; when false it's plain 200.
    health_endpoint_generation: bool,
}

/// Parse `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION`, matching Python's `EnvBool`
/// (`true/1/yes/y` → true, `false/0/no/n` → false). Unset or unrecognized → the
/// `true` default (health generation on).
fn read_health_endpoint_generation() -> bool {
    match std::env::var("SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "false" | "0" | "no" | "n"
        ),
        Err(_) => true,
    }
}

pub async fn serve(
    listener: std::net::TcpListener,
    senders: Senders,
    id_gen: Arc<RequestIdGen>,
    egress_buf: usize,
    server_args: Arc<ServerArgs>,
    egress_activity: ActivityCounter,
    shutdown: flume::Receiver<()>,
) {
    let state = AppState {
        senders,
        id_gen,
        egress_buf,
        server_args,
        egress_activity,
        health_endpoint_generation: read_health_endpoint_generation(),
    };
    let app = Router::new()
        .route("/generate", post(generate))
        // Health: `/health` runs the generation round-trip by default
        // (`SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION`, else plain 200);
        // `/health_generate` always does. Mirrors the Python handler.
        .route("/health", get(health))
        .route("/health_generate", get(health_generate))
        // Control-plane endpoints: each reuses the ingress FSM (no tokenization)
        // and returns a single, non-streamed JSON result from the scheduler.
        // Adding one = a route line passing its scheduler request-struct tag.
        .route("/server_info", get(server_info))
        // Static config endpoints: no scheduler round-trip. `/get_model_info`
        // (+ its `/model_info` alias) is what the SGLang lang backend
        // (`RuntimeEndpoint`, used by the gsm8k/eval benchmarks) calls at
        // startup; `/v1/models` is OpenAI-compatible.
        .route("/get_model_info", get(model_info))
        .route("/model_info", get(model_info))
        .route("/v1/models", get(openai::available_models))
        .with_state(state);

    // The listener was already bound synchronously in `runtime::start` (so a port
    // conflict fails startup); adopt it into the tokio reactor here.
    let listener = match tokio::net::TcpListener::from_std(listener) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(error = %e, "failed to adopt pre-bound listener");
            return;
        }
    };
    if let Ok(addr) = listener.local_addr() {
        tracing::info!(%addr, "sglang-server api listening");
    }
    let serve = axum::serve(listener, app).with_graceful_shutdown(async move {
        let _ = shutdown.recv_async().await;
    });
    if let Err(e) = serve.await {
        tracing::error!(error = %e, "axum serve exited");
    }
}

/// Submit a request into the ingress pipeline. Returns the per-request egress
/// receiver to read the result(s) from. The `kind` carries the variant body
/// (generate payload / control tag), so this stays generic over both.
async fn submit(
    state: &AppState,
    kind: RequestKind,
) -> Result<(RequestId, mpsc::Receiver<EgressItem>), ()> {
    let id = state.id_gen.next();
    // Move the request into the in-process pipeline. Async-aware send so a full
    // TM inbox yields (backpressure) instead of parking a worker thread; Err only
    // when the inbox is closed (runtime shutdown).
    let (tx, rx) = mpsc::channel::<EgressItem>(state.egress_buf);
    let req = Request {
        id,
        state: RequestState::Received,
        sink: EgressSink::Local(tx),
        kind,
    };
    match state.senders.tm.send_async(TmEvent::Ingress(req)).await {
        Ok(()) => Ok((id, rx)),
        Err(_) => {
            tracing::error!("tm inbox closed; request dropped");
            Err(())
        }
    }
}

/// Aborts any still-in-flight request when dropped before normal completion —
/// i.e. axum dropped the handler future / SSE stream because the HTTP client
/// disconnected. Mirrors the Python `TokenizerManager` aborting on
/// `request.is_disconnected()`. Each rid is disarmed once it finishes naturally;
/// whatever is left when the guard drops gets an abort.
struct AbortGuard {
    senders: Senders,
    rids: Vec<RequestId>,
}

impl AbortGuard {
    fn new(senders: Senders, rid: RequestId) -> Self {
        Self {
            senders,
            rids: vec![rid],
        }
    }

    /// Request `rid` finished naturally — don't abort it on drop.
    fn disarm(&mut self, rid: RequestId) {
        self.rids.retain(|r| *r != rid);
    }
}

impl Drop for AbortGuard {
    fn drop(&mut self) {
        // Best-effort, non-blocking abort of each still-in-flight rid — route a
        // `TmEvent::Abort` to the ingress loop. A full/closed channel just drops
        // it (the request then finishes at EOS, only later).
        for &rid in &self.rids {
            let _ = self.senders.tm.try_send(TmEvent::Abort(rid));
        }
    }
}

/// Submit a `Control(tag)` request through the same ingress FSM as `/generate`
/// (minus tokenization) and await the scheduler's single msgpack `Result`. The
/// scheduler pushes a *named map* (`structs.asdict` of the response struct).
/// Returns the raw msgpack bytes, or an error `Response` to return as-is.
async fn await_control_result(
    state: &AppState,
    tag: &'static str,
) -> Result<bytes::Bytes, Response> {
    let (_id, mut rx) = submit(state, RequestKind::Control(ControlRequest { tag }))
        .await
        .map_err(|()| (StatusCode::SERVICE_UNAVAILABLE, "service unavailable").into_response())?;
    match rx.recv().await {
        Some(EgressItem::Control(bytes)) => Ok(bytes),
        Some(EgressItem::Error(e)) => {
            let code =
                StatusCode::from_u16(e.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            Err((code, e.to_string()).into_response())
        }
        // A control request never receives generation frames.
        Some(EgressItem::Frame(_)) | Some(EgressItem::Done(_)) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "unexpected generation output for control request",
        )
            .into_response()),
        None => Err((StatusCode::from_u16(499).unwrap(), "request aborted").into_response()),
    }
}

/// The text slot of a `[logprob, token_id, text]` tuple: the decoded token when
/// `return_text_in_logprobs` supplied a text buffer, else `null`.
fn text_slot(texts: Option<&[String]>, j: usize) -> serde_json::Value {
    texts
        .and_then(|t| t.get(j))
        .map(|s| serde_json::json!(s))
        .unwrap_or(serde_json::Value::Null)
}

/// A decoded-text column becomes the tuples' text source only when populated
/// (`return_text_in_logprobs`); empty → `None` → null text slots.
fn opt_texts(t: &[String]) -> Option<&[String]> {
    (!t.is_empty()).then_some(t)
}

/// The logprob slot of a tuple: a finite value, or `null` for the `NaN` sentinel.
fn lp_value(v: f32) -> serde_json::Value {
    if v.is_nan() {
        serde_json::Value::Null
    } else {
        serde_json::json!(v)
    }
}

/// Build the SGLang logprob wire shape: a list of `[logprob, token_id, text]`
/// tuples. `texts` (flat, parallel to `idxs`) fills the text slot when
/// `return_text_in_logprobs` is set; otherwise it is `null`.
fn logprob_tuples(vals: &[f32], idxs: &[i32], texts: Option<&[String]>) -> serde_json::Value {
    let tuples: Vec<serde_json::Value> = vals
        .iter()
        .zip(idxs.iter())
        .enumerate()
        .map(|(j, (&v, &tid))| serde_json::json!([lp_value(v), tid, text_slot(texts, j)]))
        .collect();
    serde_json::Value::Array(tuples)
}

/// Build the ragged top-k / token-ids logprob shape: one entry per position,
/// each a list of `[logprob, token_id, text]` tuples, or `null` for an empty
/// position (`lens[p] == 0`) — mirroring `detokenize_top_logprobs_tokens`.
/// `texts` is flat, parallel to `vals`/`idxs`.
fn ragged_logprob_tuples(
    vals: &[f32],
    idxs: &[i32],
    lens: &[u32],
    texts: Option<&[String]>,
) -> serde_json::Value {
    let mut positions = Vec::with_capacity(lens.len());
    let mut off = 0usize;
    for &l in lens {
        let l = l as usize;
        if l == 0 {
            positions.push(serde_json::Value::Null);
        } else {
            let tuples: Vec<serde_json::Value> = (off..off + l)
                .map(|j| serde_json::json!([lp_value(vals[j]), idxs[j], text_slot(texts, j)]))
                .collect();
            positions.push(serde_json::Value::Array(tuples));
        }
        off += l;
    }
    serde_json::Value::Array(positions)
}

/// Reshape flat hidden-state f32s + per-row lengths into `meta_info`'s nested
/// `list[list[float]]` (one row per output position). A single row of length 1
/// wins the common last-token case; multi-row is the per-token case.
fn hidden_states_rows(vals: &[f32], lens: &[u32]) -> serde_json::Value {
    let mut rows = Vec::with_capacity(lens.len());
    let mut off = 0usize;
    for &l in lens {
        let l = l as usize;
        rows.push(serde_json::json!(&vals[off..(off + l).min(vals.len())]));
        off += l;
    }
    serde_json::Value::Array(rows)
}

/// Map a terminal `abort` finish reason that carries a `status_code` (a request
/// error the scheduler produced.
fn abort_error_response(finish_reason: &Option<serde_json::Value>) -> Option<Response> {
    let fr = finish_reason.as_ref()?;
    if fr.get("type").and_then(|t| t.as_str()) != Some("abort") {
        return None;
    }
    let code = fr.get("status_code").and_then(|s| s.as_u64())? as u16;
    let message = fr
        .get("message")
        .and_then(|m| m.as_str())
        .unwrap_or("request aborted");
    let status = StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let body = serde_json::json!({ "error": { "message": message, "code": code } });
    Some((status, Json(body)).into_response())
}

/// Format a neutral [`GenerationOutput`] as one SGLang `/generate` frame.
fn sglang_frame(out: &GenerationOutput) -> Vec<u8> {
    let mut v = serde_json::json!({
        "text": out.text,
        "meta_info": {
            "id": out.rid,
            "prompt_tokens": out.prompt_tokens,
            "completion_tokens": out.completion_tokens,
            // Full dict (type + matched + message + status_code + …), or null.
            "finish_reason": out.finish_reason,
        },
    });
    // Logprobs: SGLang shape is a list of `[logprob, token_id, token_text|null]`
    // tuples. The token text (`return_text_in_logprobs`) was decoded on the detok
    // shard into the `*_txt` columns; empty → null text slot.
    if !out.output_ids.is_empty() {
        v["output_ids"] = serde_json::json!(out.output_ids);
    }
    if !out.out_lp_val.is_empty() {
        v["meta_info"]["output_token_logprobs"] =
            logprob_tuples(&out.out_lp_val, &out.out_lp_idx, opt_texts(&out.out_lp_txt));
    }
    if !out.in_lp_val.is_empty() {
        v["meta_info"]["input_token_logprobs"] =
            logprob_tuples(&out.in_lp_val, &out.in_lp_idx, opt_texts(&out.in_lp_txt));
    }
    if !out.out_top_lens.is_empty() {
        v["meta_info"]["output_top_logprobs"] = ragged_logprob_tuples(
            &out.out_top_val,
            &out.out_top_idx,
            &out.out_top_lens,
            opt_texts(&out.out_top_txt),
        );
    }
    if !out.in_top_lens.is_empty() {
        v["meta_info"]["input_top_logprobs"] = ragged_logprob_tuples(
            &out.in_top_val,
            &out.in_top_idx,
            &out.in_top_lens,
            opt_texts(&out.in_top_txt),
        );
    }
    if !out.out_tid_lens.is_empty() {
        v["meta_info"]["output_token_ids_logprobs"] = ragged_logprob_tuples(
            &out.out_tid_val,
            &out.out_tid_idx,
            &out.out_tid_lens,
            opt_texts(&out.out_tid_txt),
        );
    }
    if !out.in_tid_lens.is_empty() {
        v["meta_info"]["input_token_ids_logprobs"] = ragged_logprob_tuples(
            &out.in_tid_val,
            &out.in_tid_idx,
            &out.in_tid_lens,
            opt_texts(&out.in_tid_txt),
        );
    }
    if !out.hidden_lens.is_empty() {
        v["meta_info"]["hidden_states"] = hidden_states_rows(&out.hidden_val, &out.hidden_lens);
    }
    serde_json::to_vec(&v).unwrap_or_default()
}

/// Generic control endpoint: returns the scheduler's response rendered straight
/// to JSON (`tag` = the scheduler request-struct name). Used by control
/// endpoints whose response needs no shaping.
#[allow(dead_code)] // first non-/server_info control endpoint will use this
async fn control(State(state): State<AppState>, tag: &'static str) -> Response {
    match await_control_result(&state, tag).await {
        Ok(bytes) => match msgpack_to_json(&bytes) {
            Ok(json) => {
                (StatusCode::OK, [("content-type", "application/json")], json).into_response()
            }
            Err(e) => {
                tracing::error!(error = %e, "control: msgpack→json failed");
                (StatusCode::INTERNAL_SERVER_ERROR, "bad control response").into_response()
            }
        },
        Err(resp) => resp,
    }
}

/// `GET /health` — liveness. By default (`SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION`
/// True, mirroring Python) this runs the same 1-token round-trip as
/// `/health_generate`; when the env is set false it returns a plain 200 (the
/// api-server routing the request already proves the frontend is up).
async fn health(State(state): State<AppState>) -> Response {
    if state.health_endpoint_generation {
        health_generate_inner(&state).await
    } else {
        StatusCode::OK.into_response()
    }
}

/// `GET /health_generate` — always the deep round-trip, regardless of the env.
async fn health_generate(State(state): State<AppState>) -> Response {
    health_generate_inner(&state).await
}

/// Deep health: confirm the scheduler → detok path is actually producing output.
/// Returns 200 iff the egress heartbeat advances within the timeout, else 503.
///
/// It fires a pre-tokenized 1-token probe (`input_ids = [0]`, which skips the
/// tokenizer via `classify → AlreadyTokenized`) so an **idle** pipeline produces
/// a frame, then watches the **global** [`AppState::egress_activity`] counter
/// (not the probe's own rid). So a **busy** server passes immediately on
/// concurrent traffic and a backlogged queue never causes a false 503 — the
/// rust-native analogue of the Python health check's `last_receive_tstamp`. The
/// scheduler's `HEALTH_CHECK`-rid skip and `http_worker_ipc` ack are irrelevant
/// here: those exist for the multi-tokenizer-*process* setup, whereas this
/// single-process server owns the egress ring and observes activity directly.
async fn health_generate_inner(state: &AppState) -> Response {
    let baseline = state
        .egress_activity
        .load(std::sync::atomic::Ordering::Relaxed);

    // Fire the probe (we do not await its own response; the heartbeat is the
    // signal). One decode step, so it self-completes with nothing to abort.
    let sampling_params = rmpv::Value::Map(vec![
        (rmpv::Value::from("max_new_tokens"), rmpv::Value::from(1)),
        (rmpv::Value::from("temperature"), rmpv::Value::F64(0.0)),
    ]);
    let payload = GeneratePayload {
        input_ids: Some(vec![0]),
        sampling_params: Some(sampling_params),
        ..Default::default()
    };
    let kind = RequestKind::Generate(GenerateRequest {
        payload,
        input_ids: None,
        stream: false,
        // The scheduler skips this when busy so it never occupies a queue slot.
        is_health_check: true,
    });
    let _keepalive = match submit(state, kind).await {
        // Hold the receiver so the probe's sink stays open until it completes.
        Ok((_rid, rx)) => rx,
        Err(()) => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
    };

    // Watch the heartbeat advance. `SGLANG_HEALTH_CHECK_TIMEOUT` defaults to 20s.
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        if state
            .egress_activity
            .load(std::sync::atomic::Ordering::Relaxed)
            != baseline
        {
            return StatusCode::OK.into_response();
        }
        if tokio::time::Instant::now() >= deadline {
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

/// `GET /get_model_info` (+ `/model_info` alias) — static model metadata read
/// from `server_args`; no scheduler round-trip, mirroring `/v1/models`. The
/// SGLang lang backend (`RuntimeEndpoint`) calls this at startup and only reads
/// `model_path` (for chat-template detection); the gsm8k/eval benchmark scripts
/// go through it. `is_generation` is always true — this server is generation
/// only.
async fn model_info(State(state): State<AppState>) -> Response {
    let sa = &state.server_args;
    let body = serde_json::json!({
        "model_path": sa.model_path(),
        "tokenizer_path": sa.tokenizer_path(),
        "is_generation": true,
        "preferred_sampling_params": serde_json::Value::Null,
        "weight_version": serde_json::Value::Null,
    });
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        serde_json::to_vec(&body).unwrap_or_default(),
    )
        .into_response()
}

/// `GET /server_info` — shapes the scheduler's `GetInternalStateReqOutput` the
/// way `tokenizer_control_mixin.get_internal_state` does: lift `server_args` to
/// the top, drop null fields, merge (internal-state fields win on collision).
///
/// TODO(server_info): the original Python endpoint also includes `version`,
/// `kv_events`, and scheduler init info (`max_total_num_tokens`,
/// `max_req_input_len`). Those are dropped for now — add them here once the
/// values are plumbed through (e.g. captured at `Server.start` / a richer
/// scheduler response).
async fn server_info(State(state): State<AppState>) -> Response {
    let bytes = match await_control_result(&state, "GetInternalStateReq").await {
        Ok(b) => b,
        Err(resp) => return resp,
    };
    match shape_server_info(&bytes) {
        Ok(json) => (StatusCode::OK, [("content-type", "application/json")], json).into_response(),
        Err(e) => {
            tracing::error!(error = %e, "server_info: shaping failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "bad server_info response",
            )
                .into_response()
        }
    }
}

fn shape_server_info(msgpack: &[u8]) -> Result<Vec<u8>, String> {
    // Decode the msgpack named map directly into a JSON object.
    let mut obj: serde_json::Map<String, serde_json::Value> =
        rmp_serde::from_slice(msgpack).map_err(|e| e.to_string())?;

    // server_args = res.pop("server_args", {})
    let mut merged = match obj.remove("server_args") {
        Some(serde_json::Value::Object(m)) => m,
        _ => serde_json::Map::new(),
    };
    // res = {k: v for k, v in res.items() if v is not None}; merged = server_args | res
    for (k, v) in obj {
        if !v.is_null() {
            merged.insert(k, v);
        }
    }

    let response = serde_json::json!({ "internal_states": [serde_json::Value::Object(merged)] });
    serde_json::to_vec(&response).map_err(|e| e.to_string())
}

/// Convert a msgpack control response (the scheduler's native ring format) into
/// JSON bytes for the HTTP client.
fn msgpack_to_json(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let val = rmpv::decode::read_value(&mut &*bytes).map_err(|e| e.to_string())?;
    serde_json::to_vec(&val).map_err(|e| e.to_string())
}

/// Folds the per-chunk [`GenerationOutput`] deltas the detok emits into a
/// cumulative view. Used by the drain loops that need cumulative output — every
/// unary response and the cumulative SGLang `/generate` stream. OpenAI streaming
/// forwards deltas directly and doesn't use this. Shared with the [`openai`]
/// submodule (`super::OutputAccumulator`).
#[derive(Default)]
struct OutputAccumulator {
    text: String,
    output_ids: Vec<i32>,
    prompt_tokens: u32,
    completion_tokens: u64,
    finish_reason: Option<serde_json::Value>,
    /// Output-token logprobs, concatenated across chunks (parallel val/idx).
    out_lp_val: Vec<f32>,
    out_lp_idx: Vec<i32>,
    /// Input (prefill) token logprobs — set once (only the first chunk carries
    /// them).
    in_lp_val: Vec<f32>,
    in_lp_idx: Vec<i32>,
    /// Top-k logprobs, concatenated across chunks (2-level ragged: flat val/idx
    /// + per-position lens). Output concatenates; input is set once.
    out_top_val: Vec<f32>,
    out_top_idx: Vec<i32>,
    out_top_lens: Vec<u32>,
    in_top_val: Vec<f32>,
    in_top_idx: Vec<i32>,
    in_top_lens: Vec<u32>,
    /// Token-ids logprobs (same layout).
    out_tid_val: Vec<f32>,
    out_tid_idx: Vec<i32>,
    out_tid_lens: Vec<u32>,
    in_tid_val: Vec<f32>,
    in_tid_idx: Vec<i32>,
    in_tid_lens: Vec<u32>,
    /// Hidden states — last-writer-wins (final chunk carries the full set).
    hidden_val: Vec<f32>,
    hidden_lens: Vec<u32>,
    /// Decoded logprob token text (parallel to each `*_idx`); output concatenates,
    /// input is set once, same as the val/idx columns.
    out_lp_txt: Vec<String>,
    in_lp_txt: Vec<String>,
    out_top_txt: Vec<String>,
    in_top_txt: Vec<String>,
    out_tid_txt: Vec<String>,
    in_tid_txt: Vec<String>,
}

impl OutputAccumulator {
    /// Fold one delta frame in.
    fn fold(&mut self, d: &GenerationOutput) {
        self.text.push_str(&d.text);
        self.output_ids.extend_from_slice(&d.output_ids);
        self.completion_tokens += d.completion_tokens;
        self.prompt_tokens = d.prompt_tokens; // constant across the request
        self.out_lp_val.extend_from_slice(&d.out_lp_val);
        self.out_lp_idx.extend_from_slice(&d.out_lp_idx);
        self.out_top_val.extend_from_slice(&d.out_top_val);
        self.out_top_idx.extend_from_slice(&d.out_top_idx);
        self.out_top_lens.extend_from_slice(&d.out_top_lens);
        self.out_tid_val.extend_from_slice(&d.out_tid_val);
        self.out_tid_idx.extend_from_slice(&d.out_tid_idx);
        self.out_tid_lens.extend_from_slice(&d.out_tid_lens);
        self.out_lp_txt.extend_from_slice(&d.out_lp_txt);
        self.out_top_txt.extend_from_slice(&d.out_top_txt);
        self.out_tid_txt.extend_from_slice(&d.out_tid_txt);
        if !d.in_lp_val.is_empty() {
            self.in_lp_val = d.in_lp_val.clone();
            self.in_lp_idx = d.in_lp_idx.clone();
            self.in_lp_txt = d.in_lp_txt.clone();
        }
        // Input families ride once (prefill); `lens` non-empty marks their arrival.
        if !d.in_top_lens.is_empty() {
            self.in_top_val = d.in_top_val.clone();
            self.in_top_idx = d.in_top_idx.clone();
            self.in_top_lens = d.in_top_lens.clone();
            self.in_top_txt = d.in_top_txt.clone();
        }
        if !d.in_tid_lens.is_empty() {
            self.in_tid_val = d.in_tid_val.clone();
            self.in_tid_idx = d.in_tid_idx.clone();
            self.in_tid_lens = d.in_tid_lens.clone();
            self.in_tid_txt = d.in_tid_txt.clone();
        }
        // Hidden states are non-cumulative: the latest non-empty set wins.
        if !d.hidden_lens.is_empty() {
            self.hidden_val = d.hidden_val.clone();
            self.hidden_lens = d.hidden_lens.clone();
        }
        if d.finish_reason.is_some() {
            self.finish_reason = d.finish_reason.clone();
        }
    }

    /// Cumulative snapshot for an intermediate streaming frame (clones — a
    /// cumulative protocol like SGLang `/generate` needs the full text per frame).
    fn snapshot(&self, rid: &str) -> GenerationOutput {
        GenerationOutput {
            rid: rid.to_string(),
            text: self.text.clone(),
            output_ids: self.output_ids.clone(),
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            finish_reason: self.finish_reason.clone(),
            out_lp_val: self.out_lp_val.clone(),
            out_lp_idx: self.out_lp_idx.clone(),
            in_lp_val: self.in_lp_val.clone(),
            in_lp_idx: self.in_lp_idx.clone(),
            out_top_val: self.out_top_val.clone(),
            out_top_idx: self.out_top_idx.clone(),
            out_top_lens: self.out_top_lens.clone(),
            in_top_val: self.in_top_val.clone(),
            in_top_idx: self.in_top_idx.clone(),
            in_top_lens: self.in_top_lens.clone(),
            out_tid_val: self.out_tid_val.clone(),
            out_tid_idx: self.out_tid_idx.clone(),
            out_tid_lens: self.out_tid_lens.clone(),
            in_tid_val: self.in_tid_val.clone(),
            in_tid_idx: self.in_tid_idx.clone(),
            in_tid_lens: self.in_tid_lens.clone(),
            hidden_val: self.hidden_val.clone(),
            hidden_lens: self.hidden_lens.clone(),
            out_lp_txt: self.out_lp_txt.clone(),
            in_lp_txt: self.in_lp_txt.clone(),
            out_top_txt: self.out_top_txt.clone(),
            in_top_txt: self.in_top_txt.clone(),
            out_tid_txt: self.out_tid_txt.clone(),
            in_tid_txt: self.in_tid_txt.clone(),
        }
    }

    /// Consume into the final cumulative output (moves; for a unary response or a
    /// stream's final frame).
    fn into_output(self, rid: String) -> GenerationOutput {
        GenerationOutput {
            rid,
            text: self.text,
            output_ids: self.output_ids,
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            finish_reason: self.finish_reason,
            out_lp_val: self.out_lp_val,
            out_lp_idx: self.out_lp_idx,
            in_lp_val: self.in_lp_val,
            in_lp_idx: self.in_lp_idx,
            out_top_val: self.out_top_val,
            out_top_idx: self.out_top_idx,
            out_top_lens: self.out_top_lens,
            in_top_val: self.in_top_val,
            in_top_idx: self.in_top_idx,
            in_top_lens: self.in_top_lens,
            out_tid_val: self.out_tid_val,
            out_tid_idx: self.out_tid_idx,
            out_tid_lens: self.out_tid_lens,
            in_tid_val: self.in_tid_val,
            in_tid_idx: self.in_tid_idx,
            in_tid_lens: self.in_tid_lens,
            hidden_val: self.hidden_val,
            hidden_lens: self.hidden_lens,
            out_lp_txt: self.out_lp_txt,
            in_lp_txt: self.in_lp_txt,
            out_top_txt: self.out_top_txt,
            in_top_txt: self.in_top_txt,
            out_tid_txt: self.out_tid_txt,
            in_tid_txt: self.in_tid_txt,
        }
    }
}

async fn generate(State(state): State<AppState>, Json(payload): Json<GeneratePayload>) -> Response {
    let stream = payload.stream;
    // `return_text_in_logprobs` is decoded on the detok shard (see
    // DetokMsg::Register.decode_logprob_text) into the `*_txt` columns, so
    // `sglang_frame` just reads them — no tokenizer needed here.
    let kind = RequestKind::Generate(GenerateRequest {
        payload,
        input_ids: None,
        stream,
        is_health_check: false,
    });
    let (rid, mut rx) = match submit(&state, kind).await {
        Ok(v) => v,
        Err(()) => {
            return (StatusCode::SERVICE_UNAVAILABLE, "service unavailable").into_response();
        }
    };
    // Abort the request if the client disconnects: the guard fires `try_abort`
    // when dropped before the request finishes — i.e. axum drops this handler /
    // SSE stream because the connection closed. Disarmed on a natural terminal.
    let mut guard = AbortGuard::new(state.senders.clone(), rid);

    if stream {
        // SSE: the SGLang `/generate` protocol carries **cumulative** text per
        // frame, so fold the detok deltas back up before formatting each frame.
        let s = async_stream::stream! {
            let mut acc = OutputAccumulator::default();
            while let Some(item) = rx.recv().await {
                match item {
                    EgressItem::Frame(out) => {
                        acc.fold(&out);
                        let f = sglang_frame(&acc.snapshot(&out.rid));
                        yield Ok::<_, Infallible>(Event::default().data(String::from_utf8_lossy(&f)));
                    }
                    EgressItem::Done(out) => {
                        acc.fold(&out);
                        let f = sglang_frame(&acc.into_output(out.rid));
                        yield Ok(Event::default().data(String::from_utf8_lossy(&f)));
                        break;
                    }
                    EgressItem::Error(e) => {
                        let body = serde_json::json!({
                            "error": { "message": e.to_string(), "code": e.http_status() }
                        });
                        yield Ok(Event::default().data(body.to_string()));
                        break;
                    }
                    // Control results never arrive on a `/generate` request.
                    EgressItem::Control(_) => break,
                }
            }
            // Reached only on a terminal / closed channel (not a disconnect, which
            // drops the suspended generator before here) — so the request is done.
            guard.disarm(rid);
            yield Ok(Event::default().data("[DONE]"));
        };
        Sse::new(s).into_response()
    } else {
        // Unary: fold every delta, respond once from the cumulative result.
        let mut acc = OutputAccumulator::default();
        while let Some(item) = rx.recv().await {
            match item {
                EgressItem::Frame(out) => acc.fold(&out),
                EgressItem::Done(out) => {
                    acc.fold(&out);
                    guard.disarm(rid);
                    let final_out = acc.into_output(out.rid);
                    // A validation abort carries its own HTTP status + diagnostic.
                    if let Some(resp) = abort_error_response(&final_out.finish_reason) {
                        return resp;
                    }
                    return (
                        StatusCode::OK,
                        [("content-type", "application/json")],
                        sglang_frame(&final_out),
                    )
                        .into_response();
                }
                EgressItem::Error(e) => {
                    guard.disarm(rid);
                    let code = StatusCode::from_u16(e.http_status())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    let body = serde_json::json!({
                        "error": { "message": e.to_string(), "code": e.http_status() }
                    });
                    return (code, Json(body)).into_response();
                }
                EgressItem::Control(_) => continue, // never on `/generate`
            }
        }
        // Sender dropped without a terminal item → request already gone.
        guard.disarm(rid);
        (StatusCode::from_u16(499).unwrap(), "request aborted").into_response()
    }
}

#[cfg(test)]
mod logprob_shape_tests {
    use super::*;

    #[test]
    fn flat_logprob_tuples_shape() {
        let v = logprob_tuples(&[-0.5, -1.5], &[10, 20], None);
        assert_eq!(
            v,
            serde_json::json!([
                [-0.5f32, 10, serde_json::Value::Null],
                [-1.5f32, 20, serde_json::Value::Null]
            ])
        );
    }

    /// With a text buffer, the tuple's third slot carries the decoded token.
    #[test]
    fn flat_logprob_tuples_with_text() {
        let texts = vec!["a".to_string(), "b".to_string()];
        let v = logprob_tuples(&[-0.5, -1.5], &[10, 20], Some(&texts));
        assert_eq!(
            v,
            serde_json::json!([[-0.5f32, 10, "a"], [-1.5f32, 20, "b"]])
        );
    }

    /// Ragged reshape restores null positions (len 0) — mirrors
    /// detokenize_top_logprobs_tokens emitting None for empty positions.
    #[test]
    fn ragged_logprob_tuples_restores_null_positions() {
        // 2 positions: first null (len 0), second k=1.
        let v = ragged_logprob_tuples(&[-0.3], &[9], &[0, 1], None);
        assert_eq!(
            v,
            serde_json::json!([
                serde_json::Value::Null,
                [[-0.3f32, 9, serde_json::Value::Null]]
            ])
        );
    }

    /// The `NaN` sentinel (the Python `None` logprob for the first prompt token)
    /// becomes a JSON `null` logprob, while its token id in the parallel `idx`
    /// column is preserved. Guards the scheduler-killing prompt-logprob crash.
    #[test]
    fn nan_sentinel_becomes_null_logprob() {
        // Flat (input/output logprobs): first value absent, second present.
        let flat = logprob_tuples(&[f32::NAN, -0.5], &[10, 20], None);
        assert_eq!(
            flat,
            serde_json::json!([
                [serde_json::Value::Null, 10, serde_json::Value::Null],
                [-0.5f32, 20, serde_json::Value::Null],
            ])
        );
        // Ragged (top-k / token-ids logprobs): a NaN inside a position → null.
        let ragged = ragged_logprob_tuples(&[f32::NAN], &[7], &[1], None);
        assert_eq!(
            ragged,
            serde_json::json!([[[serde_json::Value::Null, 7, serde_json::Value::Null]]])
        );
    }

    /// End-to-end: a `GenerationOutput` carrying a prompt-logprob request (first
    /// input logprob is the `NaN` sentinel) formats without panicking and emits
    /// `input_token_logprobs` with a leading `[null, token_id, text]`.
    #[test]
    fn prompt_logprob_frame_emits_null_first() {
        let out = GenerationOutput {
            rid: "1".into(),
            in_lp_val: vec![f32::NAN, -0.5],
            in_lp_idx: vec![10, 20],
            in_lp_txt: vec!["<s>".into(), "hi".into()],
            ..Default::default()
        };
        let frame: serde_json::Value = serde_json::from_slice(&sglang_frame(&out)).unwrap();
        assert_eq!(
            frame["meta_info"]["input_token_logprobs"],
            serde_json::json!([[serde_json::Value::Null, 10, "<s>"], [-0.5f32, 20, "hi"]])
        );
    }

    /// A populated text column (decoded on the detok shard) → `Some`; empty
    /// (`return_text_in_logprobs` off) → `None` → null text slots.
    #[test]
    fn opt_texts_gates_on_population() {
        assert!(opt_texts(&[]).is_none());
        let t = vec!["x".to_string()];
        assert_eq!(opt_texts(&t), Some(t.as_slice()));
    }

    /// A validation abort (`status_code` set) maps to that HTTP status — the
    /// over-context request must not come back as a 200.
    #[test]
    fn validation_abort_maps_to_its_status() {
        let fr = Some(serde_json::json!({
            "type": "abort", "message": "over the 40954-token limit", "status_code": 400
        }));
        let resp = abort_error_response(&fr).expect("abort with status → error response");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// A normal finish, a bare abort (no status), and no finish all stay 200.
    #[test]
    fn non_error_finishes_stay_ok() {
        assert!(
            abort_error_response(&Some(serde_json::json!({"type": "stop", "matched": 5})))
                .is_none()
        );
        assert!(
            abort_error_response(&Some(serde_json::json!({"type": "length", "length": 8})))
                .is_none()
        );
        assert!(
            abort_error_response(&Some(
                serde_json::json!({"type": "abort", "message": "Aborted"})
            ))
            .is_none()
        );
        assert!(abort_error_response(&None).is_none());
    }
}
