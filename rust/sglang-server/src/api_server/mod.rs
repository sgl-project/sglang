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
    ChunkEvent, ControlRequest, EgressItem, EgressSink, GenerateBody, GeneratePayload,
    GenerateRequest, Request, RequestKind,
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
        // TODO(auth): no API-key boundary yet. The Python server gates every route
        // (except /health*, /metrics*, OPTIONS) with `add_api_key_middleware`
        // (`api_key` / `admin_api_key`); until that is ported here, a configured
        // `api_key` does NOT protect these routes.
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
    // Non-graceful shutdown: on the signal, stop accepting and RETURN — do NOT
    // wait for in-flight handlers (a `/generate` blocked on its egress channel
    // would never complete, wedging the join). Returning drops `serve` (stops
    // accepts) and unwinds `block_on` in `runtime::start`, so the api thread's
    // tokio runtime is dropped — which cancels the detached in-flight handler
    // tasks. Each cancelled handler's `AbortGuard` fires, releasing its `Senders`
    // clone; once every clone is gone the detok/tok channels close and those
    // workers exit. Full graceful drain is deferred (see `request_shutdown`).
    let serve = axum::serve(listener, app);
    tokio::select! {
        r = serve => {
            if let Err(e) = r {
                tracing::error!(error = %e, "axum serve exited");
            }
        }
        _ = shutdown.recv_async() => {
            tracing::info!("shutdown: stopping accepts, aborting in-flight handlers");
        }
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

    /// Guard covering no rids yet — a batch arms each as it's submitted so a
    /// mid-fan-out disconnect aborts every request already handed to the scheduler.
    fn new_empty(senders: Senders) -> Self {
        Self {
            senders,
            rids: Vec::new(),
        }
    }

    /// Track `rid` for abort-on-drop.
    fn arm(&mut self, rid: RequestId) {
        self.rids.push(rid);
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

/// Classify a terminal finish reason: `Some((code, message))` when it's an
/// `abort` carrying a `status_code` (a request error the scheduler produced, e.g.
/// an over-context request → 400). The status lives only inside `meta_info`, so
/// both the unary and streaming paths must inspect it rather than treat the
/// `Done` frame as a normal completion.
fn abort_status(finish_reason: &Option<serde_json::Value>) -> Option<(u16, String)> {
    let fr = finish_reason.as_ref()?;
    if fr.get("type").and_then(|t| t.as_str()) != Some("abort") {
        return None;
    }
    let code = fr.get("status_code").and_then(|s| s.as_u64())? as u16;
    let message = fr
        .get("message")
        .and_then(|m| m.as_str())
        .unwrap_or("request aborted")
        .to_string();
    Some((code, message))
}

/// The `{ "error": { message, code } }` object every error path emits (an SSE
/// event's data, a unary body, or one entry of a batch array).
fn error_value(code: u16, message: &str) -> serde_json::Value {
    serde_json::json!({ "error": { "message": message, "code": code } })
}

/// Format a decoded [`ChunkEvent`] as one SGLang `/generate` frame's JSON. `rid`
/// (the response `meta_info.id`) is passed in as its string form — the api-server
/// owns the canonical id; the event's numeric `rid` is just the shard routing key.
fn sglang_frame_value(out: &ChunkEvent, rid: &str) -> serde_json::Value {
    let mut v = serde_json::json!({
        "text": out.text,
        "meta_info": {
            "id": rid,
            "prompt_tokens": out.prompt_tokens,
            "completion_tokens": out.completion_tokens,
            // Full dict (type + matched + message + status_code + …), or null.
            "finish_reason": out.finish_reason,
        },
    });
    // Logprobs: SGLang shape is a list of `[logprob, token_id, token_text|null]`
    // tuples. The token text (`return_text_in_logprobs`) was decoded on the detok
    // shard into the `*_txt` columns; empty → null text slot.
    if !out.token_ids.is_empty() {
        v["output_ids"] = serde_json::json!(out.token_ids);
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
    v
}

/// Format one streaming frame. Non-incremental (SGLang default): the accumulator's
/// **cumulative** view. Incremental (`incremental_streaming_output`): this step's
/// **delta** `text`/`output_ids`/logprobs, but with the cumulative token count in
/// `meta_info` (matching the Python `TokenizerManager`). `delta` is this chunk's
/// event; `acc` is the cumulative fold used for both the cumulative view and the
/// running `completion_tokens`.
fn stream_frame_value(
    delta: ChunkEvent,
    acc: &OutputAccumulator,
    incremental: bool,
    rid_str: &str,
) -> serde_json::Value {
    if incremental {
        let mut d = delta;
        d.completion_tokens = acc.snapshot().completion_tokens;
        sglang_frame_value(&d, rid_str)
    } else {
        sglang_frame_value(acc.snapshot(), rid_str)
    }
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
    // signal). We must NOT assume it self-completes: a busy scheduler skips the
    // health request without emitting any terminal frame, so its detok
    // registration is cleaned up only by the `AbortGuard` below.
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
    let (rid, _keepalive) = match submit(state, kind).await {
        // Hold the receiver so the probe's sink stays open until it completes.
        Ok((rid, rx)) => (rid, rx),
        Err(()) => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
    };
    // Deregister on drop (never disarmed): a busy-skipped probe has no terminal
    // frame, so without this abort it leaks one detok entry per call.
    let _abort_guard = AbortGuard::new(state.senders.clone(), rid);

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

/// `GET /get_model_info` (+ `/model_info` alias) — static model metadata from
/// `server_args` (no scheduler round-trip); `is_generation` always true.
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

/// `GET /server_info` — surface only an allowlist ([`INTERNAL_STATE_ALLOWLIST`] +
/// curated [`ServerArgs`] accessors), never the scheduler's raw server-args dump,
/// which embeds `api_key`/`admin_api_key` (see [`shape_server_info`]).
///
/// TODO(server_info): the Python endpoint also includes `version` and `kv_events`;
/// add them once plumbed through (captured at `Server.start` / a richer response).
async fn server_info(State(state): State<AppState>) -> Response {
    let bytes = match await_control_result(&state, "GetInternalStateReq").await {
        Ok(b) => b,
        Err(resp) => return resp,
    };
    match shape_server_info(&bytes, &state.server_args) {
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

/// Runtime-metric keys the scheduler's `get_internal_state` adds on top of the
/// server-args dump. We copy ONLY these out of `internal_state` — an allowlist, so
/// the co-mingled `api_key`/`admin_api_key` (and any secret added later) can never
/// reach the response.
const INTERNAL_STATE_ALLOWLIST: &[&str] = &[
    "last_gen_throughput",
    "memory_usage",
    "effective_max_running_requests_per_dp",
    "avg_spec_accept_length",
    "step_time_dict",
];

fn shape_server_info(msgpack: &[u8], server_args: &ServerArgs) -> Result<Vec<u8>, String> {
    // GetInternalStateReqOutput asdict → `{ "internal_state": { server-args dump +
    // metrics }, ... }`. Pull that inner map out (it is NOT safe to expose whole).
    let mut obj: serde_json::Map<String, serde_json::Value> =
        rmp_serde::from_slice(msgpack).map_err(|e| e.to_string())?;
    let internal = match obj.remove("internal_state") {
        Some(serde_json::Value::Object(m)) => m,
        _ => serde_json::Map::new(),
    };

    // Copy only the allowlisted runtime metrics — never the raw server-args dump.
    let mut state_out = serde_json::Map::new();
    for &k in INTERNAL_STATE_ALLOWLIST {
        match internal.get(k) {
            Some(v) if !v.is_null() => {
                state_out.insert(k.to_string(), v.clone());
            }
            _ => {}
        }
    }

    // Top-level non-secret config from typed accessors — these structurally cannot
    // surface a key field, unlike the raw dump.
    let response = serde_json::json!({
        "model_path": server_args.model_path(),
        "served_model_name": server_args.served_model_name(),
        "tokenizer_path": server_args.tokenizer_path(),
        "max_context_length": server_args.context_len(),
        "internal_states": [serde_json::Value::Object(state_out)],
    });
    serde_json::to_vec(&response).map_err(|e| e.to_string())
}

/// Convert a msgpack control response (the scheduler's native ring format) into
/// JSON bytes for the HTTP client.
fn msgpack_to_json(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let val = rmpv::decode::read_value(&mut &*bytes).map_err(|e| e.to_string())?;
    serde_json::to_vec(&val).map_err(|e| e.to_string())
}

/// Folds the per-chunk [`ChunkEvent`] deltas the detok emits into a cumulative
/// view. Used by the drain loops that need cumulative output — every unary
/// response and the cumulative SGLang `/generate` stream. OpenAI streaming
/// forwards deltas directly and doesn't use this. Shared with the [`openai`]
/// submodule (`super::OutputAccumulator`).
/// Wraps a single cumulative [`ChunkEvent`] built up across delta chunks.
/// Holding it (rather than mirroring its fields) lets `snapshot` hand back a
/// **borrow** for each streaming frame — no per-frame deep clone of the growing
/// buffers (that made token-by-token streaming O(T²) *extra* on top of the wire
/// format's inherent O(T²)).
#[derive(Default)]
struct OutputAccumulator {
    out: ChunkEvent,
}

impl OutputAccumulator {
    /// Fold one delta frame in. Output families concatenate; input families and
    /// hidden states are set-once / last-writer-wins (they ride the prefill/final
    /// chunk), matching the Python `meta_info` assignment.
    fn fold(&mut self, d: &ChunkEvent) {
        let o = &mut self.out;
        o.rid = d.rid; // constant across the request; keeps the accumulated view coherent
        o.text.push_str(&d.text);
        o.token_ids.extend_from_slice(&d.token_ids); // token_ids doubles as output_ids
        o.completion_tokens += d.completion_tokens;
        o.prompt_tokens = d.prompt_tokens; // constant across the request
        o.out_lp_val.extend_from_slice(&d.out_lp_val);
        o.out_lp_idx.extend_from_slice(&d.out_lp_idx);
        o.out_top_val.extend_from_slice(&d.out_top_val);
        o.out_top_idx.extend_from_slice(&d.out_top_idx);
        o.out_top_lens.extend_from_slice(&d.out_top_lens);
        o.out_tid_val.extend_from_slice(&d.out_tid_val);
        o.out_tid_idx.extend_from_slice(&d.out_tid_idx);
        o.out_tid_lens.extend_from_slice(&d.out_tid_lens);
        o.out_lp_txt.extend_from_slice(&d.out_lp_txt);
        o.out_top_txt.extend_from_slice(&d.out_top_txt);
        o.out_tid_txt.extend_from_slice(&d.out_tid_txt);
        if !d.in_lp_val.is_empty() {
            o.in_lp_val = d.in_lp_val.clone();
            o.in_lp_idx = d.in_lp_idx.clone();
            o.in_lp_txt = d.in_lp_txt.clone();
        }
        // Input families ride once (prefill); `lens` non-empty marks their arrival.
        if !d.in_top_lens.is_empty() {
            o.in_top_val = d.in_top_val.clone();
            o.in_top_idx = d.in_top_idx.clone();
            o.in_top_lens = d.in_top_lens.clone();
            o.in_top_txt = d.in_top_txt.clone();
        }
        if !d.in_tid_lens.is_empty() {
            o.in_tid_val = d.in_tid_val.clone();
            o.in_tid_idx = d.in_tid_idx.clone();
            o.in_tid_lens = d.in_tid_lens.clone();
            o.in_tid_txt = d.in_tid_txt.clone();
        }
        // Hidden states are non-cumulative: the latest non-empty set wins.
        if !d.hidden_lens.is_empty() {
            o.hidden_val = d.hidden_val.clone();
            o.hidden_lens = d.hidden_lens.clone();
        }
        if d.finish_reason.is_some() {
            o.finish_reason = d.finish_reason.clone();
        }
    }

    /// Borrow the cumulative output for an intermediate streaming frame.
    fn snapshot(&self) -> &ChunkEvent {
        &self.out
    }

    /// Consume into the final cumulative output.
    fn into_output(self) -> ChunkEvent {
        self.out
    }
}

async fn generate(State(state): State<AppState>, Json(body): Json<GenerateBody>) -> Response {
    let stream = body.stream;
    // Fan `text`/`input_ids`/`sampling_params` (scalar or list) into per-request
    // payloads. `is_batch` = list form → the response is a JSON array.
    let (payloads, is_batch) = match body.split() {
        Ok(v) => v,
        Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
    };
    if !is_batch {
        // `split` guarantees exactly one payload for a non-batch body.
        let payload = payloads
            .into_iter()
            .next()
            .expect("split yields >=1 payload");
        generate_single(&state, payload, stream).await
    } else {
        generate_batch(&state, payloads, stream).await
    }
}

/// A single (non-batched) `/generate`: submit one request, then either stream its
/// SSE frames or fold to one unary response.
async fn generate_single(state: &AppState, payload: GeneratePayload, stream: bool) -> Response {
    // `return_text_in_logprobs` is decoded on the detok shard (see
    // DetokMsg::Register.decode_logprob_text) into the `*_txt` columns, so
    // `sglang_frame_value` just reads them — no tokenizer needed here.
    let kind = RequestKind::Generate(GenerateRequest {
        payload,
        input_ids: None,
        stream,
        is_health_check: false,
    });
    let (rid, mut rx) = match submit(state, kind).await {
        Ok(v) => v,
        Err(()) => {
            return (StatusCode::SERVICE_UNAVAILABLE, "service unavailable").into_response();
        }
    };
    // Abort the request if the client disconnects: the guard fires `try_abort`
    // when dropped before the request finishes — i.e. axum drops this handler /
    // SSE stream because the connection closed. Disarmed on a natural terminal.
    let mut guard = AbortGuard::new(state.senders.clone(), rid);
    // Response `meta_info.id`, stringified once and reused for every frame.
    let rid_str = rid.0.to_string();
    // Cumulative frames (SGLang default) vs per-step deltas.
    let incremental = state.server_args.incremental_streaming_output();

    if stream {
        // A single request is a 1-element batch without the `index` field — reuse
        // the same stream so the frame/abort/truncation logic lives in one place.
        use futures::StreamExt;
        let s = generation_event_stream(vec![(rid, rx)], guard, incremental, false)
            .map(|data| Ok::<_, Infallible>(Event::default().data(data)));
        Sse::new(s).into_response()
    } else {
        // Unary: fold to the terminal, respond once. Disarm only on a real terminal
        // (a truncation leaves the guard armed so the scheduler work is aborted).
        let (status, value, terminal) = drain_unary(&mut rx, &rid_str).await;
        if terminal {
            guard.disarm(rid);
        }
        (status, Json(value)).into_response()
    }
}

/// Fold a unary request to its terminal → (HTTP status, result/`error` JSON, saw-terminal); `false` = truncation, caller keeps the abort guard armed. Shared by single + batch.
async fn drain_unary(
    rx: &mut mpsc::Receiver<EgressItem>,
    rid_str: &str,
) -> (StatusCode, serde_json::Value, bool) {
    let mut acc = OutputAccumulator::default();
    while let Some(item) = rx.recv().await {
        match item {
            EgressItem::Frame(out) => acc.fold(&out),
            EgressItem::Done(out) => {
                acc.fold(&out);
                let final_out = acc.into_output();
                // A validation abort carries its own HTTP status + diagnostic.
                if let Some((code, message)) = abort_status(&final_out.finish_reason) {
                    let status =
                        StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    return (status, error_value(code, &message), true);
                }
                return (
                    StatusCode::OK,
                    sglang_frame_value(&final_out, rid_str),
                    true,
                );
            }
            EgressItem::Error(e) => {
                let code = e.http_status();
                let status =
                    StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                return (status, error_value(code, &e.to_string()), true);
            }
            EgressItem::Control(_) => continue, // never on `/generate`
        }
    }
    // Sender dropped without a terminal item: the shard dropped this request (a
    // truncation — a client disconnect would have dropped the handler future).
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        error_value(500, "response truncated before completion"),
        false,
    )
}

/// Batch `/generate`: submit every sub-request first so the scheduler runs them
/// together, then either (unary) drain each in order into a JSON **array** of
/// results, or (streaming) multiplex their egress streams into one SSE response
/// where each frame carries its batch `index`. One [`AbortGuard`] covers the whole
/// batch, so a client disconnect aborts them all. A failed unary item is its own
/// `{ "error": … }` array entry; the batch response is 200.
async fn generate_batch(
    state: &AppState,
    payloads: Vec<GeneratePayload>,
    stream: bool,
) -> Response {
    let mut guard = AbortGuard::new_empty(state.senders.clone());
    let mut receivers = Vec::with_capacity(payloads.len());
    for payload in payloads {
        let kind = RequestKind::Generate(GenerateRequest {
            payload,
            input_ids: None,
            stream,
            is_health_check: false,
        });
        match submit(state, kind).await {
            Ok((rid, rx)) => {
                guard.arm(rid);
                receivers.push((rid, rx));
            }
            Err(()) => {
                return (StatusCode::SERVICE_UNAVAILABLE, "service unavailable").into_response();
            }
        }
    }

    if stream {
        // Multiplex the N streams (mirrors the Python `_handle_batch_request` path);
        // `guard` moves into the stream so a disconnect aborts what's unfinished.
        use futures::StreamExt;
        let incremental = state.server_args.incremental_streaming_output();
        let s = generation_event_stream(receivers, guard, incremental, true)
            .map(|data| Ok::<_, Infallible>(Event::default().data(data)));
        Sse::new(s).into_response()
    } else {
        // Unary: drain each in order (already all submitted, so they run together).
        let mut results = Vec::with_capacity(receivers.len());
        for (rid, mut rx) in receivers {
            let rid_str = rid.0.to_string();
            let (_status, value, terminal) = drain_unary(&mut rx, &rid_str).await;
            if terminal {
                guard.disarm(rid);
            }
            results.push(value);
        }
        (StatusCode::OK, Json(serde_json::Value::Array(results))).into_response()
    }
}

/// Await the next item from `rx`, handing the receiver back so a `FuturesUnordered`
/// (which owns one future per poll) can re-poll it. `None` = the channel closed.
async fn recv_indexed(
    index: usize,
    mut rx: mpsc::Receiver<EgressItem>,
) -> (usize, mpsc::Receiver<EgressItem>, Option<EgressItem>) {
    let item = rx.recv().await;
    (index, rx, item)
}

/// Multiplex `receivers` (one per request) into SSE `data` strings + a final `[DONE]`;
/// `with_index` tags each frame (batch only), `incremental` = delta vs cumulative,
/// `guard` aborts unfinished on drop.
fn generation_event_stream(
    receivers: Vec<(RequestId, mpsc::Receiver<EgressItem>)>,
    mut guard: AbortGuard,
    incremental: bool,
    with_index: bool,
) -> impl futures::Stream<Item = String> {
    async_stream::stream! {
        use futures::StreamExt;

        let n = receivers.len();
        let rids: Vec<RequestId> = receivers.iter().map(|(rid, _)| *rid).collect();
        let rid_strs: Vec<String> = rids.iter().map(|r| r.0.to_string()).collect();
        let mut accs: Vec<OutputAccumulator> =
            (0..n).map(|_| OutputAccumulator::default()).collect();

        // Tag a frame with its batch position (batch only; a single request omits it).
        let tag = |mut v: serde_json::Value, i: usize| {
            if with_index {
                v["index"] = serde_json::json!(i);
            }
            v.to_string()
        };

        // Poll all receivers concurrently; re-arm a receiver's future after each
        // non-terminal frame so its stream keeps flowing.
        let mut futs = futures::stream::FuturesUnordered::new();
        for (i, (_, rx)) in receivers.into_iter().enumerate() {
            futs.push(recv_indexed(i, rx));
        }

        while let Some((i, rx, item)) = futs.next().await {
            match item {
                Some(EgressItem::Frame(out)) => {
                    accs[i].fold(&out);
                    let v = stream_frame_value(out, &accs[i], incremental, &rid_strs[i]);
                    yield tag(v, i);
                    futs.push(recv_indexed(i, rx)); // keep this item flowing
                }
                Some(EgressItem::Done(out)) => {
                    accs[i].fold(&out);
                    // A validation abort → an error object, not a frame.
                    let v = match abort_status(&out.finish_reason) {
                        Some((code, message)) => error_value(code, &message),
                        None => stream_frame_value(out, &accs[i], incremental, &rid_strs[i]),
                    };
                    yield tag(v, i);
                    guard.disarm(rids[i]); // terminal → not re-pushed
                }
                Some(EgressItem::Error(e)) => {
                    yield tag(error_value(e.http_status(), &e.to_string()), i);
                    guard.disarm(rids[i]);
                }
                Some(EgressItem::Control(_)) => {
                    futs.push(recv_indexed(i, rx)); // never on /generate; keep polling
                }
                None => {
                    // Channel closed with no terminal → truncation for this item;
                    // leave its rid armed so the scheduler work is aborted.
                    yield tag(error_value(500, "response truncated before completion"), i);
                }
            }
        }
        yield "[DONE]".to_string();
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

    /// End-to-end: a `ChunkEvent` carrying a prompt-logprob request (first input
    /// logprob is the `NaN` sentinel) formats without panicking and emits
    /// `input_token_logprobs` with a leading `[null, token_id, text]`.
    #[test]
    fn prompt_logprob_frame_emits_null_first() {
        let out = ChunkEvent {
            in_lp_val: vec![f32::NAN, -0.5],
            in_lp_idx: vec![10, 20],
            in_lp_txt: vec!["<s>".into(), "hi".into()],
            ..Default::default()
        };
        let frame = sglang_frame_value(&out, "1");
        assert_eq!(
            frame["meta_info"]["input_token_logprobs"],
            serde_json::json!([[serde_json::Value::Null, 10, "<s>"], [-0.5f32, 20, "hi"]])
        );
    }

    /// The accumulator folds deltas cumulatively and `snapshot` borrows the
    /// running state (no per-frame clone); `into_output` moves the same state.
    #[test]
    fn accumulator_snapshot_is_cumulative() {
        let mut acc = OutputAccumulator::default();
        acc.fold(&ChunkEvent {
            text: "he".into(),
            token_ids: vec![1, 2],
            completion_tokens: 2,
            ..Default::default()
        });
        {
            let s = acc.snapshot();
            assert_eq!(s.text, "he");
            assert_eq!(s.token_ids, vec![1, 2]);
        }
        acc.fold(&ChunkEvent {
            text: "llo".into(),
            token_ids: vec![3],
            completion_tokens: 1,
            ..Default::default()
        });
        {
            let s = acc.snapshot();
            assert_eq!(s.text, "hello"); // cumulative
            assert_eq!(s.token_ids, vec![1, 2, 3]);
            assert_eq!(s.completion_tokens, 3);
        }
        let out = acc.into_output();
        assert_eq!(out.text, "hello");
    }

    /// A populated text column (decoded on the detok shard) → `Some`; empty
    /// (`return_text_in_logprobs` off) → `None` → null text slots.
    #[test]
    fn opt_texts_gates_on_population() {
        assert!(opt_texts(&[]).is_none());
        let t = vec!["x".to_string()];
        assert_eq!(opt_texts(&t), Some(t.as_slice()));
    }

    /// The shared classifier both paths use: a validation abort yields its
    /// `(code, message)` (the streaming path turns this into an SSE error event
    /// instead of a normal `Done` frame); anything else yields `None`.
    #[test]
    fn abort_status_extracts_code_and_message() {
        let (code, msg) = abort_status(&Some(serde_json::json!({
            "type": "abort", "message": "over the limit", "status_code": 400
        })))
        .expect("validation abort → (code, message)");
        assert_eq!(code, 400);
        assert_eq!(msg, "over the limit");
        // Normal finish, bare abort (no status), and no finish → not an error.
        assert!(abort_status(&Some(serde_json::json!({"type": "stop"}))).is_none());
        assert!(abort_status(&Some(serde_json::json!({"type": "abort"}))).is_none());
        assert!(abort_status(&None).is_none());
    }

    /// A normal finish, a bare abort (no status), and no finish are not errors
    /// (the unary path returns them as a 200 result frame).
    #[test]
    fn non_error_finishes_stay_ok() {
        assert!(abort_status(&Some(serde_json::json!({"type": "stop", "matched": 5}))).is_none());
        assert!(abort_status(&Some(serde_json::json!({"type": "length", "length": 8}))).is_none());
        assert!(
            abort_status(&Some(
                serde_json::json!({"type": "abort", "message": "Aborted"})
            ))
            .is_none()
        );
        assert!(abort_status(&None).is_none());
    }
}

#[cfg(test)]
mod server_info_tests {
    use super::*;

    /// The scheduler's `internal_state` embeds the full server-args dump (incl.
    /// `api_key`/`admin_api_key`). `/server_info` must surface only the allowlisted
    /// runtime metrics + curated config — never the secrets — and must not re-nest
    /// the dump under `internal_states[].internal_state`.
    #[test]
    fn shape_server_info_excludes_secrets_and_dump() {
        // GetInternalStateReqOutput.asdict → { "internal_state": { …dump+metrics… } }.
        let internal = rmpv::Value::Map(vec![
            (
                rmpv::Value::from("api_key"),
                rmpv::Value::from("secret-token"),
            ),
            (
                rmpv::Value::from("admin_api_key"),
                rmpv::Value::from("admin-token"),
            ),
            (rmpv::Value::from("model_path"), rmpv::Value::from("/m")),
            (
                rmpv::Value::from("last_gen_throughput"),
                rmpv::Value::from(1.5),
            ),
            (
                rmpv::Value::from("effective_max_running_requests_per_dp"),
                rmpv::Value::from(32),
            ),
        ]);
        let outer = rmpv::Value::Map(vec![(rmpv::Value::from("internal_state"), internal)]);
        let mut msgpack = Vec::new();
        rmpv::encode::write_value(&mut msgpack, &outer).unwrap();

        let sa =
            ServerArgs::from_json(r#"{"model_path": "/m", "api_key": "secret-token"}"#).unwrap();
        let out = shape_server_info(&msgpack, &sa).unwrap();
        let text = String::from_utf8(out.clone()).unwrap();
        // No secret leaks anywhere in the serialized response.
        assert!(!text.contains("secret-token"), "api_key leaked: {text}");
        assert!(
            !text.contains("admin-token"),
            "admin_api_key leaked: {text}"
        );

        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        // Allowlisted metric surfaced; the whole dump did not.
        let state0 = &v["internal_states"][0];
        assert_eq!(state0["last_gen_throughput"], 1.5);
        assert_eq!(state0["effective_max_running_requests_per_dp"], 32);
        assert!(
            state0.get("internal_state").is_none(),
            "must not re-nest the dump under internal_state"
        );
        assert!(state0.get("api_key").is_none());
        // Curated top-level config comes from typed accessors, not the dump.
        assert_eq!(v["model_path"], "/m");
    }
}

#[cfg(test)]
mod abort_guard_tests {
    use super::*;

    fn senders(tm: flume::Sender<TmEvent>) -> Senders {
        Senders {
            tm,
            tok: flume::unbounded().0,
            detok: vec![],
        }
    }

    /// An armed guard aborts its rid on drop — exactly the cleanup a busy-skipped
    /// `/health_generate` probe relies on. It never sees a terminal frame here, so
    /// dropping the guard is the only path that deregisters its detok sink (via the
    /// ingress `on_abort`). Regression for the detok-entry leak per health probe.
    #[test]
    fn armed_guard_aborts_on_drop() {
        let (tm_tx, tm_rx) = flume::unbounded();
        drop(AbortGuard::new(senders(tm_tx), RequestId(7)));
        assert!(
            matches!(tm_rx.try_recv(), Ok(TmEvent::Abort(id)) if id == RequestId(7)),
            "armed guard must abort its rid on drop",
        );
        assert!(tm_rx.try_recv().is_err(), "exactly one abort");
    }

    /// A disarmed rid (finished naturally) is not aborted on drop.
    #[test]
    fn disarmed_guard_does_not_abort() {
        let (tm_tx, tm_rx) = flume::unbounded();
        let mut guard = AbortGuard::new(senders(tm_tx), RequestId(9));
        guard.disarm(RequestId(9));
        drop(guard);
        assert!(tm_rx.try_recv().is_err(), "disarmed rid must not abort");
    }
}

#[cfg(test)]
mod batch_stream_tests {
    use super::*;
    use futures::StreamExt;

    fn senders() -> Senders {
        Senders {
            tm: flume::unbounded().0,
            tok: flume::unbounded().0,
            detok: vec![],
        }
    }

    fn frame(rid: u64, text: &str) -> EgressItem {
        EgressItem::Frame(ChunkEvent {
            rid,
            text: text.into(),
            completion_tokens: 1,
            ..Default::default()
        })
    }
    fn done(rid: u64, text: &str) -> EgressItem {
        EgressItem::Done(ChunkEvent {
            rid,
            text: text.into(),
            completion_tokens: 1,
            finish_reason: Some(serde_json::json!({ "type": "length" })),
            ..Default::default()
        })
    }
    fn parse(s: &str) -> serde_json::Value {
        serde_json::from_str(s).expect("frame is JSON")
    }

    /// Two sub-requests' frames interleave into one stream, each tagged with its
    /// batch `index`; text accumulates per item; `[DONE]` comes only after both
    /// terminate, then the stream ends.
    #[tokio::test]
    async fn interleaves_indexes_and_accumulates() {
        let (tx0, rx0) = mpsc::channel(8);
        let (tx1, rx1) = mpsc::channel(8);
        let receivers = vec![(RequestId(10), rx0), (RequestId(11), rx1)];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), false, true);
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

        tx1.send(done(11, "?")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 1);
        assert_eq!(v["text"], "b?");

        assert_eq!(stream.next().await.unwrap(), "[DONE]");
        assert!(stream.next().await.is_none());
    }

    /// A per-item error is surfaced with its `index` and doesn't end the batch;
    /// `[DONE]` still waits for the other item.
    #[tokio::test]
    async fn per_item_error_carries_index() {
        let (tx0, rx0) = mpsc::channel(8);
        let (tx1, rx1) = mpsc::channel(8);
        let receivers = vec![(RequestId(10), rx0), (RequestId(11), rx1)];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), false, true);
        futures::pin_mut!(stream);

        tx0.send(EgressItem::Error(crate::error::Error::Validation(
            "bad".into(),
        )))
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
        let receivers = vec![(RequestId(10), rx)];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), true, true);
        futures::pin_mut!(stream);

        tx.send(frame(10, "Hello")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "Hello");
        assert_eq!(v["meta_info"]["completion_tokens"], 1);

        tx.send(frame(10, " world")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], " world", "delta, not cumulative 'Hello world'");
        assert_eq!(
            v["meta_info"]["completion_tokens"], 2,
            "count stays cumulative"
        );

        tx.send(done(10, "!")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "!");
        assert_eq!(v["meta_info"]["completion_tokens"], 3);
        assert_eq!(v["meta_info"]["finish_reason"]["type"], "length");

        assert_eq!(stream.next().await.unwrap(), "[DONE]");
    }

    /// The single-request shape (`with_index=false`, one receiver) omits the
    /// `index` field entirely, and still terminates with `[DONE]`.
    #[tokio::test]
    async fn single_shape_omits_index() {
        let (tx, rx) = mpsc::channel(8);
        let receivers = vec![(RequestId(10), rx)];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), false, false);
        futures::pin_mut!(stream);

        tx.send(done(10, "hi")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "hi");
        assert!(v.get("index").is_none(), "single response has no index");

        assert_eq!(stream.next().await.unwrap(), "[DONE]");
    }
}
