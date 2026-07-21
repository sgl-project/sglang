//! API server (axum / tokio). I/O-bound; own pinned multi-thread runtime. Only
//! this module knows HTTP, so other protocols can mount the same `AppState`.
//! `/generate` submits a `Request` then awaits one `Done` (unary) or relays SSE
//! frames (`data: {json}` … `[DONE]`), byte-compatible with Python
//! `http_server.generate_request`; `/server_info` reuses it for one control result.
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
use crate::ids::RidHash;

use crate::message::{
    ChunkEvent, ChunkExtras, ControlRequest, EgressItem, EgressSink, GenerateBody, GenerateRequest,
    Request, RequestKind,
};
use crate::runtime::ServerArgs;
use crate::runtime::channels::{Senders, TmEvent};
use crate::tokenizer_manager::ActivityCounter;

/// Shared handler state: the submit machinery (`senders`, `egress_buf`)
/// + shared tokenizer. `Clone` is cheap refcount bumps (every field is `Arc`).
#[derive(Clone)]
struct AppState {
    senders: Senders,
    egress_buf: usize,
    server_args: Arc<ServerArgs>,
    /// Egress heartbeat (bumped per drained ring frame). `/health_generate`
    /// watches it advance to confirm the scheduler → detok path is alive.
    egress_activity: ActivityCounter,
    /// `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` (default true). When true,
    /// `/health` also runs the generation round-trip; when false it's plain 200.
    health_endpoint_generation: bool,
}

/// Parse `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` (Python `EnvBool`:
/// `false/0/no/n` → false, else true; unset → true).
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
    egress_buf: usize,
    server_args: Arc<ServerArgs>,
    egress_activity: ActivityCounter,
    shutdown: flume::Receiver<()>,
) {
    let access_log_enabled = server_args.http_access_log_enabled();
    let state = AppState {
        senders,
        egress_buf,
        server_args,
        egress_activity,
        health_endpoint_generation: read_health_endpoint_generation(),
    };
    let app = Router::new()
        .route("/generate", post(generate))
        // `/health` runs the generation round-trip by default (env-gated, else
        // plain 200); `/health_generate` always does. Mirrors Python.
        .route("/health", get(health))
        .route("/health_generate", get(health_generate))
        // Control-plane: reuses the ingress FSM (no tokenization), returns one
        // non-streamed JSON result. Adding one = a route line + its struct tag.
        .route("/server_info", get(server_info))
        // Static config, no scheduler round-trip. `/get_model_info` (+ `/model_info`
        // alias) is what the SGLang lang backend (`RuntimeEndpoint`, gsm8k/eval)
        // calls at startup; `/v1/models` is OpenAI-compatible.
        .route("/get_model_info", get(model_info))
        .route("/model_info", get(model_info))
        .route("/v1/models", get(openai::available_models))
        // TODO(auth): no API-key boundary yet. Python gates every route (except
        // /health*, /metrics*, OPTIONS) via `add_api_key_middleware`; until ported,
        // a configured `api_key` does NOT protect these routes.
        .with_state(state);
    // Access log gated exactly like uvicorn's (`--log-level-http warning` turns
    // it off); when disabled the middleware isn't even installed — zero cost.
    let app = if access_log_enabled {
        app.layer(axum::middleware::from_fn(access_log))
    } else {
        app
    };

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
    // Non-graceful shutdown: on the signal, stop accepting and RETURN without
    // waiting for in-flight handlers (a `/generate` blocked on egress would wedge
    // the join). Returning unwinds `block_on` in `runtime::start` → the api tokio
    // runtime drops → detached handlers cancel → their `AbortGuard`s fire, release
    // `Senders` clones → tok/detok channels close → workers exit. Full drain is
    // deferred (see `request_shutdown`).
    // `with_connect_info` exposes the peer address to the access-log middleware.
    let serve = axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    );
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

/// Access log — one INFO line per request, content-matching the Python server's
/// uvicorn access log (`127.0.0.1:54232 - "GET /model_info HTTP/1.1" 200 OK`).
/// Logged when the response head is ready; for SSE that's stream start, same as
/// uvicorn.
async fn access_log(
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<std::net::SocketAddr>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let version = req.version();
    let res = next.run(req).await;
    let status = res.status();
    tracing::info!(
        "{peer} - \"{method} {uri} {version:?}\" {} {}",
        status.as_u16(),
        status.canonical_reason().unwrap_or("")
    );
    res
}

/// Submit a request into the ingress pipeline; returns the client-visible rid
/// (uuid hex, Python-parity), its hashed routing key, and the egress receiver.
/// `kind` carries the variant body (generate / control), so this is generic over
/// both.
async fn submit(
    state: &AppState,
    kind: RequestKind,
) -> Result<(RidHash, String, mpsc::Receiver<EgressItem>), ()> {
    // Health probes get the Python server's `HEALTH_CHECK_<uuid>` rid form so
    // scheduler logs and prefix-gated handling recognize them; a client-supplied
    // rid (already fanned out per item by `split`) wins over minting.
    let rid = match &kind {
        RequestKind::Generate(g) if g.is_health_check => crate::ids::new_health_check_rid(),
        RequestKind::Generate(g) => g.rid.clone().unwrap_or_else(crate::ids::new_rid),
        RequestKind::Control(_) => crate::ids::new_rid(),
    };
    let id = RidHash::from_rid(&rid);
    // Async-aware send so a full TM inbox yields (backpressure) instead of parking
    // a thread; Err only when the inbox is closed (shutdown).
    let (tx, rx) = mpsc::channel::<EgressItem>(state.egress_buf);
    let req = Request {
        rid_hash: id,
        rid: rid.clone(),
        state: RequestState::Received,
        sink: EgressSink::Local(tx),
        kind,
    };
    match state.senders.tm.send_async(TmEvent::Ingress(req)).await {
        Ok(()) => Ok((id, rid, rx)),
        Err(_) => {
            tracing::error!("tm inbox closed; request dropped");
            Err(())
        }
    }
}

/// Aborts still-in-flight rids on drop — i.e. axum dropped the handler/SSE stream
/// because the client disconnected (mirrors Python's `is_disconnected` abort).
/// Each rid is disarmed on natural finish; whatever remains at drop is aborted.
struct AbortGuard {
    senders: Senders,
    /// `(routing key, rid string)` — the string is what `AbortReq` needs on the
    /// scheduler wire (unrecoverable from the hashed key), the key is what
    /// callers disarm by.
    rids: Vec<(RidHash, String)>,
}

impl AbortGuard {
    fn new(senders: Senders, id: RidHash, rid: String) -> Self {
        Self {
            senders,
            rids: vec![(id, rid)],
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

    /// Track a request for abort-on-drop.
    fn arm(&mut self, id: RidHash, rid: String) {
        self.rids.push((id, rid));
    }

    /// Request finished naturally — don't abort it on drop.
    fn disarm(&mut self, id: RidHash) {
        self.rids.retain(|(r, _)| *r != id);
    }
}

impl Drop for AbortGuard {
    fn drop(&mut self) {
        // Best-effort non-blocking abort per rid; a full/closed channel just drops
        // it (the request then finishes at EOS, only later).
        for (_, rid) in self.rids.drain(..) {
            let _ = self.senders.tm.try_send(TmEvent::Abort(rid));
        }
    }
}

/// Submit a `Control(tag)` through the ingress FSM (no tokenization) and await the
/// scheduler's single msgpack result (a `structs.asdict` named map). Returns the
/// raw bytes, or an error `Response` to return as-is.
async fn await_control_result(
    state: &AppState,
    tag: &'static str,
) -> Result<bytes::Bytes, Response> {
    let (_id, _rid, mut rx) = submit(state, RequestKind::Control(ControlRequest { tag }))
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

/// SGLang logprob shape: a list of `[logprob, token_id, text]` tuples. `texts`
/// (parallel to `idxs`) fills the text slot when set, else `null`.
fn logprob_tuples(vals: &[f32], idxs: &[i32], texts: Option<&[String]>) -> serde_json::Value {
    let tuples: Vec<serde_json::Value> = vals
        .iter()
        .zip(idxs.iter())
        .enumerate()
        .map(|(j, (&v, &tid))| serde_json::json!([lp_value(v), tid, text_slot(texts, j)]))
        .collect();
    serde_json::Value::Array(tuples)
}

/// Ragged top-k / token-ids shape: one entry per position — a list of
/// `[logprob, token_id, text]` tuples, or `null` when `lens[p] == 0` (mirrors
/// `detokenize_top_logprobs_tokens`). `texts` is parallel to `vals`/`idxs`.
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
/// `list[list[float]]` (one row per output position).
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

/// Classify a terminal finish reason: `Some((code, message))` when it's an `abort`
/// carrying a `status_code` (a scheduler request error, e.g. over-context → 400).
/// Both unary + streaming paths inspect this instead of treating `Done` as normal.
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
/// (response `meta_info.id`) is passed as a string; the event's numeric `rid` is
/// just the shard routing key.
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
    if !out.token_ids.is_empty() {
        v["output_ids"] = serde_json::json!(out.token_ids);
    }
    // Logprobs + hidden states ride behind the boxed extras (absent for a plain
    // token/text frame). `[logprob, token_id, text|null]` tuples; text
    // (`return_text_in_logprobs`) was decoded on the detok shard into `*_txt`.
    let Some(ex) = out.extras.as_deref() else {
        return v;
    };
    if !ex.out_lp_val.is_empty() {
        v["meta_info"]["output_token_logprobs"] =
            logprob_tuples(&ex.out_lp_val, &ex.out_lp_idx, opt_texts(&ex.out_lp_txt));
    }
    if !ex.in_lp_val.is_empty() {
        v["meta_info"]["input_token_logprobs"] =
            logprob_tuples(&ex.in_lp_val, &ex.in_lp_idx, opt_texts(&ex.in_lp_txt));
    }
    if !ex.out_top_lens.is_empty() {
        v["meta_info"]["output_top_logprobs"] = ragged_logprob_tuples(
            &ex.out_top_val,
            &ex.out_top_idx,
            &ex.out_top_lens,
            opt_texts(&ex.out_top_txt),
        );
    }
    if !ex.in_top_lens.is_empty() {
        v["meta_info"]["input_top_logprobs"] = ragged_logprob_tuples(
            &ex.in_top_val,
            &ex.in_top_idx,
            &ex.in_top_lens,
            opt_texts(&ex.in_top_txt),
        );
    }
    if !ex.out_tid_lens.is_empty() {
        v["meta_info"]["output_token_ids_logprobs"] = ragged_logprob_tuples(
            &ex.out_tid_val,
            &ex.out_tid_idx,
            &ex.out_tid_lens,
            opt_texts(&ex.out_tid_txt),
        );
    }
    if !ex.in_tid_lens.is_empty() {
        v["meta_info"]["input_token_ids_logprobs"] = ragged_logprob_tuples(
            &ex.in_tid_val,
            &ex.in_tid_idx,
            &ex.in_tid_lens,
            opt_texts(&ex.in_tid_txt),
        );
    }
    if !ex.hidden_lens.is_empty() {
        v["meta_info"]["hidden_states"] = hidden_states_rows(&ex.hidden_val, &ex.hidden_lens);
    }
    v
}

/// Cumulative frame JSON from the accumulator's memoized ids/text — O(T), not O(T²).
/// Byte-identical to `sglang_frame_value(..).to_string()` (a `BTreeMap` keeps keys
/// alphabetical); pinned by `cumulative_frame_json_matches_serde`. `None` on extras.
fn cumulative_frame_json(
    acc: &OutputAccumulator,
    rid: &str,
    index: Option<usize>,
) -> Option<String> {
    use std::fmt::Write;

    let o = acc.snapshot();
    if o.extras.is_some() {
        return None;
    }
    // Fixed size regardless of T, so this stays O(1) per frame.
    let meta = serde_json::json!({
        "id": rid,
        "prompt_tokens": o.prompt_tokens,
        "completion_tokens": o.completion_tokens,
        "finish_reason": o.finish_reason,
    })
    .to_string();

    let mut s = String::with_capacity(acc.text_json.len() + acc.ids_json.len() + meta.len() + 40);
    s.push('{');
    if let Some(i) = index {
        let _ = write!(s, "\"index\":{i},");
    }
    s.push_str("\"meta_info\":");
    s.push_str(&meta);
    if !acc.ids_json.is_empty() {
        s.push_str(",\"output_ids\":[");
        s.push_str(&acc.ids_json);
        s.push(']');
    }
    s.push_str(",\"text\":\"");
    s.push_str(&acc.text_json);
    s.push_str("\"}");
    Some(s)
}

/// Attach the batch `index` (batch streams only) and render to the SSE `data` text.
fn tag_value(mut v: serde_json::Value, index: Option<usize>) -> String {
    if let Some(i) = index {
        v["index"] = serde_json::json!(i);
    }
    v.to_string()
}

/// One streaming frame's JSON: cumulative ignores `delta`, incremental ships it.
fn stream_frame_string(
    delta: ChunkEvent,
    acc: &OutputAccumulator,
    incremental: bool,
    rid_str: &str,
    index: Option<usize>,
) -> String {
    if !incremental {
        return cumulative_frame_string(acc, rid_str, index);
    }
    tag_value(stream_frame_value(delta, acc, true, rid_str), index)
}

/// A cumulative frame's JSON, built purely from the accumulator (which is why a
/// backlog can coalesce to its last); falls back to the `Value` builder on extras.
fn cumulative_frame_string(acc: &OutputAccumulator, rid_str: &str, index: Option<usize>) -> String {
    cumulative_frame_json(acc, rid_str, index)
        .unwrap_or_else(|| tag_value(sglang_frame_value(acc.snapshot(), rid_str), index))
}

/// Format one streaming frame: the accumulator's cumulative view (default), or this
/// step's delta with the cumulative token count in `meta_info` (matching Python).
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

/// Generic control endpoint: the scheduler's response straight to JSON (`tag` =
/// request-struct name). For control endpoints whose response needs no shaping.
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

/// `GET /health` — liveness. By default (env true, mirroring Python) runs the same
/// 1-token round-trip as `/health_generate`; env false → plain 200 (routing the
/// request already proves the frontend is up).
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

/// Deep health: confirm the scheduler → detok path is producing output. 200 iff
/// the egress heartbeat advances within the timeout, else 503.
///
/// Fires a pre-tokenized 1-token probe (`input_ids = [0]`, skips the tokenizer) so
/// an idle pipeline produces a frame, then watches the *global*
/// [`AppState::egress_activity`] counter (not the probe's own rid) — so a busy
/// server passes immediately and a backlog never false-503s (the analogue of
/// Python's `last_receive_tstamp`). The `HEALTH_CHECK` skip + `http_worker_ipc`
/// ack are irrelevant here: this single-process server owns the egress ring.
async fn health_generate_inner(state: &AppState) -> Response {
    let baseline = state
        .egress_activity
        .load(std::sync::atomic::Ordering::Relaxed);

    // Fire the probe (the heartbeat is the signal, not its own response). A busy
    // scheduler skips it with no terminal frame, so its detok registration is
    // cleaned up only by the `AbortGuard` below.
    let sampling_params = rmpv::Value::Map(vec![
        (rmpv::Value::from("max_new_tokens"), rmpv::Value::from(1)),
        (rmpv::Value::from("temperature"), rmpv::Value::F64(0.0)),
    ]);
    let kind = RequestKind::Generate(GenerateRequest {
        input_ids: Some(vec![0]),
        sampling_params: Some(sampling_params),
        stream: false,
        // The scheduler skips this when busy so it never occupies a queue slot.
        is_health_check: true,
        ..Default::default()
    });
    let (id, rid, _keepalive) = match submit(state, kind).await {
        // Hold the receiver so the probe's sink stays open until it completes.
        Ok((id, rid, rx)) => (id, rid, rx),
        Err(()) => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
    };
    // Deregister on drop (never disarmed): a busy-skipped probe has no terminal
    // frame, so without this abort it leaks one detok entry per call.
    let _abort_guard = AbortGuard::new(state.senders.clone(), id, rid);

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
/// curated [`ServerArgs`] accessors), never the raw server-args dump (embeds
/// `api_key`/`admin_api_key`; see [`shape_server_info`]).
///
/// TODO(server_info): Python also includes `kv_events`; add once plumbed.
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

/// Runtime-metric keys `get_internal_state` adds atop the server-args dump. We copy
/// ONLY these out of `internal_state` (an allowlist), so the co-mingled
/// `api_key`/`admin_api_key` can never reach the response.
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

    // Top-level non-secret config from typed accessors (structurally can't surface
    // a key field, unlike the raw dump).
    let response = serde_json::json!({
        "model_path": server_args.model_path(),
        "served_model_name": server_args.served_model_name(),
        "tokenizer_path": server_args.tokenizer_path(),
        "max_context_length": server_args.context_len(),
        "max_total_num_tokens": server_args.max_total_num_tokens(),
        "version": server_args.version(),
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

/// Folds per-chunk [`ChunkEvent`] deltas into a cumulative view — used by the drain
/// loops needing cumulative output (every unary response + the cumulative SGLang
/// stream; OpenAI streaming forwards deltas and skips this). Holds a single
/// [`ChunkEvent`] so `snapshot` hands back a **borrow** per frame — no per-frame
/// clone of the growing buffers (that added O(T²) atop the wire's inherent O(T²)).
/// Shared with the [`openai`] submodule.
#[derive(Default)]
struct OutputAccumulator {
    out: ChunkEvent,
    /// Serialized cumulative `output_ids` body (`"1,2,3"`, no brackets), appended per
    /// delta so a frame memcpy's it instead of rebuilding the array — O(T), not O(T²).
    ids_json: String,
    /// JSON-escaped cumulative text, without the surrounding quotes. Escaping is
    /// per-character, so `escape(a + b) == escape(a) + escape(b)` and deltas append.
    text_json: String,
}

/// Append `s` JSON-escaped (no surrounding quotes) — `serde_json` quotes it, and the
/// quotes are the first and last bytes of a string encoding.
fn push_escaped(dst: &mut String, s: &str) {
    if s.is_empty() {
        return;
    }
    let quoted = serde_json::to_string(s).expect("str-to-json should never fail");
    dst.push_str(&quoted[1..quoted.len() - 1]);
}

impl OutputAccumulator {
    /// Fold one delta frame in. Output families concatenate; input families and
    /// hidden states are set-once / last-writer-wins (they ride the prefill/final
    /// chunk), matching the Python `meta_info` assignment.
    fn fold(&mut self, d: &ChunkEvent) {
        use std::fmt::Write;

        // Grow the memoized serializations alongside the raw cumulative buffers.
        push_escaped(&mut self.text_json, &d.text);
        for &id in &d.token_ids {
            if !self.ids_json.is_empty() {
                self.ids_json.push(',');
            }
            let _ = write!(self.ids_json, "{id}");
        }

        let o = &mut self.out;
        o.rid_hash = d.rid_hash; // constant across the request; keeps the accumulated view coherent
        o.text.push_str(&d.text);
        o.token_ids.extend_from_slice(&d.token_ids); // token_ids doubles as output_ids
        o.completion_tokens += d.completion_tokens;
        o.prompt_tokens = d.prompt_tokens; // constant across the request
        if d.finish_reason.is_some() {
            o.finish_reason = d.finish_reason.clone();
        }
        // Logprobs/hidden ride behind the boxed extras — most frames have none, so
        // only allocate the accumulator's box once a delta actually carries some.
        let Some(de) = d.extras.as_deref() else {
            return;
        };
        let oe = o
            .extras
            .get_or_insert_with(|| Box::new(ChunkExtras::default()));
        oe.out_lp_val.extend_from_slice(&de.out_lp_val);
        oe.out_lp_idx.extend_from_slice(&de.out_lp_idx);
        oe.out_top_val.extend_from_slice(&de.out_top_val);
        oe.out_top_idx.extend_from_slice(&de.out_top_idx);
        oe.out_top_lens.extend_from_slice(&de.out_top_lens);
        oe.out_tid_val.extend_from_slice(&de.out_tid_val);
        oe.out_tid_idx.extend_from_slice(&de.out_tid_idx);
        oe.out_tid_lens.extend_from_slice(&de.out_tid_lens);
        oe.out_lp_txt.extend_from_slice(&de.out_lp_txt);
        oe.out_top_txt.extend_from_slice(&de.out_top_txt);
        oe.out_tid_txt.extend_from_slice(&de.out_tid_txt);
        if !de.in_lp_val.is_empty() {
            oe.in_lp_val = de.in_lp_val.clone();
            oe.in_lp_idx = de.in_lp_idx.clone();
            oe.in_lp_txt = de.in_lp_txt.clone();
        }
        // Input families ride once (prefill); `lens` non-empty marks their arrival.
        if !de.in_top_lens.is_empty() {
            oe.in_top_val = de.in_top_val.clone();
            oe.in_top_idx = de.in_top_idx.clone();
            oe.in_top_lens = de.in_top_lens.clone();
            oe.in_top_txt = de.in_top_txt.clone();
        }
        if !de.in_tid_lens.is_empty() {
            oe.in_tid_val = de.in_tid_val.clone();
            oe.in_tid_idx = de.in_tid_idx.clone();
            oe.in_tid_lens = de.in_tid_lens.clone();
            oe.in_tid_txt = de.in_tid_txt.clone();
        }
        // Hidden states are non-cumulative: the latest non-empty set wins.
        if !de.hidden_lens.is_empty() {
            oe.hidden_val = de.hidden_val.clone();
            oe.hidden_lens = de.hidden_lens.clone();
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
async fn generate_single(state: &AppState, req: GenerateRequest, stream: bool) -> Response {
    // `return_text_in_logprobs` is decoded on the detok shard into `*_txt`, so
    // `sglang_frame_value` just reads them — no tokenizer needed here.
    let (id, rid_str, mut rx) = match submit(state, RequestKind::Generate(req)).await {
        Ok(v) => v,
        Err(()) => {
            return (StatusCode::SERVICE_UNAVAILABLE, "service unavailable").into_response();
        }
    };
    // Abort on client disconnect: the guard fires when dropped before the request
    // finishes (axum drops the handler/SSE stream). Disarmed on a natural terminal.
    // `rid_str` is the response `meta_info.id`, reused for every frame.
    let mut guard = AbortGuard::new(state.senders.clone(), id, rid_str.clone());
    // Cumulative frames (SGLang default) vs per-step deltas.
    let incremental = state.server_args.incremental_streaming_output();

    if stream {
        // A single request is a 1-element batch without the `index` field — reuse
        // the same stream so the frame/abort/truncation logic lives in one place.
        use futures::StreamExt;
        let s = generation_event_stream(vec![(id, rid_str, rx)], guard, incremental, false)
            .map(|data| Ok::<_, Infallible>(Event::default().data(data)));
        Sse::new(s).into_response()
    } else {
        // Unary: fold to the terminal, respond once. Disarm only on a real terminal
        // (a truncation leaves the guard armed so the scheduler work is aborted).
        let (status, value, terminal) = drain_unary(&mut rx, &rid_str).await;
        if terminal {
            guard.disarm(id);
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

/// Batch `/generate`: submit all sub-requests first (scheduler runs them together),
/// then either (unary) drain each in order into a JSON array, or (streaming)
/// multiplex their streams into one SSE response, each frame carrying its `index`.
/// One [`AbortGuard`] covers the batch. A failed unary item is its own
/// `{ "error": … }` entry; the batch response is 200.
async fn generate_batch(
    state: &AppState,
    requests: Vec<GenerateRequest>,
    stream: bool,
) -> Response {
    let mut guard = AbortGuard::new_empty(state.senders.clone());
    let mut receivers = Vec::with_capacity(requests.len());
    for req in requests {
        match submit(state, RequestKind::Generate(req)).await {
            Ok((id, rid, rx)) => {
                guard.arm(id, rid.clone());
                receivers.push((id, rid, rx));
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
        for (id, rid_str, mut rx) in receivers {
            let (_status, value, terminal) = drain_unary(&mut rx, &rid_str).await;
            if terminal {
                guard.disarm(id);
            }
            results.push(value);
        }
        (StatusCode::OK, Json(serde_json::Value::Array(results))).into_response()
    }
}

/// Await the next item from `rx`, then drain whatever queued behind it (so the caller
/// can coalesce a backlog, as Python's `state.out_list` does), handing the receiver
/// back for `FuturesUnordered` to re-poll. Empty result = channel closed.
async fn recv_indexed(
    index: usize,
    mut rx: mpsc::Receiver<EgressItem>,
) -> (usize, mpsc::Receiver<EgressItem>, Vec<EgressItem>) {
    let mut items = Vec::new();
    match rx.recv().await {
        Some(item) => items.push(item),
        None => return (index, rx, items), // closed
    }
    while let Ok(item) = rx.try_recv() {
        items.push(item);
    }
    (index, rx, items)
}

/// Multiplex `receivers` (one per request) into SSE `data` strings + a final `[DONE]`;
/// `with_index` tags each frame (batch only), `incremental` = delta vs cumulative,
/// `guard` aborts unfinished on drop.
fn generation_event_stream(
    receivers: Vec<(RidHash, String, mpsc::Receiver<EgressItem>)>,
    mut guard: AbortGuard,
    incremental: bool,
    with_index: bool,
) -> impl futures::Stream<Item = String> {
    async_stream::stream! {
        use futures::StreamExt;

        let n = receivers.len();
        let rids: Vec<RidHash> = receivers.iter().map(|(id, _, _)| *id).collect();
        let rid_strs: Vec<String> = receivers.iter().map(|(_, rid, _)| rid.clone()).collect();
        let mut accs: Vec<OutputAccumulator> =
            (0..n).map(|_| OutputAccumulator::default()).collect();

        // Batch position, tagged onto every frame (a single request omits it).
        let idx = |i: usize| with_index.then_some(i);

        // Poll all receivers concurrently; re-arm a receiver's future after each
        // non-terminal frame so its stream keeps flowing.
        let mut futs = futures::stream::FuturesUnordered::new();
        for (i, (_, _, rx)) in receivers.into_iter().enumerate() {
            futs.push(recv_indexed(i, rx));
        }

        while let Some((i, rx, items)) = futs.next().await {
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
                    EgressItem::Frame(out) => {
                        accs[i].fold(&out);
                        if incremental {
                            yield stream_frame_string(out, &accs[i], true, &rid_strs[i], idx(i));
                        } else {
                            coalesced = true;
                        }
                    }
                    EgressItem::Done(out) => {
                        accs[i].fold(&out);
                        terminal = Some(out);
                    }
                    EgressItem::Error(e) => failed = Some(e),
                    EgressItem::Control(_) => {} // never on /generate
                }
            }

            if let Some(e) = failed {
                yield tag_value(error_value(e.http_status(), &e.to_string()), idx(i));
                guard.disarm(rids[i]);
            } else if let Some(out) = terminal {
                // A validation abort → an error object, not a frame. The final frame
                // carries the full cumulative state, so any coalesced ones are moot.
                yield match abort_status(&out.finish_reason) {
                    Some((code, message)) => tag_value(error_value(code, &message), idx(i)),
                    None => stream_frame_string(out, &accs[i], incremental, &rid_strs[i], idx(i)),
                };
                guard.disarm(rids[i]); // terminal → not re-pushed
            } else {
                if coalesced {
                    yield cumulative_frame_string(&accs[i], &rid_strs[i], idx(i));
                }
                futs.push(recv_indexed(i, rx)); // keep this item flowing
            }
        }
        yield "[DONE]".to_string();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

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
            extras: Some(Box::new(ChunkExtras {
                in_lp_val: vec![f32::NAN, -0.5],
                in_lp_idx: vec![10, 20],
                in_lp_txt: vec!["<s>".into(), "hi".into()],
                ..Default::default()
            })),
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

    fn senders_with_tm(tm: flume::Sender<TmEvent>) -> Senders {
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
        drop(AbortGuard::new(
            senders_with_tm(tm_tx),
            RidHash::from_rid("r7"),
            "r7".to_string(),
        ));
        assert!(
            matches!(tm_rx.try_recv(), Ok(TmEvent::Abort(rid)) if rid == "r7"),
            "armed guard must abort its rid on drop",
        );
        assert!(tm_rx.try_recv().is_err(), "exactly one abort");
    }

    /// A disarmed rid (finished naturally) is not aborted on drop.
    #[test]
    fn disarmed_guard_does_not_abort() {
        let (tm_tx, tm_rx) = flume::unbounded();
        let id = RidHash::from_rid("r9");
        let mut guard = AbortGuard::new(senders_with_tm(tm_tx), id, "r9".to_string());
        guard.disarm(id);
        drop(guard);
        assert!(tm_rx.try_recv().is_err(), "disarmed rid must not abort");
    }

    fn senders() -> Senders {
        Senders {
            tm: flume::unbounded().0,
            tok: flume::unbounded().0,
            detok: vec![],
        }
    }

    fn frame(rid: u64, text: &str) -> EgressItem {
        EgressItem::Frame(ChunkEvent {
            rid_hash: rid,
            text: text.into(),
            completion_tokens: 1,
            ..Default::default()
        })
    }
    fn done(rid: u64, text: &str) -> EgressItem {
        EgressItem::Done(ChunkEvent {
            rid_hash: rid,
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
        let receivers = vec![
            (RidHash(10), "10".to_string(), rx0),
            (RidHash(11), "11".to_string(), rx1),
        ];
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
        let receivers = vec![
            (RidHash(10), "10".to_string(), rx0),
            (RidHash(11), "11".to_string(), rx1),
        ];
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
        let receivers = vec![(RidHash(10), "10".to_string(), rx)];
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
        let receivers = vec![(RidHash(10), "10".to_string(), rx)];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), false, false);
        futures::pin_mut!(stream);

        tx.send(done(10, "hi")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "hi");
        assert!(v.get("index").is_none(), "single response has no index");

        assert_eq!(stream.next().await.unwrap(), "[DONE]");
    }

    /// A backlog of cumulative chunks collapses to a single frame carrying the latest
    /// state — each cumulative frame supersedes the last, so emitting the intermediate
    /// ones ships the full O(T) payload again for nothing. Mirrors the Python waiter's
    /// `out = out_list[-1]`. This is the whole point of draining in `recv_indexed`.
    #[tokio::test]
    async fn cumulative_backlog_coalesces_to_latest() {
        let (tx, rx) = mpsc::channel(8);
        let receivers = vec![(RidHash(10), "10".to_string(), rx)];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), false, false);
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
        let receivers = vec![(RidHash(10), "10".to_string(), rx)];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), true, false);
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

    /// The memoized cumulative fast path must emit **byte-identical** JSON to the
    /// `serde_json::Value` builder it replaces — same keys, same alphabetical order
    /// (`Map` is a `BTreeMap`; no `preserve_order`), same escaping. Covers unicode
    /// and control chars, an empty-ids first frame, a finish_reason, and the batch
    /// `index`. Guards the O(T) rewrite of the O(T²) `output_ids` serialization.
    #[test]
    fn cumulative_frame_json_matches_serde() {
        let deltas = [
            ChunkEvent {
                rid_hash: 7,
                text: String::new(),
                token_ids: vec![],
                completion_tokens: 0,
                prompt_tokens: 128,
                ..Default::default()
            },
            ChunkEvent {
                rid_hash: 7,
                text: "He\"llo\n\t".into(),
                token_ids: vec![1000],
                completion_tokens: 1,
                prompt_tokens: 128,
                ..Default::default()
            },
            ChunkEvent {
                rid_hash: 7,
                text: " 世界 🌍 \\".into(),
                token_ids: vec![-2, 3],
                completion_tokens: 2,
                prompt_tokens: 128,
                ..Default::default()
            },
            ChunkEvent {
                rid_hash: 7,
                text: "!".into(),
                token_ids: vec![9],
                completion_tokens: 1,
                prompt_tokens: 128,
                finish_reason: Some(serde_json::json!({"type": "stop", "matched": 9})),
                ..Default::default()
            },
        ];

        for index in [None, Some(3usize)] {
            let mut acc = OutputAccumulator::default();
            for d in &deltas {
                acc.fold(d);
                let fast = cumulative_frame_json(&acc, "7", index).expect("no extras → fast path");
                let slow = tag_value(sglang_frame_value(acc.snapshot(), "7"), index);
                assert_eq!(fast, slow, "index={index:?} text={:?}", acc.snapshot().text);
            }
        }
    }

    /// A frame carrying logprobs falls back to the `Value` builder (the fast path
    /// only knows the plain text/ids shape).
    #[test]
    fn cumulative_frame_json_defers_on_extras() {
        let mut acc = OutputAccumulator::default();
        acc.fold(&ChunkEvent {
            rid_hash: 1,
            token_ids: vec![5],
            text: "x".into(),
            extras: Some(Box::new(ChunkExtras {
                out_lp_val: vec![-0.5],
                out_lp_idx: vec![5],
                ..Default::default()
            })),
            ..Default::default()
        });
        assert!(cumulative_frame_json(&acc, "1", None).is_none());
    }
}
