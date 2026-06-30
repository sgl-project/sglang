//! API server (axum / tokio). I/O-bound; runs on its own pinned multi-thread
//! runtime. Designed so additional protocols (h2/h3/websocket/grpc) can mount
//! the same `AppState` later — only this module knows about HTTP.
//!
//! `/generate` opens a per-request egress channel, moves a `Request` into the
//! ingress pipeline, and then either awaits a single `Done` (unary) or relays
//! frames as Server-Sent Events (streaming), byte-compatible with the Python
//! `http_server.generate_request` (`data: {json}\n\n` … `data: [DONE]\n\n`).
//!
//! The OpenAI-compatible endpoints (`/v1/*`) live in the [`openai`] submodule;
//! they share this module's [`AppState`] and submit machinery. Future protocols
//! (e.g. Anthropic) get their own sibling submodule the same way.

mod openai;

use std::net::SocketAddr;
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
use crate::ids::RequestIdGen;
use dynamo_renderer::PromptFormatter;

use crate::message::{
    ControlRequest, EgressItem, EgressSink, GeneratePayload, GenerateRequest, GenerationOutput,
    Request, RequestKind,
};
use crate::runtime::ServerArgs;
use crate::runtime::channels::{Senders, TmEvent};
use crate::transport::NetReady;

/// How the api-server reaches the pipeline. Embedded (`dp_size == 1`) uses the
/// in-process `flume` channels; the standalone api-server process (`dp_size > 1`)
/// reaches each headless DP rank over TCP. The [`NetReady`] pool is established
/// lazily, once every rank has registered (`POST /internal/register`), so
/// `submit` answers 503 until then. `submit` is the only place that branches;
/// every handler is transport-agnostic.
#[derive(Clone)]
pub enum Transport {
    Local(Senders),
    Net(Arc<NetReady>),
}

/// Built once at startup from the model's `tokenizer_config.json` (`None` when
/// the model has no chat template, or under `skip_tokenizer_init`). `Clone` is a
/// refcount bump (the formatter is `Arc`-backed), so it rides on `AppState`.
#[derive(Clone)]
struct ChatFormatter(PromptFormatter);

/// Shared state for every handler. Holds the submit machinery (`senders`,
/// `id_gen`, `egress_buf`) plus the static `ServerArgs` read by `/v1/models`.
/// `server_args` is an `Arc`, so the per-request clone axum makes is a refcount
/// bump.
#[derive(Clone)]
struct AppState {
    transport: Transport,
    id_gen: Arc<RequestIdGen>,
    egress_buf: usize,
    server_args: Arc<ServerArgs>,
    /// `None` when the model has no chat template → `/v1/chat/completions` 400s.
    chat: Option<ChatFormatter>,
}

pub async fn serve(
    bind: SocketAddr,
    transport: Transport,
    id_gen: Arc<RequestIdGen>,
    egress_buf: usize,
    server_args: Arc<ServerArgs>,
) {
    let chat = openai::load_chat_formatter(&server_args).map(ChatFormatter);
    let state = AppState {
        transport,
        id_gen,
        egress_buf,
        server_args,
        chat,
    };
    let app = Router::new()
        .route("/generate", post(generate))
        // OpenAI-compatible: same tokenize→generate→detok pipeline, OpenAI shape.
        .route("/v1/completions", post(openai::openai_completions))
        .route(
            "/v1/chat/completions",
            post(openai::openai_chat_completions),
        )
        // Control-plane endpoints: each reuses the ingress FSM (no tokenization)
        // and returns a single, non-streamed JSON result from the scheduler.
        // Adding one = a route line passing its scheduler request-struct tag.
        .route("/server_info", get(server_info))
        // Static config endpoint (OpenAI-compatible): no scheduler round-trip.
        .route("/v1/models", get(openai::available_models))
        // Internal: a headless DP rank reports its TCP endpoint so the standalone
        // api-server can build its pool (Mode A registration). No-op (400) on the
        // embedded transport.
        .route("/internal/register", post(register))
        .with_state(state);

    match tokio::net::TcpListener::bind(bind).await {
        Ok(listener) => {
            tracing::info!(%bind, "sglang-server api listening");
            if let Err(e) = axum::serve(listener, app).await {
                tracing::error!(error = %e, "axum serve exited");
            }
        }
        Err(e) => tracing::error!(error = %e, %bind, "failed to bind api server"),
    }
}

/// Body of `POST /internal/register`: one headless DP rank announcing the TCP
/// address the api-server should dial it at.
#[derive(serde::Deserialize)]
struct RegisterBody {
    dp_rank: usize,
    endpoint: String,
}

/// `POST /internal/register` — a headless DP rank reports its TCP endpoint. The
/// standalone api-server collects all `dp_size` of them, then connects its pool
/// (handled inside [`NetReady::register`]); request handlers return 503 until
/// it's up. Not applicable to the embedded (in-process) transport.
async fn register(State(state): State<AppState>, Json(body): Json<RegisterBody>) -> Response {
    match &state.transport {
        Transport::Net(ready) => match body.endpoint.parse() {
            Ok(addr) => {
                ready.register(body.dp_rank, addr);
                (StatusCode::OK, "registered").into_response()
            }
            Err(_) => (StatusCode::BAD_REQUEST, "bad endpoint").into_response(),
        },
        Transport::Local(_) => {
            (StatusCode::BAD_REQUEST, "registration not applicable").into_response()
        }
    }
}

/// Submit a request into the ingress pipeline. Returns the per-request egress
/// receiver to read the result(s) from. The `kind` carries the variant body
/// (generate payload / control tag), so this stays generic over both.
async fn submit(state: &AppState, kind: RequestKind) -> Result<mpsc::Receiver<EgressItem>, ()> {
    let id = state.id_gen.next();
    match &state.transport {
        // Embedded: move the request into the in-process pipeline. Async-aware
        // send so a full TM inbox yields (backpressure) instead of parking a
        // worker thread; Err only when the inbox is closed (runtime shutdown).
        Transport::Local(senders) => {
            let (tx, rx) = mpsc::channel::<EgressItem>(state.egress_buf);
            let req = Request {
                id,
                state: RequestState::Received,
                sink: EgressSink::Local(tx),
                kind,
            };
            match senders.tm.send_async(TmEvent::Ingress(req)).await {
                Ok(()) => Ok(rx),
                Err(_) => {
                    tracing::error!("tm inbox closed; request dropped");
                    Err(())
                }
            }
        }
        // Standalone: serialize the request and route it over TCP. The pool is
        // up only after every rank has registered — until then `client()` is
        // `None` and we fail (→ 503). Generate requests load-balance to one rank;
        // **control** requests (e.g. `/server_info`) are answered per-rank, so
        // they're broadcast to all and the first response wins — round-robin
        // could pick a rank whose answer never returns, hanging the caller.
        Transport::Net(ready) => match ready.client() {
            Some(client) => match kind {
                RequestKind::Control(_) => {
                    client.submit_broadcast(id, kind, state.egress_buf).await
                }
                RequestKind::Generate(_) => client.submit(id, kind, state.egress_buf).await,
            },
            None => Err(()),
        },
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
    let mut rx = submit(state, RequestKind::Control(ControlRequest { tag }))
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

/// Format a neutral [`GenerationOutput`] as one SGLang `/generate` frame. Lives
/// in the handler now that the detok shard is protocol-neutral. `output_ids` is
/// surfaced only in `skip_tokenizer_init` mode (cumulative, non-empty there).
fn sglang_frame(out: &GenerationOutput) -> Vec<u8> {
    let mut v = serde_json::json!({
        "text": out.text,
        "meta_info": {
            "id": out.rid,
            "prompt_tokens": out.prompt_tokens,
            "completion_tokens": out.completion_tokens,
            "finish_reason": out.finish_reason.as_deref().map(|r| serde_json::json!({ "type": r })),
        },
    });
    if !out.output_ids.is_empty() {
        v["output_ids"] = serde_json::json!(out.output_ids);
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
    finish_reason: Option<String>,
}

impl OutputAccumulator {
    /// Fold one delta frame in.
    fn fold(&mut self, d: &GenerationOutput) {
        self.text.push_str(&d.text);
        self.output_ids.extend_from_slice(&d.output_ids);
        self.completion_tokens += d.completion_tokens;
        self.prompt_tokens = d.prompt_tokens; // constant across the request
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
        }
    }
}

async fn generate(State(state): State<AppState>, Json(payload): Json<GeneratePayload>) -> Response {
    let stream = payload.stream;
    let kind = RequestKind::Generate(GenerateRequest {
        payload,
        input_ids: None,
        stream,
    });
    let mut rx = match submit(&state, kind).await {
        Ok(rx) => rx,
        Err(()) => {
            return (StatusCode::SERVICE_UNAVAILABLE, "service unavailable").into_response();
        }
    };

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
                    return (
                        StatusCode::OK,
                        [("content-type", "application/json")],
                        sglang_frame(&acc.into_output(out.rid)),
                    )
                        .into_response();
                }
                EgressItem::Error(e) => {
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
        // Sender dropped without a terminal item → treated as aborted.
        (StatusCode::from_u16(499).unwrap(), "request aborted").into_response()
    }
}
