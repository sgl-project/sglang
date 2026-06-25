//! API server (axum / tokio). I/O-bound; runs on its own pinned multi-thread
//! runtime. Designed so additional protocols (h2/h3/websocket/grpc) can mount
//! the same `AppState` later — only this module knows about HTTP.
//!
//! `/generate` opens a per-request egress channel, moves a `Request` into the
//! ingress pipeline, and then either awaits a single `Done` (unary) or relays
//! frames as Server-Sent Events (streaming), byte-compatible with the Python
//! `http_server.generate_request` (`data: {json}\n\n` … `data: [DONE]\n\n`).

use std::convert::Infallible;
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
use tokio::sync::mpsc;

use crate::fsm::RequestState;
use crate::ids::RequestIdGen;
use crate::message::{EgressItem, GeneratePayload, Request, RequestKind};
use crate::runtime::ServerArgs;
use crate::runtime::channels::{Senders, TmEvent};

/// Shared state for every handler. Holds the submit machinery (`senders`,
/// `id_gen`, `egress_buf`) plus the static `ServerArgs` read by `/v1/models`.
/// `server_args` is an `Arc`, so the per-request clone axum makes is a refcount
/// bump.
#[derive(Clone)]
struct AppState {
    senders: Senders,
    id_gen: Arc<RequestIdGen>,
    egress_buf: usize,
    server_args: Arc<ServerArgs>,
}

pub async fn serve(
    bind: SocketAddr,
    senders: Senders,
    id_gen: Arc<RequestIdGen>,
    egress_buf: usize,
    server_args: Arc<ServerArgs>,
) {
    let state = AppState {
        senders,
        id_gen,
        egress_buf,
        server_args,
    };
    let app = Router::new()
        .route("/generate", post(generate))
        // Control-plane endpoints: each reuses the ingress FSM (no tokenization)
        // and returns a single, non-streamed JSON result from the scheduler.
        // Adding one = a route line passing its scheduler request-struct tag.
        .route("/server_info", get(server_info))
        // Static config endpoint (OpenAI-compatible): no scheduler round-trip.
        .route("/v1/models", get(available_models))
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

/// `GET /v1/models` — OpenAI-compatible model list. Served from `server_args`;
/// no scheduler round-trip. Mirrors `http_server.available_models`.
///
/// TODO(v1/models): when `--enable-lora`, append a `ModelCard` per loaded LoRA
/// adapter (`id=lora_name, root=lora_path, parent=served_model_name,
/// max_model_len=None`). Adapters load/unload at runtime, so that part needs a
/// control-request query to the scheduler's LoRA registry.
async fn available_models(State(state): State<AppState>) -> Response {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let name = state.server_args.served_model_name();
    let base = serde_json::json!({
        "id": name,
        "object": "model",
        "created": created,
        "owned_by": "sglang",
        "root": name,
        "parent": serde_json::Value::Null,
        "max_model_len": state.server_args.context_len(),
    });
    let list = serde_json::json!({ "object": "list", "data": [base] });
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        serde_json::to_vec(&list).unwrap_or_default(),
    )
        .into_response()
}

/// Submit a request into the ingress pipeline. Returns the per-request egress
/// receiver to read the result(s) from.
async fn submit(
    state: &AppState,
    kind: RequestKind,
    payload: GeneratePayload,
) -> Result<mpsc::Receiver<EgressItem>, ()> {
    let (tx, rx) = mpsc::channel::<EgressItem>(state.egress_buf);
    let id = state.id_gen.next();
    let stream = payload.stream;
    let req = Request {
        id,
        kind,
        state: RequestState::Received,
        payload,
        input_ids: None,
        sink: tx,
        stream,
    };
    // Async-aware send from this tokio task: under a full TM inbox it yields
    // (backpressure) instead of parking a worker thread, which flume's sync
    // `send` would do. Err only when the inbox is closed (runtime shutdown).
    match state.senders.tm.send_async(TmEvent::Ingress(req)).await {
        Ok(()) => Ok(rx),
        Err(_) => {
            tracing::error!("tm inbox closed; request dropped");
            Err(())
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
    let mut rx = submit(state, RequestKind::Control(tag), GeneratePayload::default())
        .await
        .map_err(|()| (StatusCode::SERVICE_UNAVAILABLE, "service unavailable").into_response())?;
    match rx.recv().await {
        Some(EgressItem::Done(bytes)) | Some(EgressItem::Frame(bytes)) => Ok(bytes),
        Some(EgressItem::Error(e)) => {
            let code =
                StatusCode::from_u16(e.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            Err((code, e.to_string()).into_response())
        }
        None => Err((StatusCode::from_u16(499).unwrap(), "request aborted").into_response()),
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

async fn generate(State(state): State<AppState>, Json(payload): Json<GeneratePayload>) -> Response {
    let stream = payload.stream;
    let mut rx = match submit(&state, RequestKind::Generate, payload).await {
        Ok(rx) => rx,
        Err(()) => {
            return (StatusCode::SERVICE_UNAVAILABLE, "service unavailable").into_response();
        }
    };

    if stream {
        // SSE: relay frames until Done/Error, then `data: [DONE]`.
        let s = async_stream::stream! {
            while let Some(item) = rx.recv().await {
                match item {
                    EgressItem::Frame(bytes) => {
                        yield Ok::<_, Infallible>(Event::default().data(String::from_utf8_lossy(&bytes)));
                    }
                    EgressItem::Done(bytes) => {
                        yield Ok(Event::default().data(String::from_utf8_lossy(&bytes)));
                        break;
                    }
                    EgressItem::Error(e) => {
                        let body = serde_json::json!({
                            "error": { "message": e.to_string(), "code": e.http_status() }
                        });
                        yield Ok(Event::default().data(body.to_string()));
                        break;
                    }
                }
            }
            yield Ok(Event::default().data("[DONE]"));
        };
        Sse::new(s).into_response()
    } else {
        // Unary: drain until the terminal frame.
        while let Some(item) = rx.recv().await {
            match item {
                EgressItem::Frame(_) => continue, // ignore intermediate for unary
                EgressItem::Done(bytes) => {
                    return (
                        StatusCode::OK,
                        [("content-type", "application/json")],
                        bytes,
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
            }
        }
        // Sender dropped without a terminal item → treated as aborted.
        (StatusCode::from_u16(499).unwrap(), "request aborted").into_response()
    }
}
