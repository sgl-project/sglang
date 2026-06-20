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
    routing::post,
};
use tokio::sync::mpsc;

use crate::fsm::RequestState;
use crate::ids::RequestIdGen;
use crate::message::{EgressItem, GeneratePayload, Request};
use crate::runtime::channels::{Senders, TmEvent};

#[derive(Clone)]
struct AppState {
    senders: Senders,
    id_gen: Arc<RequestIdGen>,
    egress_buf: usize,
}

pub async fn serve(bind: SocketAddr, senders: Senders, id_gen: Arc<RequestIdGen>, egress_buf: usize) {
    let state = AppState {
        senders,
        id_gen,
        egress_buf,
    };
    let app = Router::new()
        .route("/generate", post(generate))
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

/// Submit a request into the ingress pipeline. Returns the per-request egress
/// receiver to read the result(s) from.
async fn submit(
    state: &AppState,
    payload: GeneratePayload,
) -> Result<mpsc::Receiver<EgressItem>, ()> {
    let (tx, rx) = mpsc::channel::<EgressItem>(state.egress_buf);
    let id = state.id_gen.next();
    let stream = payload.stream;
    let req = Request {
        id,
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

async fn generate(
    State(state): State<AppState>,
    Json(payload): Json<GeneratePayload>,
) -> Response {
    let stream = payload.stream;
    let mut rx = match submit(&state, payload).await {
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
