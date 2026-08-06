//! Request submission into the ingress pipeline, shared by every endpoint
//! module: mint the client-visible rid (uuid hex, Python-parity), build the
//! `Request`, and hand it to the TM with an egress receiver for the response.

use axum::{http::StatusCode, response::Response};
use tokio::sync::mpsc;

use super::AppState;
use super::guard::AbortGuard;
use crate::fsm::RequestState;
use crate::ids::Rid;
use crate::message::{EgressItem, EgressSink, GenerateRequest, Request, RequestKind};
use crate::tokenizer_manager::TmEvent;
use crate::utils::response::{error_response, error_value};

/// The single submission failure: the TM inbox is closed.
struct ChannelClosed;

/// Submit one request; returns the rid, its hashed routing key, and the egress
/// receiver.
async fn submit(
    state: &AppState,
    kind: RequestKind,
) -> Result<(Rid, mpsc::Receiver<EgressItem>), ChannelClosed> {
    let rid = match &kind {
        // Generate rids are already final: `GenerateBody::into_requests` normalized the
        // client's, or minted one. Control requests have no client-facing rid.
        RequestKind::Generate(g) => g.rid.clone(),
        RequestKind::Control(c) => c.rid().into(),
        // Internal service call — no client-facing rid; mint a fresh one.
        RequestKind::Detokenize { .. } => Rid::new(),
    };
    // Two in-flight requests can name the same client rid, but they cannot share a
    // `Rid`: `into_requests` built each through `Rid::from_client`, which appends a
    // uniquifier.
    let (tx, rx) = mpsc::channel::<EgressItem>(state.egress_buf);
    let request = Request {
        rid: rid.clone(),
        state: RequestState::Received,
        sink: EgressSink::Local(tx),
        kind,
    };
    match state.senders.tm.send_async(TmEvent::Ingress(request)).await {
        Ok(()) => Ok((rid, rx)),
        // `SendError` has a single meaning — the channel is disconnected.
        Err(_) => {
            tracing::error!(%rid, "tm ingress channel closed; request rejected");
            Err(ChannelClosed)
        }
    }
}

/// Native-frontend submission.
pub(super) async fn submit_native(
    state: &AppState,
    kind: RequestKind,
    stream: bool,
) -> Result<(Rid, mpsc::Receiver<EgressItem>), Response> {
    match submit(state, kind).await {
        Ok(submitted) => Ok(submitted),
        Err(ChannelClosed) => Err(error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            error_value(503, "service unavailable"),
            stream,
        )),
    }
}

/// OpenAI-frontend submission.
pub(super) async fn submit_openai(
    state: &AppState,
    request: GenerateRequest,
    stream: bool,
    guard: &mut AbortGuard,
) -> Result<mpsc::Receiver<EgressItem>, Response> {
    match submit(state, RequestKind::Generate(Box::new(request))).await {
        Ok((rid, rx)) => {
            guard.arm(rid);
            Ok(rx)
        }
        Err(ChannelClosed) => Err(error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            super::openai::error_payload(StatusCode::SERVICE_UNAVAILABLE, "service unavailable"),
            stream,
        )),
    }
}
