//! Request submission into the ingress pipeline, shared by every endpoint
//! module: mint the client-visible rid (uuid hex, Python-parity), build the
//! `Request`, and hand it to the TM with an egress receiver for the response.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use tokio::sync::mpsc;

use super::AppState;
use crate::fsm::RequestState;
use crate::ids::RidHash;
use crate::message::{EgressItem, EgressSink, Request, RequestKind};
use crate::runtime::channels::TmEvent;

/// Submit one request; returns the rid, its hashed routing key, and the egress
/// receiver. Rid policy by kind: generate health probes get the Python server's
/// `HEALTH_CHECK_<uuid>` form (so scheduler logs and prefix-gated handling
/// recognize them), a client-supplied generate rid (already fanned out per item
/// by `split`) wins over minting, and control requests always get a rust-minted
/// rid (their responses are routed by it, so no client-supplied form exists).
pub(super) async fn submit(
    state: &AppState,
    kind: RequestKind,
) -> Result<(RidHash, String, mpsc::Receiver<EgressItem>), Response> {
    let rid = match &kind {
        RequestKind::Generate(g) if g.is_health_check => crate::ids::new_health_check_rid(),
        RequestKind::Generate(g) => g.rid.clone().unwrap_or_else(crate::ids::new_rid),
        RequestKind::Control(_) => crate::ids::new_rid(),
    };
    let id = RidHash::from_rid(&rid);
    // Async-aware send so a full TM inbox yields (backpressure) instead of parking
    // a thread; Err only when the inbox is closed (shutdown).
    let (tx, rx) = mpsc::channel::<EgressItem>(state.egress_buf);
    let request = Request {
        rid_hash: id,
        rid: rid.clone(),
        state: RequestState::Received,
        sink: EgressSink::Local(tx),
        kind,
    };
    match state.senders.tm.send_async(TmEvent::Ingress(request)).await {
        Ok(()) => Ok((id, rid, rx)),
        // `SendError` has a single meaning — the channel is disconnected.
        Err(_) => {
            tracing::error!(%rid, "tm inbox closed; request rejected");
            // Return 503 so the client can retry.
            Err((StatusCode::SERVICE_UNAVAILABLE, "service unavailable").into_response())
        }
    }
}
