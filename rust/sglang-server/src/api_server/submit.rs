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
use crate::tokenizer_manager::TmEvent;

/// Submit one request; returns the rid, its hashed routing key, and the egress
/// receiver. Every request arrives with its final rid — a generate request from
/// `into_requests` (or the `HEALTH_CHECK_<uuid>` the health probe sets), a
/// control request from its constructor — so this only echoes it back.
pub(super) async fn submit(
    state: &AppState,
    kind: RequestKind,
) -> Result<(RidHash, String, mpsc::Receiver<EgressItem>), Response> {
    let rid = match &kind {
        // Generate rids are already final: `GenerateBody::into_requests` normalized the
        // client's, or minted one. Control requests have no client-facing rid.
        RequestKind::Generate(g) => g.rid.clone(),
        RequestKind::Control(c) => c.rid().to_string(),
    };
    // Generate rids can be client-supplied, so two in-flight requests may share
    // one. Detok `Register` is an insert-overwrite: the second would evict the
    // first's sink, 500 that client mid-generation, and deliver its remaining
    // chunks to the second's connection. Reject instead, as Python's
    // `TokenizerManager` does. The entry is dropped by the caller's `AbortGuard`.
    let mut lease = None;
    if matches!(kind, RequestKind::Generate(_)) {
        if !state
            .live_rids
            .lock()
            .expect("live_rids poisoned")
            .insert(rid.clone())
        {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("Duplicate request ID detected: {rid}"),
            )
                .into_response());
        }
        // RAII from here on. Every path out of this function that is NOT a
        // successful hand-off must release the rid, including the one that never
        // returns: `send_async` below is an await point, and a client disconnecting
        // there drops this future mid-flight. A leaked entry makes the rid
        // permanently unusable — every retry 400s — which defeats client-supplied
        // rids as idempotency keys and is drivable by saturating the inbox.
        lease = Some(RidLease {
            live_rids: state.live_rids.clone(),
            rid: rid.clone(),
        });
    }
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
        Ok(()) => {
            // Handed off: the caller's `AbortGuard` owns the release from here.
            if let Some(lease) = lease.as_mut() {
                lease.disarm();
            }
            Ok((id, rid, rx))
        }
        // `SendError` has a single meaning — the channel is disconnected.
        Err(_) => {
            tracing::error!(%rid, "tm inbox closed; request rejected");
            // Return 503 so the client can retry.
            Err((StatusCode::SERVICE_UNAVAILABLE, "service unavailable").into_response())
        }
    }
}

/// Holds a `live_rids` entry until the request is safely handed to the TM, then
/// hands ownership to the caller's [`AbortGuard`](super::guard::AbortGuard) via
/// [`disarm`](Self::disarm). Dropped undisarmed — an early return, or the future
/// being dropped at the `send_async` await — it releases the rid.
struct RidLease {
    live_rids: super::LiveRids,
    rid: String,
}

impl RidLease {
    fn disarm(&mut self) {
        self.rid.clear();
    }
}

impl Drop for RidLease {
    fn drop(&mut self) {
        if self.rid.is_empty() {
            return; // disarmed: the AbortGuard owns it now
        }
        if let Ok(mut live) = self.live_rids.lock() {
            live.remove(&self.rid);
        }
    }
}
