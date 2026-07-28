//! Request submission into the ingress pipeline, shared by every endpoint
//! module: mint the client-visible rid (uuid hex, Python-parity), build the
//! `Request`, and hand it to the TM with an egress receiver for the response.

use axum::{http::StatusCode, response::Response};
use tokio::sync::mpsc;

use super::AppState;
use crate::fsm::RequestState;
use crate::ids::Rid;
use crate::message::{EgressItem, EgressSink, Request, RequestKind};
use crate::tokenizer_manager::TmEvent;

/// Submit one request; returns the rid, its hashed routing key, and the egress
/// receiver. Every request arrives with its final rid — a generate request from
/// `into_requests` (or the `HEALTH_CHECK_<uuid>` the health probe sets), a
/// control request from its constructor — so this only echoes it back.
pub(super) async fn submit(
    state: &AppState,
    kind: RequestKind,
    // `stream`: the client is reading an SSE stream, so it expects 200 plus an
    // error frame rather than a 4xx — same rule `pre_submit_error` applies
    // everywhere else.
    stream: bool,
) -> Result<(Rid, mpsc::Receiver<EgressItem>), Response> {
    let rid = match &kind {
        // Generate rids are already final: `GenerateBody::into_requests` normalized the
        // client's, or minted one. Control requests have no client-facing rid.
        RequestKind::Generate(g) => g.rid.clone(),
        RequestKind::Control(c) => c.rid().into(),
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
            return Err(super::native_api::pre_submit_error(
                StatusCode::BAD_REQUEST,
                &format!("Duplicate request ID detected: {rid}"),
                stream,
            ));
        }
        // RAII from here on. Every path out of this function that is NOT a
        // successful hand-off must release the rid, including the one that never
        // returns: `send_async` below is an await point, and a client disconnecting
        // there drops this future mid-flight. A leaked entry makes the rid
        // permanently unusable — every retry 400s — which defeats client-supplied
        // rids as idempotency keys and is drivable by saturating the inbox.
        lease = Some(RidLease {
            live_rids: state.live_rids.clone(),
            rid: Some(rid.clone()),
        });
    }
    // Async-aware send so a full TM inbox yields (backpressure) instead of parking
    // a thread; Err only when the inbox is closed (shutdown).
    let (tx, rx) = mpsc::channel::<EgressItem>(state.egress_buf);
    let request = Request {
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
            Ok((rid, rx))
        }
        // `SendError` has a single meaning — the channel is disconnected.
        Err(_) => {
            tracing::error!(%rid, "tm inbox closed; request rejected");
            // Return 503 so the client can retry.
            Err(super::native_api::pre_submit_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "service unavailable",
                stream,
            ))
        }
    }
}

/// Holds a `live_rids` entry until the request is safely handed to the TM, then
/// hands ownership to the caller's [`AbortGuard`](super::guard::AbortGuard) via
/// [`disarm`](Self::disarm). Dropped undisarmed — an early return, or the future
/// being dropped at the `send_async` await — it releases the rid.
struct RidLease {
    live_rids: super::LiveRids,
    /// `None` once disarmed. An `Option`, not an emptied `String`: a client may
    /// send `rid: ""`, which is a real (if odd) rid — with a "cleared" sentinel it
    /// was indistinguishable from a disarmed lease, so the 503 and cancellation
    /// paths left `""` in the set forever and every later `rid: ""` got a 400.
    rid: Option<Rid>,
}

impl RidLease {
    fn disarm(&mut self) {
        self.rid = None;
    }
}

impl Drop for RidLease {
    fn drop(&mut self) {
        let Some(rid) = self.rid.take() else {
            return; // disarmed: the AbortGuard owns it now
        };
        if let Ok(mut live) = self.live_rids.lock() {
            live.remove(&rid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::GenerateRequest;
    use crate::tokenizer_manager::{LiveRids, Senders};
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;
    use std::task::{Context, Waker};

    /// `AppState` with a caller-supplied tm inbox and rid registry, so a test can
    /// observe both sides of a submit.
    fn state_with(tm: flume::Sender<TmEvent>, live_rids: LiveRids) -> AppState {
        AppState {
            senders: Senders {
                tm,
                abort: flume::unbounded().0,
                tok: flume::unbounded().0,
                detok: vec![flume::unbounded().0],
            },
            egress_buf: 8,
            server_args: Arc::new(
                crate::runtime::ServerArgs::from_json(r#"{"model_path": "/m"}"#).unwrap(),
            ),
            egress_activity: Arc::new(AtomicU64::new(0)),
            live_rids,
        }
    }

    fn generate(rid: &str) -> RequestKind {
        RequestKind::Generate(Box::new(GenerateRequest {
            rid: rid.into(),
            input_ids: Some(vec![1, 2, 3]),
            ..Default::default()
        }))
    }

    /// Two in-flight requests may not share a rid: detok `Register` is an
    /// insert-overwrite, so the second would evict the first's sink, 500 that
    /// client mid-generation, and deliver its remaining chunks to the second
    /// connection. Python raises the same way (`tokenizer_manager.py`,
    /// "Duplicate request ID detected").
    ///
    /// The second half is the part that is easy to get wrong: the REJECTED request
    /// must not touch the holder's registry entry on its way out. `RidLease` is
    /// only constructed after a successful `insert`, so the rejection path has
    /// nothing to drop — this pins that.
    #[tokio::test]
    async fn duplicate_rid_is_rejected_and_the_holder_keeps_its_entry() {
        let live: LiveRids = Default::default();
        let (tm_tx, tm_rx) = flume::unbounded();
        let state = state_with(tm_tx, live.clone());

        submit(&state, generate("dup"), false)
            .await
            .expect("first submit succeeds");
        assert!(live.lock().unwrap().contains(&Rid::from("dup")));

        let Err(err) = submit(&state, generate("dup"), false).await else {
            panic!("second submit with the same rid must be rejected");
        };
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
        assert!(
            live.lock().unwrap().contains(&Rid::from("dup")),
            "the rejected duplicate must not release the holder's entry"
        );
        // Exactly one request reached the scheduler.
        assert!(tm_rx.try_recv().is_ok());
        assert!(tm_rx.try_recv().is_err());
    }

    /// A closed tm inbox (shutdown) returns 503 — and must release the rid on the
    /// way out. `submit` inserts before the send, and the `AbortGuard` that
    /// normally owns the release is only built by the caller on `Ok`, so an early
    /// return with no lease would strand the rid forever: every retry 400s, which
    /// defeats client-supplied rids as idempotency keys.
    #[tokio::test]
    async fn unavailable_tm_releases_the_rid() {
        let live: LiveRids = Default::default();
        let (tm_tx, tm_rx) = flume::unbounded();
        drop(tm_rx); // inbox closed
        let state = state_with(tm_tx, live.clone());

        let Err(err) = submit(&state, generate("r"), false).await else {
            panic!("a closed tm inbox must be a 503");
        };
        assert_eq!(err.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            live.lock().unwrap().is_empty(),
            "503 must not strand the rid in the registry"
        );
    }

    /// The hard one: `send_async` is an await point, so a client disconnecting
    /// while the inbox is full drops this future mid-flight — no early return runs.
    /// Only `RidLease`'s `Drop` releases the rid here, which is why the lease is
    /// RAII rather than a cleanup on each error path.
    ///
    /// `rid: ""` on purpose: the lease used to signal "disarmed" by clearing the
    /// string, so an empty rid was indistinguishable from a handed-off one and this
    /// path leaked it permanently.
    #[tokio::test]
    async fn cancelled_submit_releases_the_rid_even_when_empty() {
        let live: LiveRids = Default::default();
        // Capacity 1, pre-filled: `send_async` must pend.
        let (tm_tx, _tm_rx) = flume::bounded(1);
        tm_tx
            .try_send(TmEvent::Ingress(match generate("filler") {
                RequestKind::Generate(g) => Request {
                    rid: "filler".into(),
                    state: RequestState::Received,
                    sink: EgressSink::Local(mpsc::channel(1).0),
                    kind: RequestKind::Generate(g),
                },
                _ => unreachable!(),
            }))
            .unwrap();
        let state = state_with(tm_tx, live.clone());

        // Keep the future ALIVE across the poll: asserting only that the registry
        // is empty afterwards is satisfied just as well by a rid that was never
        // inserted, so a `submit` that reserved AFTER the send — or that gained an
        // await point BEFORE the reservation — would pass vacuously. Pinning the
        // rid as present AT the pend point is what makes this a real assertion.
        let mut fut = Box::pin(submit(&state, generate(""), false));
        let mut cx = Context::from_waker(Waker::noop());
        assert!(
            fut.as_mut().poll(&mut cx).is_pending(),
            "the full inbox must make submit pend"
        );
        assert!(
            live.lock().unwrap().contains(&Rid::from("")),
            "the rid must already be reserved when submit parks on the send — \
             reserving after it leaves a window where a duplicate is admitted"
        );
        drop(fut); // the client disconnected
        assert!(
            live.lock().unwrap().is_empty(),
            "a submit cancelled at the send_async await must release its rid"
        );
    }
}
