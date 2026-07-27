//! Abort-on-disconnect guard for in-flight requests. Handlers arm a guard per
//! submitted rid; axum dropping the handler/SSE stream (client disconnected)
//! drops the guard, which aborts whatever wasn't disarmed (mirrors Python's
//! `is_disconnected` abort).

use crate::ids::RidHash;
use crate::tokenizer_manager::Senders;

/// Aborts still-in-flight rids on drop. Each rid is disarmed on natural finish;
/// whatever remains at drop is aborted.
pub(super) struct AbortGuard {
    senders: Senders,
    /// Registry to release on drop, plus every rid this guard ever covered —
    /// including disarmed ones, which are finished and must not stay "in flight".
    live_rids: super::LiveRids,
    owned: Vec<String>,
    /// `(routing key, rid string)` — the string is what `AbortReq` needs on the
    /// scheduler wire (unrecoverable from the hashed key), the key is what
    /// callers disarm by.
    rids: Vec<(RidHash, String)>,
}

impl AbortGuard {
    pub(super) fn new(
        senders: Senders,
        live_rids: super::LiveRids,
        id: RidHash,
        rid: String,
    ) -> Self {
        Self {
            senders,
            live_rids,
            owned: vec![rid.clone()],
            rids: vec![(id, rid)],
        }
    }

    /// Guard covering no rids yet — a batch arms each as it's submitted so a
    /// mid-fan-out disconnect aborts every request already handed to the scheduler.
    pub(super) fn new_empty(senders: Senders, live_rids: super::LiveRids) -> Self {
        Self {
            senders,
            live_rids,
            owned: Vec::new(),
            rids: Vec::new(),
        }
    }

    /// Track a request for abort-on-drop.
    pub(super) fn arm(&mut self, id: RidHash, rid: String) {
        self.owned.push(rid.clone());
        self.rids.push((id, rid));
    }

    /// Request finished naturally — don't abort it on drop.
    pub(super) fn disarm(&mut self, id: RidHash) {
        self.rids.retain(|(r, _)| *r != id);
    }
}

impl Drop for AbortGuard {
    fn drop(&mut self) {
        // Best-effort non-blocking abort per rid; a full/closed channel just drops
        // it (the request then finishes at EOS, only later).
        // A rid whose abort could NOT be queued must stay held. The inbox is
        // bounded and this send is best-effort, so under load the abort is dropped
        // while the scheduler keeps generating and the detok entry stays live —
        // releasing the rid there would let a resubmit pass the duplicate check and
        // overwrite that entry, which is the cross-client delivery the registry
        // exists to prevent. Holding it costs the client a 400 on retry; releasing
        // it costs another client's tokens.
        // Unbounded lane: this send only fails at shutdown, when the loop is gone
        // and nothing is generating anyway. So the release below is unconditional
        // again — no rid is held back, and none is released while its abort is
        // still undelivered.
        for (_, rid) in self.rids.drain(..) {
            let _ = self.senders.abort.send(rid);
        }
        // Release the in-flight rids so the client can reuse them. Every rid the
        // guard covered, disarmed or not — a disarmed one finished, which is
        // exactly when it stops being in flight.
        if let Ok(mut live) = self.live_rids.lock() {
            for rid in self.owned.drain(..) {
                live.remove(&rid);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn senders_with_abort(abort: flume::Sender<String>) -> Senders {
        Senders {
            tm: flume::unbounded().0,
            abort,
            tok: flume::unbounded().0,
            detok: vec![],
        }
    }

    /// An armed guard aborts its rid on drop — exactly the cleanup a busy-skipped
    /// `/health_generate` probe relies on. It never sees a terminal frame here, so
    /// dropping the guard is the only path that deregisters its detok sink (via the
    /// ingress `on_abort`). Regression for the detok-entry leak per health probe.
    /// The guard owns the `live_rids` release: without it a finished rid stays "in
    /// flight" forever and every retry 400s, which defeats client-supplied rids as
    /// idempotency keys. Disarmed rids count too — disarmed means finished.
    #[test]
    fn guard_releases_live_rids_on_drop() {
        let live: super::super::LiveRids = Default::default();
        live.lock().unwrap().insert("r1".to_string());
        live.lock().unwrap().insert("r2".to_string());
        let (tm_tx, _tm_rx) = flume::unbounded();
        let id1 = RidHash::from_rid("r1");
        let mut guard = AbortGuard::new(senders_with_abort(tm_tx), live.clone(), id1, "r1".into());
        guard.arm(RidHash::from_rid("r2"), "r2".to_string());
        guard.disarm(id1); // finished naturally — still must be released
        drop(guard);
        assert!(
            live.lock().unwrap().is_empty(),
            "every rid the guard covered must be released, disarmed or not"
        );
    }

    #[test]
    fn armed_guard_aborts_on_drop() {
        let (tm_tx, tm_rx) = flume::unbounded();
        drop(AbortGuard::new(
            senders_with_abort(tm_tx),
            Default::default(),
            RidHash::from_rid("r7"),
            "r7".to_string(),
        ));
        assert!(
            matches!(tm_rx.try_recv(), Ok(rid) if rid == "r7"),
            "armed guard must abort its rid on drop",
        );
        assert!(tm_rx.try_recv().is_err(), "exactly one abort");
    }

    /// A disarmed rid (finished naturally) is not aborted on drop.
    #[test]
    fn disarmed_guard_does_not_abort() {
        let (tm_tx, tm_rx) = flume::unbounded();
        let id = RidHash::from_rid("r9");
        let mut guard = AbortGuard::new(
            senders_with_abort(tm_tx),
            Default::default(),
            id,
            "r9".to_string(),
        );
        guard.disarm(id);
        drop(guard);
        assert!(tm_rx.try_recv().is_err(), "disarmed rid must not abort");
    }
}
