//! Abort-on-disconnect guard for in-flight requests. Handlers arm a guard per
//! submitted rid; axum dropping the handler/SSE stream (client disconnected)
//! drops the guard, which aborts whatever wasn't disarmed (mirrors Python's
//! `is_disconnected` abort).

use crate::ids::Rid;
use crate::tokenizer_manager::{AbortSource, Senders};

/// Aborts still-in-flight rids on drop. Each rid is disarmed on natural finish;
/// whatever remains at drop is aborted.
pub(super) struct AbortGuard {
    senders: Senders,
    /// Registry to release on drop, plus every rid this guard ever covered —
    /// including disarmed ones, which are finished and must not stay "in flight".
    live_rids: super::LiveRids,
    owned: Vec<Rid>,
    /// Rids still in flight. `Rid` carries its own partition key, so there is no
    /// separate routing value to keep alongside it.
    rids: Vec<Rid>,
}

impl AbortGuard {
    pub(super) fn new(senders: Senders, live_rids: super::LiveRids, rid: Rid) -> Self {
        Self {
            senders,
            live_rids,
            owned: vec![rid.clone()],
            rids: vec![rid],
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
    pub(super) fn arm(&mut self, rid: Rid) {
        self.owned.push(rid.clone());
        self.rids.push(rid);
    }

    /// Request finished naturally — don't abort it on drop.
    pub(super) fn disarm(&mut self, rid: &Rid) {
        self.rids.retain(|r| r != rid);
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
        // Aborted rids are released by `Ingress::on_abort`, after the deregister
        // and the ring push actually happen. Releasing them here would order the
        // send rather than the effect, and a retry could then register ahead of the
        // stale abort.
        let aborted: Vec<Rid> = self.rids.drain(..).collect();
        for rid in &aborted {
            let _ = self.senders.abort.send(AbortSource::Guard(rid.clone()));
        }
        // Disarmed rids finished naturally: no abort was sent for them, so nothing
        // downstream will ever release them.
        self.owned.retain(|rid| !aborted.contains(rid));
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

    fn senders_with_abort(abort: flume::Sender<AbortSource>) -> Senders {
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
    /// Release ownership is SPLIT, and that split is the fix for the
    /// disconnect/resubmit race: a rid the guard aborted stays held until
    /// `Ingress::on_abort` has actually deregistered it and pushed the `AbortReq`.
    /// Releasing it here would order the send, not the effect, and a retry could
    /// register ahead of the stale abort. A DISARMED rid finished naturally — no
    /// abort was sent, so nothing downstream would ever release it.
    #[test]
    fn guard_releases_only_the_rids_it_did_not_abort() {
        let live: crate::tokenizer_manager::LiveRids = Default::default();
        live.lock().unwrap().insert("done".into());
        live.lock().unwrap().insert("aborted".into());
        let (abort_tx, abort_rx) = flume::unbounded();
        let done: Rid = "done".into();
        let mut guard = AbortGuard::new(senders_with_abort(abort_tx), live.clone(), "done".into());
        guard.arm("aborted".into());
        guard.disarm(&done); // finished naturally
        drop(guard);

        let held = live.lock().unwrap();
        assert!(
            !held.contains(&Rid::from("done")),
            "a finished rid must be released here"
        );
        assert!(
            held.contains(&Rid::from("aborted")),
            "an aborted rid stays held until on_abort has issued the deregister"
        );
        assert!(
            matches!(abort_rx.try_recv().unwrap(), AbortSource::Guard(r) if r.as_str() == "aborted")
        );
    }

    #[test]
    fn armed_guard_aborts_on_drop() {
        let (tm_tx, tm_rx) = flume::unbounded();
        drop(AbortGuard::new(
            senders_with_abort(tm_tx),
            Default::default(),
            "r7".into(),
        ));
        assert!(
            matches!(tm_rx.try_recv(), Ok(AbortSource::Guard(rid)) if rid.as_str() == "r7"),
            "armed guard must abort its rid on drop",
        );
        assert!(tm_rx.try_recv().is_err(), "exactly one abort");
    }

    /// A disarmed rid (finished naturally) is not aborted on drop.
    #[test]
    fn disarmed_guard_does_not_abort() {
        let (tm_tx, tm_rx) = flume::unbounded();
        let id = Rid::from("r9");
        let mut guard = AbortGuard::new(senders_with_abort(tm_tx), Default::default(), "r9".into());
        guard.disarm(&id);
        drop(guard);
        assert!(tm_rx.try_recv().is_err(), "disarmed rid must not abort");
    }
}
