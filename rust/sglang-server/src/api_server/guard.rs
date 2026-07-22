//! Abort-on-disconnect guard for in-flight requests. Handlers arm a guard per
//! submitted rid; axum dropping the handler/SSE stream (client disconnected)
//! drops the guard, which aborts whatever wasn't disarmed (mirrors Python's
//! `is_disconnected` abort).

use crate::ids::RidHash;
use crate::runtime::channels::{Senders, TmEvent};

/// Aborts still-in-flight rids on drop. Each rid is disarmed on natural finish;
/// whatever remains at drop is aborted.
pub(super) struct AbortGuard {
    senders: Senders,
    /// `(routing key, rid string)` — the string is what `AbortReq` needs on the
    /// scheduler wire (unrecoverable from the hashed key), the key is what
    /// callers disarm by.
    rids: Vec<(RidHash, String)>,
}

impl AbortGuard {
    pub(super) fn new(senders: Senders, id: RidHash, rid: String) -> Self {
        Self {
            senders,
            rids: vec![(id, rid)],
        }
    }

    /// Guard covering no rids yet — a batch arms each as it's submitted so a
    /// mid-fan-out disconnect aborts every request already handed to the scheduler.
    pub(super) fn new_empty(senders: Senders) -> Self {
        Self {
            senders,
            rids: Vec::new(),
        }
    }

    /// Track a request for abort-on-drop.
    pub(super) fn arm(&mut self, id: RidHash, rid: String) {
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
        for (_, rid) in self.rids.drain(..) {
            let _ = self.senders.tm.try_send(TmEvent::Abort(rid));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
