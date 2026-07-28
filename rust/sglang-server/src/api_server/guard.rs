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
    /// Rids still in flight. `Rid` carries its own partition key, so there is no
    /// separate routing value to keep alongside it.
    rids: Vec<Rid>,
}

impl AbortGuard {
    pub(super) fn new(senders: Senders, rid: Rid) -> Self {
        Self {
            senders,
            rids: vec![rid],
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
    pub(super) fn arm(&mut self, rid: Rid) {
        self.rids.push(rid);
    }

    /// Request finished naturally — don't abort it on drop.
    pub(super) fn disarm(&mut self, rid: &Rid) {
        self.rids.retain(|r| r != rid);
    }
}

impl Drop for AbortGuard {
    fn drop(&mut self) {
        // Report the abort and nothing more. There is no in-flight rid registry to
        // release from: `Rid::from_client` makes each client rid internally unique,
        // so a resubmit of the "same" rid is a different `Rid` and cannot be caught
        // up in this abort. That removes the ordering hazard split ownership created.
        //
        // The lane is unbounded, so this send only fails at shutdown, when the loop
        // is gone and nothing is generating anyway.
        for rid in self.rids.drain(..) {
            let _ = self.senders.abort.send(AbortSource::Guard(rid));
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

    /// A batch guard aborts exactly the rids still armed at drop — the ones whose
    /// requests never reached a terminal — and leaves the finished ones alone.
    /// A partial disarm is the normal case for a batch: some items complete, then
    /// the client disconnects, and only the stragglers need the scheduler stopped.
    #[test]
    fn guard_aborts_only_the_rids_still_armed() {
        let (abort_tx, abort_rx) = flume::unbounded();
        let done: Rid = "done".into();
        let mut guard = AbortGuard::new(senders_with_abort(abort_tx), done.clone());
        guard.arm("aborted".into());
        guard.disarm(&done); // finished naturally
        drop(guard);

        assert!(
            matches!(abort_rx.try_recv().unwrap(), AbortSource::Guard(r) if r.as_str() == "aborted")
        );
        assert!(
            abort_rx.try_recv().is_err(),
            "a disarmed rid must not be aborted"
        );
    }

    /// An armed guard aborts its rid on drop — exactly the cleanup a busy-skipped
    /// `/health_generate` probe relies on. It never sees a terminal frame here, so
    /// dropping the guard is the only path that deregisters its detok sink (via the
    /// ingress `on_abort`). Regression for the detok-entry leak per health probe.
    #[test]
    fn armed_guard_aborts_on_drop() {
        let (tm_tx, tm_rx) = flume::unbounded();
        drop(AbortGuard::new(senders_with_abort(tm_tx), "r7".into()));
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
        let mut guard = AbortGuard::new(senders_with_abort(tm_tx), "r9".into());
        guard.disarm(&id);
        drop(guard);
        assert!(tm_rx.try_recv().is_err(), "disarmed rid must not abort");
    }
}
