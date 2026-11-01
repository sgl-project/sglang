//! Graceful drain, the Rust side of Python's SIGTERM handling
//! (`TokenizerManager.sigterm_watchdog`).
//!
//! Once a drain begins, every health endpoint answers 503 while requests keep
//! being admitted, as in Python. The listeners close when no client request
//! is left in flight — or at once, skipping the wait, when the last health
//! probe had failed or the drain was forced (Python's `UnHealthy` and
//! `SGL_FORCE_SHUTDOWN` exits). There is no deadline: like Python, the
//! orchestrator's kill is the only bound on a drain.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tokio::sync::Notify;

#[derive(Default)]
pub(crate) struct DrainState {
    draining: AtomicBool,
    /// Skip the wait for in-flight requests (`SGL_FORCE_SHUTDOWN`).
    forced: AtomicBool,
    /// The last deep health probe stalled; a later healthy one clears it, as
    /// Python moves `server_status` between `UnHealthy` and `Up`.
    unhealthy: AtomicBool,
    /// Client generation calls alive (see [`InFlight`]).
    in_flight: AtomicUsize,
    /// Latched once a drain is first seen with nothing in flight, so every
    /// waiter (each listener's shutdown, the close deadline) agrees it ended
    /// even if a request on a still-open connection is admitted afterwards.
    drained: AtomicBool,
    changed: Notify,
}

/// How a drain ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DrainEnd {
    /// Every client request finished.
    Drained,
    /// The wait was skipped: the drain was forced or the engine unhealthy.
    Forced,
}

impl DrainState {
    /// Count one client request until the guard drops.
    pub(crate) fn track(self: &Arc<Self>) -> InFlight {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        InFlight(self.clone())
    }

    /// Start draining (idempotent); `force` skips the wait for in-flight
    /// requests.
    pub(crate) fn begin(&self, force: bool) {
        if force {
            self.forced.store(true, Ordering::SeqCst);
        }
        if !self.draining.swap(true, Ordering::SeqCst) {
            tracing::warn!(
                in_flight = self.in_flight.load(Ordering::SeqCst),
                "Drain requested: health now answers 503; draining requests and shutting down"
            );
        }
        self.changed.notify_waiters();
    }

    pub(crate) fn is_draining(&self) -> bool {
        self.draining.load(Ordering::SeqCst)
    }

    pub(crate) fn set_unhealthy(&self, unhealthy: bool) {
        if self.unhealthy.swap(unhealthy, Ordering::SeqCst) != unhealthy {
            self.changed.notify_waiters();
        }
    }

    /// Why a drain skips its wait, in Python's words; `None` if it does not.
    pub(crate) fn force_reason(&self) -> Option<&'static str> {
        if !self.is_draining() {
            None
        } else if self.forced.load(Ordering::SeqCst) {
            Some("Signal SIGTERM received while force shutdown flag set. Force exiting.")
        } else if self.unhealthy.load(Ordering::SeqCst) {
            Some("Signal SIGTERM received while health check failed. Force exiting.")
        } else {
            None
        }
    }

    fn end(&self) -> Option<DrainEnd> {
        if !self.is_draining() {
            None
        } else if self.force_reason().is_some() {
            Some(DrainEnd::Forced)
        } else if self.drained.load(Ordering::SeqCst) || self.in_flight.load(Ordering::SeqCst) == 0
        {
            // Wake a waiter that checked between the load and this latch.
            if !self.drained.swap(true, Ordering::SeqCst) {
                self.changed.notify_waiters();
            }
            Some(DrainEnd::Drained)
        } else {
            None
        }
    }

    /// Wait until `check` yields a value, re-checking on every state change.
    async fn wait_until<T>(&self, check: impl Fn(&Self) -> Option<T>) -> T {
        loop {
            // Register before checking, so a change between the check and
            // the await still wakes this waiter.
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(value) = check(self) {
                return value;
            }
            notified.await;
        }
    }

    /// Resolve once the drain has ended, and how.
    pub(crate) async fn ended(&self) -> DrainEnd {
        self.wait_until(Self::end).await
    }

    /// Resolve once the drain must stop waiting for in-flight requests. A
    /// probe that stalls after the requests drained still forces the close of
    /// connections the listener is finishing.
    pub(crate) async fn forced(&self) {
        self.wait_until(Self::force_reason).await;
    }
}

/// One client request alive, counted for the drain. Health probes are not
/// counted, matching Python, whose drain never waits for one.
pub(crate) struct InFlight(Arc<DrainState>);

impl Drop for InFlight {
    fn drop(&mut self) {
        if self.0.in_flight.fetch_sub(1, Ordering::SeqCst) == 1 && self.0.is_draining() {
            self.0.changed.notify_waiters();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    async fn settles<F: std::future::Future>(future: F) -> Option<F::Output> {
        tokio::time::timeout(Duration::from_millis(100), future)
            .await
            .ok()
    }

    /// A health probe that started before the drain and stalls during it
    /// still forces the drain, as Python's watchdog re-checks `UnHealthy` on
    /// every round of its drain loop.
    #[tokio::test]
    async fn probe_stalling_during_the_drain_forces_it() {
        let state = Arc::new(DrainState::default());
        let _request = state.track();
        state.begin(false);
        assert_eq!(settles(state.ended()).await, None);
        state.set_unhealthy(true);
        assert_eq!(settles(state.ended()).await, Some(DrainEnd::Forced));
        assert!(settles(state.forced()).await.is_some());
        assert_eq!(
            state.force_reason(),
            Some("Signal SIGTERM received while health check failed. Force exiting.")
        );
    }

    /// A request admitted after one waiter saw the drain end (on a connection
    /// the listener is still finishing) must not un-end it for the others.
    #[tokio::test]
    async fn drain_end_is_latched() {
        let state = Arc::new(DrainState::default());
        state.begin(false);
        assert_eq!(settles(state.ended()).await, Some(DrainEnd::Drained));
        let _late = state.track();
        assert_eq!(settles(state.ended()).await, Some(DrainEnd::Drained));
    }
}
