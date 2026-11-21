//! Graceful drain, the Rust side of Python's SIGTERM handling
//! (`TokenizerManager.sigterm_watchdog`).
//!
//! Once a drain begins, every health endpoint answers 503 while requests keep
//! being admitted, as in Python. The listeners close when no client request
//! is left in flight — or at once, skipping the wait, when the last health
//! probe had failed or the drain was forced (Python's `UnHealthy` and
//! `SGLANG_FORCE_SHUTDOWN` exits). There is no deadline: like Python, the
//! orchestrator's kill is the only bound on a drain.
//!
//! Prefill/Decode also waits for a fresh scheduler-idle report after the drain
//! begins (see [`DrainState::report_scheduler_idle`]), so KV handoff and local
//! PD queues can finish after the frontend in-flight count has already hit
//! zero. Unified serving keeps the original in-flight-only wait.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::utils::inflight::{InFlight, InFlightCount};

#[derive(Default)]
pub(crate) struct DrainState {
    draining: AtomicBool,
    /// Skip the wait for in-flight requests (`SGLANG_FORCE_SHUTDOWN`).
    forced: AtomicBool,
    /// The last deep health probe stalled; a later healthy one clears it, as
    /// Python moves `server_status` between `UnHealthy` and `Up`.
    unhealthy: AtomicBool,
    /// Client generation / P/D pipeline work alive (see [`InFlight`]).
    in_flight: Arc<InFlightCount>,
    /// Prefill/Decode waits for [`Self::report_scheduler_idle`] after begin.
    requires_scheduler_idle: bool,
    /// Serialize P/D admission with the compound zero-work/idle close check.
    /// Independent atomics alone could latch between tracking and validation.
    admission: Mutex<()>,
    /// Fresh idle report observed after the drain began (and since the last
    /// [`Self::track`]); cleared when tracking invalidates a stale report.
    scheduler_idle: AtomicBool,
    /// Latched once a drain is first seen with nothing in flight (and, for
    /// P/D, a fresh scheduler-idle report), so every waiter (each listener's
    /// shutdown, the close deadline) agrees it ended even if a request on a
    /// still-open connection is admitted afterwards.
    drained: AtomicBool,
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
    /// `requires_scheduler_idle` is set for Prefill/Decode serving
    /// ([`CoreConfig::is_disaggregation`](crate::api_server::handlers::CoreConfig)).
    pub(crate) fn new(requires_scheduler_idle: bool) -> Self {
        Self {
            requires_scheduler_idle,
            ..Self::default()
        }
    }

    /// Count one client request (or P/D pipeline hop) until the guard drops.
    /// Invalidates any prior scheduler-idle report so a normal P/D drain must
    /// observe a fresh idle after this work.
    pub(crate) fn track(self: &Arc<Self>) -> InFlight {
        let _admission = self.admission_guard();
        self.invalidate_scheduler_idle();
        self.in_flight.track()
    }

    /// Start draining (idempotent); `force` skips the wait for in-flight
    /// requests (and the P/D idle gate).
    pub(crate) fn begin(&self, force: bool) -> bool {
        let _admission = self.admission_guard();
        if force {
            self.forced.store(true, Ordering::SeqCst);
        }
        let first = !self.draining.swap(true, Ordering::SeqCst);
        if first {
            // A pre-drain idle report is stale; wait for a fresh one.
            self.invalidate_scheduler_idle();
            tracing::warn!(
                in_flight = self.in_flight.get(),
                "Drain requested: health now answers 503; draining requests and shutting down"
            );
        }
        self.in_flight.changed.notify_waiters();
        first
    }

    pub(crate) fn is_draining(&self) -> bool {
        self.draining.load(Ordering::SeqCst)
    }

    /// Whether a normal drain has already latched its end. Prefill/Decode
    /// refuses later generations once this is set; unified serving still
    /// admits them on connections the listener is finishing.
    pub(crate) fn is_latched(&self) -> bool {
        self.drained.load(Ordering::SeqCst)
    }

    /// Whether the drain was begun forced (`SGLANG_FORCE_SHUTDOWN`). A failed
    /// health probe also skips the wait, but is not a request to force.
    pub(crate) fn is_forced(&self) -> bool {
        self.forced.load(Ordering::SeqCst)
    }

    pub(crate) fn set_unhealthy(&self, unhealthy: bool) {
        if self.unhealthy.swap(unhealthy, Ordering::SeqCst) != unhealthy {
            self.in_flight.changed.notify_waiters();
        }
    }

    /// Prefill/Decode quiescence: `true` once the scheduler reports fully idle
    /// after the drain began. Ignored for unified serving and for
    /// force/unhealthy drains.
    pub(crate) fn report_scheduler_idle(&self, idle: bool) {
        if !self.requires_scheduler_idle {
            return;
        }
        let _admission = self.admission_guard();
        if idle && !self.is_draining() {
            return;
        }
        if self.scheduler_idle.swap(idle, Ordering::SeqCst) != idle {
            self.in_flight.changed.notify_waiters();
        }
    }

    fn invalidate_scheduler_idle(&self) {
        if self.requires_scheduler_idle {
            self.scheduler_idle.store(false, Ordering::SeqCst);
        }
    }

    fn scheduler_quiesced(&self) -> bool {
        !self.requires_scheduler_idle || self.scheduler_idle.load(Ordering::SeqCst)
    }

    fn admission_guard(&self) -> Option<MutexGuard<'_, ()>> {
        self.requires_scheduler_idle
            .then(|| self.admission.lock().expect("P/D drain admission lock"))
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
        let _admission = self.admission_guard();
        if !self.is_draining() {
            None
        } else if self.force_reason().is_some() {
            Some(DrainEnd::Forced)
        } else if self.drained.load(Ordering::SeqCst)
            || (self.in_flight.get() == 0 && self.scheduler_quiesced())
        {
            // Wake a waiter that checked between the load and this latch.
            if !self.drained.swap(true, Ordering::SeqCst) {
                self.in_flight.changed.notify_waiters();
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
            let notified = self.in_flight.changed.notified();
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
        assert!(!state.is_forced(), "a failed probe is not a forced drain");
        state.begin(true);
        assert!(state.is_forced());
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

    /// Prefill/Decode waits for a fresh idle report after begin; a pre-drain
    /// idle is ignored, and tracking invalidates a stale report.
    #[tokio::test]
    async fn pd_drain_waits_for_fresh_scheduler_idle() {
        let state = Arc::new(DrainState::new(true));
        state.report_scheduler_idle(true);
        state.begin(false);
        assert_eq!(settles(state.ended()).await, None);
        state.report_scheduler_idle(true);
        assert_eq!(settles(state.ended()).await, Some(DrainEnd::Drained));
        assert!(state.is_latched());
    }

    #[tokio::test]
    async fn pd_tracking_invalidates_stale_idle_report() {
        let state = Arc::new(DrainState::new(true));
        state.begin(false);
        state.report_scheduler_idle(true);
        let _work = state.track();
        assert_eq!(settles(state.ended()).await, None);
        drop(_work);
        assert_eq!(settles(state.ended()).await, None);
        state.report_scheduler_idle(true);
        assert_eq!(settles(state.ended()).await, Some(DrainEnd::Drained));
    }

    #[tokio::test]
    async fn pd_force_and_unhealthy_bypass_idle_gate() {
        let forced = Arc::new(DrainState::new(true));
        let _keep = forced.track();
        forced.begin(true);
        assert_eq!(settles(forced.ended()).await, Some(DrainEnd::Forced));

        let unhealthy = Arc::new(DrainState::new(true));
        let _keep = unhealthy.track();
        unhealthy.begin(false);
        unhealthy.set_unhealthy(true);
        assert_eq!(settles(unhealthy.ended()).await, Some(DrainEnd::Forced));
    }

    /// An orphan pipeline guard (reader dropped while the request is still on
    /// the to-scheduler ring) keeps the P/D drain open until it is dropped.
    #[tokio::test]
    async fn orphan_pipeline_guard_holds_pd_drain() {
        let state = Arc::new(DrainState::new(true));
        state.begin(false);
        state.report_scheduler_idle(true);
        let pipeline = state.track();
        assert_eq!(settles(state.ended()).await, None);
        drop(pipeline);
        assert_eq!(settles(state.ended()).await, None);
        state.report_scheduler_idle(true);
        assert_eq!(settles(state.ended()).await, Some(DrainEnd::Drained));
    }
}
