// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

// NOTE: `opened_at` uses `tokio::time::Instant` rather than `std::time::Instant`
// so that `#[tokio::test(start_paused = true)]` + `tokio::time::advance` can
// move the clock forward in tests. `std::time::Instant` is not paused by
// tokio's mock clock, so `elapsed()` would always return near-zero inside a
// paused-time test, preventing the Open → HalfOpen transition from being
// exercised deterministically.

use std::num::NonZeroU32;
use std::sync::Mutex;
use std::time::Duration;
use tokio::time::Instant;

/// Consistent `(admit, state_code)` pair read under a single breaker lock.
/// See [`CircuitBreaker::snapshot`].
#[derive(Debug, Clone, Copy)]
pub struct CircuitSnapshot {
    /// Would the breaker admit a request right now (`would_allow` semantics).
    pub admit: bool,
    /// State code: 0=closed, 1=open, 2=half_open.
    pub state_code: u8,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub threshold: NonZeroU32,
    pub cool_down: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            threshold: NonZeroU32::new(3).expect("3 is non-zero"),
            cool_down: Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum State {
    Closed,
    Open { opened_at: Instant },
    HalfOpen { probe_in_flight: bool },
}

#[derive(Debug)]
struct Inner {
    state: State,
    consecutive_failures: u32,
}

#[derive(Debug)]
pub struct CircuitBreaker {
    inner: Mutex<Inner>,
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    pub fn new() -> Self {
        Self::with_config(CircuitBreakerConfig::default())
    }

    pub fn with_config(config: CircuitBreakerConfig) -> Self {
        Self {
            inner: Mutex::new(Inner {
                state: State::Closed,
                consecutive_failures: 0,
            }),
            config,
        }
    }

    /// Non-mutating predicate: would [`allow`] return `true` if called
    /// right now?
    ///
    /// Used by enumeration / filter paths (e.g.
    /// [`crate::workers::registry::WorkerRegistry::healthy_workers_for`])
    /// that need to inspect breaker readiness without claiming a half-open
    /// probe slot. Calling `allow()` for filtering would leak probe slots
    /// to unselected candidates and starve dispatch: the policy would
    /// enumerate a worker as "healthy", then the proxy's `allow()` at
    /// dispatch time would see `probe_in_flight=true` and reject.
    ///
    /// Semantics:
    /// - `Closed` → `true`
    /// - `Open` past `cool_down` → `true` (a probe slot is available)
    /// - `Open` within `cool_down` → `false`
    /// - `HalfOpen { probe_in_flight: true }` → `false`
    /// - `HalfOpen { probe_in_flight: false }` → `true`
    pub fn would_allow(&self) -> bool {
        let g = self.inner.lock().unwrap();
        match g.state {
            State::Closed => true,
            State::Open { opened_at } => opened_at.elapsed() >= self.config.cool_down,
            State::HalfOpen { probe_in_flight } => !probe_in_flight,
        }
    }

    /// Single-lock snapshot of `(admit, state_code)` for the `/metrics`
    /// scrape path, feeding `sgl_router_worker_health` and
    /// `sgl_router_worker_cb_state` (0=closed, 1=open, 2=half_open). Reading
    /// admit and state separately would take the lock twice and could observe
    /// a transition between the two reads, emitting a self-contradictory pair
    /// for one scrape. This reads both under one lock so they always agree.
    ///
    /// Note `admit` and `state_code` can still legitimately disagree within
    /// a *consistent* read: an `Open` breaker past its cooldown returns
    /// `admit=true` (a probe slot is available) while `state_code=1`. That
    /// is the breaker's real state, not a race.
    pub fn snapshot(&self) -> CircuitSnapshot {
        let g = self.inner.lock().unwrap();
        let (admit, state_code) = match g.state {
            State::Closed => (true, 0),
            State::Open { opened_at } => (opened_at.elapsed() >= self.config.cool_down, 1),
            State::HalfOpen { probe_in_flight } => (!probe_in_flight, 2),
        };
        CircuitSnapshot { admit, state_code }
    }

    /// True if a request may proceed. Mutates state when transitioning
    /// from Open → HalfOpen.
    pub fn allow(&self) -> bool {
        let mut g = self.inner.lock().unwrap();
        match g.state {
            State::Closed => true,
            State::Open { opened_at } => {
                if opened_at.elapsed() >= self.config.cool_down {
                    g.state = State::HalfOpen {
                        probe_in_flight: true,
                    };
                    true
                } else {
                    false
                }
            }
            State::HalfOpen { probe_in_flight } => {
                if probe_in_flight {
                    false
                } else {
                    g.state = State::HalfOpen {
                        probe_in_flight: true,
                    };
                    true
                }
            }
        }
    }

    pub fn record_success(&self) {
        let mut g = self.inner.lock().unwrap();
        g.consecutive_failures = 0;
        g.state = State::Closed;
    }

    pub fn record_failure(&self) {
        let mut g = self.inner.lock().unwrap();
        match g.state {
            State::Closed | State::HalfOpen { .. } => {
                g.consecutive_failures += 1;
                if g.consecutive_failures >= self.config.threshold.get() {
                    g.state = State::Open {
                        opened_at: Instant::now(),
                    };
                }
            }
            State::Open { .. } => {
                // Already open: ticking consecutive_failures or refreshing opened_at
                // would pin us Open during a failure storm. The cool_down is
                // measured from first-open; failures during Open are ignored.
            }
        }
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cb(threshold: u32, cool_down_secs: u64) -> CircuitBreaker {
        CircuitBreaker::with_config(CircuitBreakerConfig {
            threshold: NonZeroU32::new(threshold).unwrap(),
            cool_down: Duration::from_secs(cool_down_secs),
        })
    }

    #[test]
    fn state_code_is_closed_by_default() {
        assert_eq!(CircuitBreaker::new().snapshot().state_code, 0);
    }

    #[test]
    fn state_code_reports_open_only_after_threshold() {
        let b = cb(2, 30);
        b.record_failure();
        assert_eq!(
            b.snapshot().state_code,
            0,
            "1 failure < threshold 2 stays closed",
        );
        b.record_failure();
        assert_eq!(
            b.snapshot().state_code,
            1,
            "reaching threshold opens the breaker",
        );
    }

    #[tokio::test(start_paused = true)]
    async fn state_code_reports_half_open_after_cooldown_probe() {
        let b = cb(1, 10);
        b.record_failure();
        assert_eq!(
            b.snapshot().state_code,
            1,
            "threshold=1 opens on first failure"
        );
        tokio::time::advance(Duration::from_secs(11)).await;
        // `allow()` claims the probe slot, transitioning Open -> HalfOpen.
        assert!(b.allow());
        assert_eq!(b.snapshot().state_code, 2);
    }

    #[tokio::test(start_paused = true)]
    async fn snapshot_reports_open_but_admittable_after_cooldown() {
        // The contract the scrape path depends on: a single read can show an
        // Open breaker (state_code=1) that nonetheless admits (admit=true)
        // once cooldown has elapsed — and the two halves never disagree due
        // to a torn read because they come from one lock acquisition.
        let b = cb(1, 10);
        b.record_failure();
        let s = b.snapshot();
        assert!(!s.admit, "open within cooldown must not admit");
        assert_eq!(s.state_code, 1);

        tokio::time::advance(Duration::from_secs(11)).await;
        let s = b.snapshot();
        assert!(s.admit, "open past cooldown admits a probe");
        assert_eq!(s.state_code, 1, "...but is still reported as open");
    }
}
