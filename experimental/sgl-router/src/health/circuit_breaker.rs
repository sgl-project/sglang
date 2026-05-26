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
