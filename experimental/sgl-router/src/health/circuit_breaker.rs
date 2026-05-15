// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

// NOTE: `opened_at` uses `tokio::time::Instant` rather than `std::time::Instant`
// so that `#[tokio::test(start_paused = true)]` + `tokio::time::advance` can
// move the clock forward in tests. `std::time::Instant` is not paused by
// tokio's mock clock, so `elapsed()` would always return near-zero inside a
// paused-time test, preventing the Open → HalfOpen transition from being
// exercised deterministically.

use std::sync::Mutex;
use std::time::Duration;
use tokio::time::Instant;

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub threshold: u32,
    pub cool_down: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            threshold: 3,
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
                if g.consecutive_failures >= self.config.threshold {
                    g.state = State::Open {
                        opened_at: Instant::now(),
                    };
                }
            }
            State::Open { .. } => {
                // Already open — refresh opened_at to extend the cool-down.
                g.state = State::Open {
                    opened_at: Instant::now(),
                };
            }
        }
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}
