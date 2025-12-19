use std::{
    sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering},
    time::{Duration, Instant},
};

use tracing::info;

use crate::observability::metrics::Metrics;

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures to open the circuit
    pub failure_threshold: u32,
    /// Success threshold to close circuit from half-open
    pub success_threshold: u32,
    /// Duration to wait before attempting half-open
    pub timeout_duration: Duration,
    /// Time window for failure counting
    pub window_duration: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout_duration: Duration::from_secs(30),
            window_duration: Duration::from_secs(60),
        }
    }
}

/// Circuit breaker state constants for atomic storage
const STATE_CLOSED: u8 = 0;
const STATE_OPEN: u8 = 1;
const STATE_HALF_OPEN: u8 = 2;

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation - requests are allowed
    Closed,
    /// Circuit is open - requests are rejected
    Open,
    /// Testing if service has recovered - limited requests allowed
    HalfOpen,
}

impl std::fmt::Display for CircuitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitState::Closed => write!(f, "Closed"),
            CircuitState::Open => write!(f, "Open"),
            CircuitState::HalfOpen => write!(f, "HalfOpen"),
        }
    }
}

impl CircuitState {
    pub fn as_str(&self) -> &'static str {
        match self {
            CircuitState::Closed => "closed",
            CircuitState::Open => "open",
            CircuitState::HalfOpen => "half_open",
        }
    }

    pub fn to_int(&self) -> u8 {
        match self {
            CircuitState::Closed => STATE_CLOSED,
            CircuitState::Open => STATE_OPEN,
            CircuitState::HalfOpen => STATE_HALF_OPEN,
        }
    }

    fn from_int(v: u8) -> Self {
        match v {
            STATE_CLOSED => CircuitState::Closed,
            STATE_OPEN => CircuitState::Open,
            STATE_HALF_OPEN => CircuitState::HalfOpen,
            _ => CircuitState::Closed, // Default to closed for safety
        }
    }
}

/// Get current time as milliseconds since an arbitrary epoch.
/// Uses Instant for monotonic time, converting to ms for atomic storage.
#[inline]
fn now_ms() -> u64 {
    // Use a static reference point for consistent timing
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_millis() as u64
}

/// Circuit breaker implementation using lock-free atomics for hot paths.
///
/// This implementation avoids RwLock contention by using atomic operations
/// for state checks (the most common operation). Only state transitions
/// use compare-and-swap which is still lock-free.
#[derive(Debug)]
pub struct CircuitBreaker {
    /// Circuit state stored as atomic u8 (0=Closed, 1=Open, 2=HalfOpen)
    state: AtomicU8,
    consecutive_failures: AtomicU32,
    consecutive_successes: AtomicU32,
    total_failures: AtomicU64,
    total_successes: AtomicU64,
    /// Last failure time in milliseconds (from now_ms())
    last_failure_time_ms: AtomicU64,
    /// Last state change time in milliseconds (from now_ms())
    last_state_change_ms: AtomicU64,
    config: CircuitBreakerConfig,
    metric_label: String,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with default configuration
    pub fn new() -> Self {
        Self::with_config_and_label(CircuitBreakerConfig::default(), String::new())
    }

    /// Create a new circuit breaker with custom configuration and metric label
    pub fn with_config_and_label(config: CircuitBreakerConfig, metric_label: String) -> Self {
        let init_state = CircuitState::Closed;
        Metrics::set_worker_cb_state(&metric_label, init_state.to_int());
        Self {
            state: AtomicU8::new(STATE_CLOSED),
            consecutive_failures: AtomicU32::new(0),
            consecutive_successes: AtomicU32::new(0),
            total_failures: AtomicU64::new(0),
            total_successes: AtomicU64::new(0),
            last_failure_time_ms: AtomicU64::new(0),
            last_state_change_ms: AtomicU64::new(now_ms()),
            config,
            metric_label,
        }
    }

    /// Get the metric label
    pub fn metric_label(&self) -> &str {
        &self.metric_label
    }

    /// Check if a request can be executed (lock-free hot path)
    #[inline]
    pub fn can_execute(&self) -> bool {
        let state = self.state();
        match state {
            CircuitState::Closed => true,
            CircuitState::Open => false,
            CircuitState::HalfOpen => true,
        }
    }

    /// Get the current state (lock-free)
    #[inline]
    pub fn state(&self) -> CircuitState {
        self.check_and_update_state_returning()
    }

    /// Check and update state, returning the current state (lock-free)
    #[inline]
    fn check_and_update_state_returning(&self) -> CircuitState {
        let current_state_int = self.state.load(Ordering::Acquire);
        let current_state = CircuitState::from_int(current_state_int);

        if current_state == CircuitState::Open {
            let last_change_ms = self.last_state_change_ms.load(Ordering::Acquire);
            let elapsed_ms = now_ms().saturating_sub(last_change_ms);
            let timeout_ms = self.config.timeout_duration.as_millis() as u64;

            if elapsed_ms >= timeout_ms {
                // Try to transition to HalfOpen using CAS
                if self
                    .state
                    .compare_exchange(
                        STATE_OPEN,
                        STATE_HALF_OPEN,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    self.last_state_change_ms.store(now_ms(), Ordering::Release);
                    self.consecutive_failures.store(0, Ordering::Release);
                    self.consecutive_successes.store(0, Ordering::Release);

                    info!("Circuit breaker state transition: open -> half_open");
                    Metrics::record_worker_cb_transition(&self.metric_label, "open", "half_open");
                    Metrics::set_worker_cb_state(&self.metric_label, STATE_HALF_OPEN);
                    self.publish_gauge_metrics();
                    return CircuitState::HalfOpen;
                }
                // Another thread already transitioned, re-read the state
                return CircuitState::from_int(self.state.load(Ordering::Acquire));
            }
        }
        current_state
    }

    /// Record the outcome of a request
    pub fn record_outcome(&self, success: bool) {
        if success {
            self.record_success();
        } else {
            self.record_failure();
        }

        let outcome_str = if success { "success" } else { "failure" };
        Metrics::record_worker_cb_outcome(&self.metric_label, outcome_str);
        self.publish_gauge_metrics();
    }

    /// Record a successful request
    pub fn record_success(&self) {
        self.total_successes.fetch_add(1, Ordering::Relaxed);
        self.consecutive_failures.store(0, Ordering::Release);
        let successes = self.consecutive_successes.fetch_add(1, Ordering::AcqRel) + 1;

        let current_state = CircuitState::from_int(self.state.load(Ordering::Acquire));

        match current_state {
            CircuitState::HalfOpen => {
                if successes >= self.config.success_threshold {
                    self.transition_to(CircuitState::Closed);
                }
            }
            CircuitState::Closed => {}
            CircuitState::Open => {
                tracing::warn!("Success recorded while circuit is open");
            }
        }
    }

    /// Record a failed request
    pub fn record_failure(&self) {
        self.total_failures.fetch_add(1, Ordering::Relaxed);
        self.consecutive_successes.store(0, Ordering::Release);
        let failures = self.consecutive_failures.fetch_add(1, Ordering::AcqRel) + 1;

        // Update last failure time atomically
        self.last_failure_time_ms.store(now_ms(), Ordering::Release);

        let current_state = CircuitState::from_int(self.state.load(Ordering::Acquire));

        match current_state {
            CircuitState::Closed => {
                if failures >= self.config.failure_threshold {
                    self.transition_to(CircuitState::Open);
                }
            }
            CircuitState::HalfOpen => {
                self.transition_to(CircuitState::Open);
            }
            CircuitState::Open => {}
        }
    }

    /// Transition to a new state (uses CAS for lock-free operation)
    fn transition_to(&self, new_state: CircuitState) {
        let new_state_int = new_state.to_int();
        let old_state_int = self.state.swap(new_state_int, Ordering::AcqRel);
        let old_state = CircuitState::from_int(old_state_int);

        if old_state != new_state {
            self.last_state_change_ms.store(now_ms(), Ordering::Release);

            match new_state {
                CircuitState::Closed => {
                    self.consecutive_failures.store(0, Ordering::Release);
                    self.consecutive_successes.store(0, Ordering::Release);
                }
                CircuitState::Open => {
                    self.consecutive_successes.store(0, Ordering::Release);
                }
                CircuitState::HalfOpen => {
                    self.consecutive_failures.store(0, Ordering::Release);
                    self.consecutive_successes.store(0, Ordering::Release);
                }
            }

            let from = old_state.as_str();
            let to = new_state.as_str();
            info!("Circuit breaker state transition: {} -> {}", from, to);
            Metrics::record_worker_cb_transition(&self.metric_label, from, to);
            Metrics::set_worker_cb_state(&self.metric_label, new_state.to_int());
            self.publish_gauge_metrics();
        }
    }

    /// Get the number of consecutive failures
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures.load(Ordering::Acquire)
    }

    /// Get the number of consecutive successes
    pub fn consecutive_successes(&self) -> u32 {
        self.consecutive_successes.load(Ordering::Acquire)
    }

    /// Get total failures
    pub fn total_failures(&self) -> u64 {
        self.total_failures.load(Ordering::Relaxed)
    }

    /// Get total successes
    pub fn total_successes(&self) -> u64 {
        self.total_successes.load(Ordering::Relaxed)
    }

    /// Get time since last failure
    pub fn time_since_last_failure(&self) -> Option<Duration> {
        let last_ms = self.last_failure_time_ms.load(Ordering::Acquire);
        if last_ms == 0 {
            None
        } else {
            let elapsed_ms = now_ms().saturating_sub(last_ms);
            Some(Duration::from_millis(elapsed_ms))
        }
    }

    /// Get time since last state change
    pub fn time_since_last_state_change(&self) -> Duration {
        let last_ms = self.last_state_change_ms.load(Ordering::Acquire);
        let elapsed_ms = now_ms().saturating_sub(last_ms);
        Duration::from_millis(elapsed_ms)
    }

    /// Check if the circuit is in a half-open state
    pub fn is_half_open(&self) -> bool {
        self.state() == CircuitState::HalfOpen
    }

    /// Record a test success (for health check probing)
    pub fn record_test_success(&self) {
        if self.is_half_open() {
            self.record_success();
        }
    }

    /// Record a test failure (for health check probing)
    pub fn record_test_failure(&self) {
        if self.is_half_open() {
            self.record_failure();
        }
    }

    /// Reset the circuit breaker to closed state
    pub fn reset(&self) {
        self.transition_to(CircuitState::Closed);
        self.consecutive_failures.store(0, Ordering::Release);
        self.consecutive_successes.store(0, Ordering::Release);
        self.publish_gauge_metrics();
    }

    /// Force the circuit to open (for manual intervention)
    pub fn force_open(&self) {
        self.transition_to(CircuitState::Open);
    }

    /// Get circuit breaker statistics
    pub fn stats(&self) -> CircuitBreakerStats {
        CircuitBreakerStats {
            state: self.state(),
            consecutive_failures: self.consecutive_failures(),
            consecutive_successes: self.consecutive_successes(),
            total_failures: self.total_failures(),
            total_successes: self.total_successes(),
            time_since_last_failure: self.time_since_last_failure(),
            time_since_last_state_change: self.time_since_last_state_change(),
        }
    }

    fn publish_gauge_metrics(&self) {
        Metrics::set_worker_cb_consecutive_failures(
            &self.metric_label,
            self.consecutive_failures(),
        );
        Metrics::set_worker_cb_consecutive_successes(
            &self.metric_label,
            self.consecutive_successes(),
        );
    }
}

impl Clone for CircuitBreaker {
    fn clone(&self) -> Self {
        Self {
            state: AtomicU8::new(self.state.load(Ordering::Acquire)),
            consecutive_failures: AtomicU32::new(self.consecutive_failures.load(Ordering::Acquire)),
            consecutive_successes: AtomicU32::new(
                self.consecutive_successes.load(Ordering::Acquire),
            ),
            total_failures: AtomicU64::new(self.total_failures.load(Ordering::Relaxed)),
            total_successes: AtomicU64::new(self.total_successes.load(Ordering::Relaxed)),
            last_failure_time_ms: AtomicU64::new(self.last_failure_time_ms.load(Ordering::Acquire)),
            last_state_change_ms: AtomicU64::new(self.last_state_change_ms.load(Ordering::Acquire)),
            config: self.config.clone(),
            metric_label: self.metric_label.clone(),
        }
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    pub state: CircuitState,
    pub consecutive_failures: u32,
    pub consecutive_successes: u32,
    pub total_failures: u64,
    pub total_successes: u64,
    pub time_since_last_failure: Option<Duration>,
    pub time_since_last_state_change: Duration,
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    #[test]
    fn test_circuit_breaker_initial_state() {
        let cb = CircuitBreaker::new();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.can_execute());
        assert_eq!(cb.consecutive_failures(), 0);
        assert_eq!(cb.consecutive_successes(), 0);
    }

    #[test]
    fn test_circuit_opens_on_threshold() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config_and_label(config, String::new());

        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();

        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.can_execute());
        assert_eq!(cb.consecutive_failures(), 3);
    }

    #[test]
    fn test_circuit_half_open_after_timeout() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            timeout_duration: Duration::from_millis(100),
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config_and_label(config, String::new());

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        thread::sleep(Duration::from_millis(150));

        assert_eq!(cb.state(), CircuitState::HalfOpen);
        assert!(cb.can_execute());
    }

    #[test]
    fn test_circuit_closes_on_success_threshold() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 2,
            timeout_duration: Duration::from_millis(50),
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config_and_label(config, String::new());

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        thread::sleep(Duration::from_millis(100));
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        cb.record_success();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_success();

        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.can_execute());
    }

    #[test]
    fn test_circuit_reopens_on_half_open_failure() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            timeout_duration: Duration::from_millis(50),
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config_and_label(config, String::new());

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        thread::sleep(Duration::from_millis(100));
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        cb.record_failure();

        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.can_execute());
    }

    #[test]
    fn test_success_resets_failure_count() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config_and_label(config, String::new());

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.consecutive_failures(), 2);

        cb.record_success();
        assert_eq!(cb.consecutive_failures(), 0);
        assert_eq!(cb.consecutive_successes(), 1);

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_manual_reset() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config_and_label(config, String::new());

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.consecutive_failures(), 0);
        assert_eq!(cb.consecutive_successes(), 0);
    }

    #[test]
    fn test_force_open() {
        let cb = CircuitBreaker::new();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.force_open();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.can_execute());
    }

    #[test]
    fn test_stats() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config_and_label(config, String::new());

        cb.record_success();
        cb.record_failure();
        cb.record_failure();

        let stats = cb.stats();
        assert_eq!(stats.state, CircuitState::Open);
        assert_eq!(stats.consecutive_failures, 2);
        assert_eq!(stats.consecutive_successes, 0);
        assert_eq!(stats.total_failures, 2);
        assert_eq!(stats.total_successes, 1);
    }

    #[test]
    fn test_clone() {
        let cb1 = CircuitBreaker::new();
        cb1.record_failure();

        let cb2 = cb1.clone();
        assert_eq!(cb2.consecutive_failures(), 1);

        cb1.record_failure();
        assert_eq!(cb1.consecutive_failures(), 2);
        assert_eq!(cb2.consecutive_failures(), 1); // cb2 is unchanged
    }

    #[test]
    fn test_thread_safety() {
        use std::sync::Arc;

        let cb = Arc::new(CircuitBreaker::new());
        let mut handles = vec![];

        for _ in 0..10 {
            let cb_clone = Arc::clone(&cb);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    cb_clone.record_failure();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(cb.total_failures(), 1000);
    }
}
