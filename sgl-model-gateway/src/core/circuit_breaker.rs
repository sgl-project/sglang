use std::{
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc, RwLock,
    },
    time::{Duration, Instant},
};

use tracing::info;

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

/// Circuit breaker implementation
#[derive(Debug)]
pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    consecutive_failures: Arc<AtomicU32>,
    consecutive_successes: Arc<AtomicU32>,
    total_failures: Arc<AtomicU64>,
    total_successes: Arc<AtomicU64>,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
    last_state_change: Arc<RwLock<Instant>>,
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with default configuration
    pub fn new() -> Self {
        Self::with_config(CircuitBreakerConfig::default())
    }

    /// Create a new circuit breaker with custom configuration
    pub fn with_config(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            consecutive_failures: Arc::new(AtomicU32::new(0)),
            consecutive_successes: Arc::new(AtomicU32::new(0)),
            total_failures: Arc::new(AtomicU64::new(0)),
            total_successes: Arc::new(AtomicU64::new(0)),
            last_failure_time: Arc::new(RwLock::new(None)),
            last_state_change: Arc::new(RwLock::new(Instant::now())),
            config,
        }
    }

    /// Check if a request can be executed
    pub fn can_execute(&self) -> bool {
        let state = self.state();
        match state {
            CircuitState::Closed => true,
            CircuitState::Open => false,
            CircuitState::HalfOpen => true,
        }
    }

    /// Get the current state
    pub fn state(&self) -> CircuitState {
        self.check_and_update_state_returning()
    }

    /// Check and update state, returning the current state to avoid double lock
    fn check_and_update_state_returning(&self) -> CircuitState {
        let current_state = *self.state.read().unwrap();

        if current_state == CircuitState::Open {
            let last_change = *self.last_state_change.read().unwrap();
            if last_change.elapsed() >= self.config.timeout_duration {
                self.transition_to(CircuitState::HalfOpen);
                return CircuitState::HalfOpen;
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
    }

    /// Record a successful request
    pub fn record_success(&self) {
        self.total_successes.fetch_add(1, Ordering::Relaxed);
        self.consecutive_failures.store(0, Ordering::Release);
        let successes = self.consecutive_successes.fetch_add(1, Ordering::AcqRel) + 1;

        let current_state = *self.state.read().unwrap();

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

        {
            let mut last_failure = self.last_failure_time.write().unwrap();
            *last_failure = Some(Instant::now());
        }

        let current_state = *self.state.read().unwrap();

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

    /// Transition to a new state
    fn transition_to(&self, new_state: CircuitState) {
        let mut state = self.state.write().unwrap();
        let old_state = *state;

        if old_state != new_state {
            *state = new_state;

            let mut last_change = self.last_state_change.write().unwrap();
            *last_change = Instant::now();

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

            let from = match old_state {
                CircuitState::Closed => "closed",
                CircuitState::Open => "open",
                CircuitState::HalfOpen => "half_open",
            };
            let to = match new_state {
                CircuitState::Closed => "closed",
                CircuitState::Open => "open",
                CircuitState::HalfOpen => "half_open",
            };
            info!("Circuit breaker state transition: {} -> {}", from, to);
        }
    }

    /// Get the number of consecutive failures
    pub fn failure_count(&self) -> u32 {
        self.consecutive_failures.load(Ordering::Acquire)
    }

    /// Get the number of consecutive successes
    pub fn success_count(&self) -> u32 {
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
        self.last_failure_time.read().unwrap().map(|t| t.elapsed())
    }

    /// Get time since last state change
    pub fn time_since_last_state_change(&self) -> Duration {
        self.last_state_change.read().unwrap().elapsed()
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
    }

    /// Force the circuit to open (for manual intervention)
    pub fn force_open(&self) {
        self.transition_to(CircuitState::Open);
    }

    /// Get circuit breaker statistics
    pub fn stats(&self) -> CircuitBreakerStats {
        CircuitBreakerStats {
            state: self.state(),
            consecutive_failures: self.failure_count(),
            consecutive_successes: self.success_count(),
            total_failures: self.total_failures(),
            total_successes: self.total_successes(),
            time_since_last_failure: self.time_since_last_failure(),
            time_since_last_state_change: self.time_since_last_state_change(),
        }
    }
}

impl Clone for CircuitBreaker {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
            consecutive_failures: Arc::clone(&self.consecutive_failures),
            consecutive_successes: Arc::clone(&self.consecutive_successes),
            total_failures: Arc::clone(&self.total_failures),
            total_successes: Arc::clone(&self.total_successes),
            last_failure_time: Arc::clone(&self.last_failure_time),
            last_state_change: Arc::clone(&self.last_state_change),
            config: self.config.clone(),
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
        assert_eq!(cb.failure_count(), 0);
        assert_eq!(cb.success_count(), 0);
    }

    #[test]
    fn test_circuit_opens_on_threshold() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config(config);

        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();

        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.can_execute());
        assert_eq!(cb.failure_count(), 3);
    }

    #[test]
    fn test_circuit_half_open_after_timeout() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            timeout_duration: Duration::from_millis(100),
            ..Default::default()
        };
        let cb = CircuitBreaker::with_config(config);

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
        let cb = CircuitBreaker::with_config(config);

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
        let cb = CircuitBreaker::with_config(config);

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
        let cb = CircuitBreaker::with_config(config);

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.failure_count(), 2);

        cb.record_success();
        assert_eq!(cb.failure_count(), 0);
        assert_eq!(cb.success_count(), 1);

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
        let cb = CircuitBreaker::with_config(config);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_count(), 0);
        assert_eq!(cb.success_count(), 0);
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
        let cb = CircuitBreaker::with_config(config);

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
        assert_eq!(cb2.failure_count(), 1);

        cb1.record_failure();
        assert_eq!(cb2.failure_count(), 2);
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
