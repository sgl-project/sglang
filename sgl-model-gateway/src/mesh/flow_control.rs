//! Flow control for mesh cluster communication
//!
//! Provides:
//! - Backpressure control (channel capacity monitoring)
//! - Message size limits and validation
//! - Exponential backoff for reconnection

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::RwLock;

/// Maximum message size in bytes (default: 10MB)
pub const MAX_MESSAGE_SIZE: usize = 10 * 1024 * 1024;

/// Channel capacity threshold for backpressure (default: 20% remaining)
pub const BACKPRESSURE_THRESHOLD: usize = 25; // 25 out of 128 = ~20%

/// Backpressure controller for managing channel capacity
#[derive(Debug, Clone)]
pub struct BackpressureController {
    channel_capacity: usize,
    threshold: usize,
}

impl BackpressureController {
    pub fn new(channel_capacity: usize, threshold: usize) -> Self {
        Self {
            channel_capacity,
            threshold,
        }
    }

    /// Check if channel has capacity for sending
    pub fn can_send(&self, current_len: usize) -> bool {
        let remaining = self.channel_capacity.saturating_sub(current_len);
        remaining > self.threshold
    }

    /// Get remaining capacity
    pub fn remaining_capacity(&self, current_len: usize) -> usize {
        self.channel_capacity.saturating_sub(current_len)
    }
}

impl Default for BackpressureController {
    fn default() -> Self {
        Self::new(128, BACKPRESSURE_THRESHOLD)
    }
}

/// Message size validator
#[derive(Debug, Clone)]
pub struct MessageSizeValidator {
    max_size: usize,
}

impl MessageSizeValidator {
    pub fn new(max_size: usize) -> Self {
        Self { max_size }
    }

    /// Validate message size
    pub fn validate(&self, size: usize) -> Result<(), MessageSizeError> {
        if size > self.max_size {
            Err(MessageSizeError::TooLarge {
                size,
                max: self.max_size,
            })
        } else {
            Ok(())
        }
    }

    /// Get maximum allowed size
    pub fn max_size(&self) -> usize {
        self.max_size
    }
}

impl Default for MessageSizeValidator {
    fn default() -> Self {
        Self::new(MAX_MESSAGE_SIZE)
    }
}

/// Message size validation error
#[derive(Debug, Clone)]
pub enum MessageSizeError {
    TooLarge { size: usize, max: usize },
}

impl std::fmt::Display for MessageSizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageSizeError::TooLarge { size, max } => {
                write!(f, "Message size {} exceeds maximum {}", size, max)
            }
        }
    }
}

impl std::error::Error for MessageSizeError {}

/// Exponential backoff calculator for reconnection
#[derive(Debug, Clone)]
pub struct ExponentialBackoff {
    initial_delay: Duration,
    max_delay: Duration,
    multiplier: f64,
}

impl ExponentialBackoff {
    pub fn new(initial_delay: Duration, max_delay: Duration, multiplier: f64) -> Self {
        Self {
            initial_delay,
            max_delay,
            multiplier,
        }
    }

    /// Calculate delay for attempt number (0-indexed)
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let delay_secs = self.initial_delay.as_secs_f64() * self.multiplier.powi(attempt as i32);
        let delay = Duration::from_secs_f64(delay_secs);
        delay.min(self.max_delay)
    }
}

impl Default for ExponentialBackoff {
    fn default() -> Self {
        Self::new(Duration::from_secs(1), Duration::from_secs(60), 2.0)
    }
}

/// Connection retry manager with exponential backoff
#[derive(Debug)]
pub struct RetryManager {
    backoff: ExponentialBackoff,
    last_attempt: Arc<RwLock<Option<Instant>>>,
    attempt_count: Arc<RwLock<u32>>,
}

impl RetryManager {
    pub fn new(backoff: ExponentialBackoff) -> Self {
        Self {
            backoff,
            last_attempt: Arc::new(RwLock::new(None)),
            attempt_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Check if we should retry now (based on backoff delay)
    pub fn should_retry(&self) -> bool {
        let last = self.last_attempt.read();
        if let Some(last_attempt) = *last {
            let attempt = *self.attempt_count.read();
            let delay = self.backoff.delay_for_attempt(attempt);
            last_attempt.elapsed() >= delay
        } else {
            true // First attempt
        }
    }

    /// Record a retry attempt
    pub fn record_attempt(&self) {
        *self.last_attempt.write() = Some(Instant::now());
        *self.attempt_count.write() += 1;
    }

    /// Reset retry state (on successful connection)
    pub fn reset(&self) {
        *self.last_attempt.write() = None;
        *self.attempt_count.write() = 0;
    }

    /// Get current attempt count
    pub fn attempt_count(&self) -> u32 {
        *self.attempt_count.read()
    }

    /// Get next retry delay
    pub fn next_delay(&self) -> Duration {
        let attempt = *self.attempt_count.read();
        self.backoff.delay_for_attempt(attempt)
    }
}

impl Default for RetryManager {
    fn default() -> Self {
        Self::new(ExponentialBackoff::default())
    }
}
