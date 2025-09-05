//! Core abstractions for the SGLang router
//!
//! This module contains the fundamental types and traits used throughout the router:
//! - Worker trait and implementations
//! - Error types
//! - Circuit breaker for reliability
//! - Common utilities

pub mod circuit_breaker;
pub mod error;
pub mod retry;
pub mod token_bucket;
pub mod worker;

// Re-export commonly used types at the module level
pub use circuit_breaker::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerStats, CircuitState,
};
pub use error::{WorkerError, WorkerResult};
pub use retry::{is_retryable_status, BackoffCalculator, RetryError, RetryExecutor};
pub use worker::{
    start_health_checker, BasicWorker, ConnectionMode, DPAwareWorker, HealthChecker, HealthConfig,
    Worker, WorkerCollection, WorkerFactory, WorkerLoadGuard, WorkerType,
};
