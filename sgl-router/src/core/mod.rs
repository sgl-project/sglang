//! Core abstractions for the SGLang router
//!
//! This module contains the fundamental types and traits used throughout the router:
//! - Worker trait and implementations
//! - Error types
//! - Circuit breaker for reliability
//! - Token buckets for rate limiting
//! - Workflow engine for multi-step operations
//! - Common utilities

pub mod circuit_breaker;
pub mod error;
pub mod job_queue;
pub mod metrics_aggregator;
pub mod retry;
pub mod token_bucket;
pub mod worker;
pub mod worker_builder;
pub mod worker_manager;
pub mod worker_registry;
pub mod workflow;

pub use circuit_breaker::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerStats, CircuitState,
};
pub use error::{WorkerError, WorkerResult};
pub use job_queue::{Job, JobQueue, JobQueueConfig};
pub use retry::{is_retryable_status, BackoffCalculator, RetryError, RetryExecutor};
pub use worker::{
    worker_to_info, BasicWorker, ConnectionMode, DPAwareWorker, HealthChecker, HealthConfig,
    Worker, WorkerFactory, WorkerLoadGuard, WorkerType,
};
pub use worker_builder::{BasicWorkerBuilder, DPAwareWorkerBuilder};
pub use worker_manager::{LoadMonitor, WorkerManager};
pub use worker_registry::{WorkerId, WorkerRegistry, WorkerRegistryStats};
