//! Core abstractions for the SGLang router
//!
//! This module contains the fundamental types and traits used throughout the router:
//! - Worker trait and implementations
//! - Model types and endpoint definitions
//! - Error types
//! - Circuit breaker for reliability
//! - Token buckets for rate limiting
//! - Workflow steps for multi-step operations
//! - Common utilities

// Re-export UNKNOWN_MODEL_ID from protocols for use throughout core
pub use crate::protocols::UNKNOWN_MODEL_ID;

pub mod circuit_breaker;
pub mod error;
pub mod job_queue;
pub mod metrics_aggregator;
pub mod model_card;
pub mod model_type;
pub mod retry;
pub mod steps;
pub mod token_bucket;
pub mod worker;
pub mod worker_builder;
pub mod worker_manager;
pub mod worker_registry;
pub mod worker_service;

pub use circuit_breaker::{
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerStats, CircuitState,
};
pub use error::{WorkerError, WorkerResult};
pub use job_queue::{Job, JobQueue, JobQueueConfig};
pub use model_card::{ModelCard, ProviderType};
pub use model_type::{Endpoint, ModelType};
pub use retry::{is_retryable_status, BackoffCalculator, RetryError, RetryExecutor};
pub use worker::{
    attach_guards_to_response, worker_to_info, BasicWorker, ConnectionMode, DPAwareWorker,
    HealthChecker, HealthConfig, RuntimeType, Worker, WorkerFactory, WorkerLoadGuard, WorkerType,
};
pub use worker_builder::{BasicWorkerBuilder, DPAwareWorkerBuilder};
pub use worker_manager::{LoadMonitor, WorkerManager};
pub use worker_registry::{HashRing, WorkerId, WorkerRegistry, WorkerRegistryStats};
pub use worker_service::{WorkerService, WorkerServiceError};
