//! Core abstractions for the SGLang router
//!
//! This module contains the fundamental types and traits used throughout the router:
//! - Worker trait and implementations
//! - Error types
//! - Common utilities

pub mod error;
pub mod worker;

// Re-export commonly used types at the module level
pub use error::{WorkerError, WorkerResult};
pub use worker::{BasicWorker, Worker, WorkerCollection, WorkerFactory, WorkerType};
