//! Error types for the SGLang router core
//!
//! This module defines error types used throughout the router for worker operations.

use std::fmt;

/// Worker-related errors
#[derive(Debug)]
pub enum WorkerError {
    /// Health check failed
    HealthCheckFailed { url: String, reason: String },
    /// Worker not found
    WorkerNotFound { url: String },
    /// Invalid worker configuration
    InvalidConfiguration { message: String },
    /// Network error
    NetworkError { url: String, error: String },
    /// Worker is at capacity
    WorkerAtCapacity { url: String },
}

impl fmt::Display for WorkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkerError::HealthCheckFailed { url, reason } => {
                write!(f, "Health check failed for worker {}: {}", url, reason)
            }
            WorkerError::WorkerNotFound { url } => {
                write!(f, "Worker not found: {}", url)
            }
            WorkerError::InvalidConfiguration { message } => {
                write!(f, "Invalid worker configuration: {}", message)
            }
            WorkerError::NetworkError { url, error } => {
                write!(f, "Network error for worker {}: {}", url, error)
            }
            WorkerError::WorkerAtCapacity { url } => {
                write!(f, "Worker at capacity: {}", url)
            }
        }
    }
}

impl std::error::Error for WorkerError {}

/// Result type for worker operations
pub type WorkerResult<T> = Result<T, WorkerError>;

/// Convert from reqwest errors to worker errors
impl From<reqwest::Error> for WorkerError {
    fn from(err: reqwest::Error) -> Self {
        WorkerError::NetworkError {
            url: err.url().map(|u| u.to_string()).unwrap_or_default(),
            error: err.to_string(),
        }
    }
}
