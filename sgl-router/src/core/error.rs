//! Error types for worker operations

use std::fmt;

/// Errors that can occur during worker operations
#[derive(Debug, Clone)]
pub enum WorkerError {
    /// Worker is not responding to health checks
    HealthCheckFailed { url: String, reason: String },
    /// Network error while communicating with worker
    NetworkError { url: String, error: String },
    /// Worker returned an error status
    WorkerError {
        url: String,
        status: u16,
        body: String,
    },
    /// Timeout waiting for worker response
    Timeout { url: String, timeout_secs: u64 },
    /// Invalid worker configuration
    InvalidConfiguration { reason: String },
    /// Worker is not available
    Unavailable { url: String },
}

impl fmt::Display for WorkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkerError::HealthCheckFailed { url, reason } => {
                write!(f, "Health check failed for worker {url}: {reason}")
            }
            WorkerError::NetworkError { url, error } => {
                write!(f, "Network error for worker {url}: {error}")
            }
            WorkerError::WorkerError { url, status, body } => {
                write!(f, "Worker {url} returned error status {status}: {body}")
            }
            WorkerError::Timeout { url, timeout_secs } => {
                write!(f, "Timeout waiting for worker {url} after {timeout_secs}s")
            }
            WorkerError::InvalidConfiguration { reason } => {
                write!(f, "Invalid worker configuration: {reason}")
            }
            WorkerError::Unavailable { url } => {
                write!(f, "Worker {url} is not available")
            }
        }
    }
}

impl std::error::Error for WorkerError {}

// Conversion from common error types
impl From<reqwest::Error> for WorkerError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            WorkerError::Timeout {
                url: err.url().map(|u| u.to_string()).unwrap_or_default(),
                timeout_secs: 0, // We don't have access to the actual timeout value
            }
        } else {
            WorkerError::NetworkError {
                url: err.url().map(|u| u.to_string()).unwrap_or_default(),
                error: err.to_string(),
            }
        }
    }
}
