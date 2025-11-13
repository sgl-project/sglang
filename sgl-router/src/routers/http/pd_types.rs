// Custom error type for PD router operations
#[derive(Debug, thiserror::Error)]
pub enum PDRouterError {
    #[error("Worker already exists: {url}")]
    WorkerAlreadyExists { url: String },

    #[error("Worker not found: {url}")]
    WorkerNotFound { url: String },

    #[error("Lock acquisition failed: {operation}")]
    LockError { operation: String },

    #[error("Health check failed for worker: {url}")]
    HealthCheckFailed { url: String },

    #[error("Invalid worker configuration: {reason}")]
    InvalidConfiguration { reason: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Timeout waiting for worker: {url}")]
    Timeout { url: String },
}

// Helper functions for workers
pub fn api_path(url: &str, api_path: &str) -> String {
    if api_path.starts_with("/") {
        format!("{}{}", url, api_path)
    } else {
        format!("{}/{}", url, api_path)
    }
}

use serde::Serialize;

// Optimized bootstrap wrapper for single requests
#[derive(Serialize)]
pub struct RequestWithBootstrap<'a, T: Serialize> {
    #[serde(flatten)]
    pub original: &'a T,
    pub bootstrap_host: String,
    pub bootstrap_port: Option<u16>,
    pub bootstrap_room: u64,
}

// Optimized bootstrap wrapper for batch requests
#[derive(Serialize)]
pub struct BatchRequestWithBootstrap<'a, T: Serialize> {
    #[serde(flatten)]
    pub original: &'a T,
    pub bootstrap_host: Vec<String>,
    pub bootstrap_port: Vec<Option<u16>>,
    pub bootstrap_room: Vec<u64>,
}

// Helper to generate bootstrap room ID
pub fn generate_room_id() -> u64 {
    // Generate a value in the range [0, 2^63 - 1] to match Python's random.randint(0, 2**63 - 1)
    rand::random::<u64>() & (i64::MAX as u64)
}

// PD-specific routing policies
#[derive(Debug, Clone, PartialEq)]
pub enum PDSelectionPolicy {
    Random,
    PowerOfTwo,
    CacheAware {
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
    },
    Bucket {
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        bucket_adjust_interval_secs: usize,
    },
}
