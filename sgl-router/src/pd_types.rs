// Essential PDLB types extracted for PD routing

use crate::core::{Worker, WorkerType};
use serde::{Deserialize, Serialize};
use serde_json::Value;

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

pub fn get_hostname(url: &str) -> String {
    // Simple hostname extraction without external dependencies
    let url = url
        .trim_start_matches("http://")
        .trim_start_matches("https://");
    url.split(':').next().unwrap_or("localhost").to_string()
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
}
// Bootstrap types from PDLB
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SingleOrBatch<T> {
    Single(T),
    Batch(Vec<T>),
}

pub type InputIds = SingleOrBatch<Vec<i32>>;
pub type InputText = SingleOrBatch<String>;
pub type BootstrapHost = SingleOrBatch<String>;
pub type BootstrapPort = SingleOrBatch<Option<u16>>;
pub type BootstrapRoom = SingleOrBatch<u64>;

// Bootstrap trait for request handling
pub trait Bootstrap: Send + Sync {
    fn is_stream(&self) -> bool;
    fn get_batch_size(&self) -> Result<Option<usize>, String>;
    fn set_bootstrap_info(
        &mut self,
        bootstrap_host: BootstrapHost,
        bootstrap_port: BootstrapPort,
        bootstrap_room: BootstrapRoom,
    );

    fn add_bootstrap_info(&mut self, prefill_worker: &dyn Worker) -> Result<(), String> {
        let batch_size = self.get_batch_size()?;

        // Extract bootstrap port from prefill worker if it's a prefill type
        let bootstrap_port = match prefill_worker.worker_type() {
            WorkerType::Prefill { bootstrap_port } => bootstrap_port,
            _ => None,
        };

        let hostname = get_hostname(prefill_worker.url());

        if let Some(batch_size) = batch_size {
            self.set_bootstrap_info(
                BootstrapHost::Batch(vec![hostname; batch_size]),
                BootstrapPort::Batch(vec![bootstrap_port; batch_size]),
                // Use high-quality random numbers to minimize collision risk
                BootstrapRoom::Batch(
                    (0..batch_size)
                        .map(|_| {
                            // Combine multiple sources of randomness for better distribution
                            let r1 = rand::random::<u64>();
                            let r2 = rand::random::<u64>();
                            r1.wrapping_add(r2.rotate_left(32))
                        })
                        .collect(),
                ),
            );
        } else {
            self.set_bootstrap_info(
                BootstrapHost::Single(hostname),
                BootstrapPort::Single(bootstrap_port),
                BootstrapRoom::Single({
                    // Use high-quality random number for single requests too
                    let r1 = rand::random::<u64>();
                    let r2 = rand::random::<u64>();
                    r1.wrapping_add(r2.rotate_left(32))
                }),
            );
        }
        Ok(())
    }
}

// Request types
#[derive(Debug, Deserialize, Serialize)]
pub struct GenerateReqInput {
    pub text: Option<InputText>,
    pub input_ids: Option<InputIds>,
    #[serde(default)]
    pub stream: bool,
    pub bootstrap_host: Option<BootstrapHost>,
    pub bootstrap_port: Option<BootstrapPort>,
    pub bootstrap_room: Option<BootstrapRoom>,

    #[serde(flatten)]
    pub other: Value,
}

impl GenerateReqInput {
    pub fn get_batch_size(&self) -> Result<Option<usize>, String> {
        if self.text.is_some() && self.input_ids.is_some() {
            return Err("Both text and input_ids are present in the request".to_string());
        }

        // Check text batch
        if let Some(InputText::Batch(texts)) = &self.text {
            if texts.is_empty() {
                return Err("Batch text array is empty".to_string());
            }
            if texts.len() > 10000 {
                // Reasonable limit for production
                return Err(format!(
                    "Batch size {} exceeds maximum allowed (10000)",
                    texts.len()
                ));
            }
            return Ok(Some(texts.len()));
        }

        // Check input_ids batch
        if let Some(InputIds::Batch(ids)) = &self.input_ids {
            if ids.is_empty() {
                return Err("Batch input_ids array is empty".to_string());
            }
            if ids.len() > 10000 {
                // Reasonable limit for production
                return Err(format!(
                    "Batch size {} exceeds maximum allowed (10000)",
                    ids.len()
                ));
            }
            // Validate each sequence is not empty
            for (i, seq) in ids.iter().enumerate() {
                if seq.is_empty() {
                    return Err(format!("Input sequence at index {} is empty", i));
                }
            }
            return Ok(Some(ids.len()));
        }

        Ok(None)
    }
}

impl Bootstrap for GenerateReqInput {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_batch_size(&self) -> Result<Option<usize>, String> {
        self.get_batch_size()
    }

    fn set_bootstrap_info(
        &mut self,
        bootstrap_host: BootstrapHost,
        bootstrap_port: BootstrapPort,
        bootstrap_room: BootstrapRoom,
    ) {
        self.bootstrap_host = Some(bootstrap_host);
        self.bootstrap_port = Some(bootstrap_port);
        self.bootstrap_room = Some(bootstrap_room);
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatReqInput {
    #[serde(default)]
    pub stream: bool,
    pub bootstrap_host: Option<BootstrapHost>,
    pub bootstrap_port: Option<BootstrapPort>,
    pub bootstrap_room: Option<BootstrapRoom>,

    #[serde(flatten)]
    pub other: Value,
}

impl Bootstrap for ChatReqInput {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_batch_size(&self) -> Result<Option<usize>, String> {
        // Check if 'n' parameter is present and > 1
        if let Some(n_value) = self.other.get("n") {
            if let Some(n) = n_value.as_u64() {
                if n > 1 {
                    return Ok(Some(n as usize));
                }
            }
        }
        Ok(None)
    }

    fn set_bootstrap_info(
        &mut self,
        bootstrap_host: BootstrapHost,
        bootstrap_port: BootstrapPort,
        bootstrap_room: BootstrapRoom,
    ) {
        self.bootstrap_host = Some(bootstrap_host);
        self.bootstrap_port = Some(bootstrap_port);
        self.bootstrap_room = Some(bootstrap_room);
    }
}
