//! Worker management API specifications
//!
//! Defines the request/response structures for worker management endpoints

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Worker configuration for API requests
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkerConfigRequest {
    /// Worker URL (required)
    pub url: String,

    /// Worker API key (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// Model ID (optional, will query from server if not provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,

    /// Worker priority (optional, default: 50, higher = preferred)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<u32>,

    /// Worker cost factor (optional, default: 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f32>,

    /// Worker type (optional: "regular", "prefill", "decode")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_type: Option<String>,

    /// Bootstrap port for prefill workers (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_port: Option<u16>,

    /// Runtime type (optional: "sglang", "vllm", default: "sglang")
    /// Only relevant for gRPC workers
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime: Option<String>,

    // gRPC-specific configuration (optional, ignored in HTTP mode)
    /// Tokenizer path for gRPC mode
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_path: Option<String>,

    /// Reasoning parser type for gRPC mode
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_parser: Option<String>,

    /// Tool parser type for gRPC mode
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_parser: Option<String>,

    /// Chat template for gRPC mode
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,

    /// Additional labels (optional)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub labels: HashMap<String, String>,

    /// Health check timeout in seconds (default: 30)
    #[serde(default = "default_health_check_timeout")]
    pub health_check_timeout_secs: u64,

    /// Health check interval in seconds (default: 60)
    #[serde(default = "default_health_check_interval")]
    pub health_check_interval_secs: u64,

    /// Number of successful health checks needed to mark worker as healthy (default: 2)
    #[serde(default = "default_health_success_threshold")]
    pub health_success_threshold: u32,

    /// Number of failed health checks before marking worker as unhealthy (default: 3)
    #[serde(default = "default_health_failure_threshold")]
    pub health_failure_threshold: u32,

    /// Maximum connection attempts during worker registration (default: 20)
    #[serde(default = "default_max_connection_attempts")]
    pub max_connection_attempts: u32,

    /// Enable data parallelism aware scheduling (default: false)
    #[serde(default)]
    pub dp_aware: bool,
}

// Default value functions for serde
fn default_health_check_timeout() -> u64 {
    30
}

fn default_health_check_interval() -> u64 {
    60
}

fn default_health_success_threshold() -> u32 {
    2
}

fn default_health_failure_threshold() -> u32 {
    3
}

fn default_max_connection_attempts() -> u32 {
    20
}

/// Worker information for API responses
#[derive(Debug, Clone, Serialize)]
pub struct WorkerInfo {
    /// Worker unique identifier
    pub id: String,

    /// Worker URL
    pub url: String,

    /// Model ID this worker serves
    pub model_id: String,

    /// Worker priority
    pub priority: u32,

    /// Worker cost factor
    pub cost: f32,

    /// Worker type
    pub worker_type: String,

    /// Whether the worker is healthy
    pub is_healthy: bool,

    /// Current load on the worker
    pub load: usize,

    /// Connection mode (http or grpc)
    pub connection_mode: String,

    /// Runtime type (sglang or vllm, for gRPC workers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_type: Option<String>,

    // gRPC-specific fields (None for HTTP workers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_path: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_parser: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_parser: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,

    /// Bootstrap port for prefill workers
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_port: Option<u16>,

    /// Additional metadata
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,

    /// Job status for async operations (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_status: Option<JobStatus>,
}

/// Job status for async control plane operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStatus {
    pub job_type: String,
    pub worker_url: String,
    pub status: String,
    pub message: Option<String>,
    pub timestamp: u64,
}

/// Worker list response
#[derive(Debug, Clone, Serialize)]
pub struct WorkerListResponse {
    /// List of workers
    pub workers: Vec<WorkerInfo>,

    /// Total count
    pub total: usize,

    /// Statistics
    pub stats: WorkerStats,
}

/// Worker statistics
#[derive(Debug, Clone, Serialize)]
pub struct WorkerStats {
    pub total_workers: usize,
    pub healthy_workers: usize,
    pub total_models: usize,
    pub total_load: usize,
    pub by_type: WorkerTypeStats,
}

/// Worker statistics by type
#[derive(Debug, Clone, Serialize)]
pub struct WorkerTypeStats {
    pub regular: usize,
    pub prefill: usize,
    pub decode: usize,
}

/// Worker update request
#[derive(Debug, Clone, Deserialize)]
pub struct WorkerUpdateRequest {
    /// Update priority
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<u32>,

    /// Update cost
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f32>,

    /// Update labels
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<HashMap<String, String>>,
}

/// Generic API response
#[derive(Debug, Clone, Serialize)]
pub struct WorkerApiResponse {
    pub success: bool,
    pub message: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker: Option<WorkerInfo>,
}

/// Error response
#[derive(Debug, Clone, Serialize)]
pub struct WorkerErrorResponse {
    pub error: String,
    pub code: String,
}

/// Server info response from /get_server_info endpoint
#[derive(Debug, Clone, Deserialize)]
pub struct ServerInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_type: Option<String>,

    // gRPC-specific
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_path: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_parser: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_parser: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
}

/// Result from flush cache operations across workers
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FlushCacheResult {
    /// URLs of workers where cache flush succeeded
    pub successful: Vec<String>,
    /// URLs and error messages for workers where cache flush failed
    pub failed: Vec<(String, String)>,
    /// Total number of workers attempted
    pub total_workers: usize,
    /// Number of HTTP workers (gRPC workers don't support flush cache)
    pub http_workers: usize,
    /// Human-readable summary message
    pub message: String,
}

/// Result from getting worker loads
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkerLoadsResult {
    /// Worker URL and load pairs
    pub loads: Vec<WorkerLoadInfo>,
    /// Total number of workers
    pub total_workers: usize,
    /// Number of workers with successful load fetches
    pub successful: usize,
    /// Number of workers with failed load fetches
    pub failed: usize,
}

/// Individual worker load information
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkerLoadInfo {
    /// Worker URL
    pub worker: String,
    /// Worker type (regular, prefill, decode)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_type: Option<String>,
    /// Current load (-1 indicates failure to fetch)
    pub load: isize,
}
