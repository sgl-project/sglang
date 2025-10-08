//! Worker management API specifications
//!
//! Defines the request/response structures for worker management endpoints

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

    // gRPC-specific fields (None for HTTP workers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_path: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_parser: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_parser: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,

    /// Additional metadata
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
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
