//! WASM Module Data Structures and Types
//!
//! This module defines the core data structures for managing WebAssembly components:
//! - Module metadata (UUID, name, file path, hash, timestamps, metrics)
//! - Module types and attachment points (Middleware hooks: OnRequest, OnResponse, OnError)
//! - API request/response types for module management
//! - Execution metrics and statistics
//!
//! The module provides custom serialization for:
//! - SHA256 hashes (hex string representation)
//! - Timestamps (ISO 8601 format for JSON output)

use serde::{Deserialize, Serialize, Serializer};
use uuid::Uuid;

/// Serialize [u8; 32] as hex string
fn serialize_sha256_hash<S>(hash: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let hex_string = hash
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>();
    serializer.serialize_str(&hex_string)
}

/// Serialize u64 timestamp (nanoseconds since epoch) as ISO 8601 string
fn serialize_timestamp<S>(timestamp: &u64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    use chrono::{DateTime, Utc};

    // Convert nanoseconds to seconds and remaining nanoseconds
    let secs = (*timestamp / 1_000_000_000) as i64;
    let nanos = (*timestamp % 1_000_000_000) as u32;

    match DateTime::<Utc>::from_timestamp(secs, nanos) {
        Some(dt) => {
            let s = dt.to_rfc3339_opts(chrono::SecondsFormat::Nanos, true);
            serializer.serialize_str(&s)
        }
        None => {
            // Fallback: format manually if timestamp is out of range
            let s = format!("{}", timestamp);
            serializer.serialize_str(&s)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmModule {
    // unique identifier for the module
    pub module_uuid: Uuid,
    pub module_meta: WasmModuleMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WasmModuleAddResult {
    Success(Uuid),
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmModuleDescriptor {
    pub name: String,
    pub file_path: String,
    pub module_type: WasmModuleType,
    pub attach_points: Vec<WasmModuleAttachPoint>,
    pub add_result: Option<WasmModuleAddResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmModuleMeta {
    // module name provided by the user
    pub name: String,
    // path to the module file
    pub file_path: String,
    // sha256 hash of the module file
    #[serde(serialize_with = "serialize_sha256_hash")]
    pub sha256_hash: [u8; 32],
    // size of the module file in bytes
    pub size_bytes: u64,
    // timestamp of when the module was created (nanoseconds since epoch)
    #[serde(serialize_with = "serialize_timestamp")]
    pub created_at: u64,
    // timestamp of when the module was last accessed (nanoseconds since epoch)
    #[serde(serialize_with = "serialize_timestamp")]
    pub last_accessed_at: u64,
    // number of times the module was accessed
    pub access_count: u64,
    // attach points for the module
    pub attach_points: Vec<WasmModuleAttachPoint>,
    // Pre-loaded WASM component bytes (loaded into memory for faster execution)
    #[serde(skip)]
    pub wasm_bytes: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub enum WasmModuleType {
    Middleware,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub enum MiddlewareAttachPoint {
    OnRequest,
    OnResponse,
    OnError,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub enum WasmModuleAttachPoint {
    Middleware(MiddlewareAttachPoint),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmModuleAddRequest {
    pub modules: Vec<WasmModuleDescriptor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmModuleAddResponse {
    pub modules: Vec<WasmModuleDescriptor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmModuleListResponse {
    pub modules: Vec<WasmModule>,
    pub metrics: WasmMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmMetrics {
    pub total_executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
    pub total_execution_time_ms: u64,
    pub max_execution_time_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub average_execution_time_ms: Option<f64>,
}
