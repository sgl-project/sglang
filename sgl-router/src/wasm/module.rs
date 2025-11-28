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

/// Deserialize hex string to [u8; 32]
fn deserialize_sha256_hash<'de, D>(deserializer: D) -> Result<[u8; 32], D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let hex_string = String::deserialize(deserializer)?;

    // Parse hex string to bytes
    if hex_string.len() != 64 {
        return Err(serde::de::Error::custom(format!(
            "Invalid SHA256 hash length: expected 64 hex characters, got {}",
            hex_string.len()
        )));
    }

    let mut hash = [0u8; 32];
    for (i, chunk) in hex_string.as_bytes().chunks(2).enumerate() {
        if chunk.len() != 2 {
            return Err(serde::de::Error::custom("Invalid hex string format"));
        }
        let byte_str = std::str::from_utf8(chunk)
            .map_err(|e| serde::de::Error::custom(format!("Invalid UTF-8: {}", e)))?;
        hash[i] = u8::from_str_radix(byte_str, 16)
            .map_err(|e| serde::de::Error::custom(format!("Invalid hex digit: {}", e)))?;
    }

    Ok(hash)
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

/// Deserialize ISO 8601 string to u64 timestamp (nanoseconds since epoch)
fn deserialize_timestamp<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use chrono::{DateTime, Utc};
    use serde::Deserialize;

    let timestamp_str = String::deserialize(deserializer)?;

    // Try to parse as ISO 8601 datetime (RFC 3339)
    match DateTime::parse_from_rfc3339(&timestamp_str) {
        Ok(dt) => {
            // Convert to UTC and then to nanoseconds since epoch
            let dt_utc = dt.with_timezone(&Utc);
            let secs = dt_utc.timestamp();
            let nanos = dt_utc.timestamp_subsec_nanos();
            Ok((secs as u64) * 1_000_000_000 + (nanos as u64))
        }
        Err(_) => {
            // Fallback: try to parse as u64 directly
            timestamp_str
                .parse::<u64>()
                .map_err(|e| serde::de::Error::custom(format!("Invalid timestamp format: {}", e)))
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
    #[serde(
        serialize_with = "serialize_sha256_hash",
        deserialize_with = "deserialize_sha256_hash"
    )]
    pub sha256_hash: [u8; 32],
    // size of the module file in bytes
    pub size_bytes: u64,
    // timestamp of when the module was created (nanoseconds since epoch)
    #[serde(
        serialize_with = "serialize_timestamp",
        deserialize_with = "deserialize_timestamp"
    )]
    pub created_at: u64,
    // timestamp of when the module was last accessed (nanoseconds since epoch)
    #[serde(
        serialize_with = "serialize_timestamp",
        deserialize_with = "deserialize_timestamp"
    )]
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
