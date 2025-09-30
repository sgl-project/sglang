//! WASM Module

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmModule {
    // unique identifier for the module
    pub module_uuid: Uuid,
    pub module_meta: WasmModuleMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmModuleDescriptor {
    pub name: String,
    pub file_path: String,
    pub module_type: WasmModuleType,
    pub attach_points: HashMap<WasmModuleAttachPoint, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmModuleMeta {
    // module name provided by the user
    pub name: String,
    // path to the module file
    pub file_path: String,
    // sha256 hash of the module file
    pub sha256_hash: [u8; 32],
    // size of the module file in bytes
    pub size_bytes: u64,
    // timestamp of when the module was created (nanoseconds since epoch)
    pub created_at: u64,
    // timestamp of when the module was last accessed (nanoseconds since epoch)
    pub last_accessed_at: u64,
    // number of times the module was accessed
    pub access_count: u64,
    // attach points for the module (module type / attach poi -> Wasmfunction names)
    pub attach_points: HashMap<WasmModuleAttachPoint, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub enum WasmModuleType {
    Middleware
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