//! WASM Module

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmModule {
    // unique identifier for the module
    pub module_uuid: Uuid,
    // metadata for the module
    pub module_meta: WasmModuleMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmModuleDescriptor {
    pub name: String,
    pub file_path: String,
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
}
