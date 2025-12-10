//! WASM Error Types
//!
//! Defines comprehensive error types for the WASM subsystem,
//! including module, manager, and runtime errors.

use std::fmt;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, WasmError>;

/// SHA256 hash wrapper for display purposes
#[derive(Debug, Clone, Copy)]
pub struct Sha256Hash(pub [u8; 32]);

impl fmt::Display for Sha256Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hex_string: String = self.0.iter().map(|b| format!("{:02x}", b)).collect();
        write!(f, "{}", hex_string)
    }
}

impl From<[u8; 32]> for Sha256Hash {
    fn from(hash: [u8; 32]) -> Self {
        Sha256Hash(hash)
    }
}

#[derive(Debug, Error)]
pub enum WasmError {
    #[error(transparent)]
    Module(#[from] WasmModuleError),

    #[error(transparent)]
    Manager(#[from] WasmManagerError),

    #[error(transparent)]
    Runtime(#[from] WasmRuntimeError),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

#[derive(Debug, Error)]
pub enum WasmModuleError {
    #[error("invalid module descriptor: {0}")]
    InvalidDescriptor(String),

    #[error("module with same sha256 already exists: {0}")]
    DuplicateSha256(Sha256Hash),

    #[error("module not found: {0}")]
    NotFound(uuid::Uuid),

    #[error("failed to read module file: {0}")]
    FileRead(String),

    #[error("validation failed: {0}")]
    ValidationFailed(String),

    #[error("attach point missing: {0}")]
    AttachPointMissing(String),

    #[error("invalid function for attach point: {0}")]
    AttachPointFunctionInvalid(String),
}

#[derive(Debug, Error)]
pub enum WasmManagerError {
    #[error("failed to acquire lock: {0}")]
    LockFailed(String),

    #[error("module add failed: {0}")]
    ModuleAddFailed(String),

    #[error("module remove failed: {0}")]
    ModuleRemoveFailed(String),

    #[error("runtime unavailable")]
    RuntimeUnavailable,

    #[error("execution failed: {0}")]
    ExecutionFailed(String),

    #[error("module {0} not found")]
    ModuleNotFound(uuid::Uuid),
}

#[derive(Debug, Error)]
pub enum WasmRuntimeError {
    #[error("failed to create engine: {0}")]
    EngineCreateFailed(String),

    #[error("failed to compile module: {0}")]
    CompileFailed(String),

    #[error("failed to create instance: {0}")]
    InstanceCreateFailed(String),

    #[error("function not found: {0}")]
    FunctionNotFound(String),

    #[error("execution timeout after {0}ms")]
    Timeout(u64),

    #[error("execution failed: {0}")]
    CallFailed(String),
}

impl From<wasmtime::Error> for WasmError {
    fn from(value: wasmtime::Error) -> Self {
        WasmError::Runtime(WasmRuntimeError::CallFailed(value.to_string()))
    }
}
