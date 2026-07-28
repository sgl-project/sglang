use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeCode {
    InvalidArgument,
    TooManyRequests,
    AddressNotRegistered,
    BatchBusy,
    DeviceNotFound,
    AddressOverlapped,
    NotSupportedTransport,
    Dns,
    Socket,
    MalformedJson,
    RejectHandshake,
    Metadata,
    Endpoint,
    Context,
    Numa,
    Clock,
    Memory,
    NotImplemented,
    Unknown(i32),
}

impl NativeCode {
    pub fn from_raw(raw: i32) -> Self {
        match raw.unsigned_abs() {
            1 => Self::InvalidArgument,
            2 => Self::TooManyRequests,
            3 => Self::AddressNotRegistered,
            4 => Self::BatchBusy,
            6 => Self::DeviceNotFound,
            7 => Self::AddressOverlapped,
            8 => Self::NotSupportedTransport,
            101 => Self::Dns,
            102 => Self::Socket,
            103 => Self::MalformedJson,
            104 => Self::RejectHandshake,
            200 => Self::Metadata,
            201 => Self::Endpoint,
            202 => Self::Context,
            300 => Self::Numa,
            301 => Self::Clock,
            302 => Self::Memory,
            303 | 999 => Self::NotImplemented,
            _ => Self::Unknown(raw),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeOperation {
    SetCudaDevice,
    AllocatePinnedMemory,
    AllocateCudaMemory,
    CopyMemory,
    CreateEngine,
    GetLocalEndpoint,
    InstallTransport,
    RegisterRegion,
    UnregisterRegion,
    OpenPeer,
    ClosePeer,
    AllocateBatch,
    SubmitBatch,
    Poll,
    FreeBatch,
    UninstallTransport,
    DestroyEngine,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EngineError {
    #[error("unsupported frozen PD GPU {device}; expected 4 or 5")]
    UnsupportedGpu { device: u32 },

    #[error("invalid {field}: {detail}")]
    InvalidDescriptor { field: &'static str, detail: String },

    #[error("range overflow while validating {field}")]
    RangeOverflow { field: &'static str },

    #[error("{field} range [{offset}, {end}) exceeds registered length {registered_length}")]
    RangeOutOfBounds {
        field: &'static str,
        offset: u64,
        end: u64,
        registered_length: u64,
    },

    #[error("Mooncake owner command queue is full")]
    QueueFull,

    #[error("Mooncake owner worker is closed")]
    WorkerClosed,

    #[error("Mooncake owner response timed out during {operation}")]
    ResponseTimeout { operation: &'static str },

    #[error("native in-flight batch limit reached ({limit})")]
    InFlightLimit { limit: usize },

    #[error("{kind} {id} is closed or unknown")]
    ResourceClosed { kind: &'static str, id: u64 },

    #[error("batch {id} is not safely terminal")]
    BatchNotTerminal { id: u64 },

    #[error("native {operation:?} failed with {code:?} (raw code {raw_code})")]
    Native {
        operation: NativeOperation,
        code: NativeCode,
        raw_code: i32,
    },

    #[error("native {operation:?} returned a null/invalid handle")]
    NativeHandle { operation: NativeOperation },

    #[error("Mooncake native library is missing: {path}")]
    LibraryMissing { path: PathBuf },

    #[error("Mooncake ABI manifest is missing: {path}")]
    ManifestMissing { path: PathBuf },

    #[error("Mooncake artifact mismatch for {field}: expected {expected}, got {actual}")]
    ArtifactMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },

    #[error("Mooncake ABI mismatch: {detail}")]
    AbiMismatch { detail: String },

    #[error("Mooncake required symbol is missing: {symbol}")]
    SymbolMissing { symbol: String },

    #[error("Mooncake loader failed for {path}: {detail}")]
    LoaderFailure { path: PathBuf, detail: String },

    #[error("native engine was created in pid {creator_pid} but called in pid {current_pid}")]
    ForkDetected { creator_pid: u32, current_pid: u32 },

    #[error("owner worker failed to start: {detail}")]
    WorkerStart { detail: String },

    #[error("native rollback after {operation:?} also failed: {cleanup}")]
    Rollback {
        operation: NativeOperation,
        cleanup: String,
    },

    #[error("internal Mooncake state lock is poisoned")]
    LockPoisoned,
}

impl EngineError {
    pub(crate) fn native(operation: NativeOperation, raw_code: i32) -> Self {
        let code = match operation {
            NativeOperation::SetCudaDevice
            | NativeOperation::AllocatePinnedMemory
            | NativeOperation::AllocateCudaMemory
            | NativeOperation::CopyMemory => NativeCode::Unknown(raw_code),
            _ => NativeCode::from_raw(raw_code),
        };
        Self::Native {
            operation,
            code,
            raw_code,
        }
    }
}
