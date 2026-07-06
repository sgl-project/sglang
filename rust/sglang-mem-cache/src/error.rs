//! Crate-wide error types and their PyO3 boundary conversions.

use pyo3::PyErr;

// pyo3 0.22's `create_exception!` references the undeclared `gil-refs` feature.
#[allow(unexpected_cfgs)]
mod exceptions {
    use pyo3::create_exception;
    use pyo3::exceptions::PyException;
    create_exception!(_mem_cache_core, RadixCacheInitPyError, PyException);
    create_exception!(_mem_cache_core, RadixCacheRuntimePyError, PyException);
    create_exception!(_mem_cache_core, RadixCacheInfraPyError, PyException);
}
pub use exceptions::{RadixCacheInfraPyError, RadixCacheInitPyError, RadixCacheRuntimePyError};

/// Engine setup errors.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, thiserror::Error)]
pub enum RadixCacheInitError {
    #[error("invalid page_size: expected {expected}, got {got}")]
    InvalidPageSize { expected: &'static str, got: usize },
    #[error("unknown device: {0}")]
    InvalidDevice(String),
    #[error("invalid pool_size: must be > 0, got {got}")]
    InvalidPoolSize { got: i64 },
    #[error("invalid sliding_window_size: must be > 0 when set, got {got}")]
    InvalidSlidingWindowSize { got: usize },
    #[error("mamba_cache_chunk_size ({chunk_size}) must be >= page_size ({page_size})")]
    MambaCacheChunkSizeBelowPageSize { chunk_size: usize, page_size: usize },
    #[error(
        "SWA + Mamba combined mode is not supported: sliding_window_size and \
         mamba_cache_chunk_size are both set"
    )]
    SwaMambaComboNotSupported,
    #[error("Bigram (EAGLE) + Mamba combined mode is not supported")]
    BigramMambaNotSupported,
}

/// Engine runtime errors.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, thiserror::Error)]
pub enum RadixCacheRuntimeError {
    #[error("insert: value length ({value_len}) < aligned key length ({aligned_key_len})")]
    InsertValueLengthMismatch {
        aligned_key_len: usize,
        value_len: usize,
    },
    #[error("insert: value dtype must be Int64, got {got:?}")]
    InsertValueWrongDtype { got: tch::Kind },
    #[error("insert: value must be 1-D, got shape {got:?}")]
    InsertValueWrongShape { got: Vec<i64> },
    #[error("insert: value device {got:?} does not match cache device {expected:?}")]
    InsertValueWrongDevice {
        expected: tch::Device,
        got: tch::Device,
    },
    #[error("insert: mamba_value dtype must be Int64, got {got:?}")]
    InsertMambaValueWrongDtype { got: tch::Kind },
    #[error("insert: mamba_value must be 1-D with shape [1], got shape {got:?}")]
    InsertMambaValueWrongShape { got: Vec<i64> },
    #[error("insert: mamba_value device {got:?} does not match cache device {expected:?}")]
    InsertMambaValueWrongDevice {
        expected: tch::Device,
        got: tch::Device,
    },
    #[error("insert: mamba_value passed but mamba_cache_chunk_size is None")]
    InsertMambaValueWithoutMambaConfigured,
    #[error("apply_swa_writes: node_indices length ({indices}) != swa_values length ({values})")]
    ApplySwaWritesMismatch { indices: usize, values: usize },
    #[error("node {node_idx} already has a {slot} value")]
    DuplicateValueSet { node_idx: usize, slot: &'static str },
    #[error("cannot set a {slot} value on the root node")]
    ValueSetOnRoot { slot: &'static str },
    #[error("un-evict: device-absent node {node_idx} has a non-zero lock_ref")]
    UnevictLockedNode { node_idx: usize },
}

/// Caller requested a feature the integration does not support.
#[derive(Debug, thiserror::Error)]
pub enum RadixCacheInfraError {
    #[error("unsupported feature: {0}")]
    UnsupportedFeature(String),
}

impl From<RadixCacheInitError> for PyErr {
    fn from(err: RadixCacheInitError) -> Self {
        RadixCacheInitPyError::new_err(err.to_string())
    }
}

impl From<RadixCacheRuntimeError> for PyErr {
    fn from(err: RadixCacheRuntimeError) -> Self {
        RadixCacheRuntimePyError::new_err(err.to_string())
    }
}

impl From<RadixCacheInfraError> for PyErr {
    fn from(err: RadixCacheInfraError) -> Self {
        RadixCacheInfraPyError::new_err(err.to_string())
    }
}
