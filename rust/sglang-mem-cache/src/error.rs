//! Crate-wide error types and their PyO3 boundary conversions.
//!
//! Each error type owns a dedicated `*PyError` Python exception class (inheriting
//! `Exception` directly, not `ValueError`) so callers can catch the typed subclass.

use pyo3::PyErr;

// pyo3 0.22's `create_exception!` references the undeclared `gil-refs` feature.
// Scoped to this inner module so real unexpected_cfgs elsewhere still surface.
#[allow(unexpected_cfgs)]
mod exceptions {
    use pyo3::create_exception;
    use pyo3::exceptions::PyException;
    create_exception!(_mem_cache_core, RadixCacheInitPyError, PyException);
    create_exception!(_mem_cache_core, RadixCacheRuntimePyError, PyException);
    create_exception!(_mem_cache_core, RadixCacheInfraPyError, PyException);
}
pub use exceptions::{RadixCacheInfraPyError, RadixCacheInitPyError, RadixCacheRuntimePyError};

/// Construction-time failures from `RadixCache::new`, the wrapper layer, and
/// `parse_device` (device-string parsing is part of the init surface).
// All variants are "invalid <input>" errors — the shared `Invalid` prefix is
// semantic, not redundant.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, thiserror::Error)]
pub enum RadixCacheInitError {
    /// `page_size` value is invalid for the chosen key type — e.g. `1` for a
    /// multi-token key, or anything other than `1` for a single-token key.
    #[error("invalid page_size: expected {expected}, got {got}")]
    InvalidPageSize { expected: &'static str, got: usize },
    /// Device string passed to the wrapper could not be parsed.
    #[error("unknown device: {0}")]
    InvalidDevice(String),
    /// `pool_size` must be a positive count of KV slots.
    #[error("invalid pool_size: must be > 0, got {got}")]
    InvalidPoolSize { got: i64 },
    /// `sliding_window_size` must be a positive token count when set; pass
    /// `None` to disable SWA (`Some(0)` would gate vacuously).
    #[error("invalid sliding_window_size: must be > 0 when set, got {got}")]
    InvalidSlidingWindowSize { got: usize },
    /// Mamba SSM checkpoints land on chunk-size multiples, so chunk granularity
    /// must cover at least one page (also rejects `chunk_size == 0`).
    #[error("mamba_cache_chunk_size ({chunk_size}) must be >= page_size ({page_size})")]
    MambaCacheChunkSizeBelowPageSize { chunk_size: usize, page_size: usize },
    /// SWA + Mamba combined mode is not supported.
    #[error(
        "SWA + Mamba combined mode is not supported: sliding_window_size and \
         mamba_cache_chunk_size are both set"
    )]
    SwaMambaComboNotSupported,
    /// Bigram (EAGLE) + Mamba combined mode is not supported.
    #[error("Bigram (EAGLE) + Mamba combined mode is not supported")]
    BigramMambaNotSupported,
}

/// Errors surfaced from the runtime API (`match_prefix`, `insert`) when caller
/// inputs violate documented contracts. Maps to `PyValueError` at the wrapper
/// boundary so Python callers see structured exceptions instead of deep
/// `tch::Tensor` panics.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, thiserror::Error)]
pub enum RadixCacheRuntimeError {
    /// Value shorter than the page-aligned key length. Longer values are
    /// silently truncated (symmetric with key truncation) and do NOT error.
    #[error("insert: value length ({value_len}) < aligned key length ({aligned_key_len})")]
    InsertValueLengthMismatch {
        aligned_key_len: usize,
        value_len: usize,
    },
    /// Value tensor must be `Int64` to match the cache's slot-index dtype.
    #[error("insert: value dtype must be Int64, got {got:?}")]
    InsertValueWrongDtype { got: tch::Kind },
    /// Value tensor must be 1-D — a flat slice of KV slot indices.
    #[error("insert: value must be 1-D, got shape {got:?}")]
    InsertValueWrongShape { got: Vec<i64> },
    /// Value must live on the cache device; caught here so a mismatched device
    /// fails at the boundary instead of as an opaque `Tensor::cat` libtorch error.
    #[error("insert: value device {got:?} does not match cache device {expected:?}")]
    InsertValueWrongDevice {
        expected: tch::Device,
        got: tch::Device,
    },
    /// Mamba value must be `Int64` to match the Mamba pool slot-index dtype.
    #[error("insert: mamba_value dtype must be Int64, got {got:?}")]
    InsertMambaValueWrongDtype { got: tch::Kind },
    /// Mamba value must be 1-D with a single element (1 SSM state per node).
    #[error("insert: mamba_value must be 1-D with shape [1], got shape {got:?}")]
    InsertMambaValueWrongShape { got: Vec<i64> },
    /// Mamba value tensor device must match the cache device.
    #[error("insert: mamba_value device {got:?} does not match cache device {expected:?}")]
    InsertMambaValueWrongDevice {
        expected: tch::Device,
        got: tch::Device,
    },
    /// `mamba_value` passed but Mamba is not configured on this cache.
    #[error("insert: mamba_value passed but mamba_cache_chunk_size is None")]
    InsertMambaValueWithoutMambaConfigured,
    /// Mismatched-length argument lists to `apply_swa_writes`. These are built in
    /// lockstep from `insert`'s `Recover` actions, so a mismatch is a caller bug.
    #[error("apply_swa_writes: node_indices length ({indices}) != swa_values length ({values})")]
    ApplySwaWritesMismatch { indices: usize, values: usize },
    #[error("node {node_idx} already has a {slot} value")]
    DuplicateValueSet { node_idx: usize, slot: &'static str },
    #[error("cannot set a {slot} value on the root node")]
    ValueSetOnRoot { slot: &'static str },
    #[error("un-evict: device-absent node {node_idx} has a non-zero lock_ref")]
    UnevictLockedNode { node_idx: usize },
}

/// Child-key construction failures at insert/match time: the `ids` slice is too
/// short for the configured page size. (Page-size validation itself happens at
/// construction via `RadixCacheInitError::InvalidPageSize`.) Maps to
/// `RadixCacheInfraPyError`.
///
/// TODO(Jialin): fold into `RadixCacheInfraError` once that surface lands.
#[derive(Debug, thiserror::Error)]
pub enum ChildKeyError {
    /// `ids` slice is too short for the requested `page_size`.
    #[error("ids length ({len}) < page_size ({page_size})")]
    SliceTooShort { page_size: usize, len: usize },
}

/// Caller requested a feature the integration does not support (e.g. non-LRU
/// eviction, `enable_kv_cache_events`, `cache_ttl_seconds`). Maps to
/// `RadixCacheInfraPyError` so misconfigured deployments fail fast at
/// construction instead of drifting behaviorally.
#[derive(Debug, thiserror::Error)]
pub enum RadixCacheInfraError {
    /// The string names the feature, e.g. `"InsertParams.priority != 0"`.
    #[error("unsupported feature: {0}")]
    UnsupportedFeature(String),
}

impl From<RadixCacheInitError> for PyErr {
    fn from(err: RadixCacheInitError) -> Self {
        RadixCacheInitPyError::new_err(err.to_string())
    }
}

impl From<ChildKeyError> for PyErr {
    fn from(err: ChildKeyError) -> Self {
        RadixCacheInfraPyError::new_err(err.to_string())
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
