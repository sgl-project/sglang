//! Crate-wide error types and their PyO3 boundary conversions.
//!
//! Each error type owns a dedicated Python exception class so wrapper call
//! sites can use bare `?` and Python callers get typed exceptions to catch.
//!
//! Naming pattern (mirrors pyo3 convention): the Rust enum keeps the natural
//! name (`RadixCache{Init,Runtime,Infra}Error`); the Python-visible exception
//! class gets a `Py` suffix (`RadixCache{Init,Runtime,Infra}PyError`).
//!
//! All three Python exception classes inherit from `Exception` directly, so
//! callers must catch the typed subclass (or the broad `Exception`) — they
//! are intentionally NOT `ValueError` subclasses.

use pyo3::PyErr;

// pyo3 0.22's `create_exception!` macro references the `gil-refs` cargo
// feature, which our pyo3 dependency doesn't declare. Scoped to this inner
// module so future real unexpected_cfgs in the rest of error.rs still surface.
#[allow(unexpected_cfgs)]
mod exceptions {
    use pyo3::create_exception;
    use pyo3::exceptions::PyException;
    create_exception!(_mem_cache_core, RadixCacheInitPyError, PyException);
    create_exception!(_mem_cache_core, RadixCacheRuntimePyError, PyException);
    create_exception!(_mem_cache_core, RadixCacheInfraPyError, PyException);
}
pub use exceptions::{RadixCacheInfraPyError, RadixCacheInitPyError, RadixCacheRuntimePyError};

/// Error returned by `RadixCache::new` (and the wrapper layer that feeds it).
/// Covers all construction-time failure modes; new variants land alongside
/// operator PRs as needed.
///
/// Also returned by `parse_device` since device-string parsing is part of the
/// init surface from a Python-user perspective. The `kv_pool_allocator` test
/// wrapper inherits this error type for the same reason.
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
    /// `sliding_window_size` must be a positive token count when SWA mode
    /// is enabled. `Some(0)` is degenerate (the validator would always
    /// pass vacuously, defeating the purpose of SWA gating); pass `None`
    /// to disable SWA instead.
    #[error("invalid sliding_window_size: must be > 0 when set, got {got}")]
    InvalidSlidingWindowSize { got: usize },
    /// Mamba SSM checkpoints land on chunk-size multiples, so the chunk
    /// granularity must cover at least one page worth of tokens (page_size
    /// is itself >= 1, so this also rejects `chunk_size == 0`).
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
    /// Value tensor is shorter than the page-aligned key length, so the leaf
    /// can't hold a value matching its key. Longer-than-aligned values are
    /// silently truncated (symmetric with key truncation) and do NOT trigger
    /// this error.
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
    /// Value tensor must live on the same device as the cache. A CPU tensor
    /// passed to a CUDA cache (or vice versa) would later mismatch in
    /// `Tensor::cat` with an opaque libtorch error — catch it at the boundary.
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
    /// `apply_swa_writes` was called with mismatched-length argument lists.
    /// The orchestrator builds these lists in lockstep from the `Recover`
    /// deferred actions returned by `insert`, so a length mismatch
    /// indicates a caller bug (lost / duplicated entries between
    /// translation and write-back).
    #[error("apply_swa_writes: node_indices length ({indices}) != swa_values length ({values})")]
    ApplySwaWritesMismatch { indices: usize, values: usize },
    #[error(
        "set_host_full_values: node_indices length ({indices}) != host_values length ({values})"
    )]
    SetHostFullValuesMismatch { indices: usize, values: usize },
    #[error("set_host_full_values: node {node_idx} already has a host value")]
    HostValueAlreadyBackedUp { node_idx: usize },
    #[error(
        "set_host_full_values: node {node_idx} backed up before its parent \
         {parent_idx} (host prefix contiguity violated)"
    )]
    HostBackupParentNotBackedUp { node_idx: usize, parent_idx: usize },
    #[error("match_prefix: node {node_idx} missing FULL value with HiCache disabled")]
    MatchPrefixMissingFullValue { node_idx: usize },
    #[error(
        "prepare_load_back: node {node_idx} device-evicted but missing host value \
         (host prefix contiguity violated)"
    )]
    PrepareLoadBackMissingHostValue { node_idx: usize },
    #[error(
        "postprocess_load_back: device_values length ({got}) != chain token count ({expected})"
    )]
    PostprocessLoadBackLengthMismatch { got: usize, expected: usize },
    #[error("node {node_idx} already has a {slot} value")]
    DuplicateValueSet { node_idx: usize, slot: &'static str },
    #[error("cannot set a {slot} value on the root node")]
    ValueSetOnRoot { slot: &'static str },
    #[error("un-evict: device-absent node {node_idx} has a non-zero lock_ref")]
    UnevictLockedNode { node_idx: usize },
}

/// Error type for child key construction at insert/match time — i.e. the input
/// `ids` slice is too short for the page size already configured on the pool.
/// Page-size *validation* itself happens at construction and is reported via
/// `RadixCacheInitError::InvalidPageSize`.
///
/// TODO(Jialin): Fold into `RadixCacheInfraError` once the broader infra-error
/// surface lands (duplicate insert, evicted-node access, split-precondition
/// violations). Until then, `ChildKeyError` keeps its own dedicated type and
/// maps to the same `RadixCacheInfraPyError` Python class so callers catching
/// the eventual unified type already work.
#[derive(Debug, thiserror::Error)]
pub enum ChildKeyError {
    /// `ids` slice is too short for the requested `page_size`.
    #[error("ids length ({len}) < page_size ({page_size})")]
    SliceTooShort { page_size: usize, len: usize },
}

/// Errors surfaced when callers request features the v1 integration does not
/// support (e.g. non-LRU eviction, EAGLE bigram, SWA, Mamba, HiCache,
/// `enable_kv_cache_events`, `cache_ttl_seconds`, etc.). Maps to the typed
/// `RadixCacheInfraPyError` Python exception so deployments that misconfigure
/// fail fast at construction with a structured error instead of a silent
/// behavioral drift.
///
/// Future runtime-infra variants (per `ChildKeyError` TODO) land here too.
#[derive(Debug, thiserror::Error)]
pub enum RadixCacheInfraError {
    /// Caller requested a feature the current Rust integration does not
    /// implement. The string identifies the specific feature, e.g.
    /// `"InsertParams.priority != 0 (LRU only in v1)"`.
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
