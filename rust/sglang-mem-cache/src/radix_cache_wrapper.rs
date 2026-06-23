//! Python wrapper of Rust Radix Cache.

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::py_interop::PyTensor;

use crate::component_type::NUM_COMPONENT_TYPES;
use crate::deferred_action::DeferredAction;
use crate::radix_cache::{BigramRadixCache, InsertResult, MatchResult, PageRadixCache};
use crate::tree_node_lru::{EvictRequest, EvictResult};
use crate::utils::parse_device;

/// Per-component eviction outcome at the PyO3 boundary.
///
/// - `freed`: `list[list[torch.Tensor]]` indexed by `ComponentType`
///   discriminant (FULL=0, SWA=1, MAMBA=2); each `freed[ct]` feeds that
///   component's Python allocator free.
/// - `evicted`: per-component token-count total (sum of `t.numel()`).
///
/// `freed` is built once as a `Py<PyList>` so getters are refcount bumps;
/// `tch::Tensor` lacks `Clone`, ruling out the `#[pyo3(get)]` auto-getter.
#[pyclass]
pub struct RustEvictResult {
    freed: Py<PyList>,
    evicted: [usize; NUM_COMPONENT_TYPES],
    deferred_actions: Py<PyList>,
}

/// Serialize any `DeferredAction` to a `PyObject`.
fn deferred_action_to_py(py: Python<'_>, action: DeferredAction) -> PyObject {
    match action {
        DeferredAction::FullDupFreed { freed_indices } => {
            ("FullDupFreed", PyTensor(freed_indices).into_py(py)).into_py(py)
        }
        DeferredAction::SwaRecover {
            node_idx,
            freed_full,
            source_value,
        } => (
            "SwaRecover",
            node_idx,
            PyTensor(freed_full).into_py(py),
            PyTensor(source_value).into_py(py),
        )
            .into_py(py),
        DeferredAction::SwaStamp {
            node_idx,
            source_value,
        } => ("SwaStamp", node_idx, PyTensor(source_value).into_py(py)).into_py(py),
    }
}

/// Serialize a list of `DeferredAction`s into a Python list.
fn deferred_actions_to_py_list(py: Python<'_>, actions: Vec<DeferredAction>) -> Py<PyList> {
    let list = PyList::empty_bound(py);
    for action in actions {
        #[allow(
            clippy::expect_used,
            reason = "PyList::append on a fresh list only fails on OOM"
        )]
        list.append(deferred_action_to_py(py, action))
            .expect("list append cannot fail on a fresh empty list");
    }
    list.into()
}

impl RustEvictResult {
    /// Move a Rust-side `EvictResult` into the PyO3 boundary type, wrapping
    /// the per-component freed tensors into nested Python lists.
    fn from_evict_result(py: Python<'_>, r: EvictResult) -> Self {
        let outer = PyList::empty_bound(py);
        for ct_freed in r.freed {
            let inner =
                PyList::new_bound(py, ct_freed.into_iter().map(|t| PyTensor(t).into_py(py)));
            #[allow(
                clippy::expect_used,
                reason = "PyList::append on fresh list only fails on OOM"
            )]
            outer
                .append(inner)
                .expect("outer list append cannot fail on a fresh empty list");
        }
        Self {
            freed: outer.into(),
            evicted: r.evicted,
            deferred_actions: deferred_actions_to_py_list(py, r.deferred_actions),
        }
    }
}

#[pymethods]
impl RustEvictResult {
    /// Per-component freed tensors to feed the matching Python allocator free.
    #[getter]
    fn freed<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        self.freed.bind(py).clone()
    }

    /// Per-component freed token-count total.
    #[getter]
    fn evicted(&self) -> [usize; NUM_COMPONENT_TYPES] {
        self.evicted
    }

    /// Tagged tuples for the orchestrator to apply.
    #[getter]
    fn deferred_actions<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        self.deferred_actions.bind(py).clone()
    }
}

/// Insert result at the PyO3 boundary.
///
/// `deferred_actions` is a flat Python list of tagged tuples the
/// orchestrator pattern-matches on by string tag:
///   - `("FullDupFreed", freed_indices_tensor)`
///   - `("SwaRecover", node_idx, freed_full_tensor, source_value_tensor)`
///   - `("SwaStamp", node_idx, source_value_tensor)`
#[pyclass]
pub struct RustInsertResult {
    prefix_len: usize,
    leaf_creation_skipped: bool,
    mamba_value_exists: bool,
    deferred_actions: Py<PyList>,
}

impl RustInsertResult {
    fn from_insert_result(py: Python<'_>, r: InsertResult) -> Self {
        Self {
            prefix_len: r.prefix_len,
            leaf_creation_skipped: r.leaf_creation_skipped,
            mamba_value_exists: r.mamba_value_exists,
            deferred_actions: deferred_actions_to_py_list(py, r.deferred_actions),
        }
    }
}

#[pymethods]
impl RustInsertResult {
    #[getter]
    fn prefix_len(&self) -> usize {
        self.prefix_len
    }

    #[getter]
    fn leaf_creation_skipped(&self) -> bool {
        self.leaf_creation_skipped
    }

    #[getter]
    fn mamba_value_exists(&self) -> bool {
        self.mamba_value_exists
    }

    #[getter]
    fn deferred_actions<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        self.deferred_actions.bind(py).clone()
    }
}

/// PyO3 boundary type for `MatchResult`.
#[pyclass]
pub struct RustMatchResult {
    device_indices: PyTensor,
    last_device_node_idx: usize,
    mamba_branching_seqlen: Option<usize>,
    mamba_value: Option<PyTensor>,
}

impl RustMatchResult {
    fn from_match_result(r: MatchResult) -> Self {
        Self {
            device_indices: PyTensor(r.device_indices),
            last_device_node_idx: r.last_device_node_idx,
            mamba_branching_seqlen: r.mamba_branching_seqlen,
            mamba_value: r.mamba_value.map(PyTensor),
        }
    }
}

#[pymethods]
impl RustMatchResult {
    #[getter]
    fn device_indices(&self) -> PyTensor {
        PyTensor(self.device_indices.0.shallow_clone())
    }

    #[getter]
    fn last_device_node_idx(&self) -> usize {
        self.last_device_node_idx
    }

    #[getter]
    fn mamba_branching_seqlen(&self) -> Option<usize> {
        self.mamba_branching_seqlen
    }

    #[getter]
    fn mamba_value(&self) -> Option<PyTensor> {
        self.mamba_value
            .as_ref()
            .map(|t| PyTensor(t.0.shallow_clone()))
    }
}

/// Convert a Python int64 sequence to an owned `Vec<i64>`.
fn py_array_to_vec_i64(py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    // Special handling for empty keys, as empty pyarray might use
    // a random address to represent empty buffer which
    // non-deterministically violates alignment check
    if key.len().map(|n| n == 0).unwrap_or(false) {
        return Ok(Vec::new());
    }
    let buffer = key.extract::<PyBuffer<i64>>()?;
    if !buffer.is_c_contiguous() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Unexpected key received, expected a C-contiguous int64 buffer \
             (e.g. array.array('q'))",
        ));
    }
    buffer.to_vec(py)
}

#[pyclass]
pub struct RustPageRadixCacheWrapper {
    inner: PageRadixCache,
}

#[pymethods]
impl RustPageRadixCacheWrapper {
    #[new]
    #[pyo3(signature = (device, page_size, init_node_capacity, sliding_window_size = None, mamba_cache_chunk_size = None))]
    fn new(
        device: &str,
        page_size: usize,
        init_node_capacity: usize,
        sliding_window_size: Option<usize>,
        mamba_cache_chunk_size: Option<usize>,
    ) -> PyResult<Self> {
        let device = parse_device(device)?;
        let inner = PageRadixCache::new(
            device,
            page_size,
            init_node_capacity,
            sliding_window_size,
            mamba_cache_chunk_size,
        )?;
        Ok(Self { inner })
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn page_size(&self) -> usize {
        self.inner.page_size()
    }

    fn active_tree_node_count(&self) -> usize {
        self.inner.active_tree_node_count()
    }

    /// Sum of `key.len()` across nodes eligible for eviction
    /// (lock_ref == 0 and value present).
    fn evictable_token_size(&self) -> usize {
        self.inner.evictable_token_size()
    }

    /// Sum of `key.len()` across nodes locked by in-flight requests.
    fn protected_token_size(&self) -> usize {
        self.inner.protected_token_size()
    }

    /// Total tokens held by the tree: evictable + protected.
    fn total_token_size(&self) -> usize {
        self.inner.total_token_size()
    }

    /// Sum of `key.len()` across SWA-tracked + SWA-unlocked nodes. 0 for
    /// FULL-only configs (no path credits SWA's aggregate).
    fn swa_evictable_token_size(&self) -> usize {
        self.inner.swa_evictable_token_size()
    }

    /// Sum of `key.len()` across SWA-tracked + SWA-locked nodes.
    fn swa_protected_token_size(&self) -> usize {
        self.inner.swa_protected_token_size()
    }

    /// Count of unlocked nodes with a Mamba value populated.
    fn mamba_evictable_token_size(&self) -> usize {
        self.inner.mamba_evictable_token_size()
    }

    /// Count of locked nodes with a Mamba value populated.
    fn mamba_protected_token_size(&self) -> usize {
        self.inner.mamba_protected_token_size()
    }

    /// Total Mamba slots tracked by the tree: evictable + protected.
    fn mamba_total_size(&self) -> usize {
        self.inner.mamba_total_size()
    }

    /// Run prefix match of `key` on the tree.
    #[pyo3(signature = (key, extra_key = None))]
    fn match_prefix(
        &mut self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        extra_key: Option<String>,
    ) -> PyResult<RustMatchResult> {
        let key_vec = py_array_to_vec_i64(py, key)?;
        let r = py.allow_threads(|| self.inner.match_prefix(&key_vec, extra_key.as_deref()))?;
        Ok(RustMatchResult::from_match_result(r))
    }

    /// Insert `key`/`value`; the returned `prefix_len` is the count already
    /// cached, whose slots the caller frees as redundant duplicates.
    ///
    /// `value` must be a 1-D `Int64` tensor with length >= aligned key length;
    /// excess is silently truncated, shorter values raise `ValueError`. The
    /// cache deep-copies the stored slice, so the caller may reuse its tensor.
    #[pyo3(signature = (key, value, extra_key = None, prev_prefix_len = 0, swa_evicted_seqlen = 0, mamba_value = None))]
    #[allow(clippy::too_many_arguments)]
    fn insert(
        &mut self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: PyTensor,
        extra_key: Option<String>,
        prev_prefix_len: usize,
        swa_evicted_seqlen: usize,
        mamba_value: Option<PyTensor>,
    ) -> PyResult<RustInsertResult> {
        let key_vec = py_array_to_vec_i64(py, key)?;
        let mamba_tensor = mamba_value.map(|m| m.0);
        let r = py.allow_threads(move || {
            self.inner.insert(
                &key_vec,
                &value.0,
                extra_key.as_deref(),
                prev_prefix_len,
                swa_evicted_seqlen,
                mamba_tensor,
            )
        })?;
        Ok(RustInsertResult::from_insert_result(py, r))
    }

    /// Increment lock_ref on `node_idx` and ancestors per each component's
    /// policy (FULL: to namespace root excl.; SWA: window-bounded walk),
    /// protecting the matched prefix from eviction while a request is inflight.
    ///
    /// Returns `(delta, swa_uuid_for_lock)`:
    ///   - `delta` — signed delta to `evictable_token_size`; FULL contributes
    ///     negative (evictable -> protected), SWA contributes `0`.
    ///   - `swa_uuid_for_lock` — `Some(uuid)` when SWA is configured and the
    ///     window filled; `None` otherwise. Caller MUST pass it back to
    ///     `dec_lock_ref` so SWA's release stops at the right boundary.
    ///
    /// `node_idx` is a raw pool index whose slot the freelist may recycle once
    /// the node is evicted; call `inc_lock_ref` before yielding to any evicting
    /// operation, and never reuse the idx past `dec_lock_ref` (stale use panics
    /// or corrupts accounting via ABA).
    ///
    /// TODO(Jialin): [Safety] Replace the raw `usize` handle with a
    /// generation-tagged token so stale uses raise a typed error instead of
    /// panicking.
    fn inc_lock_ref(&mut self, node_idx: usize) -> (i64, Option<u64>) {
        let r = self.inner.inc_lock_ref(node_idx);
        (r.delta, r.swa_uuid_for_lock)
    }
    /// Decrement lock_ref on `node_idx` per each component's policy; pair
    /// exactly with `inc_lock_ref` (underflow panics). Returns the (positive)
    /// delta to `evictable_token_size`.
    ///
    /// `swa_uuid_for_lock` MUST be the value from the matching `inc_lock_ref`;
    /// `None` makes SWA's release walk to root (exclusive). Same `node_idx`
    /// lifecycle caveat as `inc_lock_ref`.
    #[pyo3(signature = (node_idx, swa_uuid_for_lock = None))]
    fn dec_lock_ref(&mut self, node_idx: usize, swa_uuid_for_lock: Option<u64>) -> i64 {
        self.inner.dec_lock_ref(node_idx, swa_uuid_for_lock)
    }

    /// Best-effort cascade evict of up to `num_tokens[ct]` tokens per component
    /// (oldest first). Leaves free whole, so `evicted[ct]` may exceed the target
    /// when the last leaf overshoots, or fall short under lock saturation; no
    /// error is raised, so the caller compares to detect partial fulfillment.
    ///
    /// `num_tokens` is positional, indexed by `ComponentType` discriminant
    /// (FULL=0, SWA=1, MAMBA=2): FULL-only callers pass `[N, 0, 0]`.
    fn evict(
        &mut self,
        py: Python<'_>,
        num_tokens: [usize; NUM_COMPONENT_TYPES],
    ) -> RustEvictResult {
        let r = self.inner.evict(EvictRequest { num_tokens });
        RustEvictResult::from_evict_result(py, r)
    }

    /// Write per-node SWA values back into the tree.
    fn apply_swa_writes(
        &mut self,
        node_indices: Vec<usize>,
        swa_values: Vec<PyTensor>,
    ) -> PyResult<()> {
        let values: Vec<tch::Tensor> = swa_values.into_iter().map(|v| v.0).collect();
        Ok(self.inner.apply_swa_writes(node_indices, values)?)
    }
}

/// Python wrapper for the EAGLE bigram radix cache (children keyed by overlap
/// bigram pairs `(t[i], t[i+1])`).
///
/// Mirrors `RustPageRadixCacheWrapper`'s API so the orchestrator can pick a
/// wrapper by `params.is_eagle` without per-method dispatch. Two differences
/// live inside the wrapper:
///   1. Each key method takes a 1-D `int64` raw-token slice and builds the
///      `Vec<(i64, i64)>` pair view Rust-side, so the pairs never cross PyO3.
///   2. `value` is trimmed N -> N-1 (one slot per pair) so the cache's
///      `value.len() >= key.len()` invariant holds in atom (= pair) units.
///
/// All sizes report in atom (= bigram-pair) units; raw-token translation
/// happens in Python.
///
/// Page-align contract (CRITICAL): the orchestrator MUST pass raw token
/// sequences untrimmed. The cache page-aligns internally in atom units
/// `(N-1) // page_size * page_size`; a raw-token trim in Python would compound
/// with the Rust trim and produce the wrong bigram count.
///
/// TODO(Jialin): [Refactor][Major] Consolidate this path back onto
/// `RustPageRadixCacheWrapper` (`K = Vec<i64>`), retiring this wrapper and the
/// `Vec<(i64, i64)>` `ChildKeyType` impl; the bigram tree branches at the same
/// points as the raw-token tree, just shifted by one index. Done naively this
/// hits subtle value-sizing, edge-label, hash-collision, and page-unit issues,
/// so the explicit `Vec<(i64, i64)>` impl ships as the conservative option.
/// Build overlap-bigram pairs `(t[i], t[i+1])` from a raw 1-D `int64`
/// token slice. Returns an empty `Vec` for inputs shorter than 2
/// tokens (no pairs possible).
fn build_bigram_pairs(raw: &[i64]) -> Vec<(i64, i64)> {
    raw.windows(2).map(|w| (w[0], w[1])).collect()
}

#[pyclass]
pub struct RustBigramRadixCacheWrapper {
    inner: BigramRadixCache,
}

#[pymethods]
impl RustBigramRadixCacheWrapper {
    #[new]
    #[pyo3(signature = (device, page_size, init_node_capacity, sliding_window_size = None, mamba_cache_chunk_size = None))]
    fn new(
        device: &str,
        page_size: usize,
        init_node_capacity: usize,
        sliding_window_size: Option<usize>,
        mamba_cache_chunk_size: Option<usize>,
    ) -> PyResult<Self> {
        if mamba_cache_chunk_size.is_some() {
            return Err(crate::error::RadixCacheInitError::BigramMambaNotSupported.into());
        }
        let device = parse_device(device)?;
        let inner = BigramRadixCache::new(
            device,
            page_size,
            init_node_capacity,
            sliding_window_size,
            mamba_cache_chunk_size,
        )?;
        Ok(Self { inner })
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn page_size(&self) -> usize {
        self.inner.page_size()
    }

    fn active_tree_node_count(&self) -> usize {
        self.inner.active_tree_node_count()
    }

    /// In atom (= bigram-pair) units: an N-token sequence reports up to N-1.
    fn evictable_token_size(&self) -> usize {
        self.inner.evictable_token_size()
    }

    fn protected_token_size(&self) -> usize {
        self.inner.protected_token_size()
    }

    fn total_token_size(&self) -> usize {
        self.inner.total_token_size()
    }

    fn swa_evictable_token_size(&self) -> usize {
        self.inner.swa_evictable_token_size()
    }

    fn swa_protected_token_size(&self) -> usize {
        self.inner.swa_protected_token_size()
    }

    fn mamba_evictable_token_size(&self) -> usize {
        self.inner.mamba_evictable_token_size()
    }

    fn mamba_protected_token_size(&self) -> usize {
        self.inner.mamba_protected_token_size()
    }

    fn mamba_total_size(&self) -> usize {
        self.inner.mamba_total_size()
    }

    /// Run prefix match of `key` on the bigram-keyed tree.
    /// `last_device_node_idx` is in atom (= bigram-pair) units.
    #[pyo3(signature = (key, extra_key = None))]
    fn match_prefix(
        &mut self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        extra_key: Option<String>,
    ) -> PyResult<RustMatchResult> {
        let key_vec = py_array_to_vec_i64(py, key)?;
        let r = py.allow_threads(|| {
            let pairs = build_bigram_pairs(&key_vec);
            self.inner.match_prefix(&pairs, extra_key.as_deref())
        })?;
        Ok(RustMatchResult::from_match_result(r))
    }

    #[pyo3(signature = (key, value, extra_key = None, prev_prefix_len = 0, swa_evicted_seqlen = 0, mamba_value = None))]
    #[allow(clippy::too_many_arguments)]
    fn insert(
        &mut self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: PyTensor,
        extra_key: Option<String>,
        prev_prefix_len: usize,
        swa_evicted_seqlen: usize,
        mamba_value: Option<PyTensor>,
    ) -> PyResult<RustInsertResult> {
        let key_vec = py_array_to_vec_i64(py, key)?;
        let mamba_tensor = mamba_value.map(|m| m.0);
        let r = py.allow_threads(move || {
            let pairs = build_bigram_pairs(&key_vec);
            // Trim value N -> N-1 (one slot per bigram). If raw.len() < 2 the
            // pair vec is empty and the cache early-returns without touching value.
            let trimmed_value = if pairs.is_empty() {
                value.0.shallow_clone()
            } else {
                value.0.narrow(0, 0, pairs.len() as i64)
            };
            self.inner.insert(
                &pairs,
                &trimmed_value,
                extra_key.as_deref(),
                prev_prefix_len,
                swa_evicted_seqlen,
                mamba_tensor,
            )
        })?;
        Ok(RustInsertResult::from_insert_result(py, r))
    }

    fn inc_lock_ref(&mut self, node_idx: usize) -> (i64, Option<u64>) {
        let r = self.inner.inc_lock_ref(node_idx);
        (r.delta, r.swa_uuid_for_lock)
    }
    #[pyo3(signature = (node_idx, swa_uuid_for_lock = None))]
    fn dec_lock_ref(&mut self, node_idx: usize, swa_uuid_for_lock: Option<u64>) -> i64 {
        self.inner.dec_lock_ref(node_idx, swa_uuid_for_lock)
    }

    fn evict(
        &mut self,
        py: Python<'_>,
        num_tokens: [usize; NUM_COMPONENT_TYPES],
    ) -> RustEvictResult {
        let r = self.inner.evict(EvictRequest { num_tokens });
        RustEvictResult::from_evict_result(py, r)
    }

    fn apply_swa_writes(
        &mut self,
        node_indices: Vec<usize>,
        swa_values: Vec<PyTensor>,
    ) -> PyResult<()> {
        let values: Vec<tch::Tensor> = swa_values.into_iter().map(|v| v.0).collect();
        Ok(self.inner.apply_swa_writes(node_indices, values)?)
    }
}
