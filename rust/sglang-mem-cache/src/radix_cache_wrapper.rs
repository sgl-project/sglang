//! Python wrapper of Rust Radix Cache.

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::py_interop::PyTensor;

use crate::component_type::NUM_COMPONENT_TYPES;
use crate::deferred_action::DeferredAction;
use crate::radix_cache::{
    BigramRadixCache, InsertResult, MatchResult, PageRadixCache, PrepareLoadBackResult,
};
use crate::tree_node_lru::{EvictRequest, EvictResult};
use crate::utils::parse_device;

/// Per-component eviction outcome at the PyO3 boundary. Mirrors the
/// Rust-internal `tree_node_lru::EvictResult` shape.
///
/// * `freed`: `list[list[torch.Tensor]]` indexed by `ComponentType`
///   discriminant (FULL=0, SWA=1, MAMBA=2). `freed[ct]` is the list
///   Python's component-`ct` allocator free should consume — FULL →
///   `allocator.free(t)`; SWA → `allocator.free_swa(t)`; future
///   Mamba → `mamba_pool.free(t)`.
/// * `evicted`: `list[int]` (length `NUM_COMPONENT_TYPES`) — the
///   running token-count total per component (sum of `t.numel()`
///   across `freed[ct]`).
///
/// Implementation note: `freed` is built once at construction as a
/// Python-managed `Py<PyList>` so the `#[getter]` returns a
/// refcount-bumped handle (cheap) without needing `Clone` on
/// `PyTensor` — `tch::Tensor` only has `shallow_clone`, not `Clone`,
/// which rules out the `#[pyo3(get)]` auto-getter shape.
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
        DeferredAction::FullWriteThroughBackup { node_idx, value } => (
            "FullWriteThroughBackup",
            node_idx,
            PyTensor(value).into_py(py),
        )
            .into_py(py),
        DeferredAction::FullDeviceEvictOnBackedUp {
            node_idx,
            device_value,
        } => (
            "FullDeviceEvictOnBackedUp",
            node_idx,
            PyTensor(device_value).into_py(py),
        )
            .into_py(py),
        DeferredAction::FullHostEvict {
            node_idx,
            host_value,
        } => ("FullHostEvict", node_idx, PyTensor(host_value).into_py(py)).into_py(py),
        DeferredAction::FullWriteBackOnEvict { node_idx, value } => (
            "FullWriteBackOnEvict",
            node_idx,
            PyTensor(value).into_py(py),
        )
            .into_py(py),
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
    /// Move a Rust-side `EvictResult` into the PyO3 boundary type,
    /// wrapping each per-component freed-tensor `Vec<Tensor>` into a
    /// Python list and the outer per-component dimension into another
    /// Python list. The result holds only Python-managed references —
    /// repeat getter access is just refcount bumps.
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
    /// `freed[ct]`: list Python's component-`ct` allocator free should
    /// consume. Indexed by `ComponentType` discriminant.
    #[getter]
    fn freed<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        self.freed.bind(py).clone()
    }

    /// `evicted[ct]`: running token-count total per component.
    /// Indexed by `ComponentType` discriminant.
    #[getter]
    fn evicted(&self) -> [usize; NUM_COMPONENT_TYPES] {
        self.evicted
    }

    /// `deferred_actions`: tagged tuples for the orchestrator to apply.
    #[getter]
    fn deferred_actions<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        self.deferred_actions.bind(py).clone()
    }
}

/// Insert result at the PyO3 boundary. Wraps the Rust-internal
/// `InsertResult` with Python-accessible fields.
///
/// `deferred_actions` is a flat Python list of tuples:
///   - `("FullDupFreed", freed_indices_tensor)`
///   - `("SwaRecover", node_idx, freed_full_tensor, source_value_tensor)`
///   - `("SwaStamp", node_idx, source_value_tensor)`
///   - `("FullWriteThroughBackup", node_idx, device_value_tensor)`
///
/// Using tagged tuples (not a pyclass enum) keeps the boundary simple and
/// avoids proliferating one-off pyclass types — the Python orchestrator
/// pattern-matches on the string tag.
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
    last_host_node_idx: usize,
    host_only_length: usize,
    mamba_branching_seqlen: Option<usize>,
    mamba_value: Option<PyTensor>,
}

impl RustMatchResult {
    fn from_match_result(r: MatchResult) -> Self {
        Self {
            device_indices: PyTensor(r.device_indices),
            last_device_node_idx: r.last_device_node_idx,
            last_host_node_idx: r.last_host_node_idx,
            host_only_length: r.host_only_length,
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
    fn last_host_node_idx(&self) -> usize {
        self.last_host_node_idx
    }

    #[getter]
    fn host_only_length(&self) -> usize {
        self.host_only_length
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

#[pyclass]
pub struct RustPrepareLoadBackResult {
    chain: Vec<usize>,
    host_indices: PyTensor,
    ancestor_node_idx: usize,
}

impl RustPrepareLoadBackResult {
    fn from_prepare_load_back_result(r: PrepareLoadBackResult) -> Self {
        Self {
            chain: r.chain,
            host_indices: PyTensor(r.host_indices),
            ancestor_node_idx: r.ancestor_node_idx,
        }
    }
}

#[pymethods]
impl RustPrepareLoadBackResult {
    #[getter]
    fn chain(&self) -> Vec<usize> {
        self.chain.clone()
    }

    #[getter]
    fn host_indices(&self) -> PyTensor {
        PyTensor(self.host_indices.0.shallow_clone())
    }

    #[getter]
    fn ancestor_node_idx(&self) -> usize {
        self.ancestor_node_idx
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
    #[pyo3(signature = (device, page_size, init_node_capacity, sliding_window_size = None, mamba_cache_chunk_size = None, enable_hicache = false, hicache_write_back = false))]
    fn new(
        device: &str,
        page_size: usize,
        init_node_capacity: usize,
        sliding_window_size: Option<usize>,
        mamba_cache_chunk_size: Option<usize>,
        enable_hicache: bool,
        hicache_write_back: bool,
    ) -> PyResult<Self> {
        let device = parse_device(device)?;
        let inner = PageRadixCache::new(
            device,
            page_size,
            init_node_capacity,
            sliding_window_size,
            mamba_cache_chunk_size,
            enable_hicache,
            hicache_write_back,
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

    /// Sum of `key.len()` across nodes currently eligible for eviction
    /// (lock_ref == 0 and value present). Mirrors Python `evictable_size()`.
    fn evictable_token_size(&self) -> usize {
        self.inner.evictable_token_size()
    }

    /// Sum of `key.len()` across nodes currently locked by in-flight
    /// requests. Mirrors Python `protected_size()`.
    fn protected_token_size(&self) -> usize {
        self.inner.protected_token_size()
    }

    /// Total tokens held by the tree right now: evictable + protected.
    /// Mirrors Python `total_size()`.
    fn total_token_size(&self) -> usize {
        self.inner.total_token_size()
    }

    /// Sum of `key.len()` across SWA-tracked + SWA-unlocked nodes.
    /// Mirrors OSS `swa_radix_cache.py`'s `swa_evictable_size`. 0
    /// for FULL-only configs (no path credits SWA's aggregate).
    fn swa_evictable_token_size(&self) -> usize {
        self.inner.swa_evictable_token_size()
    }

    /// Sum of `key.len()` across SWA-tracked + SWA-locked nodes.
    /// Mirrors OSS `swa_radix_cache.py`'s `swa_protected_size`.
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

    /// Walk the host-only (device-evicted, host-backed) chain from `node_idx`
    /// up to the first device-present node. Also lock the chain to avoid eviction.
    fn prepare_load_back(&mut self, node_idx: usize) -> PyResult<RustPrepareLoadBackResult> {
        Ok(RustPrepareLoadBackResult::from_prepare_load_back_result(
            self.inner.prepare_load_back(node_idx)?,
        ))
    }

    /// Returns `prefix_len` — number of tokens already cached before this insert.
    /// Caller (scheduler) frees `value[:prefix_len]` since those slots are now
    /// redundant duplicates of the already-cached prefix.
    ///
    /// `value` must be a 1-D `Int64` tensor with length >= aligned key length.
    /// Excess length is silently truncated (symmetric with key truncation);
    /// shorter values are rejected with `ValueError`. The cache deep-copies
    /// the value slice it stores, so callers may mutate or drop their tensor
    /// after this call returns.
    #[pyo3(signature = (key, value, extra_key = None, prev_prefix_len = 0, swa_evicted_seqlen = 0, mamba_value = None, chunked = false))]
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
        chunked: bool,
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
                chunked,
            )
        })?;
        Ok(RustInsertResult::from_insert_result(py, r))
    }

    /// Increment lock_ref on `node_idx` and every ancestor per each
    /// configured component's policy (FULL: walk to namespace root
    /// excl.; SWA: window-bounded walk to boundary). Use the
    /// `last_device_node_idx` returned by `match_prefix` to protect
    /// the matched prefix from eviction while a request is in flight.
    ///
    /// Returns `(delta, swa_uuid_for_lock)`:
    ///   * `delta` — signed delta to `evictable_token_size`; FULL
    ///     contributes a negative value (tokens shifted from
    ///     evictable to protected). SWA contributes `0` today
    ///     (SWA's per-pool size aggregates aren't wired up yet — see
    ///     the TODO at the SWA size update site in
    ///     `SwaComponent::inc_lock_ref`).
    ///   * `swa_uuid_for_lock` — `Some(uuid)` when SWA is configured
    ///     and the window filled (uuid stamped or reused at the
    ///     boundary node); `None` for FULL-only configs and for SWA
    ///     configs where `sliding_window_size > total_path_len`.
    ///     Caller MUST pass this back to `dec_lock_ref` so SWA's
    ///     release stops at the right boundary.
    ///
    /// **Caller-side handle lifecycle (read carefully):** `node_idx`
    /// is a raw pool index. The pool's freelist may recycle the slot
    /// if the node is evicted between `match_prefix` and
    /// `inc_lock_ref`. Callers (Python facade) MUST call
    /// `inc_lock_ref` before yielding to any operation that could
    /// evict — typical pattern: `idx, ... = match_prefix(); delta,
    /// uuid = inc_lock_ref(idx); /* request inflight */;
    /// dec_lock_ref(idx, uuid)`. Holding `node_idx` past
    /// `dec_lock_ref` is a caller bug; using a stale idx will either
    /// panic on access or silently corrupt accounting if the slot was
    /// reallocated (ABA).
    ///
    /// TODO(Jialin): [Safety] Replace raw `usize` handle with a
    /// generation-tagged opaque token so stale uses surface as a typed
    /// `StaleHandleError` (Python `ValueError`) instead of panicking. Out
    /// of scope for this PR — bounded by the Python facade owning the
    /// handle within a single request lifecycle.
    fn inc_lock_ref(&mut self, node_idx: usize) -> (i64, Option<u64>) {
        let r = self.inner.inc_lock_ref(node_idx);
        (r.delta, r.swa_uuid_for_lock)
    }
    /// Decrement lock_ref on `node_idx` per each configured
    /// component's policy. Pair exactly with `inc_lock_ref` —
    /// underflow panics on the per-slot mutator. Returns the
    /// (positive) delta to `evictable_token_size`.
    ///
    /// `swa_uuid_for_lock` MUST be the value returned by the matching
    /// `inc_lock_ref` call. `None` for FULL-only configs (or SWA
    /// configs where the window didn't fill) — SWA's release walks
    /// to root in that case (matches OSS `swa_radix_cache.py`'s
    /// `dec_lock_ref` "unlocks to the root, exclusive" semantics).
    ///
    /// Same `node_idx` lifecycle caveat as `inc_lock_ref` (see TODO
    /// there).
    #[pyo3(signature = (node_idx, swa_uuid_for_lock = None))]
    fn dec_lock_ref(&mut self, node_idx: usize, swa_uuid_for_lock: Option<u64>) -> i64 {
        self.inner.dec_lock_ref(node_idx, swa_uuid_for_lock)
    }

    /// Best-effort cascade evict — evict up to `num_tokens[ct]` tokens
    /// per component (oldest first by access time) and return
    /// `RustEvictResult { freed, evicted }` per component. Eviction
    /// works at leaf granularity: leaves are freed whole, so
    /// `evicted[ct]` may exceed `num_tokens[ct]` when the last popped
    /// leaf is larger than the remaining target. When the cache has
    /// fewer evictable tokens than requested (e.g., heavy lock
    /// saturation), `evicted[ct] < num_tokens[ct]` and no error is
    /// raised — caller compares to detect partial fulfillment.
    ///
    /// `num_tokens` is a positional length-`NUM_COMPONENT_TYPES` list
    /// indexed by `ComponentType` discriminant (FULL=0, SWA=1,
    /// MAMBA=2). FULL-only callers pass `[N, 0, 0]`; SWA-configured
    /// callers pass `[N_full, N_swa, 0]`.
    fn evict(
        &mut self,
        py: Python<'_>,
        num_tokens: [usize; NUM_COMPONENT_TYPES],
    ) -> RustEvictResult {
        let r = self.inner.evict(EvictRequest { num_tokens });
        RustEvictResult::from_evict_result(py, r)
    }

    /// Best-effort to evict at least `num_tokens` host FULL values.
    fn evict_host(&mut self, py: Python<'_>, num_tokens: usize) -> RustEvictResult {
        let r = self.inner.evict_host(num_tokens);
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

    /// Populate backup host values onto the given nodes.
    fn set_host_full_values(
        &mut self,
        node_indices: Vec<usize>,
        host_values: Vec<PyTensor>,
    ) -> PyResult<()> {
        let values: Vec<tch::Tensor> = host_values.into_iter().map(|v| v.0).collect();
        Ok(self.inner.set_host_full_values(node_indices, values)?)
    }

    /// Populate backup host values onto the given nodes and lock device values
    /// (to avoid async copy corruption).
    fn set_host_full_values_and_lock_device(
        &mut self,
        node_indices: Vec<usize>,
        host_values: Vec<PyTensor>,
    ) -> PyResult<()> {
        let values: Vec<tch::Tensor> = host_values.into_iter().map(|v| v.0).collect();
        self.inner
            .set_host_full_values(node_indices.clone(), values)?;
        for idx in node_indices {
            let _ = self.inner.inc_lock_ref(idx);
        }
        Ok(())
    }

    /// Commit the loadback results back to the radix tree. A `None`
    /// `device_values` indicates loadback failure.
    #[pyo3(signature = (chain, ancestor_node_idx, device_values=None))]
    fn postprocess_load_back(
        &mut self,
        chain: Vec<usize>,
        ancestor_node_idx: usize,
        device_values: Option<PyTensor>,
    ) -> PyResult<()> {
        Ok(self.inner.postprocess_load_back(
            chain,
            ancestor_node_idx,
            device_values.map(|v| v.0),
        )?)
    }
}

/// Production Python wrapper for the EAGLE bigram radix cache (children
/// keyed by overlap bigram pairs `(t[i], t[i+1])`).
///
/// Mirrors `RustPageRadixCacheWrapper`'s API exactly so the Python
/// orchestrator can pick which wrapper to instantiate based on
/// `params.is_eagle` without any per-method dispatch logic. The two
/// differences live entirely inside the wrapper:
///   1. **Bigram pair construction.** Each public method that takes a
///      key receives a 1-D `int64` raw-token slice (same shape as the
///      Page wrapper). The wrapper builds the overlap-pair view
///      `Vec<(i64, i64)>` via `windows(2).map(...).collect()` on the
///      Rust side — never crosses the Python/Rust boundary as a
///      `(N, 2)` array, so there's no extra PyO3 transport cost.
///   2. **Value tensor trim N → N−1.** Incoming `value` has one slot
///      per raw token (length N); the bigram cache stores one slot per
///      bigram pair (length N−1). The wrapper slices `value[..N-1]`
///      before forwarding so the cache's `value.len() >= key.len()`
///      invariant holds at the (atom = pair) granularity used inside.
///
/// All accessor / lock_ref / evict / apply_swa_writes methods report
/// in "atom units" = bigram pairs. Callers translating between raw
/// tokens and bigrams (e.g., for accept-length accounting against
/// OSS semantics) do that translation in Python.
///
/// **Page-align contract (CRITICAL for callers):** the Python
/// orchestrator MUST pass raw token sequences directly without any
/// pre-call trimming. The cache's `PageAlignedQueryKey::new`
/// performs the page-alignment trim internally — in bigram-atom
/// units `(N-1) // page_size * page_size`, matching OSS
/// `RadixKey.page_aligned()` semantics for `is_bigram=True`. A naive
/// raw-token trim (`token_ids[: len // page_size * page_size]`)
/// applied in Python before this wrapper would compound with Rust's
/// bigram trim and silently produce the wrong bigram count.
///
/// TODO(Jialin): [Refactor][Major] Consolidate the bigram path back
/// onto the existing `RustPageRadixCacheWrapper` (i.e., `K = Vec<i64>`)
/// and retire this wrapper + the `Vec<(i64, i64)>` `ChildKeyType`
/// impl. The bigram tree
/// `[(t0,t1), (t1,t2), (t2,t3), ...]` branches at the same points as
/// the raw-token tree because consecutive bigrams overlap by one
/// token — the branching structure is essentially identical, just
/// shifted by one index position. In principle the bigram cache can
/// be implemented as the standard `Vec<i64>` cache over the raw
/// token sequence with off-by-one indexing managed at the wrapper
/// boundary, eliminating the second `ChildKeyType` impl + this whole
/// wrapper class and any per-impl overhead on the standard `Vec<i64>`
/// path. Done naively the consolidation runs into subtle issues
/// around value-tensor sizing (N vs N−1 entries), edge label
/// semantics, hash-collision behavior between same-tail-token
/// bigrams, and page-size unit conversion — none individually hard,
/// but each requires careful design to preserve OSS bigram-mode
/// cache-hit semantics. Filed as follow-up; the explicit
/// `Vec<(i64, i64)>` instantiation lands today as the conservative
/// option that ships correct EAGLE semantics without those
/// design subtleties.
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
    #[pyo3(signature = (device, page_size, init_node_capacity, sliding_window_size = None, mamba_cache_chunk_size = None, enable_hicache = false, hicache_write_back = false))]
    fn new(
        device: &str,
        page_size: usize,
        init_node_capacity: usize,
        sliding_window_size: Option<usize>,
        mamba_cache_chunk_size: Option<usize>,
        enable_hicache: bool,
        hicache_write_back: bool,
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
            enable_hicache,
            hicache_write_back,
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

    /// In atom (= bigram-pair) units. For an N-token sequence the cache
    /// reports up to N−1 atoms.
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

    /// Walk the host-only (device-evicted, host-backed) chain from `node_idx`
    /// up to the first device-present node. Also lock the chain to avoid eviction.
    fn prepare_load_back(&mut self, node_idx: usize) -> PyResult<RustPrepareLoadBackResult> {
        Ok(RustPrepareLoadBackResult::from_prepare_load_back_result(
            self.inner.prepare_load_back(node_idx)?,
        ))
    }

    #[pyo3(signature = (key, value, extra_key = None, prev_prefix_len = 0, swa_evicted_seqlen = 0, mamba_value = None, chunked = false))]
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
        chunked: bool,
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
                chunked,
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

    /// Best-effort to evict at least `num_tokens` host FULL values.
    fn evict_host(&mut self, py: Python<'_>, num_tokens: usize) -> RustEvictResult {
        let r = self.inner.evict_host(num_tokens);
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

    /// Populate backup host values onto the given nodes.
    fn set_host_full_values(
        &mut self,
        node_indices: Vec<usize>,
        host_values: Vec<PyTensor>,
    ) -> PyResult<()> {
        let values: Vec<tch::Tensor> = host_values.into_iter().map(|v| v.0).collect();
        Ok(self.inner.set_host_full_values(node_indices, values)?)
    }

    /// Populate backup host values onto the given nodes and lock device values
    /// (to avoid async copy corruption).
    fn set_host_full_values_and_lock_device(
        &mut self,
        node_indices: Vec<usize>,
        host_values: Vec<PyTensor>,
    ) -> PyResult<()> {
        let values: Vec<tch::Tensor> = host_values.into_iter().map(|v| v.0).collect();
        self.inner
            .set_host_full_values(node_indices.clone(), values)?;
        for idx in node_indices {
            let _ = self.inner.inc_lock_ref(idx);
        }
        Ok(())
    }

    /// Commit the loadback results back to the radix tree. A `None`
    /// `device_values` indicates loadback failure.
    #[pyo3(signature = (chain, ancestor_node_idx, device_values=None))]
    fn postprocess_load_back(
        &mut self,
        chain: Vec<usize>,
        ancestor_node_idx: usize,
        device_values: Option<PyTensor>,
    ) -> PyResult<()> {
        Ok(self.inner.postprocess_load_back(
            chain,
            ancestor_node_idx,
            device_values.map(|v| v.0),
        )?)
    }
}
