//! PyO3 wrappers for the Rust radix cache.

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::py_interop::PyTensor;

use crate::component_type::{ComponentType, NUM_COMPONENT_TYPES};
use crate::components::{EvictRequest, EvictResult};
use crate::deferred_action::DeferredAction;
use crate::radix_cache::{BigramRadixCache, InsertResult, MatchResult, PageRadixCache};
use crate::utils::parse_device;

/// Per-component eviction outcome at the PyO3 boundary.
#[pyclass]
pub struct RustEvictResult {
    freed: Py<PyList>,
    evicted: [usize; NUM_COMPONENT_TYPES],
    deferred_actions: Py<PyList>,
}

fn deferred_action_to_py(py: Python<'_>, action: DeferredAction) -> PyObject {
    // Lead with the owning ComponentType so the Python consumer routes by it.
    let ct = action.component_type().into_py(py);
    match action {
        DeferredAction::FullFree { full_to_free } => {
            (ct, "FullFree", PyTensor(full_to_free).into_py(py)).into_py(py)
        }
        DeferredAction::SwaRecover {
            node_idx,
            old_full_to_free,
            new_full_value,
        } => (
            ct,
            "SwaRecover",
            node_idx,
            PyTensor(old_full_to_free).into_py(py),
            PyTensor(new_full_value).into_py(py),
        )
            .into_py(py),
        DeferredAction::SwaStamp {
            node_idx,
            full_value,
        } => (ct, "SwaStamp", node_idx, PyTensor(full_value).into_py(py)).into_py(py),
    }
}

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
    /// Per-component freed tensors.
    #[getter]
    fn freed<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        self.freed.bind(py).clone()
    }

    /// Per-component freed token-count total.
    #[getter]
    fn evicted(&self) -> [usize; NUM_COMPONENT_TYPES] {
        self.evicted
    }

    #[getter]
    fn deferred_actions<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        self.deferred_actions.bind(py).clone()
    }
}

/// Insert result at the PyO3 boundary.
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
    // Empty pyarray may use an unaligned dummy address; handle separately.
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

/// Define a PyO3 wrapper `$Wrapper` around radix cache `$Inner`: a `#[pyclass]`
/// holding `inner: $Inner` plus a `#[pymethods]` block with the delegating
/// accessors common to every cache, followed by the per-cache `$custom` methods
/// (constructor, `match_prefix`, `insert`).
macro_rules! define_radix_cache_wrapper {
    ($(#[$meta:meta])* $Wrapper:ident, $Inner:ty, { $($custom:tt)* }) => {
        $(#[$meta])*
        #[pyclass]
        pub struct $Wrapper {
            inner: $Inner,
        }

        #[pymethods]
        impl $Wrapper {
            fn reset(&mut self) {
                self.inner.reset();
            }

            fn page_size(&self) -> usize {
                self.inner.page_size()
            }

            fn active_tree_node_count(&self) -> usize {
                self.inner.active_tree_node_count()
            }

            fn component_evictable_size(&self, ct: ComponentType) -> usize {
                self.inner.component_evictable_size(ct)
            }

            fn component_protected_size(&self, ct: ComponentType) -> usize {
                self.inner.component_protected_size(ct)
            }

            fn component_total_size(&self, ct: ComponentType) -> usize {
                self.inner.component_total_size(ct)
            }

            fn total_size(&self) -> (usize, usize) {
                self.inner.total_size()
            }

            /// Lock `node_idx` and ancestors, protecting the prefix from eviction.
            /// Returns `(delta, swa_uuid_for_lock)`; pass the uuid back to `dec_lock_ref`.
            fn inc_lock_ref(&mut self, node_idx: usize) -> (i64, Option<u64>) {
                let r = self.inner.inc_lock_ref(node_idx);
                (r.delta, r.swa_uuid_for_lock)
            }

            /// Unlock `node_idx`; pair with `inc_lock_ref` and pass back its uuid.
            #[pyo3(signature = (node_idx, swa_uuid_for_lock = None))]
            fn dec_lock_ref(&mut self, node_idx: usize, swa_uuid_for_lock: Option<u64>) -> i64 {
                self.inner.dec_lock_ref(node_idx, swa_uuid_for_lock)
            }

            /// Best-effort cascade evict of up to `num_tokens[ct]` tokens per component.
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

            $($custom)*
        }
    };
}

define_radix_cache_wrapper!(RustPageRadixCacheWrapper, PageRadixCache, {
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

    /// Insert `key`/`value`; `prefix_len` counts the already-cached prefix.
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
});

/// Build EAGLE overlap bigram pairs `(t[i], t[i+1])` from raw tokens.
fn build_bigram_pairs(raw: &[i64]) -> Vec<(i64, i64)> {
    raw.windows(2).map(|w| (w[0], w[1])).collect()
}

define_radix_cache_wrapper!(
    /// Python wrapper for the EAGLE bigram radix cache (children keyed by overlap
    /// bigram pairs `(t[i], t[i+1])`). Sizes report in atom (= pair) units.
    ///
    /// Callers MUST pass raw token sequences untrimmed; the cache page-aligns and
    /// trims value N -> N-1 internally.
    RustBigramRadixCacheWrapper,
    BigramRadixCache,
    {
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
                // Trim value N -> N-1 (one slot per bigram).
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
    }
);
