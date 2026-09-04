//! Python bindings: the `mem_cache` extension module and its TreeCore adapter.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyAssertionError, PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use tch::{Device, Kind, Tensor};

use crate::components::{ComponentType, FULL, MAMBA, SWA};
use crate::node::ChildKeyType;
use crate::node::{KeyNamespaceRef, NodeId, TreeCoreRuntimeError};
use crate::unified_tree_core::KvCacheEvent;
use crate::unified_tree_core::{
    BufferBackupSnapshot, BufferBackupState, CacheAction, CacheInitParams, CacheTransferPhase,
    DecLockRefParams, EvictLayer, EvictionStepResult, InsertParams, InsertResult, InsertStepResult,
    MatchPrefixParams, MatchResult, PoolHitPolicy, PoolName, PoolTransfer, PoolTransferResult, Req,
    UnifiedTreeCore,
};

/// Parse a torch-style device string (e.g. "cpu", "cuda", "cuda:1"); a bare
/// "cuda" means index 0, so callers must resolve the index themselves.
fn parse_device(device: &str) -> PyResult<Device> {
    let device = device.to_lowercase();
    if device == "cpu" {
        return Ok(Device::Cpu);
    }
    if device == "cuda" {
        return Ok(Device::Cuda(0));
    }
    if let Some(index) = device.strip_prefix("cuda:")
        && let Ok(index) = index.parse::<usize>()
    {
        return Ok(Device::Cuda(index));
    }
    Err(PyValueError::new_err(format!(
        "unsupported device string: {device}"
    )))
}

/// Map a Python ComponentType value onto the Rust enum.
fn parse_component_type(component_type: u8) -> PyResult<ComponentType> {
    match component_type {
        0 => Ok(FULL),
        1 => Ok(SWA),
        2 => Ok(MAMBA),
        other => Err(PyValueError::new_err(format!(
            "unknown component type: {other}"
        ))),
    }
}

/// Map the Python EvictLayer IntFlag value onto the Rust enum.
fn parse_evict_layer(target: u8) -> PyResult<EvictLayer> {
    match target {
        1 => Ok(EvictLayer::Device),
        2 => Ok(EvictLayer::Host),
        3 => Ok(EvictLayer::All),
        other => Err(PyValueError::new_err(format!(
            "unknown eviction layer: {other}"
        ))),
    }
}

/// Convert an expected tree-core contract failure without unwinding through PyO3.
fn tree_core_runtime_error(error: TreeCoreRuntimeError) -> PyErr {
    match error {
        TreeCoreRuntimeError::NodeNotAllocated { node_id } => PyKeyError::new_err(node_id),
        error => PyRuntimeError::new_err(error.to_string()),
    }
}

fn tree_core_assertion_error(error: TreeCoreRuntimeError) -> PyErr {
    match error {
        TreeCoreRuntimeError::NodeNotAllocated { node_id } => PyKeyError::new_err(node_id),
        error => PyAssertionError::new_err(error.to_string()),
    }
}

/// Map the Rust enum back onto the Python ComponentType value.
fn component_type_to_u8(component_type: ComponentType) -> u8 {
    component_type as u8
}

/// Newtype bridging `tch::Tensor` and Python `torch.Tensor` over raw THPVariable
/// pointers (inlined from pyo3-tch, MIT/Apache-2.0, by Laurent Mazare).
pub struct PyTensor(pub Tensor);

impl<'py> FromPyObject<'py> for PyTensor {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let ptr = ob.as_ptr() as *mut tch::python::CPyObject;
        match unsafe { Tensor::pyobject_unpack(ptr) } {
            Ok(Some(tensor)) => Ok(PyTensor(tensor)),
            Ok(None) => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "expected a torch.Tensor, got {}",
                ob.get_type()
            ))),
            Err(err) => Err(PyValueError::new_err(format!("{err:?}"))),
        }
    }
}

impl ToPyObject for PyTensor {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        PyTensor(self.0.shallow_clone()).into_py(py)
    }
}

impl IntoPy<PyObject> for PyTensor {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let ptr = self
            .0
            .pyobject_wrap()
            .expect("failed to wrap a tensor as torch.Tensor");
        unsafe { PyObject::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject) }
    }
}

/// Convert a tensor into a Python-held torch.Tensor reference.
fn tensor_to_py(py: Python<'_>, tensor: Tensor) -> PyResult<Py<PyAny>> {
    let ptr = tensor
        .pyobject_wrap()
        .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
    Ok(unsafe { Py::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject) })
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

/// The tagged-tuple tag of a cache action.
fn cache_action_tag(action: &CacheAction) -> &'static str {
    match action {
        CacheAction::FreeDeviceKV(_) => "free_device_kv",
        CacheAction::FreeDeviceKVFullOnly(_) => "free_device_kv_full_only",
        CacheAction::BackupKV(_) => "backup_kv",
        CacheAction::ReplaceWriteThroughOnNodeSplit { .. } => "replace_write_through_on_node_split",
        CacheAction::MambaEvictExcessPathStates { .. } => "mamba_evict_excess_path_states",
        CacheAction::FreeComponentDeviceSlot { .. } => "free_component_device_slot",
        CacheAction::FreeComponentHostSlot { .. } => "free_component_host_slot",
        CacheAction::RebuildFullToSwaMapping { .. } => "rebuild_full_to_swa_mapping",
        CacheAction::RecoverSwaWithLockedFull { .. } => "recover_swa_with_locked_full",
        CacheAction::SwaRebuild { .. } => "swa_rebuild",
    }
}

/// Convert a cache action into its Python tagged tuple.
fn cache_action_to_py(py: Python<'_>, action: CacheAction) -> PyResult<Py<PyAny>> {
    let tag = cache_action_tag(&action);
    match action {
        CacheAction::FreeDeviceKV(tensors) => {
            let tensors = PyList::new_bound(py, tensors.into_iter().map(PyTensor));
            Ok((tag, tensors).into_py(py))
        }
        CacheAction::FreeDeviceKVFullOnly(tensors) => {
            let tensors = PyList::new_bound(py, tensors.into_iter().map(PyTensor));
            Ok((tag, tensors).into_py(py))
        }
        CacheAction::BackupKV(backup) => Ok((tag, backup.node_ids).into_py(py)),
        CacheAction::ReplaceWriteThroughOnNodeSplit {
            ack_id,
            old_node_id,
            new_node_id,
            new_child_node_id,
        } => Ok((tag, ack_id, old_node_id, new_node_id, new_child_node_id).into_py(py)),
        CacheAction::MambaEvictExcessPathStates { tail_node_id } => {
            Ok((tag, tail_node_id).into_py(py))
        }
        CacheAction::FreeComponentDeviceSlot {
            component_type,
            indices,
        } => {
            let indices = PyList::new_bound(py, indices.into_iter().map(PyTensor));
            Ok((tag, component_type_to_u8(component_type), indices).into_py(py))
        }
        CacheAction::FreeComponentHostSlot {
            component_type,
            host_indices,
        } => {
            let host_indices = PyList::new_bound(py, host_indices.into_iter().map(PyTensor));
            Ok((tag, component_type_to_u8(component_type), host_indices).into_py(py))
        }
        CacheAction::RebuildFullToSwaMapping {
            full_indices,
            swa_indices,
        } => {
            let full_indices = PyList::new_bound(py, full_indices.into_iter().map(PyTensor));
            let swa_indices = PyList::new_bound(py, swa_indices.into_iter().map(PyTensor));
            Ok((tag, full_indices, swa_indices).into_py(py))
        }
        CacheAction::RecoverSwaWithLockedFull {
            node_id,
            kept_full,
            incoming_full,
        } => Ok((tag, node_id, PyTensor(kept_full), PyTensor(incoming_full)).into_py(py)),
        CacheAction::SwaRebuild {
            node_id,
            source_value,
        } => Ok((tag, node_id, PyTensor(source_value)).into_py(py)),
    }
}

/// Convert cache actions into a Python list of tagged tuples.
fn cache_actions_to_py(py: Python<'_>, actions: Vec<CacheAction>) -> PyResult<Py<PyList>> {
    let list = PyList::empty_bound(py);
    for action in actions {
        list.append(cache_action_to_py(py, action)?)?;
    }
    Ok(list.unbind())
}

/// The python PoolName string for a pool.
fn pool_name_str(name: PoolName) -> &'static str {
    match name {
        PoolName::Kv => "kv",
        PoolName::Mamba => "mamba",
        PoolName::Swa => "swa",
        PoolName::Indexer => "indexer",
        PoolName::DeepseekV4C4 => "deepseek_v4_c4",
        PoolName::DeepseekV4C4Indexer => "deepseek_v4_c4_indexer",
        PoolName::DeepseekV4C4IndexerScale => "deepseek_v4_c4_indexer_scale",
        PoolName::DeepseekV4C128 => "deepseek_v4_c128",
        PoolName::DeepseekV4C4State => "deepseek_v4_c4_state",
        PoolName::DeepseekV4C4IndexerState => "deepseek_v4_c4_indexer_state",
        PoolName::DeepseekV4C128State => "deepseek_v4_c128_state",
        PoolName::Draft => "draft",
        PoolName::DraftIndexer => "draft_indexer",
        PoolName::DraftSwa => "draft_swa",
    }
}

/// Map a python PoolName string onto the Rust enum.
fn parse_pool_name(name: &str) -> PyResult<PoolName> {
    match name {
        "kv" => Ok(PoolName::Kv),
        "mamba" => Ok(PoolName::Mamba),
        "swa" => Ok(PoolName::Swa),
        "indexer" => Ok(PoolName::Indexer),
        "deepseek_v4_c4" => Ok(PoolName::DeepseekV4C4),
        "deepseek_v4_c4_indexer" => Ok(PoolName::DeepseekV4C4Indexer),
        "deepseek_v4_c4_indexer_scale" => Ok(PoolName::DeepseekV4C4IndexerScale),
        "deepseek_v4_c128" => Ok(PoolName::DeepseekV4C128),
        "deepseek_v4_c4_state" => Ok(PoolName::DeepseekV4C4State),
        "deepseek_v4_c4_indexer_state" => Ok(PoolName::DeepseekV4C4IndexerState),
        "deepseek_v4_c128_state" => Ok(PoolName::DeepseekV4C128State),
        "draft" => Ok(PoolName::Draft),
        "draft_indexer" => Ok(PoolName::DraftIndexer),
        "draft_swa" => Ok(PoolName::DraftSwa),
        other => Err(PyValueError::new_err(format!("unknown pool name: {other}"))),
    }
}

/// A pool transfer's boundary form:
/// (name, host_indices, device_indices, nodes_to_load, keys, hit_policy).
type TransferArgs = (
    String,
    Option<PyTensor>,
    Option<PyTensor>,
    Option<Vec<NodeId>>,
    Option<Vec<String>>,
    String,
);

/// Strongly typed, attribute-based input view of a Python MatchResult.
/// Cache actions are intentionally omitted: component finalizers only update
/// match metadata, while the test adapter preserves the original actions.
#[cfg(feature = "inspection")]
#[derive(FromPyObject)]
struct InspectionMatchResultInput {
    #[pyo3(attribute)]
    device_indices: PyTensor,
    #[pyo3(attribute)]
    last_device_node: NodeId,
    #[pyo3(attribute)]
    last_host_node: NodeId,
    #[pyo3(attribute)]
    best_match_node: NodeId,
    #[pyo3(attribute)]
    host_hit_length: usize,
    #[pyo3(attribute)]
    swa_host_hit_length: usize,
    #[pyo3(attribute)]
    mamba_host_hit_length: usize,
    #[pyo3(attribute)]
    mamba_branching_seqlen: Option<usize>,
    #[pyo3(attribute)]
    full_kv_hit_length: usize,
}

/// Map a python CacheTransferPhase value onto the Rust enum.
fn parse_transfer_phase(phase: &str) -> PyResult<CacheTransferPhase> {
    match phase {
        "backup_host" => Ok(CacheTransferPhase::BackupHost),
        "load_back" => Ok(CacheTransferPhase::LoadBack),
        "backup_storage" => Ok(CacheTransferPhase::BackupStorage),
        "prefetch" => Ok(CacheTransferPhase::Prefetch),
        other => Err(PyValueError::new_err(format!(
            "unknown transfer phase: {other}"
        ))),
    }
}

/// Map a python PoolHitPolicy value onto the Rust enum.
fn parse_hit_policy(hit_policy: &str) -> PyResult<PoolHitPolicy> {
    match hit_policy {
        "all_pages" => Ok(PoolHitPolicy::AllPages),
        "trailing_pages" => Ok(PoolHitPolicy::TrailingPages),
        other => Err(PyValueError::new_err(format!(
            "unknown hit policy: {other}"
        ))),
    }
}

/// Convert a pool transfer into its boundary tuple.
fn transfer_to_py(py: Python<'_>, transfer: PoolTransfer) -> PyResult<Py<PyAny>> {
    Ok((
        pool_name_str(transfer.name),
        transfer.host_indices.map(PyTensor),
        transfer.device_indices.map(PyTensor),
        transfer.nodes_to_load,
        transfer.keys,
        transfer.hit_policy.as_str(),
    )
        .into_py(py))
}

/// Build a pool transfer from its boundary tuple.
fn transfer_from_args(args: TransferArgs) -> PyResult<PoolTransfer> {
    let (name, host_indices, device_indices, nodes_to_load, keys, hit_policy) = args;
    Ok(PoolTransfer {
        name: parse_pool_name(&name)?,
        host_indices: host_indices.map(|t| t.0),
        device_indices: device_indices.map(|t| t.0),
        keys,
        hit_policy: parse_hit_policy(&hit_policy)?,
        nodes_to_load,
    })
}

/// Convert per-component transfers into a Python dict of boundary tuples.
fn comp_xfers_to_py(
    py: Python<'_>,
    comp_xfers: HashMap<ComponentType, Vec<PoolTransfer>>,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);
    for (ct, transfers) in comp_xfers {
        let list = PyList::empty_bound(py);
        for transfer in transfers {
            list.append(transfer_to_py(py, transfer)?)?;
        }
        dict.set_item(component_type_to_u8(ct), list)?;
    }
    Ok(dict.unbind())
}

/// Build per-component transfers from a Python dict of boundary tuples.
fn comp_xfers_from_args(
    comp_xfers: HashMap<u8, Vec<TransferArgs>>,
) -> PyResult<HashMap<ComponentType, Vec<PoolTransfer>>> {
    comp_xfers
        .into_iter()
        .map(|(ct, transfers)| {
            Ok((
                parse_component_type(ct)?,
                transfers
                    .into_iter()
                    .map(transfer_from_args)
                    .collect::<PyResult<Vec<_>>>()?,
            ))
        })
        .collect()
}

/// Convert per-component freed tensors into a Python dict keyed by component value.
fn frees_to_py(py: Python<'_>, frees: HashMap<ComponentType, Vec<Tensor>>) -> PyResult<Py<PyDict>> {
    let frees: HashMap<u8, Vec<PyTensor>> = frees
        .into_iter()
        .map(|(ct, tensors)| {
            (
                component_type_to_u8(ct),
                tensors.into_iter().map(PyTensor).collect(),
            )
        })
        .collect();
    let dict = PyDict::new_bound(py);
    for (ct, tensors) in frees {
        dict.set_item(ct, tensors)?;
    }
    Ok(dict.unbind())
}

/// Python-visible tree-core init params; converts into CacheInitParams.
#[pyclass(get_all, set_all)]
#[derive(Clone)]
pub struct TreeCoreInitParamsBinding {
    pub eviction_policy: String,
    pub page_size: usize,
    pub is_write_back: bool,
    pub enable_hicache: bool,
    pub write_through_threshold: i64,
    pub device: String,
    pub swa_sliding_window_size: Option<usize>,
    pub enable_kv_cache_events: bool,
    pub mamba_cache_chunk_size: Option<usize>,
    pub mamba_max_states_per_path: Option<usize>,
}

impl TreeCoreInitParamsBinding {
    /// Convert into the tree core's construction params.
    fn to_cache_init_params(&self) -> PyResult<CacheInitParams> {
        Ok(CacheInitParams {
            eviction_policy: self.eviction_policy.clone(),
            page_size: self.page_size,
            is_write_back: self.is_write_back,
            enable_hicache: self.enable_hicache,
            write_through_threshold: self.write_through_threshold,
            device: parse_device(&self.device)?,
            swa_sliding_window_size: self.swa_sliding_window_size,
            // Wired post-construction via set_has_swa_host_pool.
            has_swa_host_pool: false,
            enable_kv_cache_events: self.enable_kv_cache_events,
            mamba_cache_chunk_size: self.mamba_cache_chunk_size,
            mamba_max_states_per_path: self.mamba_max_states_per_path,
        })
    }
}

#[pymethods]
impl TreeCoreInitParamsBinding {
    #[new]
    #[pyo3(signature = (eviction_policy = "lru".to_string(), page_size = 1, is_write_back = false, enable_hicache = false, write_through_threshold = 256, device = "cpu".to_string(), swa_sliding_window_size = None, enable_kv_cache_events = false, mamba_cache_chunk_size = None, mamba_max_states_per_path = None))]
    fn new(
        eviction_policy: String,
        page_size: usize,
        is_write_back: bool,
        enable_hicache: bool,
        write_through_threshold: i64,
        device: String,
        swa_sliding_window_size: Option<usize>,
        enable_kv_cache_events: bool,
        mamba_cache_chunk_size: Option<usize>,
        mamba_max_states_per_path: Option<usize>,
    ) -> Self {
        TreeCoreInitParamsBinding {
            eviction_policy,
            page_size,
            is_write_back,
            enable_hicache,
            write_through_threshold,
            device,
            swa_sliding_window_size,
            enable_kv_cache_events,
            mamba_cache_chunk_size,
            mamba_max_states_per_path,
        }
    }
}

/// Python-visible match params; converts into MatchPrefixParams.
#[pyclass(get_all, set_all)]
#[derive(Clone)]
pub struct MatchParamsBinding {
    pub key: Vec<i64>,
    pub extra_key: Option<String>,
    pub cache_salt: Option<String>,
}

#[pymethods]
impl MatchParamsBinding {
    #[new]
    #[pyo3(signature = (key, extra_key = None, cache_salt = None))]
    fn new(
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        extra_key: Option<String>,
        cache_salt: Option<String>,
    ) -> PyResult<Self> {
        Ok(MatchParamsBinding {
            key: py_array_to_vec_i64(py, key)?,
            extra_key,
            cache_salt,
        })
    }
}

/// Python-visible insert params; converts into InsertParams. The value tensor
/// stays a Python-held reference until the insert call unwraps it.
#[pyclass(get_all, set_all)]
pub struct InsertParamsBinding {
    pub key: Vec<i64>,
    pub value: Py<PyAny>,
    pub extra_key: Option<String>,
    pub cache_salt: Option<String>,
    pub mamba_value: Option<Py<PyAny>>,
    pub prev_prefix_len: usize,
    pub swa_evicted_seqlen: usize,
    pub chunked: bool,
    pub priority: i64,
    pub track_adopted_ranges: bool,
}

#[pymethods]
impl InsertParamsBinding {
    #[new]
    #[pyo3(signature = (key, value, extra_key = None, cache_salt = None, prev_prefix_len = 0, swa_evicted_seqlen = 0, chunked = false, priority = 0, mamba_value = None, track_adopted_ranges = false))]
    fn new(
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: Py<PyAny>,
        extra_key: Option<String>,
        cache_salt: Option<String>,
        prev_prefix_len: usize,
        swa_evicted_seqlen: usize,
        chunked: bool,
        priority: i64,
        mamba_value: Option<Py<PyAny>>,
        track_adopted_ranges: bool,
    ) -> PyResult<Self> {
        Ok(InsertParamsBinding {
            key: py_array_to_vec_i64(py, key)?,
            value,
            extra_key,
            cache_salt,
            mamba_value,
            prev_prefix_len,
            swa_evicted_seqlen,
            chunked,
            priority,
            track_adopted_ranges,
        })
    }
}

/// Python-visible match result; tensors and actions are Python-held.
#[pyclass(get_all)]
pub struct MatchResultBinding {
    device_indices: Py<PyAny>,
    last_device_node_id: NodeId,
    last_host_node_id: NodeId,
    best_match_node_id: NodeId,
    host_hit_length: usize,
    swa_host_hit_length: usize,
    mamba_host_hit_length: usize,
    mamba_branching_seqlen: Option<usize>,
    full_kv_hit_length: usize,
    cache_actions: Py<PyList>,
}

impl MatchResultBinding {
    /// Move a core match result across the boundary.
    fn from_match_result(py: Python<'_>, result: MatchResult) -> PyResult<Self> {
        Ok(MatchResultBinding {
            device_indices: tensor_to_py(py, result.device_indices)?,
            last_device_node_id: result.last_device_node_id,
            last_host_node_id: result.last_host_node_id,
            best_match_node_id: result.best_match_node_id,
            host_hit_length: result.host_hit_length,
            swa_host_hit_length: result.swa_host_hit_length,
            mamba_host_hit_length: result.mamba_host_hit_length,
            mamba_branching_seqlen: result.mamba_branching_seqlen,
            full_kv_hit_length: result.full_kv_hit_length,
            cache_actions: cache_actions_to_py(py, result.cache_actions)?,
        })
    }
}

/// Python-visible insert result; actions are Python-held.
#[pyclass(get_all)]
pub struct InsertResultBinding {
    prefix_len: usize,
    total_len: usize,
    last_device_node: Option<NodeId>,
    inserted_host_node: Option<NodeId>,
    host_insert_dropped: bool,
    mamba_exist: bool,
    adopted_ranges: Option<HashMap<u8, Vec<(usize, usize)>>>,
    cache_actions: Py<PyList>,
}

/// One step of the resumable insert: the actions to apply at this barrier
/// and the final result once the walk completes.
#[pyclass(get_all)]
pub struct InsertStepResultBinding {
    actions: Py<PyList>,
    result: Option<Py<InsertResultBinding>>,
}

impl InsertStepResultBinding {
    /// Move a core insert step across the boundary.
    fn from_insert_step(py: Python<'_>, step: InsertStepResult) -> PyResult<Self> {
        let result = match step.result {
            Some(result) => Some(Py::new(
                py,
                InsertResultBinding::from_insert_result(py, result)?,
            )?),
            None => None,
        };
        Ok(InsertStepResultBinding {
            actions: cache_actions_to_py(py, step.actions)?,
            result,
        })
    }
}

impl InsertResultBinding {
    /// Move a core insert result across the boundary.
    fn from_insert_result(py: Python<'_>, result: InsertResult) -> PyResult<Self> {
        Ok(InsertResultBinding {
            prefix_len: result.prefix_len,
            total_len: result.total_len,
            last_device_node: result.last_device_node_id,
            inserted_host_node: result.inserted_host_node,
            host_insert_dropped: result.host_insert_dropped,
            mamba_exist: result.mamba_exist,
            adopted_ranges: result.adopted_ranges.map(|ranges| {
                ranges
                    .into_iter()
                    .map(|(component_type, ranges)| (component_type_to_u8(component_type), ranges))
                    .collect()
            }),
            cache_actions: cache_actions_to_py(py, result.cache_actions)?,
        })
    }
}

/// Python-visible dec-lock params; converts into DecLockRefParams.
#[pyclass(get_all, set_all)]
#[derive(Clone, Default)]
pub struct DecLockRefParamsBinding {
    pub swa_uuid_for_lock: Option<i64>,
    pub swa_uuid_for_host_lock: Option<i64>,
    pub skip_lock_node_ids: HashMap<u8, HashSet<NodeId>>,
}

#[pymethods]
impl DecLockRefParamsBinding {
    #[new]
    #[pyo3(signature = (swa_uuid_for_lock = None, swa_uuid_for_host_lock = None, skip_lock_node_ids = None))]
    fn new(
        swa_uuid_for_lock: Option<i64>,
        swa_uuid_for_host_lock: Option<i64>,
        skip_lock_node_ids: Option<HashMap<u8, HashSet<NodeId>>>,
    ) -> Self {
        DecLockRefParamsBinding {
            swa_uuid_for_lock,
            swa_uuid_for_host_lock,
            skip_lock_node_ids: skip_lock_node_ids.unwrap_or_default(),
        }
    }
}

impl DecLockRefParamsBinding {
    /// Convert into the tree core's dec-lock params.
    fn to_dec_lock_ref_params(&self) -> PyResult<DecLockRefParams> {
        Ok(DecLockRefParams {
            swa_uuid_for_lock: self.swa_uuid_for_lock,
            swa_uuid_for_host_lock: self.swa_uuid_for_host_lock,
            skip_lock_node_ids: self
                .skip_lock_node_ids
                .iter()
                .map(|(ct, node_ids)| {
                    Ok::<_, PyErr>((parse_component_type(*ct)?, node_ids.clone()))
                })
                .collect::<PyResult<_>>()?,
        })
    }
}

/// Python-visible inc_lock_ref result; hand skip_lock_node_ids back to the
/// matching dec_lock_ref.
#[pyclass(get_all)]
pub struct IncLockRefResultBinding {
    delta: Option<usize>,
    swa_uuid_for_lock: Option<i64>,
    swa_uuid_for_host_lock: Option<i64>,
    skip_lock_node_ids: HashMap<u8, HashSet<NodeId>>,
}

impl IncLockRefResultBinding {
    fn from_result(result: crate::unified_tree_core::IncLockRefResult) -> Self {
        Self {
            delta: result.delta,
            swa_uuid_for_lock: result.swa_uuid_for_lock,
            swa_uuid_for_host_lock: result.swa_uuid_for_host_lock,
            skip_lock_node_ids: result
                .skip_lock_node_ids
                .into_iter()
                .map(|(ct, node_ids)| (component_type_to_u8(ct), node_ids))
                .collect(),
        }
    }
}

/// Convert a Python component-keyed tracker into the core's counts.
fn tracker_from_py(tracker: HashMap<u8, usize>) -> PyResult<HashMap<ComponentType, usize>> {
    tracker
        .into_iter()
        .map(|(ct, freed)| Ok((parse_component_type(ct)?, freed)))
        .collect()
}

/// Convert the core's tracker back into Python component-keyed counts.
fn tracker_to_py(tracker: HashMap<ComponentType, usize>) -> HashMap<u8, usize> {
    tracker
        .into_iter()
        .map(|(ct, freed)| (component_type_to_u8(ct), freed))
        .collect()
}

/// Next-eviction-node step result: the node to evict, whether the walk made
/// progress, this step's per-component evicted counts, and this step's newly
/// freed tensors.
#[pyclass(get_all)]
pub struct EvictDeviceNextNodeResultBinding {
    node_id: Option<NodeId>,
    made_progress: bool,
    tracker: HashMap<u8, usize>,
    new_device_frees: Py<PyDict>,
    new_host_frees: Py<PyDict>,
}

/// Leaf-eviction step result: the backup action for an unbacked
/// write-back leaf (else None), this step's per-component evicted counts,
/// and this step's newly freed tensors.
#[pyclass(get_all)]
pub struct EvictDeviceLeafResultBinding {
    backup_kv: Option<Py<PyAny>>,
    tracker: HashMap<u8, usize>,
    new_device_frees: Py<PyDict>,
    new_host_frees: Py<PyDict>,
}

/// Drop-subtree result: whether the drop happened, this step's
/// per-component evicted counts, and the subtree's newly freed tensors.
#[pyclass(get_all)]
pub struct DropSubtreeResultBinding {
    dropped: bool,
    tracker: HashMap<u8, usize>,
    new_device_frees: Py<PyDict>,
    new_host_frees: Py<PyDict>,
}

/// Demote result: this step's per-component evicted counts and the
/// demoted node's newly freed tensors.
#[pyclass(get_all)]
pub struct DemoteResultBinding {
    tracker: HashMap<u8, usize>,
    new_device_frees: Py<PyDict>,
    new_host_frees: Py<PyDict>,
}

/// Python-visible device->storage backup spec.
#[pyclass(get_all)]
pub struct StorageBackupSpecBinding {
    host_value: Py<PyAny>,
    /// Raw native-endian int64 bytes; python rebuilds them via array("q").frombytes.
    token_ids: Py<PyBytes>,
    hash_value: Option<Vec<String>>,
    prefix_keys: Option<Vec<String>>,
    comp_xfers: Py<PyDict>,
}

#[pyclass(get_all)]
pub struct BufferBackupSnapshotBinding {
    node_id: NodeId,
    parent_node_id: NodeId,
    parent_is_root: bool,
    parent_last_hash: Option<String>,
    key_token_ids: Py<PyBytes>,
    extra_key: Option<String>,
    cache_salt: Option<String>,
    is_bigram: bool,
    hash_values: Vec<String>,
    prefix_keys: Option<Vec<String>>,
}

impl BufferBackupSnapshotBinding {
    fn from_snapshot(py: Python<'_>, snapshot: BufferBackupSnapshot) -> Self {
        let mut token_bytes = Vec::with_capacity(snapshot.token_ids.len() * 8);
        for token in snapshot.token_ids {
            token_bytes.extend_from_slice(&token.to_ne_bytes());
        }
        Self {
            node_id: snapshot.node_id,
            parent_node_id: snapshot.parent_node_id,
            parent_is_root: snapshot.parent_is_root,
            parent_last_hash: snapshot.parent_last_hash,
            key_token_ids: PyBytes::new_bound(py, &token_bytes).unbind(),
            extra_key: snapshot.extra_key,
            cache_salt: snapshot.cache_salt,
            is_bigram: snapshot.is_bigram,
            hash_values: snapshot.hash_values,
            prefix_keys: snapshot.prefix_keys,
        }
    }
}

#[pyclass(get_all)]
pub struct BufferBackupStateBinding {
    parent_node_id: NodeId,
    parent_is_root: bool,
    parent_last_hash: Option<String>,
}

impl From<BufferBackupState> for BufferBackupStateBinding {
    fn from(state: BufferBackupState) -> Self {
        Self {
            parent_node_id: state.parent_node_id,
            parent_is_root: state.parent_is_root,
            parent_last_hash: state.parent_last_hash,
        }
    }
}

/// Python-visible KV-canary walk rows: parallel int64 tensors of slots,
/// positions, and preceding slots.
#[pyclass(get_all)]
pub struct KvCanaryWalkResultBinding {
    slot_indices: Py<PyAny>,
    positions: Py<PyAny>,
    prev_slot_indices: Py<PyAny>,
}

/// Host-eviction drive result: this step's per-component evicted counts
/// and the drive's newly freed tensors.
#[pyclass(get_all)]
pub struct HostEvictionResultBinding {
    tracker: HashMap<u8, usize>,
    new_device_frees: Py<PyDict>,
    new_host_frees: Py<PyDict>,
}

impl HostEvictionResultBinding {
    fn from_eviction_step(py: Python<'_>, result: EvictionStepResult) -> PyResult<Self> {
        Ok(Self {
            tracker: tracker_to_py(result.tracker),
            new_device_frees: frees_to_py(py, result.device_frees)?,
            new_host_frees: frees_to_py(py, result.host_frees)?,
        })
    }
}

/// The generic UnifiedTreeCore adapter the per-key-type pyclasses delegate to;
/// the Mutex makes the Send-only core satisfy pyclass's Sync bound.
struct TreeCoreBinding<K: ChildKeyType> {
    core: Mutex<UnifiedTreeCore<K>>,
    /// The core's construction device, kept outside the Mutex for pre-lock validation.
    device: Device,
    /// The core's page size, kept outside the Mutex for pre-lock validation.
    page_size: usize,
}

// Send + Sync lets allow_threads release the GIL around core calls.
impl<K: ChildKeyType + Send + Sync> TreeCoreBinding<K> {
    /// Build a tree core for the given component types from the cache's
    /// init params.
    fn new(init_params: &TreeCoreInitParamsBinding, component_types: Vec<u8>) -> PyResult<Self> {
        let component_types = component_types
            .into_iter()
            .map(parse_component_type)
            .collect::<PyResult<Vec<_>>>()?;
        if component_types != [FULL]
            && component_types != [FULL, SWA]
            && component_types != [FULL, MAMBA]
            && component_types != [FULL, SWA, MAMBA]
        {
            return Err(PyValueError::new_err(
                "only the [Full], [Full, Swa], [Full, Mamba], and [Full, Swa, Mamba] component sets are supported",
            ));
        }
        if component_types.contains(&SWA) && init_params.swa_sliding_window_size.is_none() {
            return Err(PyValueError::new_err(
                "the Swa component requires swa_sliding_window_size",
            ));
        }
        if component_types.contains(&MAMBA) && init_params.mamba_cache_chunk_size.is_none() {
            return Err(PyValueError::new_err(
                "the Mamba component requires mamba_cache_chunk_size",
            ));
        }
        if init_params.page_size == 0 {
            return Err(PyValueError::new_err("page_size must be at least 1"));
        }
        let eviction_policy = init_params.eviction_policy.to_lowercase();
        if !matches!(
            eviction_policy.as_str(),
            "lru" | "lfu" | "fifo" | "mru" | "filo" | "priority" | "slru"
        ) {
            return Err(PyValueError::new_err(format!(
                "Unknown eviction policy: {eviction_policy}. Supported policies: \
                 'lru', 'lfu', 'fifo', 'mru', 'filo', 'priority', 'slru'."
            )));
        }
        let params = init_params.to_cache_init_params()?;
        let device = params.device;
        let page_size = params.page_size;
        Ok(TreeCoreBinding {
            core: Mutex::new(UnifiedTreeCore::new(params, component_types)),
            device,
            page_size,
        })
    }

    /// Lock the core for one adapter call. A panic can leave a mutation half-applied,
    /// so a poisoned core is never reused.
    fn core(&self) -> std::sync::MutexGuard<'_, UnifiedTreeCore<K>> {
        self.core.lock().unwrap_or_else(|_| {
            panic!("Rust TreeCore mutex poisoned; refusing to reuse state after an earlier panic")
        })
    }

    /// Reject an insert value whose dtype, device, or length cannot cover the key.
    fn validate_insert_value(&self, value: &Tensor, key_atom_len: usize) -> PyResult<()> {
        if value.kind() != Kind::Int64 {
            return Err(PyValueError::new_err(format!(
                "insert value must be an int64 tensor, got {:?}",
                value.kind()
            )));
        }
        if value.device() != self.device {
            return Err(PyValueError::new_err(format!(
                "insert value device {:?} does not match the tree core device {:?}",
                value.device(),
                self.device
            )));
        }
        let aligned_key_len = key_atom_len / self.page_size * self.page_size;
        let value_len = value.size().first().copied().unwrap_or(0);
        if value_len < aligned_key_len as i64 {
            return Err(PyValueError::new_err(format!(
                "insert value length {value_len} is shorter than the aligned key length {aligned_key_len}"
            )));
        }
        Ok(())
    }

    /// Drop the entire tree and reinitialize empty state.
    fn reset(&self, py: Python<'_>) {
        py.allow_threads(|| self.core().reset());
    }

    /// Match a key against the tree.
    fn match_prefix(
        &self,
        py: Python<'_>,
        params: &MatchParamsBinding,
    ) -> PyResult<MatchResultBinding> {
        let key = K::key_from(Cow::Borrowed(&params.key));
        let key = key.as_ref();
        let params = MatchPrefixParams {
            key,
            namespace: KeyNamespaceRef::new(
                params.extra_key.as_deref(),
                params.cache_salt.as_deref(),
            ),
        };
        let result = py.allow_threads(|| self.core().match_prefix(&params));
        MatchResultBinding::from_match_result(py, result)
    }

    /// The empty match result anchored at the root.
    fn empty_match_result(&self, py: Python<'_>) -> PyResult<MatchResultBinding> {
        let result = py.allow_threads(|| self.core().empty_match_result());
        MatchResultBinding::from_match_result(py, result)
    }

    /// Insert device values into the tree per the provided key.
    fn insert(
        &self,
        py: Python<'_>,
        params: &InsertParamsBinding,
    ) -> PyResult<InsertResultBinding> {
        let key = K::key_from(Cow::Borrowed(&params.key));
        let key = key.as_ref();
        let value: PyTensor = params.value.bind(py).extract()?;
        // The value covers key atoms (bigram: raw len - 1), so validate the converted key.
        self.validate_insert_value(&value.0, key.atom_len())?;
        let mamba_value = match &params.mamba_value {
            Some(mamba_value) => Some(mamba_value.bind(py).extract::<PyTensor>()?.0),
            None => None,
        };
        let params = InsertParams {
            key,
            namespace: KeyNamespaceRef::new(
                params.extra_key.as_deref(),
                params.cache_salt.as_deref(),
            ),
            value: value.0,
            mamba_value,
            prev_prefix_len: params.prev_prefix_len,
            swa_evicted_seqlen: params.swa_evicted_seqlen,
            chunked: params.chunked,
            priority: params.priority,
            track_adopted_ranges: params.track_adopted_ranges,
        };
        let result = py
            .allow_threads(move || self.core().try_insert(&params))
            .map_err(tree_core_runtime_error)?;
        InsertResultBinding::from_insert_result(py, result)
    }

    /// Start the resumable insert, running to its first barrier or completion.
    fn begin_insert(
        &self,
        py: Python<'_>,
        params: &InsertParamsBinding,
    ) -> PyResult<InsertStepResultBinding> {
        let key = K::key_from(Cow::Borrowed(&params.key));
        let key = key.as_ref();
        let value: PyTensor = params.value.bind(py).extract()?;
        // The value covers key atoms (bigram: raw len - 1), so validate the converted key.
        self.validate_insert_value(&value.0, key.atom_len())?;
        let mamba_value = match &params.mamba_value {
            Some(mamba_value) => Some(mamba_value.bind(py).extract::<PyTensor>()?.0),
            None => None,
        };
        let params = InsertParams {
            key,
            namespace: KeyNamespaceRef::new(
                params.extra_key.as_deref(),
                params.cache_salt.as_deref(),
            ),
            value: value.0,
            mamba_value,
            prev_prefix_len: params.prev_prefix_len,
            swa_evicted_seqlen: params.swa_evicted_seqlen,
            chunked: params.chunked,
            priority: params.priority,
            track_adopted_ranges: params.track_adopted_ranges,
        };
        let step = py
            .allow_threads(move || self.core().try_begin_insert(&params))
            .map_err(tree_core_runtime_error)?;
        InsertStepResultBinding::from_insert_step(py, step)
    }

    /// Continue the suspended insert after its step actions were applied.
    fn resume_insert(&self, py: Python<'_>) -> PyResult<InsertStepResultBinding> {
        let step = py
            .allow_threads(|| self.core().try_resume_insert())
            .map_err(tree_core_runtime_error)?;
        InsertStepResultBinding::from_insert_step(py, step)
    }

    /// Whether an insert walk is suspended at a barrier.
    fn has_ongoing_insert(&self, py: Python<'_>) -> bool {
        py.allow_threads(|| self.core().has_ongoing_insert())
    }

    /// Finish the insert (idempotent); returns still-pending actions to drain.
    fn end_insert(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let actions = py.allow_threads(|| self.core().end_insert());
        cache_actions_to_py(py, actions)
    }

    /// Bump the reference count on a node's component locks.
    fn inc_lock_ref(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        skip_lock_components: Option<Vec<u8>>,
    ) -> PyResult<IncLockRefResultBinding> {
        let skip_lock_components = skip_lock_components
            .unwrap_or_default()
            .into_iter()
            .map(parse_component_type)
            .collect::<PyResult<Vec<_>>>()?;
        let result = py.allow_threads(|| {
            self.core()
                .inc_lock_ref_with_skip(node_id, &skip_lock_components)
        });
        Ok(IncLockRefResultBinding::from_result(result))
    }

    /// Decrease the reference count on a node's component locks.
    fn dec_lock_ref(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        params: Option<&DecLockRefParamsBinding>,
        skip_swa: bool,
    ) -> PyResult<()> {
        let params = params.map(|p| p.to_dec_lock_ref_params()).transpose()?;
        py.allow_threads(|| self.core().dec_lock_ref(node_id, params.as_ref(), skip_swa));
        Ok(())
    }

    /// Early-release the SWA portion of a request's tree lock; returns this
    /// release's per-component (device_frees, host_frees).
    fn dec_swa_lock_only(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        swa_uuid_for_lock: Option<i64>,
        skip_lock_node_ids: Option<HashMap<u8, HashSet<NodeId>>>,
    ) -> PyResult<(Py<PyDict>, Py<PyDict>)> {
        let skip_lock_node_ids = skip_lock_node_ids
            .unwrap_or_default()
            .into_iter()
            .map(|(ct, node_ids)| Ok((parse_component_type(ct)?, node_ids)))
            .collect::<PyResult<HashMap<_, _>>>()?;
        let (device_frees, host_frees) = py.allow_threads(|| {
            let mut device_frees = HashMap::new();
            let mut host_frees = HashMap::new();
            self.core().dec_swa_lock_only_with_skip(
                node_id,
                swa_uuid_for_lock,
                Some(&skip_lock_node_ids),
                &mut device_frees,
                &mut host_frees,
            );
            (device_frees, host_frees)
        });
        Ok((frees_to_py(py, device_frees)?, frees_to_py(py, host_frees)?))
    }

    /// Store a component's device value on a node (the SWA rebuild write-back).
    fn set_component_device_value(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        component_type: u8,
        value: PyTensor,
    ) -> PyResult<()> {
        let component_type = parse_component_type(component_type)?;
        let value = value.0;
        if value.kind() != Kind::Int64 {
            return Err(PyValueError::new_err(format!(
                "component device value must be an int64 tensor, got {:?}",
                value.kind()
            )));
        }
        if value.device() != self.device {
            return Err(PyValueError::new_err(format!(
                "component device value device {:?} does not match the tree core device {:?}",
                value.device(),
                self.device
            )));
        }
        py.allow_threads(|| {
            self.core()
                .set_component_device_value(node_id, component_type, value)
        });
        Ok(())
    }

    /// A component's device value on a node, if set.
    fn get_component_device_value(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        component_type: u8,
    ) -> PyResult<Option<PyTensor>> {
        let component_type = parse_component_type(component_type)?;
        let value = py.allow_threads(|| {
            self.core()
                .get_component_device_value(node_id, component_type)
                .map(|tensor| tensor.shallow_clone())
        });
        Ok(value.map(PyTensor))
    }

    // TODO(jialino): batch a full no-backup eviction round in Rust (one crossing
    // instead of 2N+2); step-wise stays for the write-back interleave.
    /// Begin a component's device-eviction walk for up to request_cnt tokens.
    fn evict_device_start(
        &self,
        py: Python<'_>,
        component_type: u8,
        request_cnt: usize,
    ) -> PyResult<()> {
        let ct = parse_component_type(component_type)?;
        py.allow_threads(|| self.core().evict_device_start(ct, request_cnt));
        Ok(())
    }

    /// Advance one component eviction step. A missing node with
    /// `made_progress` set means an internal tombstone completed the step;
    /// otherwise a missing node means the walk is exhausted.
    fn evict_device_next_node(
        &self,
        py: Python<'_>,
        component_type: u8,
        tracker: HashMap<u8, usize>,
    ) -> PyResult<EvictDeviceNextNodeResultBinding> {
        let ct = parse_component_type(component_type)?;
        let baseline = tracker_from_py(tracker)?;
        let (node_id, result) =
            py.allow_threads(move || self.core().evict_device_next_node(ct, &baseline));
        let made_progress = node_id.is_some() || !result.tracker.is_empty();
        Ok(EvictDeviceNextNodeResultBinding {
            node_id,
            made_progress,
            tracker: tracker_to_py(result.tracker),
            new_device_frees: frees_to_py(py, result.device_frees)?,
            new_host_frees: frees_to_py(py, result.host_frees)?,
        })
    }

    /// Evict one device leaf; an unbacked write-back leaf returns its backup
    /// action for the caller to execute before demoting.
    fn evict_device_leaf(
        &self,
        py: Python<'_>,
        node_id: NodeId,
    ) -> PyResult<EvictDeviceLeafResultBinding> {
        let (backup, result) = py.allow_threads(move || {
            let mut core = self.core();
            let is_write_back = core.is_write_back;
            core.evict_device_leaf(node_id, is_write_back)
        });
        Ok(EvictDeviceLeafResultBinding {
            backup_kv: backup
                .map(|backup| cache_action_to_py(py, CacheAction::BackupKV(backup)))
                .transpose()?,
            tracker: tracker_to_py(result.tracker),
            new_device_frees: frees_to_py(py, result.device_frees)?,
            new_host_frees: frees_to_py(py, result.host_frees)?,
        })
    }

    /// Finish a component's device-eviction walk.
    fn evict_device_end(&self, py: Python<'_>, component_type: u8) -> PyResult<()> {
        let ct = parse_component_type(component_type)?;
        py.allow_threads(|| self.core().evict_device_end(ct));
        Ok(())
    }

    /// Verify tree-structure, leaf-set, LRU, size, and ongoing-op invariants;
    /// ongoing_* args are (id, node_id) pairs.
    fn sanity_check(
        &self,
        py: Python<'_>,
        ongoing_write_through: Vec<(i64, NodeId)>,
        ongoing_load_back: Vec<(i64, NodeId)>,
    ) -> PyResult<()> {
        py.allow_threads(|| {
            self.core()
                .try_sanity_check(&ongoing_write_through, &ongoing_load_back)
        })
        .map_err(PyAssertionError::new_err)
    }

    /// Concatenated FULL device values from from_node up to (exclusive) until_node.
    fn collect_full_device_indices(
        &self,
        py: Python<'_>,
        from_node_id: NodeId,
        until_node_id: NodeId,
    ) -> PyTensor {
        PyTensor(py.allow_threads(|| {
            self.core()
                .collect_full_device_indices(from_node_id, until_node_id)
        }))
    }

    /// Every FULL device value in the tree, concatenated.
    fn all_values_flatten(&self, py: Python<'_>) -> PyTensor {
        PyTensor(py.allow_threads(|| self.core().all_values_flatten()))
    }

    /// Every Mamba device value in the tree, concatenated.
    fn all_mamba_values_flatten(&self, py: Python<'_>) -> PyTensor {
        PyTensor(py.allow_threads(|| self.core().all_mamba_values_flatten()))
    }

    /// Flatten every FULL device slot into (slot, position, prev-slot) rows for the KV-canary sweep.
    fn walk_for_kv_canary(
        &self,
        py: Python<'_>,
        unlocked_only: bool,
        swa_resident_only: bool,
    ) -> PyResult<KvCanaryWalkResultBinding> {
        let result = py.allow_threads(|| {
            self.core()
                .walk_for_kv_canary(unlocked_only, swa_resident_only)
        });
        Ok(KvCanaryWalkResultBinding {
            slot_indices: tensor_to_py(py, Tensor::from_slice(&result.slot_indices))?,
            positions: tensor_to_py(py, Tensor::from_slice(&result.positions))?,
            prev_slot_indices: tensor_to_py(py, Tensor::from_slice(&result.prev_slot_indices))?,
        })
    }

    /// Evictable token count of the FULL (base) component.
    fn evictable_size(&self, py: Python<'_>) -> usize {
        py.allow_threads(|| self.core().evictable_size())
    }

    /// Protected (locked) token count of the FULL (base) component.
    fn protected_size(&self, py: Python<'_>) -> usize {
        py.allow_threads(|| self.core().protected_size())
    }

    /// FULL component evictable token count.
    fn full_evictable_size(&self, py: Python<'_>) -> usize {
        py.allow_threads(|| self.core().full_evictable_size())
    }

    /// FULL component protected token count.
    fn full_protected_size(&self, py: Python<'_>) -> usize {
        py.allow_threads(|| self.core().full_protected_size())
    }

    /// Evictable token count for one component (0 if the component is absent).
    fn component_evictable_size(&self, py: Python<'_>, component_type: u8) -> PyResult<usize> {
        let ct = parse_component_type(component_type)?;
        Ok(py.allow_threads(|| self.core().component_evictable_size(ct)))
    }

    /// Protected token count for one component (0 if the component is absent).
    fn component_protected_size(&self, py: Python<'_>, component_type: u8) -> PyResult<usize> {
        let ct = parse_component_type(component_type)?;
        Ok(py.allow_threads(|| self.core().component_protected_size(ct)))
    }

    /// (full_tokens, aux_tokens) summed across the whole tree.
    fn total_size(&self, py: Python<'_>) -> (usize, usize) {
        py.allow_threads(|| self.core().total_size())
    }

    /// Whether the node's FULL device value has been evicted.
    fn is_full_device_evicted(&self, py: Python<'_>, node_id: NodeId) -> bool {
        py.allow_threads(|| self.core().is_full_device_evicted(node_id))
    }

    /// Mark the host tier (HiCache) as wired.
    fn set_hicache_enabled(&self, py: Python<'_>) {
        py.allow_threads(|| self.core().set_hicache_enabled());
    }

    /// Whether the host tier (HiCache) is wired.
    fn enable_hicache(&self, py: Python<'_>) -> bool {
        py.allow_threads(|| self.core().enable_hicache)
    }

    /// Mark the SWA host pool as wired (HiCache).
    fn set_has_swa_host_pool(&self, py: Python<'_>) {
        py.allow_threads(|| self.core().set_has_swa_host_pool());
    }

    /// Whether the SWA host pool is wired.
    fn has_swa_host_pool(&self, py: Python<'_>) -> bool {
        py.allow_threads(|| self.core().has_swa_host_pool)
    }

    /// Insert a host-side (backuped) tree path descending from the given node.
    fn insert_host(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        extra_key: Option<String>,
        key: &Bound<'_, PyAny>,
        host_value: PyTensor,
        hash_value: Vec<String>,
        cache_salt: Option<String>,
    ) -> PyResult<InsertResultBinding> {
        let key = K::key_from(Cow::Owned(py_array_to_vec_i64(py, key)?)).into_owned();
        let host_value = host_value.0;
        if host_value.kind() != Kind::Int64 {
            return Err(PyValueError::new_err(format!(
                "insert_host host_value must be an int64 tensor, got {:?}",
                host_value.kind()
            )));
        }
        let result = py
            .allow_threads(move || {
                self.core().try_insert_host_in_namespace(
                    node_id,
                    KeyNamespaceRef::new(extra_key.as_deref(), cache_salt.as_deref()),
                    key,
                    host_value,
                    hash_value,
                )
            })
            .map_err(tree_core_runtime_error)?;
        InsertResultBinding::from_insert_result(py, result)
    }

    /// Gather a node's device value plus per-component BACKUP_HOST transfers.
    fn build_backup_spec(
        &self,
        py: Python<'_>,
        node_id: NodeId,
    ) -> PyResult<(PyTensor, Py<PyDict>)> {
        let (device_value, comp_xfers) =
            py.allow_threads(|| self.core().build_backup_spec(node_id));
        Ok((PyTensor(device_value), comp_xfers_to_py(py, comp_xfers)?))
    }

    /// Gather a node's device->storage backup spec; None if the node is not backuped.
    fn build_storage_backup_spec(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        pass_prefix_keys: bool,
    ) -> PyResult<Option<StorageBackupSpecBinding>> {
        let spec = py.allow_threads(|| {
            self.core()
                .build_storage_backup_spec(node_id, pass_prefix_keys)
        });
        let Some(spec) = spec else {
            return Ok(None);
        };
        let mut token_bytes = Vec::with_capacity(spec.token_ids.len() * 8);
        for token in &spec.token_ids {
            token_bytes.extend_from_slice(&token.to_ne_bytes());
        }
        Ok(Some(StorageBackupSpecBinding {
            host_value: tensor_to_py(py, spec.host_value)?,
            token_ids: PyBytes::new_bound(py, &token_bytes).unbind(),
            hash_value: spec.hash_value,
            prefix_keys: spec.prefix_keys,
            comp_xfers: comp_xfers_to_py(py, spec.comp_xfers)?,
        }))
    }

    /// Route a build_hicache_transfers call to the component for the given type.
    #[allow(clippy::too_many_arguments)]
    fn build_hicache_transfers(
        &self,
        py: Python<'_>,
        component_type: u8,
        node_id: NodeId,
        phase: &str,
        host_indices: Option<PyTensor>,
        token_ids: Option<Vec<i64>>,
        prefetch_tokens: usize,
        last_hash: Option<String>,
    ) -> PyResult<Option<Vec<Py<PyAny>>>> {
        let component_type = parse_component_type(component_type)?;
        let phase = parse_transfer_phase(phase)?;
        let host_indices = host_indices.map(|t| t.0);
        let transfers = py
            .allow_threads(|| {
                self.core().try_build_hicache_transfers(
                    component_type,
                    node_id,
                    phase,
                    host_indices,
                    token_ids.as_deref(),
                    prefetch_tokens,
                    last_hash.as_deref(),
                )
            })
            .map_err(tree_core_assertion_error)?;
        transfers
            .map(|transfers| {
                transfers
                    .into_iter()
                    .map(|transfer| transfer_to_py(py, transfer))
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()
    }

    /// The anchor node's namespace; None for root-like anchors.
    fn prefetch_anchor_info(
        &self,
        py: Python<'_>,
        node_id: NodeId,
    ) -> PyResult<(Option<String>, Option<String>)> {
        py.allow_threads(|| self.core().try_prefetch_anchor_info(node_id))
            .map_err(tree_core_runtime_error)
    }

    /// Whether the node's Full KV is present on host.
    fn node_backuped(&self, py: Python<'_>, node_id: NodeId) -> PyResult<bool> {
        py.allow_threads(|| self.core().try_node_backuped(node_id))
            .map_err(tree_core_runtime_error)
    }

    /// Whether the node is a (default or named) root.
    fn is_root(&self, py: Python<'_>, node_id: NodeId) -> PyResult<bool> {
        py.allow_threads(|| self.core().try_is_root(node_id))
            .map_err(tree_core_runtime_error)
    }

    /// The node's last page hash, or None when it was never hashed.
    fn get_last_hash_value(&self, py: Python<'_>, node_id: NodeId) -> PyResult<Option<String>> {
        py.allow_threads(|| self.core().try_get_last_hash_value(node_id))
            .map_err(tree_core_runtime_error)
    }

    /// The hash chain of the node's ancestors, in root-to-parent order.
    fn get_prefix_hash_values(&self, py: Python<'_>, node_id: NodeId) -> PyResult<Vec<String>> {
        py.allow_threads(|| self.core().try_get_prefix_hash_values(node_id))
            .map_err(tree_core_runtime_error)
    }

    fn get_hash_values(&self, py: Python<'_>, node_id: NodeId) -> PyResult<Vec<String>> {
        py.allow_threads(|| self.core().try_get_hash_values(node_id))
            .map_err(tree_core_runtime_error)
    }

    fn snapshot_buffer_backup(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        pass_prefix_keys: bool,
    ) -> Option<BufferBackupSnapshotBinding> {
        py.allow_threads(|| {
            self.core()
                .snapshot_buffer_backup(node_id, pass_prefix_keys)
        })
        .map(|snapshot| BufferBackupSnapshotBinding::from_snapshot(py, snapshot))
    }

    fn validate_buffer_backup(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        expected_key_length: usize,
    ) -> Option<BufferBackupStateBinding> {
        py.allow_threads(|| {
            self.core()
                .validate_buffer_backup(node_id, expected_key_length)
        })
        .map(BufferBackupStateBinding::from)
    }

    /// Hash every node built while storage was disabled.
    fn backfill_missing_hash_values(&self, py: Python<'_>) -> usize {
        py.allow_threads(|| self.core().backfill_missing_hash_values())
    }

    fn root_node_handle(&self, py: Python<'_>, extra_key: Option<String>) -> NodeId {
        py.allow_threads(|| self.core().root_node_handle(extra_key.as_deref()))
    }

    fn dfs_weight_order(&self, py: Python<'_>, node_ids: Vec<NodeId>) -> PyResult<Vec<usize>> {
        py.allow_threads(|| self.core().try_dfs_weight_order(&node_ids))
            .map_err(tree_core_runtime_error)
    }

    /// Commit each component's HiCache transfers; returns the new cache actions.
    fn commit_hicache_transfers(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        phase: &str,
        comp_xfers: HashMap<u8, Vec<TransferArgs>>,
        insert_result: Option<(usize, Option<NodeId>, bool)>,
        pool_storage_result: Option<(usize, HashMap<String, usize>)>,
    ) -> PyResult<(Py<PyList>, Option<bool>)> {
        let phase = parse_transfer_phase(phase)?;
        let comp_xfers = comp_xfers_from_args(comp_xfers)?;
        let insert_result =
            insert_result.map(
                |(total_len, inserted_host_node, mamba_exist)| InsertResult {
                    total_len,
                    inserted_host_node,
                    mamba_exist,
                    ..InsertResult::default()
                },
            );
        let pool_storage_result = pool_storage_result
            .map(|(kv_hit_pages, extra_pool_hit_pages)| {
                Ok::<_, PyErr>(PoolTransferResult {
                    kv_hit_pages,
                    extra_pool_hit_pages: extra_pool_hit_pages
                        .into_iter()
                        .map(|(name, pages)| Ok((parse_pool_name(&name)?, pages)))
                        .collect::<PyResult<HashMap<_, _>>>()?,
                })
            })
            .transpose()?;
        let (cache_actions, mamba_exist) = py.allow_threads(move || {
            let mut cache_actions = Vec::new();
            let mut insert_result = insert_result;
            self.core().commit_hicache_transfers(
                node_id,
                phase,
                comp_xfers,
                &mut cache_actions,
                insert_result.as_mut(),
                pool_storage_result.as_ref(),
            );
            (
                cache_actions,
                insert_result.map(|result| result.mamba_exist),
            )
        });
        Ok((cache_actions_to_py(py, cache_actions)?, mamba_exist))
    }

    /// Commit a successful backup to the node.
    fn commit_backup(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        host_indices: PyTensor,
        comp_xfers: HashMap<u8, Vec<TransferArgs>>,
    ) -> PyResult<()> {
        let comp_xfers = comp_xfers_from_args(comp_xfers)?;
        let host_indices = host_indices.0;
        py.allow_threads(move || self.core().commit_backup(node_id, host_indices, comp_xfers));
        Ok(())
    }

    /// Build the H->D load-back KV transfer plus per-component aux transfers.
    fn build_load_back_spec(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        mamba_pool_idx: Option<PyTensor>,
    ) -> PyResult<(Py<PyAny>, Py<PyDict>)> {
        let req = Req {
            mamba_pool_idx: mamba_pool_idx.map(|t| t.0),
        };
        let (kv_xfer, comp_xfers) = py
            .allow_threads(move || self.core().try_build_load_back_spec(node_id, Some(&req)))
            .map_err(tree_core_assertion_error)?;
        Ok((
            transfer_to_py(py, kv_xfer)?,
            comp_xfers_to_py(py, comp_xfers)?,
        ))
    }

    /// Commit a successful H->D load-back onto the node; returns its actions.
    fn commit_load_back(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        device_indices: PyTensor,
        kv_xfer: TransferArgs,
        comp_xfers: HashMap<u8, Vec<TransferArgs>>,
    ) -> PyResult<Py<PyList>> {
        let kv_xfer = transfer_from_args(kv_xfer)?;
        let comp_xfers = comp_xfers_from_args(comp_xfers)?;
        let device_indices = device_indices.0;
        let actions = py.allow_threads(move || {
            self.core()
                .commit_load_back(node_id, device_indices, kv_xfer, comp_xfers)
        });
        cache_actions_to_py(py, actions)
    }

    /// Release a node's device KV once its host copy exists.
    fn demote(&self, py: Python<'_>, node_id: NodeId) -> PyResult<DemoteResultBinding> {
        let result = py
            .allow_threads(move || self.core().try_demote(node_id))
            .map_err(tree_core_assertion_error)?;
        Ok(DemoteResultBinding {
            tracker: tracker_to_py(result.tracker),
            new_device_frees: frees_to_py(py, result.device_frees)?,
            new_host_frees: frees_to_py(py, result.host_frees)?,
        })
    }

    /// Evict up to num_tokens of one component's host resources.
    fn drive_host_eviction(
        &self,
        py: Python<'_>,
        component_type: u8,
        num_tokens: usize,
    ) -> PyResult<HostEvictionResultBinding> {
        let ct = parse_component_type(component_type)?;
        let result = py.allow_threads(move || self.core().drive_host_eviction(ct, num_tokens));
        Ok(HostEvictionResultBinding {
            tracker: tracker_to_py(result.tracker),
            new_device_frees: frees_to_py(py, result.device_frees)?,
            new_host_frees: frees_to_py(py, result.host_frees)?,
        })
    }

    /// Evict shallow Mamba device checkpoints beyond the per-path cap on the
    /// tail's root path; returns the step's freed tensors.
    fn evict_excess_path_states(
        &self,
        py: Python<'_>,
        tail_node_id: NodeId,
    ) -> PyResult<HostEvictionResultBinding> {
        let result = py.allow_threads(move || self.core().evict_excess_path_states(tail_node_id));
        Ok(HostEvictionResultBinding {
            tracker: tracker_to_py(result.tracker),
            new_device_frees: frees_to_py(py, result.device_frees)?,
            new_host_frees: frees_to_py(py, result.host_frees)?,
        })
    }

    /// Bump the reference count on a node's host-side component locks.
    fn inc_host_lock_ref(&self, py: Python<'_>, node_id: NodeId) -> IncLockRefResultBinding {
        let result = py.allow_threads(|| self.core().inc_host_lock_ref(node_id));
        IncLockRefResultBinding {
            delta: result.delta,
            swa_uuid_for_lock: result.swa_uuid_for_lock,
            swa_uuid_for_host_lock: result.swa_uuid_for_host_lock,
            skip_lock_node_ids: result
                .skip_lock_node_ids
                .into_iter()
                .map(|(ct, node_ids)| (component_type_to_u8(ct), node_ids))
                .collect(),
        }
    }

    /// Decrease the reference count on a node's host-side component locks.
    fn dec_host_lock_ref(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        params: Option<&DecLockRefParamsBinding>,
    ) -> PyResult<()> {
        let params = params.map(|p| p.to_dec_lock_ref_params()).transpose()?;
        py.allow_threads(|| self.core().dec_host_lock_ref(node_id, params.as_ref()));
        Ok(())
    }

    /// Set the write-back (vs write-through) policy; decided at HiCache init.
    fn set_is_write_back(&self, py: Python<'_>, is_write_back: bool) {
        py.allow_threads(|| self.core().is_write_back = is_write_back);
    }

    /// The current write-back (vs write-through) policy.
    fn is_write_back(&self, py: Python<'_>) -> bool {
        py.allow_threads(|| self.core().is_write_back)
    }

    /// Set the write-through backup hit threshold; decided at HiCache init.
    fn set_write_through_threshold(&self, py: Python<'_>, threshold: i64) {
        py.allow_threads(|| self.core().write_through_threshold = threshold);
    }

    /// The current write-through backup hit threshold.
    fn write_through_threshold(&self, py: Python<'_>) -> i64 {
        py.allow_threads(|| self.core().write_through_threshold)
    }

    /// Mark the storage tier (L3) wired; storage attaches after tree construction.
    fn set_enable_storage(&self, py: Python<'_>, value: bool) {
        py.allow_threads(|| self.core().set_enable_storage(value));
    }

    /// Whether the storage tier (L3) is wired.
    fn enable_storage(&self, py: Python<'_>) -> bool {
        py.allow_threads(|| self.core().enable_storage)
    }

    /// Queue the all-cleared placement event.
    fn record_all_cleared_event(&self, py: Python<'_>) {
        py.allow_threads(|| self.core().record_all_cleared_event());
    }

    /// Drain the queued placement events as tagged tuples.
    fn take_events(&self, py: Python<'_>) -> PyResult<Py<PyList>>
    where
        Vec<K::Atom>: IntoPy<Py<PyAny>>,
    {
        let events = py.allow_threads(|| self.core().take_events());
        let list = PyList::empty_bound(py);
        for event in events {
            match event {
                KvCacheEvent::BlockStored {
                    block_hashes,
                    parent_block_hash,
                    token_ids,
                    block_size,
                    medium,
                    cache_salt,
                } => {
                    let item: Py<PyAny> = (
                        "block_stored",
                        block_hashes,
                        parent_block_hash,
                        token_ids,
                        block_size,
                        medium.as_str(),
                        cache_salt.map(|salt| salt.to_string()),
                    )
                        .into_py(py);
                    list.append(item)?;
                }
                KvCacheEvent::BlockRemoved {
                    block_hashes,
                    medium,
                } => {
                    let item: Py<PyAny> =
                        ("block_removed", block_hashes, medium.as_str()).into_py(py);
                    list.append(item)?;
                }
                KvCacheEvent::AllBlocksCleared => {
                    let item: Py<PyAny> = ("all_blocks_cleared",).into_py(py);
                    list.append(item)?;
                }
            }
        }
        Ok(list.unbind())
    }

    /// Drop the subtree rooted at an unbacked D-leaf; not dropped when a lock
    /// blocks it.
    fn drop_subtree_no_host(
        &self,
        py: Python<'_>,
        node_id: NodeId,
    ) -> PyResult<DropSubtreeResultBinding> {
        let (dropped, result) = py.allow_threads(move || self.core().drop_subtree_no_host(node_id));
        Ok(DropSubtreeResultBinding {
            dropped,
            tracker: tracker_to_py(result.tracker),
            new_device_frees: frees_to_py(py, result.device_frees)?,
            new_host_frees: frees_to_py(py, result.host_frees)?,
        })
    }

    /// Mark a node as having an in-flight write-through backup.
    fn mark_write_through_pending(&self, py: Python<'_>, node_id: NodeId) {
        py.allow_threads(|| self.core().mark_write_through_pending(node_id));
    }

    /// Clear the write-through-pending mark on the acked nodes.
    fn finish_write_through(&self, py: Python<'_>, node_ids: Vec<NodeId>, ack_id: NodeId) {
        py.allow_threads(|| self.core().finish_write_through(node_ids, ack_id));
    }

    /// Clear the in-flight H->D marks on the anchor's root path at ack time.
    fn finish_load_back(&self, py: Python<'_>, anchor_node_id: NodeId) {
        py.allow_threads(|| self.core().finish_load_back(anchor_node_id));
    }

    /// Order-sensitive digest of reclaimed coexisting host values.
    fn write_back_coexist_reclaim_digest(&self, py: Python<'_>) -> i64 {
        py.allow_threads(|| self.core().write_back_coexist_reclaim_digest)
    }

    /// Whether the component's data is device-evicted but host-backed.
    fn component_has_host_value_only(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        component_type: u8,
    ) -> PyResult<bool> {
        let ct = parse_component_type(component_type)?;
        Ok(py.allow_threads(|| self.core().component_has_host_value_only(node_id, ct)))
    }
}

#[cfg(feature = "inspection")]
impl<K: ChildKeyType + Send + Sync> TreeCoreBinding<K> {
    fn inspect_contains_node(&self, py: Python<'_>, node_id: NodeId) -> bool {
        py.allow_threads(|| self.core().inspect_contains_node(node_id))
    }

    fn inspect_get_parent_node_id(&self, py: Python<'_>, node_id: NodeId) -> Option<NodeId> {
        py.allow_threads(|| self.core().inspect_get_parent_node_id(node_id))
    }

    fn inspect_get_child_node_ids(&self, py: Python<'_>, node_id: NodeId) -> Vec<NodeId> {
        py.allow_threads(|| self.core().inspect_get_child_node_ids(node_id))
    }

    fn inspect_get_node_key_length(&self, py: Python<'_>, node_id: NodeId) -> usize {
        py.allow_threads(|| self.core().inspect_get_node_key_length(node_id))
    }

    fn inspect_get_node_token_ids(&self, py: Python<'_>, node_id: NodeId) -> Vec<i64> {
        py.allow_threads(|| self.core().inspect_get_node_token_ids(node_id))
    }

    fn inspect_is_node_key_bigram(&self, py: Python<'_>, node_id: NodeId) -> bool {
        py.allow_threads(|| self.core().inspect_is_node_key_bigram(node_id))
    }

    fn inspect_get_component_host_value(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        component_type: u8,
    ) -> PyResult<Option<PyTensor>> {
        let component_type = parse_component_type(component_type)?;
        Ok(py
            .allow_threads(|| {
                self.core()
                    .inspect_get_component_host_value(node_id, component_type)
            })
            .map(PyTensor))
    }

    fn inspect_get_component_device_lock_ref(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        component_type: u8,
    ) -> PyResult<u32> {
        let component_type = parse_component_type(component_type)?;
        Ok(py.allow_threads(|| {
            self.core()
                .inspect_get_component_device_lock_ref(node_id, component_type)
        }))
    }

    fn inspect_get_node_hit_count(&self, py: Python<'_>, node_id: NodeId) -> i64 {
        py.allow_threads(|| self.core().inspect_get_node_hit_count(node_id))
    }

    fn inspect_get_write_through_pending_id(
        &self,
        py: Python<'_>,
        node_id: NodeId,
    ) -> Option<usize> {
        py.allow_threads(|| self.core().inspect_get_write_through_pending_id(node_id))
    }

    fn inspect_is_node_in_device_lru(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        component_type: u8,
    ) -> PyResult<bool> {
        let component_type = parse_component_type(component_type)?;
        Ok(py.allow_threads(|| {
            self.core()
                .inspect_is_node_in_device_lru(node_id, component_type)
        }))
    }

    fn inspect_is_node_in_host_lru(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        component_type: u8,
    ) -> PyResult<bool> {
        let component_type = parse_component_type(component_type)?;
        Ok(py.allow_threads(|| {
            self.core()
                .inspect_is_node_in_host_lru(node_id, component_type)
        }))
    }

    fn inspect_get_component_device_lru_node_ids(
        &self,
        py: Python<'_>,
        component_type: u8,
    ) -> PyResult<Vec<NodeId>> {
        let component_type = parse_component_type(component_type)?;
        Ok(py.allow_threads(|| {
            self.core()
                .inspect_get_component_device_lru_node_ids(component_type)
        }))
    }

    fn inspect_is_device_evictable_leaf(&self, py: Python<'_>, node_id: NodeId) -> bool {
        py.allow_threads(|| self.core().inspect_is_device_evictable_leaf(node_id))
    }

    fn inspect_is_host_evictable_leaf(&self, py: Python<'_>, node_id: NodeId) -> bool {
        py.allow_threads(|| self.core().inspect_is_host_evictable_leaf(node_id))
    }

    fn inspect_is_device_leaf(&self, py: Python<'_>, node_id: NodeId) -> bool {
        py.allow_threads(|| self.core().inspect_is_device_leaf(node_id))
    }

    fn inspect_get_all_node_ids(&self, py: Python<'_>) -> Vec<NodeId> {
        py.allow_threads(|| self.core().inspect_get_all_node_ids())
    }

    fn inspect_component_protected_size(
        &self,
        py: Python<'_>,
        component_type: u8,
    ) -> PyResult<usize> {
        let component_type = parse_component_type(component_type)?;
        Ok(py.allow_threads(|| self.core().inspect_component_protected_size(component_type)))
    }

    fn inspect_set_node_hash_values(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        hash_values: Option<Vec<String>>,
    ) {
        py.allow_threads(move || {
            self.core()
                .inspect_set_node_hash_values(node_id, hash_values)
        });
    }

    fn inspect_set_component_device_value_raw(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        component_type: u8,
        value: Option<PyTensor>,
    ) -> PyResult<()> {
        let component_type = parse_component_type(component_type)?;
        let value = value.map(|value| value.0);
        py.allow_threads(move || {
            self.core()
                .inspect_set_component_device_value_raw(node_id, component_type, value)
        });
        Ok(())
    }

    fn inspect_set_component_host_value_raw(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        component_type: u8,
        value: Option<PyTensor>,
    ) -> PyResult<()> {
        let component_type = parse_component_type(component_type)?;
        let value = value.map(|value| value.0);
        py.allow_threads(move || {
            self.core()
                .inspect_set_component_host_value_raw(node_id, component_type, value)
        });
        Ok(())
    }

    fn inspect_set_component_device_lock_ref(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        component_type: u8,
        lock_ref: u32,
    ) -> PyResult<()> {
        let component_type = parse_component_type(component_type)?;
        py.allow_threads(|| {
            self.core()
                .inspect_set_component_device_lock_ref(node_id, component_type, lock_ref)
        });
        Ok(())
    }

    fn inspect_remove_node_from_device_lru(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        component_type: u8,
    ) -> PyResult<()> {
        let component_type = parse_component_type(component_type)?;
        py.allow_threads(|| {
            self.core()
                .inspect_remove_node_from_device_lru(node_id, component_type)
        });
        Ok(())
    }

    fn inspect_insert_node_into_host_lru(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        component_type: u8,
    ) -> PyResult<()> {
        let component_type = parse_component_type(component_type)?;
        py.allow_threads(|| {
            self.core()
                .inspect_insert_node_into_host_lru(node_id, component_type)
        });
        Ok(())
    }

    fn inspect_set_component_evictable_size(
        &self,
        py: Python<'_>,
        component_type: u8,
        value: usize,
    ) -> PyResult<()> {
        let component_type = parse_component_type(component_type)?;
        py.allow_threads(|| {
            self.core()
                .inspect_set_component_evictable_size(component_type, value)
        });
        Ok(())
    }

    fn inspect_set_component_protected_size(
        &self,
        py: Python<'_>,
        component_type: u8,
        value: usize,
    ) -> PyResult<()> {
        let component_type = parse_component_type(component_type)?;
        py.allow_threads(|| {
            self.core()
                .inspect_set_component_protected_size(component_type, value)
        });
        Ok(())
    }

    fn inspect_update_duplicate_tracking(&self, py: Python<'_>, node_id: NodeId) {
        py.allow_threads(|| self.core().inspect_update_duplicate_tracking(node_id));
    }

    fn inspect_advance_insert_walk_once(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.core().inspect_advance_insert_walk_once())
            .map_err(PyRuntimeError::new_err)
    }

    fn inspect_evict_component(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        component_type: u8,
        target: u8,
    ) -> PyResult<HostEvictionResultBinding> {
        let component_type = parse_component_type(component_type)?;
        let target = parse_evict_layer(target)?;
        let result = py.allow_threads(|| {
            self.core()
                .inspect_evict_component(node_id, component_type, target)
        });
        HostEvictionResultBinding::from_eviction_step(py, result)
    }

    fn inspect_validate_cascade_evict(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        component_type: u8,
        target: u8,
    ) -> PyResult<()> {
        let component_type = parse_component_type(component_type)?;
        let target = parse_evict_layer(target)?;
        py.allow_threads(|| {
            self.core()
                .inspect_validate_cascade_evict(node_id, component_type, target)
        })
        .map_err(PyAssertionError::new_err)
    }

    fn inspect_cleanup_tombstone_ancestors(
        &self,
        py: Python<'_>,
        node_id: NodeId,
    ) -> PyResult<HostEvictionResultBinding> {
        let result = py.allow_threads(|| self.core().inspect_cleanup_tombstone_ancestors(node_id));
        HostEvictionResultBinding::from_eviction_step(py, result)
    }

    #[allow(clippy::too_many_arguments)]
    fn inspect_finalize_component_match_result(
        &self,
        py: Python<'_>,
        component_type: u8,
        result: InspectionMatchResultInput,
        key: &Bound<'_, PyAny>,
        extra_key: Option<String>,
        cache_salt: Option<String>,
        value_chunks: Vec<PyTensor>,
        best_value_len: usize,
    ) -> PyResult<MatchResultBinding> {
        let component_type = parse_component_type(component_type)?;
        let key = K::key_from(Cow::Owned(py_array_to_vec_i64(py, key)?)).into_owned();
        let InspectionMatchResultInput {
            device_indices,
            last_device_node: last_device_node_id,
            last_host_node: last_host_node_id,
            best_match_node: best_match_node_id,
            host_hit_length,
            swa_host_hit_length,
            mamba_host_hit_length,
            mamba_branching_seqlen,
            full_kv_hit_length,
        } = result;
        let result = MatchResult {
            device_indices: device_indices.0,
            last_device_node_id,
            last_host_node_id,
            best_match_node_id,
            host_hit_length,
            swa_host_hit_length,
            mamba_host_hit_length,
            mamba_branching_seqlen,
            full_kv_hit_length,
            cache_actions: Vec::new(),
        };
        let value_chunks = value_chunks
            .into_iter()
            .map(|value| value.0)
            .collect::<Vec<_>>();
        let result = py.allow_threads(move || {
            let params = MatchPrefixParams {
                key: &key,
                namespace: KeyNamespaceRef::new(extra_key.as_deref(), cache_salt.as_deref()),
            };
            self.core().inspect_finalize_component_match_result(
                component_type,
                result,
                &params,
                &value_chunks,
                best_value_len,
            )
        });
        MatchResultBinding::from_match_result(py, result)
    }

    fn inspect_build_backup_node_ids(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        write_back: bool,
    ) -> Vec<NodeId> {
        py.allow_threads(|| {
            self.core()
                .inspect_build_backup_node_ids(node_id, write_back)
        })
    }
}

impl<K: ChildKeyType + Send + Sync> TreeCoreBinding<K> {
    /// Print the tree structure for debugging.
    fn pretty_print(&self, py: Python<'_>) {
        py.allow_threads(|| self.core().pretty_print());
    }
}

// The delegate surface is identical for every key type; the macro stamps the
// pyclass + pymethods pair per concrete key.
macro_rules! tree_core_binding {
    ($(#[$doc:meta])* $name:ident, $key:ty) => {
        $(#[$doc])*
        #[pyclass]
        pub struct $name {
            inner: TreeCoreBinding<$key>,
        }

        #[pymethods]
        impl $name {
            /// Build a tree core for the given component types from the cache's
            /// init params.
            #[new]
            fn new(init_params: &TreeCoreInitParamsBinding, component_types: Vec<u8>) -> PyResult<Self> {
                Ok($name {
                    inner: TreeCoreBinding::new(init_params, component_types)?,
                })
            }

            /// Drop the entire tree and reinitialize empty state.
            fn reset(&self, py: Python<'_>) {
                self.inner.reset(py)
            }

            /// Match a key against the tree.
            fn match_prefix(
                &self,
                py: Python<'_>,
                params: &MatchParamsBinding,
            ) -> PyResult<MatchResultBinding> {
                self.inner.match_prefix(py, params)
            }

            /// The empty match result anchored at the root.
            fn empty_match_result(&self, py: Python<'_>) -> PyResult<MatchResultBinding> {
                self.inner.empty_match_result(py)
            }

            /// Insert device values into the tree per the provided key.
            fn insert(
                &self,
                py: Python<'_>,
                params: &InsertParamsBinding,
            ) -> PyResult<InsertResultBinding> {
                self.inner.insert(py, params)
            }

            /// Start the resumable insert, running to its first barrier or completion.
            fn begin_insert(
                &self,
                py: Python<'_>,
                params: &InsertParamsBinding,
            ) -> PyResult<InsertStepResultBinding> {
                self.inner.begin_insert(py, params)
            }

            /// Continue the suspended insert after its step actions were applied.
            fn resume_insert(&self, py: Python<'_>) -> PyResult<InsertStepResultBinding> {
                self.inner.resume_insert(py)
            }

            /// Whether an insert walk is suspended at a barrier.
            fn has_ongoing_insert(&self, py: Python<'_>) -> bool {
                self.inner.has_ongoing_insert(py)
            }

            /// Finish the insert (idempotent); returns still-pending actions to drain.
            fn end_insert(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
                self.inner.end_insert(py)
            }

            /// Bump the reference count on a node's component locks.
            #[pyo3(signature = (node_id, skip_lock_components = None))]
            fn inc_lock_ref(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                skip_lock_components: Option<Vec<u8>>,
            ) -> PyResult<IncLockRefResultBinding> {
                self.inner.inc_lock_ref(py, node_id, skip_lock_components)
            }

            /// Decrease the reference count on a node's component locks.
            #[pyo3(signature = (node_id, params = None, skip_swa = false))]
            fn dec_lock_ref(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                params: Option<&DecLockRefParamsBinding>,
                skip_swa: bool,
            ) -> PyResult<()> {
                self.inner.dec_lock_ref(py, node_id, params, skip_swa)
            }

            /// Early-release the SWA portion of a request's tree lock; returns this
            /// release's per-component (device_frees, host_frees).
            #[pyo3(signature = (node_id, swa_uuid_for_lock = None, skip_lock_node_ids = None))]
            fn dec_swa_lock_only(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                swa_uuid_for_lock: Option<i64>,
                skip_lock_node_ids: Option<HashMap<u8, HashSet<NodeId>>>,
            ) -> PyResult<(Py<PyDict>, Py<PyDict>)> {
                self.inner.dec_swa_lock_only(
                    py,
                    node_id,
                    swa_uuid_for_lock,
                    skip_lock_node_ids,
                )
            }

            /// Store a component's device value on a node (the SWA rebuild write-back).
            fn set_component_device_value(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                component_type: u8,
                value: PyTensor,
            ) -> PyResult<()> {
                self.inner
                    .set_component_device_value(py, node_id, component_type, value)
            }

            /// A component's device value on a node, if set.
            fn get_component_device_value(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                component_type: u8,
            ) -> PyResult<Option<PyTensor>> {
                self.inner
                    .get_component_device_value(py, node_id, component_type)
            }

            /// Begin a component's device-eviction walk for up to request_cnt tokens.
            fn evict_device_start(
                &self,
                py: Python<'_>,
                component_type: u8,
                request_cnt: usize,
            ) -> PyResult<()> {
                self.inner
                    .evict_device_start(py, component_type, request_cnt)
            }

            /// The next device leaf to evict, or None when the walk is done; the
            /// passed running tracker gates the budget, and the result carries
            /// this step's deltas.
            fn evict_device_next_node(
                &self,
                py: Python<'_>,
                component_type: u8,
                tracker: HashMap<u8, usize>,
            ) -> PyResult<EvictDeviceNextNodeResultBinding> {
                self.inner
                    .evict_device_next_node(py, component_type, tracker)
            }

            /// Evict one device leaf; an unbacked write-back leaf returns its backup
            /// action for the caller to execute before demoting.
            fn evict_device_leaf(
                &self,
                py: Python<'_>,
                node_id: NodeId,
            ) -> PyResult<EvictDeviceLeafResultBinding> {
                self.inner.evict_device_leaf(py, node_id)
            }

            /// Finish a component's device-eviction walk.
            fn evict_device_end(&self, py: Python<'_>, component_type: u8) -> PyResult<()> {
                self.inner.evict_device_end(py, component_type)
            }

            /// Verify tree-structure, leaf-set, LRU, size, and ongoing-op invariants;
            /// ongoing_* args are (id, node_id) pairs.
            fn sanity_check(
                &self,
                py: Python<'_>,
                ongoing_write_through: Vec<(i64, NodeId)>,
                ongoing_load_back: Vec<(i64, NodeId)>,
            ) -> PyResult<()> {
                self.inner
                    .sanity_check(py, ongoing_write_through, ongoing_load_back)
            }

            /// Concatenated FULL device values from from_node up to (exclusive) until_node.
            fn collect_full_device_indices(
                &self,
                py: Python<'_>,
                from_node_id: NodeId,
                until_node_id: NodeId,
            ) -> PyTensor {
                self.inner
                    .collect_full_device_indices(py, from_node_id, until_node_id)
            }

            /// Every FULL device value in the tree, concatenated.
            fn all_values_flatten(&self, py: Python<'_>) -> PyTensor {
                self.inner.all_values_flatten(py)
            }

            /// Every Mamba device value in the tree, concatenated.
            fn all_mamba_values_flatten(&self, py: Python<'_>) -> PyTensor {
                self.inner.all_mamba_values_flatten(py)
            }

            /// Flatten every FULL device slot into (slot, position, prev-slot) rows for the KV-canary sweep.
            fn walk_for_kv_canary(
                &self,
                py: Python<'_>,
                unlocked_only: bool,
                swa_resident_only: bool,
            ) -> PyResult<KvCanaryWalkResultBinding> {
                self.inner
                    .walk_for_kv_canary(py, unlocked_only, swa_resident_only)
            }

            /// Evictable token count of the FULL (base) component.
            fn evictable_size(&self, py: Python<'_>) -> usize {
                self.inner.evictable_size(py)
            }

            /// Protected (locked) token count of the FULL (base) component.
            fn protected_size(&self, py: Python<'_>) -> usize {
                self.inner.protected_size(py)
            }

            /// FULL component evictable token count.
            fn full_evictable_size(&self, py: Python<'_>) -> usize {
                self.inner.full_evictable_size(py)
            }

            /// FULL component protected token count.
            fn full_protected_size(&self, py: Python<'_>) -> usize {
                self.inner.full_protected_size(py)
            }

            /// Evictable token count for one component (0 if the component is absent).
            fn component_evictable_size(&self, py: Python<'_>, component_type: u8) -> PyResult<usize> {
                self.inner.component_evictable_size(py, component_type)
            }

            /// Protected token count for one component (0 if the component is absent).
            fn component_protected_size(&self, py: Python<'_>, component_type: u8) -> PyResult<usize> {
                self.inner.component_protected_size(py, component_type)
            }

            /// (full_tokens, aux_tokens) summed across the whole tree.
            fn total_size(&self, py: Python<'_>) -> (usize, usize) {
                self.inner.total_size(py)
            }

            /// Whether the node's FULL device value has been evicted.
            fn is_full_device_evicted(&self, py: Python<'_>, node_id: NodeId) -> bool {
                self.inner.is_full_device_evicted(py, node_id)
            }

            /// Mark the host tier (HiCache) as wired.
            fn set_hicache_enabled(&self, py: Python<'_>) {
                self.inner.set_hicache_enabled(py)
            }

            /// Whether the host tier (HiCache) is wired.
            fn enable_hicache(&self, py: Python<'_>) -> bool {
                self.inner.enable_hicache(py)
            }

            /// Mark the SWA host pool as wired (HiCache).
            fn set_has_swa_host_pool(&self, py: Python<'_>) {
                self.inner.set_has_swa_host_pool(py)
            }

            /// Whether the SWA host pool is wired.
            fn has_swa_host_pool(&self, py: Python<'_>) -> bool {
                self.inner.has_swa_host_pool(py)
            }

            /// Insert a host-side (backuped) tree path descending from the given node.
            #[pyo3(signature = (node_id, extra_key, key, host_value, hash_value, cache_salt = None))]
            fn insert_host(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                extra_key: Option<String>,
                key: &Bound<'_, PyAny>,
                host_value: PyTensor,
                hash_value: Vec<String>,
                cache_salt: Option<String>,
            ) -> PyResult<InsertResultBinding> {
                self.inner.insert_host(
                    py,
                    node_id,
                    extra_key,
                    key,
                    host_value,
                    hash_value,
                    cache_salt,
                )
            }

            /// Gather a node's device value plus per-component BACKUP_HOST transfers.
            fn build_backup_spec(
                &self,
                py: Python<'_>,
                node_id: NodeId,
            ) -> PyResult<(PyTensor, Py<PyDict>)> {
                self.inner.build_backup_spec(py, node_id)
            }

            /// Gather a node's device->storage backup spec; None if the node is not backuped.
            fn build_storage_backup_spec(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                pass_prefix_keys: bool,
            ) -> PyResult<Option<StorageBackupSpecBinding>> {
                self.inner
                    .build_storage_backup_spec(py, node_id, pass_prefix_keys)
            }

            /// Route a build_hicache_transfers call to the component for the given type.
            #[allow(clippy::too_many_arguments)]
            #[pyo3(signature = (component_type, node_id, phase, host_indices = None, token_ids = None, prefetch_tokens = 0, last_hash = None))]
            fn build_hicache_transfers(
                &self,
                py: Python<'_>,
                component_type: u8,
                node_id: NodeId,
                phase: &str,
                host_indices: Option<PyTensor>,
                token_ids: Option<Vec<i64>>,
                prefetch_tokens: usize,
                last_hash: Option<String>,
            ) -> PyResult<Option<Vec<Py<PyAny>>>> {
                self.inner.build_hicache_transfers(
                    py,
                    component_type,
                    node_id,
                    phase,
                    host_indices,
                    token_ids,
                    prefetch_tokens,
                    last_hash,
                )
            }

            /// The anchor node's caller-defined key and cache salt.
            fn prefetch_anchor_info(
                &self,
                py: Python<'_>,
                node_id: NodeId,
            ) -> PyResult<(Option<String>, Option<String>)> {
                self.inner.prefetch_anchor_info(py, node_id)
            }

            /// Whether the node's Full KV is present on host.
            fn node_backuped(&self, py: Python<'_>, node_id: NodeId) -> PyResult<bool> {
                self.inner.node_backuped(py, node_id)
            }

            /// Whether the node is a (default or named) root.
            fn is_root(&self, py: Python<'_>, node_id: NodeId) -> PyResult<bool> {
                self.inner.is_root(py, node_id)
            }

            /// The node's last page hash, or None when it was never hashed.
            fn get_last_hash_value(
                &self,
                py: Python<'_>,
                node_id: NodeId,
            ) -> PyResult<Option<String>> {
                self.inner.get_last_hash_value(py, node_id)
            }

            /// The hash chain of the node's ancestors, in root-to-parent order.
            fn get_prefix_hash_values(
                &self,
                py: Python<'_>,
                node_id: NodeId,
            ) -> PyResult<Vec<String>> {
                self.inner.get_prefix_hash_values(py, node_id)
            }

            fn get_hash_values(&self, py: Python<'_>, node_id: NodeId) -> PyResult<Vec<String>> {
                self.inner.get_hash_values(py, node_id)
            }

            fn snapshot_buffer_backup(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                pass_prefix_keys: bool,
            ) -> Option<BufferBackupSnapshotBinding> {
                self.inner
                    .snapshot_buffer_backup(py, node_id, pass_prefix_keys)
            }

            fn validate_buffer_backup(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                expected_key_length: usize,
            ) -> Option<BufferBackupStateBinding> {
                self.inner
                    .validate_buffer_backup(py, node_id, expected_key_length)
            }

            /// Hash every node built while storage was disabled.
            fn backfill_missing_hash_values(&self, py: Python<'_>) -> usize {
                self.inner.backfill_missing_hash_values(py)
            }

            #[pyo3(signature = (extra_key = None))]
            fn root_node_handle(&self, py: Python<'_>, extra_key: Option<String>) -> NodeId {
                self.inner.root_node_handle(py, extra_key)
            }

            fn dfs_weight_order(
                &self,
                py: Python<'_>,
                node_ids: Vec<NodeId>,
            ) -> PyResult<Vec<usize>> {
                self.inner.dfs_weight_order(py, node_ids)
            }

            /// Commit each component's HiCache transfers; returns the new cache actions.
            #[pyo3(signature = (node_id, phase, comp_xfers, insert_result = None, pool_storage_result = None))]
            fn commit_hicache_transfers(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                phase: &str,
                comp_xfers: HashMap<u8, Vec<TransferArgs>>,
                insert_result: Option<(usize, Option<NodeId>, bool)>,
                pool_storage_result: Option<(usize, HashMap<String, usize>)>,
            ) -> PyResult<(Py<PyList>, Option<bool>)> {
                self.inner.commit_hicache_transfers(
                    py,
                    node_id,
                    phase,
                    comp_xfers,
                    insert_result,
                    pool_storage_result,
                )
            }

            /// Commit a successful backup to the node.
            fn commit_backup(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                host_indices: PyTensor,
                comp_xfers: HashMap<u8, Vec<TransferArgs>>,
            ) -> PyResult<()> {
                self.inner
                    .commit_backup(py, node_id, host_indices, comp_xfers)
            }

            /// Build the H->D load-back KV transfer plus per-component aux transfers.
            #[pyo3(signature = (node_id, mamba_pool_idx = None))]
            fn build_load_back_spec(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                mamba_pool_idx: Option<PyTensor>,
            ) -> PyResult<(Py<PyAny>, Py<PyDict>)> {
                self.inner.build_load_back_spec(py, node_id, mamba_pool_idx)
            }

            /// Commit a successful H->D load-back onto the node; returns its actions.
            fn commit_load_back(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                device_indices: PyTensor,
                kv_xfer: TransferArgs,
                comp_xfers: HashMap<u8, Vec<TransferArgs>>,
            ) -> PyResult<Py<PyList>> {
                self.inner
                    .commit_load_back(py, node_id, device_indices, kv_xfer, comp_xfers)
            }

            /// Release a node's device KV once its host copy exists.
            fn demote(&self, py: Python<'_>, node_id: NodeId) -> PyResult<DemoteResultBinding> {
                self.inner.demote(py, node_id)
            }

            /// Evict up to num_tokens of one component's host resources.
            fn drive_host_eviction(
                &self,
                py: Python<'_>,
                component_type: u8,
                num_tokens: usize,
            ) -> PyResult<HostEvictionResultBinding> {
                self.inner.drive_host_eviction(py, component_type, num_tokens)
            }

            /// Evict shallow Mamba device checkpoints beyond the per-path cap
            /// on the tail's root path.
            fn evict_excess_path_states(
                &self,
                py: Python<'_>,
                tail_node_id: NodeId,
            ) -> PyResult<HostEvictionResultBinding> {
                self.inner.evict_excess_path_states(py, tail_node_id)
            }

            /// Bump the reference count on a node's host-side component locks.
            fn inc_host_lock_ref(&self, py: Python<'_>, node_id: NodeId) -> IncLockRefResultBinding {
                self.inner.inc_host_lock_ref(py, node_id)
            }

            /// Decrease the reference count on a node's host-side component locks.
            #[pyo3(signature = (node_id, params = None))]
            fn dec_host_lock_ref(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                params: Option<&DecLockRefParamsBinding>,
            ) -> PyResult<()> {
                self.inner.dec_host_lock_ref(py, node_id, params)
            }

            /// Set the write-back (vs write-through) policy; decided at HiCache init.
            fn set_is_write_back(&self, py: Python<'_>, is_write_back: bool) {
                self.inner.set_is_write_back(py, is_write_back)
            }

            /// The current write-back (vs write-through) policy.
            fn is_write_back(&self, py: Python<'_>) -> bool {
                self.inner.is_write_back(py)
            }

            /// Set the write-through backup hit threshold; decided at HiCache init.
            fn set_write_through_threshold(&self, py: Python<'_>, threshold: i64) {
                self.inner.set_write_through_threshold(py, threshold)
            }

            /// The current write-through backup hit threshold.
            fn write_through_threshold(&self, py: Python<'_>) -> i64 {
                self.inner.write_through_threshold(py)
            }

            /// Mark the storage tier (L3) wired; storage attaches after tree construction.
            fn set_enable_storage(&self, py: Python<'_>, value: bool) {
                self.inner.set_enable_storage(py, value)
            }

            /// Whether the storage tier (L3) is wired.
            fn enable_storage(&self, py: Python<'_>) -> bool {
                self.inner.enable_storage(py)
            }

            /// Queue the all-cleared placement event.
            fn record_all_cleared_event(&self, py: Python<'_>) {
                self.inner.record_all_cleared_event(py)
            }

            /// Drain the queued placement events as tagged tuples.
            fn take_events(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
                self.inner.take_events(py)
            }

            /// Drop the subtree rooted at an unbacked D-leaf; not dropped when a lock
            /// blocks it.
            fn drop_subtree_no_host(
                &self,
                py: Python<'_>,
                node_id: NodeId,
            ) -> PyResult<DropSubtreeResultBinding> {
                self.inner.drop_subtree_no_host(py, node_id)
            }

            /// Mark a node as having an in-flight write-through backup.
            fn mark_write_through_pending(&self, py: Python<'_>, node_id: NodeId) {
                self.inner.mark_write_through_pending(py, node_id)
            }

            /// Clear the write-through-pending mark on the acked nodes.
            fn finish_write_through(&self, py: Python<'_>, node_ids: Vec<NodeId>, ack_id: NodeId) {
                self.inner.finish_write_through(py, node_ids, ack_id)
            }

            /// Clear the in-flight H->D marks on the anchor's root path at ack time.
            fn finish_load_back(&self, py: Python<'_>, anchor_node_id: NodeId) {
                self.inner.finish_load_back(py, anchor_node_id)
            }

            /// Order-sensitive digest of reclaimed coexisting host values.
            #[pyo3(name = "write_back_duplicate_reclaim_digest")]
            fn write_back_coexist_reclaim_digest(&self, py: Python<'_>) -> i64 {
                self.inner.write_back_coexist_reclaim_digest(py)
            }

            /// Whether the component's data is device-evicted but host-backed.
            fn component_has_host_value_only(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                component_type: u8,
            ) -> PyResult<bool> {
                self.inner
                    .component_has_host_value_only(py, node_id, component_type)
            }

            // ==== Test-only inspection surface ====

            #[cfg(feature = "inspection")]
            fn inspect_contains_node(&self, py: Python<'_>, node_id: NodeId) -> bool {
                self.inner.inspect_contains_node(py, node_id)
            }

            #[cfg(feature = "inspection")]
            fn inspect_get_parent_node_id(
                &self,
                py: Python<'_>,
                node_id: NodeId,
            ) -> Option<NodeId> {
                self.inner.inspect_get_parent_node_id(py, node_id)
            }

            #[cfg(feature = "inspection")]
            fn inspect_get_child_node_ids(
                &self,
                py: Python<'_>,
                node_id: NodeId,
            ) -> Vec<NodeId> {
                self.inner.inspect_get_child_node_ids(py, node_id)
            }

            #[cfg(feature = "inspection")]
            fn inspect_get_node_key_length(&self, py: Python<'_>, node_id: NodeId) -> usize {
                self.inner.inspect_get_node_key_length(py, node_id)
            }

            #[cfg(feature = "inspection")]
            fn inspect_get_node_token_ids(&self, py: Python<'_>, node_id: NodeId) -> Vec<i64> {
                self.inner.inspect_get_node_token_ids(py, node_id)
            }

            #[cfg(feature = "inspection")]
            fn inspect_is_node_key_bigram(&self, py: Python<'_>, node_id: NodeId) -> bool {
                self.inner.inspect_is_node_key_bigram(py, node_id)
            }

            #[cfg(feature = "inspection")]
            fn inspect_get_component_host_value(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                component_type: u8,
            ) -> PyResult<Option<PyTensor>> {
                self.inner
                    .inspect_get_component_host_value(py, node_id, component_type)
            }

            #[cfg(feature = "inspection")]
            fn inspect_get_component_device_lock_ref(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                component_type: u8,
            ) -> PyResult<u32> {
                self.inner
                    .inspect_get_component_device_lock_ref(py, node_id, component_type)
            }

            #[cfg(feature = "inspection")]
            fn inspect_get_node_hit_count(&self, py: Python<'_>, node_id: NodeId) -> i64 {
                self.inner.inspect_get_node_hit_count(py, node_id)
            }

            #[cfg(feature = "inspection")]
            fn inspect_get_write_through_pending_id(
                &self,
                py: Python<'_>,
                node_id: NodeId,
            ) -> Option<usize> {
                self.inner
                    .inspect_get_write_through_pending_id(py, node_id)
            }

            #[cfg(feature = "inspection")]
            fn inspect_is_node_in_device_lru(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                component_type: u8,
            ) -> PyResult<bool> {
                self.inner
                    .inspect_is_node_in_device_lru(py, node_id, component_type)
            }

            #[cfg(feature = "inspection")]
            fn inspect_is_node_in_host_lru(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                component_type: u8,
            ) -> PyResult<bool> {
                self.inner
                    .inspect_is_node_in_host_lru(py, node_id, component_type)
            }

            #[cfg(feature = "inspection")]
            fn inspect_get_component_device_lru_node_ids(
                &self,
                py: Python<'_>,
                component_type: u8,
            ) -> PyResult<Vec<NodeId>> {
                self.inner
                    .inspect_get_component_device_lru_node_ids(py, component_type)
            }

            #[cfg(feature = "inspection")]
            fn inspect_is_device_evictable_leaf(
                &self,
                py: Python<'_>,
                node_id: NodeId,
            ) -> bool {
                self.inner.inspect_is_device_evictable_leaf(py, node_id)
            }

            #[cfg(feature = "inspection")]
            fn inspect_is_host_evictable_leaf(
                &self,
                py: Python<'_>,
                node_id: NodeId,
            ) -> bool {
                self.inner.inspect_is_host_evictable_leaf(py, node_id)
            }

            #[cfg(feature = "inspection")]
            fn inspect_is_device_leaf(&self, py: Python<'_>, node_id: NodeId) -> bool {
                self.inner.inspect_is_device_leaf(py, node_id)
            }

            #[cfg(feature = "inspection")]
            fn inspect_get_all_node_ids(&self, py: Python<'_>) -> Vec<NodeId> {
                self.inner.inspect_get_all_node_ids(py)
            }

            #[cfg(feature = "inspection")]
            fn inspect_component_protected_size(
                &self,
                py: Python<'_>,
                component_type: u8,
            ) -> PyResult<usize> {
                self.inner
                    .inspect_component_protected_size(py, component_type)
            }

            #[cfg(feature = "inspection")]
            #[pyo3(signature = (node_id, hash_values = None))]
            fn inspect_set_node_hash_values(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                hash_values: Option<Vec<String>>,
            ) {
                self.inner
                    .inspect_set_node_hash_values(py, node_id, hash_values)
            }

            #[cfg(feature = "inspection")]
            #[pyo3(signature = (node_id, component_type, value = None))]
            fn inspect_set_component_device_value_raw(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                component_type: u8,
                value: Option<PyTensor>,
            ) -> PyResult<()> {
                self.inner.inspect_set_component_device_value_raw(
                    py,
                    node_id,
                    component_type,
                    value,
                )
            }

            #[cfg(feature = "inspection")]
            #[pyo3(signature = (node_id, component_type, value = None))]
            fn inspect_set_component_host_value_raw(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                component_type: u8,
                value: Option<PyTensor>,
            ) -> PyResult<()> {
                self.inner.inspect_set_component_host_value_raw(
                    py,
                    node_id,
                    component_type,
                    value,
                )
            }

            #[cfg(feature = "inspection")]
            fn inspect_set_component_device_lock_ref(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                component_type: u8,
                lock_ref: u32,
            ) -> PyResult<()> {
                self.inner.inspect_set_component_device_lock_ref(
                    py,
                    node_id,
                    component_type,
                    lock_ref,
                )
            }

            #[cfg(feature = "inspection")]
            fn inspect_remove_node_from_device_lru(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                component_type: u8,
            ) -> PyResult<()> {
                self.inner
                    .inspect_remove_node_from_device_lru(py, node_id, component_type)
            }

            #[cfg(feature = "inspection")]
            fn inspect_insert_node_into_host_lru(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                component_type: u8,
            ) -> PyResult<()> {
                self.inner
                    .inspect_insert_node_into_host_lru(py, node_id, component_type)
            }

            #[cfg(feature = "inspection")]
            fn inspect_set_component_evictable_size(
                &self,
                py: Python<'_>,
                component_type: u8,
                value: usize,
            ) -> PyResult<()> {
                self.inner
                    .inspect_set_component_evictable_size(py, component_type, value)
            }

            #[cfg(feature = "inspection")]
            fn inspect_set_component_protected_size(
                &self,
                py: Python<'_>,
                component_type: u8,
                value: usize,
            ) -> PyResult<()> {
                self.inner
                    .inspect_set_component_protected_size(py, component_type, value)
            }

            #[cfg(feature = "inspection")]
            fn inspect_update_duplicate_tracking(&self, py: Python<'_>, node_id: NodeId) {
                self.inner.inspect_update_duplicate_tracking(py, node_id)
            }

            #[cfg(feature = "inspection")]
            fn inspect_advance_insert_walk_once(&self, py: Python<'_>) -> PyResult<()> {
                self.inner.inspect_advance_insert_walk_once(py)
            }

            #[cfg(feature = "inspection")]
            fn inspect_evict_component(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                component_type: u8,
                target: u8,
            ) -> PyResult<HostEvictionResultBinding> {
                self.inner
                    .inspect_evict_component(py, node_id, component_type, target)
            }

            #[cfg(feature = "inspection")]
            fn inspect_validate_cascade_evict(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                component_type: u8,
                target: u8,
            ) -> PyResult<()> {
                self.inner
                    .inspect_validate_cascade_evict(py, node_id, component_type, target)
            }

            #[cfg(feature = "inspection")]
            fn inspect_cleanup_tombstone_ancestors(
                &self,
                py: Python<'_>,
                node_id: NodeId,
            ) -> PyResult<HostEvictionResultBinding> {
                self.inner
                    .inspect_cleanup_tombstone_ancestors(py, node_id)
            }

            #[cfg(feature = "inspection")]
            #[allow(clippy::too_many_arguments)]
            #[pyo3(signature = (component_type, result, key, extra_key, cache_salt, value_chunks, best_value_len))]
            fn inspect_finalize_component_match_result(
                &self,
                py: Python<'_>,
                component_type: u8,
                result: InspectionMatchResultInput,
                key: &Bound<'_, PyAny>,
                extra_key: Option<String>,
                cache_salt: Option<String>,
                value_chunks: Vec<PyTensor>,
                best_value_len: usize,
            ) -> PyResult<MatchResultBinding> {
                self.inner.inspect_finalize_component_match_result(
                    py,
                    component_type,
                    result,
                    key,
                    extra_key,
                    cache_salt,
                    value_chunks,
                    best_value_len,
                )
            }

            #[cfg(feature = "inspection")]
            #[pyo3(signature = (node_id, write_back = false))]
            fn inspect_build_backup_node_ids(
                &self,
                py: Python<'_>,
                node_id: NodeId,
                write_back: bool,
            ) -> Vec<NodeId> {
                self.inner
                    .inspect_build_backup_node_ids(py, node_id, write_back)
            }

            /// Print the tree structure for debugging.
            fn pretty_print(&self, py: Python<'_>) {
                self.inner.pretty_print(py)
            }
        }
    };
}

tree_core_binding!(
    /// The UnifiedTreeCore Python adapter over single-token (unigram) child keys.
    RustUnifiedTreeCoreBinding,
    Vec<i64>
);

tree_core_binding!(
    /// The UnifiedTreeCore Python adapter over bigram (EAGLE) child keys; keys
    /// cross the boundary as raw token ids and pair up rust-side.
    RustBigramUnifiedTreeCoreBinding,
    Vec<(i64, i64)>
);

/// Per-page chained hashes over raw token ids.
#[pyfunction]
#[pyo3(signature = (token_ids, prior_hash, page_size, is_bigram = false))]
fn get_hash_str(
    py: Python<'_>,
    token_ids: &Bound<'_, PyAny>,
    prior_hash: Option<String>,
    page_size: usize,
    is_bigram: bool,
) -> PyResult<Vec<String>> {
    let raw = py_array_to_vec_i64(py, token_ids)?;
    if page_size == 0 {
        return Err(PyValueError::new_err("page_size must be positive"));
    }
    if let Some(prior_hash) = prior_hash.as_deref().filter(|hash| !hash.is_empty())
        && (prior_hash.len() != 64 || !prior_hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        return Err(PyValueError::new_err(
            "prior_hash must be a 64-character hexadecimal digest",
        ));
    }
    if let Some(token_id) = raw
        .iter()
        .find(|token_id| u32::try_from(**token_id).is_err())
    {
        return Err(PyValueError::new_err(format!(
            "token id {token_id} does not fit in uint32"
        )));
    }
    Ok(py.allow_threads(move || {
        if is_bigram {
            let key = <Vec<(i64, i64)> as ChildKeyType>::key_from(Cow::Owned(raw)).into_owned();
            crate::node::get_hash_str::<Vec<(i64, i64)>>(&key, prior_hash.as_deref(), page_size)
        } else {
            crate::node::get_hash_str::<Vec<i64>>(&raw, prior_hash.as_deref(), page_size)
        }
    }))
}

fn register_mem_cache_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_hash_str, m)?)?;
    m.add_class::<TreeCoreInitParamsBinding>()?;
    m.add_class::<MatchParamsBinding>()?;
    m.add_class::<InsertParamsBinding>()?;
    m.add_class::<MatchResultBinding>()?;
    m.add_class::<InsertResultBinding>()?;
    m.add_class::<IncLockRefResultBinding>()?;
    m.add_class::<DecLockRefParamsBinding>()?;
    m.add_class::<EvictDeviceNextNodeResultBinding>()?;
    m.add_class::<EvictDeviceLeafResultBinding>()?;
    m.add_class::<DropSubtreeResultBinding>()?;
    m.add_class::<DemoteResultBinding>()?;
    m.add_class::<KvCanaryWalkResultBinding>()?;
    m.add_class::<StorageBackupSpecBinding>()?;
    m.add_class::<BufferBackupSnapshotBinding>()?;
    m.add_class::<BufferBackupStateBinding>()?;
    m.add_class::<HostEvictionResultBinding>()?;
    m.add_class::<RustUnifiedTreeCoreBinding>()?;
    m.add_class::<RustBigramUnifiedTreeCoreBinding>()?;
    Ok(())
}

/// The production TreeCore extension module.
#[pymodule]
fn mem_cache(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_mem_cache_module(m)
}

/// White-box variant used only by the shared test inspector.
#[cfg(feature = "inspection")]
#[pymodule]
fn mem_cache_inspection(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_mem_cache_module(m)
}
