//! Cache actions returned to the cache for execution.

use std::fmt::Debug;

use tch::Tensor;

use crate::components::ComponentType;
use crate::node::NodeId;

pub trait CacheAction: Debug + Send {
    fn tag(&self) -> &'static str;

    /// Whether insertion must pause for the cache to apply this action.
    fn requires_insert_barrier(&self) -> bool {
        true
    }

    #[cfg(feature = "python-extension")]
    fn into_py_tagged(
        self: Box<Self>,
        py: pyo3::Python<'_>,
    ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>>;

    /// Rust tests inspect emitted payloads without a production downcast API.
    #[cfg(test)]
    fn as_any(&self) -> &dyn std::any::Any;
}

pub type CacheActions = Vec<Box<dyn CacheAction>>;

macro_rules! impl_cache_action {
    ($action:ty, $tag:literal $(, insert_barrier = $barrier:expr)? $(,)?) => {
        impl $crate::cache_action::CacheAction for $action {
            fn tag(&self) -> &'static str {
                $tag
            }

            $(fn requires_insert_barrier(&self) -> bool {
                $barrier
            })?

            #[cfg(feature = "python-extension")]
            fn into_py_tagged(
                self: Box<Self>,
                py: pyo3::Python<'_>,
            ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                <$action>::into_py_tagged(*self, py)
            }

            #[cfg(test)]
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
    };
}
pub(crate) use impl_cache_action;

/// Duplicate device KV slices the cache frees after the insert.
#[derive(Debug)]
pub struct FreeDeviceKV {
    pub indices: Vec<Tensor>,
}

/// Free the full side only, for a tombstoned node whose SWA peers are gone;
/// FreeDeviceKV would release the SWA side twice.
#[derive(Debug)]
pub struct FreeDeviceKVFullOnly {
    pub indices: Vec<Tensor>,
}

/// Update cache-side write bookkeeping after a node split:
///
///     parent -> node    =>    parent -> new_node -> new_child
///
/// old_node_id (the pre-split node) is replaced by new_node_id + new_child_node_id.
#[derive(Debug)]
pub struct UpdatePendingBackupOnNodeSplit {
    /// None when buffer-only bookkeeping has no tree-tracked write.
    pub ack_id: Option<usize>,
    pub old_node_id: NodeId,
    pub new_node_id: NodeId,
    pub new_child_node_id: NodeId,
}

/// Per-path Mamba state-cap eviction from the tail's root path; applied at
/// the insert's commit barrier, after the walk-time backups whose
/// write-through locks shield the backed-up chain.
#[derive(Debug)]
pub struct MambaEvictExcessPathStates {
    pub tail_node_id: NodeId,
}

/// Free only the given component's device KV slots.
#[derive(Debug)]
pub struct FreeComponentDeviceSlot {
    pub component_type: ComponentType,
    pub indices: Vec<Tensor>,
}

/// Free the given component's host KV pages.
#[derive(Debug)]
pub struct FreeComponentHostSlot {
    pub component_type: ComponentType,
    pub host_indices: Vec<Tensor>,
}

/// Rebuild the SWA allocator's full->swa index mapping for loaded chunks.
#[derive(Debug)]
pub struct RebuildFullToSwaMapping {
    pub full_indices: Vec<Tensor>,
    pub swa_indices: Vec<Tensor>,
}

/// Recover an SWA tombstone whose full is locked: keep the locked full, remap
/// it onto the incoming full's SWA translation, and free only the incoming full.
#[derive(Debug)]
pub struct RecoverSwaWithLockedFull {
    pub node_id: NodeId,
    pub kept_full: Tensor,
    pub incoming_full: Tensor,
}

/// Rebuild a node's SWA value by translating its source full value, then store it.
#[derive(Debug)]
pub struct SwaRebuild {
    pub node_id: NodeId,
    pub source_value: Tensor,
}

/// A device->host backup work item for the cache to execute.
#[derive(Default, Debug)]
pub struct BackupKV {
    /// Backup these nodes device->host in order, stopping at the first failure; the
    /// caller orders them parent-before-child for write-through and child-first for
    /// write-back. External handles: the list crosses to the orchestrator.
    pub node_ids: Vec<NodeId>,
}

self::impl_cache_action!(FreeDeviceKV, "free_device_kv", insert_barrier = false);
self::impl_cache_action!(
    FreeDeviceKVFullOnly,
    "free_device_kv_full_only",
    insert_barrier = false
);
self::impl_cache_action!(BackupKV, "backup_kv");
self::impl_cache_action!(
    UpdatePendingBackupOnNodeSplit,
    "update_pending_backup_on_node_split",
    insert_barrier = false
);
self::impl_cache_action!(MambaEvictExcessPathStates, "mamba_evict_excess_path_states");
self::impl_cache_action!(FreeComponentDeviceSlot, "free_component_device_slot");
self::impl_cache_action!(FreeComponentHostSlot, "free_component_host_slot");
self::impl_cache_action!(RebuildFullToSwaMapping, "rebuild_full_to_swa_mapping");
self::impl_cache_action!(RecoverSwaWithLockedFull, "recover_swa_with_locked_full");
self::impl_cache_action!(SwaRebuild, "swa_rebuild");
