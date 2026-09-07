//! Shared helpers for the crate's unit tests.

use std::collections::HashMap;

use tch::Tensor;

use crate::components::ComponentType;
use crate::unified_tree_core::{CacheAction, EvictionStepResult};

/// Fold an eviction step into a caller's running accumulators (the Controller
/// consumption contract: deltas add, freed tensors append).
pub(crate) fn accumulate_step(
    step: EvictionStepResult,
    tracker: &mut HashMap<ComponentType, usize>,
    device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
) {
    for (ct, delta) in step.tracker {
        *tracker.entry(ct).or_insert(0) += delta;
    }
    for (ct, tensors) in step.device_frees {
        device_frees.entry(ct).or_default().extend(tensors);
    }
    for (ct, tensors) in step.host_frees {
        host_frees.entry(ct).or_default().extend(tensors);
    }
}

/// Short variant names for diagnosing an action sequence's shape.
pub(crate) fn action_kinds(actions: &[CacheAction]) -> Vec<&'static str> {
    actions
        .iter()
        .map(|action| match action {
            CacheAction::FreeDeviceKV(_) => "FreeDeviceKV",
            CacheAction::FreeDeviceKVFullOnly(_) => "FreeDeviceKVFullOnly",
            CacheAction::BackupKV(_) => "BackupKV",
            CacheAction::ReplaceWriteThroughOnNodeSplit { .. } => "ReplaceWriteThroughOnNodeSplit",
            CacheAction::MambaEvictExcessPathStates { .. } => "MambaEvictExcessPathStates",
            CacheAction::FreeComponentDeviceSlot { .. } => "FreeComponentDeviceSlot",
            CacheAction::FreeComponentHostSlot { .. } => "FreeComponentHostSlot",
            CacheAction::RebuildFullToSwaMapping { .. } => "RebuildFullToSwaMapping",
            CacheAction::RecoverSwaWithLockedFull { .. } => "RecoverSwaWithLockedFull",
            CacheAction::SwaRebuild { .. } => "SwaRebuild",
        })
        .collect()
}
