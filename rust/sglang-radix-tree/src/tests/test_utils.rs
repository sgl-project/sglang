//! Shared helpers for the crate's unit tests.

use std::collections::HashMap;

use tch::Tensor;

use crate::components::ComponentType;
use crate::node::ChildKeyType;
use crate::unified_tree_core::{CacheAction, EvictionStepResult, MatchResult, UnifiedTreeCore};

pub(crate) fn matched_device_indices<K: ChildKeyType>(
    tc: &UnifiedTreeCore<K>,
    result: &MatchResult,
) -> Tensor {
    tc.collect_full_device_indices(result.last_device_node_id, tc.root_node_handle(None))
        .expect("live match node")
}

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

/// Tags for diagnosing an action sequence's shape.
pub(crate) fn action_kinds(actions: &[Box<dyn CacheAction>]) -> Vec<&'static str> {
    actions.iter().map(|action| action.tag()).collect()
}

pub(crate) fn expect_action<T: std::any::Any>(action: &dyn CacheAction) -> &T {
    action
        .as_any()
        .downcast_ref::<T>()
        .unwrap_or_else(|| panic!("expected {}, got {:?}", std::any::type_name::<T>(), action))
}
