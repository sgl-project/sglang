//! SWA KV kept only for the window before each cached Mamba checkpoint.

use std::collections::HashMap;

use tch::Tensor;

use crate::components::{ComponentType, MAMBA, SWA};
use crate::node::{ChildKeyType, NodeIdx_, ValueSlotIdx};
use crate::unified_tree_core::{CacheActions, EvictLayer, UnifiedTreeCore};

fn swa_slot(host: bool) -> ValueSlotIdx {
    if host {
        ValueSlotIdx::host(SWA)
    } else {
        ValueSlotIdx::device(SWA)
    }
}

fn is_reclaimable<K: ChildKeyType>(
    tree_core: &UnifiedTreeCore<K>,
    node_id: NodeIdx_,
    host: bool,
) -> bool {
    let state = tree_core.arena.node(node_id).state_(swa_slot(host));
    state.value.is_some() && state.lock_ref == 0 && state.cover_ref == 0
}

/// Nodes of the window ending at checkpoint, deepest first.
fn window<K: ChildKeyType>(tree_core: &UnifiedTreeCore<K>, checkpoint: NodeIdx_) -> Vec<NodeIdx_> {
    let window_tokens = tree_core
        .checkpoint_window_tokens
        .expect("checkpoint windows are disabled");
    let mut nodes = Vec::new();
    let mut covered = 0;
    let mut cur = checkpoint;
    while !tree_core.arena.node(cur).is_root() && covered < window_tokens {
        nodes.push(cur);
        covered += tree_core.arena.node(cur).key.atom_len();
        cur = tree_core.arena.node(cur).parent();
    }
    assert!(
        covered <= window_tokens,
        "window of node {checkpoint} does not start on a node boundary \
         ({covered} > {window_tokens} tokens)"
    );
    nodes
}

/// Split the node that straddles a new checkpoint's window start, then add the window to its nodes' counts.
pub(crate) fn store_checkpoint<K: ChildKeyType>(
    tree_core: &mut UnifiedTreeCore<K>,
    checkpoint: NodeIdx_,
    cache_actions: &mut CacheActions,
) {
    let window_tokens = tree_core
        .checkpoint_window_tokens
        .expect("checkpoint windows are disabled");
    let mut covered = 0;
    let mut cur = checkpoint;
    while !tree_core.arena.node(cur).is_root() && covered < window_tokens {
        let len = tree_core.arena.node(cur).key.atom_len();
        if covered + len > window_tokens {
            let (_, action) = tree_core.split_node_(cur, covered + len - window_tokens);
            if let Some(action) = action {
                cache_actions.push(Box::new(action));
            }
            break;
        }
        covered += len;
        cur = tree_core.arena.node(cur).parent();
    }
    acquire_window(tree_core, checkpoint, false);
}

/// Add a new checkpoint's window to the count of every node it includes.
pub(crate) fn acquire_window<K: ChildKeyType>(
    tree_core: &mut UnifiedTreeCore<K>,
    checkpoint: NodeIdx_,
    host: bool,
) {
    for node_id in window(tree_core, checkpoint) {
        tree_core
            .arena
            .node_mut(node_id)
            .state_mut_(swa_slot(host))
            .cover_ref += 1;
        tree_core.component_state_mut(SWA).uncovered[usize::from(host)].discard(node_id);
    }
}

/// Drop a checkpoint's window, freeing KV no other window covers.
pub(crate) fn release_window<K: ChildKeyType>(
    tree_core: &mut UnifiedTreeCore<K>,
    checkpoint: NodeIdx_,
    host: bool,
    device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    cache_actions: &mut CacheActions,
    mut tracker: Option<&mut HashMap<ComponentType, usize>>,
) {
    for node_id in window(tree_core, checkpoint) {
        let state = tree_core.arena.node_mut(node_id).state_mut_(swa_slot(host));
        assert!(
            state.cover_ref > 0,
            "window release hit {} cover_ref=0 on node {node_id}",
            swa_slot(host).tier()
        );
        state.cover_ref -= 1;
        if !is_reclaimable(tree_core, node_id, host) {
            continue;
        }
        if host {
            free(
                tree_core,
                node_id,
                true,
                device_frees,
                host_frees,
                cache_actions,
                tracker.as_deref_mut(),
            );
        } else {
            release_device_kv(
                tree_core,
                node_id,
                device_frees,
                host_frees,
                cache_actions,
                tracker.as_deref_mut(),
            );
        }
    }
}

/// Free a node's unlocked device SWA KV outside every window, or queue it while a host window still needs it.
pub(crate) fn release_device_kv<K: ChildKeyType>(
    tree_core: &mut UnifiedTreeCore<K>,
    node_id: NodeIdx_,
    device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    cache_actions: &mut CacheActions,
    tracker: Option<&mut HashMap<ComponentType, usize>>,
) {
    let host_state = tree_core.arena.node(node_id).state_(swa_slot(true));
    // Hack: a write-through backup stops at an in-flight ancestor, so a host window can lack this node's SWA.
    // TODO: back up that SWA when the ancestor's ack lands, then hold the node only until then.
    if host_state.cover_ref > 0 && host_state.value.is_none() {
        queue_if_uncovered(tree_core, node_id, false);
    } else {
        free(
            tree_core,
            node_id,
            false,
            device_frees,
            host_frees,
            cache_actions,
            tracker,
        );
    }
}

fn free<K: ChildKeyType>(
    tree_core: &mut UnifiedTreeCore<K>,
    node_id: NodeIdx_,
    host: bool,
    device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    cache_actions: &mut CacheActions,
    tracker: Option<&mut HashMap<ComponentType, usize>>,
) {
    let target = if host {
        EvictLayer::Host
    } else {
        EvictLayer::Device
    };
    tree_core.evict_component_and_detach_lru_(
        node_id,
        SWA,
        device_frees,
        host_frees,
        cache_actions,
        target,
        tracker,
    );
    tree_core.update_evictable_leaf_sets_(node_id);
}

/// Queue the node to be freed if its SWA KV on this tier is unlocked and outside every window.
pub(crate) fn queue_if_uncovered<K: ChildKeyType>(
    tree_core: &mut UnifiedTreeCore<K>,
    node_id: NodeIdx_,
    host: bool,
) {
    if is_reclaimable(tree_core, node_id, host) {
        tree_core.component_state_mut(SWA).uncovered[usize::from(host)].add(node_id);
    }
}

pub(crate) fn dequeue<K: ChildKeyType>(
    tree_core: &mut UnifiedTreeCore<K>,
    node_id: NodeIdx_,
    target: EvictLayer,
) {
    let uncovered = &mut tree_core.component_state_mut(SWA).uncovered;
    if target.contains(EvictLayer::Device) {
        uncovered[0].discard(node_id);
    }
    if target.contains(EvictLayer::Host) {
        uncovered[1].discard(node_id);
    }
}

fn pop_reclaimable<K: ChildKeyType>(
    tree_core: &mut UnifiedTreeCore<K>,
    host: bool,
) -> Option<NodeIdx_> {
    loop {
        let uncovered = &mut tree_core.component_state_mut(SWA).uncovered[usize::from(host)];
        let node_id = uncovered.iter().next()?;
        uncovered.discard(node_id);
        if is_reclaimable(tree_core, node_id, host) {
            return Some(node_id);
        }
    }
}

/// Copy the child's window counts and free-queue entry to the new parent of a split.
pub(crate) fn redistribute_on_node_split<K: ChildKeyType>(
    tree_core: &mut UnifiedTreeCore<K>,
    new_parent_id: NodeIdx_,
    child_id: NodeIdx_,
) {
    for host in [false, true] {
        let cover_ref = tree_core
            .arena
            .node(child_id)
            .state_(swa_slot(host))
            .cover_ref;
        tree_core
            .arena
            .node_mut(new_parent_id)
            .state_mut_(swa_slot(host))
            .cover_ref = cover_ref;
        queue_if_uncovered(tree_core, new_parent_id, host);
    }
}

/// Free one node's SWA KV that is outside every window. Windowed KV is freed with its checkpoint.
pub(crate) fn evict_device_next_node<K: ChildKeyType>(
    tree_core: &mut UnifiedTreeCore<K>,
    tracker: &mut HashMap<ComponentType, usize>,
    device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    cache_actions: &mut CacheActions,
) -> Option<NodeIdx_> {
    if tracker[&SWA] < tree_core.component_state(SWA).evict_device_request_cnt
        && let Some(node_id) = pop_reclaimable(tree_core, false)
    {
        free(
            tree_core,
            node_id,
            false,
            device_frees,
            host_frees,
            cache_actions,
            Some(tracker),
        );
    }
    None
}

/// Free host SWA KV outside every window, then evict host checkpoints with their windows in LRU order.
pub(crate) fn drive_host_eviction<K: ChildKeyType>(
    tree_core: &mut UnifiedTreeCore<K>,
    num_tokens: usize,
    tracker: &mut HashMap<ComponentType, usize>,
    device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    cache_actions: &mut CacheActions,
) {
    while tracker[&SWA] < num_tokens {
        let Some(node_id) = pop_reclaimable(tree_core, true) else {
            break;
        };
        free(
            tree_core,
            node_id,
            true,
            device_frees,
            host_frees,
            cache_actions,
            Some(tracker),
        );
    }
    let mamba = tree_core.component_by_type_(MAMBA);
    while tracker[&SWA] < num_tokens {
        let mut checkpoint_tracker = HashMap::from([(MAMBA, 0)]);
        mamba.drive_host_eviction(
            tree_core,
            1,
            &mut checkpoint_tracker,
            device_frees,
            host_frees,
            cache_actions,
        );
        if checkpoint_tracker.values().all(|&count| count == 0) {
            break;
        }
        for (component_type, count) in checkpoint_tracker {
            *tracker.entry(component_type).or_insert(0) += count;
        }
    }
}

/// Report window counts and reclaim queues that disagree with the tree.
pub(crate) fn validate<K: ChildKeyType>(
    tree_core: &UnifiedTreeCore<K>,
    nodes: &[NodeIdx_],
    errors: &mut Vec<String>,
) {
    let nodes: Vec<NodeIdx_> = nodes
        .iter()
        .copied()
        .filter(|&node_id| !tree_core.arena.node(node_id).is_root())
        .collect();
    for host in [false, true] {
        let tier = swa_slot(host).tier();
        let mamba_slot = if host {
            ValueSlotIdx::host(MAMBA)
        } else {
            ValueSlotIdx::device(MAMBA)
        };
        let mut expected: HashMap<NodeIdx_, u32> = HashMap::new();
        for &node_id in &nodes {
            if tree_core.arena.node(node_id).has_value_(mamba_slot) {
                for covered in window(tree_core, node_id) {
                    *expected.entry(covered).or_insert(0) += 1;
                }
            }
        }
        let uncovered = &tree_core.component_state(SWA).uncovered[usize::from(host)];
        for &node_id in &nodes {
            let state = tree_core.arena.node(node_id).state_(swa_slot(host));
            let want = expected.get(&node_id).copied().unwrap_or(0);
            if state.cover_ref != want {
                errors.push(format!(
                    "[CheckpointWindow] SWA node {node_id} {tier} cover_ref={}, expected {want}",
                    state.cover_ref
                ));
            }
            if is_reclaimable(tree_core, node_id, host) && !uncovered.contains(node_id) {
                errors.push(format!(
                    "[CheckpointWindow] SWA node {node_id} holds uncovered {tier} KV unqueued"
                ));
            }
        }
    }
}
