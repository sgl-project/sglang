//! Cache-residency switches: Mamba states on backed-up nodes under write_back,
//! and host evict-first marks. Every switch is off by default; the "off" cases
//! pin the stock decisions the switches change.

use super::*;
use crate::unified_tree_core::CacheResidencyOptions;
use tch::{Device, Kind};

fn residency(keep_backed_mamba_states: bool, host_evict_first: bool) -> CacheResidencyOptions {
    CacheResidencyOptions {
        keep_backed_mamba_states,
        host_evict_first,
    }
}

fn write_back_core() -> UnifiedTreeCore<Vec<i64>> {
    let mut tc = mamba_core(/* page_size = */ 1);
    tc.set_hicache_enabled();
    tc.is_write_back = true;
    tc
}

fn node_of(tc: &mut UnifiedTreeCore<Vec<i64>>, key: Vec<i64>) -> NodeId {
    let id = tc.match_prefix(&match_params(&key)).best_match_node_id;
    assert_ne!(id, tc.root_node_handle(None), "{key:?} is not cached");
    id
}

fn idx(tc: &UnifiedTreeCore<Vec<i64>>, id: NodeId) -> NodeIdx_ {
    tc.arena.resolve(id).expect("live test node")
}

/// Commit and settle one node's D->H backup, as the controller does after the
/// copy's ack: Full host slots at `full_host` (unless Full is already on host)
/// and a Mamba host slot at `mamba_host` when the spec carries one.
fn commit_host_copy(
    tc: &mut UnifiedTreeCore<Vec<i64>>,
    node: NodeId,
    full_host: i64,
    mamba_host: i64,
) {
    let (full, mut transfers) = tc.build_backup_spec(node).unwrap();
    if let Some(xfers) = transfers.get_mut(&MAMBA) {
        xfers[0].host_indices = Some(Tensor::from_slice(&[mamba_host]));
    }
    let host = if full.numel() == 0 {
        full
    } else {
        Tensor::arange_start(
            full_host,
            full_host + full.numel() as i64,
            (Kind::Int64, Device::Cpu),
        )
    };
    tc.commit_backup(node, host, transfers).unwrap();
    tc.mark_write_through_pending(vec![node], node).unwrap();
    tc.finish_write_through(vec![node], node).unwrap();
}

/// Give a node a Full-only host copy, leaving its Mamba state device-only: the
/// shape a state attached after the node's Full KV reached host has.
fn commit_full_only_host_copy(tc: &mut UnifiedTreeCore<Vec<i64>>, node: NodeId, full_host: i64) {
    let len = tc.arena.node(idx(tc, node)).device_value(FULL).numel() as i64;
    tc.commit_backup(
        node,
        Tensor::arange_start(full_host, full_host + len, (Kind::Int64, Device::Cpu)),
        HashMap::new(),
    )
    .unwrap();
    tc.mark_write_through_pending(vec![node], node).unwrap();
    tc.finish_write_through(vec![node], node).unwrap();
    let node = tc.arena.node(idx(tc, node));
    assert!(node.backuped() && node.has_device_value(MAMBA) && !node.has_host_value(MAMBA));
}

// ---- keep_backed_mamba_states: internal nodes ----

#[test]
fn stock_walk_drops_a_device_only_state_on_a_backed_internal_node() {
    let (mut tc, parent, _leaf) = internal_write_back_fixture();
    commit_full_only_host_copy(&mut tc, parent, 100);
    tc.evict_device_start(MAMBA, 1);
    let (next, step) = tc.evict_device_next_node(MAMBA, &HashMap::new());
    tc.evict_device_end(MAMBA);
    assert_eq!(next, None);
    assert_eq!(step.mamba_backup_node_id, None);
    assert_eq!(step.tracker, HashMap::from([(MAMBA, 1)]));
    let node = tc.arena.node(idx(&tc, parent));
    assert!(!node.has_device_value(MAMBA) && !node.has_host_value(MAMBA));
    assert_eq!(tc.cache_residency_stats().backed_node_state_backups, 0);
    tc.sanity_check(&[], &[]);
}

#[test]
fn keep_backed_mamba_states_backs_up_a_state_on_a_backed_internal_node() {
    let (mut tc, parent, _leaf) = internal_write_back_fixture();
    tc.configure_cache_residency(residency(true, false));
    commit_full_only_host_copy(&mut tc, parent, 100);
    tc.evict_device_start(MAMBA, 1);
    let (next, step) = tc.evict_device_next_node(MAMBA, &HashMap::new());
    assert_eq!(next, None);
    assert_eq!(step.mamba_backup_node_id, Some(parent));
    assert!(step.tracker.is_empty() && step.device_frees.is_empty());
    assert_eq!(tc.cache_residency_stats().backed_node_state_backups, 1);

    // The controller's copy: no Full tokens (already on host), one Mamba state.
    let (full, transfers) = tc.build_backup_spec(parent).unwrap();
    assert_eq!(full.numel(), 0);
    assert!(
        transfers[&MAMBA][0]
            .device_indices
            .as_ref()
            .unwrap()
            .equal(&Tensor::from_slice(&[7i64]))
    );
    commit_host_copy(&mut tc, parent, 0, 70);
    let step = tc.finish_mamba_state_eviction(parent);
    tc.evict_device_end(MAMBA);
    assert_eq!(step.tracker, HashMap::from([(MAMBA, 1)]));
    assert!(step.device_frees[&MAMBA][0].equal(&Tensor::from_slice(&[7i64])));
    let node = tc.arena.node(idx(&tc, parent));
    assert!(node.device_value(FULL).equal(&Tensor::from_slice(&[10i64])));
    assert!(node.host_value(FULL).equal(&Tensor::from_slice(&[100i64])));
    assert!(node.host_value(MAMBA).equal(&Tensor::from_slice(&[70i64])));
    assert!(!node.has_device_value(MAMBA));
    // The state is still a hit: a match ending at [1] loads it back.
    let matched = tc.match_prefix(&match_params(&vec![1]));
    assert_eq!(matched.best_match_node_id, parent);
    assert_eq!(matched.mamba_host_hit_length, 1);
    tc.sanity_check(&[], &[]);
}

#[test]
fn keep_backed_mamba_states_leaves_other_policies_and_settled_states_alone() {
    for mode in ["write_through", "no_hicache", "state_on_host"] {
        let (mut tc, parent, _leaf) = internal_write_back_fixture();
        tc.configure_cache_residency(residency(true, false));
        match mode {
            "write_through" => {
                commit_full_only_host_copy(&mut tc, parent, 100);
                tc.is_write_back = false;
            }
            "no_hicache" => {
                commit_full_only_host_copy(&mut tc, parent, 100);
                tc.enable_hicache = false;
            }
            "state_on_host" => commit_host_copy(&mut tc, parent, 100, 70),
            _ => unreachable!(),
        }
        tc.evict_device_start(MAMBA, 1);
        let (_, step) = tc.evict_device_next_node(MAMBA, &HashMap::new());
        tc.evict_device_end(MAMBA);
        assert_eq!(step.mamba_backup_node_id, None, "{mode}");
        assert_eq!(step.tracker, HashMap::from([(MAMBA, 1)]), "{mode}");
        assert_eq!(
            tc.cache_residency_stats().backed_node_state_backups,
            0,
            "{mode}"
        );
    }
}

// ---- keep_backed_mamba_states: device leaves ----

#[test]
fn needs_state_backup_before_demote_only_for_a_backed_leaf_with_an_unbacked_state() {
    let (mut tc, _parent, leaf) = internal_write_back_fixture();
    // Off: the stock demote path.
    commit_full_only_host_copy(&mut tc, leaf, 200);
    assert!(!tc.needs_state_backup_before_demote(leaf).unwrap());
    tc.configure_cache_residency(residency(true, false));
    assert!(tc.needs_state_backup_before_demote(leaf).unwrap());
    assert_eq!(tc.cache_residency_stats().state_backups_before_demote, 1);
    tc.is_write_back = false;
    assert!(!tc.needs_state_backup_before_demote(leaf).unwrap());
    tc.is_write_back = true;
    commit_host_copy(&mut tc, leaf, 0, 80);
    assert!(!tc.needs_state_backup_before_demote(leaf).unwrap());

    // A leaf whose Full KV is not on host takes the ordinary BackupKV path.
    let (mut tc, _parent, leaf) = internal_write_back_fixture();
    tc.configure_cache_residency(residency(true, false));
    assert!(!tc.needs_state_backup_before_demote(leaf).unwrap());
}

#[test]
fn a_state_copied_before_the_demote_survives_on_host() {
    for top_up in [false, true] {
        let (mut tc, _parent, leaf) = internal_write_back_fixture();
        tc.configure_cache_residency(residency(true, false));
        commit_full_only_host_copy(&mut tc, leaf, 200);
        if top_up {
            assert!(tc.needs_state_backup_before_demote(leaf).unwrap());
            commit_host_copy(&mut tc, leaf, 0, 80);
        }
        let (backup, step) = tc
            .evict_device_leaf(leaf, /* is_write_back = */ true)
            .unwrap();
        assert!(backup.is_none());
        assert!(step.device_frees[&MAMBA][0].equal(&Tensor::from_slice(&[8i64])));
        let node = tc.arena.node(idx(&tc, leaf));
        assert!(node.evicted() && node.backuped());
        assert_eq!(node.has_host_value(MAMBA), top_up);
        assert_eq!(
            tc.host_lru_list(MAMBA).in_list(Some(idx(&tc, leaf))),
            top_up
        );
        let matched = tc.match_prefix(&match_params(&vec![1, 2]));
        if top_up {
            assert_eq!(matched.best_match_node_id, leaf);
            assert_eq!(matched.mamba_host_hit_length, 1);
        } else {
            // Stock: the turn-end state is gone, so the hit falls back to [1].
            assert_ne!(matched.best_match_node_id, leaf);
        }
        tc.sanity_check(&[], &[]);
    }
}

// ---- host_evict_first ----

/// Two finished root children [1] and [2] on host only (Full and Mamba), [1]
/// accessed first.
fn two_host_leaves(host_evict_first: bool) -> (UnifiedTreeCore<Vec<i64>>, NodeId, NodeId) {
    let mut tc = write_back_core();
    tc.configure_cache_residency(residency(false, host_evict_first));
    tc.insert(&insert_params_mamba(&vec![1], &[10], Some(7)));
    tc.insert(&insert_params_mamba(&vec![2], &[20], Some(8)));
    let a = node_of(&mut tc, vec![1]);
    let b = node_of(&mut tc, vec![2]);
    for (node, full_host, mamba_host) in [(a, 100, 70), (b, 200, 80)] {
        commit_host_copy(&mut tc, node, full_host, mamba_host);
        tc.evict_device_leaf(node, /* is_write_back = */ true)
            .unwrap();
        assert!(tc.arena.node(idx(&tc, node)).evicted());
    }
    (tc, a, b)
}

fn first_host_victim(
    tc: &mut UnifiedTreeCore<Vec<i64>>,
    component: ComponentType,
    a: NodeId,
    b: NodeId,
) -> NodeId {
    let step = tc.drive_host_eviction(component, /* num_tokens = */ 1);
    assert_eq!(step.tracker[&component], 1);
    match (tc.arena.resolve(a).is_ok(), tc.arena.resolve(b).is_ok()) {
        (false, true) => a,
        (true, false) => b,
        other => panic!("expected one host leaf evicted, got {other:?}"),
    }
}

#[test]
fn host_evict_first_ranks_a_marked_leaf_ahead_of_the_lru() {
    for component in [FULL, MAMBA] {
        let (mut tc, a, b) = two_host_leaves(false);
        let lru_victim = first_host_victim(&mut tc, component, a, b);
        let other = if lru_victim == a { b } else { a };

        // Off: marking is a no-op.
        let (mut tc, a, b) = two_host_leaves(false);
        assert_eq!(tc.mark_host_evict_first(vec![other]), 0);
        assert_eq!(first_host_victim(&mut tc, component, a, b), lru_victim);

        let (mut tc, a, b) = two_host_leaves(true);
        assert_eq!(tc.mark_host_evict_first(vec![other]), 1);
        assert_eq!(
            first_host_victim(&mut tc, component, a, b),
            other,
            "{component:?}"
        );
        let stats = tc.cache_residency_stats();
        assert_eq!(stats.host_evict_first_marked, 1);
        if component == FULL {
            assert_eq!(stats.host_evict_first_full_tokens, 1);
        } else {
            assert_eq!(stats.host_evict_first_mamba_states, 1);
        }
        tc.sanity_check(&[], &[]);
    }
}

#[test]
fn an_access_clears_the_host_evict_first_mark() {
    let (mut tc, a, b) = two_host_leaves(true);
    // Mark both, then a match of [2] re-stamps it: only [1] stays marked.
    assert_eq!(tc.mark_host_evict_first(vec![a, b]), 2);
    tc.match_prefix(&match_params(&vec![2]));
    assert!(tc.is_host_evict_first_(tc.arena.node(idx(&tc, a))));
    assert!(!tc.is_host_evict_first_(tc.arena.node(idx(&tc, b))));
    assert_eq!(first_host_victim(&mut tc, FULL, a, b), a);
}

#[test]
fn host_evict_first_climbs_only_through_fully_marked_parents() {
    let mut tc = write_back_core();
    tc.configure_cache_residency(residency(false, true));
    tc.insert(&insert_params_mamba(&vec![1], &[10], Some(7)));
    tc.insert(&insert_params_mamba(&vec![1, 2], &[10, 11], Some(8)));
    tc.insert(&insert_params_mamba(&vec![1, 3], &[10, 12], Some(9)));
    let parent = node_of(&mut tc, vec![1]);
    let left = node_of(&mut tc, vec![1, 2]);
    let right = node_of(&mut tc, vec![1, 3]);
    let marked = |tc: &UnifiedTreeCore<Vec<i64>>, id: NodeId| {
        tc.is_host_evict_first_(tc.arena.node(tc.arena.resolve(id).unwrap()))
    };

    // A live (locked) branch is never marked, and nothing above it is.
    let lock = tc
        .inc_lock_ref(right, ComponentSet::EMPTY)
        .unwrap()
        .to_dec_params();
    assert_eq!(tc.mark_host_evict_first(vec![right]), 0);
    assert!(!marked(&tc, right));
    tc.dec_lock_ref(right, &lock, false).unwrap();

    assert_eq!(tc.mark_host_evict_first(vec![left]), 1);
    assert!(marked(&tc, left) && !marked(&tc, parent));
    assert_eq!(tc.mark_host_evict_first(vec![right]), 2);
    assert!(marked(&tc, right) && marked(&tc, parent));
    // A repeated mark adds nothing.
    assert_eq!(tc.mark_host_evict_first(vec![left, right]), 0);
    assert_eq!(tc.cache_residency_stats().host_evict_first_marked, 3);
    tc.sanity_check(&[], &[]);
}

#[test]
fn a_mark_on_a_device_node_applies_once_it_is_demoted() {
    let mut tc = write_back_core();
    tc.configure_cache_residency(residency(false, true));
    tc.insert(&insert_params_mamba(&vec![1], &[10], Some(7)));
    tc.insert(&insert_params_mamba(&vec![2], &[20], Some(8)));
    let a = node_of(&mut tc, vec![1]);
    let b = node_of(&mut tc, vec![2]);
    // [2] is the newer state; mark it while still on device, then demote both,
    // [2] last, so stock order would put its host state at the MRU end.
    assert_eq!(tc.mark_host_evict_first(vec![b]), 1);
    for (node, full_host, mamba_host) in [(a, 100, 70), (b, 200, 80)] {
        commit_host_copy(&mut tc, node, full_host, mamba_host);
        tc.evict_device_leaf(node, /* is_write_back = */ true)
            .unwrap();
    }
    assert_eq!(
        lru_order(tc.host_lru_list(MAMBA)).last(),
        Some(&idx(&tc, b))
    );
    assert_eq!(first_host_victim(&mut tc, MAMBA, a, b), b);
    tc.sanity_check(&[], &[]);
}

#[test]
fn host_evict_first_skips_stale_ids() {
    let (mut tc, a, b) = two_host_leaves(true);
    let gone = first_host_victim(&mut tc, FULL, a, b);
    assert_eq!(tc.mark_host_evict_first(vec![gone, 1_000_000]), 0);
    tc.sanity_check(&[], &[]);
}

#[test]
fn move_to_lru_puts_a_member_at_the_eviction_end() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a, b, c] = chain::<3>(&mut tc);
    let lru = tc.host_lru_list_mut(MAMBA);
    for node in [a, b, c] {
        lru.insert_mru(node);
    }
    lru.move_to_lru(c);
    assert_eq!(lru_order(tc.host_lru_list(MAMBA)), vec![b, a, c]);
    tc.host_lru_list_mut(MAMBA).move_to_lru(c);
    assert_eq!(lru_order(tc.host_lru_list(MAMBA)), vec![b, a, c]);
}
