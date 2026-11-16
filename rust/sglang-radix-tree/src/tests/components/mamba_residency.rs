//! Host evict-first marks (off by default; the "off" cases pin the stock order).

use super::*;
use crate::unified_tree_core::CacheResidencyOptions;
use tch::{Device, Kind};

fn residency(host_evict_first: bool) -> CacheResidencyOptions {
    CacheResidencyOptions { host_evict_first }
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

// ---- host_evict_first ----

/// Two finished root children [1] and [2] on host only (Full and Mamba), [1]
/// accessed first.
fn two_host_leaves(host_evict_first: bool) -> (UnifiedTreeCore<Vec<i64>>, NodeId, NodeId) {
    let mut tc = write_back_core();
    tc.configure_cache_residency(residency(host_evict_first));
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
    tc.configure_cache_residency(residency(true));
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
    tc.configure_cache_residency(residency(true));
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
