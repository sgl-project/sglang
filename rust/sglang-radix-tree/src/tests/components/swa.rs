use super::*;
use crate::components::{FULL, MAMBA, SWA};
use crate::test_utils::{accumulate_step, action_kinds};
use crate::unified_tree_core::CacheInitParams;

#[test]
fn component_type_is_swa() {
    let swa = SwaComponent::new(&swa_params());
    assert_eq!(
        <SwaComponent as TreeComponent<Vec<i64>>>::component_type(&swa),
        SWA
    );
}

fn swa_params() -> CacheInitParams {
    CacheInitParams {
        swa_sliding_window_size: Some(4096),
        ..Default::default()
    }
}

#[test]
#[should_panic(expected = "requires swa_sliding_window_size")]
fn new_panics_without_a_sliding_window_size() {
    SwaComponent::new(&CacheInitParams::default());
}

#[test]
fn new_stores_the_sliding_window_size() {
    let params = CacheInitParams {
        swa_sliding_window_size: Some(4096),
        ..Default::default()
    };
    assert_eq!(SwaComponent::new(&params).sliding_window_size, 4096);
}

#[test]
fn construction_with_full_and_swa_locks_both_roots() {
    let tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(swa_params(), vec![FULL, SWA]);
    let root = tc.arena.root();
    let root_node = tc.arena.node(root);
    assert_eq!(root_node.values[FULL.idx()].lock_ref, 1);
    assert_eq!(root_node.values[SWA.idx()].lock_ref, 1);
    assert_eq!(root_node.values[MAMBA.idx()].lock_ref, 0);
}

#[test]
fn swa_sizes_read_zero_on_a_fresh_tree() {
    let tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(swa_params(), vec![FULL, SWA]);
    assert_eq!(tc.swa_evictable_size(), 0);
    assert_eq!(tc.swa_protected_size(), 0);
}

#[test]
fn node_has_component_data_tracks_each_slot() {
    let mut tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(swa_params(), vec![FULL, SWA]);
    let root = tc.arena.root();
    let node = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let swa = SwaComponent::new(&swa_params());
    assert!(!crate::components::node_has_component_data(
        &tc.arena,
        node,
        SWA,
        EvictLayer::Device
    ));
    assert!(!crate::components::node_has_component_data(
        &tc.arena,
        node,
        SWA,
        EvictLayer::Host
    ));

    tc.arena
        .set_device_value(node, SWA, Tensor::from_slice(&[10i64]));
    assert!(crate::components::node_has_component_data(
        &tc.arena,
        node,
        SWA,
        EvictLayer::Device
    ));
    assert!(!crate::components::node_has_component_data(
        &tc.arena,
        node,
        SWA,
        EvictLayer::Host
    ));

    tc.arena
        .set_host_value(node, SWA, Tensor::from_slice(&[20i64]));
    assert!(crate::components::node_has_component_data(
        &tc.arena,
        node,
        SWA,
        EvictLayer::Host
    ));
}

// A [Full, Swa] core with the given sliding window and page size.
fn swa_core(window: usize, page_size: usize) -> UnifiedTreeCore<Vec<i64>> {
    UnifiedTreeCore::new(
        CacheInitParams {
            page_size,
            ..swa_params_with_window(window)
        },
        vec![FULL, SWA],
    )
}

fn swa_params_with_window(window: usize) -> CacheInitParams {
    CacheInitParams {
        swa_sliding_window_size: Some(window),
        ..Default::default()
    }
}

fn swa_component(window: usize) -> SwaComponent {
    SwaComponent::new(&swa_params_with_window(window))
}

// A chain of single-atom children under the default root; returns the node ids.
fn chain<const N: usize>(tc: &mut UnifiedTreeCore<Vec<i64>>) -> [NodeIdx_; N] {
    let mut parent = tc.arena.root();
    let mut nodes = [NodeIdx_(0); N];
    for (i, node) in nodes.iter_mut().enumerate() {
        let id = tc
            .arena
            .alloc_child(
                parent,
                /* key = */ vec![i as i64 + 1],
                /* priority = */ 0,
                /* extra_key = */ None,
            )
            .unwrap();
        *node = id;
        parent = id;
    }
    nodes
}

fn set_swa_device(tc: &mut UnifiedTreeCore<Vec<i64>>, node: NodeIdx_) {
    let len = tc.arena.node(node).key.atom_len();
    tc.arena
        .set_device_value(node, SWA, Tensor::from_slice(&vec![0i64; len]));
}

fn set_swa_host(tc: &mut UnifiedTreeCore<Vec<i64>>, node: NodeIdx_) {
    let len = tc.arena.node(node).key.atom_len();
    tc.arena
        .set_host_value(node, SWA, Tensor::from_slice(&vec![0i64; len]));
}

fn node_swa_uuid(tc: &UnifiedTreeCore<Vec<i64>>, node: NodeIdx_) -> Option<i64> {
    tc.arena.node(node).swa_uuid
}

fn node_swa_host_uuid(tc: &UnifiedTreeCore<Vec<i64>>, node: NodeIdx_) -> Option<i64> {
    tc.arena.node(node).swa_host_uuid
}

// Give `node` an SWA device value via the auxiliary store (sizes + LRU stamped).
fn store_swa_device(tc: &mut UnifiedTreeCore<Vec<i64>>, node: NodeIdx_) {
    let len = tc.arena.node(node).key.atom_len();
    tc.set_component_device_value(
        tc.arena.node(node).id,
        SWA,
        Tensor::from_slice(&vec![0i64; len]),
    );
}

#[test]
fn match_validator_accepts_valued_nodes_before_any_gap() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a, b] = chain(&mut tc);
    set_swa_device(&mut tc, a);
    set_swa_device(&mut tc, b);
    // The window starts unbounded: a 2-atom valued span validates under a
    // window of 4 because no gap has been seen yet.
    let mut validator =
        swa_component(4).create_match_validator(&tc, /* match_device_only = */ true);
    assert!(validator(&tc, a));
    assert!(validator(&tc, b));
}

#[test]
fn match_validator_gap_resets_until_the_window_is_reached() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, t, c, d, e] = chain(&mut tc);
    set_swa_device(&mut tc, a);
    set_swa_device(&mut tc, c);
    set_swa_device(&mut tc, d);
    set_swa_device(&mut tc, e);
    let mut validator =
        swa_component(2).create_match_validator(&tc, /* match_device_only = */ true);
    assert!(validator(&tc, a));
    // The tombstone resets the run; below the window stays invalid, the
    // exact window boundary revalidates, and beyond it stays valid.
    assert!(!validator(&tc, t));
    assert!(!validator(&tc, c));
    assert!(validator(&tc, d));
    assert!(validator(&tc, e));
}

#[test]
fn match_validator_window_larger_than_the_remaining_span_never_revalidates() {
    let mut tc = swa_core(/* window = */ 3, /* page_size = */ 1);
    let [_a, t, c, d] = chain(&mut tc);
    set_swa_device(&mut tc, c);
    set_swa_device(&mut tc, d);
    let mut validator =
        swa_component(3).create_match_validator(&tc, /* match_device_only = */ true);
    assert!(!validator(&tc, t));
    assert!(!validator(&tc, c));
    assert!(!validator(&tc, d));
}

#[test]
fn match_validator_device_only_treats_host_only_swa_as_a_gap() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, h, c] = chain(&mut tc);
    set_swa_device(&mut tc, a);
    set_swa_host(&mut tc, h);
    set_swa_device(&mut tc, c);
    let mut validator =
        swa_component(2).create_match_validator(&tc, /* match_device_only = */ true);
    assert!(validator(&tc, a));
    assert!(!validator(&tc, h));
    assert!(!validator(&tc, c));
}

#[test]
fn match_validator_host_backed_nodes_extend_the_window() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [t, h, c] = chain(&mut tc);
    set_swa_host(&mut tc, h);
    set_swa_device(&mut tc, c);
    // Without match_device_only, a host-backed node is not a gap: it
    // counts toward the window like a device-backed one.
    let mut validator =
        swa_component(2).create_match_validator(&tc, /* match_device_only = */ false);
    assert!(!validator(&tc, t));
    assert!(!validator(&tc, h));
    assert!(validator(&tc, c));
}

#[test]
fn match_validator_hicache_accepts_live_or_backuped_swa_tombstones() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    tc.set_hicache_enabled();
    let [live, backuped, dead] = chain(&mut tc);
    tc.arena
        .set_device_value(live, FULL, Tensor::from_slice(&[0i64]));
    tc.arena
        .set_host_value(backuped, FULL, Tensor::from_slice(&[0i64]));
    let mut validator =
        swa_component(2).create_match_validator(&tc, /* match_device_only = */ true);
    assert!(validator(&tc, live));
    assert!(validator(&tc, backuped));
    assert!(!validator(&tc, dead));
}

#[test]
fn match_validator_without_hicache_rejects_every_swa_tombstone() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [live, backuped, dead] = chain(&mut tc);
    tc.arena
        .set_device_value(live, FULL, Tensor::from_slice(&[0i64]));
    tc.arena
        .set_host_value(backuped, FULL, Tensor::from_slice(&[0i64]));
    let mut validator =
        swa_component(2).create_match_validator(&tc, /* match_device_only = */ true);
    assert!(!validator(&tc, live));
    assert!(!validator(&tc, backuped));
    assert!(!validator(&tc, dead));
}

#[test]
fn match_validator_hicache_with_a_host_pool_rejects_swa_tombstones() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    tc.set_hicache_enabled();
    let [live] = chain(&mut tc);
    tc.arena
        .set_device_value(live, FULL, Tensor::from_slice(&[0i64]));
    // A wired host SWA pool means tombstones must gate the match again.
    tc.set_has_swa_host_pool();
    let swa = SwaComponent {
        sliding_window_size: 2,
    };
    let mut validator = <SwaComponent as TreeComponent<Vec<i64>>>::create_match_validator(
        &swa, &tc, /* match_device_only = */ true,
    );
    assert!(!validator(&tc, live));
}

#[test]
fn match_validator_hicache_tombstone_acceptance_still_resets_the_window() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    tc.set_hicache_enabled();
    let [live, c] = chain(&mut tc);
    tc.arena
        .set_device_value(live, FULL, Tensor::from_slice(&[0i64]));
    set_swa_device(&mut tc, c);
    let mut validator =
        swa_component(2).create_match_validator(&tc, /* match_device_only = */ true);
    // The accepted tombstone still zeroes the run: the next valued node
    // sits below the window.
    assert!(validator(&tc, live));
    assert!(!validator(&tc, c));
}

// The SWA LRU order, MRU to LRU.
fn swa_lru_order(tc: &UnifiedTreeCore<Vec<i64>>) -> Vec<NodeIdx_> {
    tc.device_lru_list(SWA).iter().collect()
}

#[test]
fn refresh_lru_walkdown_is_a_noop() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let root = tc.arena.root();
    let [a, b] = chain(&mut tc);
    set_swa_device(&mut tc, a);
    set_swa_device(&mut tc, b);
    tc.device_lru_list_mut(SWA).insert_mru(a);
    tc.device_lru_list_mut(SWA).insert_mru(b);
    swa_component(2).refresh_lru(&mut tc, LRURefreshPhase::Walkdown, a);
    assert_eq!(swa_lru_order(&tc), vec![b, a]);
}

#[test]
fn refresh_lru_match_end_reranks_the_window_run_deepest_first() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let root = tc.arena.root();
    let [a, b, c, d] = chain(&mut tc);
    for node in [d, c, b, a] {
        set_swa_device(&mut tc, node);
        tc.device_lru_list_mut(SWA).insert_mru(node);
    }
    // The walk window is sliding_window_size + page_size = 3: d, c, b are
    // re-ranked deepest first; a stays beyond the window.
    swa_component(2).refresh_lru(&mut tc, LRURefreshPhase::MatchEnd, d);
    assert_eq!(swa_lru_order(&tc), vec![d, c, b, a]);
}

#[test]
fn refresh_lru_insert_end_matches_the_match_end_walk() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let root = tc.arena.root();
    let [a, b, c, d] = chain(&mut tc);
    for node in [d, c, b, a] {
        set_swa_device(&mut tc, node);
        tc.device_lru_list_mut(SWA).insert_mru(node);
    }
    swa_component(2).refresh_lru(&mut tc, LRURefreshPhase::InsertEnd, d);
    assert_eq!(swa_lru_order(&tc), vec![d, c, b, a]);
}

#[test]
fn refresh_lru_window_walk_skips_tombstones_but_counts_their_span() {
    let mut tc = swa_core(/* window = */ 1, /* page_size = */ 1);
    let root = tc.arena.root();
    let [a, _t, c] = chain(&mut tc);
    let s = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![9],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    for node in [c, a, s] {
        set_swa_device(&mut tc, node);
        tc.device_lru_list_mut(SWA).insert_mru(node);
    }
    // The walk window is 2: the unlisted tombstone t is skipped but its
    // atom consumes the window, so a is never re-ranked.
    swa_component(1).refresh_lru(&mut tc, LRURefreshPhase::MatchEnd, c);
    assert_eq!(swa_lru_order(&tc), vec![c, s, a]);
}

// Finalize `best` against an otherwise-empty match result carrying a prior
// SWA host hit.
fn finalize(
    tc: &UnifiedTreeCore<Vec<i64>>,
    swa: &SwaComponent,
    best: NodeIdx_,
    prior_swa_host_hit: usize,
) -> MatchResult {
    swa.finalize_match_result_in_tree_core(
        tc,
        MatchResult {
            best_match_node_id: tc.arena.node(best).id,
            swa_host_hit_length: prior_swa_host_hit,
            ..tc.empty_match_result()
        },
        &MatchPrefixParams {
            key: &Vec::new(),
            namespace: Default::default(),
        },
        &[],
        0,
    )
}

#[test]
fn finalize_without_host_chunks_leaves_the_result_unchanged() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a, b] = chain(&mut tc);
    set_swa_device(&mut tc, a);
    set_swa_device(&mut tc, b);
    let out = finalize(&tc, &swa_component(4), b, /* prior = */ 0);
    assert_eq!(out.swa_host_hit_length, 0);
    assert_eq!(out.best_match_node_id, tc.arena.node(b).id);
}

#[test]
fn finalize_sums_swa_host_chunks_within_the_window() {
    let mut tc = swa_core(/* window = */ 5, /* page_size = */ 1);
    let [a, h, b, c] = chain(&mut tc);
    set_swa_host(&mut tc, a);
    set_swa_host(&mut tc, h);
    set_swa_device(&mut tc, b);
    set_swa_device(&mut tc, c);
    // From c up: device 1 + 1, then host 1 + 1 — all inside the window of 5.
    let out = finalize(&tc, &swa_component(5), c, /* prior = */ 0);
    assert_eq!(out.swa_host_hit_length, 2);
}

#[test]
fn finalize_stops_at_the_window_before_higher_host_chunks() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [h, b, c] = chain(&mut tc);
    set_swa_host(&mut tc, h);
    set_swa_device(&mut tc, b);
    set_swa_device(&mut tc, c);
    // The device span alone covers the window of 2: the host chunk above
    // the boundary is never counted.
    let out = finalize(&tc, &swa_component(2), c, /* prior = */ 0);
    assert_eq!(out.swa_host_hit_length, 0);
}

#[test]
fn finalize_counts_the_straddling_host_chunk_in_full() {
    let mut tc = swa_core(/* window = */ 3, /* page_size = */ 1);
    let root = tc.arena.root();
    let h = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2, 3, 4],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let c = tc
        .arena
        .alloc_child(
            h,
            /* key = */ vec![5, 6],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    set_swa_host(&mut tc, h);
    set_swa_device(&mut tc, c);
    // The host chunk straddles the window boundary (2 of its 4 tokens are
    // in-window) and is counted in full, uncapped.
    let out = finalize(&tc, &swa_component(3), c, /* prior = */ 0);
    assert_eq!(out.swa_host_hit_length, 4);
}

#[test]
fn finalize_breaks_at_an_swa_gap() {
    let mut tc = swa_core(/* window = */ 5, /* page_size = */ 1);
    let [h, _t, c] = chain(&mut tc);
    set_swa_host(&mut tc, h);
    set_swa_device(&mut tc, c);
    // The tombstone between c and h ends the walk: the host chunk above
    // the gap is unreachable.
    let out = finalize(&tc, &swa_component(5), c, /* prior = */ 0);
    assert_eq!(out.swa_host_hit_length, 0);
}

#[test]
fn finalize_keeps_a_larger_existing_swa_host_hit() {
    let mut tc = swa_core(/* window = */ 5, /* page_size = */ 1);
    let [h, c] = chain(&mut tc);
    set_swa_host(&mut tc, h);
    set_swa_device(&mut tc, c);
    let out = finalize(&tc, &swa_component(5), c, /* prior = */ 100);
    assert_eq!(out.swa_host_hit_length, 100);
}

#[test]
fn finalize_overrides_a_smaller_existing_swa_host_hit() {
    let mut tc = swa_core(/* window = */ 5, /* page_size = */ 1);
    let [h, hh, c] = chain(&mut tc);
    set_swa_host(&mut tc, h);
    set_swa_host(&mut tc, hh);
    set_swa_device(&mut tc, c);
    let out = finalize(&tc, &swa_component(5), c, /* prior = */ 1);
    assert_eq!(out.swa_host_hit_length, 2);
}

#[test]
fn finalize_walk_stops_at_the_root() {
    let mut tc = swa_core(/* window = */ 5, /* page_size = */ 1);
    let lora = tc.arena.root();
    let c = tc
        .arena
        .alloc_child(
            lora,
            /* key = */ vec![1],
            /* priority = */ 0,
            Some("lora-1"),
        )
        .unwrap();
    set_swa_device(&mut tc, c);
    // A host value on the root itself is never counted: the walk ends there.
    tc.arena
        .node_mut(lora)
        .state_mut_(ValueSlotIdx::host(SWA))
        .value = Some(Tensor::from_slice(&[0i64]));
    let out = finalize(&tc, &swa_component(5), c, /* prior = */ 0);
    assert_eq!(out.swa_host_hit_length, 0);
}

fn insert_params_swa<'k>(
    key: &'k Vec<i64>,
    value: &[i64],
    prev_prefix_len: usize,
    swa_evicted_seqlen: usize,
) -> InsertParams<'k, Vec<i64>> {
    InsertParams {
        key,
        namespace: Default::default(),
        value: Tensor::from_slice(value),
        mamba_value: None,
        prev_prefix_len,
        swa_evicted_seqlen,
        chunked: false,
        priority: 0,
        track_adopted_ranges: false,
    }
}

// The child of `node` along the edge keyed by `page` (default namespace).
fn child_of(tc: &UnifiedTreeCore<Vec<i64>>, node: NodeIdx_, page: &[i64]) -> NodeIdx_ {
    tc.arena
        .child_on_page(node, /* extra_key = */ None, page)
        .expect("child on page")
}

// Tombstone a leaf's FULL device value the way eviction leaves it.
fn evict_full(tc: &mut UnifiedTreeCore<Vec<i64>>, leaf: NodeIdx_, remaining_size: usize) {
    let _ = tc.arena.take_device_value(leaf, FULL);
    tc.component_state_mut(FULL).evictable_size = remaining_size;
    tc.evictable_device_leaves.discard(leaf);
}

#[test]
fn insert_new_leaf_in_window_emits_one_leaf_rebuild() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3],
        &[10, 11, 12],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 0,
    ));
    assert_eq!(result.prefix_len, 0);
    let root = tc.arena.root();
    let leaf = child_of(&tc, root, &[1]);
    let [
        CacheAction::SwaRebuild {
            node_id,
            source_value,
        },
    ] = result.cache_actions.as_slice()
    else {
        panic!("expected one SwaRebuild action");
    };
    assert_eq!(*node_id, tc.arena.node(leaf).id);
    assert!(source_value.equal(&Tensor::from_slice(&[10i64, 11, 12])));
    // The rebuild is deferred to apply time: the leaf is still an SWA tombstone.
    assert!(!tc.arena.has_device_value(leaf, SWA));
}

#[test]
fn insert_new_leaf_straddling_the_boundary_splits_and_rebuilds_the_tail() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4],
        &[10, 11, 12, 13],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 2,
    ));
    let root = tc.arena.root();
    let parent = child_of(&tc, root, &[1]);
    let child = child_of(&tc, parent, &[3]);
    assert_eq!(tc.arena.node(parent).key, vec![1, 2]);
    assert_eq!(tc.arena.node(child).key, vec![3, 4]);
    // The out-of-window parent is an SWA tombstone holding its own Full span.
    assert!(!tc.arena.has_device_value(parent, SWA));
    assert!(
        tc.arena
            .device_value(parent, FULL)
            .equal(&Tensor::from_slice(&[10i64, 11]))
    );
    let [
        CacheAction::SwaRebuild {
            node_id,
            source_value,
        },
    ] = result.cache_actions.as_slice()
    else {
        panic!("expected one SwaRebuild action");
    };
    assert_eq!(*node_id, tc.arena.node(child).id);
    assert!(source_value.equal(&Tensor::from_slice(&[12i64, 13])));
}

#[test]
fn insert_new_leaf_outside_the_window_stays_a_tombstone() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    // The boundary equals the leaf end: no split, no rebuild.
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3],
        &[10, 11, 12],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 3,
    ));
    assert!(result.cache_actions.is_empty());
    let root = tc.arena.root();
    let leaf = child_of(&tc, root, &[1]);
    assert_eq!(tc.arena.node(leaf).key, vec![1, 2, 3]);
    assert!(!tc.arena.has_device_value(leaf, SWA));
}

#[test]
fn insert_long_leaf_caps_the_window_and_rebuilds_the_older_prefix_first() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4, 5, 6],
        &[10, 11, 12, 13, 14, 15],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 0,
    ));
    let root = tc.arena.root();
    let capped = child_of(&tc, root, &[1]);
    let tail = child_of(&tc, capped, &[5]);
    assert_eq!(tc.arena.node(capped).key, vec![1, 2, 3, 4]);
    assert_eq!(tc.arena.node(tail).key, vec![5, 6]);
    let [
        CacheAction::SwaRebuild {
            node_id: first_id,
            source_value: first_value,
        },
        CacheAction::SwaRebuild {
            node_id: second_id,
            source_value: second_value,
        },
    ] = result.cache_actions.as_slice()
    else {
        panic!("expected two SwaRebuild actions");
    };
    assert_eq!(*first_id, tc.arena.node(capped).id);
    assert!(first_value.equal(&Tensor::from_slice(&[10i64, 11, 12, 13])));
    assert_eq!(*second_id, tc.arena.node(tail).id);
    assert!(second_value.equal(&Tensor::from_slice(&[14i64, 15])));
}

#[test]
fn cap_split_uses_the_page_rounded_window() {
    let mut tc = swa_core(/* window = */ 3, /* page_size = */ 2);
    tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4, 5, 6],
        &[10, 11, 12, 13, 14, 15],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 0,
    ));
    let root = tc.arena.root();
    let capped = child_of(&tc, root, &[1, 2]);
    // tail_size rounds the window of 3 up to 2 pages: the tail keeps 4
    // atoms, the parent 2.
    assert_eq!(tc.arena.node(capped).key, vec![1, 2]);
    let tail = child_of(&tc, capped, &[3, 4]);
    assert_eq!(tc.arena.node(tail).key, vec![3, 4, 5, 6]);
}

#[test]
fn insert_overlap_with_live_swa_frees_the_whole_duplicate() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[10, 11, 12], 0, 0));
    let root = tc.arena.root();
    let leaf = child_of(&tc, root, &[1]);
    tc.set_component_device_value(
        tc.arena.node(leaf).id,
        SWA,
        Tensor::from_slice(&[50i64, 51, 52]),
    );
    let result = tc.insert(&insert_params_swa(&vec![1, 2, 3], &[20, 21, 22], 0, 0));
    assert_eq!(result.prefix_len, 3);
    let [CacheAction::FreeDeviceKV(freed)] = result.cache_actions.as_slice() else {
        panic!("expected one FreeDeviceKV action");
    };
    assert_eq!(freed.len(), 1);
    assert!(freed[0].equal(&Tensor::from_slice(&[20i64, 21, 22])));
    // The node keeps its original Full and SWA values.
    assert!(
        tc.arena
            .device_value(leaf, FULL)
            .equal(&Tensor::from_slice(&[10i64, 11, 12]))
    );
    assert!(
        tc.arena
            .device_value(leaf, SWA)
            .equal(&Tensor::from_slice(&[50i64, 51, 52]))
    );
}

#[test]
fn insert_overlap_recovers_a_tombstone_inside_the_window() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[10, 11, 12], 0, 0));
    let root = tc.arena.root();
    let leaf = child_of(&tc, root, &[1]);
    let result = tc.insert(&InsertParams {
        track_adopted_ranges: true,
        ..insert_params_swa(&vec![1, 2, 3], &[20, 21, 22], 0, 0)
    });
    // SWA consumed the whole slice: the node adopts the fresh Full KV and
    // the old Full is freed instead of the duplicates.
    assert!(
        tc.arena
            .device_value(leaf, FULL)
            .equal(&Tensor::from_slice(&[20i64, 21, 22]))
    );
    let [
        CacheAction::FreeDeviceKVFullOnly(freed),
        CacheAction::SwaRebuild {
            node_id,
            source_value,
        },
    ] = result.cache_actions.as_slice()
    else {
        panic!("expected FreeDeviceKVFullOnly then SwaRebuild");
    };
    assert!(freed[0].equal(&Tensor::from_slice(&[10i64, 11, 12])));
    assert_eq!(*node_id, tc.arena.node(leaf).id);
    assert!(source_value.equal(&Tensor::from_slice(&[20i64, 21, 22])));
    let adopted = result.adopted_ranges.as_ref().unwrap();
    assert_eq!(adopted[&FULL], [(0, 3)]);
    assert_eq!(adopted[&SWA], [(0, 3)]);
}

#[test]
#[should_panic(expected = "tombstone Swa lock_ref should be 0, node")]
fn insert_overlap_panics_on_a_locked_swa_tombstone() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[10, 11, 12], 0, 0));
    let root = tc.arena.root();
    let leaf = child_of(&tc, root, &[1]);
    // The rebuild is deferred, so the leaf is still an SWA tombstone; a raw
    // lock on it breaks the tombstones-are-unlocked contract.
    tc.arena
        .node_mut(leaf)
        .set_lock_ref_(ValueSlotIdx::device(SWA), 1);
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[20, 21, 22], 0, 0));
}

#[test]
fn insert_overlap_with_a_locked_full_emits_the_recover_action() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[10, 11, 12], 0, 0));
    let root = tc.arena.root();
    let leaf = child_of(&tc, root, &[1]);
    tc.arena
        .node_mut(leaf)
        .set_lock_ref_(ValueSlotIdx::device(FULL), 1);
    let result = tc.insert(&InsertParams {
        track_adopted_ranges: true,
        ..insert_params_swa(&vec![1, 2, 3], &[20, 21, 22], 0, 0)
    });
    // The locked Full stays on the node; the cache resolves the recover action.
    assert!(
        tc.arena
            .device_value(leaf, FULL)
            .equal(&Tensor::from_slice(&[10i64, 11, 12]))
    );
    let [
        CacheAction::RecoverSwaWithLockedFull {
            node_id,
            kept_full,
            incoming_full,
        },
    ] = result.cache_actions.as_slice()
    else {
        panic!("expected one RecoverSwaWithLockedFull action");
    };
    assert_eq!(*node_id, tc.arena.node(leaf).id);
    assert!(kept_full.equal(&Tensor::from_slice(&[10i64, 11, 12])));
    assert!(incoming_full.equal(&Tensor::from_slice(&[20i64, 21, 22])));
    let adopted = result.adopted_ranges.as_ref().unwrap();
    assert!(!adopted.contains_key(&FULL));
    assert_eq!(adopted[&SWA], [(0, 3)]);
}

#[test]
fn insert_overlap_straddling_the_boundary_splits_and_recovers_the_tail() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4],
        &[10, 11, 12, 13],
        0,
        0,
    ));
    let root = tc.arena.root();
    let node = child_of(&tc, root, &[1]);
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4],
        &[20, 21, 22, 23],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 2,
    ));
    assert_eq!(result.prefix_len, 4);
    // The node split at the boundary: the parent keeps the old
    // out-of-window Full span, the tail adopts the fresh KV.
    let parent = child_of(&tc, root, &[1]);
    assert_eq!(tc.arena.node(parent).key, vec![1, 2]);
    assert_eq!(tc.arena.node(node).key, vec![3, 4]);
    assert!(
        tc.arena
            .device_value(parent, FULL)
            .equal(&Tensor::from_slice(&[10i64, 11]))
    );
    assert!(
        tc.arena
            .device_value(node, FULL)
            .equal(&Tensor::from_slice(&[22i64, 23]))
    );
    let [
        CacheAction::FreeDeviceKVFullOnly(old_tail),
        CacheAction::SwaRebuild {
            node_id,
            source_value,
        },
        CacheAction::FreeDeviceKV(duplicates),
    ] = result.cache_actions.as_slice()
    else {
        panic!(
            "expected FreeDeviceKVFullOnly, SwaRebuild, FreeDeviceKV, got {:?}",
            action_kinds(&result.cache_actions)
        );
    };
    assert!(old_tail[0].equal(&Tensor::from_slice(&[12i64, 13])));
    assert_eq!(*node_id, tc.arena.node(node).id);
    assert!(source_value.equal(&Tensor::from_slice(&[22i64, 23])));
    // Only the out-of-window head is duplicate; the consumed tail is not re-freed.
    assert!(duplicates[0].equal(&Tensor::from_slice(&[20i64, 21])));
}

#[test]
fn insert_overlap_straddling_with_a_locked_full_defers_the_tail() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4],
        &[10, 11, 12, 13],
        0,
        0,
    ));
    let root = tc.arena.root();
    let node = child_of(&tc, root, &[1]);
    tc.arena
        .node_mut(node)
        .set_lock_ref_(ValueSlotIdx::device(FULL), 1);
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4],
        &[20, 21, 22, 23],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 2,
    ));
    // The locked tail keeps its Full value; only the recover action crosses.
    assert!(
        tc.arena
            .device_value(node, FULL)
            .equal(&Tensor::from_slice(&[12i64, 13]))
    );
    let [
        CacheAction::RecoverSwaWithLockedFull {
            node_id,
            kept_full,
            incoming_full,
        },
        CacheAction::FreeDeviceKV(duplicates),
    ] = result.cache_actions.as_slice()
    else {
        panic!("expected RecoverSwaWithLockedFull then FreeDeviceKV");
    };
    assert_eq!(*node_id, tc.arena.node(node).id);
    assert!(kept_full.equal(&Tensor::from_slice(&[12i64, 13])));
    assert!(incoming_full.equal(&Tensor::from_slice(&[22i64, 23])));
    assert!(duplicates[0].equal(&Tensor::from_slice(&[20i64, 21])));
}

#[test]
fn insert_overlap_entirely_outside_the_window_is_all_duplicate() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[10, 11, 12], 0, 0));
    // The boundary sits at the node end: nothing consumed, no recovery.
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3],
        &[20, 21, 22],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 3,
    ));
    let [CacheAction::FreeDeviceKV(freed)] = result.cache_actions.as_slice() else {
        panic!("expected one FreeDeviceKV action");
    };
    assert!(freed[0].equal(&Tensor::from_slice(&[20i64, 21, 22])));
}

#[test]
fn insert_overlap_already_cached_prefix_skips_the_tombstone_recovery() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[10, 11, 12], 0, 0));
    // prev_prefix_len covers the node: no recovery and no duplicate free.
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3],
        &[20, 21, 22],
        /* prev_prefix_len = */ 3,
        /* swa_evicted_seqlen = */ 0,
    ));
    assert!(result.cache_actions.is_empty());
}

#[test]
fn insert_overlap_boundary_at_the_node_start_recovers_the_whole_node() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[10, 11, 12], 0, 0));
    tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4, 5],
        &[30, 31, 32, 13, 14],
        0,
        0,
    ));
    let root = tc.arena.root();
    let a = child_of(&tc, root, &[1]);
    let b = child_of(&tc, a, &[4]);
    tc.set_component_device_value(
        tc.arena.node(a).id,
        SWA,
        Tensor::from_slice(&[50i64, 51, 52]),
    );
    // The boundary lands exactly on b's start: full recovery, no split.
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4, 5],
        &[20, 21, 22, 23, 24],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 3,
    ));
    assert_eq!(tc.arena.node(b).key, vec![4, 5]);
    assert!(
        tc.arena
            .device_value(b, FULL)
            .equal(&Tensor::from_slice(&[23i64, 24]))
    );
    let [
        CacheAction::FreeDeviceKV(duplicates),
        CacheAction::FreeDeviceKVFullOnly(old_full),
        CacheAction::SwaRebuild {
            node_id,
            source_value,
        },
    ] = result.cache_actions.as_slice()
    else {
        panic!(
            "expected FreeDeviceKV, FreeDeviceKVFullOnly, SwaRebuild, got {:?}",
            action_kinds(&result.cache_actions)
        );
    };
    assert!(duplicates[0].equal(&Tensor::from_slice(&[20i64, 21, 22])));
    assert!(old_full[0].equal(&Tensor::from_slice(&[13i64, 14])));
    assert_eq!(*node_id, tc.arena.node(b).id);
    assert!(source_value.equal(&Tensor::from_slice(&[23i64, 24])));
}

#[test]
fn insert_overlap_straddling_a_second_level_node_recovers_the_tail() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[10, 11, 12], 0, 0));
    tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4, 5],
        &[10, 11, 12, 13, 14],
        0,
        0,
    ));
    let root = tc.arena.root();
    let a = child_of(&tc, root, &[1]);
    let b = child_of(&tc, a, &[4]);
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4, 5],
        &[20, 21, 22, 23, 24],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 4,
    ));
    assert_eq!(result.prefix_len, 5);
    // The boundary lands one atom into b (total_prefix_len 3): b splits at
    // its node-relative offset 1, not at the absolute seqlen.
    let p = child_of(&tc, a, &[4]);
    assert_eq!(tc.arena.node(p).key, vec![4]);
    assert_eq!(tc.arena.node(b).key, vec![5]);
    assert!(
        tc.arena
            .device_value(p, FULL)
            .equal(&Tensor::from_slice(&[13i64]))
    );
    assert!(
        tc.arena
            .device_value(b, FULL)
            .equal(&Tensor::from_slice(&[24i64]))
    );
    let [
        CacheAction::FreeDeviceKV(duplicates_head),
        CacheAction::FreeDeviceKVFullOnly(old_tail),
        CacheAction::SwaRebuild {
            node_id,
            source_value,
        },
        CacheAction::FreeDeviceKV(duplicates_tail),
    ] = result.cache_actions.as_slice()
    else {
        panic!(
            "expected FreeDeviceKV, FreeDeviceKVFullOnly, SwaRebuild, FreeDeviceKV, got {:?}",
            action_kinds(&result.cache_actions)
        );
    };
    assert!(duplicates_head[0].equal(&Tensor::from_slice(&[20i64, 21, 22])));
    assert!(old_tail[0].equal(&Tensor::from_slice(&[14i64])));
    assert_eq!(*node_id, tc.arena.node(b).id);
    assert!(source_value.equal(&Tensor::from_slice(&[24i64])));
    assert!(duplicates_tail[0].equal(&Tensor::from_slice(&[23i64])));
}

#[test]
fn insert_overlap_straddling_a_second_level_locked_node_defers_the_tail() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[10, 11, 12], 0, 0));
    tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4, 5],
        &[10, 11, 12, 13, 14],
        0,
        0,
    ));
    let root = tc.arena.root();
    let a = child_of(&tc, root, &[1]);
    let b = child_of(&tc, a, &[4]);
    tc.arena
        .node_mut(b)
        .set_lock_ref_(ValueSlotIdx::device(FULL), 1);
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4, 5],
        &[20, 21, 22, 23, 24],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 4,
    ));
    // The locked tail keeps its old Full value; recovery is deferred.
    let p = child_of(&tc, a, &[4]);
    assert_eq!(tc.arena.node(p).key, vec![4]);
    assert_eq!(tc.arena.node(b).key, vec![5]);
    assert!(
        tc.arena
            .device_value(b, FULL)
            .equal(&Tensor::from_slice(&[14i64]))
    );
    let [
        CacheAction::FreeDeviceKV(duplicates_head),
        CacheAction::RecoverSwaWithLockedFull {
            node_id,
            kept_full,
            incoming_full,
        },
        CacheAction::FreeDeviceKV(duplicates_tail),
    ] = result.cache_actions.as_slice()
    else {
        panic!(
            "expected FreeDeviceKV, RecoverSwaWithLockedFull, FreeDeviceKV, got {:?}",
            action_kinds(&result.cache_actions)
        );
    };
    assert!(duplicates_head[0].equal(&Tensor::from_slice(&[20i64, 21, 22])));
    assert_eq!(*node_id, tc.arena.node(b).id);
    assert!(kept_full.equal(&Tensor::from_slice(&[14i64])));
    assert!(incoming_full.equal(&Tensor::from_slice(&[24i64])));
    assert!(duplicates_tail[0].equal(&Tensor::from_slice(&[23i64])));
}

#[test]
fn insert_overlap_prev_prefix_strictly_inside_the_node_still_recovers() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4],
        &[10, 11, 12, 13],
        0,
        0,
    ));
    let root = tc.arena.root();
    let node = child_of(&tc, root, &[1]);
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4],
        &[20, 21, 22, 23],
        /* prev_prefix_len = */ 2,
        /* swa_evicted_seqlen = */ 0,
    ));
    // prev covers only part of the node, so the tombstone recovery runs.
    assert!(
        tc.arena
            .device_value(node, FULL)
            .equal(&Tensor::from_slice(&[20i64, 21, 22, 23]))
    );
    assert!(result.cache_actions.iter().any(|action| matches!(
        action,
        CacheAction::SwaRebuild { node_id, .. } if *node_id == tc.arena.node(node).id
    )));
}

#[test]
#[should_panic(expected = "swa_evicted_seqlen must be page-aligned")]
fn insert_overlap_rejects_a_page_misaligned_boundary() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 2);
    tc.insert(&insert_params_swa(&vec![1, 2], &[10, 11], 0, 0));
    tc.insert(&insert_params_swa(
        &vec![1, 2],
        &[20, 21],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 1,
    ));
}

#[test]
fn reinsert_after_full_eviction_rebuilds_swa_from_the_fresh_kv() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[10, 11, 12], 0, 0));
    let root = tc.arena.root();
    let leaf = child_of(&tc, root, &[1]);
    evict_full(&mut tc, leaf, /* remaining_size = */ 0);
    let result = tc.insert(&insert_params_swa(&vec![1, 2, 3], &[20, 21, 22], 0, 0));
    assert_eq!(result.prefix_len, 3);
    assert!(
        tc.arena
            .device_value(leaf, FULL)
            .equal(&Tensor::from_slice(&[20i64, 21, 22]))
    );
    let [
        CacheAction::SwaRebuild {
            node_id,
            source_value,
        },
    ] = result.cache_actions.as_slice()
    else {
        panic!("expected one SwaRebuild action");
    };
    assert_eq!(*node_id, tc.arena.node(leaf).id);
    assert!(source_value.equal(&Tensor::from_slice(&[20i64, 21, 22])));
}

#[test]
fn reinsert_straddling_the_boundary_splits_before_the_rebuild() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4],
        &[10, 11, 12, 13],
        0,
        0,
    ));
    let root = tc.arena.root();
    let node = child_of(&tc, root, &[1]);
    evict_full(&mut tc, node, /* remaining_size = */ 0);
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4],
        &[20, 21, 22, 23],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 2,
    ));
    // The out-of-window head stays an SWA tombstone on the split-off parent.
    let parent = child_of(&tc, root, &[1]);
    assert_eq!(tc.arena.node(parent).key, vec![1, 2]);
    assert_eq!(tc.arena.node(node).key, vec![3, 4]);
    assert!(
        tc.arena
            .device_value(parent, FULL)
            .equal(&Tensor::from_slice(&[20i64, 21]))
    );
    assert!(!tc.arena.has_device_value(parent, SWA));
    let [
        CacheAction::SwaRebuild {
            node_id,
            source_value,
        },
    ] = result.cache_actions.as_slice()
    else {
        panic!("expected one SwaRebuild action");
    };
    assert_eq!(*node_id, tc.arena.node(node).id);
    assert!(source_value.equal(&Tensor::from_slice(&[22i64, 23])));
}

#[test]
fn reinsert_straddling_the_boundary_at_a_second_level_node_splits_before_the_rebuild() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[10, 11, 12], 0, 0));
    tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4, 5],
        &[10, 11, 12, 13, 14],
        0,
        0,
    ));
    let root = tc.arena.root();
    let a = child_of(&tc, root, &[1]);
    let b = child_of(&tc, a, &[4]);
    evict_full(&mut tc, b, /* remaining_size = */ 0);
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4, 5],
        &[20, 21, 22, 23, 24],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 4,
    ));
    // The unevicted b splits at its node-relative offset 1; only the
    // in-window tail is rebuilt.
    let p = child_of(&tc, a, &[4]);
    assert_eq!(tc.arena.node(p).key, vec![4]);
    assert_eq!(tc.arena.node(b).key, vec![5]);
    assert!(
        tc.arena
            .device_value(p, FULL)
            .equal(&Tensor::from_slice(&[23i64]))
    );
    assert!(
        tc.arena
            .device_value(b, FULL)
            .equal(&Tensor::from_slice(&[24i64]))
    );
    assert!(!tc.arena.has_device_value(p, SWA));
    assert!(result.cache_actions.iter().any(|action| matches!(
        action,
        CacheAction::SwaRebuild { node_id, source_value }
            if *node_id == tc.arena.node(b).id && source_value.equal(&Tensor::from_slice(&[24i64]))
    )));
    assert!(!result.cache_actions.iter().any(|action| matches!(
        action,
        CacheAction::SwaRebuild { node_id, .. } if *node_id == tc.arena.node(p).id
    )));
}

#[test]
fn reinsert_entirely_outside_the_window_skips_the_rebuild() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[10, 11, 12], 0, 0));
    let root = tc.arena.root();
    let leaf = child_of(&tc, root, &[1]);
    evict_full(&mut tc, leaf, /* remaining_size = */ 0);
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3],
        &[20, 21, 22],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 3,
    ));
    assert!(result.cache_actions.is_empty());
    assert!(!tc.arena.has_device_value(leaf, SWA));
}

#[test]
fn reinsert_with_live_swa_skips_recovery() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[10, 11, 12], 0, 0));
    let root = tc.arena.root();
    let leaf = child_of(&tc, root, &[1]);
    tc.set_component_device_value(
        tc.arena.node(leaf).id,
        SWA,
        Tensor::from_slice(&[50i64, 51, 52]),
    );
    evict_full(&mut tc, leaf, /* remaining_size = */ 0);
    let result = tc.insert(&insert_params_swa(&vec![1, 2, 3], &[20, 21, 22], 0, 0));
    // The SWA value is already live: no rebuild is emitted.
    assert!(result.cache_actions.is_empty());
    assert!(
        tc.arena
            .device_value(leaf, SWA)
            .equal(&Tensor::from_slice(&[50i64, 51, 52]))
    );
}

#[test]
fn reinsert_boundary_at_the_node_start_rebuilds_the_whole_node() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[10, 11, 12], 0, 0));
    tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4, 5],
        &[30, 31, 32, 13, 14],
        0,
        0,
    ));
    let root = tc.arena.root();
    let a = child_of(&tc, root, &[1]);
    let b = child_of(&tc, a, &[4]);
    evict_full(&mut tc, b, /* remaining_size = */ 3);
    // a is an out-of-window tombstone (all duplicate); b unevicts and
    // rebuilds in full, no split.
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4, 5],
        &[20, 21, 22, 23, 24],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 3,
    ));
    assert_eq!(tc.arena.node(b).key, vec![4, 5]);
    let [
        CacheAction::FreeDeviceKV(duplicates),
        CacheAction::SwaRebuild {
            node_id,
            source_value,
        },
    ] = result.cache_actions.as_slice()
    else {
        panic!(
            "expected FreeDeviceKV then SwaRebuild, got {:?}",
            action_kinds(&result.cache_actions)
        );
    };
    assert!(duplicates[0].equal(&Tensor::from_slice(&[20i64, 21, 22])));
    assert_eq!(*node_id, tc.arena.node(b).id);
    assert!(source_value.equal(&Tensor::from_slice(&[23i64, 24])));
}

#[test]
fn walk_split_redistributes_the_live_swa_value() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4],
        &[10, 11, 12, 13],
        0,
        0,
    ));
    let root = tc.arena.root();
    let node = child_of(&tc, root, &[1]);
    tc.set_component_device_value(
        tc.arena.node(node).id,
        SWA,
        Tensor::from_slice(&[50i64, 51, 52, 53]),
    );
    let result = tc.insert(&insert_params_swa(&vec![1, 2, 9], &[20, 21, 29], 0, 0));
    assert_eq!(result.prefix_len, 2);
    let parent = child_of(&tc, root, &[1]);
    let leaf = child_of(&tc, parent, &[9]);
    // The split slices the SWA value alongside the Full value; both sides
    // stay in the SWA device LRU.
    assert!(
        tc.arena
            .device_value(parent, SWA)
            .equal(&Tensor::from_slice(&[50i64, 51]))
    );
    assert!(
        tc.arena
            .device_value(node, SWA)
            .equal(&Tensor::from_slice(&[52i64, 53]))
    );
    assert!(tc.device_lru_list(SWA).in_list(Some(parent)));
    assert!(tc.device_lru_list(SWA).in_list(Some(node)));
    let [
        CacheAction::FreeDeviceKV(duplicates),
        CacheAction::SwaRebuild {
            node_id,
            source_value,
        },
    ] = result.cache_actions.as_slice()
    else {
        panic!("expected FreeDeviceKV then SwaRebuild");
    };
    assert!(duplicates[0].equal(&Tensor::from_slice(&[20i64, 21])));
    assert_eq!(*node_id, tc.arena.node(leaf).id);
    assert!(source_value.equal(&Tensor::from_slice(&[29i64])));
}

#[test]
fn redistribute_on_node_split_slices_host_values_and_parks_tombstones() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2], &[10, 11], 0, 0));
    let root = tc.arena.root();
    let node = child_of(&tc, root, &[1]);
    tc.arena
        .set_host_value(node, SWA, Tensor::from_slice(&[70i64, 71]));
    tc.host_lru_list_mut(SWA).insert_mru(node);
    let (parent, action) = tc.split_node_(node, /* split_len = */ 1);
    assert!(action.is_none());
    assert!(
        tc.arena
            .host_value(parent, SWA)
            .equal(&Tensor::from_slice(&[70i64]))
    );
    assert!(
        tc.arena
            .host_value(node, SWA)
            .equal(&Tensor::from_slice(&[71i64]))
    );
    // Both sides are device tombstones: the parent joins the host LRU, the
    // child stays listed.
    assert!(tc.host_lru_list(SWA).in_list(Some(parent)));
    assert!(tc.host_lru_list(SWA).in_list(Some(node)));
}

#[test]
fn redistribute_on_node_split_keeps_device_valued_sides_off_the_host_lru() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2], &[10, 11], 0, 0));
    let root = tc.arena.root();
    let node = child_of(&tc, root, &[1]);
    tc.set_component_device_value(
        tc.arena.node(node).id,
        SWA,
        Tensor::from_slice(&[50i64, 51]),
    );
    tc.arena
        .set_host_value(node, SWA, Tensor::from_slice(&[70i64, 71]));
    tc.arena
        .node_mut(node)
        .set_lock_ref_(ValueSlotIdx::device(SWA), 2);
    let (parent, _) = tc.split_node_(node, /* split_len = */ 1);
    // Device-valued sides slice both tiers and inherit the SWA lock_ref;
    // neither enters the host LRU.
    assert!(
        tc.arena
            .device_value(parent, SWA)
            .equal(&Tensor::from_slice(&[50i64]))
    );
    assert!(
        tc.arena
            .host_value(parent, SWA)
            .equal(&Tensor::from_slice(&[70i64]))
    );
    assert_eq!(tc.arena.device_lock_ref(parent, SWA), 2);
    assert_eq!(tc.arena.device_lock_ref(node, SWA), 2);
    assert_eq!(tc.host_lru_list(SWA).len(), 0);
}

#[test]
fn redistribute_on_node_split_lists_an_unlisted_tombstone_child() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2], &[10, 11], 0, 0));
    let root = tc.arena.root();
    let node = child_of(&tc, root, &[1]);
    tc.arena
        .set_host_value(node, SWA, Tensor::from_slice(&[70i64, 71]));
    let (parent, _) = tc.split_node_(node, /* split_len = */ 1);
    // Neither side was in the host LRU; both device tombstones join it.
    assert!(tc.host_lru_list(SWA).in_list(Some(parent)));
    assert!(tc.host_lru_list(SWA).in_list(Some(node)));
}

#[test]
fn acquire_lock_walks_until_the_window_fills_and_stamps_the_crossing_node() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, b);
    store_swa_device(&mut tc, c);
    let result = swa_component(2).acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 1);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 1);
    assert_eq!(tc.arena.device_lock_ref(a, SWA), 0);
    // The walk fills the 2-atom window at b; b carries the first minted uuid.
    assert_eq!(result.swa_uuid_for_lock, Some(2));
    assert_eq!(result.swa_uuid_for_host_lock, None);
    assert_eq!(node_swa_uuid(&tc, b), Some(2));
    assert_eq!(node_swa_uuid(&tc, c), None);
    assert_eq!(tc.swa_evictable_size(), 1);
    assert_eq!(tc.swa_protected_size(), 2);
}

#[test]
fn acquire_lock_overshooting_the_window_stops_at_the_crossing_node() {
    let mut tc = swa_core(/* window = */ 3, /* page_size = */ 1);
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let b = tc
        .arena
        .alloc_child(
            a,
            /* key = */ vec![3, 4],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let c = tc
        .arena
        .alloc_child(
            b,
            /* key = */ vec![5, 6],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, b);
    store_swa_device(&mut tc, c);
    let result = swa_component(3).acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    // The 2-atom nodes overshoot the 3-atom window at b (2 -> 4): the walk
    // stops there, stamps b, and leaves a untouched.
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 1);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 1);
    assert_eq!(tc.arena.device_lock_ref(a, SWA), 0);
    assert_eq!(result.swa_uuid_for_lock, Some(2));
    assert_eq!(node_swa_uuid(&tc, b), Some(2));
    assert_eq!(node_swa_uuid(&tc, a), None);
    assert_eq!(tc.swa_evictable_size(), 2);
    assert_eq!(tc.swa_protected_size(), 4);
}

#[test]
fn acquire_lock_reuses_the_stamped_uuid_and_shifts_sizes_once() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, b);
    store_swa_device(&mut tc, c);
    let swa = swa_component(2);
    let first = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    let second = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    assert_eq!(second.swa_uuid_for_lock, first.swa_uuid_for_lock);
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 2);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 2);
    assert_eq!(tc.swa_evictable_size(), 1);
    assert_eq!(tc.swa_protected_size(), 2);
}

#[test]
fn acquire_lock_skips_tombstones_and_records_them() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, c);
    let result = swa_component(2).acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    // The valueless b is recorded and skipped; the window fills at a.
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 1);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(a, SWA), 1);
    assert_eq!(result.skip_lock_node_ids[&SWA].len(), 1);
    assert!(result.skip_lock_node_ids[&SWA].contains(&tc.arena.node(b).id));
    assert!(result.swa_uuid_for_lock.is_some());
    assert_eq!(node_swa_uuid(&tc, a), result.swa_uuid_for_lock);
}

#[test]
fn acquire_lock_under_the_window_reaches_the_root_without_a_uuid() {
    let mut tc = swa_core(/* window = */ 100, /* page_size = */ 1);
    let [a, b] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, b);
    let result = swa_component(100).acquire_component_lock(
        &mut tc,
        b,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    assert_eq!(result.swa_uuid_for_lock, None);
    assert_eq!(tc.arena.device_lock_ref(a, SWA), 1);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 1);
    assert_eq!(tc.swa_evictable_size(), 0);
    assert_eq!(tc.swa_protected_size(), 2);
}

#[test]
fn inc_lock_ref_runs_full_and_swa_walks_together() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, b);
    store_swa_device(&mut tc, c);
    let result = tc.inc_lock_ref(tc.arena.node(c).id);
    // FULL sees a valueless path (skip segment only); SWA locks its window.
    assert_eq!(result.delta, Some(0));
    assert_eq!(result.skip_lock_node_ids[&FULL].len(), 3);
    assert!(result.swa_uuid_for_lock.is_some());
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 1);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 1);
    assert_eq!(tc.arena.device_lock_ref(a, SWA), 0);
}

#[test]
fn inc_host_lock_ref_runs_full_and_swa_host_arms_together() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    for node in [a, b, c] {
        set_swa_host(&mut tc, node);
    }
    tc.arena
        .set_host_value(c, FULL, Tensor::from_slice(&[0i64]));
    let result = tc.inc_host_lock_ref(tc.arena.node(c).id);
    // FULL pins only the anchor; SWA walks its host window up to b.
    assert_eq!(tc.arena.host_lock_ref(c, FULL), 1);
    assert_eq!(tc.arena.host_lock_ref(b, FULL), 0);
    assert_eq!(tc.arena.host_lock_ref(c, SWA), 1);
    assert_eq!(tc.arena.host_lock_ref(b, SWA), 1);
    assert_eq!(tc.arena.host_lock_ref(a, SWA), 0);
    assert!(result.swa_uuid_for_host_lock.is_some());
    // The release replays the acquire's uuid and unwinds both arms.
    let params = DecLockRefParams {
        swa_uuid_for_host_lock: result.swa_uuid_for_host_lock,
        skip_lock_node_ids: result.skip_lock_node_ids,
        ..Default::default()
    };
    tc.dec_host_lock_ref(tc.arena.node(c).id, Some(&params));
    assert_eq!(tc.arena.host_lock_ref(c, FULL), 0);
    assert_eq!(tc.arena.host_lock_ref(c, SWA), 0);
    assert_eq!(tc.arena.host_lock_ref(b, SWA), 0);
    // Only the dispatcher's final leaf-set pass sees the fully-unlocked anchor.
    assert!(tc.evictable_host_leaves.contains(c));
}

#[test]
fn dec_host_lock_ref_with_the_inner_uuid_leaves_an_outer_window_pinned() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    for node in [a, b, c] {
        set_swa_host(&mut tc, node);
    }
    // Overlapping host windows: {c, b} stamps its uuid at b, {b, a} at a.
    let inner = tc.inc_host_lock_ref(tc.arena.node(c).id);
    tc.inc_host_lock_ref(tc.arena.node(b).id);
    assert!(inner.swa_uuid_for_host_lock.is_some());
    assert_eq!(tc.arena.host_lock_ref(c, SWA), 1);
    assert_eq!(tc.arena.host_lock_ref(b, SWA), 2);
    assert_eq!(tc.arena.host_lock_ref(a, SWA), 1);
    // Releasing the inner window with its own uuid stops at b; the outer
    // window's lock above the boundary survives.
    let params = DecLockRefParams {
        swa_uuid_for_host_lock: inner.swa_uuid_for_host_lock,
        skip_lock_node_ids: inner.skip_lock_node_ids,
        ..Default::default()
    };
    tc.dec_host_lock_ref(tc.arena.node(c).id, Some(&params));
    assert_eq!(tc.arena.host_lock_ref(c, SWA), 0);
    assert_eq!(tc.arena.host_lock_ref(b, SWA), 1);
    assert_eq!(tc.arena.host_lock_ref(a, SWA), 1);
}

#[test]
fn acquire_host_lock_walks_until_the_window_fills_and_stamps_the_host_uuid() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    for node in [a, b, c] {
        set_swa_host(&mut tc, node);
        tc.host_lru_list_mut(SWA).insert_mru(node);
    }
    let result = swa_component(2).acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ true,
    );
    assert_eq!(tc.arena.host_lock_ref(c, SWA), 1);
    assert_eq!(tc.arena.host_lock_ref(b, SWA), 1);
    assert_eq!(tc.arena.host_lock_ref(a, SWA), 0);
    // The window fills at b; b carries the host uuid and leaves the host LRU.
    assert_eq!(result.swa_uuid_for_host_lock, Some(2));
    assert_eq!(result.swa_uuid_for_lock, None);
    assert_eq!(node_swa_host_uuid(&tc, b), Some(2));
    assert_eq!(node_swa_host_uuid(&tc, c), None);
    assert!(!tc.host_lru_list(SWA).in_list(Some(c)));
    assert!(!tc.host_lru_list(SWA).in_list(Some(b)));
    assert!(tc.host_lru_list(SWA).in_list(Some(a)));
    // Host locks never touch the device tier or its sizes.
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 0);
    assert_eq!(tc.swa_evictable_size(), 0);
    assert_eq!(tc.swa_protected_size(), 0);
}

#[test]
fn acquire_host_lock_reuses_the_stamped_uuid_and_skips_unlisted_nodes() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    for node in [a, b, c] {
        set_swa_host(&mut tc, node);
    }
    let swa = swa_component(2);
    let first = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ true,
    );
    let second = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ true,
    );
    assert_eq!(second.swa_uuid_for_host_lock, first.swa_uuid_for_host_lock);
    assert_eq!(tc.arena.host_lock_ref(c, SWA), 2);
    assert_eq!(tc.arena.host_lock_ref(b, SWA), 2);
    assert_eq!(tc.arena.host_lock_ref(a, SWA), 0);
}

#[test]
fn acquire_host_lock_skips_host_tombstones_and_records_them() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    set_swa_host(&mut tc, a);
    set_swa_host(&mut tc, c);
    let result = swa_component(2).acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ true,
    );
    assert_eq!(tc.arena.host_lock_ref(c, SWA), 1);
    assert_eq!(tc.arena.host_lock_ref(b, SWA), 0);
    assert_eq!(tc.arena.host_lock_ref(a, SWA), 1);
    assert_eq!(result.skip_lock_node_ids[&SWA].len(), 1);
    assert!(result.skip_lock_node_ids[&SWA].contains(&tc.arena.node(b).id));
    assert_eq!(node_swa_host_uuid(&tc, a), result.swa_uuid_for_host_lock);
    assert!(result.swa_uuid_for_host_lock.is_some());
}

#[test]
fn acquire_host_lock_under_the_window_reaches_the_root_without_a_uuid() {
    let mut tc = swa_core(/* window = */ 100, /* page_size = */ 1);
    let [a, b] = chain(&mut tc);
    set_swa_host(&mut tc, a);
    set_swa_host(&mut tc, b);
    let result = swa_component(100).acquire_component_lock(
        &mut tc,
        b,
        IncLockRefResult::default(),
        /* lock_host = */ true,
    );
    assert_eq!(result.swa_uuid_for_host_lock, None);
    assert_eq!(tc.arena.host_lock_ref(a, SWA), 1);
    assert_eq!(tc.arena.host_lock_ref(b, SWA), 1);
}

#[test]
fn acquire_host_lock_delists_only_on_the_host_ref_transition() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    for node in [a, b, c] {
        store_swa_device(&mut tc, node);
        set_swa_host(&mut tc, node);
        tc.host_lru_list_mut(SWA).insert_mru(node);
    }
    let swa = swa_component(2);
    // A held device lock must not stand in for the host ref transition.
    let _ = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    let _ = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ true,
    );
    assert!(!tc.host_lru_list(SWA).in_list(Some(c)));
    assert!(!tc.host_lru_list(SWA).in_list(Some(b)));
    assert!(tc.host_lru_list(SWA).in_list(Some(a)));
    // A re-listed node with a live host lock stays listed on re-acquire.
    tc.host_lru_list_mut(SWA).insert_mru(b);
    let _ = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ true,
    );
    assert!(tc.host_lru_list(SWA).in_list(Some(b)));
    assert_eq!(tc.arena.host_lock_ref(c, SWA), 2);
    assert_eq!(tc.arena.host_lock_ref(b, SWA), 2);
}

#[test]
fn acquire_host_lock_stamps_the_host_tier_uuid_field_only() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    for node in [a, b, c] {
        set_swa_host(&mut tc, node);
    }
    let result = swa_component(2).acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ true,
    );
    // The boundary uuid lands on the host-tier field; the device field stays clear.
    assert_eq!(result.swa_uuid_for_host_lock, Some(2));
    assert_eq!(node_swa_host_uuid(&tc, b), Some(2));
    assert_eq!(node_swa_uuid(&tc, b), None);
    assert_eq!(node_swa_host_uuid(&tc, c), None);
    assert_eq!(node_swa_uuid(&tc, c), None);
    let _ = a;
}

#[test]
fn device_and_host_lock_walks_mint_independent_uuids() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    for node in [a, b, c] {
        store_swa_device(&mut tc, node);
        set_swa_host(&mut tc, node);
    }
    let swa = swa_component(2);
    let device = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    let host = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ true,
    );
    assert_eq!(device.swa_uuid_for_lock, Some(2));
    assert_eq!(host.swa_uuid_for_host_lock, Some(3));
    assert_eq!(node_swa_uuid(&tc, b), Some(2));
    assert_eq!(node_swa_host_uuid(&tc, b), Some(3));
}

#[test]
fn eviction_priority_is_zero_for_leaf_one_for_internal() {
    let swa = swa_component(4);
    assert_eq!(
        <SwaComponent as TreeComponent<Vec<i64>>>::eviction_priority(&swa, true),
        0
    );
    assert_eq!(
        <SwaComponent as TreeComponent<Vec<i64>>>::eviction_priority(&swa, false),
        1
    );
}

#[test]
fn evict_component_device_frees_the_full_indices_and_tombstones_swa() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2], &[10, 11], 0, 0));
    let root = tc.arena.root();
    let node = child_of(&tc, root, &[1]);
    tc.set_component_device_value(
        tc.arena.node(node).id,
        SWA,
        Tensor::from_slice(&[50i64, 51]),
    );
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    let (freed, host_freed) = swa_component(4).evict_component(
        &mut tc,
        node,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Device,
    );
    assert_eq!((freed, host_freed), (2, 0));
    // The freed indices are the FULL slice (SWA slots pair through it);
    // the FULL value itself stays on the node.
    assert!(device_frees[&SWA][0].equal(&Tensor::from_slice(&[10i64, 11])));
    assert!(!tc.arena.has_device_value(node, SWA));
    assert!(
        tc.arena
            .device_value(node, FULL)
            .equal(&Tensor::from_slice(&[10i64, 11]))
    );
    assert_eq!(tc.swa_evictable_size(), 0);
}

#[test]
fn evict_component_device_parks_a_remaining_host_value() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2], &[10, 11], 0, 0));
    let root = tc.arena.root();
    let node = child_of(&tc, root, &[1]);
    tc.set_component_device_value(
        tc.arena.node(node).id,
        SWA,
        Tensor::from_slice(&[50i64, 51]),
    );
    set_swa_host(&mut tc, node);
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    swa_component(4).evict_component(
        &mut tc,
        node,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Device,
    );
    assert!(tc.arena.has_host_value(node, SWA));
    assert!(tc.host_lru_list(SWA).in_list(Some(node)));
    assert!(host_frees.is_empty());
}

#[test]
fn evict_component_host_frees_and_delists_the_host_value() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2], &[10, 11], 0, 0));
    let root = tc.arena.root();
    let node = child_of(&tc, root, &[1]);
    set_swa_host(&mut tc, node);
    tc.host_lru_list_mut(SWA).insert_mru(node);
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    let (freed, host_freed) = swa_component(4).evict_component(
        &mut tc,
        node,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Host,
    );
    assert_eq!((freed, host_freed), (0, 2));
    assert!(host_frees[&SWA][0].equal(&Tensor::from_slice(&[0i64, 0])));
    assert!(!tc.arena.has_host_value(node, SWA));
    assert!(!tc.host_lru_list(SWA).in_list(Some(node)));
    assert!(device_frees.is_empty());
}

#[test]
fn release_lock_returns_the_window_to_evictable() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, b);
    store_swa_device(&mut tc, c);
    let swa = swa_component(2);
    let result = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    let params = DecLockRefParams {
        swa_uuid_for_lock: result.swa_uuid_for_lock,
        swa_uuid_for_host_lock: result.swa_uuid_for_host_lock,
        skip_lock_node_ids: result.skip_lock_node_ids,
    };
    swa.release_component_lock(&mut tc, c, Some(&params), /* lock_host = */ false);
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(a, SWA), 0);
    assert_eq!(tc.swa_evictable_size(), 3);
    assert_eq!(tc.swa_protected_size(), 0);
}

#[test]
fn release_lock_keeps_sizes_while_other_locks_remain() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, b);
    store_swa_device(&mut tc, c);
    let swa = swa_component(2);
    let first = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    let _ = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    let params = DecLockRefParams {
        swa_uuid_for_lock: first.swa_uuid_for_lock,
        swa_uuid_for_host_lock: first.swa_uuid_for_host_lock,
        skip_lock_node_ids: first.skip_lock_node_ids,
    };
    swa.release_component_lock(&mut tc, c, Some(&params), /* lock_host = */ false);
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 1);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 1);
    assert_eq!(tc.swa_evictable_size(), 1);
    assert_eq!(tc.swa_protected_size(), 2);
}

#[test]
fn release_lock_replays_the_tombstone_skips() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, c);
    let swa = swa_component(2);
    let result = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    // b gained a device value AFTER the acquire recorded it as a tombstone.
    store_swa_device(&mut tc, b);
    let params = DecLockRefParams {
        swa_uuid_for_lock: result.swa_uuid_for_lock,
        swa_uuid_for_host_lock: result.swa_uuid_for_host_lock,
        skip_lock_node_ids: result.skip_lock_node_ids,
    };
    swa.release_component_lock(&mut tc, c, Some(&params), /* lock_host = */ false);
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(a, SWA), 0);
    assert_eq!(tc.swa_evictable_size(), 3);
    assert_eq!(tc.swa_protected_size(), 0);
}

#[test]
fn release_lock_stops_at_the_window_uuid() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, b);
    store_swa_device(&mut tc, c);
    let swa = swa_component(2);
    let result = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    // A manually locked ancestor above the window must stay untouched.
    tc.arena
        .node_mut(a)
        .set_lock_ref_(ValueSlotIdx::device(SWA), 1);
    let params = DecLockRefParams {
        swa_uuid_for_lock: result.swa_uuid_for_lock,
        swa_uuid_for_host_lock: result.swa_uuid_for_host_lock,
        skip_lock_node_ids: result.skip_lock_node_ids,
    };
    swa.release_component_lock(&mut tc, c, Some(&params), /* lock_host = */ false);
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(a, SWA), 1);
}

#[test]
fn release_host_lock_stops_at_the_host_uuid_boundary() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    for node in [a, b, c] {
        set_swa_host(&mut tc, node);
        tc.host_lru_list_mut(SWA).insert_mru(node);
    }
    let swa = swa_component(2);
    let result = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ true,
    );
    // A second request holds its own host lock on a, above the boundary b.
    let _ = swa.acquire_component_lock(
        &mut tc,
        a,
        IncLockRefResult::default(),
        /* lock_host = */ true,
    );
    let params = DecLockRefParams {
        swa_uuid_for_lock: result.swa_uuid_for_lock,
        swa_uuid_for_host_lock: result.swa_uuid_for_host_lock,
        skip_lock_node_ids: result.skip_lock_node_ids,
    };
    swa.release_component_lock(&mut tc, c, Some(&params), /* lock_host = */ true);
    assert_eq!(tc.arena.host_lock_ref(c, SWA), 0);
    assert_eq!(tc.arena.host_lock_ref(b, SWA), 0);
    assert_eq!(tc.arena.host_lock_ref(a, SWA), 1);
    assert!(!tc.host_lru_list(SWA).in_list(Some(a)));
}

#[test]
fn release_lock_without_params_passes_over_an_unlocked_middle_node() {
    let mut tc = swa_core(/* window = */ 1, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, b);
    store_swa_device(&mut tc, c);
    let swa = swa_component(1);
    // The 1-atom window locks only the acquired node: c and a, never b.
    let _ = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    let _ = swa.acquire_component_lock(
        &mut tc,
        a,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    swa.release_component_lock(
        &mut tc, c, /* params = */ None, /* lock_host = */ false,
    );
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(a, SWA), 0);
    assert_eq!(tc.swa_evictable_size(), 3);
    assert_eq!(tc.swa_protected_size(), 0);
}

#[test]
fn release_host_lock_reparks_tombstoned_host_nodes() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    for node in [a, b, c] {
        set_swa_host(&mut tc, node);
        tc.host_lru_list_mut(SWA).insert_mru(node);
    }
    let swa = swa_component(2);
    let result = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ true,
    );
    let params = DecLockRefParams {
        swa_uuid_for_lock: result.swa_uuid_for_lock,
        swa_uuid_for_host_lock: result.swa_uuid_for_host_lock,
        skip_lock_node_ids: result.skip_lock_node_ids,
    };
    swa.release_component_lock(&mut tc, c, Some(&params), /* lock_host = */ true);
    assert_eq!(tc.arena.host_lock_ref(c, SWA), 0);
    assert_eq!(tc.arena.host_lock_ref(b, SWA), 0);
    assert!(tc.host_lru_list(SWA).in_list(Some(c)));
    assert!(tc.host_lru_list(SWA).in_list(Some(b)));
    assert!(tc.host_lru_list(SWA).in_list(Some(a)));
}

#[test]
fn inc_then_dec_lock_ref_roundtrips_with_dec_params() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, b);
    store_swa_device(&mut tc, c);
    let result = tc.inc_lock_ref(tc.arena.node(c).id);
    let params = DecLockRefParams {
        swa_uuid_for_lock: result.swa_uuid_for_lock,
        swa_uuid_for_host_lock: result.swa_uuid_for_host_lock,
        skip_lock_node_ids: result.skip_lock_node_ids,
    };
    tc.dec_lock_ref(
        tc.arena.node(c).id,
        Some(&params),
        /* skip_swa = */ false,
    );
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 0);
    assert_eq!(tc.swa_evictable_size(), 3);
    assert_eq!(tc.swa_protected_size(), 0);
}

#[test]
fn dec_swa_lock_only_releases_swa_while_full_stays_locked() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    for node in [a, b, c] {
        store_swa_device(&mut tc, node);
        let len = tc.arena.node(node).key.atom_len();
        tc.arena
            .set_device_value(node, FULL, Tensor::from_slice(&vec![9i64; len]));
    }
    // Fund FULL's evictable counter for its lock walk (raw slot sets skip it).
    tc.component_state_mut(FULL).evictable_size = 3;
    let result = tc.inc_lock_ref(tc.arena.node(c).id);
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    tc.dec_swa_lock_only(
        tc.arena.node(c).id,
        result.swa_uuid_for_lock,
        &mut device_frees,
        &mut host_frees,
    );
    // SWA is early-released; the FULL locks on the path stay.
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(c, FULL), 1);
    assert_eq!(tc.arena.device_lock_ref(b, FULL), 1);
    // Full still locks the nodes, so nothing is device-leaf evictable.
    assert!(device_frees.is_empty());
    assert_eq!(tc.swa_evictable_size(), 3);
    assert_eq!(tc.swa_protected_size(), 0);
}

#[test]
fn dec_swa_lock_only_evicts_a_fully_unlocked_device_leaf() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    for node in [a, b, c] {
        store_swa_device(&mut tc, node);
        let len = tc.arena.node(node).key.atom_len();
        tc.arena
            .set_device_value(node, FULL, Tensor::from_slice(&vec![9i64; len]));
    }
    let swa = swa_component(2);
    let result = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    tc.dec_swa_lock_only(
        tc.arena.node(c).id,
        result.swa_uuid_for_lock,
        &mut device_frees,
        &mut host_frees,
    );
    // The fully unlocked leaf c is device-evicted on release; b keeps its
    // SWA value because its child still holds FULL KV.
    assert!(!tc.arena.has_device_value(c, SWA));
    assert!(device_frees[&SWA][0].equal(&Tensor::from_slice(&[9i64])));
    assert!(tc.arena.has_device_value(b, SWA));
    assert_eq!(tc.swa_evictable_size(), 2);
    assert_eq!(tc.swa_protected_size(), 0);
}

#[test]
fn dec_swa_lock_only_is_a_noop_without_the_swa_component() {
    let mut tc: UnifiedTreeCore<Vec<i64>> =
        UnifiedTreeCore::new(CacheInitParams::default(), vec![FULL]);
    let root = tc.arena.root();
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    tc.dec_swa_lock_only(
        tc.arena.node(root).id,
        None,
        &mut device_frees,
        &mut host_frees,
    );
    assert!(device_frees.is_empty());
}

#[test]
fn release_window_lock_breaks_on_a_tombstone_carrying_the_uuid() {
    let mut tc = swa_core(/* window = */ 100, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, c);
    let swa = swa_component(100);
    let _ = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    // A stale uuid on the tombstone b ends the walk before a.
    tc.arena.node_mut(b).swa_uuid = Some(99);
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    <SwaComponent as TreeComponent<Vec<i64>>>::release_window_lock(
        &swa,
        &mut tc,
        c,
        Some(99),
        &mut device_frees,
        &mut host_frees,
    );
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(a, SWA), 1);
}

#[test]
#[should_panic(expected = "release_window_lock is SWA-only")]
fn release_window_lock_panics_on_a_non_swa_component() {
    use crate::components::full::FullComponent;
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let root = tc.arena.root();
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    <FullComponent as TreeComponent<Vec<i64>>>::release_window_lock(
        &FullComponent,
        &mut tc,
        root,
        None,
        &mut device_frees,
        &mut host_frees,
    );
}

#[test]
fn release_lock_skip_set_leaves_a_relocked_tombstone_credited() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, c);
    let swa = swa_component(2);
    let first = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    // b regains a value and a second request locks it before the first
    // release replays its tombstone skip set.
    store_swa_device(&mut tc, b);
    let _ = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    let params = DecLockRefParams {
        swa_uuid_for_lock: first.swa_uuid_for_lock,
        swa_uuid_for_host_lock: first.swa_uuid_for_host_lock,
        skip_lock_node_ids: first.skip_lock_node_ids,
    };
    swa.release_component_lock(&mut tc, c, Some(&params), /* lock_host = */ false);
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 1);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 1);
    assert_eq!(tc.arena.device_lock_ref(a, SWA), 0);
    assert_eq!(tc.swa_evictable_size(), 1);
    assert_eq!(tc.swa_protected_size(), 2);
}

#[test]
fn release_lock_passes_over_uncredited_nodes_without_params() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, b);
    store_swa_device(&mut tc, c);
    let swa = swa_component(2);
    let _ = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    // No params: the walk crosses the never-credited a up to the root.
    swa.release_component_lock(
        &mut tc, c, /* params = */ None, /* lock_host = */ false,
    );
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(a, SWA), 0);
    assert_eq!(tc.swa_evictable_size(), 3);
    assert_eq!(tc.swa_protected_size(), 0);
}

#[test]
fn dec_swa_lock_only_releases_the_window_exactly_once() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, b);
    store_swa_device(&mut tc, c);
    let swa = swa_component(2);
    let first = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    let _ = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    tc.dec_swa_lock_only(
        tc.arena.node(c).id,
        first.swa_uuid_for_lock,
        &mut device_frees,
        &mut host_frees,
    );
    // The second window still holds the lock: refs drop to 1, sizes stay.
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 1);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 1);
    assert_eq!(tc.swa_evictable_size(), 1);
    assert_eq!(tc.swa_protected_size(), 2);
    tc.dec_swa_lock_only(
        tc.arena.node(c).id,
        first.swa_uuid_for_lock,
        &mut device_frees,
        &mut host_frees,
    );
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 0);
    assert_eq!(tc.swa_evictable_size(), 3);
    assert_eq!(tc.swa_protected_size(), 0);
    let _ = a;
}

#[test]
fn dec_swa_lock_only_leaves_out_of_window_swa_locks_alone() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, b);
    store_swa_device(&mut tc, c);
    let swa = swa_component(2);
    let result = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    // A holds a lock beyond the window (e.g. another request's window).
    tc.arena
        .node_mut(a)
        .set_lock_ref_(ValueSlotIdx::device(SWA), 1);
    tc.dec_evictable_size(SWA, 1);
    tc.inc_protected_size(SWA, 1);
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    tc.dec_swa_lock_only(
        tc.arena.node(c).id,
        result.swa_uuid_for_lock,
        &mut device_frees,
        &mut host_frees,
    );
    // Only the SWA window is released; a's out-of-window lock survives.
    assert_eq!(tc.arena.device_lock_ref(a, SWA), 1);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 0);
    assert_eq!(tc.swa_protected_size(), 1);
}

#[test]
fn release_window_lock_passes_over_an_unlocked_valued_node() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, b);
    store_swa_device(&mut tc, c);
    let swa = swa_component(2);
    let _ = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    // No uuid bound: the walk crosses the valued-but-unlocked a to the root.
    swa.release_window_lock(&mut tc, c, None, &mut device_frees, &mut host_frees);
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(a, SWA), 0);
}

#[test]
fn release_window_lock_passes_over_a_mid_chain_tombstone_without_a_uuid() {
    let mut tc = swa_core(/* window = */ 100, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    store_swa_device(&mut tc, c);
    let swa = swa_component(100);
    let _ = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    // No uuid bound: the walk crosses the mid-chain tombstone b and releases a.
    swa.release_window_lock(&mut tc, c, None, &mut device_frees, &mut host_frees);
    assert_eq!(tc.arena.device_lock_ref(c, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(b, SWA), 0);
    assert_eq!(tc.arena.device_lock_ref(a, SWA), 0);
    assert_eq!(tc.swa_protected_size(), 0);
}

#[test]
fn release_host_lock_does_not_repark_a_node_whose_host_value_was_taken() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a] = chain(&mut tc);
    set_swa_host(&mut tc, a);
    tc.host_lru_list_mut(SWA).insert_mru(a);
    let swa = swa_component(2);
    let result = swa.acquire_component_lock(
        &mut tc,
        a,
        IncLockRefResult::default(),
        /* lock_host = */ true,
    );
    // The host value moved out while the lock was held; the node has no
    // device value either, so the release has nothing to park.
    let _ = tc.arena.take_host_value(a, SWA);
    let params = DecLockRefParams {
        swa_uuid_for_lock: result.swa_uuid_for_lock,
        swa_uuid_for_host_lock: result.swa_uuid_for_host_lock,
        skip_lock_node_ids: result.skip_lock_node_ids,
    };
    swa.release_component_lock(&mut tc, a, Some(&params), /* lock_host = */ true);
    assert_eq!(tc.arena.host_lock_ref(a, SWA), 0);
    assert!(!tc.host_lru_list(SWA).in_list(Some(a)));
}

#[test]
fn release_host_lock_skips_reparking_device_valued_nodes() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    for node in [a, b, c] {
        store_swa_device(&mut tc, node);
        set_swa_host(&mut tc, node);
        tc.host_lru_list_mut(SWA).insert_mru(node);
    }
    let swa = swa_component(2);
    let result = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ true,
    );
    let params = DecLockRefParams {
        swa_uuid_for_lock: result.swa_uuid_for_lock,
        swa_uuid_for_host_lock: result.swa_uuid_for_host_lock,
        skip_lock_node_ids: result.skip_lock_node_ids,
    };
    swa.release_component_lock(&mut tc, c, Some(&params), /* lock_host = */ true);
    // Device-valued nodes never re-park in the host LRU on host release.
    assert!(!tc.host_lru_list(SWA).in_list(Some(c)));
    assert!(!tc.host_lru_list(SWA).in_list(Some(b)));
}

#[test]
fn release_host_lock_leaves_an_already_listed_node_listed() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain(&mut tc);
    for node in [a, b, c] {
        set_swa_host(&mut tc, node);
        tc.host_lru_list_mut(SWA).insert_mru(node);
    }
    let swa = swa_component(2);
    let result = swa.acquire_component_lock(
        &mut tc,
        c,
        IncLockRefResult::default(),
        /* lock_host = */ true,
    );
    // Something re-listed b while the lock was held (e.g. a split re-park).
    tc.host_lru_list_mut(SWA).insert_mru(b);
    let params = DecLockRefParams {
        swa_uuid_for_lock: result.swa_uuid_for_lock,
        swa_uuid_for_host_lock: result.swa_uuid_for_host_lock,
        skip_lock_node_ids: result.skip_lock_node_ids,
    };
    swa.release_component_lock(&mut tc, c, Some(&params), /* lock_host = */ true);
    assert!(tc.host_lru_list(SWA).in_list(Some(b)));
    assert!(tc.host_lru_list(SWA).in_list(Some(c)));
    let _ = a;
}

// A three-node chain built through insert (leaf sets maintained), with
// SWA device values stored on every node.
fn swa_evict_chain(tc: &mut UnifiedTreeCore<Vec<i64>>) -> [NodeIdx_; 3] {
    tc.insert(&insert_params_swa(&vec![1], &[10], 0, 0));
    tc.insert(&insert_params_swa(&vec![1, 2], &[10, 11], 0, 0));
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[10, 11, 12], 0, 0));
    let root = tc.arena.root();
    let a = child_of(tc, root, &[1]);
    let b = child_of(tc, a, &[2]);
    let c = child_of(tc, b, &[3]);
    for node in [a, b, c] {
        store_swa_device(tc, node);
    }
    [a, b, c]
}

fn swa_tracker() -> HashMap<ComponentType, usize> {
    HashMap::from([(FULL, 0), (SWA, 0)])
}

#[test]
fn evict_walk_advances_one_allocator_mutation_per_call() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = swa_evict_chain(&mut tc);
    let mut tracker = swa_tracker();
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    tc.evict_device_start(SWA, /* request_cnt = */ 100);
    let (first, step) = tc.evict_device_next_node(SWA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    // Each internal tombstone is its own step so the allocator can observe
    // and reuse the freed slice before the walk mutates another node.
    assert_eq!(first, None);
    assert!(!tc.arena.has_device_value(a, SWA));
    assert!(tc.arena.has_device_value(b, SWA));
    assert!(tc.arena.has_device_value(c, SWA));
    assert_eq!(tracker[&SWA], 1);
    assert_eq!(device_frees[&SWA].len(), 1);
    assert!(device_frees[&SWA][0].equal(&Tensor::from_slice(&[10i64])));

    let (second, step) = tc.evict_device_next_node(SWA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    assert_eq!(second, None);
    assert!(!tc.arena.has_device_value(b, SWA));
    assert!(tc.arena.has_device_value(c, SWA));
    assert!(tc.arena.has_device_value(a, FULL));
    assert!(tc.arena.has_device_value(b, FULL));
    assert_eq!(tracker[&SWA], 2);
    assert_eq!(device_frees[&SWA].len(), 2);
    assert!(device_frees[&SWA][1].equal(&Tensor::from_slice(&[11i64])));

    let (third, step) = tc.evict_device_next_node(SWA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    assert_eq!(third, Some(tc.arena.node(c).id));
    assert_eq!(tracker[&SWA], 2);
    tc.evict_device_end(SWA);
}

#[test]
fn evict_walk_stops_at_the_token_budget() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = swa_evict_chain(&mut tc);
    let mut tracker = swa_tracker();
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    tc.evict_device_start(SWA, /* request_cnt = */ 1);
    let (next, step) = tc.evict_device_next_node(SWA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    // The first inline tombstone (a) fills the budget; b and c survive.
    assert_eq!(next, None);
    assert!(!tc.arena.has_device_value(a, SWA));
    assert!(tc.arena.has_device_value(b, SWA));
    assert!(tc.arena.has_device_value(c, SWA));
    assert_eq!(tracker[&SWA], 1);
    tc.evict_device_end(SWA);
}

#[test]
fn evict_walk_step_tracker_carries_only_the_deltas_over_the_baseline() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [_a, _b, _c] = swa_evict_chain(&mut tc);
    // A non-zero baseline stands in for prior steps' evictions.
    let baseline = HashMap::from([(FULL, 0), (SWA, 3)]);
    tc.evict_device_start(SWA, /* request_cnt = */ 100);
    let (next, step) = tc.evict_device_next_node(SWA, &baseline);
    assert_eq!(next, None);
    // One internal node tombstones: the step reports 1, not the running total.
    assert_eq!(step.tracker[&SWA], 1);
    tc.evict_device_end(SWA);
}

#[test]
fn evict_walk_skips_locked_nodes() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = swa_evict_chain(&mut tc);
    tc.arena
        .node_mut(c)
        .set_lock_ref_(ValueSlotIdx::device(SWA), 1);
    let mut tracker = swa_tracker();
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    tc.evict_device_start(SWA, /* request_cnt = */ 100);
    let (first, step) = tc.evict_device_next_node(SWA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    assert_eq!(first, None);
    assert_eq!(tracker[&SWA], 1);
    let (second, step) = tc.evict_device_next_node(SWA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    // a and b tombstone in separate steps; the locked c is invisible.
    assert_eq!(second, None);
    assert!(tc.arena.has_device_value(c, SWA));
    assert_eq!(tracker[&SWA], 2);
    let _ = (a, b);
    tc.evict_device_end(SWA);
}

#[test]
fn evict_walk_revalidates_a_delisted_cursor() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = swa_evict_chain(&mut tc);
    tc.evict_device_start(SWA, /* request_cnt = */ 100);
    // The cursor (LRU-most a) is delisted before the walk resumes.
    tc.device_lru_list_mut(SWA).remove_node(a);
    let mut tracker = swa_tracker();
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    let (next, step) = tc.evict_device_next_node(SWA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    // The walk resets to the list's LRU end (b) and performs one tombstone.
    assert_eq!(next, None);
    assert!(tc.arena.has_device_value(a, SWA));
    assert!(!tc.arena.has_device_value(b, SWA));
    tc.evict_device_end(SWA);
}

#[test]
fn evict_walk_second_call_resumes_past_the_returned_leaf() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = swa_evict_chain(&mut tc);
    let mut tracker = swa_tracker();
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    tc.evict_device_start(SWA, /* request_cnt = */ 100);
    let (first, step) = tc.evict_device_next_node(SWA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    assert_eq!(first, None);
    let (second, step) = tc.evict_device_next_node(SWA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    assert_eq!(second, None);
    let (third, step) = tc.evict_device_next_node(SWA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    assert_eq!(third, Some(tc.arena.node(c).id));
    // The driver has not delisted c yet: the pre-advanced cursor must not
    // hand the same leaf out again.
    let (fourth, step) = tc.evict_device_next_node(SWA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    assert_eq!(fourth, None);
    tc.evict_device_end(SWA);
    let _ = (a, b);
}

#[test]
fn evict_walk_start_skips_a_locked_lru_end() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = swa_evict_chain(&mut tc);
    tc.arena
        .node_mut(a)
        .set_lock_ref_(ValueSlotIdx::device(SWA), 1);
    let mut tracker = swa_tracker();
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    tc.evict_device_start(SWA, /* request_cnt = */ 100);
    let (first, step) = tc.evict_device_next_node(SWA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    // The locked LRU-end a is invisible; the walk starts at b.
    assert_eq!(first, None);
    assert!(tc.arena.has_device_value(a, SWA));
    assert!(!tc.arena.has_device_value(b, SWA));
    assert_eq!(tracker[&SWA], 1);
    let (second, step) = tc.evict_device_next_node(SWA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    assert_eq!(second, Some(tc.arena.node(c).id));
    tc.evict_device_end(SWA);
}

#[test]
fn evict_walk_revalidation_skips_a_locked_lru_end() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = swa_evict_chain(&mut tc);
    tc.evict_device_start(SWA, /* request_cnt = */ 100);
    // The cursor's node leaves the list and the new LRU end gets locked
    // before the walk resumes.
    tc.device_lru_list_mut(SWA).remove_node(a);
    tc.arena
        .node_mut(b)
        .set_lock_ref_(ValueSlotIdx::device(SWA), 1);
    let mut tracker = swa_tracker();
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    let (next, step) = tc.evict_device_next_node(SWA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    assert_eq!(next, Some(tc.arena.node(c).id));
    assert!(tc.arena.has_device_value(b, SWA));
    assert_eq!(tracker[&SWA], 0);
    tc.evict_device_end(SWA);
}

#[test]
fn evict_device_end_clears_the_walk_state() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    tc.evict_device_start(SWA, /* request_cnt = */ 1);
    tc.evict_device_end(SWA);
    // A second walk only starts cleanly when the end cleared the state.
    tc.evict_device_start(SWA, /* request_cnt = */ 1);
    tc.evict_device_end(SWA);
}

#[test]
#[should_panic(expected = "valueless node")]
fn evict_walk_asserts_a_valued_cursor_node() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, _b, _c] = swa_evict_chain(&mut tc);
    // a stays in the SWA LRU but loses its device value out of band.
    let _ = tc.arena.take_device_value(a, SWA);
    let tracker = swa_tracker();
    tc.evict_device_start(SWA, /* request_cnt = */ 100);
    tc.evict_device_next_node(SWA, &tracker);
}

#[test]
#[should_panic(expected = "Swa device eviction not started")]
fn evict_walk_requires_a_start() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let tracker = swa_tracker();
    tc.evict_device_next_node(SWA, &tracker);
}

#[test]
fn try_device_value_and_evictable_size_read_the_swa_slots() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b] = chain(&mut tc);
    store_swa_device(&mut tc, a);
    assert!(
        tc.get_component_device_value(tc.arena.node(a).id, SWA)
            .unwrap()
            .equal(&Tensor::from_slice(&[0i64]))
    );
    assert!(
        tc.get_component_device_value(tc.arena.node(b).id, SWA)
            .is_none()
    );
    assert_eq!(tc.evictable_size_(SWA), 1);
}

#[test]
fn redistribute_on_node_split_moves_the_swa_uuid_to_the_parent() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2], &[10, 11], 0, 0));
    let root = tc.arena.root();
    let node = child_of(&tc, root, &[1]);
    tc.arena.node_mut(node).swa_uuid = Some(7);
    let (parent, _) = tc.split_node_(node, /* split_len = */ 1);
    assert_eq!(node_swa_uuid(&tc, parent), Some(7));
    assert_eq!(node_swa_uuid(&tc, node), None);
}

#[test]
fn finalize_window_arithmetic_at_page_boundaries() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 2);
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let h = tc
        .arena
        .alloc_child(
            a,
            /* key = */ vec![3, 4],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let c = tc
        .arena
        .alloc_child(
            h,
            /* key = */ vec![5, 6],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    set_swa_host(&mut tc, a);
    set_swa_host(&mut tc, h);
    set_swa_device(&mut tc, c);
    // The window of 4 lands exactly on the c/h page boundary sum: h is
    // counted, the page above it is not.
    let out = finalize(&tc, &swa_component(4), c, /* prior = */ 0);
    assert_eq!(out.swa_host_hit_length, 2);
}

#[test]
fn new_leaf_after_a_cached_prefix_splits_at_the_leaf_relative_boundary() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[10, 11, 12], 0, 0));
    let root = tc.arena.root();
    let a = child_of(&tc, root, &[1]);
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4, 5, 6],
        &[20, 21, 22, 23, 24, 25],
        /* prev_prefix_len = */ 3,
        /* swa_evicted_seqlen = */ 4,
    ));
    assert_eq!(result.prefix_len, 3);
    // The leaf starts at prefix 3, so the boundary at seqlen 4 is
    // leaf-relative offset 1: parent [4] tombstone, tail [5, 6] rebuilt.
    let p = child_of(&tc, a, &[4]);
    let leaf = child_of(&tc, p, &[5]);
    assert_eq!(tc.arena.node(p).key, vec![4]);
    assert_eq!(tc.arena.node(leaf).key, vec![5, 6]);
    assert!(
        tc.arena
            .device_value(p, FULL)
            .equal(&Tensor::from_slice(&[23i64]))
    );
    assert!(
        tc.arena
            .device_value(leaf, FULL)
            .equal(&Tensor::from_slice(&[24i64, 25]))
    );
    assert!(result.cache_actions.iter().any(|action| matches!(
        action,
        CacheAction::SwaRebuild { node_id, source_value }
            if *node_id == tc.arena.node(leaf).id && source_value.equal(&Tensor::from_slice(&[24i64, 25]))
    )));
    assert!(!result.cache_actions.iter().any(|action| matches!(
        action,
        CacheAction::SwaRebuild { node_id, .. } if *node_id == tc.arena.node(p).id
    )));
}

#[test]
fn new_leaf_boundary_split_and_window_cap_compose_in_one_commit() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2], &[10, 11], 0, 0));
    let root = tc.arena.root();
    let a = child_of(&tc, root, &[1]);
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4, 5, 6, 7],
        &[20, 21, 22, 23, 24, 25, 26],
        /* prev_prefix_len = */ 2,
        /* swa_evicted_seqlen = */ 3,
    ));
    // Boundary split first ([3] tombstone), then the window cap splits the
    // in-window run into [4, 5] + [6, 7], rebuilt older-prefix-first.
    let p = child_of(&tc, a, &[3]);
    let capped = child_of(&tc, p, &[4]);
    let leaf = child_of(&tc, capped, &[6]);
    assert_eq!(tc.arena.node(p).key, vec![3]);
    assert_eq!(tc.arena.node(capped).key, vec![4, 5]);
    assert_eq!(tc.arena.node(leaf).key, vec![6, 7]);
    let rebuilds: Vec<NodeId> = result
        .cache_actions
        .iter()
        .filter_map(|action| match action {
            CacheAction::SwaRebuild { node_id, .. } => Some(*node_id),
            _ => None,
        })
        .collect();
    assert_eq!(
        rebuilds,
        vec![tc.arena.node(capped).id, tc.arena.node(leaf).id]
    );
    assert!(result.cache_actions.iter().any(|action| matches!(
        action,
        CacheAction::SwaRebuild { node_id, source_value }
            if *node_id == tc.arena.node(capped).id && source_value.equal(&Tensor::from_slice(&[23i64, 24]))
    )));
    assert!(result.cache_actions.iter().any(|action| matches!(
        action,
        CacheAction::SwaRebuild { node_id, source_value }
            if *node_id == tc.arena.node(leaf).id && source_value.equal(&Tensor::from_slice(&[25i64, 26]))
    )));
}

#[test]
fn finalize_counts_a_dual_tier_node_as_a_device_hit() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b] = chain(&mut tc);
    set_swa_device(&mut tc, a);
    set_swa_host(&mut tc, a);
    set_swa_device(&mut tc, b);
    set_swa_host(&mut tc, b);
    let out = finalize(&tc, &swa_component(2), b, /* prior = */ 0);
    assert_eq!(out.swa_host_hit_length, 0);
}

#[test]
fn finalize_at_the_root_leaves_the_host_hit_untouched() {
    let tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let root = tc.arena.root();
    let out = finalize(&tc, &swa_component(2), root, /* prior = */ 0);
    assert_eq!(out.swa_host_hit_length, 0);
}

#[test]
fn match_prefix_with_an_empty_key_on_a_swa_core_is_a_clean_miss() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2], &[10, 11], 0, 0));
    let result = tc.match_prefix(&MatchPrefixParams {
        key: &Vec::new(),
        namespace: Default::default(),
    });
    assert_eq!(result.device_indices.size()[0], 0);
    assert_eq!(result.swa_host_hit_length, 0);
    let root = tc.arena.root();
    assert_eq!(result.best_match_node_id, tc.arena.node(root).id);
}

#[test]
fn refresh_window_extends_by_a_full_page_beyond_the_sliding_window() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 4);
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2, 3, 4],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let b = tc
        .arena
        .alloc_child(
            a,
            /* key = */ vec![5, 6, 7, 8],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let c = tc
        .arena
        .alloc_child(
            b,
            /* key = */ vec![9, 10, 11, 12],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    for node in [c, b, a] {
        set_swa_device(&mut tc, node);
        tc.device_lru_list_mut(SWA).insert_mru(node);
    }
    // The walk window is sliding_window_size + page_size = 6: c and b
    // re-rank deepest first; a stays put.
    swa_component(2).refresh_lru(&mut tc, LRURefreshPhase::MatchEnd, c);
    assert_eq!(swa_lru_order(&tc), vec![c, b, a]);
}

#[test]
fn window_cap_skips_a_page_misaligned_leaf() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 2);
    let root = tc.arena.root();
    let leaf = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2, 3, 4, 5],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    // split_at = 5 - 2 = 3 is not page-aligned: the cap is skipped.
    let capped = swa_component(2).maybe_split_leaf_for_swa_lock_(&mut tc, leaf);
    assert_eq!(capped, None);
    assert_eq!(tc.arena.node(leaf).key, vec![1, 2, 3, 4, 5]);
    assert_eq!(tc.arena.node(leaf).parent(), root);
}

#[test]
fn new_leaf_with_the_boundary_above_its_start_skips_the_split() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[10, 11, 12], 0, 0));
    let root = tc.arena.root();
    let a = child_of(&tc, root, &[1]);
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4, 5],
        &[20, 21, 22, 23, 24],
        /* prev_prefix_len = */ 3,
        /* swa_evicted_seqlen = */ 2,
    ));
    assert_eq!(result.prefix_len, 3);
    // The boundary (2) sits above the leaf start (3): the whole leaf is
    // in-window and stays unsplit.
    let leaf = child_of(&tc, a, &[4]);
    assert_eq!(tc.arena.node(leaf).key, vec![4, 5]);
    assert!(result.cache_actions.iter().any(|action| matches!(
        action,
        CacheAction::SwaRebuild { node_id, source_value }
            if *node_id == tc.arena.node(leaf).id && source_value.equal(&Tensor::from_slice(&[23i64, 24]))
    )));
}

#[test]
fn insert_overlap_straddling_with_a_partial_prev_prefix_recovers_the_tail() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4],
        &[10, 11, 12, 13],
        0,
        0,
    ));
    let root = tc.arena.root();
    let node = child_of(&tc, root, &[1]);
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4],
        &[20, 21, 22, 23],
        /* prev_prefix_len = */ 1,
        /* swa_evicted_seqlen = */ 2,
    ));
    // prev covers only one atom, so the straddle recovery still runs.
    let parent = child_of(&tc, root, &[1]);
    assert_eq!(tc.arena.node(parent).key, vec![1, 2]);
    assert_eq!(tc.arena.node(node).key, vec![3, 4]);
    assert!(
        tc.arena
            .device_value(parent, FULL)
            .equal(&Tensor::from_slice(&[10i64, 11]))
    );
    assert!(
        tc.arena
            .device_value(node, FULL)
            .equal(&Tensor::from_slice(&[22i64, 23]))
    );
    let [
        CacheAction::FreeDeviceKVFullOnly(old_tail),
        CacheAction::SwaRebuild {
            node_id,
            source_value,
        },
        CacheAction::FreeDeviceKV(duplicates),
    ] = result.cache_actions.as_slice()
    else {
        panic!(
            "expected FreeDeviceKVFullOnly, SwaRebuild, FreeDeviceKV, got {:?}",
            action_kinds(&result.cache_actions)
        );
    };
    assert!(old_tail[0].equal(&Tensor::from_slice(&[12i64, 13])));
    assert_eq!(*node_id, tc.arena.node(node).id);
    assert!(source_value.equal(&Tensor::from_slice(&[22i64, 23])));
    // The request already owns the first prev_prefix_len token; only the
    // stretch between prev and the boundary is duplicate.
    assert!(duplicates[0].equal(&Tensor::from_slice(&[21i64])));
}

#[test]
fn insert_overlap_straddling_recovers_across_page_two() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 2);
    tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4],
        &[10, 11, 12, 13],
        0,
        0,
    ));
    let root = tc.arena.root();
    let node = child_of(&tc, root, &[1, 2]);
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4],
        &[20, 21, 22, 23],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 2,
    ));
    let parent = child_of(&tc, root, &[1, 2]);
    assert_eq!(tc.arena.node(parent).key, vec![1, 2]);
    assert_eq!(tc.arena.node(node).key, vec![3, 4]);
    assert!(
        tc.arena
            .device_value(node, FULL)
            .equal(&Tensor::from_slice(&[22i64, 23]))
    );
    assert!(result.cache_actions.iter().any(|action| matches!(
        action,
        CacheAction::SwaRebuild { node_id, source_value }
            if *node_id == tc.arena.node(node).id && source_value.equal(&Tensor::from_slice(&[22i64, 23]))
    )));
}

#[test]
fn reinsert_straddling_recovers_across_page_two() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 2);
    tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4],
        &[10, 11, 12, 13],
        0,
        0,
    ));
    let root = tc.arena.root();
    let node = child_of(&tc, root, &[1, 2]);
    evict_full(&mut tc, node, /* remaining_size = */ 0);
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4],
        &[20, 21, 22, 23],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 2,
    ));
    let parent = child_of(&tc, root, &[1, 2]);
    assert_eq!(tc.arena.node(parent).key, vec![1, 2]);
    assert_eq!(tc.arena.node(node).key, vec![3, 4]);
    assert!(
        tc.arena
            .device_value(parent, FULL)
            .equal(&Tensor::from_slice(&[20i64, 21]))
    );
    assert!(
        tc.arena
            .device_value(node, FULL)
            .equal(&Tensor::from_slice(&[22i64, 23]))
    );
    assert!(result.cache_actions.iter().any(|action| matches!(
        action,
        CacheAction::SwaRebuild { node_id, source_value }
            if *node_id == tc.arena.node(node).id && source_value.equal(&Tensor::from_slice(&[22i64, 23]))
    )));
    assert!(!result.cache_actions.iter().any(|action| matches!(
        action,
        CacheAction::SwaRebuild { node_id, .. } if *node_id == tc.arena.node(parent).id
    )));
}

#[test]
#[should_panic(expected = "swa_evicted_seqlen must be page-aligned")]
fn reinsert_rejects_a_page_misaligned_boundary() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 2);
    tc.insert(&insert_params_swa(&vec![1, 2], &[10, 11], 0, 0));
    let root = tc.arena.root();
    let node = child_of(&tc, root, &[1, 2]);
    evict_full(&mut tc, node, /* remaining_size = */ 0);
    tc.insert(&insert_params_swa(
        &vec![1, 2],
        &[20, 21],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 1,
    ));
}

#[test]
#[should_panic(expected = "tombstone Swa lock_ref should be 0 on unevict")]
fn reinsert_rejects_a_locked_tombstone() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    tc.insert(&insert_params_swa(&vec![1, 2], &[10, 11], 0, 0));
    let root = tc.arena.root();
    let node = child_of(&tc, root, &[1]);
    evict_full(&mut tc, node, /* remaining_size = */ 0);
    tc.arena
        .node_mut(node)
        .set_lock_ref_(ValueSlotIdx::device(SWA), 1);
    tc.insert(&insert_params_swa(&vec![1, 2], &[20, 21], 0, 0));
}

fn set_full_host(tc: &mut UnifiedTreeCore<Vec<i64>>, node: NodeIdx_) {
    let len = tc.arena.node(node).key.atom_len();
    tc.arena
        .set_host_value(node, FULL, Tensor::from_slice(&vec![0i64; len]));
}

fn host_drive_state() -> (
    HashMap<ComponentType, usize>,
    HashMap<ComponentType, Vec<Tensor>>,
    HashMap<ComponentType, Vec<Tensor>>,
) {
    (
        HashMap::from([(FULL, 0), (SWA, 0)]),
        HashMap::new(),
        HashMap::new(),
    )
}

#[test]
fn host_drive_tombstones_internal_nodes_and_evicts_host_leaves() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [p, c] = chain(&mut tc);
    set_full_host(&mut tc, p);
    set_swa_host(&mut tc, p);
    set_full_host(&mut tc, c);
    set_swa_host(&mut tc, c);
    // p enters the LRU first so the walk reaches it before c.
    tc.host_lru_list_mut(SWA).insert_mru(p);
    tc.host_lru_list_mut(SWA).insert_mru(c);
    tc.evictable_host_leaves.add(c);
    let (mut tr, mut df, mut hf) = host_drive_state();
    accumulate_step(
        tc.drive_host_eviction(SWA, /* num_tokens = */ 100),
        &mut tr,
        &mut df,
        &mut hf,
    );
    // p: private tombstone (SWA host only); c: atomic H-leaf teardown.
    assert_eq!(tr[&SWA], 2);
    assert_eq!(tr[&FULL], 1);
    assert_eq!(hf[&SWA].len(), 2);
    assert_eq!(hf[&FULL].len(), 1);
    assert!(tc.arena.has_host_value(p, FULL));
    assert!(!tc.arena.has_host_value(p, SWA));
    assert_eq!(tc.arena.len(), 2);
    assert!(tc.evictable_host_leaves.contains(p));
    assert_eq!(tc.host_lru_list(SWA).len(), 0);
    tc.sanity_check(&[], &[]);
}

#[test]
fn host_drive_skips_host_locked_nodes() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let root = tc.arena.root();
    let locked = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let victim = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    for node in [locked, victim] {
        set_full_host(&mut tc, node);
        set_swa_host(&mut tc, node);
        tc.host_lru_list_mut(SWA).insert_mru(node);
    }
    tc.arena
        .node_mut(locked)
        .set_lock_ref_(ValueSlotIdx::host(SWA), 1);
    tc.evictable_host_leaves.add(victim);
    let (mut tr, mut df, mut hf) = host_drive_state();
    accumulate_step(
        tc.drive_host_eviction(SWA, /* num_tokens = */ 100),
        &mut tr,
        &mut df,
        &mut hf,
    );
    // The locked LRU-most node is passed over; the unlocked one is evicted.
    assert!(tc.arena.has_host_value(locked, SWA));
    assert!(tc.host_lru_list(SWA).in_list(Some(locked)));
    assert_eq!(tr[&SWA], 1);
    assert_eq!(tr[&FULL], 1);
    assert_eq!(tc.arena.len(), 2);
    tc.sanity_check(&[], &[]);
}

#[test]
fn host_drive_stops_at_the_token_budget_consuming_lru_first() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let root = tc.arena.root();
    let h1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let h2 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    for node in [h1, h2] {
        set_full_host(&mut tc, node);
        set_swa_host(&mut tc, node);
        tc.host_lru_list_mut(SWA).insert_mru(node);
        tc.evictable_host_leaves.add(node);
    }
    let (mut tr, mut df, mut hf) = host_drive_state();
    accumulate_step(
        tc.drive_host_eviction(SWA, /* num_tokens = */ 1),
        &mut tr,
        &mut df,
        &mut hf,
    );
    // The LRU-most h1 fills the budget; h2 survives untouched.
    assert_eq!(tr[&SWA], 1);
    assert!(tc.evictable_host_leaves.contains(h2));
    assert!(tc.host_lru_list(SWA).in_list(Some(h2)));
    assert_eq!(tc.arena.len(), 2);
    tc.sanity_check(&[], &[]);
}

#[test]
fn host_drive_ends_when_the_next_candidate_left_the_lru() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let root = tc.arena.root();
    let p = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let c = tc
        .arena
        .alloc_child(
            p,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let s = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![9],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    // p holds only a transient SWA host chunk: c's teardown tombstone-walks
    // p away, so the captured next candidate leaves the list mid-drive.
    set_swa_host(&mut tc, p);
    set_full_host(&mut tc, c);
    set_swa_host(&mut tc, c);
    set_full_host(&mut tc, s);
    set_swa_host(&mut tc, s);
    tc.host_lru_list_mut(SWA).insert_mru(c);
    tc.host_lru_list_mut(SWA).insert_mru(p);
    tc.host_lru_list_mut(SWA).insert_mru(s);
    tc.evictable_host_leaves.add(c);
    tc.evictable_host_leaves.add(s);
    let (mut tr, mut df, mut hf) = host_drive_state();
    accumulate_step(
        tc.drive_host_eviction(SWA, /* num_tokens = */ 100),
        &mut tr,
        &mut df,
        &mut hf,
    );
    // The walk stops at the vanished p; the more-recent s is never reached.
    assert!(tc.arena.has_host_value(s, SWA));
    assert!(tc.host_lru_list(SWA).in_list(Some(s)));
    assert_eq!(tr[&SWA], 2);
    assert_eq!(tr[&FULL], 1);
    assert_eq!(tc.arena.len(), 2);
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "has no host value")]
fn host_drive_panics_on_an_lru_member_without_a_swa_host_value() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let root = tc.arena.root();
    let n = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    set_full_host(&mut tc, n);
    tc.host_lru_list_mut(SWA).insert_mru(n);
    tc.drive_host_eviction(SWA, /* num_tokens = */ 100);
}

#[test]
fn backup_storage_transfers_carry_trailing_page_keys() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    tc.arena
        .set_host_value(a, SWA, Tensor::from_slice(&[21i64]));
    tc.arena.node_mut(a).hash_value = Some(vec!["h0".to_string()]);
    let transfers = swa_component(4)
        .build_hicache_transfers(
            &tc,
            a,
            CacheTransferPhase::BackupStorage,
            /* mamba_pool_idx = */ None,
            /* host_indices = */ None,
            /* token_ids = */ None,
            /* prefetch_tokens = */ 0,
            /* last_hash = */ None,
        )
        .unwrap()
        .unwrap();
    assert_eq!(transfers.len(), 1);
    assert_eq!(transfers[0].name, PoolName::Swa);
    assert!(
        transfers[0]
            .host_indices
            .as_ref()
            .unwrap()
            .equal(&Tensor::from_slice(&[21i64]))
    );
    assert_eq!(transfers[0].keys, Some(vec!["h0".to_string()]));
    assert_eq!(transfers[0].hit_policy, PoolHitPolicy::TrailingPages);
}

#[test]
fn backup_storage_is_none_without_host_value_or_hashes() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    let build = |tc: &UnifiedTreeCore<Vec<i64>>| {
        swa_component(4).build_hicache_transfers(
            tc,
            a,
            CacheTransferPhase::BackupStorage,
            /* mamba_pool_idx = */ None,
            /* host_indices = */ None,
            /* token_ids = */ None,
            /* prefetch_tokens = */ 0,
            /* last_hash = */ None,
        )
    };
    assert!(build(&tc).unwrap().is_none());
    tc.arena
        .set_host_value(a, SWA, Tensor::from_slice(&[21i64]));
    tc.arena.node_mut(a).hash_value = None;
    assert!(build(&tc).unwrap().is_none());
}

#[test]
fn build_transfers_are_gated_off_until_the_swa_host_pool_is_wired() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    tc.set_hicache_enabled();
    let [a] = chain::<1>(&mut tc);
    set_swa_device(&mut tc, a);
    for phase in [CacheTransferPhase::BackupHost, CacheTransferPhase::LoadBack] {
        let transfers = swa_component(4)
            .build_hicache_transfers(
                &tc, a, phase, /* mamba_pool_idx = */ None, /* host_indices = */ None,
                /* token_ids = */ None, /* prefetch_tokens = */ 0,
                /* last_hash = */ None,
            )
            .unwrap();
        assert!(transfers.is_none());
    }
    // Wiring the pool opens the gate.
    tc.set_has_swa_host_pool();
    let transfers = swa_component(4)
        .build_hicache_transfers(
            &tc,
            a,
            CacheTransferPhase::BackupHost,
            /* mamba_pool_idx = */ None,
            /* host_indices = */ None,
            /* token_ids = */ None,
            /* prefetch_tokens = */ 0,
            /* last_hash = */ None,
        )
        .unwrap();
    assert!(transfers.is_some());
}

#[test]
fn backup_host_build_wraps_the_device_value_as_int64() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    tc.arena
        .set_device_value(a, SWA, Tensor::from_slice(&[5i32]));
    let transfers = swa_component(4)
        .build_hicache_transfers(
            &tc,
            a,
            CacheTransferPhase::BackupHost,
            /* mamba_pool_idx = */ None,
            /* host_indices = */ None,
            /* token_ids = */ None,
            /* prefetch_tokens = */ 0,
            /* last_hash = */ None,
        )
        .unwrap()
        .unwrap();
    assert_eq!(transfers.len(), 1);
    let xfer = &transfers[0];
    assert_eq!(xfer.name, PoolName::Swa);
    let device_indices = xfer.device_indices.as_ref().unwrap();
    assert_eq!(device_indices.kind(), Kind::Int64);
    assert!(device_indices.equal(&Tensor::from_slice(&[5i64])));
    assert!(xfer.host_indices.is_none());
    assert!(xfer.nodes_to_load.is_none());
}

#[test]
fn backup_host_build_returns_none_for_a_tombstone() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    let transfers = swa_component(4)
        .build_hicache_transfers(
            &tc,
            a,
            CacheTransferPhase::BackupHost,
            /* mamba_pool_idx = */ None,
            /* host_indices = */ None,
            /* token_ids = */ None,
            /* prefetch_tokens = */ 0,
            /* last_hash = */ None,
        )
        .unwrap();
    assert!(transfers.is_none());
}

#[test]
fn backup_spec_reads_the_swa_value_recovered_by_an_earlier_action() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    tc.arena
        .set_device_value(a, FULL, Tensor::from_slice(&[9i64]));
    let a_id = tc.arena.node(a).id;
    // The tombstone carries no SWA transfer into the backup spec.
    let (_, xfers) = tc.build_backup_spec(a_id);
    assert!(xfers.is_empty());
    // The cache resolves the recover/rebuild action, then rebuilds the spec:
    // the deferred read now captures the freshly stored SWA value.
    tc.set_component_device_value(a_id, SWA, Tensor::from_slice(&[50i64]));
    let (_, xfers) = tc.build_backup_spec(a_id);
    let swa_xfer = &xfers[&SWA][0];
    assert!(
        swa_xfer
            .device_indices
            .as_ref()
            .unwrap()
            .equal(&Tensor::from_slice(&[50i64]))
    );
}

#[test]
fn load_back_build_collects_host_only_nodes_within_the_window() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a, b, c] = chain::<3>(&mut tc);
    set_swa_device(&mut tc, a);
    tc.arena
        .set_host_value(b, SWA, Tensor::from_slice(&[21i64]));
    tc.arena
        .set_host_value(c, SWA, Tensor::from_slice(&[22i64]));
    let transfers = swa_component(4)
        .build_hicache_transfers(
            &tc,
            c,
            CacheTransferPhase::LoadBack,
            /* mamba_pool_idx = */ None,
            /* host_indices = */ None,
            /* token_ids = */ None,
            /* prefetch_tokens = */ 0,
            /* last_hash = */ None,
        )
        .unwrap()
        .unwrap();
    assert_eq!(transfers.len(), 1);
    let xfer = &transfers[0];
    assert_eq!(xfer.name, PoolName::Swa);
    // Ancestor-first; a's device value is skipped, not collected.
    assert!(
        xfer.host_indices
            .as_ref()
            .unwrap()
            .equal(&Tensor::from_slice(&[21i64, 22]))
    );
    assert!(xfer.device_indices.is_none());
    assert_eq!(
        xfer.nodes_to_load,
        Some(vec![tc.arena.node(b).id, tc.arena.node(c).id])
    );
}

#[test]
fn load_back_build_stops_at_the_window_boundary() {
    let mut tc = swa_core(/* window = */ 2, /* page_size = */ 1);
    let [a, b, c] = chain::<3>(&mut tc);
    tc.arena
        .set_host_value(a, SWA, Tensor::from_slice(&[20i64]));
    tc.arena
        .set_host_value(b, SWA, Tensor::from_slice(&[21i64]));
    tc.arena
        .set_host_value(c, SWA, Tensor::from_slice(&[22i64]));
    let transfers = swa_component(2)
        .build_hicache_transfers(
            &tc,
            c,
            CacheTransferPhase::LoadBack,
            /* mamba_pool_idx = */ None,
            /* host_indices = */ None,
            /* token_ids = */ None,
            /* prefetch_tokens = */ 0,
            /* last_hash = */ None,
        )
        .unwrap()
        .unwrap();
    // The two-token window covers c and b; a stays out of the transfer.
    let xfer = &transfers[0];
    assert!(
        xfer.host_indices
            .as_ref()
            .unwrap()
            .equal(&Tensor::from_slice(&[21i64, 22]))
    );
    assert_eq!(
        xfer.nodes_to_load,
        Some(vec![tc.arena.node(b).id, tc.arena.node(c).id])
    );
}

#[test]
fn load_back_build_returns_none_when_the_window_is_on_device() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a, b] = chain::<2>(&mut tc);
    set_swa_device(&mut tc, a);
    set_swa_device(&mut tc, b);
    let transfers = swa_component(4)
        .build_hicache_transfers(
            &tc,
            b,
            CacheTransferPhase::LoadBack,
            /* mamba_pool_idx = */ None,
            /* host_indices = */ None,
            /* token_ids = */ None,
            /* prefetch_tokens = */ 0,
            /* last_hash = */ None,
        )
        .unwrap();
    assert!(transfers.is_none());
}

#[test]
fn load_back_build_rejects_a_bare_window_node() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    assert!(matches!(
        swa_component(4).build_hicache_transfers(
            &tc,
            a,
            CacheTransferPhase::LoadBack,
            /* mamba_pool_idx = */ None,
            /* host_indices = */ None,
            /* token_ids = */ None,
            /* prefetch_tokens = */ 0,
            /* last_hash = */ None,
        ),
        Err(TreeCoreRuntimeError::SwaLoadBackMissingValue { node_id })
            if node_id == tc.arena.node(a).id
    ));
}

#[test]
fn fallible_load_back_boundaries_reject_a_bare_window_node() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    tc.arena
        .set_device_value(a, FULL, Tensor::from_slice(&[10i64]));
    let node_id = tc.arena.node(a).id;

    assert!(matches!(
        tc.try_build_hicache_transfers(
            SWA,
            node_id,
            CacheTransferPhase::LoadBack,
            /* host_indices = */ None,
            /* token_ids = */ None,
            /* prefetch_tokens = */ 0,
            /* last_hash = */ None,
        ),
        Err(TreeCoreRuntimeError::SwaLoadBackMissingValue { node_id: missing })
            if missing == node_id
    ));
    assert!(matches!(
        tc.try_build_load_back_spec(node_id, /* req = */ None),
        Err(TreeCoreRuntimeError::SwaLoadBackMissingValue { node_id: missing })
            if missing == node_id
    ));
}

#[test]
fn load_back_commit_attaches_chunks_and_emits_the_rebuild_action() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a, b] = chain::<2>(&mut tc);
    tc.arena
        .set_device_value(a, FULL, Tensor::from_slice(&[40i64]));
    tc.arena
        .set_device_value(b, FULL, Tensor::from_slice(&[41i64]));
    tc.arena
        .set_host_value(a, SWA, Tensor::from_slice(&[21i64]));
    tc.arena
        .set_host_value(b, SWA, Tensor::from_slice(&[22i64]));
    let mut cache_actions = Vec::new();
    let transfer = PoolTransfer {
        name: PoolName::Swa,
        host_indices: Some(Tensor::from_slice(&[21i64, 22])),
        device_indices: Some(Tensor::from_slice(&[50i64, 51])),
        nodes_to_load: Some(vec![tc.arena.node(a).id, tc.arena.node(b).id]),
        ..Default::default()
    };
    swa_component(4).commit_hicache_transfer(
        &mut tc,
        b,
        CacheTransferPhase::LoadBack,
        vec![transfer],
        &mut cache_actions,
        /* insert_result = */ None,
        /* pool_storage_result = */ None,
    );
    assert!(
        tc.arena
            .device_value(a, SWA)
            .equal(&Tensor::from_slice(&[50i64]))
    );
    assert!(
        tc.arena
            .device_value(b, SWA)
            .equal(&Tensor::from_slice(&[51i64]))
    );
    // The attach path restamps the SWA device LRU and evictable size.
    assert!(tc.device_lru_list(SWA).in_list(Some(a)));
    assert!(tc.device_lru_list(SWA).in_list(Some(b)));
    assert_eq!(tc.swa_evictable_size(), 2);
    assert_eq!(cache_actions.len(), 1);
    let CacheAction::RebuildFullToSwaMapping {
        full_indices,
        swa_indices,
    } = &cache_actions[0]
    else {
        panic!("expected a RebuildFullToSwaMapping action");
    };
    assert_eq!(full_indices.len(), 2);
    assert!(full_indices[0].equal(&Tensor::from_slice(&[40i64])));
    assert!(full_indices[1].equal(&Tensor::from_slice(&[41i64])));
    assert_eq!(swa_indices.len(), 2);
    assert!(swa_indices[0].equal(&Tensor::from_slice(&[50i64])));
    assert!(swa_indices[1].equal(&Tensor::from_slice(&[51i64])));
}

#[test]
#[should_panic(expected = "SWA LOAD_BACK commit requires device indices")]
fn load_back_commit_panics_without_device_indices() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    let transfer = PoolTransfer {
        name: PoolName::Swa,
        host_indices: Some(Tensor::from_slice(&[21i64])),
        nodes_to_load: Some(vec![tc.arena.node(a).id]),
        ..Default::default()
    };
    swa_component(4).commit_hicache_transfer(
        &mut tc,
        a,
        CacheTransferPhase::LoadBack,
        vec![transfer],
        &mut Vec::new(),
        /* insert_result = */ None,
        /* pool_storage_result = */ None,
    );
}

#[test]
#[should_panic(expected = "left == right")]
fn load_back_commit_asserts_the_loaded_length_matches_the_host_indices() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    tc.arena
        .set_device_value(a, FULL, Tensor::from_slice(&[40i64]));
    tc.arena
        .set_host_value(a, SWA, Tensor::from_slice(&[21i64]));
    // Two host indices but only one loaded token: the commit must fail loudly.
    let transfer = PoolTransfer {
        name: PoolName::Swa,
        host_indices: Some(Tensor::from_slice(&[21i64, 22])),
        device_indices: Some(Tensor::from_slice(&[50i64, 51])),
        nodes_to_load: Some(vec![tc.arena.node(a).id]),
        ..Default::default()
    };
    swa_component(4).commit_hicache_transfer(
        &mut tc,
        a,
        CacheTransferPhase::LoadBack,
        vec![transfer],
        &mut Vec::new(),
        /* insert_result = */ None,
        /* pool_storage_result = */ None,
    );
}

#[test]
fn backup_host_commit_sets_the_host_value_once() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    for host in [30i64, 31] {
        let transfer = PoolTransfer {
            name: PoolName::Swa,
            host_indices: Some(Tensor::from_slice(&[host])),
            ..Default::default()
        };
        swa_component(4).commit_hicache_transfer(
            &mut tc,
            a,
            CacheTransferPhase::BackupHost,
            vec![transfer],
            &mut Vec::new(),
            /* insert_result = */ None,
            /* pool_storage_result = */ None,
        );
    }
    // The second commit is a no-op: the first host value sticks.
    assert!(
        tc.arena
            .host_value(a, SWA)
            .equal(&Tensor::from_slice(&[30i64]))
    );
}

#[test]
fn backup_host_commit_ignores_transfers_without_host_indices() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    for transfers in [vec![], vec![PoolTransfer::default()]] {
        swa_component(4).commit_hicache_transfer(
            &mut tc,
            a,
            CacheTransferPhase::BackupHost,
            transfers,
            &mut Vec::new(),
            /* insert_result = */ None,
            /* pool_storage_result = */ None,
        );
    }
    assert!(!tc.arena.has_host_value(a, SWA));
}

#[test]
fn backup_storage_commit_is_a_noop() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    let mut cache_actions = Vec::new();
    swa_component(4).commit_hicache_transfer(
        &mut tc,
        a,
        CacheTransferPhase::BackupStorage,
        vec![],
        &mut cache_actions,
        /* insert_result = */ None,
        /* pool_storage_result = */ None,
    );
    assert!(cache_actions.is_empty());
}

#[test]
fn commit_hicache_transfers_routes_to_the_component() {
    // Underloaded prefetch through the core dispatcher releases the buffer.
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [_a] = chain::<1>(&mut tc);
    let mut cache_actions = Vec::new();
    let root = tc.arena.root();
    tc.commit_hicache_transfers(
        tc.arena.node(root).id,
        CacheTransferPhase::Prefetch,
        HashMap::from([(
            SWA,
            vec![PoolTransfer {
                name: PoolName::Swa,
                host_indices: Some(Tensor::from_slice(&[30i64])),
                ..Default::default()
            }],
        )]),
        &mut cache_actions,
        /* insert_result = */ None,
        /* pool_storage_result = */ None,
    );
    assert_eq!(cache_actions.len(), 1);
}

#[test]
fn prefetch_build_wraps_the_host_buffer_with_placeholder_keys() {
    let tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let transfers = swa_component(4)
        .build_hicache_transfers(
            &tc,
            tc.arena.root(),
            CacheTransferPhase::Prefetch,
            /* mamba_pool_idx = */ None,
            /* host_indices = */ Some(Tensor::from_slice(&[30i64, 31])),
            /* token_ids = */ None,
            /* prefetch_tokens = */ 0,
            /* last_hash = */ None,
        )
        .unwrap()
        .unwrap();
    assert_eq!(transfers.len(), 1);
    assert_eq!(
        transfers[0].keys,
        Some(vec![
            "__placeholder__".to_string(),
            "__placeholder__".to_string()
        ])
    );
    assert_eq!(transfers[0].hit_policy, PoolHitPolicy::TrailingPages);
    assert!(
        transfers[0]
            .host_indices
            .as_ref()
            .unwrap()
            .equal(&Tensor::from_slice(&[30i64, 31]))
    );
}

#[test]
fn prefetch_commit_drops_the_whole_window_when_underloaded() {
    // loaded_pages (1) < window_require_pages (2): all-or-nothing releases
    // the full buffer and attaches nothing.
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    let mut cache_actions = Vec::new();
    let mut insert_result = InsertResult {
        total_len: 2,
        inserted_host_node: Some(tc.arena.node(a).id),
        ..InsertResult::default()
    };
    let storage_result = PoolTransferResult {
        kv_hit_pages: 2,
        extra_pool_hit_pages: HashMap::from([(PoolName::Swa, 1)]),
    };
    let root = tc.arena.root();
    swa_component(4).commit_hicache_transfer(
        &mut tc,
        root,
        CacheTransferPhase::Prefetch,
        vec![PoolTransfer {
            name: PoolName::Swa,
            host_indices: Some(Tensor::from_slice(&[30i64, 31])),
            ..Default::default()
        }],
        &mut cache_actions,
        Some(&mut insert_result),
        Some(&storage_result),
    );
    assert!(!tc.arena.node(a).has_host_value(SWA));
    assert_eq!(cache_actions.len(), 1);
    let CacheAction::FreeComponentHostSlot { host_indices, .. } = &cache_actions[0] else {
        panic!("expected a host free");
    };
    assert!(host_indices[0].equal(&Tensor::from_slice(&[30i64, 31])));
}

#[test]
fn prefetch_commit_without_a_target_releases_the_whole_buffer() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    let storage_result = PoolTransferResult {
        kv_hit_pages: 2,
        extra_pool_hit_pages: HashMap::from([(PoolName::Swa, 2)]),
    };
    let root = tc.arena.root();
    // No insert result at all: the buffer has no anchor and fully releases.
    let mut cache_actions = Vec::new();
    swa_component(4).commit_hicache_transfer(
        &mut tc,
        root,
        CacheTransferPhase::Prefetch,
        vec![PoolTransfer {
            name: PoolName::Swa,
            host_indices: Some(Tensor::from_slice(&[30i64, 31])),
            ..Default::default()
        }],
        &mut cache_actions,
        /* insert_result = */ None,
        Some(&storage_result),
    );
    assert_eq!(cache_actions.len(), 1);
    let CacheAction::FreeComponentHostSlot { host_indices, .. } = &cache_actions[0] else {
        panic!("expected a host free");
    };
    assert!(host_indices[0].equal(&Tensor::from_slice(&[30i64, 31])));

    // An insert result without an inserted host node releases the same way.
    let mut insert_result = InsertResult {
        total_len: 2,
        inserted_host_node: None,
        ..InsertResult::default()
    };
    let mut cache_actions = Vec::new();
    swa_component(4).commit_hicache_transfer(
        &mut tc,
        root,
        CacheTransferPhase::Prefetch,
        vec![PoolTransfer {
            name: PoolName::Swa,
            host_indices: Some(Tensor::from_slice(&[30i64, 31])),
            ..Default::default()
        }],
        &mut cache_actions,
        Some(&mut insert_result),
        Some(&storage_result),
    );
    assert_eq!(cache_actions.len(), 1);
    let CacheAction::FreeComponentHostSlot { host_indices, .. } = &cache_actions[0] else {
        panic!("expected a host free");
    };
    assert!(host_indices[0].equal(&Tensor::from_slice(&[30i64, 31])));
    assert!(!tc.arena.node(a).has_host_value(SWA));
}

#[test]
fn prefetch_commit_releases_the_out_of_path_prefix() {
    // root -> a -> b -> c, one token each; anchor b, target c: the loaded
    // window spans two tokens but the leaf->anchor path covers only c's one.
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a, b, c] = chain::<3>(&mut tc);
    let mut cache_actions = Vec::new();
    let mut insert_result = InsertResult {
        total_len: 3,
        inserted_host_node: Some(tc.arena.node(c).id),
        ..InsertResult::default()
    };
    let storage_result = PoolTransferResult {
        kv_hit_pages: 3,
        extra_pool_hit_pages: HashMap::from([(PoolName::Swa, 2)]),
    };
    swa_component(4).commit_hicache_transfer(
        &mut tc,
        b,
        CacheTransferPhase::Prefetch,
        vec![PoolTransfer {
            name: PoolName::Swa,
            host_indices: Some(Tensor::from_slice(&[30i64, 31])),
            ..Default::default()
        }],
        &mut cache_actions,
        Some(&mut insert_result),
        Some(&storage_result),
    );
    // c (on path) fills with the buffer tail; the out-of-path prefix releases.
    assert!(
        tc.arena
            .node(c)
            .host_value(SWA)
            .equal(&Tensor::from_slice(&[31i64]))
    );
    assert!(!tc.arena.node(b).has_host_value(SWA));
    assert!(!tc.arena.node(a).has_host_value(SWA));
    assert!(tc.host_lru_list(SWA).in_list(Some(c)));
    assert_eq!(cache_actions.len(), 1);
    let CacheAction::FreeComponentHostSlot { host_indices, .. } = &cache_actions[0] else {
        panic!("expected a host free");
    };
    assert!(host_indices[0].equal(&Tensor::from_slice(&[30i64])));
}

#[test]
fn prefetch_commit_fills_tombstoned_nodes_and_releases_covered_ones() {
    // Chain root -> a -> b (one token each); b holds SWA host already, a is
    // a tombstone: b's slice releases, a's fills.
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a, b] = chain::<2>(&mut tc);
    tc.arena
        .set_host_value(b, SWA, Tensor::from_slice(&[40i64]));
    let mut cache_actions = Vec::new();
    let mut insert_result = InsertResult {
        total_len: 2,
        inserted_host_node: Some(tc.arena.node(b).id),
        ..InsertResult::default()
    };
    let storage_result = PoolTransferResult {
        kv_hit_pages: 2,
        extra_pool_hit_pages: HashMap::from([(PoolName::Swa, 2)]),
    };
    let root = tc.arena.root();
    swa_component(4).commit_hicache_transfer(
        &mut tc,
        root,
        CacheTransferPhase::Prefetch,
        vec![PoolTransfer {
            name: PoolName::Swa,
            host_indices: Some(Tensor::from_slice(&[30i64, 31])),
            ..Default::default()
        }],
        &mut cache_actions,
        Some(&mut insert_result),
        Some(&storage_result),
    );
    // b (already hosted) released its slice [31]; a filled with [30].
    assert!(
        tc.arena
            .node(a)
            .host_value(SWA)
            .equal(&Tensor::from_slice(&[30i64]))
    );
    assert!(
        tc.arena
            .node(b)
            .host_value(SWA)
            .equal(&Tensor::from_slice(&[40i64]))
    );
    assert_eq!(cache_actions.len(), 1);
    let CacheAction::FreeComponentHostSlot { host_indices, .. } = &cache_actions[0] else {
        panic!("expected a host free");
    };
    assert!(host_indices[0].equal(&Tensor::from_slice(&[31i64])));
}

#[test]
fn prefetch_commit_splits_a_partially_covered_tombstone() {
    // One two-token tombstone node; the buffer covers only its tail token,
    // so the node splits and the tail attaches.
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(a, FULL, Tensor::from_slice(&[10i64, 11]));
    tc.update_evictable_leaf_sets_(a);
    let mut cache_actions = Vec::new();
    let mut insert_result = InsertResult {
        total_len: 2,
        inserted_host_node: Some(tc.arena.node(a).id),
        ..InsertResult::default()
    };
    let storage_result = PoolTransferResult {
        kv_hit_pages: 1,
        extra_pool_hit_pages: HashMap::from([(PoolName::Swa, 1)]),
    };
    swa_component(4).commit_hicache_transfer(
        &mut tc,
        root,
        CacheTransferPhase::Prefetch,
        vec![PoolTransfer {
            name: PoolName::Swa,
            host_indices: Some(Tensor::from_slice(&[31i64])),
            ..Default::default()
        }],
        &mut cache_actions,
        Some(&mut insert_result),
        Some(&storage_result),
    );
    // The node split at token 1; its tail (still node a) got the slice.
    let node = tc.arena.node(a);
    assert_eq!(node.key.atom_len(), 1);
    assert!(node.host_value(SWA).equal(&Tensor::from_slice(&[31i64])));
    assert!(cache_actions.is_empty());
}

#[test]
fn release_swa_host_queues_a_free_action_for_non_empty_indices() {
    let mut cache_actions = Vec::new();
    swa_component(4).release_swa_host_(Tensor::from_slice(&[7i64]), &mut cache_actions);
    assert_eq!(cache_actions.len(), 1);
    let CacheAction::FreeComponentHostSlot {
        component_type,
        host_indices,
    } = &cache_actions[0]
    else {
        panic!("expected a FreeComponentHostSlot action");
    };
    assert_eq!(*component_type, SWA);
    assert_eq!(host_indices.len(), 1);
    assert!(host_indices[0].equal(&Tensor::from_slice(&[7i64])));
    // Empty indices queue nothing.
    let empty: [i64; 0] = [];
    swa_component(4).release_swa_host_(Tensor::from_slice(&empty), &mut cache_actions);
    assert_eq!(cache_actions.len(), 1);
}

#[test]
fn attach_swa_host_value_inserts_tombstones_into_the_host_lru() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a, b] = chain::<2>(&mut tc);
    set_swa_device(&mut tc, a);
    let swa = swa_component(4);
    // A tombstone lands in the host LRU; a device-backed node does not.
    swa.attach_swa_host_value_(&mut tc, b, Tensor::from_slice(&[9i64]));
    assert!(
        tc.arena
            .host_value(b, SWA)
            .equal(&Tensor::from_slice(&[9i64]))
    );
    assert!(tc.host_lru_list(SWA).in_list(Some(b)));
    swa.attach_swa_host_value_(&mut tc, a, Tensor::from_slice(&[8i64]));
    assert!(tc.arena.has_host_value(a, SWA));
    assert!(!tc.host_lru_list(SWA).in_list(Some(a)));
}

#[test]
fn attach_swa_host_value_skips_reinsertion_when_already_listed() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    tc.host_lru_list_mut(SWA).insert_mru(a);
    swa_component(4).attach_swa_host_value_(&mut tc, a, Tensor::from_slice(&[9i64]));
    assert!(tc.host_lru_list(SWA).in_list(Some(a)));
}

#[test]
fn build_load_back_spec_includes_the_swa_transfers() {
    let mut tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(
        CacheInitParams {
            enable_hicache: true,
            has_swa_host_pool: true,
            ..swa_params_with_window(4)
        },
        vec![FULL, SWA],
    );
    let [n] = chain::<1>(&mut tc);
    set_full_host(&mut tc, n);
    tc.arena
        .set_host_value(n, SWA, Tensor::from_slice(&[30i64]));
    let (kv_xfer, mut comp_xfers) =
        tc.build_load_back_spec(tc.arena.node(n).id, /* req = */ None);
    assert_eq!(kv_xfer.nodes_to_load, Some(vec![tc.arena.node(n).id]));
    let swa_xfers = comp_xfers.get_mut(&SWA).unwrap();
    assert_eq!(swa_xfers.len(), 1);
    assert!(
        swa_xfers[0]
            .host_indices
            .as_ref()
            .unwrap()
            .equal(&Tensor::from_slice(&[30i64]))
    );
    assert_eq!(swa_xfers[0].nodes_to_load, Some(vec![tc.arena.node(n).id]));
    // The orchestrator fills each transfer's device side from the pool load.
    swa_xfers[0].device_indices = Some(Tensor::from_slice(&[60i64]));
    let actions = tc.commit_load_back(
        tc.arena.node(n).id,
        Tensor::from_slice(&[50i64]),
        kv_xfer,
        comp_xfers,
    );
    assert!(
        tc.arena
            .device_value(n, FULL)
            .equal(&Tensor::from_slice(&[50i64]))
    );
    assert!(
        tc.arena
            .device_value(n, SWA)
            .equal(&Tensor::from_slice(&[60i64]))
    );
    assert_eq!(actions.len(), 1);
    assert!(matches!(
        actions[0],
        CacheAction::RebuildFullToSwaMapping { .. }
    ));
}

#[test]
fn auxiliary_load_does_not_reuse_a_full_pending_pin() {
    let mut tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(
        CacheInitParams {
            is_write_back: true,
            enable_hicache: true,
            has_swa_host_pool: true,
            ..swa_params_with_window(4)
        },
        vec![FULL, SWA],
    );
    let [shared, anchor_b] = chain::<2>(&mut tc);
    set_full_host(&mut tc, shared);
    set_full_host(&mut tc, anchor_b);
    set_swa_host(&mut tc, shared);

    let shared_id = tc.arena.node(shared).id;
    let anchor_b_id = tc.arena.node(anchor_b).id;
    tc.commit_load_back(
        shared_id,
        Tensor::from_slice(&[10i64]),
        PoolTransfer {
            name: PoolName::Kv,
            host_indices: Some(Tensor::from_slice(&[1i64])),
            nodes_to_load: Some(vec![shared_id]),
            ..Default::default()
        },
        HashMap::new(),
    );
    assert_eq!(tc.arena.node(shared).load_back_pending_id, Some(shared_id));

    let swa_xfer = PoolTransfer {
        name: PoolName::Swa,
        host_indices: Some(Tensor::from_slice(&[2i64])),
        device_indices: Some(Tensor::from_slice(&[20i64])),
        nodes_to_load: Some(vec![shared_id]),
        ..Default::default()
    };
    tc.commit_load_back(
        anchor_b_id,
        Tensor::from_slice(&[30i64]),
        PoolTransfer {
            name: PoolName::Kv,
            host_indices: Some(Tensor::from_slice(&[3i64])),
            nodes_to_load: Some(vec![anchor_b_id]),
            ..Default::default()
        },
        HashMap::from([(SWA, vec![swa_xfer])]),
    );

    assert_eq!(tc.arena.node(shared).load_back_pending_id, Some(shared_id));
    assert_eq!(
        tc.arena.node(anchor_b).load_back_pending_id,
        Some(anchor_b_id)
    );
}

#[test]
fn swa_device_eviction_skips_a_load_back_pinned_node() {
    let mut tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(
        CacheInitParams {
            is_write_back: true,
            enable_hicache: true,
            has_swa_host_pool: true,
            ..swa_params_with_window(4)
        },
        vec![FULL, SWA],
    );
    let [n] = chain::<1>(&mut tc);
    set_full_host(&mut tc, n);
    set_swa_host(&mut tc, n);
    let (kv_xfer, mut comp_xfers) =
        tc.build_load_back_spec(tc.arena.node(n).id, /* req = */ None);
    comp_xfers.get_mut(&SWA).unwrap()[0].device_indices = Some(Tensor::from_slice(&[60i64]));
    tc.commit_load_back(
        tc.arena.node(n).id,
        Tensor::from_slice(&[50i64]),
        kv_xfer,
        comp_xfers,
    );
    // The pin alone keeps the in-flight SWA slice out of every eviction branch.
    tc.evict_device_start(SWA, 4);
    let (next, _) = tc.evict_device_next_node(SWA, &HashMap::new());
    assert_eq!(next, None);
    tc.evict_device_end(SWA);
    assert!(tc.arena.has_device_value(n, SWA));
    tc.finish_load_back(tc.arena.node(n).id);
    tc.evict_device_start(SWA, 4);
    let (next, _) = tc.evict_device_next_node(SWA, &HashMap::new());
    assert_eq!(next, Some(tc.arena.node(n).id));
    tc.evict_device_end(SWA);
}

#[test]
fn swa_host_eviction_skips_a_load_back_pinned_node() {
    let mut tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(
        CacheInitParams {
            is_write_back: true,
            enable_hicache: true,
            has_swa_host_pool: true,
            ..swa_params_with_window(1)
        },
        vec![FULL, SWA],
    );
    let [a, b] = chain::<2>(&mut tc);
    set_full_host(&mut tc, a);
    set_swa_host(&mut tc, a);
    set_full_host(&mut tc, b);
    set_swa_host(&mut tc, b);
    tc.host_lru_list_mut(SWA).insert_mru(a);
    let (kv_xfer, mut comp_xfers) =
        tc.build_load_back_spec(tc.arena.node(b).id, /* req = */ None);
    assert_eq!(
        comp_xfers.get(&SWA).unwrap()[0].nodes_to_load,
        Some(vec![tc.arena.node(b).id])
    );
    comp_xfers.get_mut(&SWA).unwrap()[0].device_indices = Some(Tensor::from_slice(&[60i64]));
    tc.commit_load_back(
        tc.arena.node(b).id,
        Tensor::from_slice(&[50i64, 51]),
        kv_xfer,
        comp_xfers,
    );

    let result = tc.drive_host_eviction(SWA, /* num_tokens = */ 1);
    assert_eq!(result.tracker[&SWA], 0);
    assert!(result.host_frees.is_empty());
    assert!(tc.arena.has_host_value(a, SWA));

    tc.finish_load_back(tc.arena.node(b).id);
    let result = tc.drive_host_eviction(SWA, /* num_tokens = */ 1);
    assert_eq!(result.tracker[&SWA], 1);
    assert_eq!(result.host_frees[&SWA].len(), 1);
    // Write-back reclaims the loaded node's coexisting host duplicate first.
    assert!(tc.arena.has_host_value(a, SWA));
    assert!(!tc.arena.has_host_value(b, SWA));
    tc.sanity_check(&[], &[]);
}

#[test]
fn build_load_back_spec_degrades_to_empty_on_a_foreign_pin() {
    let mut tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(
        CacheInitParams {
            is_write_back: true,
            enable_hicache: true,
            has_swa_host_pool: true,
            ..swa_params_with_window(4)
        },
        vec![FULL, SWA],
    );
    let [a, b] = chain::<2>(&mut tc);
    set_full_host(&mut tc, a);
    set_swa_host(&mut tc, a);
    set_full_host(&mut tc, b);
    set_swa_host(&mut tc, b);
    // Anchor `a` models a Full-only load whose SWA slice remains host-only.
    let (kv_xfer, _comp_xfers) =
        tc.build_load_back_spec(tc.arena.node(a).id, /* req = */ None);
    tc.commit_load_back(
        tc.arena.node(a).id,
        Tensor::from_slice(&[50i64]),
        kv_xfer,
        HashMap::new(),
    );
    // Anchor `b` must reject its SWA window because `a` has a foreign pin.
    let (kv_xfer, comp_xfers) = tc.build_load_back_spec(tc.arena.node(b).id, /* req = */ None);
    assert_eq!(kv_xfer.host_indices.unwrap().numel(), 0);
    assert_eq!(kv_xfer.nodes_to_load, Some(vec![]));
    assert!(comp_xfers.is_empty());
    tc.finish_load_back(tc.arena.node(a).id);
    let (kv_xfer, comp_xfers) = tc.build_load_back_spec(tc.arena.node(b).id, /* req = */ None);
    assert_eq!(kv_xfer.nodes_to_load, Some(vec![tc.arena.node(b).id]));
    assert_eq!(
        comp_xfers.get(&SWA).unwrap()[0].nodes_to_load,
        Some(vec![tc.arena.node(a).id, tc.arena.node(b).id])
    );
}

#[test]
fn host_drive_reclaims_swa_coexisting_host_values_when_the_host_lru_is_empty() {
    let mut tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(
        CacheInitParams {
            is_write_back: true,
            enable_hicache: true,
            has_swa_host_pool: true,
            ..swa_params_with_window(4)
        },
        vec![FULL, SWA],
    );
    tc.insert(&insert_params_swa(&vec![1, 2], &[10, 11], 0, 0));
    tc.insert(&insert_params_swa(&vec![1, 2, 3], &[10, 11, 12], 0, 0));
    let root = tc.arena.root();
    let parent_idx = child_of(&tc, root, &[1]);
    let leaf_idx = child_of(&tc, parent_idx, &[3]);
    let (parent, leaf) = (tc.arena.node(parent_idx).id, tc.arena.node(leaf_idx).id);
    for (handle, slots) in [(parent, vec![30i64, 31]), (leaf, vec![32i64])] {
        tc.set_component_device_value(handle, SWA, Tensor::from_slice(&slots));
    }
    for (handle, host) in [(parent, vec![20i64, 21]), (leaf, vec![22i64])] {
        let swa_xfer = PoolTransfer {
            name: PoolName::Swa,
            host_indices: Some(Tensor::from_slice(&host)),
            ..Default::default()
        };
        tc.commit_backup(
            handle,
            Tensor::from_slice(&host),
            HashMap::from([(SWA, vec![swa_xfer])]),
        );
    }
    assert_eq!(tc.host_lru_list(SWA).len(), 0);

    let mut tracker = swa_tracker();
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    accumulate_step(
        tc.drive_host_eviction(SWA, /* num_tokens = */ 2),
        &mut tracker,
        &mut df,
        &mut hf,
    );
    assert_eq!(tracker[&SWA], 2);
    assert!(!tc.arena.node(parent_idx).has_host_value(SWA));
    assert!(tc.arena.node(parent_idx).has_device_value(SWA));
    assert!(tc.arena.node(parent_idx).has_host_value(FULL));
    assert!(tc.arena.node(leaf_idx).has_host_value(SWA));
    tc.sanity_check(&[], &[]);
}

fn match_params(key: &Vec<i64>) -> MatchPrefixParams<'_, Vec<i64>> {
    MatchPrefixParams {
        key,
        namespace: Default::default(),
    }
}

#[test]
fn swa_evict_on_a_full_locked_leaf_tombstones_only_the_swa_slot() {
    let mut tc = swa_core(/* window = */ 4, /* page_size = */ 1);
    let [a] = chain(&mut tc);
    tc.arena
        .set_device_value(a, FULL, Tensor::from_slice(&[9i64]));
    store_swa_device(&mut tc, a);
    // The held Full lock keeps the leaf out of the D-leaf set.
    tc.arena
        .node_mut(a)
        .set_lock_ref_(ValueSlotIdx::device(FULL), 1);
    tc.update_evictable_leaf_sets_(a);
    assert!(!tc.evictable_device_leaves.contains(a));
    let mut tracker = swa_tracker();
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    tc.evict_device_start(SWA, /* request_cnt = */ 10);
    let (next, step) = tc.evict_device_next_node(SWA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    assert_eq!(next, None);
    tc.evict_device_end(SWA);
    // The SWA slot tombstoned inline; its frees report the node's Full indices.
    assert!(!tc.arena.has_device_value(a, SWA));
    assert!(device_frees[&SWA][0].equal(&Tensor::from_slice(&[9i64])));
    assert_eq!(tracker[&SWA], 1);
    assert_eq!(tc.swa_evictable_size(), 0);
    assert!(host_frees.is_empty());
    // The higher-tier locked Full is spared and pins the node in the tree.
    assert!(
        tc.arena
            .device_value(a, FULL)
            .equal(&Tensor::from_slice(&[9i64]))
    );
    assert_eq!(tc.arena.device_lock_ref(a, FULL), 1);
    assert_eq!(tc.arena.len(), 2);
    assert!(!tc.device_lru_list(SWA).in_list(Some(a)));
    assert!(!tc.evictable_device_leaves.contains(a));
}

#[test]
fn write_through_offloads_a_boundary_split_leaf() {
    let mut tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(
        CacheInitParams {
            write_through_threshold: 1,
            enable_hicache: true,
            ..swa_params_with_window(8)
        },
        vec![FULL, SWA],
    );
    let result = tc.insert(&insert_params_swa(
        &vec![1, 2, 3, 4],
        &[10, 11, 12, 13],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 2,
    ));
    let root = tc.arena.root();
    let parent = child_of(&tc, root, &[1]);
    let leaf = child_of(&tc, parent, &[3]);
    assert_eq!(tc.arena.node(parent).key, vec![1, 2]);
    assert_eq!(tc.arena.node(leaf).key, vec![3, 4]);
    // The threshold crossing backs up both split fragments ancestors-first.
    let backups: Vec<_> = result
        .cache_actions
        .iter()
        .filter_map(|action| match action {
            CacheAction::BackupKV(backup) => Some(backup.node_ids.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        backups,
        vec![vec![tc.arena.node(parent).id, tc.arena.node(leaf).id]]
    );
    tc.commit_backup(
        tc.arena.node(parent).id,
        Tensor::from_slice(&[100i64, 101]),
        HashMap::new(),
    );
    tc.commit_backup(
        tc.arena.node(leaf).id,
        Tensor::from_slice(&[102i64, 103]),
        HashMap::new(),
    );
    let mut tracker = swa_tracker();
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.evict_device_start(FULL, /* request_cnt = */ 100);
    loop {
        let (next, step) = tc.evict_device_next_node(FULL, &tracker);
        accumulate_step(step, &mut tracker, &mut df, &mut hf);
        let Some(next) = next else { break };
        let (backup, step) = tc.evict_device_leaf(next, /* is_write_back = */ false);
        assert!(backup.is_none());
        accumulate_step(step, &mut tracker, &mut df, &mut hf);
    }
    tc.evict_device_end(FULL);
    // The split leaf demoted like any backuped leaf and awaits host eviction.
    let leaf_node = tc.arena.node(leaf);
    assert!(leaf_node.evicted() && leaf_node.backuped());
    assert!(tc.evictable_host_leaves.contains(leaf));
    tc.sanity_check(&[], &[]);
}

#[test]
fn deep_swa_tree_survives_backup_evict_and_load_back_rounds() {
    let mut tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(
        CacheInitParams {
            enable_hicache: true,
            ..swa_params_with_window(4)
        },
        vec![FULL, SWA],
    );
    // Varied topology: capped long leaf, decode-evicted chain, depth
    // extension, and two branches off the shared prefix.
    let inserts: [(Vec<i64>, Vec<i64>, usize); 5] = [
        (vec![1, 2, 3, 4, 5, 6], vec![10, 11, 12, 13, 14, 15], 0),
        (
            vec![21, 22, 23, 24, 25, 26],
            vec![30, 31, 32, 33, 34, 35],
            1,
        ),
        (
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![10, 11, 12, 13, 14, 15, 16, 17],
            0,
        ),
        (vec![1, 2, 41, 42, 43], vec![10, 11, 50, 51, 52], 0),
        (vec![1, 2, 61, 62, 63], vec![10, 11, 70, 71, 72], 0),
    ];
    for (key, kv, swa_evicted_seqlen) in inserts {
        let result = tc.insert(&insert_params_swa(&key, &kv, 0, swa_evicted_seqlen));
        // Apply the emitted rebuilds the way the cache would.
        for action in &result.cache_actions {
            if let CacheAction::SwaRebuild {
                node_id,
                source_value,
            } = action
            {
                tc.set_component_device_value(*node_id, SWA, source_value.copy());
            }
        }
        tc.sanity_check(&[], &[]);
    }
    let nodes = tc.collect_all_nodes_();
    assert!(nodes.len() >= 6);
    // Back up every non-root node so device eviction demotes instead of deleting.
    for node in nodes {
        if tc.arena.node(node).is_root() {
            continue;
        }
        let len = tc.arena.node(node).key.atom_len();
        tc.commit_backup(
            tc.arena.node(node).id,
            Tensor::from_slice(&vec![0i64; len]),
            HashMap::new(),
        );
    }
    tc.sanity_check(&[], &[]);
    // Stepwise eviction rounds: half the Full budget, then the whole SWA budget.
    for _ in 0..4 {
        let full_budget = (tc.full_evictable_size() / 2).max(1);
        let mut tracker = swa_tracker();
        let (mut df, mut hf) = (HashMap::new(), HashMap::new());
        tc.evict_device_start(FULL, full_budget);
        loop {
            let (leaf, step) = tc.evict_device_next_node(FULL, &tracker);
            accumulate_step(step, &mut tracker, &mut df, &mut hf);
            let Some(leaf) = leaf else { break };
            let (_, step) = tc.evict_device_leaf(leaf, /* is_write_back = */ false);
            accumulate_step(step, &mut tracker, &mut df, &mut hf);
        }
        tc.evict_device_end(FULL);
        let swa_budget = tc.swa_evictable_size();
        if swa_budget > 0 {
            let mut tracker = swa_tracker();
            let (mut df, mut hf) = (HashMap::new(), HashMap::new());
            tc.evict_device_start(SWA, swa_budget);
            loop {
                let (leaf, step) = tc.evict_device_next_node(SWA, &tracker);
                accumulate_step(step, &mut tracker, &mut df, &mut hf);
                let Some(leaf) = leaf else { break };
                let (_, step) = tc.evict_device_leaf(leaf, /* is_write_back = */ false);
                accumulate_step(step, &mut tracker, &mut df, &mut hf);
            }
            tc.evict_device_end(SWA);
        }
        tc.sanity_check(&[], &[]);
    }
    // Load evicted prefixes back from host, mirroring the orchestrator's
    // commit-then-lock sequence.
    for key in [vec![1i64, 2, 3, 4, 5, 6], vec![1i64, 2]] {
        let anchor = tc.match_prefix(&match_params(&key)).best_match_node_id;
        if !tc.is_root(anchor) && tc.is_full_device_evicted(anchor) {
            let (kv_xfer, comp_xfers) = tc.build_load_back_spec(anchor, /* req = */ None);
            let loaded = kv_xfer.host_indices.as_ref().unwrap().numel();
            let actions = tc.commit_load_back(
                anchor,
                Tensor::from_slice(&vec![0i64; loaded]),
                kv_xfer,
                comp_xfers,
            );
            assert!(actions.is_empty());
            let lock = tc.inc_lock_ref(anchor);
            let params = DecLockRefParams {
                swa_uuid_for_lock: lock.swa_uuid_for_lock,
                swa_uuid_for_host_lock: lock.swa_uuid_for_host_lock,
                skip_lock_node_ids: lock.skip_lock_node_ids,
            };
            tc.dec_lock_ref(anchor, Some(&params), /* skip_swa = */ false);
            tc.finish_load_back(anchor);
        }
        tc.sanity_check(&[], &[]);
    }
    let matched = tc.match_prefix(&match_params(&vec![1, 2, 3, 4, 5, 6]));
    assert_eq!(matched.device_indices.numel(), 6);
    tc.sanity_check(&[], &[]);
}

#[test]
fn recovered_swa_span_evicts_before_the_window_leaf() {
    let mut tc = swa_core(/* window = */ 8, /* page_size = */ 1);
    let key: Vec<i64> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let result = tc.insert(&insert_params_swa(
        &key,
        &[
            100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
        ],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 8,
    ));
    let root = tc.arena.root();
    let prefix = child_of(&tc, root, &[1]);
    let leaf = child_of(&tc, prefix, &[9]);
    assert_eq!(tc.arena.node(prefix).key, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let [
        CacheAction::SwaRebuild {
            node_id,
            source_value,
        },
    ] = result.cache_actions.as_slice()
    else {
        panic!(
            "expected one SwaRebuild action, got {:?}",
            action_kinds(&result.cache_actions)
        );
    };
    assert_eq!(*node_id, tc.arena.node(leaf).id);
    tc.set_component_device_value(*node_id, SWA, source_value.copy());
    assert!(!tc.arena.has_device_value(prefix, SWA));

    // The fully-in-window re-insert recovers the prefix at its walk barrier.
    let step = tc.begin_insert(&insert_params_swa(
        &key,
        &[
            200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
        ],
        /* prev_prefix_len = */ 0,
        /* swa_evicted_seqlen = */ 0,
    ));
    assert!(step.result.is_none());
    let [
        CacheAction::FreeDeviceKVFullOnly(old_full),
        CacheAction::SwaRebuild {
            node_id,
            source_value,
        },
    ] = step.actions.as_slice()
    else {
        panic!(
            "expected FreeDeviceKVFullOnly then SwaRebuild, got {:?}",
            action_kinds(&step.actions)
        );
    };
    assert!(old_full[0].equal(&Tensor::from_slice(&[
        100i64, 101, 102, 103, 104, 105, 106, 107
    ])));
    assert_eq!(*node_id, tc.arena.node(prefix).id);
    tc.set_component_device_value(*node_id, SWA, source_value.copy());
    let done = tc.resume_insert();
    assert_eq!(
        done.result.expect("the resumed walk completes").prefix_len,
        16
    );
    // The insert-end window refresh parks the recovered prefix below the leaf.
    assert_eq!(swa_lru_order(&tc), vec![leaf, prefix]);

    let mut tracker = swa_tracker();
    let (mut device_frees, mut host_frees) = (HashMap::new(), HashMap::new());
    tc.evict_device_start(SWA, /* request_cnt = */ 8);
    let (next, step) = tc.evict_device_next_node(SWA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    assert_eq!(next, None);
    tc.evict_device_end(SWA);
    // The recovered span is retaken first; the window leaf survives whole.
    assert!(!tc.arena.has_device_value(prefix, SWA));
    assert!(device_frees[&SWA][0].equal(&Tensor::from_slice(&[
        200i64, 201, 202, 203, 204, 205, 206, 207
    ])));
    assert_eq!(tracker[&SWA], 8);
    assert!(tc.arena.has_device_value(leaf, SWA));
    assert!(tc.arena.has_device_value(leaf, FULL));
    tc.sanity_check(&[], &[]);
}
