use super::*;
use crate::components::FULL;
use crate::test_utils::accumulate_step;
use crate::unified_tree_core::CacheInitParams;

fn core() -> UnifiedTreeCore<Vec<i64>> {
    UnifiedTreeCore::new(CacheInitParams::default(), vec![FULL])
}

// Raw seeding for states set_value rejects: mid-split (key trimmed before the value
// splits) and present-but-empty semantics pins.
fn set_value_no_check<K: ChildKeyType>(
    tc: &mut UnifiedTreeCore<K>,
    node: NodeIdx_,
    slot: ValueSlotIdx,
    value: Tensor,
) {
    tc.arena.node_mut(node).state_mut_(slot).value = Some(value);
}

#[test]
fn eviction_priority_is_lower_for_leaf_than_internal() {
    let full = FullComponent;
    assert_eq!(
        <FullComponent as TreeComponent<Vec<i64>>>::eviction_priority(&full, true),
        0
    );
    assert_eq!(
        <FullComponent as TreeComponent<Vec<i64>>>::eviction_priority(&full, false),
        2
    );
}

// Three value-bearing root children in the D-leaf set; ticks anti-correlated
// with allocation order so priority order differs from NodeIdx_ order.
fn evict_walk_setup(tc: &mut UnifiedTreeCore<Vec<i64>>) -> (NodeIdx_, NodeIdx_, NodeIdx_) {
    let root = tc.arena.root();
    let ticks = [30i64, 10, 20];
    let mut nodes = [NodeIdx_(0); 3];
    for (i, node) in nodes.iter_mut().enumerate() {
        let id = tc
            .arena
            .alloc_child(
                root,
                /* key = */ vec![i as i64],
                /* priority = */ 0,
                /* extra_key = */ None,
            )
            .unwrap();
        tc.arena
            .set_device_value(id, FULL, Tensor::from_slice(&[i as i64]));
        tc.arena.node_mut(id).last_access_counter = ticks[i];
        tc.evictable_device_leaves.add(id);
        *node = id;
    }
    (nodes[0], nodes[1], nodes[2])
}

fn match_params(key: &Vec<i64>) -> MatchPrefixParams<'_, Vec<i64>> {
    MatchPrefixParams {
        key,
        namespace: Default::default(),
    }
}

fn insert(tc: &mut UnifiedTreeCore<Vec<i64>>, key: &Vec<i64>, value: &[i64]) {
    tc.insert(&crate::unified_tree_core::InsertParams {
        key,
        namespace: Default::default(),
        value: Tensor::from_slice(value),
        mamba_value: None,
        prev_prefix_len: 0,
        swa_evicted_seqlen: 0,
        chunked: false,
        priority: 0,
        track_adopted_ranges: false,
    });
}

fn tracker() -> HashMap<ComponentType, usize> {
    HashMap::from([(FULL, 0)])
}

fn frees() -> HashMap<ComponentType, Vec<Tensor>> {
    HashMap::new()
}

#[test]
fn evict_walk_pops_leaves_lowest_priority_first() {
    let mut tc = core();
    let (a, b, c) = evict_walk_setup(&mut tc);
    let (mut tr, mut df, mut hf) = (tracker(), frees(), frees());
    tc.evict_device_start(FULL, /* request_cnt = */ 100);
    let (node, step) = tc.evict_device_next_node(FULL, &tr);
    accumulate_step(step, &mut tr, &mut df, &mut hf);
    assert_eq!(node, Some(tc.arena.node(b).id));
    // The driver evicts each returned leaf before asking for the next.
    tc.evictable_device_leaves.discard(b);
    let (node, step) = tc.evict_device_next_node(FULL, &tr);
    accumulate_step(step, &mut tr, &mut df, &mut hf);
    assert_eq!(node, Some(tc.arena.node(c).id));
    tc.evictable_device_leaves.discard(c);
    let (node, step) = tc.evict_device_next_node(FULL, &tr);
    accumulate_step(step, &mut tr, &mut df, &mut hf);
    assert_eq!(node, Some(tc.arena.node(a).id));
    tc.evictable_device_leaves.discard(a);
    let (node, step) = tc.evict_device_next_node(FULL, &tr);
    accumulate_step(step, &mut tr, &mut df, &mut hf);
    assert_eq!(node, None);
    tc.evict_device_end(FULL);
}

#[test]
fn evict_walk_stops_at_the_token_budget() {
    let mut tc = core();
    let (_a, b, _c) = evict_walk_setup(&mut tc);
    let (mut tr, mut df, mut hf) = (tracker(), frees(), frees());
    tc.evict_device_start(FULL, /* request_cnt = */ 5);
    let (node, step) = tc.evict_device_next_node(FULL, &tr);
    accumulate_step(step, &mut tr, &mut df, &mut hf);
    assert_eq!(node, Some(tc.arena.node(b).id));
    // The driver's tracker reaching the budget ends the walk.
    *tr.get_mut(&FULL).unwrap() = 5;
    let (node, step) = tc.evict_device_next_node(FULL, &tr);
    accumulate_step(step, &mut tr, &mut df, &mut hf);
    assert_eq!(node, None);
    // The budget-None return still resets the cursor.
    assert_eq!(tc.component_state(FULL).evict_device_cursor, None);
    tc.evict_device_end(FULL);
}

#[test]
fn evict_walk_stops_when_the_tracker_overshoots_the_budget() {
    let mut tc = core();
    let (_a, b, _c) = evict_walk_setup(&mut tc);
    let (mut tr, mut df, mut hf) = (tracker(), frees(), frees());
    tc.evict_device_start(FULL, /* request_cnt = */ 5);
    let (node, step) = tc.evict_device_next_node(FULL, &tr);
    accumulate_step(step, &mut tr, &mut df, &mut hf);
    assert_eq!(node, Some(tc.arena.node(b).id));
    // Multi-token evictions jump past the budget; the gate must still fire.
    *tr.get_mut(&FULL).unwrap() = 7;
    let (node, step) = tc.evict_device_next_node(FULL, &tr);
    accumulate_step(step, &mut tr, &mut df, &mut hf);
    assert_eq!(node, None);
    assert_eq!(tc.component_state(FULL).evict_device_cursor, None);
    tc.evict_device_end(FULL);
}

#[test]
fn evict_walk_reports_done_when_the_baseline_already_meets_the_budget() {
    let mut tc = core();
    let (_a, _b, _c) = evict_walk_setup(&mut tc);
    tc.evict_device_start(FULL, /* request_cnt = */ 5);
    // Prior steps' evictions reach the gate through the baseline.
    let baseline = HashMap::from([(FULL, 5)]);
    let (node, _step) = tc.evict_device_next_node(FULL, &baseline);
    assert_eq!(node, None);
    tc.evict_device_end(FULL);
}

#[test]
fn evict_walk_step_tracker_stays_empty_when_nothing_was_evicted() {
    let mut tc = core();
    let (_a, b, _c) = evict_walk_setup(&mut tc);
    tc.evict_device_start(FULL, /* request_cnt = */ 100);
    let (node, step) = tc.evict_device_next_node(FULL, &tracker());
    assert_eq!(node, Some(tc.arena.node(b).id));
    // The FULL walk frees nothing itself: no zero-delta entries leak out.
    assert!(step.tracker.is_empty());
    tc.evict_device_end(FULL);
}

#[test]
fn evict_walk_skips_nodes_that_left_the_leaf_set() {
    let mut tc = core();
    let (_a, b, c) = evict_walk_setup(&mut tc);
    let (mut tr, mut df, mut hf) = (tracker(), frees(), frees());
    tc.evict_device_start(FULL, /* request_cnt = */ 100);
    // The lowest-priority b stops being a D-leaf after the heap was built
    // (e.g. locked): the walk skips it for the next candidate.
    tc.evictable_device_leaves.discard(b);
    let (node, step) = tc.evict_device_next_node(FULL, &tr);
    accumulate_step(step, &mut tr, &mut df, &mut hf);
    assert_eq!(node, Some(tc.arena.node(c).id));
    tc.evict_device_end(FULL);
}

#[test]
fn evict_walk_readmits_the_freed_leafs_parent() {
    let mut tc = core();
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
    // Two live sibling leaves whose ticks straddle the parent's tick, so
    // the readmitted parent must compete on its own priority.
    let s1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![3],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let s2 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![4],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    for (id, tick) in [(c, 5i64), (s1, 10), (p, 20), (s2, 30)] {
        tc.arena
            .set_device_value(id, FULL, Tensor::from_slice(&[tick]));
        tc.arena.node_mut(id).last_access_counter = tick;
    }
    tc.evictable_device_leaves.add(c);
    tc.evictable_device_leaves.add(s1);
    tc.evictable_device_leaves.add(s2);
    let (mut tr, mut df, mut hf) = (tracker(), frees(), frees());
    tc.evict_device_start(FULL, /* request_cnt = */ 100);
    let (node, step) = tc.evict_device_next_node(FULL, &tr);
    accumulate_step(step, &mut tr, &mut df, &mut hf);
    assert_eq!(node, Some(tc.arena.node(c).id));
    // The driver deletes the write-through leaf outright and its parent
    // becomes the new D-leaf; the walk must still find it via the parent
    // captured before the free, ordered between the surviving siblings.
    tc.evictable_device_leaves.discard(c);
    let _ = tc.arena.take_device_value(c, FULL);
    tc.arena.free_leaf(c).unwrap();
    tc.evictable_device_leaves.add(p);
    let (node, step) = tc.evict_device_next_node(FULL, &tr);
    accumulate_step(step, &mut tr, &mut df, &mut hf);
    assert_eq!(node, Some(tc.arena.node(s1).id));
    tc.evictable_device_leaves.discard(s1);
    let (node, step) = tc.evict_device_next_node(FULL, &tr);
    accumulate_step(step, &mut tr, &mut df, &mut hf);
    assert_eq!(node, Some(tc.arena.node(p).id));
    tc.evictable_device_leaves.discard(p);
    let (node, step) = tc.evict_device_next_node(FULL, &tr);
    accumulate_step(step, &mut tr, &mut df, &mut hf);
    assert_eq!(node, Some(tc.arena.node(s2).id));
    tc.evict_device_end(FULL);
}

#[test]
fn evict_end_clears_the_walk_and_allows_a_restart() {
    let mut tc = core();
    let (_a, b, _c) = evict_walk_setup(&mut tc);
    let (mut tr, mut df, mut hf) = (tracker(), frees(), frees());
    tc.evict_device_start(FULL, /* request_cnt = */ 100);
    tc.evict_device_end(FULL);
    // A fresh walk rebuilds the heap from the leaf set.
    tc.evict_device_start(FULL, /* request_cnt = */ 100);
    let (node, step) = tc.evict_device_next_node(FULL, &tr);
    accumulate_step(step, &mut tr, &mut df, &mut hf);
    assert_eq!(node, Some(tc.arena.node(b).id));
    tc.evict_device_end(FULL);
}

#[test]
#[should_panic(expected = "Full device eviction already in progress")]
fn evict_start_panics_when_already_ongoing() {
    let mut tc = core();
    tc.evict_device_start(FULL, /* request_cnt = */ 1);
    tc.evict_device_start(FULL, /* request_cnt = */ 1);
}

#[test]
#[should_panic(expected = "Full device eviction not started")]
fn evict_next_panics_before_start() {
    let mut tc = core();
    let tr = tracker();
    tc.evict_device_next_node(FULL, &tr);
}

// Three host-backed root children in the H-leaf set; ticks anti-correlated
// with allocation order so priority order differs from NodeIdx_ order.
fn host_walk_setup(tc: &mut UnifiedTreeCore<Vec<i64>>) -> (NodeIdx_, NodeIdx_, NodeIdx_) {
    let root = tc.arena.root();
    let ticks = [30i64, 10, 20];
    let mut nodes = [NodeIdx_(0); 3];
    for (i, node) in nodes.iter_mut().enumerate() {
        let id = tc
            .arena
            .alloc_child(
                root,
                /* key = */ vec![i as i64],
                /* priority = */ 0,
                /* extra_key = */ None,
            )
            .unwrap();
        tc.arena
            .set_host_value(id, FULL, Tensor::from_slice(&[i as i64]));
        tc.arena.node_mut(id).last_access_counter = ticks[i];
        tc.evictable_host_leaves.add(id);
        *node = id;
    }
    (nodes[0], nodes[1], nodes[2])
}

#[test]
fn host_drive_evicts_leaves_lowest_priority_first_until_the_budget() {
    let mut tc = core();
    let (a, b, c) = host_walk_setup(&mut tc);
    let (mut tr, mut df, mut hf) = (tracker(), frees(), frees());
    accumulate_step(
        tc.drive_host_eviction(FULL, /* num_tokens = */ 2),
        &mut tr,
        &mut df,
        &mut hf,
    );
    // b (tick 10) and c (tick 20) go; a (tick 30) survives.
    assert_eq!(tr[&FULL], 2);
    assert!(!tc.evictable_host_leaves.contains(b));
    assert!(!tc.evictable_host_leaves.contains(c));
    assert!(tc.evictable_host_leaves.contains(a));
    assert_eq!(hf[&FULL].len(), 2);
    assert!(df.is_empty());
    assert_eq!(tc.arena.len(), 2);
    tc.sanity_check(&[], &[]);
}

#[test]
fn host_drive_stops_when_a_leaf_overshoots_the_budget() {
    let mut tc = core();
    let root = tc.arena.root();
    let big = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2, 3],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let small = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![9],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(big, FULL, Tensor::from_slice(&[10i64, 11, 12]));
    tc.arena
        .set_host_value(small, FULL, Tensor::from_slice(&[20i64]));
    tc.arena.node_mut(big).last_access_counter = 1;
    tc.arena.node_mut(small).last_access_counter = 2;
    tc.evictable_host_leaves.add(big);
    tc.evictable_host_leaves.add(small);
    let (mut tr, mut df, mut hf) = (tracker(), frees(), frees());
    accumulate_step(
        tc.drive_host_eviction(FULL, /* num_tokens = */ 2),
        &mut tr,
        &mut df,
        &mut hf,
    );
    // The 3-token leaf jumps past the 2-token budget; the walk still stops.
    assert_eq!(tr[&FULL], 3);
    assert!(tc.evictable_host_leaves.contains(small));
    assert_eq!(hf[&FULL].len(), 1);
    tc.sanity_check(&[], &[]);
}

#[test]
fn host_drive_reclaims_coexisting_host_values_while_sparing_the_device_leaf() {
    let mut tc = write_back_core();
    insert(&mut tc, &vec![1, 2], &[10, 11]);
    insert(&mut tc, &vec![1, 2, 3], &[10, 11, 12]);
    let leaf_handle = tc
        .match_prefix(&match_params(&vec![1, 2, 3]))
        .best_match_node_id;
    let leaf = tc.arena.resolve(leaf_handle);
    let parent = tc.arena.node(leaf).parent();
    tc.commit_backup(
        tc.arena.node(parent).id,
        Tensor::from_slice(&[20i64, 21]),
        HashMap::new(),
    );
    tc.commit_backup(leaf_handle, Tensor::from_slice(&[22i64]), HashMap::new());
    assert!(tc.evictable_host_leaves.is_empty());

    let (mut tr, mut df, mut hf) = (tracker(), frees(), frees());
    accumulate_step(
        tc.drive_host_eviction(FULL, /* num_tokens = */ 2),
        &mut tr,
        &mut df,
        &mut hf,
    );
    assert_eq!(tr[&FULL], 2);
    assert!(!tc.arena.node(parent).has_host_value(FULL));
    assert!(tc.arena.node(parent).has_device_value(FULL));
    assert!(tc.arena.node(leaf).has_host_value(FULL));
    assert_ne!(tc.write_back_coexist_reclaim_digest, 0);
    tc.sanity_check(&[], &[]);
}

#[test]
fn host_drive_spares_coexisting_host_values_under_an_in_flight_transfer() {
    let mut tc = write_back_core();
    insert(&mut tc, &vec![1, 2], &[10, 11]);
    let handle = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    tc.commit_backup(handle, Tensor::from_slice(&[20i64, 21]), HashMap::new());
    tc.mark_write_through_pending(vec![handle], /* ack_id = */ handle);

    let (mut tr, mut df, mut hf) = (tracker(), frees(), frees());
    accumulate_step(
        tc.drive_host_eviction(FULL, /* num_tokens = */ 2),
        &mut tr,
        &mut df,
        &mut hf,
    );
    assert_eq!(tr[&FULL], 0);
    assert!(tc.arena.node(tc.arena.resolve(handle)).has_host_value(FULL));

    tc.finish_write_through(vec![handle], handle);
    accumulate_step(
        tc.drive_host_eviction(FULL, /* num_tokens = */ 2),
        &mut tr,
        &mut df,
        &mut hf,
    );
    assert_eq!(tr[&FULL], 2);
    assert!(!tc.arena.node(tc.arena.resolve(handle)).has_host_value(FULL));
    tc.sanity_check(&[], &[]);
}

#[test]
fn host_drive_is_a_noop_without_host_leaves() {
    let mut tc = core();
    tc.insert(&crate::unified_tree_core::InsertParams {
        key: &vec![1, 2],
        namespace: Default::default(),
        value: Tensor::from_slice(&[10i64, 11]),
        mamba_value: None,
        prev_prefix_len: 0,
        swa_evicted_seqlen: 0,
        chunked: false,
        priority: 0,
        track_adopted_ranges: false,
    });
    let (mut tr, mut df, mut hf) = (tracker(), frees(), frees());
    accumulate_step(
        tc.drive_host_eviction(FULL, /* num_tokens = */ 5),
        &mut tr,
        &mut df,
        &mut hf,
    );
    assert_eq!(tr[&FULL], 0);
    assert!(hf.is_empty());
    assert_eq!(tc.arena.len(), 2);
    tc.sanity_check(&[], &[]);
}

#[test]
fn host_drive_readmits_the_freed_leafs_parent() {
    let mut tc = core();
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
    let c1 = tc
        .arena
        .alloc_child(
            p,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let c2 = tc
        .arena
        .alloc_child(
            p,
            /* key = */ vec![3],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    for (id, tick) in [(p, 30i64), (c1, 10), (c2, 20)] {
        tc.arena
            .set_host_value(id, FULL, Tensor::from_slice(&[tick]));
        tc.arena.node_mut(id).last_access_counter = tick;
    }
    tc.evictable_host_leaves.add(c1);
    tc.evictable_host_leaves.add(c2);
    let (mut tr, mut df, mut hf) = (tracker(), frees(), frees());
    // p only becomes an H-leaf once both children are gone; the readmission
    // after c2 lets one drive drain the whole chain.
    accumulate_step(
        tc.drive_host_eviction(FULL, /* num_tokens = */ 100),
        &mut tr,
        &mut df,
        &mut hf,
    );
    assert_eq!(tr[&FULL], 3);
    assert_eq!(hf[&FULL].len(), 3);
    assert_eq!(tc.arena.len(), 1);
    assert!(tc.evictable_host_leaves.is_empty());
    tc.sanity_check(&[], &[]);
}

#[test]
fn host_drive_skips_a_stale_heap_entry_for_an_already_freed_leaf() {
    let mut tc = core();
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
    tc.arena
        .set_host_value(p, FULL, Tensor::from_slice(&[10i64]));
    tc.arena
        .set_host_value(c, FULL, Tensor::from_slice(&[20i64]));
    tc.arena.node_mut(p).last_access_counter = 2;
    tc.arena.node_mut(c).last_access_counter = 1;
    // p staged in the set despite its child: the initial heap entry goes
    // stale once c's eviction readmits (and then frees) p.
    tc.evictable_host_leaves.add(p);
    tc.evictable_host_leaves.add(c);
    let (mut tr, mut df, mut hf) = (tracker(), frees(), frees());
    accumulate_step(
        tc.drive_host_eviction(FULL, /* num_tokens = */ 100),
        &mut tr,
        &mut df,
        &mut hf,
    );
    // The duplicate p entry is skipped instead of double-freeing.
    assert_eq!(tr[&FULL], 2);
    assert_eq!(hf[&FULL].len(), 2);
    assert_eq!(tc.arena.len(), 1);
    tc.sanity_check(&[], &[]);
}

// Chain root -> n1 (len 2) -> n2 (len 3) with FULL device values; evictable seeded to 5.
fn lock_chain(tc: &mut UnifiedTreeCore<Vec<i64>>) -> (NodeIdx_, NodeIdx_) {
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 11],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let n2 = tc
        .arena
        .alloc_child(
            n1,
            /* key = */ vec![2, 22, 222],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(n1, FULL, Tensor::from_slice(&[0i64, 1]));
    tc.arena
        .set_device_value(n2, FULL, Tensor::from_slice(&[0i64, 1, 2]));
    tc.component_state_mut(FULL).evictable_size = 5;
    tc.evictable_device_leaves.add(n2);
    (n1, n2)
}

#[test]
fn inc_lock_ref_locks_the_device_path() {
    let mut tc = core();
    let (n1, n2) = lock_chain(&mut tc);
    let result = tc.inc_lock_ref(tc.arena.node(n2).id);
    assert_eq!(result.delta, Some(5));
    assert!(result.skip_lock_node_ids.is_empty());
    assert_eq!(tc.arena.device_lock_ref(n1, FULL), 1);
    assert_eq!(tc.arena.device_lock_ref(n2, FULL), 1);
    let state = tc.component_state(FULL);
    assert_eq!(state.evictable_size, 0);
    assert_eq!(state.protected_size, 5);
    assert!(!tc.evictable_device_leaves.contains(n2));
}

#[test]
fn inc_lock_ref_again_only_bumps_the_refs() {
    let mut tc = core();
    let (n1, n2) = lock_chain(&mut tc);
    tc.inc_lock_ref(tc.arena.node(n2).id);
    let result = tc.inc_lock_ref(tc.arena.node(n2).id);
    assert_eq!(result.delta, Some(0));
    assert_eq!(tc.arena.device_lock_ref(n1, FULL), 2);
    assert_eq!(tc.arena.device_lock_ref(n2, FULL), 2);
    let state = tc.component_state(FULL);
    assert_eq!(state.evictable_size, 0);
    assert_eq!(state.protected_size, 5);
}

#[test]
fn inc_lock_ref_counts_only_newly_locked_nodes() {
    // n1 is already locked via its own path; locking n2 moves only n2's tokens.
    let mut tc = core();
    let (n1, n2) = lock_chain(&mut tc);
    tc.inc_lock_ref(tc.arena.node(n1).id);
    let result = tc.inc_lock_ref(tc.arena.node(n2).id);
    assert_eq!(result.delta, Some(3));
    assert_eq!(tc.arena.device_lock_ref(n1, FULL), 2);
    assert_eq!(tc.arena.device_lock_ref(n2, FULL), 1);
    let state = tc.component_state(FULL);
    assert_eq!(state.evictable_size, 0);
    assert_eq!(state.protected_size, 5);
}

#[test]
fn inc_lock_ref_collects_the_evicted_bottom_segment() {
    // n2 and n3 are evicted (no device value): the walk records both and locks only n1.
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 11],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let n2 = tc
        .arena
        .alloc_child(
            n1,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let n3 = tc
        .arena
        .alloc_child(
            n2,
            /* key = */ vec![3],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(n1, FULL, Tensor::from_slice(&[0i64, 1]));
    tc.component_state_mut(FULL).evictable_size = 2;
    tc.evictable_device_leaves.add(n1);
    let result = tc.inc_lock_ref(tc.arena.node(n3).id);
    assert_eq!(result.delta, Some(2));
    assert_eq!(
        result.skip_lock_node_ids[&FULL],
        HashSet::from([tc.arena.node(n2).id, tc.arena.node(n3).id])
    );
    assert_eq!(tc.arena.device_lock_ref(n1, FULL), 1);
    assert_eq!(tc.arena.device_lock_ref(n2, FULL), 0);
    assert_eq!(tc.arena.device_lock_ref(n3, FULL), 0);
    // The locked ancestor leaves the D-leaf set.
    assert!(!tc.evictable_device_leaves.contains(n1));
}

#[test]
fn lock_round_trips_on_a_root_anchor_are_noops() {
    let mut tc = core();
    let root = tc.arena.root();
    let result = tc.inc_lock_ref(tc.arena.node(root).id);
    assert_eq!(result.delta, Some(0));
    assert!(result.skip_lock_node_ids.is_empty());
    // The protected root keeps its construction-time lock through the pair.
    assert_eq!(tc.arena.device_lock_ref(root, FULL), 1);
    tc.dec_lock_ref(
        tc.arena.node(root).id,
        /* params = */ None,
        /* skip_swa = */ false,
    );
    assert_eq!(tc.arena.device_lock_ref(root, FULL), 1);
}

#[test]
fn lock_walks_stop_at_the_root_of_a_salted_chain() {
    let mut tc = core();
    let lora = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            lora,
            /* key = */ vec![1, 11],
            /* priority = */ 0,
            Some("lora-1"),
        )
        .unwrap();
    tc.arena
        .set_device_value(n1, FULL, Tensor::from_slice(&[0i64, 1]));
    tc.component_state_mut(FULL).evictable_size = 2;
    let result = tc.inc_lock_ref(tc.arena.node(n1).id);
    assert_eq!(result.delta, Some(2));
    assert_eq!(tc.arena.device_lock_ref(n1, FULL), 1);
    // The root keeps its construction-time lock untouched.
    assert_eq!(tc.arena.device_lock_ref(lora, FULL), 1);
    // The release walk stops at the same boundary.
    tc.dec_lock_ref(
        tc.arena.node(n1).id,
        /* params = */ None,
        /* skip_swa = */ false,
    );
    assert_eq!(tc.arena.device_lock_ref(n1, FULL), 0);
    assert_eq!(tc.arena.device_lock_ref(lora, FULL), 1);
}

#[test]
fn lock_walks_treat_a_present_but_empty_value_as_device_on() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let empty: [i64; 0] = [];
    set_value_no_check(
        &mut tc,
        n1,
        ValueSlotIdx::device(FULL),
        Tensor::from_slice(&empty),
    );
    let result = tc.inc_lock_ref(tc.arena.node(n1).id);
    // A present-but-empty value is device-on (Python `value is not None`):
    // locked, zero tokens moved.
    assert_eq!(result.delta, Some(0));
    assert_eq!(tc.arena.device_lock_ref(n1, FULL), 1);
    assert!(result.skip_lock_node_ids.is_empty());
    // The release side moves the same zero tokens back.
    tc.dec_lock_ref(
        tc.arena.node(n1).id,
        /* params = */ None,
        /* skip_swa = */ false,
    );
    assert_eq!(tc.arena.device_lock_ref(n1, FULL), 0);
    let state = tc.component_state(FULL);
    assert_eq!(state.evictable_size, 0);
    assert_eq!(state.protected_size, 0);
}

#[test]
fn dec_lock_ref_unlocks_and_restores_sizes() {
    let mut tc = core();
    let (n1, n2) = lock_chain(&mut tc);
    tc.inc_lock_ref(tc.arena.node(n2).id);
    tc.dec_lock_ref(
        tc.arena.node(n2).id,
        /* params = */ None,
        /* skip_swa = */ false,
    );
    assert_eq!(tc.arena.device_lock_ref(n1, FULL), 0);
    assert_eq!(tc.arena.device_lock_ref(n2, FULL), 0);
    let state = tc.component_state(FULL);
    assert_eq!(state.evictable_size, 5);
    assert_eq!(state.protected_size, 0);
    // The unlocked leaf re-enters the D-leaf set; its valued-child parent does not.
    assert!(tc.evictable_device_leaves.contains(n2));
    assert!(!tc.evictable_device_leaves.contains(n1));
}

#[test]
fn dec_lock_ref_replays_the_skip_set() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 11],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let n2 = tc
        .arena
        .alloc_child(
            n1,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let n3 = tc
        .arena
        .alloc_child(
            n2,
            /* key = */ vec![3],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(n1, FULL, Tensor::from_slice(&[0i64, 1]));
    tc.component_state_mut(FULL).evictable_size = 2;
    let result = tc.inc_lock_ref(tc.arena.node(n3).id);
    let params = DecLockRefParams {
        skip_lock_node_ids: result.skip_lock_node_ids,
        ..Default::default()
    };
    // The still-evicted n2 and n3 are skipped instead of tripping the lock asserts.
    tc.dec_lock_ref(
        tc.arena.node(n3).id,
        Some(&params),
        /* skip_swa = */ false,
    );
    assert_eq!(tc.arena.device_lock_ref(n1, FULL), 0);
    assert_eq!(tc.arena.device_lock_ref(n2, FULL), 0);
    assert_eq!(tc.arena.device_lock_ref(n3, FULL), 0);
    assert_eq!(tc.evictable_size_(FULL), 2);
    // The unlocked ancestor (whose child is valueless) re-enters the D-leaf set.
    assert!(tc.evictable_device_leaves.contains(n1));
}

#[test]
fn temp_lock_skips_the_evicted_anchor_and_mirrors_on_release() {
    // Chain root -> a -> y -> anchor with FULL device values; the anchor is evicted.
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let y = tc
        .arena
        .alloc_child(
            a,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let anchor = tc
        .arena
        .alloc_child(
            y,
            /* key = */ vec![3],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(a, FULL, Tensor::from_slice(&[0i64]));
    tc.arena
        .set_device_value(y, FULL, Tensor::from_slice(&[0i64]));
    tc.component_state_mut(FULL).evictable_size = 3;
    // The temp lock records the evicted anchor and locks only its ancestors.
    let temp_lock = tc.inc_lock_ref(tc.arena.node(anchor).id);
    assert_eq!(tc.arena.device_lock_ref(anchor, FULL), 0);
    assert_eq!(tc.arena.device_lock_ref(y, FULL), 1);
    assert_eq!(tc.arena.device_lock_ref(a, FULL), 1);
    assert_eq!(
        temp_lock.skip_lock_node_ids[&FULL],
        HashSet::from([tc.arena.node(anchor).id])
    );
    // A load-back restores the anchor; the second acquire covers it.
    tc.arena
        .set_device_value(anchor, FULL, Tensor::from_slice(&[0i64]));
    let second_lock = tc.inc_lock_ref(tc.arena.node(anchor).id);
    assert_eq!(tc.arena.device_lock_ref(anchor, FULL), 1);
    assert_eq!(tc.arena.device_lock_ref(y, FULL), 2);
    assert_eq!(tc.arena.device_lock_ref(a, FULL), 2);
    // Releasing the temp lock mirrors its skip set: the anchor keeps its lock.
    let temp_params = DecLockRefParams {
        skip_lock_node_ids: temp_lock.skip_lock_node_ids,
        ..Default::default()
    };
    tc.dec_lock_ref(
        tc.arena.node(anchor).id,
        Some(&temp_params),
        /* skip_swa = */ false,
    );
    assert_eq!(tc.arena.device_lock_ref(anchor, FULL), 1);
    assert_eq!(tc.arena.device_lock_ref(y, FULL), 1);
    assert_eq!(tc.arena.device_lock_ref(a, FULL), 1);
    let second_params = DecLockRefParams {
        skip_lock_node_ids: second_lock.skip_lock_node_ids,
        ..Default::default()
    };
    tc.dec_lock_ref(
        tc.arena.node(anchor).id,
        Some(&second_params),
        /* skip_swa = */ false,
    );
    assert_eq!(tc.arena.device_lock_ref(anchor, FULL), 0);
    assert_eq!(tc.arena.device_lock_ref(y, FULL), 0);
    assert_eq!(tc.arena.device_lock_ref(a, FULL), 0);
}

#[test]
#[should_panic(expected = "has no FULL device value")]
fn dec_lock_ref_panics_without_replaying_the_skip_set() {
    // Dropping the acquire's skip set makes the release walk hit the tombstone.
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 11],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let n2 = tc
        .arena
        .alloc_child(
            n1,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(n1, FULL, Tensor::from_slice(&[0i64, 1]));
    tc.component_state_mut(FULL).evictable_size = 2;
    tc.inc_lock_ref(tc.arena.node(n2).id);
    tc.dec_lock_ref(
        tc.arena.node(n2).id,
        /* params = */ None,
        /* skip_swa = */ false,
    );
}

#[test]
fn dec_lock_ref_with_skip_swa_still_releases_full() {
    let mut tc = core();
    let (_n1, n2) = lock_chain(&mut tc);
    tc.inc_lock_ref(tc.arena.node(n2).id);
    tc.dec_lock_ref(
        tc.arena.node(n2).id,
        /* params = */ None,
        /* skip_swa = */ true,
    );
    assert_eq!(tc.arena.device_lock_ref(n2, FULL), 0);
}

#[test]
fn nested_locks_release_pairwise() {
    // Two acquires then two releases: sizes move only on the outermost pair.
    let mut tc = core();
    let (_n1, n2) = lock_chain(&mut tc);
    tc.inc_lock_ref(tc.arena.node(n2).id);
    tc.inc_lock_ref(tc.arena.node(n2).id);
    tc.dec_lock_ref(
        tc.arena.node(n2).id,
        /* params = */ None,
        /* skip_swa = */ false,
    );
    let state = tc.component_state(FULL);
    assert_eq!(state.evictable_size, 0);
    assert_eq!(state.protected_size, 5);
    assert!(!tc.evictable_device_leaves.contains(n2));
    tc.dec_lock_ref(
        tc.arena.node(n2).id,
        /* params = */ None,
        /* skip_swa = */ false,
    );
    let state = tc.component_state(FULL);
    assert_eq!(state.evictable_size, 5);
    assert_eq!(state.protected_size, 0);
    assert!(tc.evictable_device_leaves.contains(n2));
}

#[test]
#[should_panic(expected = "is not locked")]
fn dec_lock_ref_panics_on_an_unlocked_node() {
    let mut tc = core();
    let (_n1, n2) = lock_chain(&mut tc);
    tc.dec_lock_ref(
        tc.arena.node(n2).id,
        /* params = */ None,
        /* skip_swa = */ false,
    );
}

#[test]
#[should_panic(expected = "FULL invariant broken: evicted ancestor")]
fn inc_lock_ref_panics_on_an_evicted_ancestor() {
    // A value-less n1 below a valued n2 breaks the FULL bottom-up invariant.
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let n2 = tc
        .arena
        .alloc_child(
            n1,
            /* key = */ vec![2, 22, 222],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(n2, FULL, Tensor::from_slice(&[0i64, 1, 2]));
    tc.component_state_mut(FULL).evictable_size = 3;
    tc.inc_lock_ref(tc.arena.node(n2).id);
}

#[test]
#[should_panic(expected = "evictable size underflow")]
fn inc_lock_ref_panics_when_evictable_size_is_unaccounted() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(n1, FULL, Tensor::from_slice(&[0i64]));
    tc.inc_lock_ref(tc.arena.node(n1).id);
}

#[test]
#[should_panic(expected = "protected size underflow")]
fn dec_lock_ref_panics_on_protected_underflow() {
    // A lock ref not accounted through acquire trips the checked release.
    let mut tc = core();
    let (_n1, n2) = lock_chain(&mut tc);
    tc.arena
        .node_mut(n2)
        .set_lock_ref_(ValueSlotIdx::device(FULL), 1);
    tc.dec_lock_ref(
        tc.arena.node(n2).id,
        /* params = */ None,
        /* skip_swa = */ false,
    );
}

fn write_back_core() -> UnifiedTreeCore<Vec<i64>> {
    UnifiedTreeCore::new(
        CacheInitParams {
            is_write_back: true,
            ..Default::default()
        },
        vec![FULL],
    )
}

// A backuped anchor: host value only, seeded into the H-leaf set.
fn host_lock_anchor(tc: &mut UnifiedTreeCore<Vec<i64>>) -> NodeIdx_ {
    let root = tc.arena.root();
    let node = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(node, FULL, Tensor::from_slice(&[10i64, 11]));
    tc.evictable_host_leaves.add(node);
    node
}

#[test]
fn inc_host_lock_ref_pins_the_backuped_anchor() {
    let mut tc = core();
    let node = host_lock_anchor(&mut tc);
    tc.component_state_mut(FULL).evictable_size = 7;
    let result = tc.inc_host_lock_ref(tc.arena.node(node).id);
    assert_eq!(result.delta, None);
    assert!(result.skip_lock_node_ids.is_empty());
    assert_eq!(tc.arena.host_lock_ref(node, FULL), 1);
    // The pinned anchor leaves the H-leaf set; the device tier is untouched.
    assert!(!tc.evictable_host_leaves.contains(node));
    assert_eq!(tc.arena.device_lock_ref(node, FULL), 0);
    let state = tc.component_state(FULL);
    assert_eq!(state.evictable_size, 7);
    assert_eq!(state.protected_size, 0);
}

#[test]
fn inc_host_lock_ref_again_only_bumps_the_counter() {
    let mut tc = core();
    let node = host_lock_anchor(&mut tc);
    tc.inc_host_lock_ref(tc.arena.node(node).id);
    tc.inc_host_lock_ref(tc.arena.node(node).id);
    assert_eq!(tc.arena.host_lock_ref(node, FULL), 2);
    assert!(!tc.evictable_host_leaves.contains(node));
}

#[test]
fn inc_host_lock_ref_pins_only_the_anchor_not_its_ancestors() {
    // Both chain nodes are backuped; only the anchor's host counter moves.
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let n2 = tc
        .arena
        .alloc_child(
            n1,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(n1, FULL, Tensor::from_slice(&[0i64]));
    tc.arena
        .set_host_value(n2, FULL, Tensor::from_slice(&[0i64]));
    tc.inc_host_lock_ref(tc.arena.node(n2).id);
    assert_eq!(tc.arena.host_lock_ref(n2, FULL), 1);
    assert_eq!(tc.arena.host_lock_ref(n1, FULL), 0);
}

#[test]
fn inc_host_lock_ref_skips_an_anchor_without_a_host_value() {
    let mut tc = core();
    let (_n1, n2) = lock_chain(&mut tc);
    tc.inc_host_lock_ref(tc.arena.node(n2).id);
    assert_eq!(tc.arena.host_lock_ref(n2, FULL), 0);
}

#[test]
fn host_lock_round_trips_on_a_root_anchor_are_noops() {
    let mut tc = core();
    let root = tc.arena.root();
    let result = tc.inc_host_lock_ref(tc.arena.node(root).id);
    assert_eq!(result.delta, None);
    assert_eq!(tc.arena.host_lock_ref(root, FULL), 0);
    tc.dec_host_lock_ref(tc.arena.node(root).id, /* params = */ None);
    assert_eq!(tc.arena.host_lock_ref(root, FULL), 0);
}

#[test]
fn inc_host_lock_ref_under_write_back_pins_a_device_only_anchor() {
    let mut tc = write_back_core();
    let (_n1, n2) = lock_chain(&mut tc);
    tc.inc_host_lock_ref(tc.arena.node(n2).id);
    assert_eq!(tc.arena.host_lock_ref(n2, FULL), 1);
    // The write-back host lock is a pure counter: no size shifts.
    let state = tc.component_state(FULL);
    assert_eq!(state.evictable_size, 5);
    assert_eq!(state.protected_size, 0);
}

#[test]
fn dec_host_lock_ref_unpins_and_restores_the_h_leaf_set() {
    let mut tc = core();
    let node = host_lock_anchor(&mut tc);
    tc.component_state_mut(FULL).evictable_size = 7;
    tc.inc_host_lock_ref(tc.arena.node(node).id);
    tc.dec_host_lock_ref(tc.arena.node(node).id, /* params = */ None);
    assert_eq!(tc.arena.host_lock_ref(node, FULL), 0);
    assert!(tc.evictable_host_leaves.contains(node));
    let state = tc.component_state(FULL);
    assert_eq!(state.evictable_size, 7);
    assert_eq!(state.protected_size, 0);
}

#[test]
fn dec_host_lock_ref_on_an_unlocked_anchor_is_a_noop() {
    let mut tc = core();
    let node = host_lock_anchor(&mut tc);
    tc.dec_host_lock_ref(tc.arena.node(node).id, /* params = */ None);
    assert_eq!(tc.arena.host_lock_ref(node, FULL), 0);
}

#[test]
fn dec_host_lock_ref_keeps_the_counter_when_the_host_value_is_gone() {
    // A host-evicted anchor keeps its pin count under write-through.
    let mut tc = core();
    let node = host_lock_anchor(&mut tc);
    tc.inc_host_lock_ref(tc.arena.node(node).id);
    let _ = tc.arena.take_host_value(node, FULL);
    tc.dec_host_lock_ref(tc.arena.node(node).id, /* params = */ None);
    assert_eq!(tc.arena.host_lock_ref(node, FULL), 1);
}

#[test]
fn host_lock_round_trip_under_write_back_is_a_pure_counter() {
    let mut tc = write_back_core();
    let (_n1, n2) = lock_chain(&mut tc);
    tc.inc_host_lock_ref(tc.arena.node(n2).id);
    tc.dec_host_lock_ref(tc.arena.node(n2).id, /* params = */ None);
    assert_eq!(tc.arena.host_lock_ref(n2, FULL), 0);
    let state = tc.component_state(FULL);
    assert_eq!(state.evictable_size, 5);
    assert_eq!(state.protected_size, 0);
}

#[test]
fn acquire_host_arm_updates_the_h_leaf_set_without_the_dispatcher() {
    let mut tc = core();
    let node = host_lock_anchor(&mut tc);
    FullComponent.acquire_component_lock(
        &mut tc,
        node,
        IncLockRefResult::default(),
        /* lock_host = */ true,
    );
    assert!(!tc.evictable_host_leaves.contains(node));
}

#[test]
fn release_host_arm_updates_the_h_leaf_set_without_the_dispatcher() {
    let mut tc = core();
    let node = host_lock_anchor(&mut tc);
    tc.inc_host_lock_ref(tc.arena.node(node).id);
    FullComponent.release_component_lock(
        &mut tc, node, /* params = */ None, /* lock_host = */ true,
    );
    assert!(tc.evictable_host_leaves.contains(node));
}

#[test]
fn nested_host_locks_release_pairwise() {
    let mut tc = core();
    let node = host_lock_anchor(&mut tc);
    tc.inc_host_lock_ref(tc.arena.node(node).id);
    tc.inc_host_lock_ref(tc.arena.node(node).id);
    tc.dec_host_lock_ref(tc.arena.node(node).id, /* params = */ None);
    assert_eq!(tc.arena.host_lock_ref(node, FULL), 1);
    assert!(!tc.evictable_host_leaves.contains(node));
    tc.dec_host_lock_ref(tc.arena.node(node).id, /* params = */ None);
    assert_eq!(tc.arena.host_lock_ref(node, FULL), 0);
    assert!(tc.evictable_host_leaves.contains(node));
}

#[test]
fn node_has_component_data_tracks_each_slot() {
    let mut tc = core();
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
    assert!(!crate::components::node_has_component_data(
        &tc.arena,
        node,
        FULL,
        EvictLayer::Device
    ));
    assert!(!crate::components::node_has_component_data(
        &tc.arena,
        node,
        FULL,
        EvictLayer::Host
    ));

    tc.arena
        .set_device_value(node, FULL, Tensor::from_slice(&[10i64]));
    assert!(crate::components::node_has_component_data(
        &tc.arena,
        node,
        FULL,
        EvictLayer::Device
    ));
    assert!(!crate::components::node_has_component_data(
        &tc.arena,
        node,
        FULL,
        EvictLayer::Host
    ));

    tc.arena
        .set_host_value(node, FULL, Tensor::from_slice(&[20i64]));
    assert!(crate::components::node_has_component_data(
        &tc.arena,
        node,
        FULL,
        EvictLayer::Host
    ));
}

#[test]
fn match_validator_device_only_accepts_device_backed_only() {
    let mut tc = core();
    let root = tc.arena.root();
    let dev = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let host = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let empty = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![3],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(dev, FULL, Tensor::from_slice(&[0i64]));
    tc.arena
        .set_host_value(host, FULL, Tensor::from_slice(&[0i64]));
    let mut validator = <FullComponent as TreeComponent<Vec<i64>>>::create_match_validator(
        &FullComponent,
        &tc,
        true,
    );
    assert!(validator(&tc, dev));
    assert!(!validator(&tc, host));
    assert!(!validator(&tc, empty));
}

#[test]
fn match_validator_hicache_accepts_device_or_host_backed() {
    let mut tc = core();
    let root = tc.arena.root();
    let dev = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let host = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let empty = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![3],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(dev, FULL, Tensor::from_slice(&[0i64]));
    tc.arena
        .set_host_value(host, FULL, Tensor::from_slice(&[0i64]));
    let mut validator = <FullComponent as TreeComponent<Vec<i64>>>::create_match_validator(
        &FullComponent,
        &tc,
        false,
    );
    assert!(validator(&tc, dev));
    assert!(validator(&tc, host));
    assert!(!validator(&tc, empty));
}

#[test]
fn match_validator_accepts_node_with_both_device_and_host() {
    let mut tc = core();
    let root = tc.arena.root();
    let both = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(both, FULL, Tensor::from_slice(&[0i64]));
    tc.arena
        .set_host_value(both, FULL, Tensor::from_slice(&[0i64]));
    let mut device_only = <FullComponent as TreeComponent<Vec<i64>>>::create_match_validator(
        &FullComponent,
        &tc,
        true,
    );
    let mut hicache = <FullComponent as TreeComponent<Vec<i64>>>::create_match_validator(
        &FullComponent,
        &tc,
        false,
    );
    assert!(device_only(&tc, both));
    assert!(hicache(&tc, both));
}

#[test]
fn match_validator_verdict_is_per_node_not_stateful() {
    let mut tc = core();
    let root = tc.arena.root();
    let dev = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let empty = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(dev, FULL, Tensor::from_slice(&[0i64]));
    // One validator reused across nodes returns each node's own verdict (FULL is stateless).
    let mut validator = <FullComponent as TreeComponent<Vec<i64>>>::create_match_validator(
        &FullComponent,
        &tc,
        true,
    );
    assert!(validator(&tc, dev));
    assert!(!validator(&tc, empty));
    assert!(validator(&tc, dev));
    assert!(!validator(&tc, empty));
}

#[test]
fn match_validator_accepts_present_but_empty_device_value() {
    let mut tc = core();
    let root = tc.arena.root();
    let empty_dev = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let no_value = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let empty: [i64; 0] = [];
    set_value_no_check(
        &mut tc,
        empty_dev,
        ValueSlotIdx::device(FULL),
        Tensor::from_slice(&empty),
    );
    // A present-but-empty device value is a boundary (Python `value is not None`); a
    // truly value-less node is not.
    let mut device_only = <FullComponent as TreeComponent<Vec<i64>>>::create_match_validator(
        &FullComponent,
        &tc,
        true,
    );
    assert!(device_only(&tc, empty_dev));
    assert!(!device_only(&tc, no_value));
}

// Chain root -> n1 -> n2 -> n3 with FULL host values on n2 (len 2) and n3 (len 3).
fn host_hit_chain() -> (UnifiedTreeCore<Vec<i64>>, NodeIdx_, NodeIdx_) {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let n2 = tc
        .arena
        .alloc_child(
            n1,
            /* key = */ vec![2, 22],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let n3 = tc
        .arena
        .alloc_child(
            n2,
            /* key = */ vec![3, 33, 333],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(n2, FULL, Tensor::from_slice(&[0i64, 1]));
    tc.arena
        .set_host_value(n3, FULL, Tensor::from_slice(&[0i64, 1, 2]));
    (tc, n1, n3)
}

fn finalize(tc: &UnifiedTreeCore<Vec<i64>>, result: MatchResult) -> MatchResult {
    FullComponent.finalize_match_result_in_tree_core(
        tc,
        result,
        &MatchPrefixParams {
            key: &Vec::new(),
            namespace: Default::default(),
        },
        &[],
        0,
    )
}

#[test]
fn finalize_sums_full_host_hits_between_best_and_last_device() {
    let (tc, n1, n3) = host_hit_chain();
    let out = finalize(
        &tc,
        MatchResult {
            last_device_node_id: tc.arena.node(n1).id,
            best_match_node_id: tc.arena.node(n3).id,
            host_hit_length: 0,
            ..tc.empty_match_result()
        },
    );
    assert_eq!(out.host_hit_length, 5);
    // Other MatchResult fields pass through unchanged (Python returns via _replace).
    assert_eq!(out.last_device_node_id, tc.arena.node(n1).id);
    assert_eq!(out.best_match_node_id, tc.arena.node(n3).id);
}

#[test]
fn finalize_keeps_larger_existing_host_hit_length() {
    let (tc, n1, n3) = host_hit_chain();
    let out = finalize(
        &tc,
        MatchResult {
            last_device_node_id: tc.arena.node(n1).id,
            best_match_node_id: tc.arena.node(n3).id,
            host_hit_length: 100,
            ..tc.empty_match_result()
        },
    );
    assert_eq!(out.host_hit_length, 100);
}

#[test]
fn finalize_overrides_smaller_existing_host_hit_length() {
    let (tc, n1, n3) = host_hit_chain();
    let out = finalize(
        &tc,
        MatchResult {
            last_device_node_id: tc.arena.node(n1).id,
            best_match_node_id: tc.arena.node(n3).id,
            host_hit_length: 2,
            ..tc.empty_match_result()
        },
    );
    assert_eq!(out.host_hit_length, 5);
}

#[test]
fn finalize_is_noop_without_full_host_values() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let n2 = tc
        .arena
        .alloc_child(
            n1,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let out = finalize(
        &tc,
        MatchResult {
            last_device_node_id: tc.arena.node(root).id,
            best_match_node_id: tc.arena.node(n2).id,
            host_hit_length: 4,
            ..tc.empty_match_result()
        },
    );
    assert_eq!(out.host_hit_length, 4);
}

#[test]
fn finalize_walks_a_salted_chain_up_to_the_root() {
    let mut tc = core();
    let lora = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            lora,
            /* key = */ vec![1, 11],
            /* priority = */ 0,
            Some("lora-1"),
        )
        .unwrap();
    let b = tc
        .arena
        .alloc_child(
            a,
            /* key = */ vec![2, 22, 222],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(a, FULL, Tensor::from_slice(&[0i64, 1]));
    tc.arena
        .set_host_value(b, FULL, Tensor::from_slice(&[0i64, 1, 2]));
    // The root is last_device_node_id; its own host value is never counted.
    set_value_no_check(
        &mut tc,
        lora,
        ValueSlotIdx::host(FULL),
        Tensor::from_slice(&[0i64, 1, 2, 3]),
    );
    let out = finalize(
        &tc,
        MatchResult {
            last_device_node_id: tc.arena.node(lora).id,
            best_match_node_id: tc.arena.node(b).id,
            host_hit_length: 0,
            ..tc.empty_match_result()
        },
    );
    assert_eq!(out.host_hit_length, 5);
}

#[test]
#[should_panic(expected = "hit root")]
fn finalize_panics_when_walk_hits_root_before_last_device() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    // last_device_node_id is not an ancestor of best_match_node_id -> corrupt result.
    finalize(
        &tc,
        MatchResult {
            last_device_node_id: tc.arena.node(n1).id,
            best_match_node_id: tc.arena.node(root).id,
            host_hit_length: 0,
            ..tc.empty_match_result()
        },
    );
}

#[test]
fn match_validator_hicache_accepts_present_but_empty_host_value() {
    let mut tc = core();
    let root = tc.arena.root();
    let host = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let empty: [i64; 0] = [];
    set_value_no_check(
        &mut tc,
        host,
        ValueSlotIdx::host(FULL),
        Tensor::from_slice(&empty),
    );
    // A present-but-empty host value is backuped (Python `host_value is not None`).
    let mut hicache = <FullComponent as TreeComponent<Vec<i64>>>::create_match_validator(
        &FullComponent,
        &tc,
        false,
    );
    let mut device_only = <FullComponent as TreeComponent<Vec<i64>>>::create_match_validator(
        &FullComponent,
        &tc,
        true,
    );
    assert!(hicache(&tc, host));
    assert!(!device_only(&tc, host));
}

#[test]
#[should_panic(expected = "out of bounds")]
fn match_validator_panics_on_missing_node() {
    let tc = core();
    let mut validator = <FullComponent as TreeComponent<Vec<i64>>>::create_match_validator(
        &FullComponent,
        &tc,
        true,
    );
    validator(&tc, NodeIdx_(999));
}

#[test]
#[should_panic(expected = "is not allocated")]
fn finalize_panics_on_missing_best_match_node() {
    let tc = core();
    let root = tc.arena.root();
    finalize(
        &tc,
        MatchResult {
            last_device_node_id: tc.arena.node(root).id,
            best_match_node_id: 999,
            host_hit_length: 0,
            ..tc.empty_match_result()
        },
    );
}

#[test]
fn finalize_empty_walk_when_best_equals_last_device() {
    let (tc, _n1, n3) = host_hit_chain();
    // best_match_node_id == last_device_node_id -> no hops -> unchanged.
    let out = finalize(
        &tc,
        MatchResult {
            last_device_node_id: tc.arena.node(n3).id,
            best_match_node_id: tc.arena.node(n3).id,
            host_hit_length: 2,
            ..tc.empty_match_result()
        },
    );
    assert_eq!(out.host_hit_length, 2);
}

#[test]
fn finalize_excludes_last_device_nodes_own_host_value() {
    // root -> n1 -> n2 -> n3, host on all three; last_device=n2 counts only n3.
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 11, 111, 1111],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let n2 = tc
        .arena
        .alloc_child(
            n1,
            /* key = */ vec![2, 22],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let n3 = tc
        .arena
        .alloc_child(
            n2,
            /* key = */ vec![3, 33, 333],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(n1, FULL, Tensor::from_slice(&[0i64, 1, 2, 3]));
    tc.arena
        .set_host_value(n2, FULL, Tensor::from_slice(&[0i64, 1]));
    tc.arena
        .set_host_value(n3, FULL, Tensor::from_slice(&[0i64, 1, 2]));
    let out = finalize(
        &tc,
        MatchResult {
            last_device_node_id: tc.arena.node(n2).id,
            best_match_node_id: tc.arena.node(n3).id,
            host_hit_length: 0,
            ..tc.empty_match_result()
        },
    );
    assert_eq!(out.host_hit_length, 3);
}

#[test]
fn finalize_skips_nodes_without_host_value() {
    // root -> n1 -> n2 -> n3, host only on n1 and n3; a value-less n2 contributes 0.
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let n2 = tc
        .arena
        .alloc_child(
            n1,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let n3 = tc
        .arena
        .alloc_child(
            n2,
            /* key = */ vec![3, 33, 333],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(n1, FULL, Tensor::from_slice(&[0i64]));
    tc.arena
        .set_host_value(n3, FULL, Tensor::from_slice(&[0i64, 1, 2]));
    let out = finalize(
        &tc,
        MatchResult {
            last_device_node_id: tc.arena.node(root).id,
            best_match_node_id: tc.arena.node(n3).id,
            host_hit_length: 0,
            ..tc.empty_match_result()
        },
    );
    assert_eq!(out.host_hit_length, 4);
}

// Test-only split setup: root -> parent (key len 2) -> child, as _split_node leaves them.
fn nodes(tc: &mut UnifiedTreeCore<Vec<i64>>) -> (NodeIdx_, NodeIdx_) {
    let root = tc.arena.root();
    let parent = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let child = tc
        .arena
        .alloc_child(
            parent,
            /* key = */ vec![3],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    (parent, child)
}

#[test]
fn redistribute_copies_device_lock_ref_to_new_parent() {
    let mut tc = core();
    let (parent, child) = nodes(&mut tc);
    tc.arena
        .node_mut(child)
        .set_lock_ref_(ValueSlotIdx::device(FULL), 3);
    FullComponent.redistribute_on_node_split(&mut tc, parent, child);
    assert_eq!(tc.arena.device_lock_ref(parent, FULL), 3);
    assert_eq!(tc.arena.device_lock_ref(child, FULL), 3);
}

#[test]
fn redistribute_splits_device_value() {
    let mut tc = core();
    let (parent, child) = nodes(&mut tc);
    let original = Tensor::from_slice(&[10i64, 11, 12]);
    // Mid-split state: the child's key is already trimmed, its value not yet split.
    set_value_no_check(&mut tc, child, ValueSlotIdx::device(FULL), original.copy());
    FullComponent.redistribute_on_node_split(&mut tc, parent, child);
    let parent_node = tc.arena.node(parent);
    assert!(
        parent_node
            .device_value(FULL)
            .equal(&original.narrow(0, 0, 2))
    );
    let child_node = tc.arena.node(child);
    assert!(
        child_node
            .device_value(FULL)
            .equal(&original.narrow(0, 2, 1))
    );
}

#[test]
fn redistribute_splits_host_value() {
    let mut tc = core();
    let (parent, child) = nodes(&mut tc);
    let original = Tensor::from_slice(&[20i64, 21, 22]);
    set_value_no_check(&mut tc, child, ValueSlotIdx::host(FULL), original.copy());
    FullComponent.redistribute_on_node_split(&mut tc, parent, child);
    let parent_node = tc.arena.node(parent);
    assert!(
        parent_node
            .host_value(FULL)
            .equal(&original.narrow(0, 0, 2))
    );
    let child_node = tc.arena.node(child);
    assert!(child_node.host_value(FULL).equal(&original.narrow(0, 2, 1)));
}

#[test]
fn redistribute_splits_device_value_with_bigram_key() {
    let mut tc: UnifiedTreeCore<Vec<(i64, i64)>> =
        UnifiedTreeCore::new(CacheInitParams::default(), vec![FULL]);
    let root = tc.arena.root();
    let parent = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![(1, 2), (3, 4)],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let child = tc
        .arena
        .alloc_child(
            parent,
            /* key = */ vec![(5, 6)],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    // One value row per atom: a 2-atom bigram parent key takes 2 rows.
    let original = Tensor::from_slice(&[10i64, 11, 12]);
    // Mid-split state: the child's key is already trimmed, its value not yet split.
    set_value_no_check(&mut tc, child, ValueSlotIdx::device(FULL), original.copy());
    FullComponent.redistribute_on_node_split(&mut tc, parent, child);
    let parent_node = tc.arena.node(parent);
    assert!(
        parent_node
            .device_value(FULL)
            .equal(&original.narrow(0, 0, 2))
    );
    let child_node = tc.arena.node(child);
    assert!(
        child_node
            .device_value(FULL)
            .equal(&original.narrow(0, 2, 1))
    );
}

#[test]
fn redistribute_leaves_tombstoned_child_values_none() {
    let mut tc = core();
    let (parent, child) = nodes(&mut tc);
    FullComponent.redistribute_on_node_split(&mut tc, parent, child);
    let parent_node = tc.arena.node(parent);
    assert!(!parent_node.has_device_value(FULL));
    assert!(!parent_node.has_host_value(FULL));
    assert_eq!(parent_node.device_lock_ref(FULL), 0);
    let child_node = tc.arena.node(child);
    assert!(!child_node.has_device_value(FULL));
    assert!(!child_node.has_host_value(FULL));
}

#[test]
fn redistribute_device_only_child_leaves_host_none() {
    let mut tc = core();
    let (parent, child) = nodes(&mut tc);
    set_value_no_check(
        &mut tc,
        child,
        ValueSlotIdx::device(FULL),
        Tensor::from_slice(&[10i64, 11, 12]),
    );
    FullComponent.redistribute_on_node_split(&mut tc, parent, child);
    let parent_node = tc.arena.node(parent);
    assert!(parent_node.has_device_value(FULL));
    assert!(!parent_node.has_host_value(FULL));
    let child_node = tc.arena.node(child);
    assert!(!child_node.has_host_value(FULL));
}

#[test]
fn redistribute_halves_do_not_alias_source() {
    let mut tc = core();
    let (parent, child) = nodes(&mut tc);
    let mut source = Tensor::from_slice(&[10i64, 11, 12]);
    let expected = source.copy();
    set_value_no_check(
        &mut tc,
        child,
        ValueSlotIdx::device(FULL),
        source.shallow_clone(),
    );
    FullComponent.redistribute_on_node_split(&mut tc, parent, child);
    // Writing through the source storage must not leak into either half.
    let _ = source.fill_(99);
    let parent_node = tc.arena.node(parent);
    assert!(
        parent_node
            .device_value(FULL)
            .equal(&expected.narrow(0, 0, 2))
    );
    let child_node = tc.arena.node(child);
    assert!(
        child_node
            .device_value(FULL)
            .equal(&expected.narrow(0, 2, 1))
    );
}

#[test]
fn redistribute_preserves_preexisting_parent_slot_value() {
    let mut tc = core();
    let (parent, child) = nodes(&mut tc);
    let sentinel = Tensor::from_slice(&[7i64, 8]);
    tc.arena.set_device_value(parent, FULL, sentinel.copy());
    // Child has no device value: the parent's existing slot must be left untouched.
    FullComponent.redistribute_on_node_split(&mut tc, parent, child);
    let parent_node = tc.arena.node(parent);
    assert!(parent_node.device_value(FULL).equal(&sentinel));
}

#[test]
fn redistribute_does_not_copy_host_lock_ref() {
    let mut tc = core();
    let (parent, child) = nodes(&mut tc);
    tc.arena
        .node_mut(child)
        .set_lock_ref_(ValueSlotIdx::host(FULL), 5);
    FullComponent.redistribute_on_node_split(&mut tc, parent, child);
    assert_eq!(tc.arena.host_lock_ref(parent, FULL), 0);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn redistribute_panics_on_missing_new_parent() {
    let mut tc = core();
    let (_parent, child) = nodes(&mut tc);
    FullComponent.redistribute_on_node_split(&mut tc, NodeIdx_(999), child);
}

#[test]
fn evict_device_pushes_value_and_decrements_size() {
    let mut tc = core();
    let root = tc.arena.root();
    let node = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2, 3],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let original = Tensor::from_slice(&[10i64, 11, 12]);
    tc.arena.set_device_value(node, FULL, original.copy());
    tc.component_state_mut(FULL).evictable_size = 5;
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    let (freed, host_freed) = FullComponent.evict_component(
        &mut tc,
        node,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Device,
    );
    assert_eq!(freed, 3);
    assert_eq!(host_freed, 0);
    assert_eq!(tc.evictable_size_(FULL), 2);
    let pushed = &device_frees[&FULL];
    assert_eq!(pushed.len(), 1);
    assert!(pushed[0].equal(&original));
    // The device value is NOT tombstoned here (deferred to the cascade).
    assert!(tc.arena.has_device_value(node, FULL));
}

#[test]
fn evict_device_on_valueless_node_is_noop() {
    let mut tc = core();
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
    tc.component_state_mut(FULL).evictable_size = 5;
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    let (freed, host_freed) = FullComponent.evict_component(
        &mut tc,
        node,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Device,
    );
    assert_eq!(freed, 0);
    assert_eq!(host_freed, 0);
    assert_eq!(tc.evictable_size_(FULL), 5);
    assert!(!device_frees.contains_key(&FULL));
}

#[test]
fn evict_device_pushes_a_present_but_empty_value() {
    let mut tc = core();
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
    let empty: [i64; 0] = [];
    set_value_no_check(
        &mut tc,
        node,
        ValueSlotIdx::device(FULL),
        Tensor::from_slice(&empty),
    );
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    let (freed, _) = FullComponent.evict_component(
        &mut tc,
        node,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Device,
    );
    assert_eq!(freed, 0);
    assert_eq!(device_frees[&FULL].len(), 1);
}

#[test]
fn evict_host_pushes_host_value_and_tombstones() {
    let mut tc = core();
    let root = tc.arena.root();
    let node = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let host_original = Tensor::from_slice(&[20i64, 21]);
    tc.arena
        .set_device_value(node, FULL, Tensor::from_slice(&[10i64, 11]));
    tc.arena.set_host_value(node, FULL, host_original.copy());
    tc.component_state_mut(FULL).evictable_size = 5;
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    let (freed, host_freed) = FullComponent.evict_component(
        &mut tc,
        node,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Host,
    );
    assert_eq!(freed, 0);
    assert_eq!(host_freed, 2);
    let pushed = &host_frees[&FULL];
    assert_eq!(pushed.len(), 1);
    assert!(pushed[0].equal(&host_original));
    // Host is tombstoned; device data and the evictable counter are untouched.
    let node_ref = tc.arena.node(node);
    assert!(!node_ref.has_host_value(FULL));
    assert!(node_ref.has_device_value(FULL));
    assert!(device_frees.is_empty());
    assert_eq!(tc.evictable_size_(FULL), 5);
}

#[test]
fn evict_host_on_node_without_host_value_is_noop() {
    let mut tc = core();
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
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    let (freed, host_freed) = FullComponent.evict_component(
        &mut tc,
        node,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Host,
    );
    assert_eq!(freed, 0);
    assert_eq!(host_freed, 0);
    assert!(host_frees.is_empty());
}

#[test]
fn evict_host_pushes_a_present_but_empty_host_value() {
    let mut tc = core();
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
    let empty: [i64; 0] = [];
    set_value_no_check(
        &mut tc,
        node,
        ValueSlotIdx::host(FULL),
        Tensor::from_slice(&empty),
    );
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    let (_, host_freed) = FullComponent.evict_component(
        &mut tc,
        node,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Host,
    );
    assert_eq!(host_freed, 0);
    assert_eq!(host_frees[&FULL].len(), 1);
    assert!(!tc.arena.has_host_value(node, FULL));
}

#[test]
fn evict_host_accumulates_into_shared_host_frees() {
    let mut tc = core();
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
            root,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let host_a = Tensor::from_slice(&[20i64, 21]);
    let host_b = Tensor::from_slice(&[30i64]);
    tc.arena.set_host_value(a, FULL, host_a.copy());
    tc.arena.set_host_value(b, FULL, host_b.copy());
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    let freed_a = FullComponent.evict_component(
        &mut tc,
        a,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Host,
    );
    let freed_b = FullComponent.evict_component(
        &mut tc,
        b,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Host,
    );
    assert_eq!(freed_a, (0, 2));
    assert_eq!(freed_b, (0, 1));
    let pushed = &host_frees[&FULL];
    assert_eq!(pushed.len(), 2);
    assert!(pushed[0].equal(&host_a));
    assert!(pushed[1].equal(&host_b));
}

#[test]
fn evict_device_leaves_host_value_untouched() {
    let mut tc = core();
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
    tc.arena
        .set_device_value(node, FULL, Tensor::from_slice(&[10i64]));
    tc.arena
        .set_host_value(node, FULL, Tensor::from_slice(&[20i64]));
    tc.component_state_mut(FULL).evictable_size = 1;
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    let (freed, host_freed) = FullComponent.evict_component(
        &mut tc,
        node,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Device,
    );
    assert_eq!(freed, 1);
    assert_eq!(host_freed, 0);
    assert!(host_frees.is_empty());
    assert!(tc.arena.has_host_value(node, FULL));
}

#[test]
fn evict_all_frees_both_layers() {
    let mut tc = core();
    let root = tc.arena.root();
    let node = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2, 3],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let device_original = Tensor::from_slice(&[10i64, 11, 12]);
    let host_original = Tensor::from_slice(&[20i64, 21, 22]);
    tc.arena
        .set_device_value(node, FULL, device_original.copy());
    tc.arena.set_host_value(node, FULL, host_original.copy());
    tc.component_state_mut(FULL).evictable_size = 3;
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    let (freed, host_freed) = FullComponent.evict_component(
        &mut tc,
        node,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::All,
    );
    assert_eq!(freed, 3);
    assert_eq!(host_freed, 3);
    assert!(device_frees[&FULL][0].equal(&device_original));
    assert!(host_frees[&FULL][0].equal(&host_original));
    // Device stays for the cascade to tombstone; host is tombstoned inline.
    let node_ref = tc.arena.node(node);
    assert!(node_ref.has_device_value(FULL));
    assert!(!node_ref.has_host_value(FULL));
    assert_eq!(tc.evictable_size_(FULL), 0);
}

#[test]
fn evict_device_accumulates_into_shared_device_frees() {
    let mut tc = core();
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
            root,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let value_a = Tensor::from_slice(&[10i64, 11]);
    let value_b = Tensor::from_slice(&[20i64]);
    tc.arena.set_device_value(a, FULL, value_a.copy());
    tc.arena.set_device_value(b, FULL, value_b.copy());
    tc.component_state_mut(FULL).evictable_size = 3;
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    FullComponent.evict_component(
        &mut tc,
        a,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Device,
    );
    FullComponent.evict_component(
        &mut tc,
        b,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Device,
    );
    let pushed = &device_frees[&FULL];
    assert_eq!(pushed.len(), 2);
    assert!(pushed[0].equal(&value_a));
    assert!(pushed[1].equal(&value_b));
    assert_eq!(tc.evictable_size_(FULL), 0);
}

#[test]
fn evict_device_pushed_entry_aliases_the_node_value() {
    let mut tc = core();
    let root = tc.arena.root();
    let node = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2, 3],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(node, FULL, Tensor::from_slice(&[10i64, 11, 12]));
    tc.component_state_mut(FULL).evictable_size = 3;
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    FullComponent.evict_component(
        &mut tc,
        node,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Device,
    );
    // The pushed entry shares storage with the node's value (Python appends cd.value).
    let _ = device_frees.get_mut(&FULL).unwrap()[0].fill_(99);
    assert!(
        tc.arena
            .device_value(node, FULL)
            .equal(&Tensor::from_slice(&[99i64, 99, 99]))
    );
}

#[test]
#[should_panic(expected = "out of bounds")]
fn evict_panics_on_missing_node() {
    let mut tc = core();
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    FullComponent.evict_component(
        &mut tc,
        NodeIdx_(999),
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Device,
    );
}

#[test]
#[should_panic(expected = "evictable size underflow")]
fn evict_panics_when_size_would_underflow() {
    let mut tc = core();
    let root = tc.arena.root();
    let node = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2, 3],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(node, FULL, Tensor::from_slice(&[10i64, 11, 12]));
    // Counter still 0: evicting 3 tokens must fail loudly, not wrap.
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    FullComponent.evict_component(
        &mut tc,
        node,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Device,
    );
}

#[test]
#[should_panic(expected = "EvictLayer::All")]
fn node_has_component_data_panics_on_all_layer() {
    let mut tc = core();
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
    crate::components::node_has_component_data(&tc.arena, node, FULL, EvictLayer::All);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn node_has_component_data_panics_on_missing_node() {
    let tc = core();
    crate::components::node_has_component_data(&tc.arena, NodeIdx_(999), FULL, EvictLayer::Device);
}

// Chain root -> a (device-on, backuped) -> b (host-only) -> c (host-only).
fn load_back_chain(tc: &mut UnifiedTreeCore<Vec<i64>>) -> (NodeIdx_, NodeIdx_, NodeIdx_) {
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(a, FULL, Tensor::from_slice(&[10i64]));
    tc.arena
        .set_host_value(a, FULL, Tensor::from_slice(&[20i64]));
    let b = tc
        .arena
        .alloc_child(
            a,
            /* key = */ vec![2, 3],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(b, FULL, Tensor::from_slice(&[21i64, 22]));
    let c = tc
        .arena
        .alloc_child(
            b,
            /* key = */ vec![4],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(c, FULL, Tensor::from_slice(&[23i64]));
    (a, b, c)
}

#[test]
fn build_hicache_transfers_returns_none_for_non_load_back_phases() {
    let mut tc = core();
    let (a, _b, _c) = load_back_chain(&mut tc);
    for phase in [
        CacheTransferPhase::BackupHost,
        CacheTransferPhase::BackupStorage,
        CacheTransferPhase::Prefetch,
    ] {
        let transfers = FullComponent
            .build_hicache_transfers(
                &tc, a, phase, /* mamba_pool_idx = */ None, /* host_indices = */ None,
                /* token_ids = */ None, /* prefetch_tokens = */ 0,
                /* last_hash = */ None,
            )
            .unwrap();
        assert!(transfers.is_none());
    }
}

#[test]
fn load_back_build_collects_the_evicted_suffix_ancestors_first() {
    let mut tc = core();
    let (_a, b, c) = load_back_chain(&mut tc);
    let transfers = FullComponent
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
    assert_eq!(xfer.name, PoolName::Kv);
    // The walk stops at a's device value; b's host pages precede c's.
    assert!(
        xfer.host_indices
            .as_ref()
            .unwrap()
            .equal(&Tensor::from_slice(&[21i64, 22, 23]))
    );
    assert!(xfer.device_indices.is_none());
    assert_eq!(
        xfer.nodes_to_load,
        Some(vec![tc.arena.node(b).id, tc.arena.node(c).id])
    );
}

#[test]
fn load_back_build_returns_an_empty_cpu_transfer_for_a_device_backed_node() {
    let mut tc = core();
    let (a, _b, _c) = load_back_chain(&mut tc);
    let transfers = FullComponent
        .build_hicache_transfers(
            &tc,
            a,
            CacheTransferPhase::LoadBack,
            /* mamba_pool_idx = */ None,
            /* host_indices = */ None,
            /* token_ids = */ None,
            /* prefetch_tokens = */ 0,
            /* last_hash = */ None,
        )
        .unwrap()
        .unwrap();
    let host_indices = transfers[0].host_indices.as_ref().unwrap();
    assert_eq!(host_indices.numel(), 0);
    assert_eq!(host_indices.kind(), Kind::Int64);
    assert_eq!(host_indices.device(), tch::Device::Cpu);
    assert_eq!(transfers[0].nodes_to_load, Some(vec![]));
}

#[test]
#[should_panic(expected = "value: Full/host slot has no value")]
fn load_back_build_panics_on_an_evicted_unbacked_node() {
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let _ = FullComponent.build_hicache_transfers(
        &tc,
        a,
        CacheTransferPhase::LoadBack,
        /* mamba_pool_idx = */ None,
        /* host_indices = */ None,
        /* token_ids = */ None,
        /* prefetch_tokens = */ 0,
        /* last_hash = */ None,
    );
}

#[test]
fn load_back_commit_attaches_device_slices_in_chain_order() {
    let mut tc = core();
    let (_a, b, c) = load_back_chain(&mut tc);
    let mut cache_actions = Vec::new();
    let transfer = PoolTransfer {
        name: PoolName::Kv,
        host_indices: Some(Tensor::from_slice(&[21i64, 22, 23])),
        device_indices: Some(Tensor::from_slice(&[50i64, 51, 52])),
        nodes_to_load: Some(vec![tc.arena.node(b).id, tc.arena.node(c).id]),
        ..Default::default()
    };
    FullComponent.commit_hicache_transfer(
        &mut tc,
        c,
        CacheTransferPhase::LoadBack,
        vec![transfer],
        &mut cache_actions,
        /* insert_result = */ None,
        /* pool_storage_result = */ None,
    );
    assert!(
        tc.arena
            .device_value(b, FULL)
            .equal(&Tensor::from_slice(&[50i64, 51]))
    );
    assert!(
        tc.arena
            .device_value(c, FULL)
            .equal(&Tensor::from_slice(&[52i64]))
    );
    assert_eq!(tc.full_evictable_size(), 3);
    assert!(cache_actions.is_empty());
    assert!(tc.evictable_device_leaves.contains(c));
    // b entered the D-leaf set while c was still evicted and the per-node update
    // never revisits it; the orchestrator's post-commit path re-lock corrects it.
    assert!(tc.evictable_device_leaves.contains(b));
}

#[test]
fn load_back_commit_without_transfers_leaves_the_node_evicted() {
    let mut tc = core();
    let (_a, _b, c) = load_back_chain(&mut tc);
    let mut cache_actions = Vec::new();
    FullComponent.commit_hicache_transfer(
        &mut tc,
        c,
        CacheTransferPhase::LoadBack,
        vec![],
        &mut cache_actions,
        /* insert_result = */ None,
        /* pool_storage_result = */ None,
    );
    assert!(tc.arena.node(c).evicted());
    assert!(cache_actions.is_empty());
}

#[test]
fn load_back_commit_without_device_indices_leaves_the_node_evicted() {
    let mut tc = core();
    let (_a, b, c) = load_back_chain(&mut tc);
    let mut cache_actions = Vec::new();
    let transfer = PoolTransfer {
        name: PoolName::Kv,
        host_indices: Some(Tensor::from_slice(&[21i64, 22, 23])),
        nodes_to_load: Some(vec![tc.arena.node(b).id, tc.arena.node(c).id]),
        ..Default::default()
    };
    FullComponent.commit_hicache_transfer(
        &mut tc,
        c,
        CacheTransferPhase::LoadBack,
        vec![transfer],
        &mut cache_actions,
        /* insert_result = */ None,
        /* pool_storage_result = */ None,
    );
    assert!(tc.arena.node(b).evicted());
    assert!(tc.arena.node(c).evicted());
    assert_eq!(tc.full_evictable_size(), 0);
}
