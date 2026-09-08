use super::*;
use crate::components::{FULL, MAMBA, SWA};
use crate::test_utils::{accumulate_step, action_kinds};
use crate::unified_lru_list::UnifiedLRUList;

fn mamba_core(page_size: usize) -> UnifiedTreeCore<Vec<i64>> {
    mamba_core_with_chunk(page_size, /* chunk = */ 256)
}

fn mamba_core_with_cap(cap: usize) -> UnifiedTreeCore<Vec<i64>> {
    UnifiedTreeCore::new(
        CacheInitParams {
            page_size: 1,
            mamba_cache_chunk_size: Some(256),
            mamba_max_states_per_path: Some(cap),
            ..CacheInitParams::default()
        },
        vec![FULL, MAMBA],
    )
}

fn mamba_core_with_chunk(page_size: usize, chunk: usize) -> UnifiedTreeCore<Vec<i64>> {
    UnifiedTreeCore::new(
        CacheInitParams {
            page_size,
            mamba_cache_chunk_size: Some(chunk),
            ..CacheInitParams::default()
        },
        vec![FULL, MAMBA],
    )
}

fn hybrid_lock_core() -> (UnifiedTreeCore<Vec<i64>>, NodeIdx_, NodeIdx_) {
    let mut tc = UnifiedTreeCore::new(
        CacheInitParams {
            page_size: 1,
            swa_sliding_window_size: Some(2),
            mamba_cache_chunk_size: Some(256),
            ..CacheInitParams::default()
        },
        vec![FULL, SWA, MAMBA],
    );
    let [parent, leaf] = chain::<2>(&mut tc);
    for (node, full_slot, swa_slot, mamba_slot) in [(parent, 10, 20, 30), (leaf, 11, 21, 31)] {
        tc.arena
            .set_device_value(node, FULL, Tensor::from_slice(&[full_slot]));
        tc.set_component_device_value(tc.arena.node(node).id, SWA, Tensor::from_slice(&[swa_slot]));
        set_mamba_device(&mut tc, node, mamba_slot);
        tc.update_evictable_leaf_sets_(node);
    }
    tc.component_state_mut(FULL).evictable_size = 2;
    (tc, parent, leaf)
}

fn insert_params_mamba<'k>(
    key: &'k Vec<i64>,
    value: &[i64],
    mamba_slot: Option<i64>,
) -> InsertParams<'k, Vec<i64>> {
    InsertParams {
        key,
        namespace: Default::default(),
        value: Tensor::from_slice(value),
        mamba_value: mamba_slot.map(|slot| Tensor::from_slice(&[slot])),
        prev_prefix_len: 0,
        swa_evicted_seqlen: 0,
        chunked: false,
        priority: 0,
        track_adopted_ranges: false,
    }
}

fn match_params(key: &Vec<i64>) -> MatchPrefixParams<'_, Vec<i64>> {
    MatchPrefixParams {
        key,
        namespace: Default::default(),
    }
}

// A two-node arena-built path: A[1,2] and B[3,4], both with FULL device
// values; mamba data is seeded by the caller.
fn two_node_path(tc: &mut UnifiedTreeCore<Vec<i64>>) -> (NodeIdx_, NodeIdx_) {
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
    let b = tc
        .arena
        .alloc_child(
            a,
            /* key = */ vec![3, 4],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(b, FULL, Tensor::from_slice(&[12i64, 13]));
    (a, b)
}

fn mamba_component() -> MambaComponent {
    MambaComponent::new(&CacheInitParams {
        mamba_cache_chunk_size: Some(256),
        ..CacheInitParams::default()
    })
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

// Seed a one-slot mamba device value with its LRU and size bookkeeping.
fn set_mamba_device(tc: &mut UnifiedTreeCore<Vec<i64>>, node: NodeIdx_, slot: i64) {
    tc.arena
        .set_device_value(node, MAMBA, Tensor::from_slice(&[slot]));
    tc.device_lru_list_mut(MAMBA).insert_mru(node);
    tc.inc_evictable_size(MAMBA, 1);
}

fn set_mamba_host(tc: &mut UnifiedTreeCore<Vec<i64>>, node: NodeIdx_, slot: i64) {
    tc.arena
        .set_host_value(node, MAMBA, Tensor::from_slice(&[slot]));
}

fn set_full_host(tc: &mut UnifiedTreeCore<Vec<i64>>, node: NodeIdx_, slot: i64) {
    tc.arena
        .set_host_value(node, FULL, Tensor::from_slice(&[slot]));
}

fn lru_order(lru: &UnifiedLRUList) -> Vec<NodeIdx_> {
    lru.iter().collect()
}

#[test]
fn node_has_component_data_reads_each_layer() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    let mamba = mamba_component();
    assert!(!crate::components::node_has_component_data(
        &tc.arena,
        a,
        MAMBA,
        EvictLayer::Device
    ));
    assert!(!crate::components::node_has_component_data(
        &tc.arena,
        a,
        MAMBA,
        EvictLayer::Host
    ));
    set_mamba_device(&mut tc, a, 7);
    set_mamba_host(&mut tc, a, 8);
    assert!(crate::components::node_has_component_data(
        &tc.arena,
        a,
        MAMBA,
        EvictLayer::Device
    ));
    assert!(crate::components::node_has_component_data(
        &tc.arena,
        a,
        MAMBA,
        EvictLayer::Host
    ));
}

#[test]
fn refresh_lru_walkdown_is_a_no_op() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a, b] = chain::<2>(&mut tc);
    set_mamba_device(&mut tc, a, 7);
    set_mamba_device(&mut tc, b, 8);
    let mamba = mamba_component();
    // The walk never reorders mamba states; commit and match own the stamps.
    mamba.refresh_lru(&mut tc, LRURefreshPhase::Walkdown, a);
    assert_eq!(lru_order(tc.device_lru_list(MAMBA)), vec![b, a]);
}

#[test]
fn refresh_lru_match_end_touches_only_the_matched_node() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a, b, c] = chain::<3>(&mut tc);
    set_mamba_device(&mut tc, c, 9);
    set_mamba_device(&mut tc, b, 8);
    set_mamba_device(&mut tc, a, 7);
    let mamba = mamba_component();
    // Only the consumed state re-ranks; its valued ancestors stay put.
    mamba.refresh_lru(&mut tc, LRURefreshPhase::MatchEnd, c);
    assert_eq!(lru_order(tc.device_lru_list(MAMBA)), vec![c, a, b]);
}

#[test]
fn refresh_lru_insert_end_is_a_noop() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a, b] = chain::<2>(&mut tc);
    set_mamba_device(&mut tc, a, 7);
    set_mamba_device(&mut tc, b, 8);
    mamba_component().refresh_lru(&mut tc, LRURefreshPhase::InsertEnd, a);
    assert_eq!(lru_order(tc.device_lru_list(MAMBA)), vec![b, a]);
}

#[test]
fn device_value_round_trips_through_the_component() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    let mamba = mamba_component();
    assert!(tc.arena.try_device_value(a, MAMBA).is_none());
    tc.set_component_device_value(tc.arena.node(a).id, MAMBA, Tensor::from_slice(&[42i64]));
    assert!(
        tc.arena
            .try_device_value(a, MAMBA)
            .unwrap()
            .equal(&Tensor::from_slice(&[42i64]))
    );
}

#[test]
fn match_validator_accepts_device_and_optionally_host() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a, b, c] = chain::<3>(&mut tc);
    set_mamba_device(&mut tc, a, 7);
    set_mamba_host(&mut tc, b, 8);
    let mamba = mamba_component();
    let mut hicache = mamba.create_match_validator(&tc, /* match_device_only = */ false);
    assert!(hicache(&tc, a));
    assert!(hicache(&tc, b));
    assert!(!hicache(&tc, c));
    let mut device_only = mamba.create_match_validator(&tc, /* match_device_only = */ true);
    assert!(device_only(&tc, a));
    assert!(!device_only(&tc, b));
}

#[test]
fn split_keeps_mamba_data_on_the_leaf() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let root = tc.arena.root();
    let leaf = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(leaf, FULL, Tensor::from_slice(&[10i64, 11]));
    tc.update_evictable_leaf_sets_(leaf);
    set_mamba_device(&mut tc, leaf, 7);
    let (new_parent, action) = tc.split_node_(leaf, /* split_len = */ 1);
    assert!(action.is_none());
    assert!(tc.arena.node(leaf).has_device_value(MAMBA));
    assert!(!tc.arena.node(new_parent).has_device_value(MAMBA));
    assert!(!tc.arena.node(new_parent).has_host_value(MAMBA));
}

#[test]
fn device_lock_moves_the_slot_between_evictable_and_protected_once() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    set_mamba_device(&mut tc, a, 7);
    let mamba = mamba_component();
    let result = mamba.acquire_component_lock(
        &mut tc,
        a,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    assert!(result.skip_lock_node_ids.is_empty());
    assert_eq!(tc.evictable_size_(MAMBA), 0);
    assert_eq!(tc.protected_size_(MAMBA), 1);
    mamba.acquire_component_lock(
        &mut tc,
        a,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    assert_eq!(tc.protected_size_(MAMBA), 1);
    assert_eq!(tc.arena.node(a).device_lock_ref(MAMBA), 2);
    mamba.release_component_lock(&mut tc, a, None, /* lock_host = */ false);
    assert_eq!(tc.protected_size_(MAMBA), 1);
    mamba.release_component_lock(&mut tc, a, None, /* lock_host = */ false);
    assert_eq!(tc.evictable_size_(MAMBA), 1);
    assert_eq!(tc.protected_size_(MAMBA), 0);
    assert_eq!(tc.arena.node(a).device_lock_ref(MAMBA), 0);
}

#[test]
fn skip_aware_lock_records_only_the_mamba_target() {
    let (mut tc, parent, leaf) = hybrid_lock_core();
    let leaf_handle = tc.arena.node(leaf).id;

    let result = tc.inc_lock_ref_with_skip(leaf_handle, &[MAMBA]);

    assert_eq!(result.skip_lock_node_ids[&MAMBA].len(), 1);
    assert!(result.skip_lock_node_ids[&MAMBA].contains(&leaf_handle));
    assert_eq!(tc.arena.node(parent).device_lock_ref(MAMBA), 0);
    assert_eq!(tc.arena.node(leaf).device_lock_ref(MAMBA), 0);
    assert_eq!(tc.evictable_size_(MAMBA), 2);
    assert_eq!(tc.protected_size_(MAMBA), 0);
    assert_eq!(tc.arena.node(parent).device_lock_ref(FULL), 1);
    assert_eq!(tc.arena.node(leaf).device_lock_ref(FULL), 1);

    tc.dec_lock_ref(
        leaf_handle,
        Some(&DecLockRefParams {
            swa_uuid_for_lock: result.swa_uuid_for_lock,
            skip_lock_node_ids: result.skip_lock_node_ids,
            ..Default::default()
        }),
        /* skip_swa = */ false,
    );
    assert_eq!(tc.arena.node(parent).device_lock_ref(FULL), 0);
    assert_eq!(tc.arena.node(leaf).device_lock_ref(FULL), 0);
}

#[test]
fn swa_only_release_honors_a_skipped_mamba_target() {
    let (mut tc, _parent, leaf) = hybrid_lock_core();
    let leaf_handle = tc.arena.node(leaf).id;
    let owner = tc.inc_lock_ref(leaf_handle);
    let skipped = tc.inc_lock_ref_with_skip(leaf_handle, &[MAMBA]);
    assert_eq!(tc.arena.node(leaf).device_lock_ref(MAMBA), 1);

    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    tc.dec_swa_lock_only_with_skip(
        leaf_handle,
        skipped.swa_uuid_for_lock,
        Some(&skipped.skip_lock_node_ids),
        &mut device_frees,
        &mut host_frees,
    );

    assert!(device_frees.is_empty());
    assert!(host_frees.is_empty());
    assert_eq!(tc.arena.node(leaf).device_lock_ref(MAMBA), 1);
    assert_eq!(tc.protected_size_(MAMBA), 1);

    let skipped_params = DecLockRefParams {
        swa_uuid_for_lock: skipped.swa_uuid_for_lock,
        skip_lock_node_ids: skipped.skip_lock_node_ids,
        ..Default::default()
    };
    tc.dec_lock_ref(
        leaf_handle,
        Some(&skipped_params),
        /* skip_swa = */ true,
    );
    let owner_params = DecLockRefParams {
        swa_uuid_for_lock: owner.swa_uuid_for_lock,
        skip_lock_node_ids: owner.skip_lock_node_ids,
        ..Default::default()
    };
    tc.dec_lock_ref(
        leaf_handle,
        Some(&owner_params),
        /* skip_swa = */ false,
    );
    assert_eq!(tc.protected_size_(MAMBA), 0);
}

#[test]
fn tombstone_lock_is_recorded_and_replayed_at_release() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    let mamba = mamba_component();
    let result = mamba.acquire_component_lock(
        &mut tc,
        a,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    assert!(result.skip_lock_node_ids[&MAMBA].contains(&tc.arena.node(a).id));
    assert_eq!(tc.arena.node(a).device_lock_ref(MAMBA), 0);
    // The replayed skip set keeps the release from touching the node.
    let params = DecLockRefParams {
        skip_lock_node_ids: result.skip_lock_node_ids.clone(),
        ..DecLockRefParams::default()
    };
    mamba.release_component_lock(&mut tc, a, Some(&params), /* lock_host = */ false);
    assert_eq!(tc.arena.node(a).device_lock_ref(MAMBA), 0);
    assert_eq!(tc.evictable_size_(MAMBA), 0);
}

#[test]
fn root_locks_are_noops() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let root = tc.arena.root();
    let mamba = mamba_component();
    let result = mamba.acquire_component_lock(
        &mut tc,
        root,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    assert!(result.skip_lock_node_ids.is_empty());
    mamba.release_component_lock(&mut tc, root, None, /* lock_host = */ false);
    assert_eq!(tc.evictable_size_(MAMBA), 0);
}

#[test]
fn host_lock_detaches_and_reattaches_the_host_lru() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    set_mamba_host(&mut tc, a, 8);
    tc.host_lru_list_mut(MAMBA).insert_mru(a);
    let mamba = mamba_component();
    mamba.acquire_component_lock(
        &mut tc,
        a,
        IncLockRefResult::default(),
        /* lock_host = */ true,
    );
    assert!(!tc.host_lru_list(MAMBA).in_list(Some(a)));
    assert_eq!(tc.arena.node(a).host_lock_ref(MAMBA), 1);
    mamba.release_component_lock(&mut tc, a, None, /* lock_host = */ true);
    assert!(tc.host_lru_list(MAMBA).in_list(Some(a)));
    assert_eq!(tc.arena.node(a).host_lock_ref(MAMBA), 0);
}

#[test]
fn host_unlock_skips_the_lru_for_device_backed_nodes() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    set_mamba_host(&mut tc, a, 8);
    set_mamba_device(&mut tc, a, 7);
    let mamba = mamba_component();
    mamba.acquire_component_lock(
        &mut tc,
        a,
        IncLockRefResult::default(),
        /* lock_host = */ true,
    );
    mamba.release_component_lock(&mut tc, a, None, /* lock_host = */ true);
    assert!(!tc.host_lru_list(MAMBA).in_list(Some(a)));
}

#[test]
fn eviction_priority_is_the_lowest_tier_everywhere() {
    let mamba = mamba_component();
    assert_eq!(
        TreeComponent::<Vec<i64>>::eviction_priority(&mamba, /* is_leaf = */ true),
        0
    );
    assert_eq!(
        TreeComponent::<Vec<i64>>::eviction_priority(&mamba, /* is_leaf = */ false),
        0
    );
}

#[test]
fn insert_attaches_the_donated_slot_to_the_new_leaf() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let result = tc.insert(&insert_params_mamba(&vec![1, 2], &[10, 11], Some(7)));
    assert!(!result.mamba_exist);
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    assert!(
        tc.arena
            .node(tc.arena.resolve(leaf))
            .try_device_value(MAMBA)
            .unwrap()
            .equal(&Tensor::from_slice(&[7i64]))
    );
    assert!(
        tc.device_lru_list(MAMBA)
            .in_list(Some(tc.arena.resolve(leaf)))
    );
    assert_eq!(tc.evictable_size_(MAMBA), 1);
}

#[test]
fn reinsert_keeps_the_existing_slot_and_flags_the_caller() {
    let mut tc = mamba_core(/* page_size = */ 1);
    tc.insert(&insert_params_mamba(&vec![1, 2], &[10, 11], Some(7)));
    let result = tc.insert(&insert_params_mamba(&vec![1, 2], &[10, 11], Some(8)));
    assert!(result.mamba_exist);
    assert_eq!(tc.evictable_size_(MAMBA), 1);
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    // The original slot stays; the caller frees the unused donated one.
    assert!(
        tc.arena
            .node(tc.arena.resolve(leaf))
            .try_device_value(MAMBA)
            .unwrap()
            .equal(&Tensor::from_slice(&[7i64]))
    );
}

#[test]
fn reinsert_full_backed_target_schedules_mamba_only_backup() {
    let mut tc = mamba_core(/* page_size = */ 1);
    tc.set_hicache_enabled();
    let key = vec![1, 2];
    tc.insert(&insert_params_mamba(&key, &[10, 11], Some(7)));
    let leaf = tc.match_prefix(&match_params(&key)).best_match_node_id;
    tc.commit_backup(leaf, Tensor::from_slice(&[100i64, 101]), HashMap::new());

    let result = tc.insert(&insert_params_mamba(&key, &[20, 21], Some(8)));
    let backups = result
        .cache_actions
        .iter()
        .filter_map(|action| match action {
            CacheAction::BackupKV(backup) => Some(backup),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(backups.len(), 1);
    assert_eq!(backups[0].node_ids, vec![leaf]);

    let (full_device_indices, comp_xfers) = tc.build_backup_spec(leaf);
    assert_eq!(full_device_indices.numel(), 0);
    let mamba_xfers = &comp_xfers[&MAMBA];
    assert_eq!(mamba_xfers.len(), 1);
    assert!(
        mamba_xfers[0]
            .device_indices
            .as_ref()
            .unwrap()
            .equal(&Tensor::from_slice(&[7i64]))
    );

    tc.mark_write_through_pending(vec![leaf], /* ack_id = */ leaf);
    let pending = tc.insert(&insert_params_mamba(&key, &[30, 31], Some(9)));
    assert!(
        !pending
            .cache_actions
            .iter()
            .any(|action| matches!(action, CacheAction::BackupKV(_)))
    );
}

#[test]
fn tombstone_refill_moves_the_node_from_host_to_device_lru() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    set_mamba_host(&mut tc, a, 8);
    tc.host_lru_list_mut(MAMBA).insert_mru(a);
    let mut result = InsertResult::default();
    mamba_component().commit_insert_component_data(
        &mut tc,
        a,
        /* is_new_leaf = */ false,
        &insert_params_mamba(&vec![1], &[10], Some(9)),
        &mut result,
        &mut Vec::new(),
    );
    assert!(!result.mamba_exist);
    assert!(
        tc.arena
            .node(a)
            .try_device_value(MAMBA)
            .unwrap()
            .equal(&Tensor::from_slice(&[9i64]))
    );
    assert!(!tc.host_lru_list(MAMBA).in_list(Some(a)));
    assert!(tc.device_lru_list(MAMBA).in_list(Some(a)));
    assert_eq!(tc.evictable_size_(MAMBA), 1);
}

#[test]
#[should_panic(expected = "requires a donated mamba_value")]
fn insert_without_a_mamba_value_panics() {
    let mut tc = mamba_core(/* page_size = */ 1);
    tc.insert(&insert_params_mamba(&vec![1], &[10], None));
}

#[test]
fn match_reports_the_chunk_aligned_branching_seqlen() {
    let mut tc = mamba_core_with_chunk(/* page_size = */ 1, /* chunk = */ 3);
    let (a, _b) = two_node_path(&mut tc);
    set_mamba_device(&mut tc, a, 7);
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3, 4]));
    // The walk covers 4 tokens past the mamba anchor; 4 aligns down to 3.
    assert_eq!(result.best_match_node_id, tc.arena.node(a).id);
    assert_eq!(result.mamba_branching_seqlen, Some(3));
    assert_eq!(result.mamba_host_hit_length, 0);
}

#[test]
fn branching_seqlen_uses_the_joint_chunk_and_tree_page_grid() {
    let mut tc = mamba_core_with_chunk(/* page_size = */ 3, /* chunk = */ 2);
    let root = tc.arena.root();
    let node = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena.set_device_value(
        node,
        FULL,
        Tensor::from_slice(&[10i64, 11, 12, 13, 14, 15, 16, 17, 18]),
    );

    let result = tc.match_prefix(&match_params(&vec![1, 2, 3, 4, 5, 6, 7, 8, 9]));

    assert_eq!(result.full_kv_hit_length, 9);
    // lcm(chunk=2, page=3) is 6; chunk-only alignment would incorrectly yield 8.
    assert_eq!(result.mamba_branching_seqlen, Some(6));
}

#[test]
fn short_walks_have_no_branching_seqlen() {
    let mut tc = mamba_core_with_chunk(/* page_size = */ 1, /* chunk = */ 8);
    let (a, _b) = two_node_path(&mut tc);
    set_mamba_device(&mut tc, a, 7);
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3, 4]));
    // 4 walked tokens align down to zero at chunk 8.
    assert_eq!(result.mamba_branching_seqlen, None);
}

#[test]
fn matches_ending_on_the_mamba_anchor_have_no_branching_seqlen() {
    let mut tc = mamba_core_with_chunk(/* page_size = */ 1, /* chunk = */ 2);
    let (a, b) = two_node_path(&mut tc);
    set_mamba_device(&mut tc, a, 7);
    set_mamba_device(&mut tc, b, 9);
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3, 4]));
    assert_eq!(result.best_match_node_id, tc.arena.node(b).id);
    assert_eq!(result.mamba_branching_seqlen, None);
}

#[test]
fn hicache_branching_seqlen_uses_the_full_kv_hit() {
    let mut tc = mamba_core_with_chunk(/* page_size = */ 1, /* chunk = */ 3);
    tc.set_hicache_enabled();
    let (a, _b) = two_node_path(&mut tc);
    set_mamba_device(&mut tc, a, 7);
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3, 4]));
    // The full walk hit 4 tokens; chunk-3 alignment lands past the 2-token
    // mamba boundary, so the branch point fills even under HiCache.
    assert_eq!(result.full_kv_hit_length, 4);
    assert_eq!(result.mamba_branching_seqlen, Some(3));
}

#[test]
fn host_only_anchor_bumps_the_mamba_host_hit() {
    let mut tc = mamba_core(/* page_size = */ 1);
    tc.set_hicache_enabled();
    let (a, _b) = two_node_path(&mut tc);
    set_mamba_host(&mut tc, a, 8);
    let result = tc.match_prefix(&match_params(&vec![1, 2]));
    assert_eq!(result.best_match_node_id, tc.arena.node(a).id);
    assert_eq!(result.mamba_host_hit_length, 1);
}

#[test]
fn evict_component_device_frees_and_tombstones_the_slot() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    set_mamba_device(&mut tc, a, 7);
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    let (freed, host_freed) = mamba_component().evict_component(
        &mut tc,
        a,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Device,
    );
    assert_eq!((freed, host_freed), (1, 0));
    assert!(!tc.arena.node(a).has_device_value(MAMBA));
    assert_eq!(tc.evictable_size_(MAMBA), 0);
    assert!(device_frees[&MAMBA][0].equal(&Tensor::from_slice(&[7i64])));
    assert!(host_frees.is_empty());
    // No host backup: the node does not enter the host LRU.
    assert!(!tc.host_lru_list(MAMBA).in_list(Some(a)));
}

#[test]
fn evict_component_device_moves_a_host_backed_node_into_the_host_lru() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    set_mamba_device(&mut tc, a, 7);
    set_mamba_host(&mut tc, a, 8);
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    mamba_component().evict_component(
        &mut tc,
        a,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Device,
    );
    assert!(tc.arena.node(a).has_host_value(MAMBA));
    assert!(tc.host_lru_list(MAMBA).in_list(Some(a)));
}

#[test]
fn evict_component_host_frees_and_leaves_the_host_lru() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    set_mamba_host(&mut tc, a, 8);
    tc.host_lru_list_mut(MAMBA).insert_mru(a);
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    let (freed, host_freed) = mamba_component().evict_component(
        &mut tc,
        a,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Host,
    );
    assert_eq!((freed, host_freed), (0, 1));
    assert!(!tc.arena.node(a).has_host_value(MAMBA));
    assert!(!tc.host_lru_list(MAMBA).in_list(Some(a)));
    assert!(host_frees[&MAMBA][0].equal(&Tensor::from_slice(&[8i64])));
}

#[test]
fn evict_component_all_frees_both_tiers_without_the_host_lru() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    set_mamba_device(&mut tc, a, 7);
    set_mamba_host(&mut tc, a, 8);
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    let (freed, host_freed) = mamba_component().evict_component(
        &mut tc,
        a,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::All,
    );
    assert_eq!((freed, host_freed), (1, 1));
    assert!(!tc.arena.node(a).has_device_value(MAMBA));
    assert!(!tc.arena.node(a).has_host_value(MAMBA));
    // ALL means the node dies: it must not re-enter the host LRU.
    assert!(!tc.host_lru_list(MAMBA).in_list(Some(a)));
}

#[test]
fn device_walk_advances_one_allocator_mutation_per_call() {
    let mut tc = mamba_core(/* page_size = */ 1);
    tc.insert(&insert_params_mamba(&vec![1], &[10], Some(7)));
    tc.insert(&insert_params_mamba(&vec![1, 2], &[10, 11], Some(8)));
    let a = tc.match_prefix(&match_params(&vec![1])).best_match_node_id;
    let b = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    let mut tracker = HashMap::from([(MAMBA, 0)]);
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    tc.evict_device_start(MAMBA, /* request_cnt = */ 2);
    let (first, step) = tc.evict_device_next_node(MAMBA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    // The internal node is a complete step so its free can be reused before
    // the walk hands out another victim.
    assert_eq!(first, None);
    assert!(!tc.arena.node(tc.arena.resolve(a)).has_device_value(MAMBA));
    assert!(tc.arena.node(tc.arena.resolve(b)).has_device_value(MAMBA));
    assert!(tc.arena.has_device_value(tc.arena.resolve(a), FULL));
    assert_eq!(tracker[&MAMBA], 1);
    assert!(!tc.device_lru_list(MAMBA).in_list(Some(tc.arena.resolve(a))));

    let (second, step) = tc.evict_device_next_node(MAMBA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    assert_eq!(second, Some(b));
    assert_eq!(tracker[&MAMBA], 1);
    tc.evict_device_end(MAMBA);
}

#[test]
fn device_walk_skips_locked_nodes() {
    let mut tc = mamba_core(/* page_size = */ 1);
    tc.insert(&insert_params_mamba(&vec![1], &[10], Some(7)));
    tc.insert(&insert_params_mamba(&vec![1, 2], &[10, 11], Some(8)));
    let a = tc.match_prefix(&match_params(&vec![1])).best_match_node_id;
    let b = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    let a_idx = tc.arena.resolve(a);
    mamba_component().acquire_component_lock(
        &mut tc,
        a_idx,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    let mut tracker = HashMap::from([(MAMBA, 0)]);
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    tc.evict_device_start(MAMBA, /* request_cnt = */ 2);
    let (next, step) = tc.evict_device_next_node(MAMBA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    // The locked internal node stays; the cursor starts on the leaf.
    assert_eq!(next, Some(b));
    assert!(tc.arena.node(tc.arena.resolve(a)).has_device_value(MAMBA));
    assert_eq!(tracker[&MAMBA], 0);
    tc.evict_device_end(MAMBA);
}

#[test]
#[should_panic(expected = "Mamba device eviction not started")]
fn device_walk_requires_a_start() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let tracker = HashMap::from([(MAMBA, 0)]);
    tc.evict_device_next_node(MAMBA, &tracker);
}

#[test]
#[should_panic(expected = "Mamba eviction cursor on a valueless node")]
fn device_walk_asserts_a_valued_cursor_node() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    set_mamba_device(&mut tc, a, 7);
    // a stays in the Mamba LRU but loses its device value out of band.
    let _ = tc.arena.node_mut(a).take_device_value(MAMBA);
    let tracker = HashMap::from([(MAMBA, 0)]);
    tc.evict_device_start(MAMBA, /* request_cnt = */ 100);
    tc.evict_device_next_node(MAMBA, &tracker);
}

#[test]
fn host_eviction_tombstones_internal_host_values() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a, _b] = chain::<2>(&mut tc);
    set_mamba_host(&mut tc, a, 8);
    tc.host_lru_list_mut(MAMBA).insert_mru(a);
    let mut tracker = HashMap::from([(MAMBA, 0)]);
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    mamba_component().drive_host_eviction(
        &mut tc,
        /* num_tokens = */ 1,
        &mut tracker,
        &mut device_frees,
        &mut host_frees,
    );
    assert!(!tc.arena.node(a).has_host_value(MAMBA));
    assert!(!tc.host_lru_list(MAMBA).in_list(Some(a)));
    assert_eq!(tracker[&MAMBA], 1);
    assert!(host_frees[&MAMBA][0].equal(&Tensor::from_slice(&[8i64])));
}

#[test]
fn host_eviction_takes_a_host_leaf_atomically() {
    let mut tc = mamba_core(/* page_size = */ 1);
    tc.set_hicache_enabled();
    let root = tc.arena.root();
    let leaf = tc
        .insert_host(
            tc.arena.node(root).id,
            /* extra_key = */ None,
            vec![1],
            Tensor::from_slice(&[100i64]),
            vec!["h0".to_string()],
        )
        .inserted_host_node
        .unwrap();
    let leaf_idx = tc.arena.resolve(leaf);
    set_mamba_host(&mut tc, leaf_idx, 8);
    tc.host_lru_list_mut(MAMBA).insert_mru(leaf_idx);
    assert!(tc.evictable_host_leaves.contains(tc.arena.resolve(leaf)));
    let mut tracker = HashMap::from([(MAMBA, 0)]);
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    mamba_component().drive_host_eviction(
        &mut tc,
        /* num_tokens = */ 1,
        &mut tracker,
        &mut device_frees,
        &mut host_frees,
    );
    // The atomic host-leaf eviction frees both components and the node.
    assert_eq!(tracker[&MAMBA], 1);
    assert!(host_frees[&MAMBA][0].equal(&Tensor::from_slice(&[8i64])));
    assert!(host_frees[&FULL][0].equal(&Tensor::from_slice(&[100i64])));
    assert!(tc.arena.try_resolve(leaf).is_none());
    assert!(!tc.host_lru_list(MAMBA).in_list(Some(leaf_idx)));
}

#[test]
#[should_panic(expected = "has no host value")]
fn host_drive_panics_on_an_lru_member_without_a_mamba_host_value() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [n] = chain::<1>(&mut tc);
    tc.arena
        .set_host_value(n, FULL, Tensor::from_slice(&[100i64]));
    tc.host_lru_list_mut(MAMBA).insert_mru(n);
    let mut tracker = HashMap::from([(MAMBA, 0)]);
    mamba_component().drive_host_eviction(
        &mut tc,
        /* num_tokens = */ 100,
        &mut tracker,
        &mut HashMap::new(),
        &mut HashMap::new(),
    );
}

#[test]
fn swa_triggered_cascade_takes_the_mamba_slot() {
    let mut tc = UnifiedTreeCore::<Vec<i64>>::new(
        CacheInitParams {
            page_size: 1,
            swa_sliding_window_size: Some(4),
            mamba_cache_chunk_size: Some(256),
            ..CacheInitParams::default()
        },
        vec![FULL, SWA, MAMBA],
    );
    let [a] = chain::<1>(&mut tc);
    set_mamba_device(&mut tc, a, 7);
    let mut tracker = HashMap::from([(SWA, 0)]);
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    // The SWA internal tier (1) outranks mamba (0): the cascade takes it.
    tc.cascade_evict_(
        a,
        SWA,
        &mut tracker,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Device,
    );
    assert!(!tc.arena.node(a).has_device_value(MAMBA));
    assert_eq!(tracker[&MAMBA], 1);
}

#[test]
fn mamba_triggered_cascade_spares_the_higher_tiers() {
    let mut tc = UnifiedTreeCore::<Vec<i64>>::new(
        CacheInitParams {
            page_size: 1,
            swa_sliding_window_size: Some(4),
            mamba_cache_chunk_size: Some(256),
            ..CacheInitParams::default()
        },
        vec![FULL, SWA, MAMBA],
    );
    let [a] = chain::<1>(&mut tc);
    tc.arena
        .set_device_value(a, FULL, Tensor::from_slice(&[10i64]));
    tc.arena
        .set_device_value(a, SWA, Tensor::from_slice(&[20i64]));
    let mut tracker = HashMap::from([(MAMBA, 0)]);
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    tc.cascade_evict_(
        a,
        MAMBA,
        &mut tracker,
        &mut device_frees,
        &mut host_frees,
        EvictLayer::Device,
    );
    // FULL (2) and SWA (1) both outrank the mamba trigger (0).
    assert!(tc.arena.has_device_value(a, FULL));
    assert!(tc.arena.has_device_value(a, SWA));
    assert!(device_frees.is_empty());
}

#[test]
fn backup_host_build_carries_the_device_slot() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    set_mamba_device(&mut tc, a, 7);
    let transfers = tc
        .build_hicache_transfers(
            MAMBA,
            tc.arena.node(a).id,
            CacheTransferPhase::BackupHost,
            None,
            None,
            0,
            None,
        )
        .unwrap();
    assert_eq!(transfers.len(), 1);
    assert_eq!(transfers[0].name, PoolName::Mamba);
    assert!(
        transfers[0]
            .device_indices
            .as_ref()
            .unwrap()
            .equal(&Tensor::from_slice(&[7i64]))
    );
    // A tombstone has nothing to back up.
    let [b] = [tc
        .arena
        .alloc_child(
            a,
            /* key = */ vec![9],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap()];
    assert!(
        tc.build_hicache_transfers(
            MAMBA,
            tc.arena.node(b).id,
            CacheTransferPhase::BackupHost,
            None,
            None,
            0,
            None,
        )
        .is_none()
    );
}

#[test]
fn load_back_build_restores_the_host_only_node() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    set_mamba_host(&mut tc, a, 8);
    let transfers = tc
        .build_hicache_transfers(
            MAMBA,
            tc.arena.node(a).id,
            CacheTransferPhase::LoadBack,
            None,
            None,
            0,
            None,
        )
        .unwrap();
    assert_eq!(transfers.len(), 1);
    assert!(
        transfers[0]
            .host_indices
            .as_ref()
            .unwrap()
            .equal(&Tensor::from_slice(&[8i64]))
    );
    assert_eq!(transfers[0].nodes_to_load, Some(vec![tc.arena.node(a).id]));
}

#[test]
fn load_back_build_skips_device_backed_and_bare_nodes() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a, b] = chain::<2>(&mut tc);
    set_mamba_device(&mut tc, a, 7);
    for node in [a, b] {
        assert!(
            tc.build_hicache_transfers(
                MAMBA,
                tc.arena.node(node).id,
                CacheTransferPhase::LoadBack,
                None,
                None,
                0,
                None,
            )
            .is_none()
        );
    }
}

#[test]
fn load_back_build_adds_the_per_request_cow_transfer() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    set_mamba_host(&mut tc, a, 8);
    let transfers = mamba_component()
        .build_hicache_transfers(
            &tc,
            a,
            CacheTransferPhase::LoadBack,
            /* mamba_pool_idx = */ Some(Tensor::from_slice(&[3i64]).squeeze()),
            None,
            None,
            0,
            None,
        )
        .unwrap()
        .unwrap();
    assert_eq!(transfers.len(), 2);
    // The CoW transfer copies the host slot into the request's device slot.
    assert!(
        transfers[1]
            .device_indices
            .as_ref()
            .unwrap()
            .equal(&Tensor::from_slice(&[3i64]))
    );
    assert!(transfers[1].nodes_to_load.is_none());
}

#[test]
fn backup_host_commit_stores_the_host_slot_once() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    let mut cache_actions = Vec::new();
    tc.commit_hicache_transfers(
        tc.arena.node(a).id,
        CacheTransferPhase::BackupHost,
        HashMap::from([(
            MAMBA,
            vec![PoolTransfer {
                name: PoolName::Mamba,
                host_indices: Some(Tensor::from_slice(&[30i64])),
                ..Default::default()
            }],
        )]),
        &mut cache_actions,
        None,
        None,
    );
    assert!(
        tc.arena
            .node(a)
            .try_host_value(MAMBA)
            .unwrap()
            .equal(&Tensor::from_slice(&[30i64]))
    );
    // A second backup keeps the existing host slot.
    tc.commit_hicache_transfers(
        tc.arena.node(a).id,
        CacheTransferPhase::BackupHost,
        HashMap::from([(
            MAMBA,
            vec![PoolTransfer {
                name: PoolName::Mamba,
                host_indices: Some(Tensor::from_slice(&[31i64])),
                ..Default::default()
            }],
        )]),
        &mut cache_actions,
        None,
        None,
    );
    assert!(
        tc.arena
            .node(a)
            .try_host_value(MAMBA)
            .unwrap()
            .equal(&Tensor::from_slice(&[30i64]))
    );
    assert!(cache_actions.is_empty());
}

#[test]
fn load_back_commit_moves_the_node_onto_the_device_tier() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    set_mamba_host(&mut tc, a, 8);
    tc.host_lru_list_mut(MAMBA).insert_mru(a);
    let mut cache_actions = Vec::new();
    tc.commit_hicache_transfers(
        tc.arena.node(a).id,
        CacheTransferPhase::LoadBack,
        HashMap::from([(
            MAMBA,
            vec![PoolTransfer {
                name: PoolName::Mamba,
                host_indices: Some(Tensor::from_slice(&[8i64])),
                device_indices: Some(Tensor::from_slice(&[40i64])),
                ..Default::default()
            }],
        )]),
        &mut cache_actions,
        None,
        None,
    );
    assert!(
        tc.arena
            .node(a)
            .try_device_value(MAMBA)
            .unwrap()
            .equal(&Tensor::from_slice(&[40i64]))
    );
    assert!(!tc.host_lru_list(MAMBA).in_list(Some(a)));
    assert!(tc.device_lru_list(MAMBA).in_list(Some(a)));
    assert_eq!(tc.evictable_size_(MAMBA), 1);
    assert!(cache_actions.is_empty());
}

#[test]
fn mamba_device_eviction_skips_a_load_back_pinned_node() {
    let mut tc = mamba_core(/* page_size = */ 1);
    tc.is_write_back = true;
    let [n] = chain::<1>(&mut tc);
    set_full_host(&mut tc, n, 10);
    set_mamba_host(&mut tc, n, 20);
    let (kv_xfer, mut comp_xfers) =
        tc.build_load_back_spec(tc.arena.node(n).id, /* req = */ None);
    comp_xfers.get_mut(&MAMBA).unwrap()[0].device_indices = Some(Tensor::from_slice(&[40i64]));
    tc.commit_load_back(
        tc.arena.node(n).id,
        Tensor::from_slice(&[30i64]),
        kv_xfer,
        comp_xfers,
    );

    tc.evict_device_start(MAMBA, /* request_cnt = */ 1);
    let (next, _) = tc.evict_device_next_node(MAMBA, &HashMap::new());
    assert_eq!(next, None);
    tc.evict_device_end(MAMBA);
    assert!(tc.arena.has_device_value(n, MAMBA));

    tc.finish_load_back(tc.arena.node(n).id);
    tc.evict_device_start(MAMBA, /* request_cnt = */ 1);
    let (next, _) = tc.evict_device_next_node(MAMBA, &HashMap::new());
    assert_eq!(next, Some(tc.arena.node(n).id));
    tc.evict_device_end(MAMBA);
}

#[test]
fn mamba_host_eviction_skips_a_load_back_pinned_node() {
    let mut tc = mamba_core(/* page_size = */ 1);
    tc.is_write_back = true;
    let [a, b] = chain::<2>(&mut tc);
    set_full_host(&mut tc, a, 10);
    set_full_host(&mut tc, b, 11);
    set_mamba_host(&mut tc, a, 20);
    tc.host_lru_list_mut(MAMBA).insert_mru(a);
    let (kv_xfer, comp_xfers) = tc.build_load_back_spec(tc.arena.node(b).id, /* req = */ None);
    assert!(comp_xfers.is_empty());
    tc.commit_load_back(
        tc.arena.node(b).id,
        Tensor::from_slice(&[30i64, 31]),
        kv_xfer,
        comp_xfers,
    );

    let result = tc.drive_host_eviction(MAMBA, /* num_tokens = */ 1);
    assert_eq!(result.tracker[&MAMBA], 0);
    assert!(result.host_frees.is_empty());
    assert!(tc.arena.has_host_value(a, MAMBA));

    tc.finish_load_back(tc.arena.node(b).id);
    let result = tc.drive_host_eviction(MAMBA, /* num_tokens = */ 1);
    assert_eq!(result.tracker[&MAMBA], 1);
    assert_eq!(result.host_frees[&MAMBA].len(), 1);
    assert!(!tc.arena.has_host_value(a, MAMBA));
    tc.sanity_check(&[], &[]);
}

#[test]
fn backup_storage_commit_is_a_noop() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    set_mamba_host(&mut tc, a, 8);
    let mut cache_actions = Vec::new();
    tc.commit_hicache_transfers(
        tc.arena.node(a).id,
        CacheTransferPhase::BackupStorage,
        HashMap::from([(
            MAMBA,
            vec![PoolTransfer {
                name: PoolName::Mamba,
                host_indices: Some(Tensor::from_slice(&[8i64])),
                ..Default::default()
            }],
        )]),
        &mut cache_actions,
        None,
        None,
    );
    assert!(tc.arena.node(a).has_host_value(MAMBA));
    assert!(cache_actions.is_empty());
}

#[test]
fn backup_storage_build_keys_the_trailing_hash() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    // No host value yet: nothing to publish.
    assert!(
        tc.build_hicache_transfers(
            MAMBA,
            tc.arena.node(a).id,
            CacheTransferPhase::BackupStorage,
            None,
            None,
            0,
            None,
        )
        .is_none()
    );
    set_mamba_host(&mut tc, a, 8);
    // Host value but no hash chain: still nothing.
    assert!(
        tc.build_hicache_transfers(
            MAMBA,
            tc.arena.node(a).id,
            CacheTransferPhase::BackupStorage,
            None,
            None,
            0,
            None,
        )
        .is_none()
    );
    tc.arena.node_mut(a).hash_value = Some(vec!["h0".to_string(), "h1".to_string()]);
    let transfers = tc
        .build_hicache_transfers(
            MAMBA,
            tc.arena.node(a).id,
            CacheTransferPhase::BackupStorage,
            None,
            None,
            0,
            None,
        )
        .unwrap();
    assert_eq!(transfers.len(), 1);
    assert_eq!(transfers[0].keys, Some(vec!["h1".to_string()]));
    assert_eq!(transfers[0].hit_policy, PoolHitPolicy::TrailingPages);
    assert!(
        transfers[0]
            .host_indices
            .as_ref()
            .unwrap()
            .equal(&Tensor::from_slice(&[8i64]))
    );
}

#[test]
fn prefetch_build_wraps_the_host_buffer_with_a_placeholder_key() {
    let tc = mamba_core(/* page_size = */ 1);
    let transfers = tc
        .build_hicache_transfers(
            MAMBA,
            tc.arena.node(tc.arena.root()).id,
            CacheTransferPhase::Prefetch,
            Some(Tensor::from_slice(&[30i64])),
            None,
            0,
            None,
        )
        .unwrap();
    assert_eq!(transfers.len(), 1);
    assert_eq!(transfers[0].keys, Some(vec!["__placeholder__".to_string()]));
    assert_eq!(transfers[0].hit_policy, PoolHitPolicy::TrailingPages);
}

#[test]
fn prefetch_commit_attaches_the_loaded_slot_to_the_inserted_node() {
    let mut tc = mamba_core(/* page_size = */ 1);
    tc.set_hicache_enabled();
    let root = tc.arena.root();
    let target = tc
        .insert_host(
            tc.arena.node(root).id,
            /* extra_key = */ None,
            vec![1],
            Tensor::from_slice(&[100i64]),
            vec!["h0".to_string()],
        )
        .inserted_host_node
        .unwrap();
    let mut insert_result = InsertResult {
        total_len: 1,
        inserted_host_node: Some(target),
        ..InsertResult::default()
    };
    let mut cache_actions = Vec::new();
    tc.commit_hicache_transfers(
        tc.arena.node(root).id,
        CacheTransferPhase::Prefetch,
        HashMap::from([(
            MAMBA,
            vec![PoolTransfer {
                name: PoolName::Mamba,
                host_indices: Some(Tensor::from_slice(&[50i64])),
                ..Default::default()
            }],
        )]),
        &mut cache_actions,
        Some(&mut insert_result),
        Some(&PoolTransferResult {
            kv_hit_pages: 1,
            extra_pool_hit_pages: HashMap::from([(PoolName::Mamba, 1)]),
        }),
    );
    assert!(
        tc.arena
            .node(tc.arena.resolve(target))
            .try_host_value(MAMBA)
            .unwrap()
            .equal(&Tensor::from_slice(&[50i64]))
    );
    assert!(
        tc.host_lru_list(MAMBA)
            .in_list(Some(tc.arena.resolve(target)))
    );
    assert!(!insert_result.mamba_exist);
    assert!(cache_actions.is_empty());
}

#[test]
fn prefetch_commit_frees_the_buffer_when_it_cannot_attach() {
    let mut tc = mamba_core(/* page_size = */ 1);
    tc.set_hicache_enabled();
    let root = tc.arena.root();
    let target = tc
        .insert_host(
            tc.arena.node(root).id,
            /* extra_key = */ None,
            vec![1],
            Tensor::from_slice(&[100i64]),
            vec!["h0".to_string()],
        )
        .inserted_host_node
        .unwrap();
    // Not loaded: the buffer frees and the caller keeps its slot flag.
    let mut insert_result = InsertResult {
        total_len: 1,
        inserted_host_node: Some(target),
        ..InsertResult::default()
    };
    let mut cache_actions = Vec::new();
    tc.commit_hicache_transfers(
        tc.arena.node(root).id,
        CacheTransferPhase::Prefetch,
        HashMap::from([(
            MAMBA,
            vec![PoolTransfer {
                name: PoolName::Mamba,
                host_indices: Some(Tensor::from_slice(&[50i64])),
                ..Default::default()
            }],
        )]),
        &mut cache_actions,
        Some(&mut insert_result),
        Some(&PoolTransferResult {
            kv_hit_pages: 1,
            extra_pool_hit_pages: HashMap::new(),
        }),
    );
    assert!(
        !tc.arena
            .node(tc.arena.resolve(target))
            .has_host_value(MAMBA)
    );
    assert!(insert_result.mamba_exist);
    let CacheAction::FreeComponentHostSlot {
        component_type,
        host_indices,
    } = &cache_actions[0]
    else {
        panic!("expected a FreeComponentHostSlot action");
    };
    assert_eq!(*component_type, MAMBA);
    assert!(host_indices[0].equal(&Tensor::from_slice(&[50i64])));

    // An already-hosted target frees the buffer too.
    let target_idx = tc.arena.resolve(target);
    set_mamba_host(&mut tc, target_idx, 8);
    let mut insert_result = InsertResult {
        total_len: 1,
        inserted_host_node: Some(target),
        ..InsertResult::default()
    };
    let mut cache_actions = Vec::new();
    tc.commit_hicache_transfers(
        tc.arena.node(root).id,
        CacheTransferPhase::Prefetch,
        HashMap::from([(
            MAMBA,
            vec![PoolTransfer {
                name: PoolName::Mamba,
                host_indices: Some(Tensor::from_slice(&[51i64])),
                ..Default::default()
            }],
        )]),
        &mut cache_actions,
        Some(&mut insert_result),
        Some(&PoolTransferResult {
            kv_hit_pages: 1,
            extra_pool_hit_pages: HashMap::from([(PoolName::Mamba, 1)]),
        }),
    );
    assert!(insert_result.mamba_exist);
    assert_eq!(cache_actions.len(), 1);
    assert!(
        tc.arena
            .node(tc.arena.resolve(target))
            .try_host_value(MAMBA)
            .unwrap()
            .equal(&Tensor::from_slice(&[8i64]))
    );
}

#[test]
fn new_combines_the_chunk_and_tree_page_grids() {
    let component = MambaComponent::new(&CacheInitParams {
        page_size: 6,
        mamba_cache_chunk_size: Some(4),
        ..CacheInitParams::default()
    });
    assert_eq!(component.mamba_checkpoint_grid, 12);
}

#[test]
#[should_panic(expected = "requires mamba_cache_chunk_size")]
fn new_panics_without_a_chunk_size() {
    MambaComponent::new(&CacheInitParams::default());
}

#[test]
fn evict_excess_path_states_removes_the_shallowest_states_beyond_the_cap() {
    let mut tc = mamba_core_with_cap(2);
    let [a, b, c] = chain::<3>(&mut tc);
    set_mamba_device(&mut tc, a, 7);
    set_mamba_device(&mut tc, b, 8);
    set_mamba_device(&mut tc, c, 9);
    let mut result = tc.evict_excess_path_states(tc.arena.node(c).id);
    let freed = result
        .device_frees
        .remove(&MAMBA)
        .expect("the excess state frees its slot");
    assert_eq!(freed.len(), 1);
    assert!(freed[0].equal(&Tensor::from_slice(&[7i64])));
    assert!(result.host_frees.is_empty());
    assert!(tc.arena.node(a).try_device_value(MAMBA).is_none());
    assert!(tc.arena.node(b).try_device_value(MAMBA).is_some());
    assert!(tc.arena.node(c).try_device_value(MAMBA).is_some());
}

#[test]
fn evict_excess_path_states_preserves_forks_locked_nodes_and_the_tail() {
    let mut tc = mamba_core_with_cap(1);
    let [a, b, c] = chain::<3>(&mut tc);
    set_mamba_device(&mut tc, a, 7);
    set_mamba_device(&mut tc, b, 8);
    set_mamba_device(&mut tc, c, 9);
    // a forks; b is locked: the cap is soft and neither state is removed.
    tc.arena
        .alloc_child(
            a,
            /* key = */ vec![9],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .node_mut(b)
        .set_lock_ref_(ValueSlotIdx::device(MAMBA), 1);
    let result = tc.evict_excess_path_states(tc.arena.node(c).id);
    assert!(result.device_frees.is_empty());
    assert!(result.host_frees.is_empty());
    assert!(tc.arena.node(a).try_device_value(MAMBA).is_some());
    assert!(tc.arena.node(b).try_device_value(MAMBA).is_some());
    assert!(tc.arena.node(c).try_device_value(MAMBA).is_some());
}

#[test]
fn evict_excess_path_states_without_a_cap_is_a_no_op() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a, b] = chain::<2>(&mut tc);
    set_mamba_device(&mut tc, a, 7);
    set_mamba_device(&mut tc, b, 8);
    let result = tc.evict_excess_path_states(tc.arena.node(b).id);
    assert!(result.device_frees.is_empty());
    assert!(result.host_frees.is_empty());
    assert!(tc.arena.node(a).try_device_value(MAMBA).is_some());
}

#[test]
fn insert_commit_emits_the_path_cap_action_only_when_capped() {
    let mut tc = mamba_core_with_cap(1);
    let result = tc.insert(&insert_params_mamba(&vec![1, 2], &[10, 11], Some(7)));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    let [CacheAction::MambaEvictExcessPathStates { tail_node_id }] =
        result.cache_actions.as_slice()
    else {
        panic!(
            "expected one MambaEvictExcessPathStates, got {:?}",
            action_kinds(&result.cache_actions)
        );
    };
    assert_eq!(*tail_node_id, leaf);

    let mut uncapped = mamba_core(/* page_size = */ 1);
    let result = uncapped.insert(&insert_params_mamba(&vec![1, 2], &[10, 11], Some(7)));
    assert!(result.cache_actions.is_empty());
}

#[test]
fn swa_evict_on_a_full_locked_leaf_sweeps_mamba_and_spares_full() {
    let mut tc = UnifiedTreeCore::<Vec<i64>>::new(
        CacheInitParams {
            page_size: 1,
            swa_sliding_window_size: Some(4),
            mamba_cache_chunk_size: Some(256),
            ..CacheInitParams::default()
        },
        vec![FULL, SWA, MAMBA],
    );
    let [a] = chain::<1>(&mut tc);
    tc.arena
        .set_device_value(a, FULL, Tensor::from_slice(&[10i64]));
    tc.set_component_device_value(tc.arena.node(a).id, SWA, Tensor::from_slice(&[20i64]));
    set_mamba_device(&mut tc, a, 7);
    // The held Full lock keeps the leaf out of the D-leaf set.
    tc.arena
        .node_mut(a)
        .set_lock_ref_(ValueSlotIdx::device(FULL), 1);
    tc.update_evictable_leaf_sets_(a);
    assert!(!tc.evictable_device_leaves.contains(a));
    let mut tracker = HashMap::from([(FULL, 0), (SWA, 0), (MAMBA, 0)]);
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    tc.evict_device_start(SWA, /* request_cnt = */ 10);
    let (next, step) = tc.evict_device_next_node(SWA, &tracker);
    accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    assert_eq!(next, None);
    tc.evict_device_end(SWA);
    // The walk tombstoned SWA inline and cascaded the lower-tier mamba slot.
    assert!(!tc.arena.has_device_value(a, SWA));
    assert!(!tc.arena.node(a).has_device_value(MAMBA));
    assert_eq!(tracker[&SWA], 1);
    assert_eq!(tracker[&MAMBA], 1);
    assert!(device_frees[&SWA][0].equal(&Tensor::from_slice(&[10i64])));
    assert!(device_frees[&MAMBA][0].equal(&Tensor::from_slice(&[7i64])));
    // The higher-tier locked Full is spared and pins the node in the tree.
    assert!(
        tc.arena
            .device_value(a, FULL)
            .equal(&Tensor::from_slice(&[10i64]))
    );
    assert_eq!(tc.arena.device_lock_ref(a, FULL), 1);
    assert_eq!(tc.arena.len(), 2);
    assert!(!tc.device_lru_list(SWA).in_list(Some(a)));
    assert!(!tc.device_lru_list(MAMBA).in_list(Some(a)));
    assert!(!tc.evictable_device_leaves.contains(a));
    assert_eq!(tc.evictable_size_(SWA), 0);
    assert_eq!(tc.evictable_size_(MAMBA), 0);
}

#[test]
fn host_only_mamba_anchor_disables_branching_under_hicache() {
    // Without hicache, the mamba-less walk reports the chunk-aligned branch point.
    let mut plain = mamba_core_with_chunk(/* page_size = */ 1, /* chunk = */ 3);
    let root = plain.arena.root();
    let n = plain
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2, 3, 4],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    plain
        .arena
        .set_device_value(n, FULL, Tensor::from_slice(&[10i64, 11, 12, 13]));
    let result = plain.match_prefix(&match_params(&vec![1, 2, 3, 4]));
    assert_eq!(result.best_match_node_id, plain.arena.node(root).id);
    assert_eq!(result.last_device_node_id, plain.arena.node(root).id);
    assert_eq!(result.mamba_branching_seqlen, Some(3));

    // HiCache: the host-only anchor advances the best match; no branch point.
    let mut tc = mamba_core_with_chunk(/* page_size = */ 1, /* chunk = */ 3);
    tc.set_hicache_enabled();
    let root = tc.arena.root();
    let n = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2, 3, 4],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(n, FULL, Tensor::from_slice(&[10i64, 11, 12, 13]));
    set_mamba_host(&mut tc, n, 8);
    tc.host_lru_list_mut(MAMBA).insert_mru(n);
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3, 4]));
    assert_eq!(result.best_match_node_id, tc.arena.node(n).id);
    assert_eq!(result.last_device_node_id, tc.arena.node(root).id);
    assert_eq!(result.mamba_branching_seqlen, None);
    assert!(result.mamba_host_hit_length >= 1);
    assert_eq!(result.host_hit_length, 4);
}

#[test]
fn branching_from_a_host_full_hit_is_reusable_after_insert() {
    let mut tc = mamba_core_with_chunk(/* page_size = */ 1, /* chunk = */ 3);
    tc.set_hicache_enabled();
    tc.insert(&insert_params_mamba(&vec![1, 2, 3], &[10, 11, 12], Some(7)));
    tc.insert(&insert_params_mamba(
        &vec![1, 2, 3, 4, 5, 6, 7],
        &[10, 11, 12, 13, 14, 15, 16],
        Some(8),
    ));
    let a = tc
        .match_prefix(&match_params(&vec![1, 2, 3]))
        .best_match_node_id;
    let b = tc
        .match_prefix(&match_params(&vec![1, 2, 3, 4, 5, 6, 7]))
        .best_match_node_id;
    tc.commit_backup(
        b,
        Tensor::from_slice(&[100i64, 101, 102, 103]),
        HashMap::new(),
    );
    tc.demote(b);
    // The demote's cascade swept b's mamba slot: b is Full-host-only, no mamba.
    assert!(!tc.arena.node(tc.arena.resolve(b)).has_device_value(MAMBA));
    assert!(!tc.arena.node(tc.arena.resolve(b)).has_host_value(MAMBA));
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3, 4, 5, 6, 7]));
    assert_eq!(result.best_match_node_id, a);
    assert_eq!(result.last_device_node_id, a);
    assert_eq!(result.device_indices.numel(), 3);
    assert_eq!(result.host_hit_length, 0);
    assert_eq!(result.full_kv_hit_length, 7);
    assert_eq!(result.mamba_branching_seqlen, Some(6));
    // Re-inserting up to the branch point makes the span device-reusable.
    let insert_result = tc.insert(&insert_params_mamba(
        &vec![1, 2, 3, 4, 5, 6],
        &[20, 21, 22, 23, 24, 25],
        Some(9),
    ));
    assert!(!insert_result.mamba_exist);
    let second = tc.match_prefix(&match_params(&vec![1, 2, 3, 4, 5, 6, 7]));
    assert_eq!(second.device_indices.numel(), 6);
    assert_eq!(second.mamba_branching_seqlen, None);
}

#[test]
fn skip_set_release_after_a_restore_and_relock_keeps_the_new_lock() {
    let mut tc = mamba_core(/* page_size = */ 1);
    let [a] = chain::<1>(&mut tc);
    let mamba = mamba_component();
    let first = mamba.acquire_component_lock(
        &mut tc,
        a,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    assert!(first.skip_lock_node_ids[&MAMBA].contains(&tc.arena.node(a).id));
    // The tombstone is restored and a second request locks it before the
    // first release replays its skip set.
    set_mamba_device(&mut tc, a, 7);
    let _ = mamba.acquire_component_lock(
        &mut tc,
        a,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    assert_eq!(tc.arena.node(a).device_lock_ref(MAMBA), 1);
    assert_eq!(tc.evictable_size_(MAMBA), 0);
    assert_eq!(tc.protected_size_(MAMBA), 1);
    let params = DecLockRefParams {
        skip_lock_node_ids: first.skip_lock_node_ids.clone(),
        ..DecLockRefParams::default()
    };
    mamba.release_component_lock(&mut tc, a, Some(&params), /* lock_host = */ false);
    // The replayed skip keeps the restored node's fresh lock intact.
    assert_eq!(tc.arena.node(a).device_lock_ref(MAMBA), 1);
    assert_eq!(tc.protected_size_(MAMBA), 1);
    mamba.release_component_lock(&mut tc, a, None, /* lock_host = */ false);
    assert_eq!(tc.arena.node(a).device_lock_ref(MAMBA), 0);
    assert_eq!(tc.evictable_size_(MAMBA), 1);
    assert_eq!(tc.protected_size_(MAMBA), 0);
}
