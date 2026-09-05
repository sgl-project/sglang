use std::collections::HashMap;

use sglang_radix_tree::{
    CacheInitParams, FULL, InsertParams, InsertResult, KeyNamespaceRef, MAMBA, MatchPrefixParams,
    PageValue, RadixValue, TreeCoreRuntimeError, UnifiedTreeCore,
};

type TestCore = UnifiedTreeCore<Vec<i64>, PageValue<u32>>;

fn core(eviction_policy: &str) -> TestCore {
    TestCore::new_with_empty(
        CacheInitParams {
            eviction_policy: eviction_policy.to_string(),
            ..Default::default()
        },
        vec![FULL],
        PageValue::default(),
    )
}

fn insert(core: &mut TestCore, key: &[i64], values: &[u32]) -> InsertResult<PageValue<u32>> {
    core.insert(&InsertParams {
        key: &key.to_vec(),
        namespace: KeyNamespaceRef::default(),
        value: PageValue::from_vec(values.to_vec()),
        prev_prefix_len: 0,
        swa_evicted_seqlen: 0,
        mamba_value: None,
        chunked: false,
        priority: 0,
        track_adopted_ranges: false,
    })
}

#[test]
fn page_value_core_supports_read_only_match_and_continuation_insert() {
    let mut tree = core("lru");
    insert(&mut tree, &[10, 20, 30, 40], &[1, 2, 3, 4]);

    let partial_key = vec![10, 20, 50, 60];
    assert_eq!(
        tree.prefix_match_len(&MatchPrefixParams {
            key: &partial_key,
            namespace: KeyNamespaceRef::default(),
        }),
        2
    );
    assert_eq!(
        tree.prefix_match_atoms_len(&partial_key, KeyNamespaceRef::default()),
        2
    );

    let prefix_key = vec![10, 20];
    let prefix = tree.match_prefix(&MatchPrefixParams {
        key: &prefix_key,
        namespace: KeyNamespaceRef::default(),
    });
    assert_eq!(prefix.device_indices.as_slice(), &[1, 2]);
    tree.inc_lock_ref(prefix.last_device_node_id);
    assert_eq!(tree.protected_size(), 2);

    let result = tree.insert_suffix_from_node(
        prefix.last_device_node_id,
        2,
        &InsertParams {
            key: &partial_key,
            namespace: KeyNamespaceRef::default(),
            value: PageValue::from_vec(vec![5, 6]),
            prev_prefix_len: 2,
            swa_evicted_seqlen: 0,
            mamba_value: None,
            chunked: false,
            priority: 0,
            track_adopted_ranges: false,
        },
    );
    assert_eq!(result.prefix_len, 2);
    assert_eq!(result.total_len, 4);
    assert!(result.cache_actions.is_empty());

    let canonical_suffix = tree.collect_full_device_indices(
        result.last_device_node_id.expect("inserted node"),
        prefix.last_device_node_id,
    );
    assert_eq!(canonical_suffix.as_slice(), &[5, 6]);

    let inserted_node = result.last_device_node_id.expect("inserted node");
    tree.inc_lock_ref(inserted_node);
    tree.dec_lock_ref(prefix.last_device_node_id, None, false);
    assert_eq!(tree.protected_size(), 4);
    tree.dec_lock_ref(inserted_node, None, false);
    assert_eq!(tree.protected_size(), 0);
    // The original [30, 40] branch plus the new branch remain cacheable.
    assert_eq!(tree.evictable_size(), 6);

    let matched = tree.match_prefix(&MatchPrefixParams {
        key: &partial_key,
        namespace: KeyNamespaceRef::default(),
    });
    assert_eq!(matched.device_indices.as_slice(), &[1, 2, 5, 6]);
}

#[test]
fn continuation_insert_rejects_a_host_only_anchor() {
    let mut tree = TestCore::new_with_empty(
        CacheInitParams {
            enable_hicache: true,
            ..Default::default()
        },
        vec![FULL],
        PageValue::default(),
    );
    insert(&mut tree, &[10, 20, 30, 40], &[1, 2, 3, 4]);
    let anchor = tree
        .match_prefix(&MatchPrefixParams {
            key: &vec![10, 20],
            namespace: KeyNamespaceRef::default(),
        })
        .last_device_node_id;
    let leaf = tree
        .match_prefix(&MatchPrefixParams {
            key: &vec![10, 20, 30, 40],
            namespace: KeyNamespaceRef::default(),
        })
        .last_device_node_id;
    // Back both nodes up and drop their device copies: the anchor is now a
    // host-only tombstone.
    tree.commit_backup(anchor, PageValue::from_vec(vec![101, 102]), HashMap::new());
    tree.commit_backup(leaf, PageValue::from_vec(vec![103, 104]), HashMap::new());
    let _ = tree.demote(leaf);
    let _ = tree.demote(anchor);

    let key = vec![10, 20, 50, 60];
    let error = tree
        .try_insert_suffix_from_node(
            anchor,
            2,
            &InsertParams {
                key: &key,
                namespace: KeyNamespaceRef::default(),
                value: PageValue::from_vec(vec![5, 6]),
                prev_prefix_len: 2,
                swa_evicted_seqlen: 0,
                mamba_value: None,
                chunked: false,
                priority: 0,
                track_adopted_ranges: false,
            },
        )
        .err()
        .expect("a host-only anchor must be rejected");
    assert!(matches!(
        error,
        TreeCoreRuntimeError::InsertAnchorNotDeviceResident { .. }
    ));
    tree.try_sanity_check(&[], &[])
        .expect("the rejected insert leaves the tree consistent");
}

#[test]
fn empty_insert_leaves_the_donated_mamba_slot_with_the_caller() {
    let mut tree = TestCore::new_with_empty(
        CacheInitParams {
            page_size: 2,
            mamba_cache_chunk_size: Some(2),
            ..Default::default()
        },
        vec![FULL, MAMBA],
        PageValue::default(),
    );
    // One token under page_size 2 aligns to an empty key: nothing is inserted,
    // so the caller must free its donated slot (mamba_exist == true).
    let key = vec![10];
    let result = tree.insert(&InsertParams {
        key: &key,
        namespace: KeyNamespaceRef::default(),
        value: PageValue::from_vec(vec![1]),
        prev_prefix_len: 0,
        swa_evicted_seqlen: 0,
        mamba_value: Some(PageValue::from_vec(vec![7])),
        chunked: false,
        priority: 0,
        track_adopted_ranges: false,
    });
    assert_eq!(result.total_len, 0);
    assert!(result.mamba_exist);
}

#[test]
fn every_eviction_policy_operates_without_torch() {
    for policy in ["lru", "lfu", "fifo", "mru", "filo", "priority", "slru"] {
        let mut tree = core(policy);
        insert(&mut tree, &[10, 20], &[1, 2]);
        assert_eq!(tree.evictable_size(), 2, "{policy}");

        tree.evict_device_start(FULL, 1);
        let (node_id, step) = tree.evict_device_next_node(FULL, &HashMap::new());
        assert!(step.tracker.is_empty(), "{policy}");
        let (_, result) = tree.evict_device_leaf(node_id.expect("eviction candidate"), false);
        tree.evict_device_end(FULL);

        assert_eq!(result.tracker.get(&FULL), Some(&2), "{policy}");
        assert_eq!(
            result.device_frees[&FULL][0].as_slice(),
            &[1, 2],
            "{policy}"
        );
    }
}

#[test]
fn continuation_insert_shares_immutable_value_storage() {
    let mut tree = core("lru");
    let initial_key = vec![10, 20];
    insert(&mut tree, &initial_key, &[1, 2]);
    let prefix = tree.match_prefix(&MatchPrefixParams {
        key: &initial_key,
        namespace: KeyNamespaceRef::default(),
    });

    let extended_key = vec![10, 20, 30, 40];
    let extended_value = PageValue::from_vec(vec![3, 4]);
    let extended_ptr = extended_value.as_slice().as_ptr();
    let result = tree.insert_suffix_from_node(
        prefix.last_device_node_id,
        2,
        &InsertParams {
            key: &extended_key,
            namespace: KeyNamespaceRef::default(),
            value: extended_value,
            prev_prefix_len: 2,
            swa_evicted_seqlen: 0,
            mamba_value: None,
            chunked: false,
            priority: 0,
            track_adopted_ranges: false,
        },
    );

    // The stored suffix still points at the caller's buffer: no page was copied.
    let stored = tree.collect_full_device_indices(
        result.last_device_node_id.expect("inserted node"),
        prefix.last_device_node_id,
    );
    assert_eq!(stored.as_slice(), &[3, 4]);
    assert_eq!(stored.as_slice().as_ptr(), extended_ptr);
    assert_eq!(stored.to_i64_vec(), vec![3, 4]);
}
