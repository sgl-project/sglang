// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use sgl_kv_indexer::PrefixOutcome;
use sgl_router::policies::kv_events::{
    compute_block_hashes, BlockSizeOracle, HashTree, KvWorkerId, Tiers,
};
use sgl_router::policies::prefix_provider::RadixTreePrefixProvider;

#[test]
fn radix_tree_reports_contiguous_prefix_depth_per_worker() {
    let tokens = [11_u32, 12, 13, 14];
    let hashes = compute_block_hashes(&tokens, 1);
    let tree = Arc::new(HashTree::new());
    let oracle = BlockSizeOracle::new();
    oracle.try_set(1).unwrap();

    tree.insert(&KvWorkerId::new("http://deep".into(), 0), None, &hashes);
    tree.insert(
        &KvWorkerId::new("http://deep".into(), 1),
        None,
        &hashes[..3],
    );
    tree.insert(
        &KvWorkerId::new("http://shallow".into(), 0),
        None,
        &hashes[..2],
    );

    let signal = RadixTreePrefixProvider::new(tree, oracle)
        .match_request_tokens(&tokens)
        .expect("established local tree must produce a prefix signal");
    let PrefixOutcome::Matched {
        matches,
        best_prefix_blocks,
    } = signal.outcome
    else {
        panic!("local radix-tree hit must be normalized as a match");
    };
    let depth_by_url: HashMap<_, _> = matches
        .into_iter()
        .map(|entry| (entry.address, entry.matched_prefix_blocks))
        .collect();

    assert_eq!(signal.query_blocks, 4);
    assert_eq!(best_prefix_blocks, 4);
    assert_eq!(depth_by_url.get("http://deep"), Some(&4));
    assert_eq!(depth_by_url.get("http://shallow"), Some(&2));
}

/// A URL can appear under several dp ranks, and only ONE of them is what
/// routing gets. Depth picks that rank; the tier only breaks a tie between
/// equally deep ranks. Taking the best depth and the best tier independently
/// would report an in-place hit for a prefix that is actually loaded back.
#[test]
fn dp_ranks_of_one_url_collapse_on_depth_then_tier() {
    let tokens = [21_u32, 22, 23];
    let hashes = compute_block_hashes(&tokens, 1);
    let tree = Arc::new(HashTree::new());
    let oracle = BlockSizeOracle::new();
    oracle.try_set(1).unwrap();

    // Deeper rank is host-only; shallower rank has the cheaper tier. Depth
    // must win, so the reported tier is the deep rank's `host`.
    let url = "http://w";
    tree.insert_tiered(&KvWorkerId::new(url.into(), 0), None, &hashes, Tiers::HOST);
    tree.insert_tiered(
        &KvWorkerId::new(url.into(), 1),
        None,
        &hashes[..1],
        Tiers::DEVICE,
    );

    let signal = RadixTreePrefixProvider::new(Arc::clone(&tree), Arc::clone(&oracle))
        .match_request_tokens(&tokens)
        .expect("established local tree must produce a prefix signal");
    let view = signal.tree_view.expect("the local tree reports tiers");
    assert_eq!(view.owner_tiers.get(url), Some(&"host"));
    assert_eq!(
        view.block0_in_tree, None,
        "a match has nothing to attribute, so the extra tree read is skipped",
    );

    // Now give an equally deep rank a device copy: same depth, cheaper tier.
    tree.insert_tiered(
        &KvWorkerId::new(url.into(), 1),
        None,
        &hashes,
        Tiers::DEVICE,
    );
    let signal = RadixTreePrefixProvider::new(tree, oracle)
        .match_request_tokens(&tokens)
        .expect("established local tree must produce a prefix signal");
    let view = signal.tree_view.expect("the local tree reports tiers");
    assert_eq!(
        view.owner_tiers.get(url),
        Some(&"device"),
        "equal depth: the cheaper tier breaks the tie",
    );
}

/// A walk that reaches nobody is an ANSWER, not a missing lookup. Returning
/// `None` here would make a tree miss indistinguishable from a router that
/// never looked — which is the difference between `no_candidates` and
/// `lookup_unavailable` on `sgl_router_cache_aware_decisions_total`.
#[test]
fn a_tree_miss_is_an_empty_match_not_a_missing_signal() {
    let tokens = [31_u32, 32];
    let tree = Arc::new(HashTree::new());
    let oracle = BlockSizeOracle::new();
    oracle.try_set(1).unwrap();

    let signal = RadixTreePrefixProvider::new(Arc::clone(&tree), Arc::clone(&oracle))
        .match_request_tokens(&tokens)
        .expect("an established block size means the lookup ran");
    assert!(matches!(signal.outcome, PrefixOutcome::Empty));
    assert_eq!(signal.query_blocks, 2);
    let view = signal
        .tree_view
        .expect("the local tree can attribute a miss");
    assert_eq!(
        view.block0_in_tree,
        Some(false),
        "nothing published: the gap is engine-side",
    );

    // Publish block 0 as a continuation of a chain the router never saw, so
    // it is carried but unreachable from the root: a router-side linkage
    // fault, which must read differently.
    let hashes = compute_block_hashes(&tokens, 1);
    tree.insert(&KvWorkerId::new("http://w".into(), 0), None, &[999]);
    tree.insert(&KvWorkerId::new("http://w".into(), 0), Some(999), &hashes);
    let signal = RadixTreePrefixProvider::new(tree, oracle)
        .match_request_tokens(&tokens)
        .expect("an established block size means the lookup ran");
    assert!(matches!(signal.outcome, PrefixOutcome::Empty));
    let view = signal
        .tree_view
        .expect("the local tree can attribute a miss");
    assert_eq!(
        view.block0_in_tree,
        Some(true),
        "carried but unreachable: the gap is router-side",
    );
}

/// Without a block size established from the fleet there is no lookup at all,
/// which is a different thing from a lookup that matched nothing.
#[test]
fn no_block_size_yet_means_no_signal() {
    let provider = RadixTreePrefixProvider::new(Arc::new(HashTree::new()), BlockSizeOracle::new());
    assert!(provider.match_request_tokens(&[41_u32, 42]).is_none());
}
