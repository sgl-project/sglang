// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use sgl_kv_indexer::PrefixOutcome;
use sgl_router::policies::kv_events::{
    compute_block_hashes, BlockSizeOracle, HashTree, KvWorkerId,
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
