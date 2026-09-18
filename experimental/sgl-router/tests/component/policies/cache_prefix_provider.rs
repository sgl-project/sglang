// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use sgl_router::policies::state::kv_events::{compute_block_hashes, KvEventIndex, KvWorkerId};

#[tokio::test]
async fn radix_tree_reports_contiguous_prefix_depth_per_worker() {
    let tokens = [11_u32, 12, 13, 14];
    let hashes = compute_block_hashes(&tokens, 1);
    let index = KvEventIndex::new();
    index.block_size_oracle().try_set(1).unwrap();
    let tree = index.tree();
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

    let lookup = index
        .match_prefix(&tokens)
        .expect("established local tree must produce a prefix signal");
    let depth_by_url: HashMap<_, _> = lookup
        .matches
        .into_iter()
        .map(|entry| (entry.address, entry.matched_prefix_blocks))
        .collect();

    assert_eq!(lookup.query_blocks, 4);
    assert_eq!(depth_by_url.get("http://deep"), Some(&4));
    assert_eq!(depth_by_url.get("http://shallow"), Some(&2));
}
