// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use sgl_kv_indexer::PrefixOutcome;
use sgl_router::state::kv_events::{
    compute_block_hashes, BlockSizeOracle, HashTree, KvWorkerId, RadixTreePrefixProvider,
};

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
        .match_request_tokens(&tokens, None)
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

/// Unconfirmed placements must not count as cache hits for another salt.
#[test]
fn pending_prefixes_respect_cache_salt_namespaces() {
    let tokens = [11_u32, 12, 13, 14];
    for bigram in [false, true] {
        let tree = Arc::new(HashTree::new());
        tree.pending().enable(Duration::from_secs(60));
        let oracle = BlockSizeOracle::new();
        oracle.try_set(1).unwrap();
        oracle.set_bigram(bigram);
        let provider = RadixTreePrefixProvider::new(tree, oracle);
        let placements = [
            (None, "http://unsalted"),
            (Some("tenant-a"), "http://tenant-a"),
            (Some("租户-B"), "http://tenant-b"),
        ];
        for (salt, url) in placements {
            let signal = provider.match_request_tokens(&tokens, salt).unwrap();
            assert!(
                matches!(signal.outcome, PrefixOutcome::Empty),
                "pending prefix leaked into salt {salt:?}, bigram={bigram}"
            );
            provider.record_route(&signal, url);
        }
        for (salt, url) in placements
            .into_iter()
            .chain([(Some(""), "http://unsalted")])
        {
            let signal = provider.match_request_tokens(&tokens, salt).unwrap();
            let PrefixOutcome::Matched {
                matches,
                best_prefix_blocks,
            } = signal.outcome
            else {
                panic!("pending placement missing for salt {salt:?}, bigram={bigram}");
            };
            assert_eq!(matches.len(), 1);
            assert_eq!(matches[0].address, url);
            assert_eq!(best_prefix_blocks as usize, signal.query_blocks);
        }
        assert!(matches!(
            provider
                .match_request_tokens(&tokens, Some("unseen"))
                .unwrap()
                .outcome,
            PrefixOutcome::Empty
        ));
    }
}
