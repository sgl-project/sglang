// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cross-check the real Router event helper against the SAME Python fixture.
//! This includes the source module, not a reimplementation of its hash logic.

#[path = "../../src/state/kv_events/hash.rs"]
mod event_hash;

use serde::Deserialize;

const GOLDEN: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/event_hash_golden.json"
));

#[derive(Deserialize)]
struct Fixture {
    page_size: usize,
    token_ids: Vec<u32>,
    digests: Vec<String>,
    event_keys: Vec<i64>,
    page64: Page64,
}

#[derive(Deserialize)]
struct Page64 {
    page_size: usize,
    token_start: u32,
    token_end: u32,
    digests: Vec<String>,
    event_keys: Vec<i64>,
}

fn check(tokens: &[u32], page_size: usize, digests: &[String], expected: &[i64]) {
    // The helper can hash partial tails; peer-L1 only queries complete pages.
    let complete = &tokens[..tokens.len() / page_size * page_size];
    let actual = event_hash::compute_block_hashes(complete, page_size);
    assert_eq!(actual, expected);
    assert_eq!(
        event_hash::CacheNamespace::NONE.block_hashes(complete, page_size, false),
        expected
    );
    assert!(actual.iter().any(|&key| key < 0));
    assert!(actual.iter().any(|&key| key > 0));
    assert_eq!(digests.len(), expected.len());
    for (hex, &key) in digests.iter().zip(expected) {
        assert_eq!(hex.len(), 64);
        let mut digest = [0_u8; 32];
        for (i, byte) in digest.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
        }
        assert_eq!(event_hash::sha256_to_i64(&digest), key);
    }
}

#[test]
fn l1_event_hash_golden_matches_python_fixture_full_digest_chain() {
    let fixture: Fixture = serde_json::from_str(GOLDEN).unwrap();
    check(
        &fixture.token_ids,
        fixture.page_size,
        &fixture.digests,
        &fixture.event_keys,
    );
}

#[test]
fn l1_event_hash_golden_page64_root_prefix_restores_only_the_tail() {
    let fixture: Fixture = serde_json::from_str(GOLDEN).unwrap();
    let fixture = fixture.page64;
    let tokens: Vec<u32> = (fixture.token_start..fixture.token_end).collect();
    assert_eq!(fixture.page_size, 64);
    check(
        &tokens,
        fixture.page_size,
        &fixture.digests,
        &fixture.event_keys,
    );
    assert_eq!(fixture.event_keys.len(), 16);
    let (local_tokens, hit_tokens) = (256, 1024);
    assert_eq!(hit_tokens - local_tokens, 768);
    assert_eq!((hit_tokens - local_tokens) / fixture.page_size, 12);
    // Hashing only the restore tail resets the digest chain and is wrong.
    assert_ne!(
        event_hash::compute_block_hashes(&tokens[local_tokens..hit_tokens], fixture.page_size),
        fixture.event_keys[local_tokens / fixture.page_size..]
    );
}
