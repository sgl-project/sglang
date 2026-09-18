// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cross-implementation parity test for the KV-event block-hash algorithm.
//!
//! The Rust implementation at `src/policies/kv_events/hash.rs` must produce
//! the same i64 block hashes as SGLang's `radix_cache::RadixKey.hash_page`
//! followed by `hash_str_to_int64`.  Hard-coded `cross_language_golden_*`
//! values inside `hash.rs` are correct but brittle: if either side's
//! algorithm changes, the comments don't get regenerated and the tests
//! pass with stale expectations.
//!
//! This test consumes a fixture produced by
//! `tests/scripts/generate_kv_events_hash_parity.py`, which replicates the
//! SGLang algorithm verbatim (see the script's docstring for authority
//! pointers). CI regenerates the fixture (see
//! `.github/workflows/pr-test-sgl-router.yml`) and diffs against the
//! committed file; this test asserts the Rust implementation matches
//! whatever fixture is checked in.

use serde::Deserialize;
use sgl_router::policies::kv_events::compute_block_hashes;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct ParityCase {
    name: String,
    tokens: Vec<u32>,
    block_size: usize,
    expected_i64_hashes: Vec<i64>,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("kv_events_hash_parity.json")
}

fn load_cases() -> Vec<ParityCase> {
    let path = fixture_path();
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("read parity fixture {}: {e}", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("decode parity fixture {}: {e}", path.display()))
}

#[test]
fn fixture_is_non_empty() {
    let cases = load_cases();
    assert!(
        !cases.is_empty(),
        "kv_events_hash_parity.json is empty — run \
         tests/scripts/generate_kv_events_hash_parity.py",
    );
}

/// Drives every case in the fixture through `compute_block_hashes` and
/// asserts equality with the Python-derived expectation.
#[test]
fn rust_block_hashes_match_python_radix_cache() {
    for case in load_cases() {
        // block_size of 0 is rejected by `compute_block_hashes` with a
        // panic; the Python generator also rejects it.  The fixture
        // doesn't include a 0 case, so unwrap is safe.
        let block_size = std::num::NonZeroUsize::new(case.block_size)
            .unwrap_or_else(|| panic!("case {} has block_size=0 which is invalid", case.name));
        let got = compute_block_hashes(&case.tokens, block_size.get());
        assert_eq!(
            got, case.expected_i64_hashes,
            "case {}: tokens={:?} block_size={} — Rust produced {:?}, fixture says {:?}",
            case.name, case.tokens, case.block_size, got, case.expected_i64_hashes,
        );
    }
}
