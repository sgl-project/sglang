// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared policy for tests that need a live Redis.

/// Reports that `test` cannot run, and decides whether that is fatal.
///
/// Locally a missing store should skip so `cargo test` stays usable without
/// Redis. In CI the store is always provisioned, so a skip means the harness
/// broke and the job would otherwise report success having asserted nothing.
/// `KV_INDEXER_REQUIRE_REDIS=1` selects the second interpretation.
pub fn skip(test: &str, reason: &str) {
    if std::env::var("KV_INDEXER_REQUIRE_REDIS").is_ok_and(|v| v == "1") {
        panic!("{test} requires a store but {reason}");
    }
    eprintln!("skipping {test}: {reason}");
}
