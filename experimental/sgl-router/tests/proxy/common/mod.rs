// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared test harness re-exports.

pub mod cache_aware_fixture;
pub mod mock_worker;
pub mod streaming;

/// Whether `rid` has the shape the router mints: a bare `uuid4` hex, identical
/// to what the engine mints for itself. Tests assert the SHAPE rather than a
/// marker, because being indistinguishable from an engine-minted rid is the
/// point — it is what keeps the client-visible response `id` unchanged.
#[allow(dead_code)] // not every test file inspects forwarded rids
pub fn is_engine_shaped_rid(rid: &str) -> bool {
    rid.len() == 32
        && rid
            .bytes()
            .all(|b| b.is_ascii_digit() || b.is_ascii_lowercase())
}
