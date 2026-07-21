//! Lightweight identifiers used across pipeline stages.

use std::{
    collections::hash_map::DefaultHasher,
    fmt::{Debug, Display, Formatter, Result},
    hash::{Hash, Hasher},
};

use uuid::Uuid;

/// Process-local request key, derived by hashing the client-visible rid string
/// (`from_rid`). Cheap to copy, used as the routing key on the egress side
/// (detok shard selection) and to correlate scheduler output chunks back to the
/// originating connection. NOT the identity the client sees — that is the rid
/// string; this is its stable 64-bit digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RidHash(pub u64);

impl RidHash {
    /// Derive the routing key from a rid string. `DefaultHasher::new()` uses
    /// fixed keys, so every stage (ingress push, egress decode, abort) computes
    /// the same id from the same rid — no shared map.
    pub fn from_rid(rid: &str) -> Self {
        let mut h = DefaultHasher::new();
        rid.hash(&mut h);
        RidHash(h.finish())
    }

    /// Shard index for `n` detokenizer shards. Pure function of the id so the
    /// ingress and egress sides agree without any shared map.
    #[inline]
    pub fn shard(self, n: usize) -> usize {
        debug_assert!(n > 0);
        (self.0 as usize) % n
    }
}

impl Display for RidHash {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "req-{}", self.0)
    }
}

/// Mint a fresh client-visible request id — uuid4 hex, matching the Python
/// server's `uuid.uuid4().hex`.
pub fn new_rid() -> String {
    Uuid::new_v4().simple().to_string()
}

/// Health-probe rid prefix — MUST match the Python server's
/// `sglang.srt.constants.HEALTH_CHECK_RID_PREFIX`, so scheduler logs / crash
/// dumps and any prefix-gated logic recognize probes from either server.
pub const HEALTH_CHECK_RID_PREFIX: &str = "HEALTH_CHECK";

/// Mint a health-probe rid: `HEALTH_CHECK_<uuid4 hex>`, the Python server's
/// `f"{HEALTH_CHECK_RID_PREFIX}_{uuid.uuid4().hex}"` format.
pub fn new_health_check_rid() -> String {
    format!("{HEALTH_CHECK_RID_PREFIX}_{}", new_rid())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-language format guard: Python rids are `uuid.uuid4().hex` — 32
    /// lowercase hex chars, no hyphens. `.simple()` is the matching uuid-crate
    /// encoding; swapping it for the default `to_string()` (36 chars,
    /// hyphenated) would silently break the parity.
    #[test]
    fn rid_matches_python_uuid4_hex_format() {
        let rid = new_rid();
        assert_eq!(rid.len(), 32);
        assert!(
            rid.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "rid must be lowercase hex: {rid}"
        );
    }

    /// Cross-language literal guard: the prefix is dictated by Python's
    /// `constants.HEALTH_CHECK_RID_PREFIX` ("HEALTH_CHECK"); drifting silently
    /// would break prefix-gated handling (e.g. the disagg encode server).
    #[test]
    fn health_rid_matches_python_convention() {
        assert_eq!(HEALTH_CHECK_RID_PREFIX, "HEALTH_CHECK");
        let rid = new_health_check_rid();
        // "HEALTH_CHECK_" + 32 hex chars
        assert!(rid.starts_with("HEALTH_CHECK_"));
        assert_eq!(rid.len(), "HEALTH_CHECK_".len() + 32);
    }
}
