//! Lightweight identifiers used across pipeline stages.

use std::{
    collections::hash_map::RandomState,
    fmt,
    hash::{BuildHasher, Hash},
    ops::Deref,
    sync::OnceLock,
};

use uuid::Uuid;

/// Health-probe rid prefix — MUST match the Python server's
/// `sglang.srt.constants.HEALTH_CHECK_RID_PREFIX`, so scheduler logs / crash
/// dumps and any prefix-gated logic recognize probes from either server.
pub const HEALTH_CHECK_RID_PREFIX: &str = "HEALTH_CHECK";

#[derive(Clone, Debug)]
pub struct Rid {
    id: String,
    /// Partition key, derived from `id`. Never part of identity — see the `Eq` /
    /// `Hash` impls below.
    hash: u64,
}

// Identity is the ID, not the digest. Deriving these would fold `hash` into both,
// which is redundant while the seed is stable and silently wrong if it ever isn't.
impl PartialEq for Rid {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Rid {}
impl Hash for Rid {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Rid {
    /// Borrow the underlying id — the form the msgpack wire and the HTTP body use.
    pub fn as_str(&self) -> &str {
        &self.id
    }

    pub fn new() -> Self {
        let id = Uuid::new_v4().simple().to_string();
        Rid::from(id)
    }

    pub fn new_health_check() -> Self {
        let id = format!("{HEALTH_CHECK_RID_PREFIX}_{}", Uuid::new_v4().simple());
        Rid::from(id)
    }

    /// Shard index for `n` detokenizer shards. Pure function of the id so the
    /// ingress and egress sides agree without any shared map.
    #[inline]
    pub fn shard(&self, n: usize) -> usize {
        debug_assert!(n > 0);
        (self.hash as usize) % n
    }
}

impl From<String> for Rid {
    fn from(id: String) -> Self {
        // ONE seed per process, not one per conversion. Ingress and egress each
        // build a `Rid` from the same string and must agree on the shard without a
        // shared map — a fresh `RandomState` here would hash the same rid two
        // different ways, so chunks would arrive at a shard that never registered
        // the request and be dropped.
        //
        // The seed is random rather than fixed because rids are client-supplied:
        // with public keys, colliding rids are an offline ~2^32 search. Collisions
        // are only a shard co-location now (identity is the string), but a keyed
        // hash also stops an attacker from stacking every request onto one shard.
        static SEED: OnceLock<RandomState> = OnceLock::new();
        let hash = SEED.get_or_init(RandomState::new).hash_one(&id);
        Rid { id, hash }
    }
}

impl From<&str> for Rid {
    fn from(id: &str) -> Self {
        Rid::from(id.to_string())
    }
}

impl Deref for Rid {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.id
    }
}

impl Default for Rid {
    fn default() -> Self {
        Rid::new()
    }
}

impl fmt::Display for Rid {
    /// The BARE rid, with no decoration. It is formatted into client-facing error
    /// messages and into wire values (`AbortReq`), so a prefix here would surface
    /// as a corrupted id rather than a nicety. `Debug` still shows `Rid("…")` if a
    /// log wants the type visible.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.id)
    }
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
        let rid = Rid::new();
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
        let rid = Rid::new_health_check();
        // "HEALTH_CHECK_" + 32 hex chars
        assert!(rid.starts_with("HEALTH_CHECK_"));
        assert_eq!(rid.len(), "HEALTH_CHECK_".len() + 32);
    }
}
