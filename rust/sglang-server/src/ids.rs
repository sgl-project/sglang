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
