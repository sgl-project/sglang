//! Lightweight identifiers used across pipeline stages.

use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic, process-local request id. Cheap to copy, used as the routing key
/// on the egress side (detok shard selection) and to correlate scheduler output
/// chunks back to the originating connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(pub u64);

impl RequestId {
    /// Shard index for `n` detokenizer shards. Pure function of the id so the
    /// ingress and egress sides agree without any shared map.
    #[inline]
    pub fn shard(self, n: usize) -> usize {
        debug_assert!(n > 0);
        (self.0 as usize) % n
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "req-{}", self.0)
    }
}

/// Source of fresh request ids. One instance lives in the API server.
#[derive(Debug, Default)]
pub struct RequestIdGen(AtomicU64);

impl RequestIdGen {
    pub fn next(&self) -> RequestId {
        // Relaxed is fine: we only need uniqueness, not cross-thread ordering.
        RequestId(self.0.fetch_add(1, Ordering::Relaxed))
    }
}
