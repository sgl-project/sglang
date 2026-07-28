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

/// Separates a client-supplied rid from the uniquifier appended to it (see
/// [`Rid::from_client`]). Deliberately a character this server never mints:
/// [`Rid::new`] is uuid hex and [`Rid::new_health_check`] adds only
/// `HEALTH_CHECK_`, so its presence at the fixed offset below is what lets
/// [`Rid::client_facing`] recognize a suffix without `Rid` carrying a flag —
/// which matters because `Rid` rides on every `ChunkEvent`.
const UNIQ_SEP: u8 = b'#';
/// Hex digits of uniquifier: 8 of a per-process random base, 8 of a counter.
const UNIQ_DIGITS: usize = 16;
/// Total bytes appended. Fixed-width by construction — both halves are `u32`
/// formatted `{:08x}` — which is what makes stripping a slice, not a search.
const UNIQ_SUFFIX_LEN: usize = 1 + UNIQ_DIGITS;

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

    /// A CLIENT-SUPPLIED rid, made unique for internal use by appending a
    /// uniquifier.
    ///
    /// Nothing stops two concurrent requests from arriving with the same rid, and
    /// the rid is an identity downstream: detok `Register` is an insert-overwrite,
    /// so the second would evict the first's sink, 500 that client mid-generation
    /// and deliver its remaining chunks to the second's connection. Uniquifying
    /// here makes the collision unrepresentable rather than something a duplicate
    /// check has to catch — no in-flight registry, no admission/release ordering
    /// to get wrong, and both clients get served instead of the second being 400'd.
    ///
    /// The client never sees this: [`client_facing`](Self::client_facing) strips it
    /// back off for `meta_info.id`. Only the scheduler wire and its logs carry the
    /// suffixed form.
    ///
    /// The counter alone would guarantee uniqueness within a process; the random
    /// base covers several HTTP worker processes feeding one scheduler, where two
    /// counters would otherwise both start at zero. `u32` (not `u64`) keeps the
    /// `{:08x}` width exactly 8 — wrapping needs 2^32 live requests sharing one
    /// client rid.
    pub fn from_client(id: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static BASE: OnceLock<u32> = OnceLock::new();
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let base = *BASE.get_or_init(|| Uuid::new_v4().as_u128() as u32);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        Rid::from(format!(
            "{id}{sep}{base:08x}{n:08x}",
            sep = UNIQ_SEP as char
        ))
    }

    /// The rid as the client wrote it — what `meta_info.id` must echo, and what
    /// the client-facing length limit applies to. Returns the whole id for a rid
    /// this server minted, which carries no suffix to strip.
    pub fn client_facing(&self) -> &str {
        let b = self.id.as_bytes();
        let Some(cut) = b.len().checked_sub(UNIQ_SUFFIX_LEN) else {
            return &self.id;
        };
        // `UNIQ_SEP` is ASCII, so a match here is also a char boundary — slicing
        // at `cut` cannot split a multi-byte character in a client's rid.
        if b[cut] == UNIQ_SEP && b[cut + 1..].iter().all(u8::is_ascii_hexdigit) {
            &self.id[..cut]
        } else {
            &self.id
        }
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

    /// The round-trip that makes the scheme invisible: whatever the client sent
    /// comes back out of `client_facing`, byte for byte, however odd it is.
    /// `meta_info.id` is the only thing the client can correlate a response by, so
    /// leaking the uniquifier — or over-stripping a rid that happens to look like
    /// one — is a client-visible bug.
    #[test]
    fn client_facing_round_trips_whatever_the_client_sent() {
        for given in [
            "r",
            "",
            "x_0",
            "0123456789abcdef0123456789abcdef",
            // Already shaped like a suffix: stripping must remove OURS, not theirs.
            "abc#0123456789abcdef",
            "#0123456789abcdef",
            // Non-ASCII, so a byte-offset slice could split a character.
            "réq-π-🎉",
            &"x".repeat(128),
        ] {
            let rid = Rid::from_client(given);
            assert_eq!(
                rid.client_facing(),
                given,
                "round-trip failed for {given:?}"
            );
            assert_ne!(rid.as_str(), given, "the internal rid must be uniquified");
        }
    }

    /// A rid this server minted carries no suffix, so `client_facing` must return
    /// it whole. Nothing strips these today, but `Rid::new` is uuid hex and a
    /// 17-byte tail of it is all hex — only the missing separator saves it.
    #[test]
    fn client_facing_leaves_minted_rids_alone() {
        for rid in [Rid::new(), Rid::new_health_check(), Rid::default()] {
            assert_eq!(rid.client_facing(), rid.as_str());
        }
    }

    /// Uniqueness is the entire point, and it must hold for the same input — that
    /// IS the collision case. Checked across threads because the counter is shared
    /// by every api thread.
    #[test]
    fn from_client_is_unique_even_for_one_repeated_rid() {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    (0..250)
                        .map(|_| Rid::from_client("same").as_str().to_string())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let all: std::collections::HashSet<String> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        assert_eq!(all.len(), 1000, "every uniquified rid must be distinct");
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
