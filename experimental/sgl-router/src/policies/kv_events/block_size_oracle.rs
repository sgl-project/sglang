// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Process-shared per-(cache-aware-zmq) hashing configuration, derived from
//! the workers themselves.
//!
//! # Why an oracle instead of a config field?
//!
//! `compute_block_hashes` must hash with the **same** block size the
//! worker uses to publish KV-cache events; otherwise every cache-aware
//! lookup misses silently. Event granularity is set by the worker's radix
//! tree page size: the worker advertises its `page_size` via `/server_info`
//! and discovery widens it by `dcp_size` (the tree adopts the DCP-widened
//! allocator page) into
//! [`crate::policies::kv_events::EventConfig::block_size`] — see
//! `kv_event_block_size`. Earlier versions of sgl-router carried a static
//! `block_size` field on `CacheAwareConfig`; nothing reconciled it with the
//! worker-reported value, so a mismatch silently destroyed cache-hit routing.
//!
//! # What this oracle does NOT catch
//!
//! It reconciles workers against **each other**, never against what a worker
//! actually publishes. A fleet that is uniformly wrong — every worker
//! reporting the same block size that the router derives incorrectly — agrees
//! perfectly here while every lookup misses. Only worker-vs-worker divergence
//! is detectable. Closing that hole needs the publisher's own `block_size`
//! off the wire; see `kv_event_block_size`.
//!
//! # Single oracle vs per-model
//!
//! For now the oracle is process-wide. Realistic deployments use one
//! `page_size` *and* one `dcp_size` across the cluster, so a single block
//! size suffices and mismatches across models indicate misconfiguration the
//! operator should see. A per-model oracle would require threading `ModelId`
//! through `KvEventIndex::add_worker`; that refactor can land later
//! without changing the oracle's public surface.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

/// First-wins, idempotent block-size publisher, plus the fleet's KV-block
/// hashing mode **derived from the live worker set**.
///
/// `block_size` is a first-wins latch ([`Self::try_set`]): workers must all
/// agree on a page size, so a mismatch is operator error reported loudly at
/// registration.
///
/// The hashing *mode* is different: EAGLE-family workers hash KV blocks over
/// token bigrams while everyone else hashes unigrams, and which modes the
/// fleet carries is a property of the **currently live workers**, not of
/// history. Deriving it from a per-worker registry keeps it accurate during a
/// rolling update that changes speculative-decoding config: every replica that
/// sees the same live set derives the same mode, the mode converges as the old
/// generation drains out of service discovery, and the peer-snapshot stamp
/// (`hash_config`) tracks the live fleet instead of permanently carrying
/// whichever worker happened to register first.
///
/// A heterogeneous fleet (EAGLE-family + non-EAGLE workers behind one base
/// model) is supported: [`Self::is_bimodal`] reports both modes present and
/// the selection path dual-hashes queries so neither family loses cache
/// affinity. `block_size` must still agree across the fleet; only the hashing
/// *mode* may differ.
#[derive(Debug, Default)]
pub struct BlockSizeOracle {
    value: AtomicU32,
    /// Live per-worker hashing modes, keyed by worker URL. The derived
    /// primary mode is the majority, with ties broken toward unigram — any
    /// deterministic rule would do; this one only matters so that two
    /// replicas holding the same live set derive the same value.
    modes: Mutex<HashMap<String, bool>>,
}

/// Returned by [`BlockSizeOracle::try_set`] when the candidate disagrees
/// with the already-established value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSizeMismatch {
    pub established: u32,
    pub candidate: u32,
}

/// The mode pair derived from a worker-mode map: `(primary, bimodal)`.
/// `None` when no worker is registered; primary is the majority mode.
fn derive(modes: &HashMap<String, bool>) -> Option<(bool, bool)> {
    if modes.is_empty() {
        return None;
    }
    let bigrams = modes.values().filter(|&&m| m).count();
    let unigrams = modes.len() - bigrams;
    Some((bigrams > unigrams, bigrams > 0 && unigrams > 0))
}

impl BlockSizeOracle {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Returns the established block size, or `None` if no worker has
    /// reported one yet. Routing-time consumers (`CacheAwareZmqPolicy`)
    /// fall back to min-load when this is `None`, because they cannot
    /// hash a prompt without a block size.
    pub fn get(&self) -> Option<u32> {
        let v = self.value.load(Ordering::Relaxed);
        if v == 0 {
            None
        } else {
            Some(v)
        }
    }

    /// Publish a live worker's hashing mode. Called from
    /// `KvEventIndex::add_worker` alongside `try_set`, and from the pump's
    /// post-restart re-introspection (`spawn_mode_recheck`); re-reporting
    /// overwrites the previous vote (a restarted worker may have changed
    /// speculative-decoding config).
    ///
    /// Transitions of the derived fleet shape are logged: entering or
    /// leaving a bimodal fleet is exactly the rolling-update window an
    /// operator needs to see, and the dual-hash selection path costs a
    /// second hash per query for as long as it lasts.
    pub(crate) fn report_worker(&self, worker_url: &str, is_bigram: bool) {
        let (before, after) = {
            let mut modes = self.modes.lock();
            let before = derive(&modes);
            modes.insert(worker_url.to_string(), is_bigram);
            (before, derive(&modes))
        };
        Self::log_transition(before, after, worker_url, true);
    }

    /// Drop a deregistered worker's mode. Called from
    /// `KvEventIndex::remove_worker`; without it a drained generation would
    /// keep voting in the derived mode forever.
    pub(crate) fn forget_worker(&self, worker_url: &str) {
        let (before, after) = {
            let mut modes = self.modes.lock();
            let before = derive(&modes);
            modes.remove(worker_url);
            (before, derive(&modes))
        };
        Self::log_transition(before, after, worker_url, false);
    }

    /// The recorded hashing mode of one registered worker, or `None` when it
    /// has not voted (or was forgotten on removal). `None` stays distinct
    /// from a unigram vote so consumers — the bootstrap acceptance log's
    /// unvoted tally, the re-introspection warn — can tell "never
    /// introspected" from "seen as unigram".
    pub(crate) fn vote_of(&self, worker_url: &str) -> Option<bool> {
        self.modes.lock().get(worker_url).copied()
    }

    fn log_transition(
        before: Option<(bool, bool)>,
        after: Option<(bool, bool)>,
        worker_url: &str,
        added: bool,
    ) {
        if before == after {
            return;
        }
        tracing::info!(
            worker_url = worker_url,
            worker_added = added,
            before = ?before,
            after = ?after,
            "kv-events: fleet hashing-mode shape changed (primary, bimodal) — derived \
             from the live worker set; bimodal selection dual-hashes every query",
        );
    }

    /// Whether the fleet currently carries workers of **both** hashing modes
    /// (an EAGLE-family bigram worker and a non-EAGLE unigram worker). Read
    /// at routing time so `CacheAwareZmqPolicy::select` dual-hashes only
    /// while the mixed fleet actually exists.
    pub fn is_bimodal(&self) -> bool {
        derive(&self.modes.lock()).is_some_and(|(_, bimodal)| bimodal)
    }

    /// The derived primary hashing mode: whether query hashing should use
    /// the bigram variant ([`super::hash::compute_block_hashes_bigram`]).
    ///
    /// This accessor collapses an empty fleet into `false`; routing-time
    /// callers must use [`Self::fleet_config`] so "no worker registered"
    /// cannot be mistaken for an established unigram fleet. Test-only so
    /// that distinction stays structural rather than documented.
    #[cfg(test)]
    pub fn is_bigram(&self) -> bool {
        derive(&self.modes.lock()).is_some_and(|(primary, _)| primary)
    }

    /// Routing-time fleet view, derived under one modes-map lock:
    /// `(block_size, primary, bimodal)`. Selection reads all three, so it
    /// must take them atomically — separate [`Self::hash_config`] +
    /// [`Self::is_bimodal`] calls can straddle a worker arriving or draining
    /// and pair a stale primary with a fresh bimodal bit.
    pub fn fleet_config(&self) -> Option<(u32, bool, bool)> {
        let (primary, bimodal) = derive(&self.modes.lock())?;
        let block_size = self.get()?;
        Some((block_size, primary, bimodal))
    }

    /// `(block_size, primary)` half of [`Self::fleet_config`], for the
    /// snapshot-stamping path, which has no use for the bimodal bit.
    /// Returning `None` in the registration window prevents a transient
    /// unigram lookup against an EAGLE tree.
    pub fn hash_config(&self) -> Option<(u32, bool)> {
        self.fleet_config()
            .map(|(size, primary, _)| (size, primary))
    }

    /// Publish a candidate block size. Returns the established value on
    /// success (idempotent: same candidate as already set is `Ok`);
    /// returns `Err(BlockSizeMismatch)` when the candidate disagrees.
    ///
    /// `candidate == 0` is rejected because 0 is reserved as the "not
    /// yet known" sentinel.
    pub fn try_set(&self, candidate: u32) -> Result<u32, BlockSizeMismatch> {
        if candidate == 0 {
            return Err(BlockSizeMismatch {
                established: self.value.load(Ordering::Relaxed),
                candidate,
            });
        }
        match self
            .value
            .compare_exchange(0, candidate, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => Ok(candidate),
            Err(existing) if existing == candidate => Ok(existing),
            Err(existing) => Err(BlockSizeMismatch {
                established: existing,
                candidate,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_oracle_returns_none() {
        let oracle = BlockSizeOracle::new();
        assert_eq!(oracle.get(), None);
        assert_eq!(oracle.hash_config(), None);
    }

    #[test]
    fn first_set_establishes_the_value() {
        let oracle = BlockSizeOracle::new();
        assert_eq!(oracle.try_set(64), Ok(64));
        assert_eq!(oracle.get(), Some(64));
    }

    #[test]
    fn matching_set_is_idempotent() {
        let oracle = BlockSizeOracle::new();
        assert_eq!(oracle.try_set(64), Ok(64));
        assert_eq!(oracle.try_set(64), Ok(64));
        assert_eq!(oracle.try_set(64), Ok(64));
        assert_eq!(oracle.get(), Some(64));
    }

    #[test]
    fn empty_fleet_reads_as_unknown_not_unigram() {
        let oracle = BlockSizeOracle::new();
        assert!(!oracle.is_bigram());
        assert!(!oracle.is_bimodal());
        assert_eq!(oracle.hash_config(), None, "no workers, no hash config");
    }

    #[test]
    fn primary_is_derived_from_the_live_set() {
        let oracle = BlockSizeOracle::new();
        oracle.report_worker("http://w1:30000", true);
        oracle.report_worker("http://w2:30000", true);
        assert!(oracle.is_bigram());
        assert!(!oracle.is_bimodal(), "a single mode is not bimodal");
        // A minority of opposite-mode workers makes the fleet bimodal but
        // does not flip the majority-derived primary.
        oracle.report_worker("http://w3:30000", false);
        assert!(oracle.is_bimodal());
        assert!(oracle.is_bigram(), "2 bigram vs 1 unigram: bigram majority");
        // Majority flips with the live set.
        oracle.report_worker("http://w4:30000", false);
        oracle.report_worker("http://w5:30000", false);
        assert!(oracle.is_bimodal());
        assert!(
            !oracle.is_bigram(),
            "3 unigram vs 2 bigram: unigram majority"
        );
        // Forgetting the minority heals the fleet back to unimodal.
        oracle.forget_worker("http://w1:30000");
        oracle.forget_worker("http://w2:30000");
        assert!(!oracle.is_bigram());
        assert!(
            !oracle.is_bimodal(),
            "once only one mode remains live the fleet is unimodal again"
        );
    }

    #[test]
    fn equal_split_breaks_toward_unigram() {
        let oracle = BlockSizeOracle::new();
        oracle.report_worker("http://w1:30000", true);
        oracle.report_worker("http://w2:30000", false);
        assert!(oracle.is_bimodal());
        assert!(
            !oracle.is_bigram(),
            "the tie-break is deterministic so replicas holding the same live \
             set derive the same primary"
        );
    }

    #[test]
    fn re_adding_a_worker_overwrites_its_mode() {
        let oracle = BlockSizeOracle::new();
        oracle.report_worker("http://w1:30000", false);
        // Same URL returning with a new speculative-decoding config.
        oracle.report_worker("http://w1:30000", true);
        assert!(oracle.is_bigram());
        assert!(
            !oracle.is_bimodal(),
            "a re-added worker has exactly one vote"
        );
    }

    #[test]
    fn vote_of_distinguishes_unvoted_from_unigram() {
        let oracle = BlockSizeOracle::new();
        assert_eq!(oracle.vote_of("http://w1:30000"), None);
        oracle.report_worker("http://w1:30000", false);
        assert_eq!(oracle.vote_of("http://w1:30000"), Some(false));
    }

    #[test]
    fn forget_is_idempotent_and_unknown_urls_are_ignored() {
        let oracle = BlockSizeOracle::new();
        oracle.forget_worker("http://never:30000");
        oracle.report_worker("http://w1:30000", true);
        oracle.forget_worker("http://w1:30000");
        oracle.forget_worker("http://w1:30000");
        assert_eq!(oracle.hash_config(), None, "the fleet is empty again");
    }

    #[test]
    fn hash_config_requires_both_worker_properties() {
        let oracle = BlockSizeOracle::new();
        assert_eq!(oracle.hash_config(), None);
        oracle.try_set(64).unwrap();
        assert_eq!(
            oracle.hash_config(),
            None,
            "a block size alone must not transiently imply unigram hashing",
        );
        oracle.report_worker("http://w1:30000", true);
        assert_eq!(oracle.hash_config(), Some((64, true)));
    }

    #[test]
    fn mismatching_set_fails_without_changing_state() {
        let oracle = BlockSizeOracle::new();
        oracle.try_set(64).unwrap();
        assert_eq!(
            oracle.try_set(128),
            Err(BlockSizeMismatch {
                established: 64,
                candidate: 128
            })
        );
        assert_eq!(
            oracle.get(),
            Some(64),
            "mismatched candidate must not overwrite established value"
        );
    }

    #[test]
    fn zero_candidate_is_rejected() {
        let oracle = BlockSizeOracle::new();
        assert!(oracle.try_set(0).is_err());
        assert_eq!(oracle.get(), None);
    }
}
