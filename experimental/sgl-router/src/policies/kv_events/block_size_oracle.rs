// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Process-shared per-(cache-aware-zmq) `block_size`, sourced from the
//! workers themselves.
//!
//! # Why an oracle instead of a config field?
//!
//! `compute_block_hashes` must hash with the **same** block size the
//! worker uses to publish KV-cache events; otherwise every cache-aware
//! lookup misses silently. The worker advertises its `page_size` via
//! `/server_info` (parsed into [`crate::policies::kv_events::EventConfig::block_size`]).
//! Earlier versions of sgl-router carried a static `block_size` field on
//! `CacheAwareConfig`; nothing reconciled it with the worker-reported
//! value, so a mismatch silently destroyed cache-hit routing.
//!
//! # Single oracle vs per-model
//!
//! For now the oracle is process-wide. Realistic deployments use one
//! `page_size` across the cluster, so a single value suffices and
//! mismatches across models indicate misconfiguration the operator
//! should see. A per-model oracle would require threading `ModelId`
//! through `KvEventIndex::add_worker`; that refactor can land later
//! without changing the oracle's public surface.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;

/// Tri-state for the bigram flag: distinguishes "not yet reported" from an
/// established `false`, so [`BlockSizeOracle::set_bigram`] can be first-wins
/// (matching `try_set`) rather than last-writer-wins.
const BIGRAM_UNKNOWN: u8 = 0;
const BIGRAM_UNIGRAM: u8 = 1;
const BIGRAM_BIGRAM: u8 = 2;

/// First-wins, idempotent block-size publisher.
///
/// Internally an `AtomicU32` where 0 means "not yet known". Use
/// [`Self::try_set`] to publish a worker-reported value and
/// [`Self::get`] to read at routing time.
///
/// Also carries a `bigram` flag — EAGLE-family workers hash KV blocks over
/// token bigrams, so the policy must pick the bigram hasher. The **primary**
/// mode (`bigram`) is established first-wins, mirroring `try_set`, and is what
/// `hash_config` / `is_bigram` report — the peer-snapshot prewarm protocol
/// vets replicas against it, so it must stay a single stable value.
///
/// A heterogeneous fleet (EAGLE-family + non-EAGLE workers behind one base
/// model — e.g. an EAGLE3 fleet with a non-EAGLE canary) is no longer treated as
/// pure misconfiguration: the second mode's arrival is recorded in
/// `secondary_seen` so [`Self::is_bimodal`] reports it, and the selection path
/// dual-hashes queries (hashing each request both ways and unioning the
/// eligible owners) so neither family loses cache affinity. `block_size` must
/// still agree across the fleet; only the hashing *mode* may differ.
#[derive(Debug, Default)]
pub struct BlockSizeOracle {
    value: AtomicU32,
    bigram: AtomicU8,
    /// Latched `true` once a worker registers with the opposite hashing mode
    /// from the established primary — i.e. the fleet is bimodal.
    secondary_seen: AtomicBool,
}

/// Returned by [`BlockSizeOracle::try_set`] when the candidate disagrees
/// with the already-established value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSizeMismatch {
    pub established: u32,
    pub candidate: u32,
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

    /// Publish whether a registering worker uses bigram (EAGLE-family) KV-block
    /// hashing. Called from `KvEventIndex::add_worker` alongside `try_set`.
    ///
    /// The **primary** mode is first-wins (the first worker establishes what
    /// `hash_config` / `is_bigram` report, which the peer-snapshot protocol
    /// depends on). A later worker of the *opposite* mode no longer just loses:
    /// it latches [`Self::is_bimodal`] so the selection path dual-hashes queries
    /// and both families keep cache affinity. `block_size` must still agree
    /// (enforced separately by [`Self::try_set`]); only the mode may differ.
    pub fn set_bigram(&self, is_bigram: bool) {
        let candidate = if is_bigram {
            BIGRAM_BIGRAM
        } else {
            BIGRAM_UNIGRAM
        };
        match self.bigram.compare_exchange(
            BIGRAM_UNKNOWN,
            candidate,
            Ordering::Release,
            Ordering::Acquire,
        ) {
            Ok(_) => {}
            Err(existing) if existing == candidate => {}
            Err(existing) => {
                // Opposite hashing mode from the established primary. In a mixed
                // EAGLE-family + non-EAGLE fleet behind one base model both modes
                // are legitimately present; record it so selection dual-hashes.
                self.secondary_seen.store(true, Ordering::Release);
                tracing::info!(
                    primary_bigram = existing == BIGRAM_BIGRAM,
                    worker_bigram = is_bigram,
                    "kv-events: worker uses the opposite KV-block hashing mode from the \
                     established primary; fleet is bimodal (EAGLE-family + non-EAGLE). \
                     Cache-aware selection will dual-hash queries so both families keep \
                     cache affinity — ensure all workers serve the same base model and \
                     the same block_size.",
                );
            }
        }
    }

    /// Whether the fleet carries workers of **both** hashing modes (an
    /// EAGLE-family bigram worker and a non-EAGLE unigram worker). Latched
    /// `true` the first time a worker of the non-primary mode registers; never
    /// clears. Read at routing time so `CacheAwareZmqPolicy::select` dual-hashes
    /// only when it must, leaving the uniform-fleet fast path untouched.
    pub fn is_bimodal(&self) -> bool {
        self.secondary_seen.load(Ordering::Acquire)
    }

    /// Whether query hashing should use the bigram variant
    /// ([`super::hash::compute_block_hashes_bigram`]).
    ///
    /// This accessor collapses "not reported" into `false`; routing-time
    /// callers must use [`Self::hash_config`] so unknown mode cannot be
    /// mistaken for established unigram hashing.
    pub fn is_bigram(&self) -> bool {
        self.bigram.load(Ordering::Acquire) == BIGRAM_BIGRAM
    }

    /// Return a coherent block-hashing configuration once both worker
    /// properties have been established.
    ///
    /// `add_worker` publishes `block_size` before `bigram`; the release/acquire
    /// pair on `bigram` makes the preceding size write visible here. Returning
    /// `None` while either half is unknown prevents a transient unigram lookup
    /// against an EAGLE tree during first-worker registration.
    pub fn hash_config(&self) -> Option<(u32, bool)> {
        let bigram = self.bigram.load(Ordering::Acquire);
        if bigram == BIGRAM_UNKNOWN {
            return None;
        }
        let block_size = self.value.load(Ordering::Relaxed);
        if block_size == 0 {
            return None;
        }
        Some((block_size, bigram == BIGRAM_BIGRAM))
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
    fn bigram_flag_defaults_false_first_wins_and_is_idempotent() {
        let oracle = BlockSizeOracle::new();
        assert!(
            !oracle.is_bigram(),
            "unknown (no worker reported yet) reads as non-bigram"
        );
        oracle.set_bigram(true);
        assert!(oracle.is_bigram(), "first worker establishes the mode");
        assert!(!oracle.is_bimodal(), "a single mode is not bimodal");
        oracle.set_bigram(true); // idempotent agreement
        assert!(oracle.is_bigram());
        assert!(!oracle.is_bimodal(), "agreement does not make it bimodal");
        // Independent of block_size establishment.
        assert_eq!(oracle.get(), None);
        // First-wins for the PRIMARY: a conflicting later worker does not flip
        // the reported mode, but it does latch the fleet as bimodal.
        oracle.set_bigram(false);
        assert!(
            oracle.is_bigram(),
            "a disagreeing worker must not flip the established primary mode"
        );
        assert!(
            oracle.is_bimodal(),
            "an opposite-mode worker latches the fleet as bimodal"
        );
    }

    #[test]
    fn bigram_flag_establishes_false_first_wins() {
        let oracle = BlockSizeOracle::new();
        oracle.set_bigram(false);
        assert!(!oracle.is_bigram(), "established as unigram");
        assert!(!oracle.is_bimodal());
        oracle.set_bigram(true); // conflicting; first (unigram) wins, fleet bimodal
        assert!(!oracle.is_bigram());
        assert!(oracle.is_bimodal(), "opposite mode latches bimodal");
    }

    #[test]
    fn is_bimodal_latches_and_never_clears() {
        let oracle = BlockSizeOracle::new();
        assert!(!oracle.is_bimodal(), "fresh oracle is unimodal");
        oracle.set_bigram(true); // primary = bigram
        oracle.set_bigram(false); // opposite → bimodal
        assert!(oracle.is_bimodal());
        // Subsequent agreeing workers of either established mode never clear it.
        oracle.set_bigram(true);
        oracle.set_bigram(false);
        assert!(oracle.is_bimodal(), "bimodal is a latch, not a live count");
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
        oracle.set_bigram(true);
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
