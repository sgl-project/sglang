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
//! Dynamo's design treats `kv_cache_block_size` as a property of the
//! `ModelDeploymentCard` populated by the worker registrar (see
//! `~/dynamo/components/src/dynamo/sglang/register.py`); a mismatch
//! across workers for the same model is rejected loudly
//! (`lib/kv-router/src/standalone_indexer/registry.rs::bail!`). The
//! oracle here is the sgl-router analog — first worker establishes the
//! value, mismatches are refused.
//!
//! # Single oracle vs per-model
//!
//! For now the oracle is process-wide. Realistic deployments use one
//! `page_size` across the cluster, so a single value suffices and
//! mismatches across models indicate misconfiguration the operator
//! should see. A per-model oracle would require threading `ModelId`
//! through `KvEventIndex::add_worker`; that refactor can land later
//! without changing the oracle's public surface.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// First-wins, idempotent block-size publisher.
///
/// Internally an `AtomicU32` where 0 means "not yet known". Use
/// [`Self::try_set`] to publish a worker-reported value and
/// [`Self::get`] to read at routing time.
#[derive(Debug, Default)]
pub struct BlockSizeOracle {
    value: AtomicU32,
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
