// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Process-local storage backend for the KV Indexer.
//!
//! The complete placement view lives behind one [`RwLock`], making an apply batch
//! atomic and every query a consistent snapshot. The state is soft: not shared
//! with another server, and lost when the process exits.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use tonic::Status;

use crate::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, ExternalKvActionType,
    ExternalKvNodeMatch, GetExternalKvHitCountsRequest, GetExternalKvHitCountsResponse,
    HitCountEntry, MatchExternalKvPrefixRequest, MatchExternalKvPrefixResponse,
    MatchExternalKvRequest, MatchExternalKvResponse, TierHashes, WorkerCacheSpec,
};
use crate::service::{assemble_prefix_response, prefix_limit, WorkerPrefixScanner};
use crate::{BlockComponents, KvIndexerBackend, WorkerPrefixInput};

#[derive(Debug, Default)]
struct BlockRecord {
    /// Shared block token count. A zero value means legacy/unspecified.
    token_count: u32,
    /// Resident component snapshot for each `(worker, tier)`.
    placements: HashMap<(String, i32), u32>,
}

#[derive(Debug, Default)]
struct WorkerRecord {
    address: String,
    spec: Option<WorkerCacheSpec>,
    /// Reverse index used by CLEAR_ALL_AT_TIER.
    holdings: HashMap<i32, HashSet<i64>>,
}

#[derive(Debug, Default)]
struct State {
    blocks: HashMap<i64, BlockRecord>,
    workers: HashMap<String, WorkerRecord>,
    hit_counts: HashMap<i64, u64>,
}

struct WorkerView {
    worker_id: String,
    address: String,
    spec: Option<WorkerCacheSpec>,
    hashes_by_tier: BTreeMap<i32, Vec<(i64, u32, u32)>>,
    blocks: Vec<Option<BlockComponents>>,
}

#[derive(Debug)]
struct PrefixCandidate {
    worker_id: String,
    address: String,
    scanner: WorkerPrefixScanner,
}

/// Single-process, soft-state KV placement index.
#[derive(Debug, Default)]
pub struct InMemoryKvIndexerBackend {
    state: RwLock<State>,
}

impl InMemoryKvIndexerBackend {
    pub fn new() -> Self {
        Self::default()
    }

    fn read_state(&self) -> Result<RwLockReadGuard<'_, State>, Status> {
        self.state
            .read()
            .map_err(|_| Status::internal("in-memory backend lock poisoned"))
    }

    fn write_state(&self) -> Result<RwLockWriteGuard<'_, State>, Status> {
        self.state
            .write()
            .map_err(|_| Status::internal("in-memory backend lock poisoned"))
    }

    fn apply(
        &self,
        req: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status> {
        let mut state = self.write_state()?;
        let worker_id = req.worker_id;

        // Address and spec are snapshots carried on every batch. Empty address
        // makes the worker unroutable; absent spec returns it to legacy mode.
        {
            let worker = state.workers.entry(worker_id.clone()).or_default();
            worker.address = req.worker_address;
            worker.spec = req.cache_spec;
        }

        for action in req.actions {
            match ExternalKvActionType::try_from(action.r#type) {
                Ok(ExternalKvActionType::ActionReport) => {
                    let has_masks = !action.component_masks.is_empty();
                    let has_sizes = !action.block_sizes.is_empty();

                    // REPORT is a REPLACE snapshot. Keep the final occurrence
                    // when a coalesced action repeats one hash.
                    let mut last_by_hash: HashMap<i64, (u32, u32)> = HashMap::new();
                    for (index, hash) in action.hashes.into_iter().enumerate() {
                        let mask = if has_masks {
                            action.component_masks[index]
                        } else {
                            0
                        };
                        let token_count = if has_sizes {
                            action.block_sizes[index]
                        } else {
                            0
                        };
                        last_by_hash.insert(hash, (mask, token_count));
                    }

                    for (hash, (mask, token_count)) in last_by_hash {
                        let block = state.blocks.entry(hash).or_default();
                        block
                            .placements
                            .insert((worker_id.clone(), action.tier), mask);
                        // A legacy report carries no size, so 0 means
                        // "unknown" and must not erase a known count.
                        if token_count > 0 {
                            block.token_count = token_count;
                        }
                        state
                            .workers
                            .entry(worker_id.clone())
                            .or_default()
                            .holdings
                            .entry(action.tier)
                            .or_default()
                            .insert(hash);
                    }
                }
                Ok(ExternalKvActionType::ActionRevoke) => {
                    for hash in action.hashes {
                        revoke_one(&mut state, &worker_id, &hash, action.tier);
                    }
                }
                Ok(ExternalKvActionType::ActionClearAllAtTier) => {
                    let hashes = state
                        .workers
                        .get(&worker_id)
                        .and_then(|worker| worker.holdings.get(&action.tier))
                        .cloned()
                        .unwrap_or_default();
                    for hash in hashes {
                        revoke_one(&mut state, &worker_id, &hash, action.tier);
                    }
                }
                Ok(ExternalKvActionType::ActionUnknown) | Err(_) => {
                    return Err(Status::invalid_argument("unsupported action type"));
                }
            }
        }

        Ok(ApplyExternalKvBatchResponse {})
    }

    fn do_match(&self, req: MatchExternalKvRequest) -> Result<MatchExternalKvResponse, Status> {
        let hashes = dedup_preserve_order(&req.hashes);
        let workers = if req.count_as_hit {
            let mut state = self.write_state()?;
            let (workers, matched_hashes) = Self::collect_worker_views(&state, &hashes, false);
            for hash in matched_hashes {
                let count = state.hit_counts.entry(hash).or_default();
                *count = count.saturating_add(1);
            }
            workers
        } else {
            let state = self.read_state()?;
            Self::collect_worker_views(&state, &hashes, false).0
        };
        let matches = workers
            .into_iter()
            .map(|worker| ExternalKvNodeMatch {
                worker_id: worker.worker_id,
                address: worker.address,
                hashes_by_tier: worker
                    .hashes_by_tier
                    .into_iter()
                    .map(|(tier, placements)| TierHashes {
                        tier,
                        hashes: placements.iter().map(|(hash, _, _)| *hash).collect(),
                        component_masks: placements.iter().map(|(_, mask, _)| *mask).collect(),
                        block_sizes: placements
                            .into_iter()
                            .map(|(_, _, block_size)| block_size)
                            .collect(),
                    })
                    .collect(),
            })
            .collect();

        Ok(MatchExternalKvResponse { matches })
    }

    fn collect_worker_views(
        state: &State,
        hashes: &[i64],
        with_blocks: bool,
    ) -> (Vec<WorkerView>, Vec<i64>) {
        // Keyed by a borrow of the stored worker id, so it is copied once per
        // worker in the result rather than once per scanned placement.
        let mut worker_order: Vec<&str> = Vec::new();
        let mut by_worker: HashMap<&str, WorkerView> = HashMap::new();
        let mut matched_hashes = Vec::new();

        for (index, hash) in hashes.iter().enumerate() {
            let Some(block) = state.blocks.get(hash) else {
                continue;
            };
            if block.placements.is_empty() {
                continue;
            }
            matched_hashes.push(*hash);
            for ((worker, tier), mask) in &block.placements {
                let view = by_worker.entry(worker.as_str()).or_insert_with(|| {
                    worker_order.push(worker.as_str());
                    let metadata = state.workers.get(worker);
                    WorkerView {
                        worker_id: worker.clone(),
                        address: metadata
                            .map(|worker| worker.address.clone())
                            .unwrap_or_default(),
                        spec: metadata.and_then(|worker| worker.spec),
                        hashes_by_tier: BTreeMap::new(),
                        blocks: if with_blocks {
                            vec![None; hashes.len()]
                        } else {
                            Vec::new()
                        },
                    }
                });
                if with_blocks {
                    let components = view.blocks[index].get_or_insert_with(|| BlockComponents {
                        token_count: block.token_count,
                        tier_masks: Vec::new(),
                    });
                    components.tier_masks.push((*tier, *mask));
                } else {
                    view.hashes_by_tier.entry(*tier).or_default().push((
                        *hash,
                        *mask,
                        block.token_count,
                    ));
                }
            }
        }

        let workers = worker_order
            .into_iter()
            .filter_map(|worker| by_worker.remove(worker))
            .collect();
        (workers, matched_hashes)
    }

    fn collect_prefix_inputs_locked(state: &State, hashes: &[i64]) -> Vec<WorkerPrefixInput> {
        Self::collect_worker_views(state, hashes, true)
            .0
            .into_iter()
            .map(|worker| WorkerPrefixInput {
                worker_id: worker.worker_id,
                address: worker.address,
                spec: worker.spec,
                blocks: worker.blocks,
            })
            .collect()
    }

    fn do_match_prefix(
        &self,
        req: MatchExternalKvPrefixRequest,
    ) -> Result<MatchExternalKvPrefixResponse, Status> {
        let limit = prefix_limit(req.hashes.len(), req.max_blocks);
        let hashes = &req.hashes[..limit];
        if hashes.is_empty() {
            return Ok(MatchExternalKvPrefixResponse::default());
        }

        // Only workers holding block zero can own a non-empty prefix, so the
        // candidate set is fixed up front and each candidate reuses one scanner and
        // one block view. Allocation is O(first-block holders) whatever the request
        // length, which is why there is no scan cap: length costs time, not memory.
        let state = self.read_state()?;
        let Some(first) = state
            .blocks
            .get(&hashes[0])
            .filter(|block| !block.placements.is_empty())
        else {
            return Ok(MatchExternalKvPrefixResponse {
                matches: Vec::new(),
                best_prefix_blocks: 0,
                blocks_read: 1,
            });
        };
        let mut seen = HashSet::new();
        let mut candidates: Vec<PrefixCandidate> = first
            .placements
            .keys()
            .filter(|(worker, _)| seen.insert(worker.as_str()))
            .map(|(worker, _)| {
                let metadata = state.workers.get(worker);
                PrefixCandidate {
                    address: metadata
                        .map(|worker| worker.address.clone())
                        .unwrap_or_default(),
                    scanner: WorkerPrefixScanner::new(
                        metadata.and_then(|worker| worker.spec.as_ref()),
                    ),
                    worker_id: worker.clone(),
                }
            })
            .collect();
        let candidate_by_id: HashMap<String, usize> = candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| (candidate.worker_id.clone(), index))
            .collect();
        let mut present = vec![false; candidates.len()];
        let mut block_views: Vec<BlockComponents> = (0..candidates.len())
            .map(|_| BlockComponents {
                token_count: 0,
                tier_masks: Vec::new(),
            })
            .collect();

        for hash in hashes {
            present.fill(false);
            for block in &mut block_views {
                block.token_count = 0;
                block.tier_masks.clear();
            }
            if let Some(block) = state.blocks.get(hash) {
                for ((worker, tier), mask) in &block.placements {
                    let Some(&index) = candidate_by_id.get(worker) else {
                        continue;
                    };
                    present[index] = true;
                    block_views[index].token_count = block.token_count;
                    block_views[index].tier_masks.push((*tier, *mask));
                }
            }
            for (index, candidate) in candidates.iter_mut().enumerate() {
                candidate
                    .scanner
                    .push(present[index].then_some(&block_views[index]));
            }
        }

        let entries = candidates
            .into_iter()
            .filter_map(|candidate| {
                let prefix = candidate.scanner.prefix();
                (!candidate.address.is_empty() && prefix > 0).then_some((
                    candidate.worker_id,
                    candidate.address,
                    prefix,
                ))
            })
            .collect();
        Ok(assemble_prefix_response(entries, limit as u32))
    }

    fn do_hit_counts(
        &self,
        req: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        let state = self.read_state()?;
        let entries = dedup_preserve_order(&req.hashes)
            .into_iter()
            .filter_map(|hash| {
                state
                    .hit_counts
                    .get(&hash)
                    .copied()
                    .map(|hit_count_total| HitCountEntry {
                        hash,
                        hit_count_total,
                    })
            })
            .collect();
        Ok(GetExternalKvHitCountsResponse { entries })
    }
}

fn revoke_one(state: &mut State, worker_id: &str, hash: &i64, tier: i32) {
    let mut remove_block = false;
    if let Some(block) = state.blocks.get_mut(hash) {
        block.placements.remove(&(worker_id.to_string(), tier));
        remove_block = block.placements.is_empty();
    }

    if let Some(worker) = state.workers.get_mut(worker_id) {
        if let Some(hashes) = worker.holdings.get_mut(&tier) {
            hashes.remove(hash);
            if hashes.is_empty() {
                worker.holdings.remove(&tier);
            }
        }
    }

    if remove_block {
        state.blocks.remove(hash);
        state.hit_counts.remove(hash);
    }
}

fn dedup_preserve_order(hashes: &[i64]) -> Vec<i64> {
    let mut seen = HashSet::new();
    hashes
        .iter()
        .filter(|hash| seen.insert(**hash))
        .copied()
        .collect()
}

#[tonic::async_trait]
impl KvIndexerBackend for InMemoryKvIndexerBackend {
    async fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status> {
        self.apply(request)
    }

    async fn match_external_kv(
        &self,
        request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        self.do_match(request)
    }

    async fn collect_worker_prefix_inputs(
        &self,
        hashes: &[i64],
    ) -> Result<Vec<WorkerPrefixInput>, Status> {
        let state = self.read_state()?;
        Ok(Self::collect_prefix_inputs_locked(&state, hashes))
    }

    async fn match_external_kv_prefix(
        &self,
        request: MatchExternalKvPrefixRequest,
    ) -> Result<MatchExternalKvPrefixResponse, Status> {
        self.do_match_prefix(request)
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        self.do_hit_counts(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    #[test]
    fn dedup_keeps_first_seen_order() {
        let hashes = vec![1, -2, 1, 3];
        assert_eq!(dedup_preserve_order(&hashes), vec![1, -2, 3]);
    }

    #[test]
    fn match_without_hit_count_uses_shared_lock() {
        let backend = Arc::new(InMemoryKvIndexerBackend::new());
        let read_guard = backend.read_state().unwrap();
        let query_backend = Arc::clone(&backend);
        let (result_tx, result_rx) = mpsc::channel();

        let query = std::thread::spawn(move || {
            result_tx
                .send(query_backend.do_match(MatchExternalKvRequest {
                    hashes: vec![-1],
                    count_as_hit: false,
                }))
                .unwrap();
        });

        result_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("read-only match should not wait for another reader")
            .unwrap();
        drop(read_guard);
        query.join().unwrap();
    }
}
