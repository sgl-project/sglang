// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Process-local storage backend for the KV Indexer.
//!
//! The complete placement view lives behind one [`RwLock`], making an apply batch
//! atomic and every query a consistent snapshot. The state is soft: not shared
//! with another server, and lost when the process exits.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use tonic::Status;

use crate::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, ExternalKvAction,
    ExternalKvActionType, ExternalKvNodeMatch, GetExternalKvHitCountsRequest,
    GetExternalKvHitCountsResponse, HitCountEntry, MatchExternalKvPrefixRequest,
    MatchExternalKvPrefixResponse, MatchExternalKvRequest, MatchExternalKvResponse, TierHashes,
    TierType, WorkerCacheSpec,
};
use crate::service::{assemble_prefix_response, prefix_limit, WorkerPrefixScanner, COMPONENT_FULL};
use crate::{BlockComponents, KvIndexerBackend, WorkerPrefixInput};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum ParentLink {
    #[default]
    Unknown,
    Root,
    Hash(i64),
}

#[derive(Debug, Default)]
struct BlockRecord {
    /// Chain relationship reported by the worker. Prefix-derived state is valid
    /// only along links rooted at `Root`.
    parent: ParentLink,
    children: HashSet<i64>,
    /// Shared block token count. A zero value means legacy/unspecified.
    token_count: u32,
    /// Resident component snapshot for each `(worker, tier)`.
    placements: HashMap<(String, i32), u32>,
    /// Workers for which the root-to-this-block prefix is complete and this
    /// boundary is servable by the Legacy/FULL-only fast path.
    prefix_complete_workers: HashSet<String>,
}

#[derive(Debug, Default)]
struct WorkerRecord {
    address: String,
    spec: Option<WorkerCacheSpec>,
    /// Reverse index used by CLEAR_ALL_AT_TIER.
    holdings: HashMap<i32, HashSet<i64>>,
    /// Number of non-legacy component placements. A spec-less worker can use
    /// the derived fast path only while this is zero.
    component_placement_count: usize,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FastPathKind {
    Legacy,
    FullOnly { tier_mask: u32 },
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

        validate_actions(&state, &req.actions)?;

        let previous_fast_path = state.workers.get(&worker_id).and_then(fast_path_kind);

        // Address and spec are snapshots carried on every batch. Empty address
        // makes the worker unroutable; absent spec returns it to legacy mode.
        {
            let worker = state.workers.entry(worker_id.clone()).or_default();
            worker.address = req.worker_address;
            worker.spec = req.cache_spec;
        }

        let mut dirty_roots = Vec::new();
        let mut reported_chains = Vec::new();
        let mut revoked_hashes = Vec::new();
        // Only a fast-path worker identity change requires a full recompute.
        // REPORT, REVOKE, and CLEAR enqueue their affected hashes directly.
        let mut recompute_from_graph_roots = false;
        for action in req.actions {
            match ExternalKvActionType::try_from(action.r#type) {
                Ok(ExternalKvActionType::ActionReport) => {
                    let has_masks = !action.component_masks.is_empty();
                    let has_sizes = !action.block_sizes.is_empty();
                    let hashes = action.hashes;
                    apply_report_chain(&mut state, action.parent_block_hash, &hashes);

                    for (index, hash) in hashes.iter().copied().enumerate() {
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
                        let old_mask = state
                            .blocks
                            .entry(hash)
                            .or_default()
                            .placements
                            .insert((worker_id.clone(), action.tier), mask);
                        adjust_component_placement_count(
                            &mut state,
                            &worker_id,
                            old_mask,
                            Some(mask),
                        );
                        // A legacy report carries no size, so 0 means
                        // "unknown" and must not erase a known count.
                        if token_count > 0 {
                            state.blocks.entry(hash).or_default().token_count = token_count;
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
                    if !hashes.is_empty() {
                        reported_chains.push(hashes);
                    }
                }
                Ok(ExternalKvActionType::ActionRevoke) => {
                    for hash in action.hashes {
                        revoke_one(&mut state, &worker_id, &hash, action.tier);
                        dirty_roots.push(hash);
                        revoked_hashes.push(hash);
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
                        dirty_roots.push(hash);
                        revoked_hashes.push(hash);
                    }
                }
                Ok(ExternalKvActionType::ActionUnknown) | Err(_) => {
                    return Err(Status::invalid_argument("unsupported action type"));
                }
            }
        }

        let current_fast_path = state.workers.get(&worker_id).and_then(fast_path_kind);
        if previous_fast_path != current_fast_path {
            for block in state.blocks.values_mut() {
                block.prefix_complete_workers.remove(&worker_id);
            }
            recompute_from_graph_roots = true;
        }
        if recompute_from_graph_roots {
            dirty_roots = state
                .blocks
                .iter()
                .filter_map(|(hash, block)| (block.parent == ParentLink::Root).then_some(*hash))
                .collect();
        } else {
            for hashes in reported_chains {
                dirty_roots.extend(refresh_linear_report_chain_prefix_completeness(
                    &mut state, &worker_id, &hashes,
                ));
            }
        }
        recompute_worker_subtrees(&mut state, &worker_id, dirty_roots);
        for hash in revoked_hashes {
            prune_empty_leaf(&mut state, hash);
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
        let known_prefix_len = known_request_prefix_len(&state, hashes).unwrap_or(0);
        let fast_worker_ids: HashSet<&str> = if known_prefix_len > 0 {
            first
                .prefix_complete_workers
                .iter()
                .filter_map(|worker_id| {
                    state.workers.get(worker_id).and_then(|worker| {
                        (fast_path_kind(worker).is_some() && !worker.address.is_empty())
                            .then_some(worker_id.as_str())
                    })
                })
                .collect()
        } else {
            HashSet::new()
        };
        let mut seen = HashSet::new();
        let mut candidates: Vec<PrefixCandidate> = first
            .placements
            .keys()
            .filter(|(worker, _)| {
                !fast_worker_ids.contains(worker.as_str()) && seen.insert(worker.as_str())
            })
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
        let mut entries = Vec::with_capacity(fast_worker_ids.len() + candidates.len());
        let mut unresolved = fast_worker_ids;
        for (index, hash) in hashes[..known_prefix_len].iter().enumerate().rev() {
            if unresolved.is_empty() {
                break;
            }
            if let Some(block) = state.blocks.get(hash) {
                for worker_id in &block.prefix_complete_workers {
                    if !unresolved.remove(worker_id.as_str()) {
                        continue;
                    }
                    if let Some(worker) = state.workers.get(worker_id) {
                        entries.push((
                            worker_id.clone(),
                            worker.address.clone(),
                            (index + 1) as u32,
                        ));
                    }
                }
            }
        }

        if !candidates.is_empty() {
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
        }

        entries.extend(candidates.into_iter().filter_map(|candidate| {
            let prefix = candidate.scanner.prefix();
            (!candidate.address.is_empty() && prefix > 0).then_some((
                candidate.worker_id,
                candidate.address,
                prefix,
            ))
        }));
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

fn fast_path_kind(worker: &WorkerRecord) -> Option<FastPathKind> {
    match worker.spec.as_ref() {
        None if worker.component_placement_count == 0 => Some(FastPathKind::Legacy),
        Some(spec) if spec.version <= 1 && spec.components == COMPONENT_FULL => {
            Some(FastPathKind::FullOnly {
                tier_mask: spec.full_tier_mask,
            })
        }
        _ => None,
    }
}

fn tier_in_mask(mask: u32, tier: i32) -> bool {
    tier >= 0 && mask & (1u32 << tier) != 0
}

fn block_servable(state: &State, hash: i64, worker_id: &str, kind: FastPathKind) -> bool {
    match kind {
        // A globally component-free legacy worker only needs membership, which
        // its reverse holdings index answers without scanning every other
        // worker placed on this popular block.
        FastPathKind::Legacy => state.workers.get(worker_id).is_some_and(|worker| {
            worker
                .holdings
                .values()
                .any(|hashes| hashes.contains(&hash))
        }),
        FastPathKind::FullOnly { tier_mask } => {
            let indexer_tiers =
                (1 << (TierType::TierHbm as u32)) | (1 << (TierType::TierDram as u32));
            state.blocks.get(&hash).is_some_and(|block| {
                block.placements.iter().any(|((worker, tier), mask)| {
                    worker == worker_id
                        && mask & COMPONENT_FULL != 0
                        && tier_in_mask(indexer_tiers & tier_mask, *tier)
                })
            })
        }
    }
}

#[cfg(test)]
fn link_report_chain(
    state: &mut State,
    parent_block_hash: Option<i64>,
    hashes: &[i64],
) -> Result<(), Status> {
    let mut planned_parents = HashMap::with_capacity(hashes.len());
    validate_report_chain(state, &mut planned_parents, parent_block_hash, hashes)?;
    validate_parent_graph_acyclic(state, &planned_parents)?;

    apply_report_chain(state, parent_block_hash, hashes);
    Ok(())
}

fn apply_report_chain(state: &mut State, parent_block_hash: Option<i64>, hashes: &[i64]) {
    let mut parent = parent_block_hash.map_or(ParentLink::Root, ParentLink::Hash);
    for hash in hashes {
        if let ParentLink::Hash(parent_hash) = parent {
            state
                .blocks
                .entry(parent_hash)
                .or_default()
                .children
                .insert(*hash);
        }
        state.blocks.entry(*hash).or_default().parent = parent;
        parent = ParentLink::Hash(*hash);
    }
}

fn validate_actions(state: &State, actions: &[ExternalKvAction]) -> Result<(), Status> {
    let mut planned_parents = HashMap::new();
    for action in actions {
        match ExternalKvActionType::try_from(action.r#type) {
            Ok(ExternalKvActionType::ActionReport) => validate_report_chain(
                state,
                &mut planned_parents,
                action.parent_block_hash,
                &action.hashes,
            )?,
            Ok(ExternalKvActionType::ActionRevoke)
            | Ok(ExternalKvActionType::ActionClearAllAtTier) => {}
            Ok(ExternalKvActionType::ActionUnknown) | Err(_) => {
                return Err(Status::invalid_argument("unsupported action type"));
            }
        }
    }
    validate_parent_graph_acyclic(state, &planned_parents)
}

fn validate_report_chain(
    state: &State,
    planned_parents: &mut HashMap<i64, ParentLink>,
    parent_block_hash: Option<i64>,
    hashes: &[i64],
) -> Result<(), Status> {
    let mut parent = parent_block_hash.map_or(ParentLink::Root, ParentLink::Hash);
    for hash in hashes {
        if parent == ParentLink::Hash(*hash) {
            return Err(Status::invalid_argument(
                "block hash cannot be its own parent",
            ));
        }
        let existing = planned_parents
            .get(hash)
            .copied()
            .or_else(|| state.blocks.get(hash).map(|block| block.parent))
            .unwrap_or_default();
        if existing != ParentLink::Unknown && existing != parent {
            return Err(Status::invalid_argument(format!(
                "block hash {hash} was reported with conflicting parents"
            )));
        }
        planned_parents.insert(*hash, parent);
        parent = ParentLink::Hash(*hash);
    }
    Ok(())
}

fn validate_parent_graph_acyclic(
    state: &State,
    planned_parents: &HashMap<i64, ParentLink>,
) -> Result<(), Status> {
    let mut complete = HashSet::new();
    for start in planned_parents.keys().copied() {
        if complete.contains(&start) {
            continue;
        }

        let mut path = Vec::new();
        let mut on_path = HashSet::new();
        let mut current = start;
        loop {
            if complete.contains(&current) {
                break;
            }
            if !on_path.insert(current) {
                return Err(Status::invalid_argument(
                    "report would create a parent cycle",
                ));
            }
            path.push(current);

            let parent = planned_parents
                .get(&current)
                .copied()
                .or_else(|| state.blocks.get(&current).map(|block| block.parent))
                .unwrap_or_default();
            match parent {
                ParentLink::Hash(parent) => current = parent,
                ParentLink::Unknown | ParentLink::Root => break,
            }
        }
        complete.extend(path);
    }
    Ok(())
}

fn adjust_component_placement_count(
    state: &mut State,
    worker_id: &str,
    old_mask: Option<u32>,
    new_mask: Option<u32>,
) {
    let worker = state.workers.entry(worker_id.to_string()).or_default();
    if old_mask.is_some_and(|mask| mask != 0) {
        worker.component_placement_count = worker.component_placement_count.saturating_sub(1);
    }
    if new_mask.is_some_and(|mask| mask != 0) {
        worker.component_placement_count = worker.component_placement_count.saturating_add(1);
    }
}

fn recompute_worker_subtrees(
    state: &mut State,
    worker_id: &str,
    roots: impl IntoIterator<Item = i64>,
) {
    let kind = state.workers.get(worker_id).and_then(fast_path_kind);
    let mut queue: VecDeque<i64> = roots.into_iter().collect();
    let mut visited = HashSet::new();
    while let Some(hash) = queue.pop_front() {
        if !visited.insert(hash) {
            continue;
        }
        let Some(block) = state.blocks.get(&hash) else {
            continue;
        };
        let parent_complete = match block.parent {
            ParentLink::Unknown => false,
            ParentLink::Root => true,
            ParentLink::Hash(parent) => state
                .blocks
                .get(&parent)
                .is_some_and(|parent| parent.prefix_complete_workers.contains(worker_id)),
        };
        let complete = kind
            .is_some_and(|kind| parent_complete && block_servable(state, hash, worker_id, kind));
        let children: Vec<i64> = block.children.iter().copied().collect();
        let block = state.blocks.get_mut(&hash).expect("block exists");
        if complete {
            block.prefix_complete_workers.insert(worker_id.to_string());
        } else {
            block.prefix_complete_workers.remove(worker_id);
        }
        queue.extend(children);
    }
}

/// Returns direct children held by this worker but outside the current REPORT chain.
fn external_children_held_by_worker(
    state: &State,
    worker_id: &str,
    reported_hashes: &HashSet<i64>,
    parent: i64,
) -> Vec<i64> {
    state
        .blocks
        .get(&parent)
        .into_iter()
        .flat_map(|block| block.children.iter().copied())
        .filter(|child| {
            !reported_hashes.contains(child)
                && state.blocks.get(child).is_some_and(|child| {
                    child
                        .placements
                        .keys()
                        .any(|(worker, _)| worker == worker_id)
                })
        })
        .collect()
}

/// Refreshes derived prefix state along a closed linear REPORT chain.
///
/// The caller has verified that no node in the chain has an external child.
fn refresh_linear_report_chain_prefix_completeness(
    state: &mut State,
    worker_id: &str,
    hashes: &[i64],
) -> Vec<i64> {
    let kind = state.workers.get(worker_id).and_then(fast_path_kind);
    let reported_hashes: HashSet<i64> = hashes.iter().copied().collect();
    let mut external_dirty_roots = Vec::new();
    let mut parent_complete = hashes
        .first()
        .and_then(|hash| state.blocks.get(hash))
        .is_some_and(|block| match block.parent {
            ParentLink::Root => true,
            ParentLink::Hash(parent) => state
                .blocks
                .get(&parent)
                .is_some_and(|parent| parent.prefix_complete_workers.contains(worker_id)),
            ParentLink::Unknown => false,
        });

    for hash in hashes {
        let was_complete = state
            .blocks
            .get(hash)
            .is_some_and(|block| block.prefix_complete_workers.contains(worker_id));
        let complete = kind
            .is_some_and(|kind| parent_complete && block_servable(state, *hash, worker_id, kind));
        if was_complete != complete {
            external_dirty_roots.extend(external_children_held_by_worker(
                state,
                worker_id,
                &reported_hashes,
                *hash,
            ));
        }
        let Some(block) = state.blocks.get_mut(hash) else {
            continue;
        };
        if complete {
            block.prefix_complete_workers.insert(worker_id.to_string());
        } else {
            block.prefix_complete_workers.remove(worker_id);
        }
        parent_complete = complete;
    }
    external_dirty_roots
}

/// Returns the length of the longest leading request chain already known to the
/// Indexer. A missing block starts the normal uncached suffix; a present block
/// with the wrong parent is a chain conflict and disables the derived fast path.
fn known_request_prefix_len(state: &State, hashes: &[i64]) -> Option<usize> {
    let mut known = 0;
    for (index, hash) in hashes.iter().enumerate() {
        let expected = if index == 0 {
            ParentLink::Root
        } else {
            ParentLink::Hash(hashes[index - 1])
        };
        let Some(block) = state.blocks.get(hash) else {
            break;
        };
        if block.parent != expected {
            return None;
        }
        known += 1;
    }
    Some(known)
}

fn revoke_one(state: &mut State, worker_id: &str, hash: &i64, tier: i32) {
    let mut removed_mask = None;
    if let Some(block) = state.blocks.get_mut(hash) {
        removed_mask = block.placements.remove(&(worker_id.to_string(), tier));
    }
    adjust_component_placement_count(state, worker_id, removed_mask, None);

    if let Some(worker) = state.workers.get_mut(worker_id) {
        if let Some(hashes) = worker.holdings.get_mut(&tier) {
            hashes.remove(hash);
            if hashes.is_empty() {
                worker.holdings.remove(&tier);
            }
        }
    }

    if state
        .blocks
        .get(hash)
        .is_some_and(|block| block.placements.is_empty())
    {
        state.hit_counts.remove(hash);
    }
}

fn prune_empty_leaf(state: &mut State, mut hash: i64) {
    loop {
        let Some(block) = state.blocks.get(&hash) else {
            return;
        };
        if !block.placements.is_empty()
            || !block.children.is_empty()
            || !block.prefix_complete_workers.is_empty()
        {
            return;
        }
        let parent = block.parent;
        state.blocks.remove(&hash);
        let ParentLink::Hash(parent_hash) = parent else {
            return;
        };
        if let Some(parent) = state.blocks.get_mut(&parent_hash) {
            parent.children.remove(&hash);
        }
        hash = parent_hash;
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

    #[test]
    fn known_request_prefix_stops_at_uncached_suffix_and_rejects_conflicts() {
        let mut state = State::default();
        link_report_chain(&mut state, None, &[1, 2, 3]).unwrap();
        link_report_chain(&mut state, None, &[9]).unwrap();

        assert_eq!(known_request_prefix_len(&state, &[1, 2, 3, 4, 5]), Some(3));
        assert_eq!(known_request_prefix_len(&state, &[1, 9]), None);
    }
    #[test]
    fn conflicting_report_chain_does_not_mutate_the_graph() {
        let mut state = State::default();

        let error = link_report_chain(&mut state, None, &[1, 2, 1]).unwrap_err();

        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        assert!(state.blocks.is_empty());
    }

    #[test]
    fn cyclic_report_chain_does_not_mutate_the_graph() {
        let mut state = State::default();

        let error = link_report_chain(&mut state, Some(2), &[1, 2]).unwrap_err();

        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        assert!(state.blocks.is_empty());
    }

    #[test]
    fn cyclic_report_chain_through_existing_graph_is_rejected() {
        let mut state = State::default();
        link_report_chain(&mut state, Some(2), &[1]).unwrap();

        let error = link_report_chain(&mut state, Some(1), &[2]).unwrap_err();

        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        assert_eq!(state.blocks[&1].parent, ParentLink::Hash(2));
        assert_eq!(state.blocks[&2].parent, ParentLink::Unknown);
    }

    #[test]
    fn external_children_only_include_the_reporting_workers_branch() {
        let mut state = State::default();

        link_report_chain(&mut state, None, &[1, 2, 3]).unwrap();
        link_report_chain(&mut state, Some(1), &[4]).unwrap();
        state
            .blocks
            .get_mut(&4)
            .unwrap()
            .placements
            .insert(("worker-b".into(), TierType::TierHbm as i32), 0);
        let reported_hashes: HashSet<i64> = [1, 2, 3].into_iter().collect();
        assert!(
            external_children_held_by_worker(&state, "worker-a", &reported_hashes, 1,).is_empty()
        );

        state
            .blocks
            .get_mut(&4)
            .unwrap()
            .placements
            .insert(("worker-a".into(), TierType::TierHbm as i32), 0);
        assert_eq!(
            external_children_held_by_worker(&state, "worker-a", &reported_hashes, 1,),
            vec![4]
        );
    }
}
