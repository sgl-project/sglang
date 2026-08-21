// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Process-local storage backend for the KV Indexer.
//!
//! The complete placement view lives behind one [`RwLock`], making an apply batch
//! atomic and every query a consistent snapshot. The state is soft: not shared
//! with another server, and lost when the process exits.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use tonic::Status;

use crate::pb::{
    AbortExternalKvSnapshotRequest, AppendExternalKvSnapshotRequest,
    AppendExternalKvSnapshotResponse, ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse,
    BeginExternalKvSnapshotRequest, BeginExternalKvSnapshotResponse,
    CommitExternalKvSnapshotRequest, CommitExternalKvSnapshotResponse,
    ConfigureExpectedWorkersRequest, ConfigureExpectedWorkersResponse, ExternalKvActionType,
    ExternalKvNodeMatch, ExternalKvPrefixMatch, ExternalKvSnapshotMetadata,
    ExternalKvSnapshotPlacement, GetExternalKvHitCountsRequest, GetExternalKvHitCountsResponse,
    HitCountEntry, InvalidateWorkerRequest, InvalidateWorkerResponse, MatchExternalKvPrefixRequest,
    MatchExternalKvPrefixResponse, MatchExternalKvRequest, MatchExternalKvResponse,
    ReplaceExternalKvSnapshotRequest, ReplaceExternalKvSnapshotResponse, StreamCoverage,
    TierHashes, WorkerCacheSpec,
};
use crate::service::{prefix_limit, WorkerPrefixScanner};
use crate::status::{IndexerStatusHandle, IndexerStreamStatus};
use crate::stream::StreamKey;
use crate::{BlockComponents, KvIndexerBackend, WorkerPrefixInput};

#[derive(Debug, Default)]
struct BlockRecord {
    /// Shared block token count. A zero value means legacy/unspecified.
    token_count: u32,
    /// Resident component snapshot for each `(stream, tier)`.
    placements: HashMap<(StreamKey, i32), u32>,
}

#[derive(Debug, Default)]
struct WorkerRecord {
    address: String,
    spec: Option<WorkerCacheSpec>,
    ready: bool,
    epoch: Option<String>,
    last_seq: Option<u64>,
    worker_generation: String,
    model: String,
    hash_schema_version: u32,
    page_size: u32,
    is_bigram: bool,
    /// Reverse index used by CLEAR_ALL_AT_TIER.
    holdings: HashMap<i32, HashSet<i64>>,
}

#[derive(Debug, Default)]
struct State {
    blocks: HashMap<i64, BlockRecord>,
    workers: HashMap<StreamKey, WorkerRecord>,
    hit_counts: HashMap<i64, u64>,
    snapshot_staging: HashMap<String, SnapshotStaging>,
}

#[derive(Debug)]
struct SnapshotStaging {
    stream: StreamKey,
    worker_address: String,
    worker_epoch: String,
    applied_seq: u64,
    cache_spec: Option<WorkerCacheSpec>,
    metadata: ExternalKvSnapshotMetadata,
    snapshot_id: String,
    expected_placements: u64,
    placements: Vec<ExternalKvSnapshotPlacement>,
}

struct WorkerView {
    stream: StreamKey,
    address: String,
    spec: Option<WorkerCacheSpec>,
    hashes_by_tier: BTreeMap<i32, Vec<(i64, u32, u32)>>,
    blocks: Vec<Option<BlockComponents>>,
}

#[derive(Debug)]
struct PrefixCandidate {
    stream: StreamKey,
    address: String,
    scanner: WorkerPrefixScanner,
}

/// Single-process, soft-state KV placement index.
#[derive(Debug)]
pub struct InMemoryKvIndexerBackend {
    state: RwLock<State>,
    status: Option<Arc<IndexerStatusHandle>>,
    instance_epoch: String,
}

impl Default for InMemoryKvIndexerBackend {
    fn default() -> Self {
        Self {
            state: RwLock::new(State::default()),
            status: None,
            instance_epoch: uuid::Uuid::new_v4().to_string(),
        }
    }
}

impl InMemoryKvIndexerBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_status(status: Arc<IndexerStatusHandle>) -> Self {
        Self {
            state: RwLock::new(State::default()),
            status: Some(status),
            instance_epoch: uuid::Uuid::new_v4().to_string(),
        }
    }

    fn refresh_status(&self, state: &State) {
        if let Some(status) = &self.status {
            let mut streams: Vec<_> = state
                .workers
                .iter()
                .map(|(stream, worker)| IndexerStreamStatus {
                    namespace: stream.namespace.clone(),
                    worker_id: stream.worker_id.clone(),
                    dp_rank: stream.dp_rank,
                    worker_address: worker.address.clone(),
                    ready: worker.ready,
                    worker_epoch: worker.epoch.clone().unwrap_or_default(),
                    watermark: worker.last_seq.unwrap_or_default(),
                    worker_generation: worker.worker_generation.clone(),
                })
                .collect();
            streams.sort_by(|left, right| {
                (&left.namespace, &left.worker_id, left.dp_rank).cmp(&(
                    &right.namespace,
                    &right.worker_id,
                    right.dp_rank,
                ))
            });
            status.set_stream_coverage(streams);
        }
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
        let ApplyExternalKvBatchRequest {
            worker_id,
            seq,
            actions,
            worker_address,
            cache_spec,
            worker_epoch,
            enforce_sequence,
            stream_id,
            worker_generation,
        } = req;
        let stream = StreamKey::from_wire(stream_id, &worker_id)?;

        if enforce_sequence {
            let Some(worker) = state.workers.get(&stream) else {
                return Err(Status::failed_precondition(
                    "stream must be configured before recovery-aware apply",
                ));
            };
            if !worker.ready
                || worker.epoch.as_deref() != Some(worker_epoch.as_str())
                || worker.worker_generation != worker_generation
            {
                return Err(Status::failed_precondition(
                    "stream has no READY snapshot for this generation and epoch",
                ));
            }
            let last = worker.last_seq.unwrap_or_default();
            if seq <= last {
                return Ok(ApplyExternalKvBatchResponse {
                    applied_seq: last,
                    duplicate: true,
                });
            }
            if seq != last.saturating_add(1) {
                clear_worker_holdings(&mut state, &stream);
                if let Some(worker) = state.workers.get_mut(&stream) {
                    worker.ready = false;
                    worker.epoch = None;
                    worker.last_seq = None;
                }
                self.refresh_status(&state);
                return Err(Status::failed_precondition(format!(
                    "event sequence gap: expected {}, got {seq}",
                    last.saturating_add(1)
                )));
            }
        }

        // Address and spec are snapshots carried on every batch. Empty address
        // makes the worker unroutable; absent spec returns it to legacy mode.
        {
            let worker = state.workers.entry(stream.clone()).or_default();
            worker.address = worker_address;
            worker.spec = cache_spec;
            if !worker_generation.is_empty() {
                worker.worker_generation = worker_generation.clone();
            }
        }

        for action in actions {
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
                        block.placements.insert((stream.clone(), action.tier), mask);
                        // A legacy report carries no size, so 0 means
                        // "unknown" and must not erase a known count.
                        if token_count > 0 {
                            block.token_count = token_count;
                        }
                        state
                            .workers
                            .entry(stream.clone())
                            .or_default()
                            .holdings
                            .entry(action.tier)
                            .or_default()
                            .insert(hash);
                    }
                }
                Ok(ExternalKvActionType::ActionRevoke) => {
                    for hash in action.hashes {
                        revoke_one(&mut state, &stream, &hash, action.tier);
                    }
                }
                Ok(ExternalKvActionType::ActionClearAllAtTier) => {
                    let hashes = state
                        .workers
                        .get(&stream)
                        .and_then(|worker| worker.holdings.get(&action.tier))
                        .cloned()
                        .unwrap_or_default();
                    for hash in hashes {
                        revoke_one(&mut state, &stream, &hash, action.tier);
                    }
                }
                Ok(ExternalKvActionType::ActionUnknown) | Err(_) => {
                    return Err(Status::invalid_argument("unsupported action type"));
                }
            }
        }

        if let Some(worker) = state.workers.get_mut(&stream) {
            worker.ready = true;
            if enforce_sequence {
                worker.epoch = Some(worker_epoch);
                worker.last_seq = Some(seq);
            }
        }
        self.refresh_status(&state);
        Ok(ApplyExternalKvBatchResponse {
            applied_seq: seq,
            duplicate: false,
        })
    }

    fn configure_workers(
        &self,
        req: ConfigureExpectedWorkersRequest,
    ) -> Result<ConfigureExpectedWorkersResponse, Status> {
        let mut state = self.write_state()?;
        let mut configured = Vec::with_capacity(req.workers.len());
        for worker in req.workers {
            let stream = StreamKey::from_wire(worker.stream_id.clone(), &worker.worker_id)?;
            configured.push((stream, worker));
        }
        let expected: HashSet<StreamKey> = configured
            .iter()
            .map(|(stream, _)| stream.clone())
            .collect();
        let removed: Vec<StreamKey> = state
            .workers
            .keys()
            .filter(|worker| !expected.contains(*worker))
            .cloned()
            .collect();
        for stream in removed {
            clear_worker_holdings(&mut state, &stream);
            state.workers.remove(&stream);
        }
        state
            .snapshot_staging
            .retain(|_, staging| expected.contains(&staging.stream));
        for (stream, expected) in configured {
            let worker = state.workers.entry(stream).or_default();
            worker.address = expected.worker_address;
            // Bridge configuration is the desired topology. Empty dynamic
            // fields mean "learn from Snapshot v2", not "erase the recovered
            // value". This method is also the idle Indexer-epoch heartbeat,
            // so blindly assigning None/empty every second would clear the
            // generation and component spec of an otherwise READY stream.
            if expected.cache_spec.is_some() {
                worker.spec = expected.cache_spec;
            }
            if !expected.worker_generation.is_empty() {
                worker.worker_generation = expected.worker_generation;
            }
            if !expected.model.is_empty() {
                worker.model = expected.model;
            }
            if expected.hash_schema_version != 0 {
                worker.hash_schema_version = expected.hash_schema_version;
                worker.is_bigram = expected.is_bigram;
            }
            if expected.page_size != 0 {
                worker.page_size = expected.page_size;
            }
        }
        self.refresh_status(&state);
        Ok(coverage_response(&state, &self.instance_epoch))
    }

    fn replace_snapshot(
        &self,
        req: ReplaceExternalKvSnapshotRequest,
    ) -> Result<ReplaceExternalKvSnapshotResponse, Status> {
        let mut state = self.write_state()?;
        let stream = StreamKey::from_wire(req.stream_id.clone(), &req.worker_id)?;
        if !state.workers.contains_key(&stream) {
            return Err(Status::failed_precondition(
                "stream must be configured before snapshot replacement",
            ));
        }
        clear_worker_holdings(&mut state, &stream);
        for tier in req.hashes_by_tier {
            let has_masks = !tier.component_masks.is_empty();
            let has_sizes = !tier.block_sizes.is_empty();
            for (index, hash) in tier.hashes.into_iter().enumerate() {
                let mask = if has_masks {
                    tier.component_masks[index]
                } else {
                    0
                };
                let token_count = if has_sizes {
                    tier.block_sizes[index]
                } else {
                    0
                };
                let block = state.blocks.entry(hash).or_default();
                block.placements.insert((stream.clone(), tier.tier), mask);
                if token_count > 0 {
                    block.token_count = token_count;
                }
                state
                    .workers
                    .get_mut(&stream)
                    .expect("configured stream exists")
                    .holdings
                    .entry(tier.tier)
                    .or_default()
                    .insert(hash);
            }
        }
        let worker = state
            .workers
            .get_mut(&stream)
            .expect("configured stream exists");
        worker.address = req.worker_address;
        worker.spec = req.cache_spec;
        worker.ready = true;
        worker.epoch = Some(req.worker_epoch);
        worker.last_seq = Some(req.applied_seq);
        if !req.worker_generation.is_empty() {
            worker.worker_generation = req.worker_generation;
        }
        self.refresh_status(&state);
        Ok(ReplaceExternalKvSnapshotResponse {
            applied_seq: req.applied_seq,
        })
    }

    fn invalidate(&self, req: InvalidateWorkerRequest) -> Result<InvalidateWorkerResponse, Status> {
        let mut state = self.write_state()?;
        let stream = StreamKey::from_wire(req.stream_id, &req.worker_id)?;
        if !state.workers.contains_key(&stream) {
            return Err(Status::not_found("stream is not configured"));
        }
        clear_worker_holdings(&mut state, &stream);
        state
            .snapshot_staging
            .retain(|_, staging| staging.stream != stream);
        if let Some(worker) = state.workers.get_mut(&stream) {
            worker.ready = false;
            worker.epoch = None;
            worker.last_seq = None;
            // A Worker process restart legitimately changes generation. The
            // next Snapshot v2 establishes the new value; an operator-pinned
            // expected generation is reapplied by ConfigureExpectedWorkers
            // before Begin and still fences mismatches.
            worker.worker_generation.clear();
        }
        self.refresh_status(&state);
        let coverage = coverage_response(&state, &self.instance_epoch);
        Ok(InvalidateWorkerResponse {
            total_workers: coverage.total_workers,
            ready_workers: coverage.ready_workers,
            indexer_epoch: coverage.indexer_epoch,
        })
    }

    fn begin_snapshot(
        &self,
        req: BeginExternalKvSnapshotRequest,
    ) -> Result<BeginExternalKvSnapshotResponse, Status> {
        let mut state = self.write_state()?;
        let stream = StreamKey::from_wire(req.stream_id, "")?;
        let metadata = req
            .metadata
            .ok_or_else(|| Status::invalid_argument("snapshot metadata is required"))?;
        let worker = state
            .workers
            .get(&stream)
            .ok_or_else(|| Status::failed_precondition("stream must be configured first"))?;
        validate_snapshot_metadata(worker, &metadata)?;

        // Only one transaction per stream is useful. Replacing an abandoned
        // staging area is safe because staging is never query-visible.
        state
            .snapshot_staging
            .retain(|_, staging| staging.stream != stream);
        let transaction_id = uuid::Uuid::new_v4().to_string();
        state.snapshot_staging.insert(
            transaction_id.clone(),
            SnapshotStaging {
                stream,
                worker_address: req.worker_address,
                worker_epoch: req.worker_epoch,
                applied_seq: req.applied_seq,
                cache_spec: req.cache_spec,
                metadata,
                snapshot_id: req.snapshot_id,
                expected_placements: req.expected_placements,
                placements: Vec::new(),
            },
        );
        Ok(BeginExternalKvSnapshotResponse {
            transaction_id,
            indexer_epoch: self.instance_epoch.clone(),
        })
    }

    fn append_snapshot(
        &self,
        req: AppendExternalKvSnapshotRequest,
    ) -> Result<AppendExternalKvSnapshotResponse, Status> {
        let mut state = self.write_state()?;
        let staging = state
            .snapshot_staging
            .get_mut(&req.transaction_id)
            .ok_or_else(|| Status::not_found("snapshot transaction not found"))?;
        let next = staging
            .placements
            .len()
            .checked_add(req.placements.len())
            .ok_or_else(|| Status::resource_exhausted("snapshot placement count overflow"))?;
        if next as u64 > staging.expected_placements {
            return Err(Status::invalid_argument(
                "snapshot append exceeds declared placement count",
            ));
        }
        staging.placements.extend(req.placements);
        Ok(AppendExternalKvSnapshotResponse {
            staged_placements: next as u64,
        })
    }

    fn commit_snapshot(
        &self,
        req: CommitExternalKvSnapshotRequest,
    ) -> Result<CommitExternalKvSnapshotResponse, Status> {
        let mut state = self.write_state()?;
        let staging = state
            .snapshot_staging
            .remove(&req.transaction_id)
            .ok_or_else(|| Status::not_found("snapshot transaction not found"))?;
        if staging.placements.len() as u64 != staging.expected_placements {
            return Err(Status::failed_precondition(format!(
                "snapshot transaction {} expected {} placements but staged {}",
                staging.snapshot_id,
                staging.expected_placements,
                staging.placements.len()
            )));
        }
        if !state.workers.contains_key(&staging.stream) {
            return Err(Status::failed_precondition(
                "stream was removed before snapshot commit",
            ));
        }

        let mut identities = HashSet::with_capacity(staging.placements.len());
        for placement in &staging.placements {
            if !identities.insert((placement.block_hash, placement.tier)) {
                return Err(Status::invalid_argument(
                    "snapshot contains duplicate (block_hash, tier) placement",
                ));
            }
        }

        clear_worker_holdings(&mut state, &staging.stream);
        for placement in staging.placements {
            let block = state.blocks.entry(placement.block_hash).or_default();
            block.placements.insert(
                (staging.stream.clone(), placement.tier),
                placement.component_mask,
            );
            if placement.block_size > 0 {
                block.token_count = placement.block_size;
            }
            state
                .workers
                .get_mut(&staging.stream)
                .expect("configured stream exists")
                .holdings
                .entry(placement.tier)
                .or_default()
                .insert(placement.block_hash);
        }

        let worker = state
            .workers
            .get_mut(&staging.stream)
            .expect("configured stream exists");
        worker.address = staging.worker_address;
        worker.spec = staging.cache_spec;
        worker.ready = true;
        worker.epoch = Some(staging.worker_epoch);
        worker.last_seq = Some(staging.applied_seq);
        worker.worker_generation = staging.metadata.worker_generation;
        worker.model = staging.metadata.model;
        worker.hash_schema_version = staging.metadata.hash_schema_version;
        worker.page_size = staging.metadata.page_size;
        worker.is_bigram = staging.metadata.is_bigram;
        self.refresh_status(&state);
        Ok(CommitExternalKvSnapshotResponse {
            applied_seq: staging.applied_seq,
        })
    }

    fn abort_snapshot(&self, req: AbortExternalKvSnapshotRequest) -> Result<(), Status> {
        let mut state = self.write_state()?;
        state.snapshot_staging.remove(&req.transaction_id);
        Ok(())
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
                worker_id: worker.stream.worker_id,
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
        let mut worker_order: Vec<&StreamKey> = Vec::new();
        let mut by_worker: HashMap<&StreamKey, WorkerView> = HashMap::new();
        let mut matched_hashes = Vec::new();

        for (index, hash) in hashes.iter().enumerate() {
            let Some(block) = state.blocks.get(hash) else {
                continue;
            };
            if block.placements.is_empty() {
                continue;
            }
            matched_hashes.push(*hash);
            for ((stream, tier), mask) in &block.placements {
                let view = by_worker.entry(stream).or_insert_with(|| {
                    worker_order.push(stream);
                    let metadata = state.workers.get(stream);
                    WorkerView {
                        stream: stream.clone(),
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
                worker_id: worker.stream.worker_id,
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
        let state = self.read_state()?;
        let (coverage, eligible_streams, uncovered_worker_addresses, complete_coverage) =
            select_coverage(&state, &req)?;
        if hashes.is_empty() {
            return Ok(MatchExternalKvPrefixResponse {
                matches: Vec::new(),
                best_prefix_blocks: 0,
                blocks_read: 0,
                coverage,
                complete_coverage,
                uncovered_worker_addresses,
            });
        }

        // Only workers holding block zero can own a non-empty prefix, so the
        // candidate set is fixed up front and each candidate reuses one scanner and
        // one block view. Allocation is O(first-block holders) whatever the request
        // length, which is why there is no scan cap: length costs time, not memory.
        let Some(first) = state
            .blocks
            .get(&hashes[0])
            .filter(|block| !block.placements.is_empty())
        else {
            return Ok(MatchExternalKvPrefixResponse {
                matches: Vec::new(),
                best_prefix_blocks: 0,
                blocks_read: 1,
                coverage,
                complete_coverage,
                uncovered_worker_addresses,
            });
        };
        let mut seen = HashSet::new();
        let mut candidates: Vec<PrefixCandidate> = first
            .placements
            .keys()
            .filter(|(stream, _)| eligible_streams.contains(stream))
            .filter(|(stream, _)| seen.insert((*stream).clone()))
            .filter_map(|(stream, _)| {
                let metadata = state.workers.get(stream)?;
                if !metadata.ready {
                    return None;
                }
                Some(PrefixCandidate {
                    address: metadata.address.clone(),
                    scanner: WorkerPrefixScanner::new(metadata.spec.as_ref()),
                    stream: stream.clone(),
                })
            })
            .collect();
        let candidate_by_id: HashMap<StreamKey, usize> = candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| (candidate.stream.clone(), index))
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
                for ((stream, tier), mask) in &block.placements {
                    let Some(&index) = candidate_by_id.get(stream) else {
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

        let mut matches: Vec<ExternalKvPrefixMatch> = candidates
            .into_iter()
            .filter_map(|candidate| {
                let prefix = candidate.scanner.prefix();
                let metadata = state.workers.get(&candidate.stream)?;
                (!candidate.address.is_empty() && prefix > 0).then_some(ExternalKvPrefixMatch {
                    worker_address: candidate.address,
                    matched_prefix_blocks: prefix,
                    worker_id: candidate.stream.worker_id.clone(),
                    stream_id: Some(candidate.stream.to_wire()),
                    worker_generation: metadata.worker_generation.clone(),
                })
            })
            .collect();
        matches.sort_by_key(|entry| std::cmp::Reverse(entry.matched_prefix_blocks));
        let best_prefix_blocks = matches
            .first()
            .map(|entry| entry.matched_prefix_blocks)
            .unwrap_or(0);
        Ok(MatchExternalKvPrefixResponse {
            matches,
            best_prefix_blocks,
            blocks_read: limit as u32,
            coverage,
            complete_coverage,
            uncovered_worker_addresses,
        })
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

fn coverage_response(state: &State, indexer_epoch: &str) -> ConfigureExpectedWorkersResponse {
    ConfigureExpectedWorkersResponse {
        total_workers: state.workers.len() as u32,
        ready_workers: state.workers.values().filter(|worker| worker.ready).count() as u32,
        indexer_epoch: indexer_epoch.to_owned(),
    }
}

fn validate_snapshot_metadata(
    worker: &WorkerRecord,
    metadata: &ExternalKvSnapshotMetadata,
) -> Result<(), Status> {
    let mismatch = (!worker.worker_generation.is_empty()
        && worker.worker_generation != metadata.worker_generation)
        || (!worker.model.is_empty() && worker.model != metadata.model)
        || (worker.hash_schema_version != 0
            && worker.hash_schema_version != metadata.hash_schema_version)
        || (worker.page_size != 0 && worker.page_size != metadata.page_size)
        || (worker.hash_schema_version != 0 && worker.is_bigram != metadata.is_bigram);
    if mismatch {
        return Err(Status::failed_precondition(
            "snapshot metadata does not match configured stream",
        ));
    }
    Ok(())
}

fn select_coverage(
    state: &State,
    request: &MatchExternalKvPrefixRequest,
) -> Result<(Vec<StreamCoverage>, HashSet<StreamKey>, Vec<String>, bool), Status> {
    let mut selected = std::collections::BTreeSet::new();
    let mut missing_streams = Vec::new();
    let mut uncovered_worker_addresses = Vec::new();

    if !request.eligible_streams.is_empty() {
        for stream in &request.eligible_streams {
            let key = StreamKey::from_wire(Some(stream.clone()), "")?;
            if state.workers.contains_key(&key) {
                selected.insert(key);
            } else {
                missing_streams.push(key);
            }
        }
    } else if !request.eligible_worker_addresses.is_empty() {
        for address in dedup_strings(&request.eligible_worker_addresses) {
            let before = selected.len();
            selected.extend(
                state
                    .workers
                    .iter()
                    .filter(|(_, worker)| worker.address == address)
                    .map(|(stream, _)| stream.clone()),
            );
            if selected.len() == before {
                uncovered_worker_addresses.push(address);
            }
        }
    } else {
        selected.extend(state.workers.keys().cloned());
    }

    let eligible_streams: HashSet<_> = selected.iter().cloned().collect();
    let mut coverage = Vec::with_capacity(selected.len() + missing_streams.len());
    for stream in selected {
        let worker = state
            .workers
            .get(&stream)
            .expect("selected stream is configured");
        coverage.push(StreamCoverage {
            stream_id: Some(stream.to_wire()),
            worker_address: worker.address.clone(),
            ready: worker.ready,
            worker_epoch: worker.epoch.clone().unwrap_or_default(),
            watermark: worker.last_seq.unwrap_or_default(),
            worker_generation: worker.worker_generation.clone(),
        });
    }
    coverage.extend(missing_streams.iter().map(|stream| StreamCoverage {
        stream_id: Some(stream.to_wire()),
        worker_address: String::new(),
        ready: false,
        worker_epoch: String::new(),
        watermark: 0,
        worker_generation: String::new(),
    }));
    let complete = !coverage.is_empty()
        && missing_streams.is_empty()
        && uncovered_worker_addresses.is_empty()
        && coverage.iter().all(|entry| entry.ready);
    Ok((
        coverage,
        eligible_streams,
        uncovered_worker_addresses,
        complete,
    ))
}

fn dedup_strings(values: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    values
        .iter()
        .filter(|value| seen.insert(value.as_str()))
        .cloned()
        .collect()
}

fn clear_worker_holdings(state: &mut State, stream: &StreamKey) {
    let holdings = state
        .workers
        .get(stream)
        .map(|worker| worker.holdings.clone())
        .unwrap_or_default();
    for (tier, hashes) in holdings {
        for hash in hashes {
            revoke_one(state, stream, &hash, tier);
        }
    }
}

fn revoke_one(state: &mut State, stream: &StreamKey, hash: &i64, tier: i32) {
    let mut remove_block = false;
    if let Some(block) = state.blocks.get_mut(hash) {
        block.placements.remove(&(stream.clone(), tier));
        remove_block = block.placements.is_empty();
    }

    if let Some(worker) = state.workers.get_mut(stream) {
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
    async fn configure_expected_workers(
        &self,
        request: ConfigureExpectedWorkersRequest,
    ) -> Result<ConfigureExpectedWorkersResponse, Status> {
        self.configure_workers(request)
    }

    async fn replace_external_kv_snapshot(
        &self,
        request: ReplaceExternalKvSnapshotRequest,
    ) -> Result<ReplaceExternalKvSnapshotResponse, Status> {
        self.replace_snapshot(request)
    }

    async fn begin_external_kv_snapshot(
        &self,
        request: BeginExternalKvSnapshotRequest,
    ) -> Result<BeginExternalKvSnapshotResponse, Status> {
        self.begin_snapshot(request)
    }

    async fn append_external_kv_snapshot(
        &self,
        request: AppendExternalKvSnapshotRequest,
    ) -> Result<AppendExternalKvSnapshotResponse, Status> {
        self.append_snapshot(request)
    }

    async fn commit_external_kv_snapshot(
        &self,
        request: CommitExternalKvSnapshotRequest,
    ) -> Result<CommitExternalKvSnapshotResponse, Status> {
        self.commit_snapshot(request)
    }

    async fn abort_external_kv_snapshot(
        &self,
        request: AbortExternalKvSnapshotRequest,
    ) -> Result<(), Status> {
        self.abort_snapshot(request)
    }

    async fn invalidate_worker(
        &self,
        request: InvalidateWorkerRequest,
    ) -> Result<InvalidateWorkerResponse, Status> {
        self.invalidate(request)
    }

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

    fn expected(id: &str, address: &str) -> crate::pb::ExpectedWorker {
        crate::pb::ExpectedWorker {
            worker_id: id.into(),
            worker_address: address.into(),
            cache_spec: None,
            ..Default::default()
        }
    }

    fn expected_stream(id: &str, rank: u32, address: &str) -> crate::pb::ExpectedWorker {
        crate::pb::ExpectedWorker {
            worker_id: id.into(),
            worker_address: address.into(),
            cache_spec: Some(WorkerCacheSpec {
                version: 1,
                components: crate::service::COMPONENT_FULL,
                swa_window_tokens: 0,
                full_tier_mask: 1u32 << crate::pb::TierType::TierHbm as u32,
                swa_tier_mask: 0,
                mamba_tier_mask: 0,
            }),
            stream_id: Some(crate::pb::StreamId {
                namespace: "ns".into(),
                worker_id: id.into(),
                dp_rank: rank,
            }),
            worker_generation: "generation-1".into(),
            model: "model".into(),
            hash_schema_version: 1,
            page_size: 1,
            is_bigram: false,
        }
    }

    fn snapshot(
        worker: &str,
        address: &str,
        epoch: &str,
        seq: u64,
        hashes: &[i64],
    ) -> ReplaceExternalKvSnapshotRequest {
        ReplaceExternalKvSnapshotRequest {
            worker_id: worker.into(),
            worker_address: address.into(),
            worker_epoch: epoch.into(),
            applied_seq: seq,
            hashes_by_tier: vec![TierHashes {
                tier: crate::pb::TierType::TierHbm as i32,
                hashes: hashes.to_vec(),
                component_masks: Vec::new(),
                block_sizes: Vec::new(),
            }],
            cache_spec: None,
            stream_id: None,
            worker_generation: String::new(),
        }
    }

    #[tokio::test]
    async fn configured_workers_become_ready_only_after_atomic_snapshots() {
        let status = Arc::new(IndexerStatusHandle::new(4));
        let backend = InMemoryKvIndexerBackend::with_status(Arc::clone(&status));
        backend
            .configure_expected_workers(ConfigureExpectedWorkersRequest {
                workers: vec![expected("w1", "http://w1"), expected("w2", "http://w2")],
            })
            .await
            .unwrap();
        assert!(!status.report("i".into(), "http://i".into()).ready);

        backend
            .replace_external_kv_snapshot(snapshot("w1", "http://w1", "e1", 2, &[1, 2]))
            .await
            .unwrap();
        assert!(!status.report("i".into(), "http://i".into()).ready);
        backend
            .replace_external_kv_snapshot(snapshot("w2", "http://w2", "e2", 4, &[1]))
            .await
            .unwrap();
        let report = status.report("i".into(), "http://i".into());
        assert!(report.ready);
        assert_eq!((report.ready_workers, report.total_workers), (2, 2));

        let matched = backend
            .match_external_kv_prefix(MatchExternalKvPrefixRequest {
                hashes: vec![1, 2],
                max_blocks: 0,
                eligible_worker_addresses: Vec::new(),
                eligible_streams: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(matched.best_prefix_blocks, 2);
    }

    #[tokio::test]
    async fn fenced_apply_deduplicates_and_gap_invalidates_worker() {
        let status = Arc::new(IndexerStatusHandle::new(4));
        let backend = InMemoryKvIndexerBackend::with_status(Arc::clone(&status));
        backend
            .configure_expected_workers(ConfigureExpectedWorkersRequest {
                workers: vec![expected("w", "http://w")],
            })
            .await
            .unwrap();
        backend
            .replace_external_kv_snapshot(snapshot("w", "http://w", "e", 2, &[1]))
            .await
            .unwrap();
        let request = ApplyExternalKvBatchRequest {
            worker_id: "w".into(),
            seq: 3,
            actions: vec![crate::pb::ExternalKvAction {
                r#type: ExternalKvActionType::ActionReport as i32,
                tier: crate::pb::TierType::TierHbm as i32,
                hashes: vec![2],
                component_masks: Vec::new(),
                block_sizes: Vec::new(),
            }],
            worker_address: "http://w".into(),
            cache_spec: None,
            worker_epoch: "e".into(),
            enforce_sequence: true,
            stream_id: None,
            worker_generation: String::new(),
        };
        assert!(!backend.apply(request.clone()).unwrap().duplicate);
        assert!(backend.apply(request).unwrap().duplicate);

        let gap = ApplyExternalKvBatchRequest {
            worker_id: "w".into(),
            seq: 5,
            actions: Vec::new(),
            worker_address: "http://w".into(),
            cache_spec: None,
            worker_epoch: "e".into(),
            enforce_sequence: true,
            stream_id: None,
            worker_generation: String::new(),
        };
        assert_eq!(
            backend.apply(gap).unwrap_err().code(),
            tonic::Code::FailedPrecondition
        );
        assert!(!status.report("i".into(), "http://i".into()).ready);
        assert!(backend.read_state().unwrap().blocks.is_empty());
    }

    #[tokio::test]
    async fn reconfiguration_removes_scaled_in_worker_holdings() {
        let backend = InMemoryKvIndexerBackend::new();
        backend
            .configure_expected_workers(ConfigureExpectedWorkersRequest {
                workers: vec![expected("w1", "http://w1"), expected("w2", "http://w2")],
            })
            .await
            .unwrap();
        backend
            .replace_external_kv_snapshot(snapshot("w1", "http://w1", "e1", 1, &[1]))
            .await
            .unwrap();
        backend
            .replace_external_kv_snapshot(snapshot("w2", "http://w2", "e2", 1, &[2]))
            .await
            .unwrap();
        backend
            .configure_expected_workers(ConfigureExpectedWorkersRequest {
                workers: vec![expected("w1", "http://w1")],
            })
            .await
            .unwrap();
        let state = backend.read_state().unwrap();
        assert!(state
            .workers
            .contains_key(&StreamKey::new(String::new(), "w1".into(), 0).unwrap()));
        assert!(!state
            .workers
            .contains_key(&StreamKey::new(String::new(), "w2".into(), 0).unwrap()));
        assert!(!state.blocks.contains_key(&2));
    }

    #[tokio::test]
    async fn indexer_epoch_changes_between_process_instances() {
        let first = InMemoryKvIndexerBackend::new()
            .configure_expected_workers(ConfigureExpectedWorkersRequest {
                workers: vec![expected("w", "http://w")],
            })
            .await
            .unwrap();
        let second = InMemoryKvIndexerBackend::new()
            .configure_expected_workers(ConfigureExpectedWorkersRequest {
                workers: vec![expected("w", "http://w")],
            })
            .await
            .unwrap();

        assert!(!first.indexer_epoch.is_empty());
        assert_ne!(first.indexer_epoch, second.indexer_epoch);
    }

    #[tokio::test]
    async fn staged_snapshot_larger_than_single_rpc_is_atomic() {
        const COUNT: usize = crate::service::MAX_HASHES_PER_REQUEST + 1;
        let backend = InMemoryKvIndexerBackend::new();
        backend
            .configure_expected_workers(ConfigureExpectedWorkersRequest {
                workers: vec![expected_stream("worker", 0, "http://worker")],
            })
            .await
            .unwrap();
        let begin = backend
            .begin_external_kv_snapshot(BeginExternalKvSnapshotRequest {
                stream_id: Some(crate::pb::StreamId {
                    namespace: "ns".into(),
                    worker_id: "worker".into(),
                    dp_rank: 0,
                }),
                worker_address: "http://worker".into(),
                worker_epoch: "epoch-1".into(),
                applied_seq: 7,
                cache_spec: expected_stream("worker", 0, "http://worker").cache_spec,
                metadata: Some(ExternalKvSnapshotMetadata {
                    model: "model".into(),
                    worker_generation: "generation-1".into(),
                    hash_schema_version: 1,
                    page_size: 1,
                    is_bigram: false,
                }),
                snapshot_id: "snapshot-1".into(),
                expected_placements: COUNT as u64,
            })
            .await
            .unwrap();
        let placements: Vec<_> = (0..COUNT)
            .map(|hash| ExternalKvSnapshotPlacement {
                block_hash: hash as i64,
                parent_block_hash: hash.checked_sub(1).map(|parent| parent as i64),
                tier: crate::pb::TierType::TierHbm as i32,
                component_mask: crate::service::COMPONENT_FULL,
                block_size: 1,
            })
            .collect();
        for chunk in placements.chunks(crate::service::MAX_HASHES_PER_REQUEST) {
            backend
                .append_external_kv_snapshot(AppendExternalKvSnapshotRequest {
                    transaction_id: begin.transaction_id.clone(),
                    placements: chunk.to_vec(),
                })
                .await
                .unwrap();
        }

        let before = backend
            .match_external_kv_prefix(MatchExternalKvPrefixRequest {
                hashes: vec![0],
                eligible_worker_addresses: vec!["http://worker".into()],
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(!before.complete_coverage);
        assert!(before.matches.is_empty());

        backend
            .commit_external_kv_snapshot(CommitExternalKvSnapshotRequest {
                transaction_id: begin.transaction_id,
            })
            .await
            .unwrap();
        // The Bridge uses this same RPC as an Indexer-process heartbeat.
        // Topology-only configuration must preserve Snapshot-owned fields.
        let mut heartbeat = expected_stream("worker", 0, "http://worker");
        heartbeat.worker_generation.clear();
        heartbeat.cache_spec = None;
        backend
            .configure_expected_workers(ConfigureExpectedWorkersRequest {
                workers: vec![heartbeat],
            })
            .await
            .unwrap();
        let after = backend
            .match_external_kv_prefix(MatchExternalKvPrefixRequest {
                hashes: (0..COUNT).map(|hash| hash as i64).collect(),
                eligible_worker_addresses: vec!["http://worker".into()],
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(after.complete_coverage);
        assert_eq!(after.best_prefix_blocks as usize, COUNT);
        assert_eq!(after.coverage[0].watermark, 7);
        assert_eq!(after.coverage[0].worker_generation, "generation-1");
    }

    #[tokio::test]
    async fn coverage_distinguishes_partial_recovery_from_no_match() {
        let backend = InMemoryKvIndexerBackend::new();
        backend
            .configure_expected_workers(ConfigureExpectedWorkersRequest {
                workers: vec![
                    expected_stream("w1", 0, "http://w1"),
                    expected_stream("w2", 0, "http://w2"),
                ],
            })
            .await
            .unwrap();
        let mut recovered = snapshot("w1", "http://w1", "e1", 1, &[1]);
        recovered.stream_id = Some(crate::pb::StreamId {
            namespace: "ns".into(),
            worker_id: "w1".into(),
            dp_rank: 0,
        });
        recovered.worker_generation = "generation-1".into();
        backend
            .replace_external_kv_snapshot(recovered)
            .await
            .unwrap();

        let response = backend
            .match_external_kv_prefix(MatchExternalKvPrefixRequest {
                hashes: vec![999],
                eligible_worker_addresses: vec!["http://w1".into(), "http://w2".into()],
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(!response.complete_coverage);
        assert!(response.matches.is_empty());
        assert_eq!(
            response.coverage.iter().filter(|entry| entry.ready).count(),
            1
        );
    }

    #[tokio::test]
    async fn same_worker_dp_streams_are_isolated() {
        let backend = InMemoryKvIndexerBackend::new();
        backend
            .configure_expected_workers(ConfigureExpectedWorkersRequest {
                workers: vec![
                    expected_stream("worker", 0, "http://worker"),
                    expected_stream("worker", 1, "http://worker"),
                ],
            })
            .await
            .unwrap();
        for rank in 0..=1 {
            let mut request = snapshot("worker", "http://worker", "epoch", 1, &[1]);
            request.stream_id = Some(crate::pb::StreamId {
                namespace: "ns".into(),
                worker_id: "worker".into(),
                dp_rank: rank,
            });
            request.worker_generation = "generation-1".into();
            backend.replace_external_kv_snapshot(request).await.unwrap();
        }
        backend
            .invalidate_worker(InvalidateWorkerRequest {
                worker_id: "worker".into(),
                stream_id: Some(crate::pb::StreamId {
                    namespace: "ns".into(),
                    worker_id: "worker".into(),
                    dp_rank: 0,
                }),
            })
            .await
            .unwrap();

        let response = backend
            .match_external_kv_prefix(MatchExternalKvPrefixRequest {
                hashes: vec![1],
                eligible_streams: vec![crate::pb::StreamId {
                    namespace: "ns".into(),
                    worker_id: "worker".into(),
                    dp_rank: 1,
                }],
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(response.complete_coverage);
        assert_eq!(response.best_prefix_blocks, 1);
        assert_eq!(response.matches[0].stream_id.as_ref().unwrap().dp_rank, 1);
    }

    #[tokio::test]
    async fn invalidation_allows_snapshot_to_establish_new_worker_generation() {
        let backend = InMemoryKvIndexerBackend::new();
        let mut configured = expected_stream("worker", 0, "http://worker");
        configured.worker_generation.clear();
        backend
            .configure_expected_workers(ConfigureExpectedWorkersRequest {
                workers: vec![configured],
            })
            .await
            .unwrap();
        let mut first = snapshot("worker", "http://worker", "epoch-1", 1, &[1]);
        first.stream_id = Some(crate::pb::StreamId {
            namespace: "ns".into(),
            worker_id: "worker".into(),
            dp_rank: 0,
        });
        first.worker_generation = "generation-1".into();
        backend.replace_external_kv_snapshot(first).await.unwrap();
        backend
            .invalidate_worker(InvalidateWorkerRequest {
                worker_id: "worker".into(),
                stream_id: Some(crate::pb::StreamId {
                    namespace: "ns".into(),
                    worker_id: "worker".into(),
                    dp_rank: 0,
                }),
            })
            .await
            .unwrap();

        backend
            .begin_external_kv_snapshot(BeginExternalKvSnapshotRequest {
                stream_id: Some(crate::pb::StreamId {
                    namespace: "ns".into(),
                    worker_id: "worker".into(),
                    dp_rank: 0,
                }),
                worker_address: "http://worker".into(),
                worker_epoch: "epoch-2".into(),
                applied_seq: 2,
                cache_spec: expected_stream("worker", 0, "http://worker").cache_spec,
                metadata: Some(ExternalKvSnapshotMetadata {
                    model: "model".into(),
                    worker_generation: "generation-2".into(),
                    hash_schema_version: 1,
                    page_size: 1,
                    is_bigram: false,
                }),
                snapshot_id: "snapshot-2".into(),
                expected_placements: 0,
            })
            .await
            .expect("new Worker generation should be accepted after invalidation");
    }
}
