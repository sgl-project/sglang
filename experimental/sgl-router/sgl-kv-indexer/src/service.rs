// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use tokio::sync::Semaphore;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

use crate::admission::{reject_if_deadline_passed, RejectionLog};
use crate::pb::kv_indexer_server::{KvIndexer, KvIndexerServer};
use crate::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, ExternalKvAction,
    ExternalKvActionType, ExternalKvPrefixMatch, GetExternalKvHitCountsRequest,
    GetExternalKvHitCountsResponse, MatchExternalKvPrefixRequest, MatchExternalKvPrefixResponse,
    MatchExternalKvRequest, MatchExternalKvResponse, TierType, WorkerCacheSpec,
};

/// Protocol-level resource bounds, enforced before a backend sees the request so
/// no caller can make it allocate work proportional to an unbounded field. The
/// prefix query is exempt from the hash bound; see [`validate_hashes`].
pub(crate) const MAX_HASHES_PER_REQUEST: usize = 16_384;
pub(crate) const MAX_ACTIONS_PER_BATCH: usize = 256;
pub const DEFAULT_PREFIX_QUERY_MAX_INFLIGHT: usize = 32;
/// Maximum encoded gRPC request size accepted by the Indexer server. With
/// packed `sfixed64` hashes this holds roughly one million blocks.
pub const MAX_GRPC_DECODING_MESSAGE_SIZE: usize = 8 * 1024 * 1024;
/// Per-connection bound on concurrently served HTTP/2 streams. Decoding happens
/// in tonic's codec before a method body runs, so `prefix_query_max_inflight`
/// bounds only the scan, not the bytes a peer makes the server buffer — left
/// unset, one connection can hold an unbounded number of
/// [`MAX_GRPC_DECODING_MESSAGE_SIZE`] messages at once. Sized well above the
/// router's own default of 32 in-flight queries so it never throttles a healthy
/// caller.
pub const MAX_CONCURRENT_STREAMS: u32 = 64;

static OVERLOAD_LOG: RejectionLog = RejectionLog::new();

/// Storage backend for the indexer. Every mutation flows through
/// `apply_external_kv_batch`, preserving one ordered write path.
///
/// Async so a backend that does IO fits without reshaping the trait, and
/// dyn-safe so the server can hold it as `Arc<dyn KvIndexerBackend>`.
#[tonic::async_trait]
pub trait KvIndexerBackend: Send + Sync + 'static {
    /// Applies a whole SGLang KVEventBatch. The actions are pre-validated and
    /// must be applied in order. Applies are unconditional: the request `seq` is
    /// informational only and a redelivered batch is applied again.
    async fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status>;

    async fn match_external_kv(
        &self,
        request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status>;

    /// Collects the per-worker, per-block component placement needed to compute a
    /// prefix, aligned with `hashes`.
    ///
    /// The default implementation is component-blind: every held block becomes a
    /// legacy whole-block placement. Component-aware backends override it to
    /// attach each worker's `WorkerCacheSpec` and the resident component set.
    async fn collect_worker_prefix_inputs(
        &self,
        hashes: &[i64],
    ) -> Result<Vec<WorkerPrefixInput>, Status> {
        let matched = self
            .match_external_kv(MatchExternalKvRequest {
                hashes: hashes.to_vec(),
                count_as_hit: false,
            })
            .await?;
        Ok(legacy_inputs_from_match(hashes, &matched))
    }

    /// Answers, per worker, the longest reusable request prefix it holds.
    ///
    /// This default implementation *is* the written definition of the prefix
    /// semantics, so a backend that overrides it for performance must stay
    /// field-for-field identical except for `blocks_read`, which is
    /// observability rather than semantics.
    ///
    /// The result is a safe lower bound: every required component's rule is
    /// applied, so an accurate index can only under-report, never over-report.
    async fn match_external_kv_prefix(
        &self,
        request: MatchExternalKvPrefixRequest,
    ) -> Result<MatchExternalKvPrefixResponse, Status> {
        let limit = prefix_limit(request.hashes.len(), request.max_blocks);
        let hashes: Vec<i64> = request.hashes.into_iter().take(limit).collect();
        if hashes.is_empty() {
            return Ok(MatchExternalKvPrefixResponse::default());
        }
        // The default path reads placement for every considered block.
        let inputs = self.collect_worker_prefix_inputs(&hashes).await?;
        Ok(compute_prefix_response(&inputs, hashes.len() as u32))
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status>;
}

/// Blanket impl so the server can hold the selected backend as
/// `Arc<dyn KvIndexerBackend>` and still satisfy `KvIndexerService<B>`.
#[tonic::async_trait]
impl KvIndexerBackend for std::sync::Arc<dyn KvIndexerBackend> {
    async fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status> {
        (**self).apply_external_kv_batch(request).await
    }

    async fn match_external_kv(
        &self,
        request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        (**self).match_external_kv(request).await
    }

    async fn collect_worker_prefix_inputs(
        &self,
        hashes: &[i64],
    ) -> Result<Vec<WorkerPrefixInput>, Status> {
        (**self).collect_worker_prefix_inputs(hashes).await
    }

    async fn match_external_kv_prefix(
        &self,
        request: MatchExternalKvPrefixRequest,
    ) -> Result<MatchExternalKvPrefixResponse, Status> {
        (**self).match_external_kv_prefix(request).await
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        (**self).get_external_kv_hit_counts(request).await
    }
}

#[derive(Debug)]
pub struct KvIndexerService<B> {
    backend: B,
    prefix_query_semaphore: Semaphore,
}

impl<B> KvIndexerService<B>
where
    B: KvIndexerBackend,
{
    pub fn new(backend: B) -> Self {
        Self::with_prefix_query_max_inflight(backend, DEFAULT_PREFIX_QUERY_MAX_INFLIGHT)
    }

    pub fn with_prefix_query_max_inflight(backend: B, max_inflight: usize) -> Self {
        assert!(
            max_inflight > 0,
            "prefix query max inflight must be greater than zero"
        );
        Self {
            backend,
            prefix_query_semaphore: Semaphore::new(max_inflight),
        }
    }

    /// Wraps the service in its generated server with the decoding limit a
    /// full-length prefix query needs. Constructing the server any other way
    /// silently reinstates tonic's 4 MiB default, so production and tests that
    /// exercise large requests go through here.
    pub fn into_server(self) -> KvIndexerServer<Self> {
        KvIndexerServer::new(self).max_decoding_message_size(MAX_GRPC_DECODING_MESSAGE_SIZE)
    }
}

/// A transport builder carrying the Indexer's stream bound. Pairs with
/// [`KvIndexerService::into_server`]: that sets the per-message ceiling, this
/// bounds how many messages can be in flight against it at once.
pub fn server_builder() -> Server {
    Server::builder().max_concurrent_streams(MAX_CONCURRENT_STREAMS)
}

#[tonic::async_trait]
impl<B> KvIndexer for KvIndexerService<B>
where
    B: KvIndexerBackend,
{
    async fn match_external_kv(
        &self,
        request: Request<MatchExternalKvRequest>,
    ) -> Result<Response<MatchExternalKvResponse>, Status> {
        let request = request.into_inner();
        validate_hashes_bounded(&request.hashes)?;
        let response = self.backend.match_external_kv(request).await?;
        Ok(Response::new(response))
    }

    async fn match_external_kv_prefix(
        &self,
        request: Request<MatchExternalKvPrefixRequest>,
    ) -> Result<Response<MatchExternalKvPrefixResponse>, Status> {
        let (metadata, extensions, request) = request.into_parts();
        // Before any work: an expired query must not spend the capacity the rest
        // of the backlog needs to drain.
        reject_if_deadline_passed(&metadata, &extensions)?;
        validate_hashes(&request.hashes)?;
        // Caps concurrent prefix queries; excess is rejected, never queued.
        let _permit = self.prefix_query_semaphore.try_acquire().map_err(|_| {
            if let Some(rejected_total) = OVERLOAD_LOG.record() {
                tracing::warn!(
                    rejected_total,
                    "rejecting prefix query: too many in-flight prefix queries"
                );
            }
            Status::resource_exhausted("too many in-flight prefix queries")
        })?;
        let response = self.backend.match_external_kv_prefix(request).await?;
        Ok(Response::new(response))
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: Request<GetExternalKvHitCountsRequest>,
    ) -> Result<Response<GetExternalKvHitCountsResponse>, Status> {
        let request = request.into_inner();
        validate_hashes_bounded(&request.hashes)?;
        let response = self.backend.get_external_kv_hit_counts(request).await?;
        Ok(Response::new(response))
    }

    async fn apply_external_kv_batch(
        &self,
        request: Request<ApplyExternalKvBatchRequest>,
    ) -> Result<Response<ApplyExternalKvBatchResponse>, Status> {
        let request = request.into_inner();
        validate_worker_id(&request.worker_id)?;
        validate_actions(&request.actions)?;
        let response = self.backend.apply_external_kv_batch(request).await?;
        Ok(Response::new(response))
    }
}

fn validate_worker_id(worker_id: &str) -> Result<(), Status> {
    if worker_id.is_empty() {
        return Err(Status::invalid_argument("worker_id must not be empty"));
    }
    Ok(())
}

/// Well-formedness plus the per-request hash ceiling, for the RPCs that mutate
/// state or build a per-hash response.
fn validate_hashes_bounded(hashes: &[i64]) -> Result<(), Status> {
    validate_hashes(hashes)?;
    if hashes.len() > MAX_HASHES_PER_REQUEST {
        return Err(Status::resource_exhausted(format!(
            "request contains {} hashes; maximum is {MAX_HASHES_PER_REQUEST}",
            hashes.len()
        )));
    }
    Ok(())
}

/// Well-formedness only: no hash ceiling. A prefix scan uses O(1) state per
/// candidate worker, and truncating it would silently understate a worker's
/// reusable prefix. Length is bounded by `max_blocks` and the transport limit,
/// not by the caller's deadline, which cannot cancel a scan already under way.
fn validate_hashes(hashes: &[i64]) -> Result<(), Status> {
    if hashes.is_empty() {
        return Err(Status::invalid_argument("hashes must not be empty"));
    }
    Ok(())
}

fn validate_tier(tier: i32) -> Result<(), Status> {
    match tier {
        1..=3 => Ok(()),
        0 => Err(Status::invalid_argument("tier must not be TIER_UNKNOWN")),
        _ => Err(Status::invalid_argument("tier is not supported")),
    }
}

fn validate_actions(actions: &[ExternalKvAction]) -> Result<(), Status> {
    // An empty actions list is a no-op that only refreshes the worker's recorded
    // address. Non-empty batches still have every action validated below.
    if actions.len() > MAX_ACTIONS_PER_BATCH {
        return Err(Status::resource_exhausted(format!(
            "batch contains {} actions; maximum is {MAX_ACTIONS_PER_BATCH}",
            actions.len()
        )));
    }
    let total_hashes: usize = actions.iter().map(|action| action.hashes.len()).sum();
    if total_hashes > MAX_HASHES_PER_REQUEST {
        return Err(Status::resource_exhausted(format!(
            "batch contains {total_hashes} hashes; maximum is {MAX_HASHES_PER_REQUEST}"
        )));
    }
    for action in actions {
        validate_tier(action.tier)?;
        match ExternalKvActionType::try_from(action.r#type) {
            Ok(ExternalKvActionType::ActionReport) | Ok(ExternalKvActionType::ActionRevoke) => {
                validate_hashes_bounded(&action.hashes)?;
            }
            // CLEAR_ALL_AT_TIER carries only a tier; hashes are ignored.
            Ok(ExternalKvActionType::ActionClearAllAtTier) => {}
            Ok(ExternalKvActionType::ActionUnknown) | Err(_) => {
                return Err(Status::invalid_argument("action type is not supported"));
            }
        }
        // The per-hash arrays are either absent (legacy) or index-aligned with
        // `hashes`; a partial array is a malformed batch, not a silent legacy hash.
        validate_aligned(
            action.component_masks.len(),
            action.hashes.len(),
            "component_masks",
        )?;
        validate_aligned(action.block_sizes.len(), action.hashes.len(), "block_sizes")?;
    }
    Ok(())
}

/// A per-hash side array must be empty (legacy) or exactly as long as `hashes`.
fn validate_aligned(array_len: usize, hashes_len: usize, field: &str) -> Result<(), Status> {
    if array_len != 0 && array_len != hashes_len {
        return Err(Status::invalid_argument(format!(
            "{field} has {array_len} entries but must be empty or match {hashes_len} hashes"
        )));
    }
    Ok(())
}

/// Number of leading blocks to consider for a prefix query: bounded by the
/// request length and, when the caller set one, by `max_blocks` (0 disables the
/// caller ceiling).
pub(crate) fn prefix_limit(len: usize, max_blocks: u32) -> usize {
    if max_blocks == 0 {
        len
    } else {
        len.min(max_blocks as usize)
    }
}

/// KV component bits. Each component's rule is a property of its type, so the
/// indexer applies fixed semantics rather than a per-worker rule binding.
pub const COMPONENT_FULL: u32 = 1 << 0;
pub const COMPONENT_SWA: u32 = 1 << 1;
pub const COMPONENT_MAMBA: u32 = 1 << 2;

/// On-wire component label to its bit; `None` for a label this build does not
/// model (ignored, so an unknown future component never counts).
pub fn component_bit(name: &str) -> Option<u32> {
    match name {
        "full" => Some(COMPONENT_FULL),
        "swa" => Some(COMPONENT_SWA),
        "mamba" => Some(COMPONENT_MAMBA),
        _ => None,
    }
}

/// Servable tiers as a `1 << TierType` bitmask. V1: HBM + DRAM, SSD excluded.
const SERVABLE_TIER_MASK: u32 =
    (1 << (TierType::TierHbm as u32)) | (1 << (TierType::TierDram as u32));

/// Highest `WorkerCacheSpec.version` this build interprets; a higher (future)
/// version fails closed. Version 0 (proto default) is accepted as current.
const SUPPORTED_SPEC_VERSION: u32 = 1;

/// Whether `tier` is set in a `1 << TierType` bitmask.
fn tier_in_mask(mask: u32, tier: i32) -> bool {
    tier >= 0 && mask & (1u32 << tier) != 0
}

/// One block's placement at one worker: token count plus, per tier held, the
/// resident component bitmask (mask `0` = legacy whole-block, held with no detail).
#[derive(Debug, Clone)]
pub struct BlockComponents {
    pub token_count: u32,
    pub tier_masks: Vec<(i32, u32)>,
}

/// One candidate worker for the rule engine: routing identity, optional spec, and
/// per-query-block placement (`None` where the worker does not hold the block).
#[derive(Debug, Clone)]
pub struct WorkerPrefixInput {
    pub worker_id: String,
    pub address: String,
    pub spec: Option<WorkerCacheSpec>,
    pub blocks: Vec<Option<BlockComponents>>,
}

/// Builds component-blind (legacy) prefix inputs from a `MatchExternalKv` result:
/// each held block becomes a whole-block placement (mask `0`, no size, no spec).
pub(crate) fn legacy_inputs_from_match(
    hashes: &[i64],
    matched: &MatchExternalKvResponse,
) -> Vec<WorkerPrefixInput> {
    matched
        .matches
        .iter()
        .map(|node| {
            let mut tiers_by_hash: HashMap<i64, Vec<i32>> = HashMap::new();
            for tier in &node.hashes_by_tier {
                for hash in &tier.hashes {
                    tiers_by_hash.entry(*hash).or_default().push(tier.tier);
                }
            }
            let blocks = hashes
                .iter()
                .map(|hash| {
                    tiers_by_hash.get(hash).map(|tiers| BlockComponents {
                        token_count: 0,
                        tier_masks: tiers.iter().map(|tier| (*tier, 0u32)).collect(),
                    })
                })
                .collect();
            WorkerPrefixInput {
                worker_id: node.worker_id.clone(),
                address: node.address.clone(),
                spec: None,
                blocks,
            }
        })
        .collect()
}

/// Runs the component-aware rule engine over each worker and assembles the
/// response. Every backend feeds this same engine, so fast paths cannot drift.
pub(crate) fn compute_prefix_response(
    inputs: &[WorkerPrefixInput],
    blocks_read: u32,
) -> MatchExternalKvPrefixResponse {
    let entries = inputs
        .iter()
        .filter_map(|worker| {
            // An empty address is unroutable (see the proto worker_address contract).
            if worker.address.is_empty() {
                return None;
            }
            let prefix = compute_worker_prefix(worker.spec.as_ref(), &worker.blocks);
            (prefix > 0).then(|| (worker.worker_id.clone(), worker.address.clone(), prefix))
        })
        .collect();
    assemble_prefix_response(entries, blocks_read)
}

/// The reusable prefix length for one worker: a safe lower bound on what it can
/// serve. Returns 0 (the worker is excluded) when a component-aware store lacks a
/// spec or the spec carries an unusable rule.
pub(crate) fn compute_worker_prefix(
    spec: Option<&WorkerCacheSpec>,
    blocks: &[Option<BlockComponents>],
) -> u32 {
    let mut scanner = WorkerPrefixScanner::new(spec);
    for block in blocks {
        scanner.push(block.as_ref());
    }
    scanner.prefix()
}

/// Incremental form of the component rule engine: one forward pass, one block at
/// a time, O(1) state. Lets a backend answer a prefix query without materializing
/// a `workers × request_blocks` placement array.
#[derive(Debug)]
pub(crate) struct WorkerPrefixScanner {
    processed: u32,
    state: PrefixScanState,
}

#[derive(Debug)]
enum PrefixScanState {
    /// A worker reporting no components: the count of leading blocks it holds,
    /// unless some block carries a component mask, which fails the whole result
    /// closed.
    Legacy {
        prefix: u32,
        /// False once a gap appears, after which `prefix` is final.
        contiguous: bool,
        saw_components: bool,
    },
    /// An unusable spec. Fails closed no matter what blocks arrive.
    Invalid,
    /// The largest boundary `N` where every required component's rule holds:
    ///   * FULL (always required)  — present on every block `0..N`.
    ///   * SWA (if present)        — an unbroken run ending at `N-1` covering
    ///     `swa_window_tokens`, or reaching the head.
    ///   * MAMBA (if present)      — present on block `N-1`.
    ComponentAware {
        /// Cleared once FULL is missing, which freezes `best`.
        active: bool,
        best: u32,
        spec: WorkerCacheSpec,
        /// Contiguous SWA tokens ending at the block just processed.
        swa_run: u64,
        swa_head_broken: bool,
    },
}

impl WorkerPrefixScanner {
    pub(crate) fn new(spec: Option<&WorkerCacheSpec>) -> Self {
        let state = match spec {
            // No spec to interpret components with: legacy until a block proves
            // otherwise.
            None => PrefixScanState::Legacy {
                prefix: 0,
                contiguous: true,
                saw_components: false,
            },
            Some(spec) if spec.components == 0 || spec.version > SUPPORTED_SPEC_VERSION => {
                PrefixScanState::Invalid
            }
            Some(spec) if spec.components & COMPONENT_SWA != 0 && spec.swa_window_tokens == 0 => {
                PrefixScanState::Invalid
            }
            Some(spec) => PrefixScanState::ComponentAware {
                active: true,
                best: 0,
                spec: *spec,
                swa_run: 0,
                swa_head_broken: false,
            },
        };
        Self {
            processed: 0,
            state,
        }
    }

    pub(crate) fn push(&mut self, block: Option<&BlockComponents>) {
        self.processed = self.processed.saturating_add(1);
        match &mut self.state {
            PrefixScanState::Legacy {
                prefix,
                contiguous,
                saw_components,
            } => {
                // Runs past the gap too: a mask on any later block still fails the
                // whole result closed.
                *saw_components |=
                    block.is_some_and(|block| block.tier_masks.iter().any(|(_, mask)| *mask != 0));
                if *contiguous {
                    match block {
                        Some(_) => *prefix = self.processed,
                        None => *contiguous = false,
                    }
                }
            }
            PrefixScanState::Invalid => {}
            PrefixScanState::ComponentAware {
                active,
                best,
                spec,
                swa_run,
                swa_head_broken,
            } => {
                if !*active {
                    return;
                }
                // FULL gates contiguity, so a block missing it settles this worker.
                if !component_available(block, COMPONENT_FULL, spec.full_tier_mask) {
                    *active = false;
                    return;
                }
                let mut boundary_ok = true;
                if spec.components & COMPONENT_SWA != 0 {
                    if component_available(block, COMPONENT_SWA, spec.swa_tier_mask) {
                        *swa_run += block.map(|block| block.token_count as u64).unwrap_or(0);
                        // Reaching the head counts as valid, matching the unified
                        // cache's accumulator seeded at infinity.
                        boundary_ok &=
                            !*swa_head_broken || *swa_run >= spec.swa_window_tokens as u64;
                    } else {
                        *swa_run = 0;
                        *swa_head_broken = true;
                        boundary_ok = false; // boundary block itself must carry SWA
                    }
                }
                if spec.components & COMPONENT_MAMBA != 0 {
                    boundary_ok &=
                        component_available(block, COMPONENT_MAMBA, spec.mamba_tier_mask);
                }
                if boundary_ok {
                    *best = self.processed;
                }
            }
        }
    }

    pub(crate) fn prefix(&self) -> u32 {
        match &self.state {
            PrefixScanState::Legacy {
                prefix,
                saw_components,
                ..
            } => {
                if *saw_components {
                    0
                } else {
                    *prefix
                }
            }
            PrefixScanState::Invalid => 0,
            PrefixScanState::ComponentAware { best, .. } => *best,
        }
    }
}

/// Whether `component` (a single bit) is resident on `block` at some tier that is
/// both declared servable for that component (`spec_tier_mask`) and servable by
/// the indexer (`SERVABLE_TIER_MASK`).
fn component_available(
    block: Option<&BlockComponents>,
    component: u32,
    spec_tier_mask: u32,
) -> bool {
    let Some(block) = block else {
        return false;
    };
    block.tier_masks.iter().any(|(tier, mask)| {
        mask & component != 0
            && tier_in_mask(SERVABLE_TIER_MASK, *tier)
            && tier_in_mask(spec_tier_mask, *tier)
    })
}

/// Sorts `(worker_id, address, prefix)` entries by prefix descending and builds
/// the response, so `best_prefix_blocks` and the order come from one place.
pub(crate) fn assemble_prefix_response(
    mut entries: Vec<(String, String, u32)>,
    blocks_read: u32,
) -> MatchExternalKvPrefixResponse {
    entries.sort_by_key(|entry| std::cmp::Reverse(entry.2));
    let best_prefix_blocks = entries.first().map(|entry| entry.2).unwrap_or(0);
    let matches = entries
        .into_iter()
        .map(
            |(worker_id, worker_address, matched_prefix_blocks)| ExternalKvPrefixMatch {
                worker_address,
                matched_prefix_blocks,
                worker_id,
            },
        )
        .collect();
    MatchExternalKvPrefixResponse {
        matches,
        best_prefix_blocks,
        blocks_read,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use tokio::time::Duration;

    #[derive(Clone)]
    struct BlockingPrefixBackend {
        entered: Arc<AtomicUsize>,
        release: Arc<Semaphore>,
    }

    #[tonic::async_trait]
    impl KvIndexerBackend for BlockingPrefixBackend {
        async fn apply_external_kv_batch(
            &self,
            _request: ApplyExternalKvBatchRequest,
        ) -> Result<ApplyExternalKvBatchResponse, Status> {
            Ok(ApplyExternalKvBatchResponse::default())
        }

        async fn match_external_kv(
            &self,
            _request: MatchExternalKvRequest,
        ) -> Result<MatchExternalKvResponse, Status> {
            Ok(MatchExternalKvResponse::default())
        }

        async fn match_external_kv_prefix(
            &self,
            _request: MatchExternalKvPrefixRequest,
        ) -> Result<MatchExternalKvPrefixResponse, Status> {
            self.entered.fetch_add(1, Ordering::SeqCst);
            let _permit = self
                .release
                .acquire()
                .await
                .expect("release semaphore closed");
            Ok(MatchExternalKvPrefixResponse::default())
        }

        async fn get_external_kv_hit_counts(
            &self,
            _request: GetExternalKvHitCountsRequest,
        ) -> Result<GetExternalKvHitCountsResponse, Status> {
            Ok(GetExternalKvHitCountsResponse::default())
        }
    }

    /// Runs the arrival stamp, waits out `queued_for` to stand in for the time a
    /// dispatched request spends waiting in the runtime, then serves it.
    async fn serve_after_queueing(
        service: &KvIndexerService<BlockingPrefixBackend>,
        caller_deadline: Duration,
        queued_for: Duration,
    ) -> Result<Response<MatchExternalKvPrefixResponse>, Status> {
        let mut arriving = Request::new(());
        arriving.set_timeout(caller_deadline);
        let (metadata, extensions, ()) = crate::admission::stamp_arrival(arriving)
            .expect("arrival stamp never rejects")
            .into_parts();

        tokio::time::sleep(queued_for).await;

        let request = Request::from_parts(
            metadata,
            extensions,
            MatchExternalKvPrefixRequest {
                hashes: vec![-1],
                max_blocks: 0,
            },
        );
        KvIndexer::match_external_kv_prefix(service, request).await
    }

    fn non_blocking_backend(entered: &Arc<AtomicUsize>) -> BlockingPrefixBackend {
        BlockingPrefixBackend {
            entered: Arc::clone(entered),
            release: Arc::new(Semaphore::new(Semaphore::MAX_PERMITS)),
        }
    }

    #[tokio::test]
    async fn query_that_outlived_its_caller_never_reaches_the_backend() {
        let entered = Arc::new(AtomicUsize::new(0));
        let service = KvIndexerService::new(non_blocking_backend(&entered));

        let status = serve_after_queueing(
            &service,
            Duration::from_millis(20),
            Duration::from_millis(60),
        )
        .await
        .unwrap_err();

        assert_eq!(status.code(), tonic::Code::DeadlineExceeded);
        assert_eq!(entered.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn query_still_inside_its_deadline_is_served() {
        let entered = Arc::new(AtomicUsize::new(0));
        let service = KvIndexerService::new(non_blocking_backend(&entered));

        serve_after_queueing(&service, Duration::from_secs(30), Duration::from_millis(10))
            .await
            .expect("a query within its deadline must still be answered");

        assert_eq!(entered.load(Ordering::SeqCst), 1);
    }

    fn hbm() -> i32 {
        crate::pb::TierType::TierHbm as i32
    }

    fn action(r#type: ExternalKvActionType, tier: i32, hashes: &[&str]) -> ExternalKvAction {
        ExternalKvAction {
            r#type: r#type as i32,
            tier,
            hashes: hashes.iter().map(|h| h.parse().unwrap()).collect(),
            component_masks: Vec::new(),
            block_sizes: Vec::new(),
        }
    }

    #[test]
    fn validate_actions_allows_empty_batch() {
        // An empty batch carries no mutation but is not an error.
        assert!(validate_actions(&[]).is_ok());
    }

    #[test]
    fn validate_actions_rejects_unknown_type() {
        let actions = [action(ExternalKvActionType::ActionUnknown, hbm(), &["1"])];
        assert!(validate_actions(&actions).is_err());
    }

    #[test]
    fn validate_actions_rejects_bad_tier() {
        let actions = [action(ExternalKvActionType::ActionReport, 0, &["1"])];
        assert!(validate_actions(&actions).is_err());
    }

    #[test]
    fn validate_actions_rejects_misaligned_side_arrays() {
        let base = action(ExternalKvActionType::ActionReport, hbm(), &["1", "2"]);
        assert!(validate_actions(std::slice::from_ref(&base)).is_ok());
        let mut aligned = base.clone();
        aligned.component_masks = vec![COMPONENT_FULL, COMPONENT_FULL];
        aligned.block_sizes = vec![16, 16];
        assert!(validate_actions(&[aligned]).is_ok());
        // A short component_masks array is a malformed batch, not a silent legacy.
        let mut bad_masks = base.clone();
        bad_masks.component_masks = vec![COMPONENT_FULL];
        assert_eq!(
            validate_actions(&[bad_masks]).unwrap_err().code(),
            tonic::Code::InvalidArgument
        );
        // A short block_sizes array is rejected too.
        let mut bad_sizes = base;
        bad_sizes.block_sizes = vec![16];
        assert_eq!(
            validate_actions(&[bad_sizes]).unwrap_err().code(),
            tonic::Code::InvalidArgument
        );
    }

    #[test]
    fn validate_actions_requires_hashes_for_report_and_revoke() {
        assert!(
            validate_actions(&[action(ExternalKvActionType::ActionReport, hbm(), &[])]).is_err()
        );
        assert!(
            validate_actions(&[action(ExternalKvActionType::ActionRevoke, hbm(), &[])]).is_err()
        );
    }

    #[test]
    fn validate_actions_allows_empty_hashes_for_clear_all_at_tier() {
        let actions = [action(
            ExternalKvActionType::ActionClearAllAtTier,
            hbm(),
            &[],
        )];
        assert!(validate_actions(&actions).is_ok());
    }

    #[test]
    fn validate_hashes_rejects_oversized_query() {
        let hashes = vec![1; MAX_HASHES_PER_REQUEST + 1];
        let error = validate_hashes_bounded(&hashes).unwrap_err();
        assert_eq!(error.code(), tonic::Code::ResourceExhausted);
    }

    /// Only the bounded variant rejects on length.
    #[test]
    fn validate_hashes_accepts_oversized_prefix_query() {
        let hashes = vec![1; MAX_HASHES_PER_REQUEST + 1];
        assert!(validate_hashes(&hashes).is_ok());
    }

    #[test]
    fn validate_actions_rejects_oversized_batch() {
        let hashes = vec!["1"; MAX_HASHES_PER_REQUEST / 2 + 1];
        let actions = [
            action(ExternalKvActionType::ActionReport, hbm(), &hashes),
            action(ExternalKvActionType::ActionReport, hbm(), &hashes),
        ];
        let error = validate_actions(&actions).unwrap_err();
        assert_eq!(error.code(), tonic::Code::ResourceExhausted);
    }

    #[test]
    fn validate_actions_rejects_too_many_actions() {
        let clear = action(ExternalKvActionType::ActionClearAllAtTier, hbm(), &[]);
        let actions = vec![clear; MAX_ACTIONS_PER_BATCH + 1];
        let error = validate_actions(&actions).unwrap_err();
        assert_eq!(error.code(), tonic::Code::ResourceExhausted);
    }

    #[test]
    fn validate_worker_id_rejects_empty_value() {
        assert!(validate_worker_id("").is_err());
        assert!(validate_worker_id("worker-1").is_ok());
    }

    // --- component-aware prefix rule engine ---

    fn dram() -> i32 {
        crate::pb::TierType::TierDram as i32
    }
    fn ssd() -> i32 {
        crate::pb::TierType::TierSsd as i32
    }

    /// OR the tiers into a `1 << TierType` bitmask.
    fn tmask(tiers: &[i32]) -> u32 {
        tiers.iter().fold(0, |m, t| m | (1u32 << t))
    }

    /// A held block with `(tier, component bitmask)` placements and a token count.
    fn blk(tiers: &[(i32, u32)], token_count: u32) -> Option<BlockComponents> {
        Some(BlockComponents {
            token_count,
            tier_masks: tiers.to_vec(),
        })
    }

    /// A legacy whole-block placement (mask 0) at HBM.
    fn legacy_blk() -> Option<BlockComponents> {
        blk(&[(hbm(), 0)], 0)
    }

    fn spec(
        components: u32,
        swa_window_tokens: u32,
        full_tiers: &[i32],
        swa_tiers: &[i32],
        mamba_tiers: &[i32],
    ) -> WorkerCacheSpec {
        WorkerCacheSpec {
            version: 1,
            components,
            swa_window_tokens,
            full_tier_mask: tmask(full_tiers),
            swa_tier_mask: tmask(swa_tiers),
            mamba_tier_mask: tmask(mamba_tiers),
        }
    }

    #[test]
    fn legacy_no_spec_is_contiguous() {
        let blocks = vec![legacy_blk(), legacy_blk(), legacy_blk(), None, legacy_blk()];
        assert_eq!(compute_worker_prefix(None, &blocks), 3);
    }

    #[test]
    fn component_report_without_spec_is_excluded() {
        // A worker that reports components but declared no spec cannot be
        // interpreted safely, so it contributes nothing (NoSignal-safe).
        let blocks = vec![
            blk(&[(hbm(), COMPONENT_FULL)], 16),
            blk(&[(hbm(), COMPONENT_FULL)], 16),
        ];
        assert_eq!(compute_worker_prefix(None, &blocks), 0);
    }

    #[test]
    fn contiguous_full_stops_at_first_gap() {
        let s = spec(COMPONENT_FULL, 0, &[hbm(), dram()], &[], &[]);
        let blocks = vec![
            blk(&[(hbm(), COMPONENT_FULL)], 16),
            blk(&[(dram(), COMPONENT_FULL)], 16), // full may live on a different servable tier
            blk(&[(hbm(), COMPONENT_SWA)], 16),   // no full here -> prefix stops
            blk(&[(hbm(), COMPONENT_FULL)], 16),
        ];
        assert_eq!(compute_worker_prefix(Some(&s), &blocks), 2);
    }

    #[test]
    fn ssd_only_is_not_servable_in_v1() {
        let s = spec(COMPONENT_FULL, 0, &[hbm(), dram()], &[], &[]);
        let blocks = vec![blk(&[(ssd(), COMPONENT_FULL)], 16)];
        assert_eq!(compute_worker_prefix(Some(&s), &blocks), 0);
    }

    #[test]
    fn trailing_window_requires_unbroken_window_before_boundary() {
        // window = 100 tokens, 50 tokens per block: two contiguous swa blocks
        // cover a window. full is present on every block.
        let s = spec(COMPONENT_FULL | COMPONENT_SWA, 100, &[hbm()], &[hbm()], &[]);
        let with_swa = || blk(&[(hbm(), COMPONENT_FULL | COMPONENT_SWA)], 50);
        let no_swa = || blk(&[(hbm(), COMPONENT_FULL)], 50);
        // swa present everywhere -> full length reusable.
        let blocks = vec![with_swa(), with_swa(), with_swa(), with_swa(), with_swa()];
        assert_eq!(compute_worker_prefix(Some(&s), &blocks), 5);
        // swa tombstoned at block index 3: the largest boundary whose trailing
        // 100-token window is unbroken is N=3 (blocks 1..2 cover 100 tokens).
        let holed = vec![with_swa(), with_swa(), with_swa(), no_swa(), with_swa()];
        assert_eq!(compute_worker_prefix(Some(&s), &holed), 3);
    }

    #[test]
    fn trailing_window_head_is_always_valid() {
        // Fewer tokens than a window, but an unbroken run from the head is valid
        // (matches the unified cache's window accumulator seeded at infinity).
        let s = spec(
            COMPONENT_FULL | COMPONENT_SWA,
            1000,
            &[hbm()],
            &[hbm()],
            &[],
        );
        let blocks = vec![blk(&[(hbm(), COMPONENT_FULL | COMPONENT_SWA)], 16); 2];
        assert_eq!(compute_worker_prefix(Some(&s), &blocks), 2);
    }

    #[test]
    fn exact_boundary_only_matches_at_a_checkpoint() {
        // mamba lives only on the 4th block (a leaf checkpoint). full is on all.
        let s = spec(
            COMPONENT_FULL | COMPONENT_MAMBA,
            0,
            &[hbm(), dram()],
            &[],
            &[hbm(), dram()],
        );
        let blocks = vec![
            blk(&[(hbm(), COMPONENT_FULL)], 16),
            blk(&[(hbm(), COMPONENT_FULL)], 16),
            blk(&[(hbm(), COMPONENT_FULL)], 16),
            blk(&[(hbm(), COMPONENT_FULL | COMPONENT_MAMBA)], 16),
        ];
        assert_eq!(compute_worker_prefix(Some(&s), &blocks), 4);
        // A shorter request that never reaches the checkpoint cannot reuse it.
        assert_eq!(compute_worker_prefix(Some(&s), &blocks[..2]), 0);
    }

    #[test]
    fn unusable_specs_are_excluded() {
        // Each of these declared specs is unusable and must fail closed: an empty
        // component set, a future/unsupported version, and SWA without a window.
        let blocks = vec![blk(&[(hbm(), COMPONENT_FULL | COMPONENT_SWA)], 16)];
        let empty = spec(0, 0, &[hbm()], &[], &[]);
        let mut future = spec(COMPONENT_FULL, 0, &[hbm()], &[], &[]);
        future.version = SUPPORTED_SPEC_VERSION + 1;
        let swa_no_window = spec(COMPONENT_FULL | COMPONENT_SWA, 0, &[hbm()], &[hbm()], &[]);
        for s in [empty, future, swa_no_window] {
            assert_eq!(compute_worker_prefix(Some(&s), &blocks), 0);
        }
    }

    #[test]
    fn missing_component_data_under_spec_excludes() {
        // Spec requires full+swa but the worker reported legacy whole-block
        // placement (mask 0), so full cannot be confirmed and it is excluded.
        let s = spec(COMPONENT_FULL | COMPONENT_SWA, 100, &[hbm()], &[hbm()], &[]);
        let blocks = vec![legacy_blk(), legacy_blk()];
        assert_eq!(compute_worker_prefix(Some(&s), &blocks), 0);
    }
}
