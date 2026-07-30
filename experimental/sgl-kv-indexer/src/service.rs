// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use tonic::{Request, Response, Status};

use crate::pb::kv_indexer_server::KvIndexer;
use crate::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, ExternalKvAction,
    ExternalKvActionType, ExternalKvPrefixMatch, GetExternalKvHitCountsRequest,
    GetExternalKvHitCountsResponse, MatchExternalKvPrefixRequest, MatchExternalKvPrefixResponse,
    MatchExternalKvRequest, MatchExternalKvResponse,
};

/// Protocol-level resource bounds. The Redis backend additionally chunks its
/// fan-out, but rejecting oversized requests here prevents any backend from
/// allocating or scheduling work proportional to an unbounded repeated field.
const MAX_HASHES_PER_REQUEST: usize = 16_384;
const MAX_ACTIONS_PER_BATCH: usize = 256;

/// Storage backend for the indexer. Deliberately narrow: every mutation flows
/// through `apply_external_kv_batch`, preserving one ordered write path.
///
/// Async because real backends (e.g. Redis) do network IO; the trait is made
/// dyn-safe via `#[tonic::async_trait]` so the server can select a backend at
/// runtime and hold it as `Arc<dyn KvIndexerBackend>`.
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

    /// Answers, per worker, the longest contiguous request prefix it holds.
    ///
    /// This default implementation *is* the written definition of the prefix
    /// semantics: it composes `match_external_kv` and walks each worker's
    /// matched set in request order, so any backend that overrides it for
    /// performance must stay field-for-field identical (except `blocks_read`,
    /// which is observability, not semantics).
    ///
    /// The semantics are deliberately stricter than `sgl-router`'s in-process
    /// `HashTree::match_prefix`: a worker's prefix stops at the first block it is
    /// missing, so the indexer never reports a worker as holding a prefix it
    /// cannot actually serve. It can only under-report, never over-report.
    async fn match_external_kv_prefix(
        &self,
        request: MatchExternalKvPrefixRequest,
    ) -> Result<MatchExternalKvPrefixResponse, Status> {
        let limit = prefix_limit(request.hashes.len(), request.max_blocks);
        let hashes: Vec<String> = request.hashes.into_iter().take(limit).collect();
        if hashes.is_empty() {
            return Ok(MatchExternalKvPrefixResponse::default());
        }
        let matched = self
            .match_external_kv(MatchExternalKvRequest {
                hashes: hashes.clone(),
                count_as_hit: false,
            })
            .await?;
        // The default path reads placement for every considered block.
        Ok(build_prefix_response(
            &hashes,
            &matched,
            hashes.len() as u32,
        ))
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
}

impl<B> KvIndexerService<B>
where
    B: KvIndexerBackend,
{
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
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
        validate_hashes(&request.hashes)?;
        let response = self.backend.match_external_kv(request).await?;
        Ok(Response::new(response))
    }

    async fn match_external_kv_prefix(
        &self,
        request: Request<MatchExternalKvPrefixRequest>,
    ) -> Result<Response<MatchExternalKvPrefixResponse>, Status> {
        let request = request.into_inner();
        validate_hashes(&request.hashes)?;
        let response = self.backend.match_external_kv_prefix(request).await?;
        Ok(Response::new(response))
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: Request<GetExternalKvHitCountsRequest>,
    ) -> Result<Response<GetExternalKvHitCountsResponse>, Status> {
        let request = request.into_inner();
        validate_hashes(&request.hashes)?;
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

fn validate_hashes(hashes: &[String]) -> Result<(), Status> {
    if hashes.is_empty() {
        return Err(Status::invalid_argument("hashes must not be empty"));
    }
    if hashes.len() > MAX_HASHES_PER_REQUEST {
        return Err(Status::resource_exhausted(format!(
            "request contains {} hashes; maximum is {MAX_HASHES_PER_REQUEST}",
            hashes.len()
        )));
    }
    if hashes.iter().any(|hash| hash.is_empty()) {
        return Err(Status::invalid_argument(
            "hashes must not contain empty values",
        ));
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
    // An empty actions list is accepted and applied as a no-op; it only refreshes
    // the worker's recorded address. Non-empty batches still have every action
    // validated below.
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
                validate_hashes(&action.hashes)?;
            }
            // CLEAR_ALL_AT_TIER carries only a tier; hashes are ignored.
            Ok(ExternalKvActionType::ActionClearAllAtTier) => {}
            Ok(ExternalKvActionType::ActionUnknown) | Err(_) => {
                return Err(Status::invalid_argument("action type is not supported"));
            }
        }
    }
    Ok(())
}

/// Number of leading blocks to consider for a prefix query: bounded by the
/// request length and, when the caller set one, by `max_blocks` (0 disables the
/// caller ceiling). Backends may impose their own additional scan cap.
pub(crate) fn prefix_limit(len: usize, max_blocks: u32) -> usize {
    if max_blocks == 0 {
        len
    } else {
        len.min(max_blocks as usize)
    }
}

/// Derives the prefix response from a `MatchExternalKv` result — the semantic
/// definition consumed by the trait default implementation. Each worker's prefix
/// is the run of leading `hashes` it holds contiguously; the walk stops at the
/// first missing block.
pub(crate) fn build_prefix_response(
    hashes: &[String],
    matched: &MatchExternalKvResponse,
    blocks_read: u32,
) -> MatchExternalKvPrefixResponse {
    let entries = matched
        .matches
        .iter()
        .filter_map(|node| {
            // An empty address is unroutable for the router (see the proto's
            // worker_address contract); drop it rather than report a match it
            // can never intersect.
            if node.address.is_empty() {
                return None;
            }
            let held: HashSet<&str> = node
                .hashes_by_tier
                .iter()
                .flat_map(|tier| tier.hashes.iter().map(String::as_str))
                .collect();
            let mut prefix = 0u32;
            for hash in hashes {
                if held.contains(hash.as_str()) {
                    prefix += 1;
                } else {
                    break;
                }
            }
            (prefix > 0).then(|| (node.worker_id.clone(), node.address.clone(), prefix))
        })
        .collect();
    assemble_prefix_response(entries, blocks_read)
}

/// Sorts `(worker_id, address, prefix)` entries by prefix descending and builds
/// the response. Shared so the Redis fast path, which computes prefixes during
/// its scan, produces byte-identical shape to the default implementation.
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

    fn hbm() -> i32 {
        crate::pb::TierType::TierHbm as i32
    }

    fn action(r#type: ExternalKvActionType, tier: i32, hashes: &[&str]) -> ExternalKvAction {
        ExternalKvAction {
            r#type: r#type as i32,
            tier,
            hashes: hashes.iter().map(|h| h.to_string()).collect(),
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
        let hashes = vec!["1".to_string(); MAX_HASHES_PER_REQUEST + 1];
        let error = validate_hashes(&hashes).unwrap_err();
        assert_eq!(error.code(), tonic::Code::ResourceExhausted);
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
}
