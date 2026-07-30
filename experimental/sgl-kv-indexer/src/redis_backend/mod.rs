// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Redis storage backend for the KV indexer.
//!
//! Data model (see [`schema`]): placement is a per-block-hash HASH of
//! `worker -> tier bitmask`; a per-worker SET is the reverse index used by
//! `CLEAR_ALL_AT_TIER`; a per-worker HASH holds the routing address; hit counts
//! live in a per-hash HASH co-located with placement (and are deleted together
//! with the placement when a block is fully revoked, so they never outlive the
//! block).
//!
//! This is the basic build, intended for bringing up the
//! SGLang -> bridge -> indexer -> Redis chain. Every apply is unconditional:
//! there is no per-worker sequence gate, no incarnation/generation fencing, no
//! restart reset, and no worker liveness TTL. Consequences to keep in mind while
//! debugging:
//!
//!   * A worker that restarts leaves its previous placement entries behind, and
//!     `match` will keep returning them.
//!   * A worker that dies stays in `match` results forever.
//!   * Batches are applied in arrival order with no deduplication, so a
//!     redelivered or reordered batch changes state.
//!
//! The individual mutations are still idempotent (bit set/clear, SADD/SREM) and
//! each block-hash mutation is atomic on its own cluster slot, but an apply
//! batch spanning many slots is not globally atomic and nothing repairs a
//! partial failure.

mod conn;
mod schema;
mod scripts;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use futures::future::try_join_all;
use redis::FromRedisValue;
use tokio::sync::Semaphore;
use tonic::Status;

use crate::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, ExternalKvActionType,
    ExternalKvNodeMatch, GetExternalKvHitCountsRequest, GetExternalKvHitCountsResponse,
    HitCountEntry, MatchExternalKvPrefixRequest, MatchExternalKvPrefixResponse,
    MatchExternalKvRequest, MatchExternalKvResponse, TierHashes,
};
use crate::service::{assemble_prefix_response, prefix_limit, KvIndexerBackend};

use conn::{ClusterConn, RedisConn, SingleConn};
use schema::{
    hit_key, placement_key, tier_bit, tiers_from_mask, worker_blocks_key, worker_meta_key,
};
use scripts::{HIT_BUMP, MATCH_HASH, PLACEMENT_CLEAR, PLACEMENT_SET};

/// Boxed error for construction paths (env parsing / connect / ping).
type BoxError = Box<dyn std::error::Error + Send + Sync>;

const DEFAULT_NAMESPACE: &str = "kvidx";
/// Per-request bound on concurrent Redis operations. Requests may contain more
/// hashes (up to the service-layer protocol limit), but every read/write path
/// processes them in sequential chunks so one request cannot create unbounded
/// in-flight work against Redis.
const REDIS_FANOUT_CHUNK: usize = 256;
/// `COUNT` hint for reverse-index `SSCAN` iteration. Sized to match the fanout
/// chunk so one scanned page turns into one round of concurrent cleanup.
const REVERSE_SCAN_PAGE: usize = 256;
/// Server-side ceiling on how many blocks a single prefix query scans, on top of
/// the caller's `max_blocks`. Bounds the Redis command count of one query even
/// if the caller passes an enormous hash list.
const PREFIX_SCAN_CAP: usize = 2048;
/// Process-wide cap on concurrent in-flight prefix queries.
///
/// Prefix queries sit on the routing hot path and share the one multiplexed
/// Redis connection with the apply (event-ingest) path. `max_blocks` and
/// `PREFIX_SCAN_CAP` bound a single query but not their number, so a burst of
/// concurrent queries could saturate the connection's request queue and push
/// apply calls past their response timeout. In this build applies carry no TTL
/// refresh, so a failed apply simply drops that mutation — placement drifts
/// stale and never self-heals. This semaphore keeps query fan-out well below the
/// apply path's per-batch concurrency (`REDIS_FANOUT_CHUNK`) so routing lookups
/// cannot starve ingestion. Routing queries are latency-sensitive, so excess
/// queries queueing briefly here is the right trade; applies are batched and
/// tolerate being a little slower.
const PREFIX_QUERY_MAX_INFLIGHT: usize = 16;

/// Resolved connection target parsed from the environment.
enum Target {
    Single(String),
    Cluster(Vec<String>),
}

pub struct RedisKvIndexerBackend {
    conn: Arc<dyn RedisConn>,
    ns: String,
    /// Bounds concurrent prefix queries; see [`PREFIX_QUERY_MAX_INFLIGHT`].
    prefix_semaphore: Arc<Semaphore>,
}

impl RedisKvIndexerBackend {
    fn new(conn: Arc<dyn RedisConn>, ns: impl Into<String>) -> Self {
        Self {
            conn,
            ns: ns.into(),
            prefix_semaphore: Arc::new(Semaphore::new(PREFIX_QUERY_MAX_INFLIGHT)),
        }
    }

    /// Connects to a single Redis/Dragonfly instance.
    pub async fn connect_single(url: &str, ns: impl Into<String>) -> Result<Self, BoxError> {
        let conn = SingleConn::connect(url).await?;
        Ok(Self::new(Arc::new(conn), ns))
    }

    /// Connects to a Redis Cluster from a list of seed node URLs.
    pub async fn connect_cluster(
        nodes: Vec<String>,
        ns: impl Into<String>,
    ) -> Result<Self, BoxError> {
        let conn = ClusterConn::connect(nodes).await?;
        Ok(Self::new(Arc::new(conn), ns))
    }

    /// Builds the backend from the environment:
    ///   * `KV_INDEXER_REDIS_NAMESPACE` (default `kvidx`)
    ///   * `KV_INDEXER_REDIS_CLUSTER_NODES` (comma-separated) → Cluster, else
    ///   * `KV_INDEXER_REDIS_URL` → single instance (required)
    ///
    /// Redis is always required: the connect is bounded by a timeout and
    /// followed by a PING, so an unreachable store is a loud startup failure.
    pub async fn from_env() -> Result<Self, BoxError> {
        let ns = std::env::var("KV_INDEXER_REDIS_NAMESPACE")
            .unwrap_or_else(|_| DEFAULT_NAMESPACE.into());

        let target = if let Ok(nodes) = std::env::var("KV_INDEXER_REDIS_CLUSTER_NODES") {
            let nodes: Vec<String> = nodes
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
            if nodes.is_empty() {
                return Err("KV_INDEXER_REDIS_CLUSTER_NODES is empty".into());
            }
            Target::Cluster(nodes)
        } else {
            let url = std::env::var("KV_INDEXER_REDIS_URL").map_err(|_| {
                "KV_INDEXER_REDIS_URL (or KV_INDEXER_REDIS_CLUSTER_NODES) is required for the redis backend"
            })?;
            Target::Single(url)
        };

        let backend = match target {
            Target::Cluster(nodes) => Self::connect_cluster(nodes, ns)
                .await
                .map_err(|e| format!("redis connect failed: {e}"))?,
            Target::Single(url) => Self::connect_single(&url, ns)
                .await
                .map_err(|e| format!("redis connect failed: {e}"))?,
        };
        backend
            .ping()
            .await
            .map_err(|e| format!("redis readiness probe (PING) failed: {e}"))?;
        Ok(backend)
    }

    async fn ping(&self) -> redis::RedisResult<()> {
        let _: redis::Value = self.conn.query(redis::cmd("PING").clone()).await?;
        Ok(())
    }

    // --- write path ---------------------------------------------------------

    async fn apply(
        &self,
        req: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status> {
        let worker = req.worker_id.as_str();

        // Record the worker's routing address so match responses can carry it.
        if !req.worker_address.is_empty() {
            self.set_worker_address(worker, &req.worker_address).await?;
        }

        // Actions are applied in order. An empty batch is a no-op.
        for action in &req.actions {
            let bit = tier_bit(action.tier);
            match ExternalKvActionType::try_from(action.r#type) {
                Ok(ExternalKvActionType::ActionReport) => {
                    for chunk in action.hashes.chunks(REDIS_FANOUT_CHUNK) {
                        try_join_all(chunk.iter().map(|hash| self.report_one(worker, hash, bit)))
                            .await?;
                    }
                }
                Ok(ExternalKvActionType::ActionRevoke) => {
                    self.revoke_many(worker, &action.hashes, bit).await?;
                }
                Ok(ExternalKvActionType::ActionClearAllAtTier) => {
                    self.clear_all_at_tier(worker, bit).await?;
                }
                _ => return Err(Status::invalid_argument("unsupported action type")),
            }
        }

        Ok(ApplyExternalKvBatchResponse {})
    }

    async fn set_worker_address(&self, worker: &str, addr: &str) -> Result<(), Status> {
        let mut cmd = redis::cmd("HSET");
        cmd.arg(worker_meta_key(&self.ns, worker))
            .arg("addr")
            .arg(addr);
        self.conn.query(cmd).await.map_err(to_status)?;
        Ok(())
    }

    /// Reads a worker's routing address, empty when it has never applied.
    async fn worker_address(&self, worker: &str) -> Result<String, Status> {
        let mut cmd = redis::cmd("HGET");
        cmd.arg(worker_meta_key(&self.ns, worker)).arg("addr");
        let v = self.conn.query(cmd).await.map_err(to_status)?;
        Ok(Option::<String>::from_redis_value(&v)
            .map_err(to_status)?
            .unwrap_or_default())
    }

    /// Reads one `SSCAN` page of a worker's reverse index, returning the next
    /// cursor (`0` once the iteration is complete) alongside the members.
    ///
    /// `SMEMBERS` would occupy Redis for the whole set and materialize it in a
    /// single reply, and a worker legitimately owns as many blocks as its cache
    /// holds. Scanning may return a member more than once and may miss members
    /// added or removed mid-iteration; the caller is idempotent, so neither
    /// matters here.
    async fn reverse_page(&self, worker: &str, cursor: u64) -> Result<(u64, Vec<String>), Status> {
        let mut cmd = redis::cmd("SSCAN");
        cmd.arg(worker_blocks_key(&self.ns, worker))
            .arg(cursor)
            .arg("COUNT")
            .arg(REVERSE_SCAN_PAGE);
        let value = self.conn.query(cmd).await.map_err(to_status)?;
        <(u64, Vec<String>)>::from_redis_value(&value).map_err(to_status)
    }

    async fn report_one(&self, worker: &str, hash: &str, bit: i64) -> Result<(), Status> {
        self.conn
            .invoke(
                &PLACEMENT_SET,
                vec![placement_key(&self.ns, hash)],
                vec![worker.to_string(), bit.to_string()],
            )
            .await
            .map_err(to_status)?;

        let mut cmd = redis::cmd("SADD");
        cmd.arg(worker_blocks_key(&self.ns, worker)).arg(hash);
        self.conn.query(cmd).await.map_err(to_status)?;
        Ok(())
    }

    async fn revoke_many(&self, worker: &str, hashes: &[String], bit: i64) -> Result<(), Status> {
        for chunk in hashes.chunks(REDIS_FANOUT_CHUNK) {
            try_join_all(chunk.iter().map(|hash| self.revoke_one(worker, hash, bit))).await?;
        }
        Ok(())
    }

    async fn revoke_one(&self, worker: &str, hash: &str, bit: i64) -> Result<(), Status> {
        // Placement and hit keys share a slot; the script drops hits when the
        // final placement disappears.
        let v = self
            .conn
            .invoke(
                &PLACEMENT_CLEAR,
                vec![placement_key(&self.ns, hash), hit_key(&self.ns, hash)],
                vec![worker.to_string(), bit.to_string()],
            )
            .await
            .map_err(to_status)?;
        let worker_gone = i64::from_redis_value(&v).map_err(to_status)? == 1;
        if worker_gone {
            let mut cmd = redis::cmd("SREM");
            cmd.arg(worker_blocks_key(&self.ns, worker)).arg(hash);
            self.conn.query(cmd).await.map_err(to_status)?;
        }
        Ok(())
    }

    async fn clear_all_at_tier(&self, worker: &str, bit: i64) -> Result<(), Status> {
        let mut cursor = 0;
        loop {
            let (next, page) = self.reverse_page(worker, cursor).await?;
            let hashes: Vec<String> = page
                .into_iter()
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            self.revoke_many(worker, &hashes, bit).await?;
            cursor = next;
            if cursor == 0 {
                return Ok(());
            }
        }
    }

    // --- read path ----------------------------------------------------------

    async fn do_match(
        &self,
        req: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        let hashes = dedup_preserve_order(&req.hashes);

        // Per-hash placement read, preserving hash order.
        let mut per_hash = Vec::with_capacity(hashes.len());
        for chunk in hashes.chunks(REDIS_FANOUT_CHUNK) {
            let values = try_join_all(chunk.iter().map(|hash| async move {
                let v = self
                    .conn
                    .invoke(&MATCH_HASH, vec![placement_key(&self.ns, hash)], Vec::new())
                    .await?;
                let flat: Vec<String> = Vec::<String>::from_redis_value(&v)?;
                Ok::<_, redis::RedisError>(flat)
            }))
            .await
            .map_err(to_status)?;
            per_hash.extend(values);
        }

        let mut worker_order: Vec<String> = Vec::new();
        let mut by_worker: HashMap<String, Vec<(String, i64)>> = HashMap::new();
        for (hash, flat) in hashes.iter().zip(per_hash) {
            for pair in flat.chunks(2) {
                if pair.len() != 2 {
                    continue;
                }
                let worker = &pair[0];
                let mask = pair[1].parse::<i64>().unwrap_or(0);
                let entry = by_worker.entry(worker.clone()).or_insert_with(|| {
                    worker_order.push(worker.clone());
                    Vec::new()
                });
                entry.push((hash.clone(), mask));
            }
        }

        // Fetch each matched worker's routing address.
        let mut addresses = Vec::with_capacity(worker_order.len());
        for chunk in worker_order.chunks(REDIS_FANOUT_CHUNK) {
            let values =
                try_join_all(chunk.iter().map(|worker| self.worker_address(worker))).await?;
            addresses.extend(values);
        }

        let mut matches = Vec::new();
        let mut matched_hashes: Vec<String> = Vec::new();
        let mut seen_hashes: HashSet<String> = HashSet::new();
        for (worker, address) in worker_order.into_iter().zip(addresses) {
            let mut by_tier: BTreeMap<i32, Vec<String>> = BTreeMap::new();
            for (hash, mask) in by_worker.remove(&worker).unwrap_or_default() {
                let tiers = tiers_from_mask(mask);
                if tiers.is_empty() {
                    continue;
                }
                for tier in tiers {
                    by_tier.entry(tier).or_default().push(hash.clone());
                }
                if seen_hashes.insert(hash.clone()) {
                    matched_hashes.push(hash);
                }
            }
            if by_tier.is_empty() {
                continue;
            }
            matches.push(ExternalKvNodeMatch {
                worker_id: worker,
                address,
                hashes_by_tier: by_tier
                    .into_iter()
                    .map(|(tier, hashes)| TierHashes { tier, hashes })
                    .collect(),
            });
        }

        if req.count_as_hit {
            let now = now_ms().to_string();
            for chunk in matched_hashes.chunks(REDIS_FANOUT_CHUNK) {
                try_join_all(chunk.iter().map(|hash| {
                    self.conn
                        .invoke(&HIT_BUMP, vec![hit_key(&self.ns, hash)], vec![now.clone()])
                }))
                .await
                .map_err(to_status)?;
            }
        }

        Ok(MatchExternalKvResponse { matches })
    }

    /// Workers that hold a block hash at any valid tier.
    async fn placement_holders(&self, hash: &str) -> Result<Vec<String>, Status> {
        let v = self
            .conn
            .invoke(&MATCH_HASH, vec![placement_key(&self.ns, hash)], Vec::new())
            .await
            .map_err(to_status)?;
        let flat = Vec::<String>::from_redis_value(&v).map_err(to_status)?;
        let mut holders = Vec::new();
        for pair in flat.chunks(2) {
            if pair.len() != 2 {
                continue;
            }
            let mask = pair[1].parse::<i64>().unwrap_or(0);
            if !tiers_from_mask(mask).is_empty() {
                holders.push(pair[0].clone());
            }
        }
        Ok(holders)
    }

    /// Three-stage forward scan producing the same prefix semantics as the trait
    /// default implementation.
    ///
    /// 1. Read `hashes[0]`'s placement to get every worker that could hold *any*
    ///    prefix. If none, one Redis command answers the whole query.
    /// 2. Read the routing registry once, for that worker set only; the candidate
    ///    set only shrinks afterwards, so later blocks never re-read the registry.
    /// 3. From index 1, scan forward in windows, advancing each surviving worker's
    ///    contiguous prefix and dropping it at its first missing block; stop when
    ///    no candidate remains or the scan/`max_blocks` cap is reached.
    ///
    /// The win here is *fewer Redis commands* — a first-block miss costs one
    /// command instead of one per block — not lower latency; a fully-cached long
    /// prefix takes more round-trips than reading every block at once. The
    /// registry is read as a snapshot: a worker that restarts mid-scan can still
    /// contribute entries this build never fences, so a prefix may come out longer
    /// than the worker truly holds (never shorter). The index is advisory, so this
    /// window is acceptable.
    async fn do_match_prefix(
        &self,
        req: MatchExternalKvPrefixRequest,
    ) -> Result<MatchExternalKvPrefixResponse, Status> {
        // Bound concurrent prefix queries so routing lookups cannot starve the
        // shared connection's ingest path (see PREFIX_QUERY_MAX_INFLIGHT).
        let _permit = self
            .prefix_semaphore
            .acquire()
            .await
            .map_err(|_| Status::unavailable("prefix query semaphore closed"))?;

        let limit = prefix_limit(req.hashes.len(), req.max_blocks).min(PREFIX_SCAN_CAP);
        let hashes = &req.hashes[..limit];
        if hashes.is_empty() {
            return Ok(MatchExternalKvPrefixResponse::default());
        }

        // Stage 1: the first block's holders bound the candidate set.
        let holders0 = self.placement_holders(&hashes[0]).await?;
        let mut blocks_read = 1u32;
        if holders0.is_empty() {
            return Ok(assemble_prefix_response(Vec::new(), blocks_read));
        }

        // Stage 2: one registry read for those workers; drop unroutable (empty
        // address) ones now (see the proto's worker_address contract).
        let mut addresses = Vec::with_capacity(holders0.len());
        for chunk in holders0.chunks(REDIS_FANOUT_CHUNK) {
            let values =
                try_join_all(chunk.iter().map(|worker| self.worker_address(worker))).await?;
            addresses.extend(values);
        }
        // (worker_id, address, prefix_so_far); every survivor holds block 0.
        let mut active: Vec<(String, String, u32)> = holders0
            .into_iter()
            .zip(addresses)
            .filter(|(_, addr)| !addr.is_empty())
            .map(|(worker, addr)| (worker, addr, 1u32))
            .collect();
        let mut done: Vec<(String, String, u32)> = Vec::new();

        // Stage 3: scan forward, windowed, until candidates or blocks run out.
        let mut idx = 1usize;
        while idx < hashes.len() && !active.is_empty() {
            let end = (idx + REDIS_FANOUT_CHUNK).min(hashes.len());
            let window = &hashes[idx..end];
            let holder_sets = try_join_all(window.iter().map(|hash| async move {
                Ok::<HashSet<String>, Status>(
                    self.placement_holders(hash).await?.into_iter().collect(),
                )
            }))
            .await?;
            blocks_read += window.len() as u32;

            for holders in holder_sets {
                let mut still = Vec::with_capacity(active.len());
                for (worker, addr, prefix) in active.drain(..) {
                    if holders.contains(&worker) {
                        still.push((worker, addr, prefix + 1));
                    } else {
                        done.push((worker, addr, prefix));
                    }
                }
                active = still;
                if active.is_empty() {
                    break;
                }
            }
            idx = end;
        }
        // Survivors reached the scan limit with an unbroken prefix.
        done.append(&mut active);

        Ok(assemble_prefix_response(done, blocks_read))
    }

    async fn do_hit_counts(
        &self,
        req: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        let hashes = dedup_preserve_order(&req.hashes);
        let mut counts = Vec::with_capacity(hashes.len());
        for chunk in hashes.chunks(REDIS_FANOUT_CHUNK) {
            let values = try_join_all(chunk.iter().map(|hash| async move {
                let mut cmd = redis::cmd("HGET");
                cmd.arg(hit_key(&self.ns, hash)).arg("c");
                let v = self.conn.query(cmd).await?;
                let count: Option<i64> = Option::<i64>::from_redis_value(&v)?;
                Ok::<_, redis::RedisError>(count)
            }))
            .await
            .map_err(to_status)?;
            counts.extend(values);
        }

        let entries = hashes
            .into_iter()
            .zip(counts)
            .filter_map(|(hash, count)| {
                count.map(|c| HitCountEntry {
                    hash,
                    hit_count_total: c.max(0) as u64,
                })
            })
            .collect();
        Ok(GetExternalKvHitCountsResponse { entries })
    }
}

#[tonic::async_trait]
impl KvIndexerBackend for RedisKvIndexerBackend {
    async fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status> {
        self.apply(request).await
    }

    async fn match_external_kv(
        &self,
        request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        self.do_match(request).await
    }

    async fn match_external_kv_prefix(
        &self,
        request: MatchExternalKvPrefixRequest,
    ) -> Result<MatchExternalKvPrefixResponse, Status> {
        self.do_match_prefix(request).await
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        self.do_hit_counts(request).await
    }
}

fn dedup_preserve_order(hashes: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    hashes
        .iter()
        .filter(|h| seen.insert(h.as_str().to_string()))
        .cloned()
        .collect()
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn to_status(err: redis::RedisError) -> Status {
    Status::unavailable(format!("redis backend error: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_preserves_first_seen_order() {
        let input = vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
            "c".to_string(),
            "b".to_string(),
        ];
        assert_eq!(dedup_preserve_order(&input), vec!["a", "b", "c"]);
    }
}
