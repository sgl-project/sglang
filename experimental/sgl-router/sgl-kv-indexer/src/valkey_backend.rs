// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Valkey-backed storage backend for the KV Indexer.
//!
//! Placement metadata lives in a Valkey (or Redis) keyspace shared by every
//! Indexer server that points at it. The server process holds no state of its
//! own: restarting it keeps the index, and several servers over one keyspace
//! answer identically, so a fleet can run the Indexer active-active behind the
//! Router instead of "exactly one process".
//!
//! Semantics follow [`crate::InMemoryKvIndexerBackend`] field for field. Prefix
//! answers come from the shared rule engine in `service.rs` over the same
//! placement inputs, so the two backends cannot drift on what a worker is
//! allowed to serve. The parity tests in `tests/valkey_parity.rs` run identical
//! scenario streams against both and compare every response.
//!
//! # Data model
//!
//! Every key carries a configurable prefix (default `{sgl-kv-indexer}:`).
//!
//! | key                     | type | contents                                                |
//! |-------------------------|------|---------------------------------------------------------|
//! | `b:<hash>`              | HASH | `p` parent (`R` = root, `<hash>` = parent; absent = unknown), `r` root hash of the chain, `t` token count, `w:<worker>:<tier>` = component mask per placement |
//! | `c:<hash>`              | SET  | child hashes                                            |
//! | `w:<worker>`            | HASH | `addr` router-facing address, `spec` encoded `WorkerCacheSpec` |
//! | `h:<worker>:<tier>`     | SET  | reverse holdings, drives `CLEAR_ALL_AT_TIER`            |
//! | `hc`                    | HASH | `<hash>` = cumulative hit count                         |
//!
//! Every field write is a single-key atomic command, so two bridges reporting
//! the same popular block never clobber each other: each writes its own
//! `w:<worker>:<tier>` field. A batch is validated against a read snapshot
//! before any command is issued, so a rejected batch leaves the keyspace
//! untouched, matching the in-memory backend's atomic rejection. Accepted
//! batches are pipelined in action order but are not one transaction; a
//! concurrent query may observe a partially applied batch. For REPORT that
//! only under-reports a prefix (the safe direction); for REVOKE the stale
//! placement is visible for one round trip longer, which is already the
//! nature of an event-fed index.
//!
//! # Chain validation without a graph walk
//!
//! The in-memory backend walks parent links to reject a report that would
//! create a cycle. Walking a chain hash by hash over the network would cost a
//! round trip per ancestor, so every block also records `r`, the root of its
//! chain. A batch can only add a parent edge to a block whose parent was
//! unknown, and such a block is the root of its own subtree, so a cycle
//! through existing state exists exactly when the root of a planned parent is
//! itself a planned block. That is a set lookup over the batch's own reads.
//! When a batch attaches an existing subtree under a new parent, the subtree's
//! `r` fields are rewritten by a breadth-first pass over `c:` sets.
//!
//! # Cluster mode
//!
//! Pipelines are routed per slot, so in cluster mode every key must live in one
//! slot: the prefix must carry a hash tag (`{...}`), which the default does.
//! The index is small (about a hundred bytes per block placement), so a single
//! slot is a capacity fit and cluster mode buys failover rather than sharding.

use std::collections::{BTreeMap, HashMap, HashSet};

use redis::aio::ConnectionManager;
use redis::cluster_async::ClusterConnection;
use redis::{FromRedisValue, Pipeline, RedisError, Value};
use tonic::Status;

use crate::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, ExternalKvActionType,
    ExternalKvNodeMatch, GetExternalKvHitCountsRequest, GetExternalKvHitCountsResponse,
    HitCountEntry, MatchExternalKvPrefixRequest, MatchExternalKvPrefixResponse,
    MatchExternalKvRequest, MatchExternalKvResponse, TierHashes, WorkerCacheSpec,
};
use crate::service::{compute_prefix_response, prefix_limit};
use crate::{BlockComponents, KvIndexerBackend, WorkerPrefixInput};

/// Default key prefix. The braces are a cluster hash tag so every key lands in
/// one slot and pipelines stay legal in cluster mode.
pub const DEFAULT_KEY_PREFIX: &str = "{sgl-kv-indexer}:";

/// Commands per pipeline. Bounds the memory one apply or query holds in flight
/// while keeping round trips low: a 16,384-hash batch is a handful of pipelines.
const PIPELINE_CHUNK: usize = 4096;

/// Lua that deletes a block record only if it is still empty: no placements and
/// no children. Runs server-side so a concurrent REPORT of the same block cannot
/// be lost between the check and the delete. Returns the parent hash to continue
/// pruning upward, or an empty string.
///
/// KEYS[1] = block hash key, KEYS[2] = children set key, KEYS[3] = hit count key,
/// ARGV[1] = hash, ARGV[2] = children key prefix (for the parent's set).
const PRUNE_SCRIPT: &str = r#"
local fields = redis.call('HKEYS', KEYS[1])
local placements = 0
for _, f in ipairs(fields) do
  if string.sub(f, 1, 2) == 'w:' then placements = placements + 1 end
end
if placements == 0 then
  redis.call('HDEL', KEYS[3], ARGV[1])
end
if placements > 0 or redis.call('SCARD', KEYS[2]) > 0 then
  return ''
end
local parent = redis.call('HGET', KEYS[1], 'p')
redis.call('DEL', KEYS[1], KEYS[2])
if parent and parent ~= 'R' then
  redis.call('SREM', ARGV[2] .. parent, ARGV[1])
  return parent
end
return ''
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValkeyConfig {
    /// `valkey://host:port/db`, `valkeys://...` (TLS), `valkey+unix:///path`, or
    /// the `redis` spellings of the same. In cluster mode, one or more seed
    /// nodes separated by commas.
    pub url: String,
    /// Prepended to every key. Must contain a `{hash tag}` in cluster mode.
    pub key_prefix: String,
    pub cluster: bool,
}

impl ValkeyConfig {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            key_prefix: DEFAULT_KEY_PREFIX.to_string(),
            cluster: false,
        }
    }

    pub fn with_key_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.key_prefix = prefix.into();
        self
    }

    pub fn with_cluster(mut self, cluster: bool) -> Self {
        self.cluster = cluster;
        self
    }
}

#[derive(Clone)]
enum Conn {
    Standalone(ConnectionManager),
    Cluster(ClusterConnection),
}

impl Conn {
    async fn run<T: FromRedisValue>(&mut self, pipe: &Pipeline) -> Result<T, Status> {
        let result = match self {
            Conn::Standalone(conn) => pipe.query_async::<T>(conn).await,
            Conn::Cluster(conn) => pipe.query_async::<T>(conn).await,
        };
        result.map_err(valkey_error)
    }

    async fn exec(&mut self, pipe: &Pipeline) -> Result<(), Status> {
        let result = match self {
            Conn::Standalone(conn) => pipe.exec_async(conn).await,
            Conn::Cluster(conn) => pipe.exec_async(conn).await,
        };
        result.map_err(valkey_error)
    }
}

fn valkey_error(error: RedisError) -> Status {
    Status::unavailable(format!("valkey backend: {error}"))
}

fn parse_error(what: &str, raw: &str) -> Status {
    Status::internal(format!("valkey backend: malformed {what}: {raw:?}"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParentLink {
    Unknown,
    Root,
    Hash(i64),
}

impl ParentLink {
    fn decode(raw: Option<&str>) -> Result<Self, Status> {
        match raw {
            None => Ok(ParentLink::Unknown),
            Some("R") => Ok(ParentLink::Root),
            Some(hash) => hash
                .parse::<i64>()
                .map(ParentLink::Hash)
                .map_err(|_| parse_error("parent link", hash)),
        }
    }

    fn encode(self) -> Option<String> {
        match self {
            ParentLink::Unknown => None,
            ParentLink::Root => Some("R".to_string()),
            ParentLink::Hash(hash) => Some(hash.to_string()),
        }
    }
}

/// A block record as read from `b:<hash>`.
#[derive(Debug)]
struct BlockRead {
    parent: ParentLink,
    root: Option<i64>,
    token_count: u32,
    /// `(worker, tier) -> component mask`, in field order.
    placements: Vec<((String, i32), u32)>,
}

impl BlockRead {
    fn from_fields(fields: HashMap<String, String>) -> Result<Self, Status> {
        let mut read = BlockRead {
            parent: ParentLink::Unknown,
            root: None,
            token_count: 0,
            placements: Vec::new(),
        };
        for (field, value) in fields {
            match field.as_str() {
                "p" => read.parent = ParentLink::decode(Some(&value))?,
                "r" => {
                    read.root = Some(
                        value
                            .parse::<i64>()
                            .map_err(|_| parse_error("root hash", &value))?,
                    )
                }
                "t" => {
                    read.token_count = value
                        .parse::<u32>()
                        .map_err(|_| parse_error("token count", &value))?
                }
                _ => {
                    if let Some(rest) = field.strip_prefix("w:") {
                        let (worker, tier) = rest
                            .rsplit_once(':')
                            .ok_or_else(|| parse_error("placement field", &field))?;
                        let tier = tier
                            .parse::<i32>()
                            .map_err(|_| parse_error("placement tier", tier))?;
                        let mask = value
                            .parse::<u32>()
                            .map_err(|_| parse_error("component mask", &value))?;
                        read.placements.push(((worker.to_string(), tier), mask));
                    }
                }
            }
        }
        // Field order from HGETALL is hash-table order; sort so responses are
        // deterministic and match the in-memory backend's per-worker grouping.
        read.placements.sort();
        Ok(read)
    }

    fn is_held(&self) -> bool {
        !self.placements.is_empty()
    }

    /// Every created record carries `r`, so its absence means no record.
    fn exists(&self) -> bool {
        self.root.is_some()
    }
}

/// Per tier, the `(hash, component mask, token count)` rows of one match.
type TierPlacements = BTreeMap<i32, Vec<(i64, u32, u32)>>;

#[derive(Debug, Default, Clone)]
struct WorkerMeta {
    address: String,
    spec: Option<WorkerCacheSpec>,
}

fn encode_spec(spec: &WorkerCacheSpec) -> String {
    format!(
        "{},{},{},{},{},{}",
        spec.version,
        spec.components,
        spec.swa_window_tokens,
        spec.full_tier_mask,
        spec.swa_tier_mask,
        spec.mamba_tier_mask
    )
}

fn decode_spec(raw: &str) -> Result<WorkerCacheSpec, Status> {
    let parts: Vec<u32> = raw
        .split(',')
        .map(|part| part.parse::<u32>())
        .collect::<Result<_, _>>()
        .map_err(|_| parse_error("worker cache spec", raw))?;
    if parts.len() != 6 {
        return Err(parse_error("worker cache spec", raw));
    }
    Ok(WorkerCacheSpec {
        version: parts[0],
        components: parts[1],
        swa_window_tokens: parts[2],
        full_tier_mask: parts[3],
        swa_tier_mask: parts[4],
        mamba_tier_mask: parts[5],
    })
}

fn worker_meta_from_fields(fields: HashMap<String, String>) -> Result<WorkerMeta, Status> {
    let address = fields.get("addr").cloned().unwrap_or_default();
    let spec = fields.get("spec").map(|raw| decode_spec(raw)).transpose()?;
    Ok(WorkerMeta { address, spec })
}

fn dedup_preserve_order(hashes: &[i64]) -> Vec<i64> {
    let mut seen = HashSet::new();
    hashes
        .iter()
        .filter(|hash| seen.insert(**hash))
        .copied()
        .collect()
}

fn convert<T: FromRedisValue>(value: Value, what: &str) -> Result<T, Status> {
    T::from_redis_value(value)
        .map_err(|error| Status::internal(format!("valkey backend: unexpected {what}: {error}")))
}

/// Shared-keyspace KV placement index over Valkey.
#[derive(Clone)]
pub struct ValkeyKvIndexerBackend {
    conn: Conn,
    prefix: String,
}

impl std::fmt::Debug for ValkeyKvIndexerBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValkeyKvIndexerBackend")
            .field("prefix", &self.prefix)
            .field(
                "mode",
                &match self.conn {
                    Conn::Standalone(_) => "standalone",
                    Conn::Cluster(_) => "cluster",
                },
            )
            .finish()
    }
}

impl ValkeyKvIndexerBackend {
    /// Connects and returns the backend. The first connection is awaited so a
    /// misconfigured URL fails at startup rather than on the first query.
    pub async fn connect(config: ValkeyConfig) -> Result<Self, Status> {
        if config.cluster && !(config.key_prefix.contains('{') && config.key_prefix.contains('}')) {
            return Err(Status::invalid_argument(
                "valkey backend: cluster mode needs a {hash tag} in the key prefix so pipelines stay in one slot",
            ));
        }
        let conn = if config.cluster {
            let nodes: Vec<String> = config
                .url
                .split(',')
                .map(|node| node.trim().to_string())
                .filter(|node| !node.is_empty())
                .collect();
            let client = redis::cluster::ClusterClient::new(nodes).map_err(valkey_error)?;
            Conn::Cluster(client.get_async_connection().await.map_err(valkey_error)?)
        } else {
            let client = redis::Client::open(config.url.as_str()).map_err(valkey_error)?;
            Conn::Standalone(
                client
                    .get_connection_manager()
                    .await
                    .map_err(valkey_error)?,
            )
        };
        Ok(Self {
            conn,
            prefix: config.key_prefix,
        })
    }

    fn block_key(&self, hash: i64) -> String {
        format!("{}b:{hash}", self.prefix)
    }

    fn children_key(&self, hash: i64) -> String {
        format!("{}c:{hash}", self.prefix)
    }

    fn children_prefix(&self) -> String {
        format!("{}c:", self.prefix)
    }

    fn worker_key(&self, worker: &str) -> String {
        format!("{}w:{worker}", self.prefix)
    }

    fn holdings_key(&self, worker: &str, tier: i32) -> String {
        format!("{}h:{worker}:{tier}", self.prefix)
    }

    fn hits_key(&self) -> String {
        format!("{}hc", self.prefix)
    }

    fn placement_field(worker: &str, tier: i32) -> String {
        format!("w:{worker}:{tier}")
    }

    /// Runs `pipe` in [`PIPELINE_CHUNK`]-sized slices, in order, collecting every
    /// reply. `commands` must equal the number of un-ignored commands queued.
    async fn run_chunked(
        &self,
        build: impl Fn(&mut Pipeline, std::ops::Range<usize>),
        commands: usize,
    ) -> Result<Vec<Value>, Status> {
        let mut conn = self.conn.clone();
        let mut out = Vec::with_capacity(commands);
        let mut start = 0;
        while start < commands {
            let end = (start + PIPELINE_CHUNK).min(commands);
            let mut pipe = redis::pipe();
            build(&mut pipe, start..end);
            let replies: Vec<Value> = conn.run(&pipe).await?;
            out.extend(replies);
            start = end;
        }
        Ok(out)
    }

    /// Reads full block records for `hashes`, aligned with the input.
    async fn read_blocks(&self, hashes: &[i64]) -> Result<Vec<BlockRead>, Status> {
        let replies = self
            .run_chunked(
                |pipe, range| {
                    for hash in &hashes[range] {
                        pipe.cmd("HGETALL").arg(self.block_key(*hash));
                    }
                },
                hashes.len(),
            )
            .await?;
        replies
            .into_iter()
            .map(|value| BlockRead::from_fields(convert(value, "block record")?))
            .collect()
    }

    /// One pipeline: full records for `hashes` and the holdings of `worker` at
    /// each of `clear_tiers`.
    async fn read_touched(
        &self,
        worker: &str,
        hashes: &[i64],
        clear_tiers: &[i32],
    ) -> Result<(HashMap<i64, BlockRead>, HashMap<i32, HashSet<i64>>), Status> {
        let commands = hashes.len() + clear_tiers.len();
        let replies = self
            .run_chunked(
                |pipe, range| {
                    for index in range {
                        if index < hashes.len() {
                            pipe.cmd("HGETALL").arg(self.block_key(hashes[index]));
                        } else {
                            let tier = clear_tiers[index - hashes.len()];
                            pipe.cmd("SMEMBERS").arg(self.holdings_key(worker, tier));
                        }
                    }
                },
                commands,
            )
            .await?;
        let mut blocks = HashMap::with_capacity(hashes.len());
        let mut holdings = HashMap::with_capacity(clear_tiers.len());
        for (index, value) in replies.into_iter().enumerate() {
            if index < hashes.len() {
                let read = BlockRead::from_fields(convert(value, "block record")?)?;
                blocks.insert(hashes[index], read);
            } else {
                let members: Vec<i64> = convert(value, "holdings set")?;
                holdings.insert(
                    clear_tiers[index - hashes.len()],
                    members.into_iter().collect(),
                );
            }
        }
        Ok((blocks, holdings))
    }

    async fn read_workers(&self, workers: &[String]) -> Result<Vec<WorkerMeta>, Status> {
        let replies = self
            .run_chunked(
                |pipe, range| {
                    for worker in &workers[range] {
                        pipe.cmd("HGETALL").arg(self.worker_key(worker));
                    }
                },
                workers.len(),
            )
            .await?;
        replies
            .into_iter()
            .map(|value| worker_meta_from_fields(convert(value, "worker record")?))
            .collect()
    }

    /// Issues write commands in [`PIPELINE_CHUNK`] slices, preserving order.
    async fn write_all(&self, commands: Vec<redis::Cmd>) -> Result<(), Status> {
        let mut conn = self.conn.clone();
        for chunk in commands.chunks(PIPELINE_CHUNK) {
            let mut pipe = redis::pipe();
            for cmd in chunk {
                pipe.add_command(cmd.clone()).ignore();
            }
            conn.exec(&pipe).await?;
        }
        Ok(())
    }

    // ---- apply ---------------------------------------------------------------

    async fn apply(
        &self,
        req: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status> {
        let worker_id = req.worker_id;

        // Phase 1: plan the chain edges this batch introduces, purely in memory.
        // A hash may appear in several REPORT actions only with the same parent.
        let mut planned_parents: HashMap<i64, ParentLink> = HashMap::new();
        let mut planned_order: Vec<i64> = Vec::new();
        let mut clear_tiers: Vec<i32> = Vec::new();
        for action in &req.actions {
            match ExternalKvActionType::try_from(action.r#type) {
                Ok(ExternalKvActionType::ActionReport) => {
                    let mut parent = action
                        .parent_block_hash
                        .map_or(ParentLink::Root, ParentLink::Hash);
                    for hash in &action.hashes {
                        if parent == ParentLink::Hash(*hash) {
                            return Err(Status::invalid_argument(
                                "block hash cannot be its own parent",
                            ));
                        }
                        match planned_parents.get(hash) {
                            Some(existing) if *existing != parent => {
                                return Err(Status::invalid_argument(format!(
                                    "block hash {hash} was reported with conflicting parents"
                                )));
                            }
                            Some(_) => {}
                            None => {
                                planned_parents.insert(*hash, parent);
                                planned_order.push(*hash);
                            }
                        }
                        parent = ParentLink::Hash(*hash);
                    }
                }
                Ok(ExternalKvActionType::ActionRevoke) => {}
                Ok(ExternalKvActionType::ActionClearAllAtTier) => clear_tiers.push(action.tier),
                Ok(ExternalKvActionType::ActionUnknown) | Err(_) => {
                    return Err(Status::invalid_argument("unsupported action type"));
                }
            }
        }

        // Phase 2: one read snapshot of every block the batch touches (parent
        // link, root, placements) plus the holdings each CLEAR_ALL_AT_TIER drains.
        let mut touched: Vec<i64> = planned_order.clone();
        for parent in planned_parents.values() {
            if let ParentLink::Hash(parent) = parent {
                touched.push(*parent);
            }
        }
        for action in &req.actions {
            if ExternalKvActionType::try_from(action.r#type)
                == Ok(ExternalKvActionType::ActionRevoke)
            {
                touched.extend(action.hashes.iter().copied());
            }
        }
        let touched = dedup_preserve_order(&touched);
        clear_tiers.sort_unstable();
        clear_tiers.dedup();
        let (mut blocks, mut holdings) = self
            .read_touched(&worker_id, &touched, &clear_tiers)
            .await?;
        // Blocks a CLEAR will revoke that no other action named.
        let extra: Vec<i64> = holdings
            .values()
            .flatten()
            .copied()
            .filter(|hash| !blocks.contains_key(hash))
            .collect();
        let extra = dedup_preserve_order(&extra);
        if !extra.is_empty() {
            for (hash, read) in extra.iter().zip(self.read_blocks(&extra).await?) {
                blocks.insert(*hash, read);
            }
        }
        let root_of_existing = |hash: i64| {
            blocks
                .get(&hash)
                .and_then(|block| block.root)
                .unwrap_or(hash)
        };

        // Phase 3: validate against the snapshot. Nothing has been written yet,
        // so a rejection leaves the keyspace exactly as it was.
        for hash in &planned_order {
            let planned = planned_parents[hash];
            let existing = blocks[hash].parent;
            if existing != ParentLink::Unknown && existing != planned {
                return Err(Status::invalid_argument(format!(
                    "block hash {hash} was reported with conflicting parents"
                )));
            }
        }
        // Cycle check: follow planned edges through the batch; on leaving it into
        // existing state, jump to that subtree's root, the only node that can lead back.
        for start in &planned_order {
            let mut on_path = HashSet::new();
            let mut current = *start;
            loop {
                if !on_path.insert(current) {
                    return Err(Status::invalid_argument(
                        "report would create a parent cycle",
                    ));
                }
                let parent = match planned_parents.get(&current) {
                    Some(parent) => *parent,
                    None => break,
                };
                match parent {
                    ParentLink::Unknown | ParentLink::Root => break,
                    ParentLink::Hash(parent) => {
                        if planned_parents.contains_key(&parent) {
                            current = parent;
                        } else {
                            let root = root_of_existing(parent);
                            if planned_parents.contains_key(&root) {
                                current = root;
                            } else {
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Roots after this batch. A planned block whose parent is outside the
        // batch inherits that parent's root; inside the batch, its parent's new root.
        let mut roots: HashMap<i64, i64> = HashMap::new();
        fn root_of(
            hash: i64,
            planned: &HashMap<i64, ParentLink>,
            existing: &dyn Fn(i64) -> i64,
            roots: &mut HashMap<i64, i64>,
        ) -> i64 {
            if let Some(root) = roots.get(&hash) {
                return *root;
            }
            let root = match planned.get(&hash) {
                Some(ParentLink::Hash(parent)) => root_of(*parent, planned, existing, roots),
                Some(ParentLink::Root) | Some(ParentLink::Unknown) => hash,
                None => existing(hash),
            };
            roots.insert(hash, root);
            root
        }
        // Acyclicity was verified above, so this recursion terminates.
        for hash in &planned_order {
            root_of(*hash, &planned_parents, &root_of_existing, &mut roots);
        }

        // Phase 4: writes, in action order.
        let mut commands: Vec<redis::Cmd> = Vec::new();
        // Address and spec are snapshots carried on every batch; an empty address
        // makes the worker unroutable and an absent spec returns it to legacy.
        let mut cmd = redis::cmd("HSET");
        cmd.arg(self.worker_key(&worker_id))
            .arg("addr")
            .arg(&req.worker_address);
        commands.push(cmd);
        commands.push(match &req.cache_spec {
            Some(spec) => {
                let mut cmd = redis::cmd("HSET");
                cmd.arg(self.worker_key(&worker_id))
                    .arg("spec")
                    .arg(encode_spec(spec));
                cmd
            }
            None => {
                let mut cmd = redis::cmd("HDEL");
                cmd.arg(self.worker_key(&worker_id)).arg("spec");
                cmd
            }
        });
        // Existing blocks that gain a parent edge here; their subtrees need `r`
        // rewritten after the pipeline.
        let mut relink: Vec<(i64, i64)> = Vec::new();
        let mut written_parent: HashSet<i64> = HashSet::new();
        let mut revoked: Vec<i64> = Vec::new();
        // In-batch view of holdings so CLEAR_ALL_AT_TIER sees earlier actions.
        let mut holdings_delta: HashMap<i32, (HashSet<i64>, HashSet<i64>)> = HashMap::new();
        // In-batch view of each block's placements; the reference drops a hit
        // count at the revoke that empties the block, before later actions run.
        let mut live: HashMap<i64, HashSet<(String, i32)>> = HashMap::new();
        fn live_placements<'a>(
            live: &'a mut HashMap<i64, HashSet<(String, i32)>>,
            blocks: &HashMap<i64, BlockRead>,
            hash: i64,
        ) -> &'a mut HashSet<(String, i32)> {
            live.entry(hash).or_insert_with(|| {
                blocks
                    .get(&hash)
                    .map(|block| {
                        block
                            .placements
                            .iter()
                            .map(|(key, _)| key.clone())
                            .collect()
                    })
                    .unwrap_or_default()
            })
        }

        for action in &req.actions {
            match ExternalKvActionType::try_from(action.r#type) {
                Ok(ExternalKvActionType::ActionReport) => {
                    let has_masks = !action.component_masks.is_empty();
                    let has_sizes = !action.block_sizes.is_empty();
                    let mut parent = action
                        .parent_block_hash
                        .map_or(ParentLink::Root, ParentLink::Hash);
                    for (index, hash) in action.hashes.iter().copied().enumerate() {
                        let snap = &blocks[&hash];
                        let root = roots[&hash];
                        if written_parent.insert(hash) {
                            let mut cmd = redis::cmd("HSET");
                            cmd.arg(self.block_key(hash)).arg("r").arg(root);
                            if let Some(encoded) = parent.encode() {
                                cmd.arg("p").arg(encoded);
                            }
                            commands.push(cmd);
                            if let ParentLink::Hash(parent_hash) = parent {
                                let mut cmd = redis::cmd("SADD");
                                cmd.arg(self.children_key(parent_hash)).arg(hash);
                                commands.push(cmd);
                                // A parent only ever referenced gets a record so
                                // the chain is walkable and prunable.
                                if !planned_parents.contains_key(&parent_hash)
                                    && !blocks.get(&parent_hash).is_some_and(BlockRead::exists)
                                    && written_parent.insert(parent_hash)
                                {
                                    let mut cmd = redis::cmd("HSET");
                                    cmd.arg(self.block_key(parent_hash))
                                        .arg("r")
                                        .arg(parent_hash);
                                    commands.push(cmd);
                                }
                            }
                            if snap.exists()
                                && snap.parent == ParentLink::Unknown
                                && parent != ParentLink::Unknown
                                && root != hash
                            {
                                relink.push((hash, root));
                            }
                        }
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
                        let mut cmd = redis::cmd("HSET");
                        cmd.arg(self.block_key(hash))
                            .arg(Self::placement_field(&worker_id, action.tier))
                            .arg(mask);
                        if token_count > 0 {
                            cmd.arg("t").arg(token_count);
                        }
                        commands.push(cmd);
                        let mut cmd = redis::cmd("SADD");
                        cmd.arg(self.holdings_key(&worker_id, action.tier))
                            .arg(hash);
                        commands.push(cmd);
                        let (added, removed) = holdings_delta.entry(action.tier).or_default();
                        added.insert(hash);
                        removed.remove(&hash);
                        live_placements(&mut live, &blocks, hash)
                            .insert((worker_id.clone(), action.tier));
                        parent = ParentLink::Hash(hash);
                    }
                }
                Ok(ExternalKvActionType::ActionRevoke) => {
                    for hash in action.hashes.iter().copied() {
                        self.push_revoke(&mut commands, &worker_id, hash, action.tier);
                        let set = live_placements(&mut live, &blocks, hash);
                        set.remove(&(worker_id.clone(), action.tier));
                        if set.is_empty() && blocks.get(&hash).is_some_and(BlockRead::exists) {
                            let mut cmd = redis::cmd("HDEL");
                            cmd.arg(self.hits_key()).arg(hash);
                            commands.push(cmd);
                        }
                        revoked.push(hash);
                        let (added, removed) = holdings_delta.entry(action.tier).or_default();
                        added.remove(&hash);
                        removed.insert(hash);
                    }
                }
                Ok(ExternalKvActionType::ActionClearAllAtTier) => {
                    let mut hashes: Vec<i64> = holdings
                        .get(&action.tier)
                        .map(|set| set.iter().copied().collect())
                        .unwrap_or_default();
                    if let Some((added, removed)) = holdings_delta.get(&action.tier) {
                        hashes.retain(|hash| !removed.contains(hash));
                        hashes.extend(added.iter().copied());
                    }
                    for hash in dedup_preserve_order(&hashes) {
                        self.push_revoke(&mut commands, &worker_id, hash, action.tier);
                        let set = live_placements(&mut live, &blocks, hash);
                        set.remove(&(worker_id.clone(), action.tier));
                        if set.is_empty() && blocks.get(&hash).is_some_and(BlockRead::exists) {
                            let mut cmd = redis::cmd("HDEL");
                            cmd.arg(self.hits_key()).arg(hash);
                            commands.push(cmd);
                        }
                        revoked.push(hash);
                    }
                    holdings_delta.remove(&action.tier);
                    holdings.remove(&action.tier);
                }
                Ok(ExternalKvActionType::ActionUnknown) | Err(_) => unreachable!("validated above"),
            }
        }
        self.write_all(commands).await?;

        // Phase 5: subtree root rewrites, then prune blocks left empty.
        for (hash, root) in relink {
            self.rewrite_subtree_root(hash, root).await?;
        }
        self.prune(dedup_preserve_order(&revoked)).await?;

        Ok(ApplyExternalKvBatchResponse {})
    }

    fn push_revoke(&self, commands: &mut Vec<redis::Cmd>, worker_id: &str, hash: i64, tier: i32) {
        let mut cmd = redis::cmd("HDEL");
        cmd.arg(self.block_key(hash))
            .arg(Self::placement_field(worker_id, tier));
        commands.push(cmd);
        let mut cmd = redis::cmd("SREM");
        cmd.arg(self.holdings_key(worker_id, tier)).arg(hash);
        commands.push(cmd);
    }

    /// Sets `r = root` on every descendant of `hash`, one pipelined BFS level at
    /// a time. Only runs when a batch attaches an existing subtree under a new
    /// parent, which a root-first bridge never does in steady state.
    async fn rewrite_subtree_root(&self, hash: i64, root: i64) -> Result<(), Status> {
        let mut frontier = vec![hash];
        let mut visited = HashSet::new();
        while !frontier.is_empty() {
            let replies = self
                .run_chunked(
                    |pipe, range| {
                        for parent in &frontier[range] {
                            pipe.cmd("SMEMBERS").arg(self.children_key(*parent));
                        }
                    },
                    frontier.len(),
                )
                .await?;
            let mut next = Vec::new();
            for value in replies {
                let children: Vec<i64> = convert(value, "children set")?;
                for child in children {
                    if visited.insert(child) && child != hash {
                        next.push(child);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            let commands = next
                .iter()
                .map(|child| {
                    let mut cmd = redis::cmd("HSET");
                    cmd.arg(self.block_key(*child)).arg("r").arg(root);
                    cmd
                })
                .collect();
            self.write_all(commands).await?;
            frontier = next;
        }
        Ok(())
    }

    /// Deletes block records that lost their last placement and have no
    /// children, walking up parents that become empty in turn. Each step is a
    /// server-side check-and-delete, so a concurrent re-report survives.
    async fn prune(&self, mut candidates: Vec<i64>) -> Result<(), Status> {
        // A parent re-enters as a candidate each time a child is deleted, so a
        // block judged non-empty early in a round is re-checked; every round deletes.
        while !candidates.is_empty() {
            let replies = self
                .run_chunked(
                    |pipe, range| {
                        for hash in &candidates[range] {
                            pipe.cmd("EVAL")
                                .arg(PRUNE_SCRIPT)
                                .arg(3)
                                .arg(self.block_key(*hash))
                                .arg(self.children_key(*hash))
                                .arg(self.hits_key())
                                .arg(*hash)
                                .arg(self.children_prefix());
                        }
                    },
                    candidates.len(),
                )
                .await?;
            let mut next = Vec::new();
            for value in replies {
                let parent: String = convert(value, "prune result")?;
                if !parent.is_empty() {
                    next.push(
                        parent
                            .parse::<i64>()
                            .map_err(|_| parse_error("pruned parent", &parent))?,
                    );
                }
            }
            candidates = dedup_preserve_order(&next);
        }
        Ok(())
    }

    // ---- queries -------------------------------------------------------------

    /// Reads the blocks and the metadata of every worker placed on them.
    /// Returns `(blocks aligned with hashes, worker metadata by id, worker order
    /// by first appearance)`.
    async fn read_placements(
        &self,
        hashes: &[i64],
    ) -> Result<(Vec<BlockRead>, HashMap<String, WorkerMeta>, Vec<String>), Status> {
        let blocks = self.read_blocks(hashes).await?;
        let mut order: Vec<String> = Vec::new();
        let mut seen: HashSet<&str> = HashSet::new();
        for block in &blocks {
            for ((worker, _), _) in &block.placements {
                if seen.insert(worker.as_str()) {
                    order.push(worker.clone());
                }
            }
        }
        let metas = self.read_workers(&order).await?;
        let by_worker = order.iter().cloned().zip(metas).collect();
        Ok((blocks, by_worker, order))
    }

    async fn do_match(
        &self,
        req: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        let hashes = dedup_preserve_order(&req.hashes);
        let (blocks, metas, order) = self.read_placements(&hashes).await?;

        let mut by_worker: HashMap<&str, TierPlacements> = HashMap::new();
        let mut matched_hashes = Vec::new();
        for (hash, block) in hashes.iter().zip(&blocks) {
            if !block.is_held() {
                continue;
            }
            matched_hashes.push(*hash);
            for ((worker, tier), mask) in &block.placements {
                by_worker
                    .entry(worker.as_str())
                    .or_default()
                    .entry(*tier)
                    .or_default()
                    .push((*hash, *mask, block.token_count));
            }
        }

        if req.count_as_hit && !matched_hashes.is_empty() {
            let commands = matched_hashes
                .iter()
                .map(|hash| {
                    let mut cmd = redis::cmd("HINCRBY");
                    cmd.arg(self.hits_key()).arg(*hash).arg(1);
                    cmd
                })
                .collect();
            self.write_all(commands).await?;
        }

        let matches = order
            .iter()
            .filter_map(|worker| {
                let tiers = by_worker.remove(worker.as_str())?;
                let meta = metas.get(worker).cloned().unwrap_or_default();
                Some(ExternalKvNodeMatch {
                    worker_id: worker.clone(),
                    address: meta.address,
                    hashes_by_tier: tiers
                        .into_iter()
                        .map(|(tier, placements)| TierHashes {
                            tier,
                            hashes: placements.iter().map(|(hash, _, _)| *hash).collect(),
                            component_masks: placements.iter().map(|(_, mask, _)| *mask).collect(),
                            block_sizes: placements.into_iter().map(|(_, _, size)| size).collect(),
                        })
                        .collect(),
                })
            })
            .collect();
        Ok(MatchExternalKvResponse { matches })
    }

    fn prefix_inputs(
        hashes: &[i64],
        blocks: &[BlockRead],
        metas: &HashMap<String, WorkerMeta>,
        order: &[String],
    ) -> Vec<WorkerPrefixInput> {
        let index_of: HashMap<&str, usize> = order
            .iter()
            .enumerate()
            .map(|(index, worker)| (worker.as_str(), index))
            .collect();
        let mut inputs: Vec<WorkerPrefixInput> = order
            .iter()
            .map(|worker| {
                let meta = metas.get(worker).cloned().unwrap_or_default();
                WorkerPrefixInput {
                    worker_id: worker.clone(),
                    address: meta.address,
                    spec: meta.spec,
                    blocks: vec![None; hashes.len()],
                }
            })
            .collect();
        for (position, block) in blocks.iter().enumerate() {
            for ((worker, tier), mask) in &block.placements {
                let index = index_of[worker.as_str()];
                let components =
                    inputs[index].blocks[position].get_or_insert_with(|| BlockComponents {
                        token_count: block.token_count,
                        tier_masks: Vec::new(),
                    });
                components.tier_masks.push((*tier, *mask));
            }
        }
        inputs
    }

    async fn do_hit_counts(
        &self,
        req: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        let hashes = dedup_preserve_order(&req.hashes);
        let mut conn = self.conn.clone();
        let mut counts: Vec<Option<u64>> = Vec::with_capacity(hashes.len());
        for chunk in hashes.chunks(PIPELINE_CHUNK) {
            let mut pipe = redis::pipe();
            let cmd = pipe.cmd("HMGET");
            cmd.arg(self.hits_key());
            for hash in chunk {
                cmd.arg(*hash);
            }
            let replies: Vec<Vec<Option<u64>>> = conn.run(&pipe).await?;
            counts.extend(replies.into_iter().flatten());
        }
        let entries = hashes
            .into_iter()
            .zip(counts)
            .filter_map(|(hash, count)| {
                count.map(|hit_count_total| HitCountEntry {
                    hash,
                    hit_count_total,
                })
            })
            .collect();
        Ok(GetExternalKvHitCountsResponse { entries })
    }
}

#[tonic::async_trait]
impl KvIndexerBackend for ValkeyKvIndexerBackend {
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

    async fn collect_worker_prefix_inputs(
        &self,
        hashes: &[i64],
    ) -> Result<Vec<WorkerPrefixInput>, Status> {
        let (blocks, metas, order) = self.read_placements(hashes).await?;
        Ok(Self::prefix_inputs(hashes, &blocks, &metas, &order))
    }

    async fn match_external_kv_prefix(
        &self,
        request: MatchExternalKvPrefixRequest,
    ) -> Result<MatchExternalKvPrefixResponse, Status> {
        let limit = prefix_limit(request.hashes.len(), request.max_blocks);
        let hashes = &request.hashes[..limit];
        if hashes.is_empty() {
            return Ok(MatchExternalKvPrefixResponse::default());
        }
        // Only holders of the first block can own a prefix: read it alone first
        // so a miss costs one round trip and reports `blocks_read = 1`.
        let first = self.read_blocks(&hashes[..1]).await?;
        if !first[0].is_held() {
            return Ok(MatchExternalKvPrefixResponse {
                matches: Vec::new(),
                best_prefix_blocks: 0,
                blocks_read: 1,
            });
        }
        let (blocks, metas, order) = self.read_placements(hashes).await?;
        let inputs = Self::prefix_inputs(hashes, &blocks, &metas, &order);
        Ok(compute_prefix_response(&inputs, limit as u32))
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        self.do_hit_counts(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_round_trips() {
        let spec = WorkerCacheSpec {
            version: 1,
            components: 3,
            swa_window_tokens: 4096,
            full_tier_mask: 6,
            swa_tier_mask: 2,
            mamba_tier_mask: 0,
        };
        assert_eq!(decode_spec(&encode_spec(&spec)).unwrap(), spec);
        assert!(decode_spec("1,2,3").is_err());
        assert!(decode_spec("a,b,c,d,e,f").is_err());
    }

    #[test]
    fn parent_link_round_trips() {
        for link in [ParentLink::Root, ParentLink::Hash(-42), ParentLink::Hash(7)] {
            let encoded = link.encode();
            assert_eq!(ParentLink::decode(encoded.as_deref()).unwrap(), link);
        }
        assert_eq!(ParentLink::decode(None).unwrap(), ParentLink::Unknown);
        assert!(ParentLink::decode(Some("nope")).is_err());
    }

    #[test]
    fn block_read_parses_placement_fields_with_colons_in_worker_ids() {
        let mut fields = HashMap::new();
        fields.insert("p".to_string(), "R".to_string());
        fields.insert("r".to_string(), "5".to_string());
        fields.insert("t".to_string(), "64".to_string());
        fields.insert("w:http://10.0.0.1:30000:1".to_string(), "1".to_string());
        fields.insert("w:b:2".to_string(), "0".to_string());
        let read = BlockRead::from_fields(fields).unwrap();
        assert_eq!(read.parent, ParentLink::Root);
        assert_eq!(read.root, Some(5));
        assert_eq!(read.token_count, 64);
        assert_eq!(
            read.placements,
            vec![
                (("b".to_string(), 2), 0),
                (("http://10.0.0.1:30000".to_string(), 1), 1),
            ]
        );
    }

    #[test]
    fn cluster_mode_requires_hash_tag() {
        let config = ValkeyConfig::new("valkey://127.0.0.1:1")
            .with_cluster(true)
            .with_key_prefix("plain:");
        let error = futures_block(ValkeyKvIndexerBackend::connect(config)).unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }

    fn futures_block<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }
}
