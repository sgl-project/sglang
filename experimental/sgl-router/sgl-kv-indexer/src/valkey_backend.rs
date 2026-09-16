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
//! | `workers`               | SET  | every worker id that ever applied a batch               |
//!
//! The event log (`events` stream, `lease:events`) lives in `stream.rs` and the
//! liveness keys (`alive:<worker>`, `hb:<worker>`) in `liveness.rs`, under the
//! same prefix.
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
//! # Pruning
//!
//! A block that lost its last placement and has no children is deleted by a
//! server-side script (loaded once, invoked by `EVALSHA`) so a concurrent
//! re-report of the same block cannot be lost between the check and the
//! delete. Every key the script touches is declared in `KEYS`; it returns the
//! deleted block's grandparent link so the walk up the chain stays declared
//! one level at a time.
//!
//! # Cluster mode
//!
//! Pipelines are routed per slot, so in cluster mode every key must live in one
//! slot: the prefix must carry a hash tag (`{...}`), which the default does.
//! The index is small (about a hundred bytes per block placement), so a single
//! slot is a capacity fit and cluster mode buys failover rather than sharding.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use redis::aio::{ConnectionManager, ConnectionManagerConfig};
use redis::cluster::ClusterClientBuilder;
use redis::cluster_async::ClusterConnection;
use redis::{FromRedisValue, Pipeline, RedisError, Script, Value};
use tonic::Status;

use crate::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, ExternalKvAction,
    ExternalKvActionType, ExternalKvNodeMatch, GetExternalKvHitCountsRequest,
    GetExternalKvHitCountsResponse, HitCountEntry, MatchExternalKvPrefixRequest,
    MatchExternalKvPrefixResponse, MatchExternalKvRequest, MatchExternalKvResponse, TierHashes,
    TierType, WorkerCacheSpec,
};
use crate::service::{compute_prefix_response, prefix_limit};
use crate::{BlockComponents, KvIndexerBackend, WorkerPrefixInput};

/// Default key prefix. The braces are a cluster hash tag so every key lands in
/// one slot and pipelines stay legal in cluster mode.
pub const DEFAULT_KEY_PREFIX: &str = "{sgl-kv-indexer}:";

/// Per-command response deadline. Well above a healthy round trip and below
/// the Router's default 100 ms query deadline times its retry-free fallback, so
/// a stalled Valkey surfaces as `Unavailable` instead of a hung apply.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(1);

/// Deadline for establishing a connection at startup or after a drop.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Commands per pipeline. Bounds the memory one apply or query holds in flight
/// while keeping round trips low: a 16,384-hash batch is a handful of pipelines.
const PIPELINE_CHUNK: usize = 4096;

/// KEYS[1] block, KEYS[2] its children set, KEYS[3] hit counts; when the block
/// has a known parent also KEYS[4] the parent's children set and KEYS[5] the
/// parent's block. ARGV[1] is the block hash. Returns nil when nothing was
/// deleted or the block had no parent, else the parent's own parent link
/// (`R`, a hash, or `U` for unknown) so the caller can continue upward.
const PRUNE_LUA: &str = r#"
local fields = redis.call('HKEYS', KEYS[1])
local placements = 0
for _, f in ipairs(fields) do
  if string.sub(f, 1, 2) == 'w:' then placements = placements + 1 end
end
if placements == 0 then
  redis.call('HDEL', KEYS[3], ARGV[1])
end
if placements > 0 or redis.call('SCARD', KEYS[2]) > 0 then
  return false
end
redis.call('DEL', KEYS[1], KEYS[2])
if #KEYS < 5 then
  return false
end
redis.call('SREM', KEYS[4], ARGV[1])
return redis.call('HGET', KEYS[5], 'p') or 'U'
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
    pub request_timeout: Duration,
    pub connect_timeout: Duration,
}

impl ValkeyConfig {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            key_prefix: DEFAULT_KEY_PREFIX.to_string(),
            cluster: false,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
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

    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }
}

/// One multiplexed connection, standalone or cluster. Cheap to clone; every
/// clone shares the underlying socket.
#[derive(Clone)]
pub(crate) enum Conn {
    Standalone(ConnectionManager),
    Cluster(ClusterConnection),
}

impl Conn {
    pub(crate) async fn run<T: FromRedisValue>(&mut self, pipe: &Pipeline) -> Result<T, Status> {
        let result = match self {
            Conn::Standalone(conn) => pipe.query_async::<T>(conn).await,
            Conn::Cluster(conn) => pipe.query_async::<T>(conn).await,
        };
        result.map_err(valkey_error)
    }

    pub(crate) async fn exec(&mut self, pipe: &Pipeline) -> Result<(), Status> {
        let result = match self {
            Conn::Standalone(conn) => pipe.exec_async(conn).await,
            Conn::Cluster(conn) => pipe.exec_async(conn).await,
        };
        result.map_err(valkey_error)
    }
}

/// Opens a connection per `config`, awaited so a bad URL fails at startup.
pub(crate) async fn connect_conn(config: &ValkeyConfig) -> Result<Conn, Status> {
    if config.cluster && !(config.key_prefix.contains('{') && config.key_prefix.contains('}')) {
        return Err(Status::invalid_argument(
            "valkey backend: cluster mode needs a {hash tag} in the key prefix so pipelines stay in one slot",
        ));
    }
    if config.cluster {
        let client = ClusterClientBuilder::new(cluster_nodes(&config.url))
            .connection_timeout(config.connect_timeout)
            .response_timeout(config.request_timeout)
            .build()
            .map_err(valkey_error)?;
        Ok(Conn::Cluster(
            client.get_async_connection().await.map_err(valkey_error)?,
        ))
    } else {
        let client = redis::Client::open(config.url.as_str()).map_err(valkey_error)?;
        // Three reconnect attempts, then commands fail as `Unavailable` and the
        // next command starts a fresh attempt; the Router falls back meanwhile.
        let manager = ConnectionManagerConfig::new()
            .set_connection_timeout(Some(config.connect_timeout))
            .set_response_timeout(Some(config.request_timeout))
            .set_number_of_retries(3)
            .set_max_delay(Duration::from_secs(1));
        Ok(Conn::Standalone(
            client
                .get_connection_manager_with_config(manager)
                .await
                .map_err(valkey_error)?,
        ))
    }
}

/// The set every worker that ever reported a placement is added to.
pub(crate) fn workers_key(prefix: &str) -> String {
    format!("{prefix}workers")
}

pub(crate) fn cluster_nodes(url: &str) -> Vec<String> {
    url.split(',')
        .map(|node| node.trim().to_string())
        .filter(|node| !node.is_empty())
        .collect()
}

pub(crate) fn valkey_status(error: RedisError) -> Status {
    valkey_error(error)
}

/// Transport failures are `Unavailable`, which the Router treats as "index
/// unreachable" and falls back on. Anything else is a server-side fault.
fn valkey_error(error: RedisError) -> Status {
    let transport = error.is_io_error()
        || error.is_timeout()
        || error.is_connection_dropped()
        || error.is_connection_refusal()
        || error.is_cluster_error();
    if transport {
        Status::unavailable(format!("valkey backend: {error}"))
    } else {
        Status::internal(format!("valkey backend: {error}"))
    }
}

fn parse_error(what: &str, raw: &str) -> Status {
    Status::internal(format!("valkey backend: malformed {what}: {raw:?}"))
}

fn missing(what: &str, hash: i64) -> Status {
    Status::internal(format!(
        "valkey backend: {what} for block {hash} missing from snapshot"
    ))
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
            None | Some("U") => Ok(ParentLink::Unknown),
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
    /// `(worker, tier) -> component mask`, sorted.
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
        // HGETALL field order is hash-table order; sort so responses are
        // deterministic and grouped like the in-memory backend's.
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

/// The chain edges one batch introduces, validated for internal consistency
/// before any read: a hash may appear in several REPORT actions only with the
/// same parent, and never as its own parent.
#[derive(Debug)]
struct BatchPlan {
    parents: HashMap<i64, ParentLink>,
    /// Planned hashes in first-appearance order.
    order: Vec<i64>,
    clear_tiers: Vec<i32>,
}

fn plan_batch(actions: &[ExternalKvAction]) -> Result<BatchPlan, Status> {
    let mut plan = BatchPlan {
        parents: HashMap::new(),
        order: Vec::new(),
        clear_tiers: Vec::new(),
    };
    for action in actions {
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
                    match plan.parents.get(hash) {
                        Some(existing) if *existing != parent => {
                            return Err(Status::invalid_argument(format!(
                                "block hash {hash} was reported with conflicting parents"
                            )));
                        }
                        Some(_) => {}
                        None => {
                            plan.parents.insert(*hash, parent);
                            plan.order.push(*hash);
                        }
                    }
                    parent = ParentLink::Hash(*hash);
                }
            }
            Ok(ExternalKvActionType::ActionRevoke) => {}
            Ok(ExternalKvActionType::ActionClearAllAtTier) => plan.clear_tiers.push(action.tier),
            Ok(ExternalKvActionType::ActionUnknown) | Err(_) => {
                return Err(Status::invalid_argument("unsupported action type"));
            }
        }
    }
    plan.clear_tiers.sort_unstable();
    plan.clear_tiers.dedup();
    Ok(plan)
}

/// What the keyspace held, for every block a batch touches, before the batch
/// wrote anything.
struct Snapshot {
    blocks: HashMap<i64, BlockRead>,
    /// Holdings of the reporting worker at each tier a CLEAR_ALL_AT_TIER names.
    holdings: HashMap<i32, HashSet<i64>>,
}

impl Snapshot {
    fn block(&self, hash: i64) -> Result<&BlockRead, Status> {
        self.blocks
            .get(&hash)
            .ok_or_else(|| missing("record", hash))
    }

    fn exists(&self, hash: i64) -> bool {
        self.blocks.get(&hash).is_some_and(BlockRead::exists)
    }

    /// Root of an existing block's chain; a block with no record is its own root.
    fn root_of_existing(&self, hash: i64) -> i64 {
        self.blocks
            .get(&hash)
            .and_then(|block| block.root)
            .unwrap_or(hash)
    }
}

/// Rejects a batch whose edges conflict with stored parents or would close a
/// cycle. Runs before any write, so a rejection leaves the keyspace untouched.
fn validate_plan(plan: &BatchPlan, snapshot: &Snapshot) -> Result<(), Status> {
    for hash in &plan.order {
        let planned = plan.parents[hash];
        let existing = snapshot.block(*hash)?.parent;
        if existing != ParentLink::Unknown && existing != planned {
            return Err(Status::invalid_argument(format!(
                "block hash {hash} was reported with conflicting parents"
            )));
        }
    }
    // Follow planned edges through the batch; on leaving it into existing
    // state, jump to that subtree's root, the only node that can lead back.
    for start in &plan.order {
        let mut on_path = HashSet::new();
        let mut current = *start;
        loop {
            if !on_path.insert(current) {
                return Err(Status::invalid_argument(
                    "report would create a parent cycle",
                ));
            }
            let Some(parent) = plan.parents.get(&current) else {
                break;
            };
            let ParentLink::Hash(parent) = *parent else {
                break;
            };
            if plan.parents.contains_key(&parent) {
                current = parent;
                continue;
            }
            let root = snapshot.root_of_existing(parent);
            if plan.parents.contains_key(&root) {
                current = root;
            } else {
                break;
            }
        }
    }
    Ok(())
}

/// Chain root of every planned block after the batch: a block attached
/// outside the batch inherits that parent's root; inside, its parent's new root.
/// `validate_plan` has ruled out cycles, so the recursion terminates.
fn roots_after(plan: &BatchPlan, snapshot: &Snapshot) -> HashMap<i64, i64> {
    fn root_of(
        hash: i64,
        plan: &BatchPlan,
        snapshot: &Snapshot,
        roots: &mut HashMap<i64, i64>,
    ) -> i64 {
        if let Some(root) = roots.get(&hash) {
            return *root;
        }
        let root = match plan.parents.get(&hash) {
            Some(ParentLink::Hash(parent)) => root_of(*parent, plan, snapshot, roots),
            Some(ParentLink::Root) | Some(ParentLink::Unknown) => hash,
            None => snapshot.root_of_existing(hash),
        };
        roots.insert(hash, root);
        root
    }
    let mut roots = HashMap::with_capacity(plan.order.len());
    for hash in &plan.order {
        root_of(*hash, plan, snapshot, &mut roots);
    }
    roots
}

/// Turns a validated batch into ordered write commands, tracking the in-batch
/// state the reference semantics depend on: holdings as seen by a later
/// CLEAR_ALL_AT_TIER, and each block's live placement set so a hit count is
/// dropped at the revoke that empties it even if a later action re-reports it.
struct WriteBuilder<'a> {
    backend: &'a ValkeyKvIndexerBackend,
    worker_id: &'a str,
    plan: &'a BatchPlan,
    snapshot: &'a Snapshot,
    roots: &'a HashMap<i64, i64>,
    commands: Vec<redis::Cmd>,
    /// Existing blocks that gain a parent edge; their subtrees need `r` rewritten.
    relink: Vec<(i64, i64)>,
    written_parent: HashSet<i64>,
    /// Revoked hashes with the parent link they will have after this batch.
    revoked: Vec<(i64, ParentLink)>,
    holdings_delta: HashMap<i32, (HashSet<i64>, HashSet<i64>)>,
    live: HashMap<i64, HashSet<(String, i32)>>,
}

impl<'a> WriteBuilder<'a> {
    fn new(
        backend: &'a ValkeyKvIndexerBackend,
        worker_id: &'a str,
        plan: &'a BatchPlan,
        snapshot: &'a Snapshot,
        roots: &'a HashMap<i64, i64>,
    ) -> Self {
        Self {
            backend,
            worker_id,
            plan,
            snapshot,
            roots,
            commands: Vec::new(),
            relink: Vec::new(),
            written_parent: HashSet::new(),
            revoked: Vec::new(),
            holdings_delta: HashMap::new(),
            live: HashMap::new(),
        }
    }

    /// Address and spec are snapshots carried on every batch; an empty address
    /// makes the worker unroutable and an absent spec returns it to legacy.
    fn worker_meta(&mut self, address: &str, spec: Option<&WorkerCacheSpec>) {
        let key = self.backend.worker_key(self.worker_id);
        let mut cmd = redis::cmd("SADD");
        cmd.arg(self.backend.workers_key()).arg(self.worker_id);
        self.commands.push(cmd);
        let mut cmd = redis::cmd("HSET");
        cmd.arg(&key).arg("addr").arg(address);
        self.commands.push(cmd);
        let cmd = match spec {
            Some(spec) => {
                let mut cmd = redis::cmd("HSET");
                cmd.arg(&key).arg("spec").arg(encode_spec(spec));
                cmd
            }
            None => {
                let mut cmd = redis::cmd("HDEL");
                cmd.arg(&key).arg("spec");
                cmd
            }
        };
        self.commands.push(cmd);
    }

    fn action(&mut self, action: &ExternalKvAction) -> Result<(), Status> {
        match ExternalKvActionType::try_from(action.r#type) {
            Ok(ExternalKvActionType::ActionReport) => self.report(action),
            Ok(ExternalKvActionType::ActionRevoke) => {
                for hash in action.hashes.iter().copied() {
                    self.revoke(hash, action.tier)?;
                    let (added, removed) = self.holdings_delta.entry(action.tier).or_default();
                    added.remove(&hash);
                    removed.insert(hash);
                }
                Ok(())
            }
            Ok(ExternalKvActionType::ActionClearAllAtTier) => self.clear(action.tier),
            Ok(ExternalKvActionType::ActionUnknown) | Err(_) => Err(Status::internal(
                "unsupported action type reached the write path",
            )),
        }
    }

    fn report(&mut self, action: &ExternalKvAction) -> Result<(), Status> {
        let mut parent = action
            .parent_block_hash
            .map_or(ParentLink::Root, ParentLink::Hash);
        for (index, hash) in action.hashes.iter().copied().enumerate() {
            if self.written_parent.insert(hash) {
                self.link(hash, parent)?;
            }
            let mask = action.component_masks.get(index).copied().unwrap_or(0);
            let token_count = action.block_sizes.get(index).copied().unwrap_or(0);
            let mut cmd = redis::cmd("HSET");
            cmd.arg(self.backend.block_key(hash))
                .arg(ValkeyKvIndexerBackend::placement_field(
                    self.worker_id,
                    action.tier,
                ))
                .arg(mask);
            // A legacy report carries no size; 0 must not erase a known count.
            if token_count > 0 {
                cmd.arg("t").arg(token_count);
            }
            self.commands.push(cmd);
            let mut cmd = redis::cmd("SADD");
            cmd.arg(self.backend.holdings_key(self.worker_id, action.tier))
                .arg(hash);
            self.commands.push(cmd);
            let (added, removed) = self.holdings_delta.entry(action.tier).or_default();
            added.insert(hash);
            removed.remove(&hash);
            let placement = (self.worker_id.to_string(), action.tier);
            self.live_placements(hash).insert(placement);
            parent = ParentLink::Hash(hash);
        }
        Ok(())
    }

    /// Writes the chain edge of a planned block: its parent link and root, the
    /// parent's child entry, and a bare record for a parent only referenced.
    fn link(&mut self, hash: i64, parent: ParentLink) -> Result<(), Status> {
        let root = *self.roots.get(&hash).ok_or_else(|| missing("root", hash))?;
        let stored = self.snapshot.block(hash)?;
        let mut cmd = redis::cmd("HSET");
        cmd.arg(self.backend.block_key(hash)).arg("r").arg(root);
        if let Some(encoded) = parent.encode() {
            cmd.arg("p").arg(encoded);
        }
        self.commands.push(cmd);
        if let ParentLink::Hash(parent_hash) = parent {
            let mut cmd = redis::cmd("SADD");
            cmd.arg(self.backend.children_key(parent_hash)).arg(hash);
            self.commands.push(cmd);
            if !self.plan.parents.contains_key(&parent_hash)
                && !self.snapshot.exists(parent_hash)
                && self.written_parent.insert(parent_hash)
            {
                let mut cmd = redis::cmd("HSET");
                cmd.arg(self.backend.block_key(parent_hash))
                    .arg("r")
                    .arg(parent_hash);
                self.commands.push(cmd);
            }
        }
        if stored.exists()
            && stored.parent == ParentLink::Unknown
            && parent != ParentLink::Unknown
            && root != hash
        {
            self.relink.push((hash, root));
        }
        Ok(())
    }

    fn revoke(&mut self, hash: i64, tier: i32) -> Result<(), Status> {
        let mut cmd = redis::cmd("HDEL");
        cmd.arg(self.backend.block_key(hash))
            .arg(ValkeyKvIndexerBackend::placement_field(
                self.worker_id,
                tier,
            ));
        self.commands.push(cmd);
        let mut cmd = redis::cmd("SREM");
        cmd.arg(self.backend.holdings_key(self.worker_id, tier))
            .arg(hash);
        self.commands.push(cmd);
        let worker_id = self.worker_id.to_string();
        let set = self.live_placements(hash);
        set.remove(&(worker_id, tier));
        let emptied = set.is_empty();
        if emptied && self.snapshot.exists(hash) {
            let mut cmd = redis::cmd("HDEL");
            cmd.arg(self.backend.hits_key()).arg(hash);
            self.commands.push(cmd);
        }
        // The parent this block will have once the batch is applied.
        let parent = match self.plan.parents.get(&hash) {
            Some(parent) => *parent,
            None => self
                .snapshot
                .blocks
                .get(&hash)
                .map_or(ParentLink::Unknown, |block| block.parent),
        };
        self.revoked.push((hash, parent));
        Ok(())
    }

    fn clear(&mut self, tier: i32) -> Result<(), Status> {
        let mut hashes: Vec<i64> = self
            .snapshot
            .holdings
            .get(&tier)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default();
        if let Some((added, removed)) = self.holdings_delta.get(&tier) {
            hashes.retain(|hash| !removed.contains(hash));
            hashes.extend(added.iter().copied());
        }
        for hash in dedup_preserve_order(&hashes) {
            self.revoke(hash, tier)?;
        }
        // Everything at this tier is gone now, including earlier in-batch adds.
        self.holdings_delta
            .insert(tier, (HashSet::new(), hashes.into_iter().collect()));
        Ok(())
    }

    fn live_placements(&mut self, hash: i64) -> &mut HashSet<(String, i32)> {
        let snapshot = self.snapshot;
        self.live.entry(hash).or_insert_with(|| {
            snapshot
                .blocks
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
}

/// Shared-keyspace KV placement index over Valkey.
#[derive(Clone)]
pub struct ValkeyKvIndexerBackend {
    conn: Conn,
    prefix: String,
    prune: Arc<Script>,
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
    /// Connects, loads the prune script, and returns the backend. Both are
    /// awaited so a misconfigured URL fails at startup rather than on the first
    /// query.
    pub async fn connect(config: ValkeyConfig) -> Result<Self, Status> {
        let conn = connect_conn(&config).await?;
        let backend = Self {
            conn,
            prefix: config.key_prefix,
            prune: Arc::new(Script::new(PRUNE_LUA)),
        };
        backend.load_prune_script().await?;
        Ok(backend)
    }

    pub fn key_prefix(&self) -> &str {
        &self.prefix
    }

    pub(crate) fn conn(&self) -> Conn {
        self.conn.clone()
    }

    /// Every worker id that ever applied a batch.
    pub async fn worker_ids(&self) -> Result<Vec<String>, Status> {
        let mut pipe = redis::pipe();
        pipe.cmd("SMEMBERS").arg(self.workers_key());
        let replies: Vec<Vec<String>> = self.conn.clone().run(&pipe).await?;
        Ok(replies.into_iter().flatten().collect())
    }

    /// Revokes every placement of `worker_id` at every tier through the normal
    /// apply path, keeping its address and spec. False when it held nothing.
    pub async fn clear_worker(&self, worker_id: &str) -> Result<bool, Status> {
        let tiers = [TierType::TierHbm, TierType::TierDram, TierType::TierSsd];
        let mut pipe = redis::pipe();
        for tier in tiers {
            pipe.cmd("SCARD")
                .arg(self.holdings_key(worker_id, tier as i32));
        }
        let held: Vec<i64> = self.conn.clone().run(&pipe).await?;
        if held.iter().sum::<i64>() == 0 {
            return Ok(false);
        }
        let meta = self
            .read_workers(&[worker_id.to_string()])
            .await?
            .into_iter()
            .next()
            .unwrap_or_default();
        let actions = tiers
            .into_iter()
            .map(|tier| ExternalKvAction {
                r#type: ExternalKvActionType::ActionClearAllAtTier as i32,
                tier: tier as i32,
                hashes: Vec::new(),
                component_masks: Vec::new(),
                block_sizes: Vec::new(),
                parent_block_hash: None,
            })
            .collect();
        self.apply(ApplyExternalKvBatchRequest {
            worker_id: worker_id.to_string(),
            seq: 0,
            actions,
            worker_address: meta.address,
            cache_spec: meta.spec,
        })
        .await?;
        Ok(true)
    }

    /// `SCRIPT LOAD` is idempotent and routed to every primary in cluster mode.
    async fn load_prune_script(&self) -> Result<(), Status> {
        let mut pipe = redis::pipe();
        pipe.cmd("SCRIPT").arg("LOAD").arg(PRUNE_LUA);
        let _: Vec<String> = self.conn.clone().run(&pipe).await?;
        Ok(())
    }

    fn block_key(&self, hash: i64) -> String {
        format!("{}b:{hash}", self.prefix)
    }

    fn children_key(&self, hash: i64) -> String {
        format!("{}c:{hash}", self.prefix)
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

    fn workers_key(&self) -> String {
        workers_key(&self.prefix)
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
    async fn write_all(&self, mut commands: Vec<redis::Cmd>) -> Result<(), Status> {
        let mut conn = self.conn.clone();
        while !commands.is_empty() {
            let take = commands.len().min(PIPELINE_CHUNK);
            let mut pipe = redis::pipe();
            for cmd in commands.drain(..take) {
                pipe.add_command(cmd).ignore();
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
        let plan = plan_batch(&req.actions)?;
        let snapshot = self.snapshot(&req.worker_id, &plan, &req.actions).await?;
        validate_plan(&plan, &snapshot)?;
        let roots = roots_after(&plan, &snapshot);

        let mut writes = WriteBuilder::new(self, &req.worker_id, &plan, &snapshot, &roots);
        writes.worker_meta(&req.worker_address, req.cache_spec.as_ref());
        for action in &req.actions {
            writes.action(action)?;
        }
        let WriteBuilder {
            commands,
            relink,
            revoked,
            ..
        } = writes;
        self.write_all(commands).await?;

        for (hash, root) in relink {
            self.rewrite_subtree_root(hash, root).await?;
        }
        self.prune(revoked).await?;
        Ok(ApplyExternalKvBatchResponse {})
    }

    /// One read of every block the batch touches (planned hashes, the parents
    /// they attach to, revoked hashes) plus the holdings each CLEAR drains, and
    /// then the blocks those holdings name that nothing else did.
    async fn snapshot(
        &self,
        worker_id: &str,
        plan: &BatchPlan,
        actions: &[ExternalKvAction],
    ) -> Result<Snapshot, Status> {
        let mut touched: Vec<i64> = plan.order.clone();
        for parent in plan.parents.values() {
            if let ParentLink::Hash(parent) = parent {
                touched.push(*parent);
            }
        }
        for action in actions {
            if ExternalKvActionType::try_from(action.r#type)
                == Ok(ExternalKvActionType::ActionRevoke)
            {
                touched.extend(action.hashes.iter().copied());
            }
        }
        let touched = dedup_preserve_order(&touched);
        let commands = touched.len() + plan.clear_tiers.len();
        let replies = self
            .run_chunked(
                |pipe, range| {
                    for index in range {
                        if index < touched.len() {
                            pipe.cmd("HGETALL").arg(self.block_key(touched[index]));
                        } else {
                            let tier = plan.clear_tiers[index - touched.len()];
                            pipe.cmd("SMEMBERS").arg(self.holdings_key(worker_id, tier));
                        }
                    }
                },
                commands,
            )
            .await?;
        let mut snapshot = Snapshot {
            blocks: HashMap::with_capacity(touched.len()),
            holdings: HashMap::with_capacity(plan.clear_tiers.len()),
        };
        for (index, value) in replies.into_iter().enumerate() {
            if index < touched.len() {
                let read = BlockRead::from_fields(convert(value, "block record")?)?;
                snapshot.blocks.insert(touched[index], read);
            } else {
                let members: Vec<i64> = convert(value, "holdings set")?;
                snapshot.holdings.insert(
                    plan.clear_tiers[index - touched.len()],
                    members.into_iter().collect(),
                );
            }
        }
        let extra: Vec<i64> = snapshot
            .holdings
            .values()
            .flatten()
            .copied()
            .filter(|hash| !snapshot.blocks.contains_key(hash))
            .collect();
        let extra = dedup_preserve_order(&extra);
        if !extra.is_empty() {
            for (hash, read) in extra.iter().zip(self.read_blocks(&extra).await?) {
                snapshot.blocks.insert(*hash, read);
            }
        }
        Ok(snapshot)
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
    /// children, walking up parents that become empty in turn. A parent
    /// re-enters as a candidate each time a child is deleted, so a block judged
    /// non-empty early in a round is re-checked; every round deletes.
    async fn prune(&self, mut candidates: Vec<(i64, ParentLink)>) -> Result<(), Status> {
        while !candidates.is_empty() {
            let mut seen = HashSet::new();
            candidates.retain(|(hash, _)| seen.insert(*hash));
            let replies = match self.prune_round(&candidates).await {
                Ok(replies) => replies,
                // A flushed or restarted Valkey forgets scripts; load and retry once.
                Err(status) if status.message().contains("NOSCRIPT") => {
                    self.load_prune_script().await?;
                    self.prune_round(&candidates).await?
                }
                Err(status) => return Err(status),
            };
            let mut next = Vec::new();
            for (value, (_, parent)) in replies.into_iter().zip(&candidates) {
                let token: Option<String> = convert(value, "prune result")?;
                if let (Some(token), ParentLink::Hash(parent)) = (token, parent) {
                    next.push((*parent, ParentLink::decode(Some(&token))?));
                }
            }
            candidates = next;
        }
        Ok(())
    }

    async fn prune_round(&self, candidates: &[(i64, ParentLink)]) -> Result<Vec<Value>, Status> {
        let sha = self.prune.get_hash();
        self.run_chunked(
            |pipe, range| {
                for (hash, parent) in &candidates[range] {
                    let cmd = pipe.cmd("EVALSHA").arg(sha);
                    match parent {
                        ParentLink::Hash(parent) => cmd
                            .arg(5)
                            .arg(self.block_key(*hash))
                            .arg(self.children_key(*hash))
                            .arg(self.hits_key())
                            .arg(self.children_key(*parent))
                            .arg(self.block_key(*parent)),
                        ParentLink::Root | ParentLink::Unknown => cmd
                            .arg(3)
                            .arg(self.block_key(*hash))
                            .arg(self.children_key(*hash))
                            .arg(self.hits_key()),
                    };
                    cmd.arg(*hash);
                }
            },
            candidates.len(),
        )
        .await
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
                let Some(&index) = index_of.get(worker.as_str()) else {
                    continue;
                };
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
        // The prune script spells an absent parent link `U`.
        assert_eq!(ParentLink::decode(Some("U")).unwrap(), ParentLink::Unknown);
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
    fn plan_rejects_self_parent_and_in_batch_conflicts() {
        let report = |parent: Option<i64>, hashes: &[i64]| ExternalKvAction {
            r#type: ExternalKvActionType::ActionReport as i32,
            tier: 1,
            hashes: hashes.to_vec(),
            component_masks: Vec::new(),
            block_sizes: Vec::new(),
            parent_block_hash: parent,
        };
        assert_eq!(
            plan_batch(&[report(Some(5), &[5])]).unwrap_err().code(),
            tonic::Code::InvalidArgument
        );
        assert_eq!(
            plan_batch(&[report(None, &[1, 2]), report(Some(9), &[2])])
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );
        let plan = plan_batch(&[report(None, &[1, 2]), report(Some(2), &[3])]).unwrap();
        assert_eq!(plan.order, vec![1, 2, 3]);
        assert_eq!(plan.parents[&3], ParentLink::Hash(2));
    }

    #[test]
    fn cluster_mode_requires_hash_tag() {
        let config = ValkeyConfig::new("valkey://127.0.0.1:1")
            .with_cluster(true)
            .with_key_prefix("plain:");
        let error = futures_block(ValkeyKvIndexerBackend::connect(config)).unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn unreachable_server_is_unavailable() {
        let config = ValkeyConfig::new("valkey://127.0.0.1:1")
            .with_connect_timeout(Duration::from_millis(200));
        let error = futures_block(ValkeyKvIndexerBackend::connect(config)).unwrap_err();
        assert_eq!(error.code(), tonic::Code::Unavailable);
    }

    fn futures_block<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }
}
