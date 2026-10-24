// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Tair KVCache backend: a [`KvIndexerBackend`] whose placement state lives in
//! a Tair KVCache Manager (KVCM) `MetaService` instead of this process.
//!
//! Ingestion is unchanged -- the bridge turns each worker's ZMQ KV-event stream
//! into [`ApplyExternalKvBatch`] RPCs -- and this backend translates those
//! actions: REPORT becomes `BLOCK_ADD` (one component spec per resident
//! component), REVOKE becomes `BLOCK_DELETE`, CLEAR_ALL_AT_TIER becomes an empty
//! `BLOCK_SNAPSHOT` (the manager's only synchronous "this reporter owns nothing"
//! primitive, sent in action order), and the reporter identity is the worker id,
//! so `GetHostCacheState` answers map straight back onto workers.
//!
//! Prefix queries are answered by `GetHostCacheState` (`QT_PREFIX_MATCH`): the
//! longest contiguous run of held blocks from block 0, which is not the rule
//! engine's answer for a component-aware worker. Workers whose cache spec needs
//! SWA or MAMBA rules are excluded rather than over-reported (see
//! [`batch_is_tair_kvcache_eligible`]) and `MatchExternalKv` fails
//! `UNIMPLEMENTED`: the prefix query is the only routing signal, always a lower
//! bound. Deliberate limitations (clear scope and cooldown, the strict-visibility
//! switch, host-dram reporting, process-local reporter addresses) carry their
//! failure modes in the README.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use tonic::transport::Channel;
use tonic::{Request, Status};

use crate::client::parse_endpoint;
use crate::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, ExternalKvActionType,
    GetExternalKvHitCountsRequest, GetExternalKvHitCountsResponse, MatchExternalKvPrefixRequest,
    MatchExternalKvPrefixResponse, MatchExternalKvRequest, MatchExternalKvResponse, TierType,
    WorkerCacheSpec,
};
use crate::service::{
    assemble_prefix_response, prefix_limit, tier_in_mask, KvIndexerBackend, COMPONENT_FULL,
    COMPONENT_MAMBA, COMPONENT_SWA, SERVABLE_TIER_MASK, SUPPORTED_SPEC_VERSION,
};
use crate::tair_kvcache_pb::meta_service_client::MetaServiceClient;
use crate::tair_kvcache_pb::{
    event_item, BlockAddEventParams, BlockDeleteEventParams, BlockSnapshotEventParams,
    CommonResponseHeader, ErrorCode, EventItem, GetHostCacheStateRequest,
    GetHostCacheStateResponse, HeartbeatEventParams, HostCacheMatch, LocationSpec,
    LocationSpecInfo, ModelDeployment, NodeRegisterEventParams, QueryType, RegisterInstanceRequest,
    ReportEventRequest, ReportEventResponse, ReportEventType, StorageType,
};

/// Per-query deadline for `GetHostCacheState`: half the router-side prefix
/// deadline, since this budget starts after the service layer's admission
/// checks, and a full router deadline here would outlive the router's ask.
pub const DEFAULT_QUERY_TIMEOUT_MS: u64 = 50;
/// Deadline for mutation and registration RPCs, shared by every chunk of one
/// apply batch. Applies are never shed by the service layer, so this is what
/// keeps a hung manager from stalling a bridge.
pub const DEFAULT_APPLY_TIMEOUT_MS: u64 = 5_000;
/// Default heartbeat cadence. Beating for every worker the backend has seen
/// keeps idle caches routable at the cost of liveness never reclaiming a dead
/// reporter; the Router's health intersection is what hides those.
pub const DEFAULT_HEARTBEAT_INTERVAL_MS: u64 = 30_000;
/// Events per `ReportEvent` RPC: a maximal apply batch would otherwise be one
/// multi-hundred-kilobyte message. Order is preserved across chunks.
const MAX_EVENTS_PER_REPORT_EVENT: usize = 4_096;

const ENV_ENDPOINT: &str = "KV_INDEXER_TAIR_KVCACHE_KVCM_ENDPOINT";
const ENV_INSTANCE_ID: &str = "KV_INDEXER_TAIR_KVCACHE_KVCM_INSTANCE_ID";
const ENV_STORAGE_TYPE: &str = "KV_INDEXER_TAIR_KVCACHE_KVCM_STORAGE_TYPE";
const ENV_REPORT_TIERS: &str = "KV_INDEXER_TAIR_KVCACHE_KVCM_REPORT_TIERS";
const ENV_AUTO_REGISTER: &str = "KV_INDEXER_TAIR_KVCACHE_KVCM_AUTO_REGISTER";
const ENV_INSTANCE_GROUP: &str = "KV_INDEXER_TAIR_KVCACHE_KVCM_INSTANCE_GROUP";
const ENV_BLOCK_SIZE: &str = "KV_INDEXER_TAIR_KVCACHE_KVCM_BLOCK_SIZE";
const ENV_MODEL_NAME: &str = "KV_INDEXER_TAIR_KVCACHE_KVCM_MODEL_NAME";
const ENV_MODEL_DTYPE: &str = "KV_INDEXER_TAIR_KVCACHE_KVCM_MODEL_DTYPE";
const ENV_QUERY_TIMEOUT_MS: &str = "KV_INDEXER_TAIR_KVCACHE_KVCM_QUERY_TIMEOUT_MS";
const ENV_APPLY_TIMEOUT_MS: &str = "KV_INDEXER_TAIR_KVCACHE_KVCM_APPLY_TIMEOUT_MS";
const ENV_HEARTBEAT_INTERVAL_MS: &str = "KV_INDEXER_TAIR_KVCACHE_KVCM_HEARTBEAT_INTERVAL_MS";

/// Location-spec names reported per component: also the `LocationSpecInfo` names
/// an auto-registered instance declares, and the `spec_names` a REVOKE deletes.
pub const SPEC_FULL: &str = "full";
pub const SPEC_SWA: &str = "swa";
pub const SPEC_MAMBA: &str = "mamba";
const SPEC_NAMES: [&str; 3] = [SPEC_FULL, SPEC_SWA, SPEC_MAMBA];

/// A misconfigured [`TairKVCacheBackendConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TairKVCacheConfigError {
    Missing(&'static str),
    Invalid { key: &'static str, reason: String },
}

impl TairKVCacheConfigError {
    fn invalid(key: &'static str, reason: impl Into<String>) -> Self {
        Self::Invalid {
            key,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for TairKVCacheConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(key) => write!(f, "{key} is required"),
            Self::Invalid { key, reason } => write!(f, "invalid {key}: {reason}"),
        }
    }
}

impl std::error::Error for TairKVCacheConfigError {}

/// Instance registration for routing-only deployments (the connector registers
/// otherwise); a second registration with different fields fails
/// `DUPLICATE_ENTITY`. A model deployment and a positive block size are both
/// required by the manager.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TairKVCacheAutoRegister {
    pub instance_group: String,
    pub block_size: u32,
    /// The routed deployment's identity: the indexer knows the fleet's workers,
    /// not their model, so the operator states the name and dtype the manager
    /// demands. Every field must match a pre-registered instance, so the rest of
    /// `ModelDeployment` keeps its defaults.
    pub model_name: String,
    pub dtype: String,
}

/// Tair KVCache backend configuration.
#[derive(Debug, Clone)]
pub struct TairKVCacheBackendConfig {
    /// gRPC endpoint of the Tair KVCache MetaService, e.g. `http://10.0.0.1:50052`.
    pub endpoint: String,
    /// Tair KVCache instance, one per model deployment across the routed fleet.
    pub instance_id: String,
    /// Event-report channel the deployment provisioned (`l1p5` or `l2`).
    pub storage_type: StorageType,
    /// Tiers whose events are forwarded to Tair KVCache. Everything else is dropped.
    pub report_tiers: Vec<TierType>,
    /// Opt-in `RegisterInstance` before the first apply batch or query.
    pub auto_register: Option<TairKVCacheAutoRegister>,
    pub query_timeout: Duration,
    pub apply_timeout: Duration,
    /// `None` disables heartbeats (Tair KVCache liveness then reclaims idle workers).
    pub heartbeat_interval: Option<Duration>,
}

impl TairKVCacheBackendConfig {
    pub fn new(endpoint: impl Into<String>, instance_id: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            instance_id: instance_id.into(),
            storage_type: StorageType::StEventReportL1p5,
            report_tiers: vec![TierType::TierHbm],
            auto_register: None,
            query_timeout: Duration::from_millis(DEFAULT_QUERY_TIMEOUT_MS),
            apply_timeout: Duration::from_millis(DEFAULT_APPLY_TIMEOUT_MS),
            heartbeat_interval: Some(Duration::from_millis(DEFAULT_HEARTBEAT_INTERVAL_MS)),
        }
    }

    /// Builds the configuration from `KV_INDEXER_TAIR_KVCACHE_KVCM_*` environment
    /// variables. `KV_INDEXER_TAIR_KVCACHE_KVCM_ENDPOINT` and
    /// `KV_INDEXER_TAIR_KVCACHE_KVCM_INSTANCE_ID` are required.
    pub fn from_env() -> Result<Self, TairKVCacheConfigError> {
        let endpoint =
            env_nonempty(ENV_ENDPOINT).ok_or(TairKVCacheConfigError::Missing(ENV_ENDPOINT))?;
        let instance_id = env_nonempty(ENV_INSTANCE_ID)
            .ok_or(TairKVCacheConfigError::Missing(ENV_INSTANCE_ID))?;
        let mut config = Self::new(endpoint, instance_id);
        config.storage_type = parse_storage_type(env_nonempty(ENV_STORAGE_TYPE).as_deref())?;
        config.report_tiers = parse_report_tiers(env_nonempty(ENV_REPORT_TIERS).as_deref())?;
        config.query_timeout = Duration::from_millis(parse_timeout_ms(
            ENV_QUERY_TIMEOUT_MS,
            DEFAULT_QUERY_TIMEOUT_MS,
        )?);
        config.apply_timeout = Duration::from_millis(parse_timeout_ms(
            ENV_APPLY_TIMEOUT_MS,
            DEFAULT_APPLY_TIMEOUT_MS,
        )?);
        let heartbeat_ms =
            parse_timeout_ms(ENV_HEARTBEAT_INTERVAL_MS, DEFAULT_HEARTBEAT_INTERVAL_MS)?;
        config.heartbeat_interval = (heartbeat_ms > 0).then(|| Duration::from_millis(heartbeat_ms));
        if let Some(flag) = env_nonempty(ENV_AUTO_REGISTER) {
            if flag == "1" || flag.eq_ignore_ascii_case("true") {
                let instance_group = env_nonempty(ENV_INSTANCE_GROUP)
                    .ok_or(TairKVCacheConfigError::Missing(ENV_INSTANCE_GROUP))?;
                let block_size = parse_u64(ENV_BLOCK_SIZE, 0)?;
                let block_size = u32::try_from(block_size).map_err(|_| {
                    TairKVCacheConfigError::invalid(ENV_BLOCK_SIZE, "value exceeds u32")
                })?;
                if block_size == 0 {
                    return Err(TairKVCacheConfigError::invalid(
                        ENV_BLOCK_SIZE,
                        "must be the fleet's block size; Tair KVCache rejects a missing or zero value",
                    ));
                }
                config.auto_register = Some(TairKVCacheAutoRegister {
                    instance_group,
                    block_size,
                    model_name: env_nonempty(ENV_MODEL_NAME)
                        .ok_or(TairKVCacheConfigError::Missing(ENV_MODEL_NAME))?,
                    dtype: env_nonempty(ENV_MODEL_DTYPE)
                        .ok_or(TairKVCacheConfigError::Missing(ENV_MODEL_DTYPE))?,
                });
            } else if flag != "0" && !flag.eq_ignore_ascii_case("false") {
                return Err(TairKVCacheConfigError::invalid(
                    ENV_AUTO_REGISTER,
                    format!("expected 0 or 1, got `{flag}`"),
                ));
            }
        }
        Ok(config)
    }

    /// Medium names for the reported tiers, sent as the `NODE_REGISTER`
    /// mediums and the `GetHostCacheState` medium filter.
    fn report_mediums(&self) -> Vec<String> {
        self.report_tiers
            .iter()
            .filter_map(|tier| tier_medium(*tier))
            .map(str::to_string)
            .collect()
    }
}

/// Medium string a tier reports under. Free-form in the Tair KVCache contract; these
/// names match the vocabulary Tair KVCache's own tools use (`hbm`/`mem`/`disk`).
fn tier_medium(tier: TierType) -> Option<&'static str> {
    match tier {
        TierType::TierHbm => Some("hbm"),
        TierType::TierDram => Some("mem"),
        TierType::TierSsd => Some("disk"),
        TierType::TierUnknown => None,
    }
}

fn parse_storage_type(raw: Option<&str>) -> Result<StorageType, TairKVCacheConfigError> {
    match raw {
        None | Some("l1p5") => Ok(StorageType::StEventReportL1p5),
        Some("l2") => Ok(StorageType::StEventReportL2),
        Some(other) => Err(TairKVCacheConfigError::invalid(
            ENV_STORAGE_TYPE,
            format!("expected `l1p5` or `l2`, got `{other}`"),
        )),
    }
}

fn parse_report_tiers(raw: Option<&str>) -> Result<Vec<TierType>, TairKVCacheConfigError> {
    let raw = raw.unwrap_or("HBM");
    let mut tiers = Vec::new();
    for part in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let tier = match part.to_ascii_uppercase().as_str() {
            "HBM" => TierType::TierHbm,
            "DRAM" => TierType::TierDram,
            "SSD" => TierType::TierSsd,
            other => {
                return Err(TairKVCacheConfigError::invalid(
                    ENV_REPORT_TIERS,
                    format!("unknown tier `{other}`; expected HBM, DRAM or SSD"),
                ))
            }
        };
        if !tiers.contains(&tier) {
            tiers.push(tier);
        }
    }
    if tiers.is_empty() {
        return Err(TairKVCacheConfigError::invalid(
            ENV_REPORT_TIERS,
            "at least one tier is required",
        ));
    }
    Ok(tiers)
}

/// Parses a millisecond timeout, `default` when unset, capped at one day:
/// the apply path adds the value to an `Instant`, which panics on overflow.
fn parse_timeout_ms(key: &'static str, default: u64) -> Result<u64, TairKVCacheConfigError> {
    let value = parse_u64(key, default)?;
    if value > 86_400_000 {
        return Err(TairKVCacheConfigError::invalid(
            key,
            format!("must be at most 86400000 (one day), got {value}"),
        ));
    }
    Ok(value)
}

/// Parses a non-negative integer, `default` when unset; a `0` is meaningful
/// rather than an error (it disables heartbeats and means "no block size").
fn parse_u64(key: &'static str, default: u64) -> Result<u64, TairKVCacheConfigError> {
    let Some(raw) = env_nonempty(key) else {
        return Ok(default);
    };
    raw.parse::<u64>().map_err(|_| {
        TairKVCacheConfigError::invalid(
            key,
            format!("expected a non-negative integer, got `{raw}`"),
        )
    })
}

fn env_nonempty(key: &str) -> Option<String> {
    match std::env::var(key) {
        Ok(value) => {
            let value = value.trim().to_string();
            (!value.is_empty()).then_some(value)
        }
        _ => None,
    }
}

/// Resident component-spec names for a reported bitmask. Legacy (mask 0) means
/// the full component alone; a mask without FULL yields no specs and the caller
/// drops the block, because the manager counts a held block by *any* spec
/// present.
fn component_spec_names(mask: u32) -> Vec<&'static str> {
    if mask == 0 {
        return vec![SPEC_FULL];
    }
    if mask & COMPONENT_FULL == 0 {
        return Vec::new();
    }
    let mut names = vec![SPEC_FULL];
    if mask & COMPONENT_SWA != 0 {
        names.push(SPEC_SWA);
    }
    if mask & COMPONENT_MAMBA != 0 {
        names.push(SPEC_MAMBA);
    }
    names
}

/// Whether one apply batch keeps its worker answerable by the manager's
/// whole-block prefix match *without over-reporting* relative to the rule engine
/// in `service::compute_worker_prefix`. That engine gates a component-aware
/// worker on more than block presence (SWA windows, MAMBA boundaries) and
/// `GetHostCacheState` answers none of it, so a worker stays answerable only
/// while every batch keeps it legacy whole-block, or FULL-only within servable
/// tiers with component detail on every block.
fn batch_is_tair_kvcache_eligible(
    request: &ApplyExternalKvBatchRequest,
    report_tiers: &[TierType],
) -> Result<bool, Status> {
    for action in &request.actions {
        match TierType::try_from(action.tier) {
            // `try_from` accepts the proto default 0; the service layer
            // rejects it, so a direct caller must not see it vanish silently.
            Err(_) | Ok(TierType::TierUnknown) => {
                return Err(Status::invalid_argument("action tier is not supported"));
            }
            Ok(_) => {}
        }
        let kind = ExternalKvActionType::try_from(action.r#type)
            .map_err(|_| Status::invalid_argument("action type is not supported"))?;
        // Defense for direct callers: the service layer already enforces
        // alignment, but the trait is public and a short side array would
        // otherwise panic in `translate_batch`.
        if !action.component_masks.is_empty() && action.component_masks.len() != action.hashes.len()
        {
            return Err(Status::invalid_argument(
                "component_masks must be empty or match the hashes length",
            ));
        }
        if kind != ExternalKvActionType::ActionReport {
            continue;
        }
        match request.cache_spec.as_ref() {
            // A legacy worker reporting component detail fails the rule engine closed.
            None => {
                if !action.component_masks.is_empty() {
                    return Ok(false);
                }
            }
            Some(spec) => {
                // Every mask must confirm FULL: the bridge encodes mask-less
                // blocks as 0 inside a component batch, and an unconfirmed
                // block must not become a reported whole-block placement.
                if !spec_is_whole_block_answerable(spec, report_tiers)
                    || action.component_masks.is_empty()
                    || !action
                        .component_masks
                        .iter()
                        .all(|mask| mask & COMPONENT_FULL != 0)
                {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

/// A declared spec is answerable by whole-block matching only when it is a
/// usable FULL-only spec (no SWA window, no MAMBA boundary) whose FULL
/// component is servable on every tier this backend would report.
fn spec_is_whole_block_answerable(spec: &WorkerCacheSpec, report_tiers: &[TierType]) -> bool {
    spec.version <= SUPPORTED_SPEC_VERSION
        && spec.components & COMPONENT_FULL != 0
        && spec.components & (COMPONENT_SWA | COMPONENT_MAMBA) == 0
        && report_tiers.iter().all(|tier| {
            tier_in_mask(SERVABLE_TIER_MASK, *tier as i32)
                && tier_in_mask(spec.full_tier_mask, *tier as i32)
        })
}

fn node_register_event(mediums: &[String]) -> EventItem {
    EventItem {
        event_type: ReportEventType::EventNodeRegister as i32,
        event_params: Some(event_item::EventParams::NodeRegister(
            NodeRegisterEventParams {
                mediums: mediums.to_vec(),
            },
        )),
    }
}

/// URI scheme Tair KVCache expects for a channel's spec URIs (l1p5 reports as
/// `event_report://`, l2 as `vineyard://`).
fn storage_uri_scheme(storage_type: StorageType) -> &'static str {
    match storage_type {
        StorageType::StEventReportL2 => "vineyard",
        _ => "event_report",
    }
}

/// The URI authority identifying the reporter: the worker address's `host:port`
/// when one is restated, else the worker id. The manager only validates the
/// structure; the data plane resolves the authority when fetching.
fn reporter_uri_authority<'a>(worker_address: &'a str, worker_id: &'a str) -> &'a str {
    let Some((scheme, rest)) = worker_address.split_once("://") else {
        return worker_id;
    };
    if scheme.is_empty() {
        return worker_id;
    }
    let authority = rest.split(['/', '?']).next().unwrap_or_default();
    if authority.is_empty() {
        worker_id
    } else {
        authority
    }
}

fn block_add_event(hash: i64, medium: &str, uri: &str, specs: Vec<&'static str>) -> EventItem {
    EventItem {
        event_type: ReportEventType::EventBlockAdd as i32,
        event_params: Some(event_item::EventParams::BlockAdd(BlockAddEventParams {
            block_key: hash.to_string(),
            uri: String::new(),
            medium: medium.to_string(),
            specs: specs
                .into_iter()
                .map(|name| LocationSpec {
                    name: name.to_string(),
                    uri: uri.to_string(),
                })
                .collect(),
        })),
    }
}

fn block_delete_event(hash: i64, medium: &str) -> EventItem {
    EventItem {
        event_type: ReportEventType::EventBlockDelete as i32,
        event_params: Some(event_item::EventParams::BlockDelete(
            BlockDeleteEventParams {
                block_key: hash.to_string(),
                medium: medium.to_string(),
                // A revoke drops the whole placement, so delete every spec this
                // backend could have reported; Tair KVCache treats missing names as no-ops.
                spec_names: SPEC_NAMES.iter().map(|name| name.to_string()).collect(),
            },
        )),
    }
}

/// `CLEAR_ALL_AT_TIER` as Tair KVCache can express it: one complete snapshot with an
/// empty block set. Tair KVCache defines empty as "this reporter owns no blocks" for
/// the whole (instance, reporter, storage channel), commits it synchronously,
/// and immediately fences every older location of that reporter behind strict
/// visibility -- the closest thing to a clear the protocol has. It must travel
/// alone in its request: Tair KVCache rejects a snapshot mixed with deltas
/// (`CacheManager::ReportEvent`).
fn block_snapshot_event() -> EventItem {
    EventItem {
        event_type: ReportEventType::EventBlockSnapshot as i32,
        event_params: Some(event_item::EventParams::BlockSnapshot(
            BlockSnapshotEventParams {
                // The deprecated single-medium form stays at its default: the
                // block set below is the complete one, and an empty set names
                // no medium at all.
                blocks: Vec::new(),
                ..Default::default()
            },
        )),
    }
}

fn heartbeat_event() -> EventItem {
    EventItem {
        event_type: ReportEventType::EventHeartbeat as i32,
        event_params: Some(event_item::EventParams::Heartbeat(
            HeartbeatEventParams::default(),
        )),
    }
}

fn carries_node_register(events: &[EventItem]) -> bool {
    events
        .iter()
        .any(|event| event.event_type == ReportEventType::EventNodeRegister as i32)
}

/// Whether an event list mutates placements. Tair KVCache answers `snapshot_required`
/// for a reporter with no committed snapshot version, and the request that
/// creates one is a delta request (see `CacheManager::refresh_snapshot_response`).
fn carries_delta(events: &[EventItem]) -> bool {
    events.iter().any(|event| {
        event.event_type == ReportEventType::EventBlockAdd as i32
            || event.event_type == ReportEventType::EventBlockDelete as i32
            || event.event_type == ReportEventType::EventBlockSnapshot as i32
    })
}

/// One apply batch, translated: the requests it takes to apply it, in order.
struct TranslatedBatch {
    segments: Vec<TranslatedSegment>,
}

/// One wire request's worth of work, in action order.
enum TranslatedSegment {
    /// ADD/DELETE events for a run of actions (chunked on the wire as before).
    Delta(Vec<EventItem>),
    /// `CLEAR_ALL_AT_TIER`: an empty all-medium snapshot, alone in its request.
    /// `register` marks the request that carries the reporter's pending
    /// `NODE_REGISTER` (a snapshot may share a request with one).
    Clear { register: bool },
}

/// Pure translation of one validated apply batch into the requests it takes to
/// apply it, in action order. Actions outside `report_tiers` are dropped first;
/// `prepend_node_register` rides the batch's first request.
fn translate_batch(
    request: &ApplyExternalKvBatchRequest,
    prepend_node_register: bool,
    report_tiers: &[TierType],
    mediums: &[String],
    uri_prefix: &str,
) -> TranslatedBatch {
    let mut segments: Vec<TranslatedSegment> = Vec::new();
    let mut events: Vec<EventItem> = Vec::new();
    // The pending registration rides the first request the batch sends: a
    // delta chunk, or the clear's snapshot when a clear opens the batch.
    let mut register_pending = prepend_node_register;
    for action in &request.actions {
        // The service layer already rejects unknown tiers and action types;
        // `batch_is_tair_kvcache_eligible` re-checks both for direct callers.
        let Ok(tier) = TierType::try_from(action.tier) else {
            continue;
        };
        if !report_tiers.contains(&tier) {
            continue;
        }
        let Some(medium) = tier_medium(tier) else {
            continue;
        };
        match ExternalKvActionType::try_from(action.r#type) {
            Ok(ExternalKvActionType::ActionReport) => {
                let has_masks = !action.component_masks.is_empty();
                for (index, hash) in action.hashes.iter().copied().enumerate() {
                    let mask = if has_masks {
                        action.component_masks[index]
                    } else {
                        0
                    };
                    let specs = component_spec_names(mask);
                    if specs.is_empty() {
                        continue;
                    }
                    events.push(block_add_event(
                        hash,
                        medium,
                        &format!("{uri_prefix}/{medium}"),
                        specs,
                    ));
                }
            }
            Ok(ExternalKvActionType::ActionRevoke) => {
                for hash in &action.hashes {
                    events.push(block_delete_event(*hash, medium));
                }
            }
            Ok(ExternalKvActionType::ActionClearAllAtTier) => {
                // The clear's position is its meaning: mutations before it are
                // fenced off by the snapshot, mutations after it land on the
                // fresh generation. Emit the pending deltas first.
                let register = if events.is_empty() {
                    // A clear that opens the batch carries the registration
                    // itself; a snapshot may share a request with one.
                    std::mem::take(&mut register_pending)
                } else {
                    // The registration rides the batch's first request (the
                    // delta segment); on the snapshot it would arrive one
                    // request late and a forgotten reporter would lose the delta.
                    if register_pending {
                        events.insert(0, node_register_event(mediums));
                        register_pending = false;
                    }
                    false
                };
                if !events.is_empty() {
                    segments.push(TranslatedSegment::Delta(std::mem::take(&mut events)));
                }
                segments.push(TranslatedSegment::Clear { register });
            }
            Ok(ExternalKvActionType::ActionUnknown) | Err(_) => {}
        }
    }
    if register_pending {
        // Nothing in this batch is forwardable, but the reporter still has to
        // be registered (the batch that discovered it owes Tair KVCache that).
        events.insert(0, node_register_event(mediums));
    }
    if !events.is_empty() {
        segments.push(TranslatedSegment::Delta(events));
    }
    TranslatedBatch { segments }
}

/// The outcome a Tair KVCache response header carries.
enum HeaderOutcome {
    Ok,
    Failed { code: ErrorCode, message: String },
}

fn header_outcome(header: Option<&CommonResponseHeader>) -> HeaderOutcome {
    let failed = |code, message| HeaderOutcome::Failed { code, message };
    let Some(header) = header else {
        return failed(
            ErrorCode::UnknownError,
            "response is missing its header".to_string(),
        );
    };
    let Some(status) = header.status.as_ref() else {
        return failed(
            ErrorCode::UnknownError,
            "response header is missing its status".to_string(),
        );
    };
    let code = ErrorCode::try_from(status.code).unwrap_or(ErrorCode::UnknownError);
    if matches!(code, ErrorCode::Ok) {
        HeaderOutcome::Ok
    } else {
        HeaderOutcome::Failed {
            code,
            message: status.message.clone(),
        }
    }
}

/// Maps a manager response header into a `Result`. `INVALID_ARGUMENT` and
/// `DUPLICATE_ENTITY` stay loud contract errors, and permanent configuration
/// faults become `FAILED_PRECONDITION` so the bridge exits instead of
/// reconnecting forever on a typo'd instance id; every other code, including
/// `NODE_NOT_REGISTERED`, is transient.
fn check_header(header: Option<&CommonResponseHeader>, what: &str) -> Result<(), Status> {
    match header_outcome(header) {
        HeaderOutcome::Ok => Ok(()),
        HeaderOutcome::Failed { code, message } => Err(match code {
            ErrorCode::InvalidArgument => {
                Status::invalid_argument(format!("Tair KVCache rejected {what}: {message}"))
            }
            ErrorCode::DuplicateEntity => Status::failed_precondition(format!(
                "Tair KVCache {what}: entity already registered with different configuration: {message}"
            )),
            ErrorCode::InstanceNotExist
            | ErrorCode::Unsupported
            | ErrorCode::ReachMaxEntityCapacity => Status::failed_precondition(format!(
                "Tair KVCache {what} failed on a permanent configuration fault: {code:?}: {message}"
            )),
            code => Status::unavailable(format!("Tair KVCache {what} failed: {code:?}: {message}")),
        }),
    }
}

/// `(worker_id, address, matched blocks)` for one host match: the worker must be
/// answerable (see [`batch_is_tair_kvcache_eligible`]) with a non-empty
/// router-facing address, and the prefix is clamped to the query length.
fn prefix_entry(
    host: HostCacheMatch,
    blocks_read: u32,
    record: &WorkerRecord,
) -> Option<(String, String, u32)> {
    if !record.tair_kvcache_eligible || record.address.is_empty() {
        return None;
    }
    let matched = u32::try_from(host.local).ok()?.min(blocks_read);
    (matched > 0).then(|| (host.host_ip_port, record.address.clone(), matched))
}

/// Router-facing bookkeeping per worker, refreshed by every apply batch.
#[derive(Debug, Default, Clone)]
struct WorkerRecord {
    address: String,
    /// Whether a `NODE_REGISTER` for this reporter has been accepted. The flag
    /// flips only on a successful report carrying one: a failed first batch must
    /// not burn the registration, and a liveness cleanup makes the manager
    /// forget it.
    node_registered: bool,
    /// Sticky: once a batch makes the worker unanswerable by whole-block
    /// matching (see [`batch_is_tair_kvcache_eligible`]), it stays excluded — the
    /// placements Tair KVCache already holds cannot be un-reported.
    tair_kvcache_eligible: bool,
}

struct Inner {
    client: MetaServiceClient<Channel>,
    config: TairKVCacheBackendConfig,
    mediums: Vec<String>,
    /// Router-facing record per worker, refreshed by every apply batch: the
    /// query side resolves Tair KVCache reporter ids back into addresses here.
    workers: RwLock<HashMap<String, WorkerRecord>>,
    instance_registered: AtomicBool,
    trace: AtomicU64,
}

impl Inner {
    fn next_trace(&self, what: &str) -> String {
        format!(
            "sgl-kv-indexer-{what}-{}",
            self.trace.fetch_add(1, Ordering::Relaxed)
        )
    }

    fn workers_read(
        &self,
    ) -> Result<std::sync::RwLockReadGuard<'_, HashMap<String, WorkerRecord>>, Status> {
        self.workers
            .read()
            .map_err(|_| Status::internal("Tair KVCache backend workers map lock poisoned"))
    }

    fn set_node_registered(&self, host: &str, registered: bool) {
        if let Ok(mut workers) = self.workers.write() {
            if let Some(record) = workers.get_mut(host) {
                record.node_registered = registered;
            }
        }
    }

    /// One `ReportEvent` on the wire, unclassified: the delta path and the
    /// flush path disagree on what a failure means (deltas fail the batch,
    /// a rate-limited flush must not), so only the transport lives here.
    async fn report_event_response(
        &self,
        host: &str,
        events: Vec<EventItem>,
        timeout: Duration,
    ) -> Result<ReportEventResponse, Status> {
        let mut client = self.client.clone();
        let mut request = Request::new(ReportEventRequest {
            trace_id: self.next_trace("apply"),
            instance_id: self.config.instance_id.clone(),
            host_ip_port: host.to_string(),
            events,
            storage_type: self.config.storage_type as i32,
        });
        request.set_timeout(timeout);
        let response = tokio::time::timeout(timeout, client.report_event(request))
            .await
            .map_err(|_| Status::deadline_exceeded("Tair KVCache ReportEvent timed out"))??;
        Ok(response.into_inner())
    }

    /// Forwards one `ReportEvent` and maintains the reporter's registration
    /// flag: success carrying `NODE_REGISTER` marks it registered, and a
    /// `NODE_NOT_REGISTERED` failure resets it so the next report re-registers.
    async fn report_event_checked(
        &self,
        host: &str,
        events: Vec<EventItem>,
        timeout: Duration,
    ) -> Result<(), Status> {
        let registered_node = carries_node_register(&events);
        let has_delta = carries_delta(&events);
        let response = self.report_event_response(host, events, timeout).await?;
        self.classify_report_response(host, &response, registered_node, has_delta)
    }

    /// Turns one `ReportEvent` response into the status the caller should see,
    /// and keeps the reporter's registration flag in step. Shared by the delta
    /// and flush paths: they differ only in what they do with a failure.
    fn classify_report_response(
        &self,
        host: &str,
        response: &ReportEventResponse,
        carried_register: bool,
        carried_delta: bool,
    ) -> Result<(), Status> {
        if let HeaderOutcome::Failed {
            code: ErrorCode::NodeNotRegistered,
            ..
        } = header_outcome(response.header.as_ref())
        {
            self.set_node_registered(host, false);
        }
        if response.snapshot_required {
            // A lifecycle start, not a fault: the first delta creates the
            // version being asked about. This backend holds no placement state
            // and cannot reconcile either way.
            if carried_delta {
                tracing::info!(
                    host,
                    "Tair KVCache has no committed snapshot for this reporter; it stays delta-only"
                );
            } else {
                tracing::debug!(
                    host,
                    "Tair KVCache still has no committed snapshot for this reporter"
                );
            }
        }
        // Header first, then items: one failure, one classification.
        check_header(response.header.as_ref(), "ReportEvent")?;
        if !response.item_results.is_empty() {
            // A structurally invalid event is a contract violation: report it
            // as INVALID_ARGUMENT so the bridge exits instead of reconnecting
            // and re-sending the same bad batch forever.
            let invalid = response
                .item_results
                .contains(&(ErrorCode::InvalidArgument as i32));
            let status = |code: tonic::Code, what: &str| {
                Status::new(
                    code,
                    format!(
                        "Tair KVCache ReportEvent partially failed ({what}): item results {:?}",
                        response.item_results
                    ),
                )
            };
            return Err(if invalid {
                status(tonic::Code::InvalidArgument, "invalid event")
            } else {
                status(tonic::Code::Internal, "transient")
            });
        }
        if carried_register {
            self.set_node_registered(host, true);
        }
        Ok(())
    }

    /// Applies one `CLEAR_ALL_AT_TIER` as an empty all-medium snapshot, in a request
    /// of its own, and returns once the manager has committed the fence.
    ///
    /// The retry policy is asymmetric on purpose:
    ///
    /// * `SNAPSHOT_IN_PROGRESS` is a transient race (a concurrent snapshot, or a
    ///   delta drain that timed out): re-send the complete snapshot after a short
    ///   backoff, bounded by the apply budget.
    /// * `SNAPSHOT_RATE_LIMITED` **drops the flush with a warning instead of
    ///   failing the batch**: the cooldown (30s by default) cannot be waited out
    ///   inside an apply budget, and a failed batch costs more than a missed flush
    ///   -- the bridge reconnects and does not replay, losing every delta that rode
    ///   with it.
    /// * `NODE_NOT_REGISTERED` means the manager forgot this reporter (liveness
    ///   cleanup): the retry carries a fresh `NODE_REGISTER`.
    ///
    /// Anything else keeps the delta path's classification. A committed snapshot
    /// leaves no local state to drop: this backend keeps none.
    async fn report_flush_snapshot(
        &self,
        host: &str,
        with_register: bool,
        deadline: tokio::time::Instant,
    ) -> Result<(), Status> {
        const ATTEMPTS: usize = 3;
        const IN_PROGRESS_BACKOFF: Duration = Duration::from_millis(100);
        let mut register = with_register;
        for attempt in 1..=ATTEMPTS {
            let mut events = Vec::with_capacity(2);
            if register {
                events.push(node_register_event(&self.mediums));
            }
            events.push(block_snapshot_event());
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Err(Status::deadline_exceeded(
                    "Tair KVCache apply budget exhausted before the flush snapshot was sent",
                ));
            }
            let response = self.report_event_response(host, events, remaining).await?;
            let outcome = header_outcome(response.header.as_ref());
            // Request-level failures land in the header alone; per-item
            // failures in both the header and `item_results`. A header naming
            // a different fault is not our code.
            let failed_with = |code: ErrorCode| match &outcome {
                HeaderOutcome::Failed { code: found, .. } => *found == code,
                HeaderOutcome::Ok => response.item_results.contains(&(code as i32)),
            };
            if matches!(outcome, HeaderOutcome::Ok) && response.item_results.is_empty() {
                if register {
                    self.set_node_registered(host, true);
                }
                tracing::info!(
                    host,
                    commit = %response.committed_snapshot_version,
                    report_tiers = self.config.report_tiers.len(),
                    "CLEAR_ALL_AT_TIER applied as an empty BLOCK_SNAPSHOT: every placement this reporter owns on the channel is fenced"
                );
                return Ok(());
            }
            if failed_with(ErrorCode::SnapshotRateLimited) {
                // Dropped on purpose: see the doc comment. The flush is a
                // best-effort improvement over a stale answer, not a promise
                // worth losing the rest of the batch for.
                tracing::warn!(
                    host,
                    retry_after_ms = response.retry_after_ms,
                    report_tiers = self.config.report_tiers.len(),
                    "flush dropped, placements stay: Tair KVCache rate-limited the snapshot; a later clear retries"
                );
                return Ok(());
            }
            if failed_with(ErrorCode::NodeNotRegistered) {
                // Tair KVCache forgot the reporter: re-register on the retry rather
                // than leaving the flush to the next batch.
                self.set_node_registered(host, false);
                register = true;
                if attempt < ATTEMPTS {
                    continue;
                }
                return self.classify_report_response(host, &response, register, true);
            }
            if failed_with(ErrorCode::SnapshotInProgress) && attempt < ATTEMPTS {
                tokio::time::sleep(IN_PROGRESS_BACKOFF * attempt as u32).await;
                continue;
            }
            // Not a flush-specific outcome: let the delta path's classification
            // decide, so the batch fails exactly as it would for a delta.
            return self.classify_report_response(host, &response, register, true);
        }
        // Every branch above returns on the final attempt; this keeps the
        // signature total without inventing a panic path.
        Err(Status::unavailable(
            "Tair KVCache reported a snapshot in progress on every flush attempt",
        ))
    }

    async fn get_host_cache_state(
        &self,
        hashes: &[i64],
    ) -> Result<GetHostCacheStateResponse, Status> {
        let mut client = self.client.clone();
        let mut request = Request::new(GetHostCacheStateRequest {
            trace_id: self.next_trace("query"),
            instance_id: self.config.instance_id.clone(),
            query_type: QueryType::QtPrefixMatch as i32,
            block_cache_keys: hashes.to_vec(),
            medium: self.mediums.clone(),
            p2p_host_count: 0,
        });
        request.set_timeout(self.config.query_timeout);
        let response = tokio::time::timeout(
            self.config.query_timeout,
            client.get_host_cache_state(request),
        )
        .await
        .map_err(|_| Status::deadline_exceeded("Tair KVCache GetHostCacheState timed out"))??;
        let response = response.into_inner();
        check_header(response.header.as_ref(), "GetHostCacheState")?;
        Ok(response)
    }

    /// One-shot `RegisterInstance` when auto-registration is configured. It runs
    /// on the apply budget (a setup call, not an answer), concurrent cold-start
    /// calls may each send one, and the flag short-circuits the rest.
    async fn register_instance(&self) -> Result<(), Status> {
        let Some(auto) = &self.config.auto_register else {
            return Ok(());
        };
        if self.instance_registered.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut client = self.client.clone();
        let mut request = Request::new(RegisterInstanceRequest {
            trace_id: self.next_trace("register"),
            instance_group: auto.instance_group.clone(),
            instance_id: self.config.instance_id.clone(),
            block_size: auto.block_size as i32,
            location_spec_infos: SPEC_NAMES
                .iter()
                .map(|&name| LocationSpecInfo {
                    name: name.to_string(),
                    size: 0,
                })
                .collect(),
            model_deployment: Some(ModelDeployment {
                model_name: auto.model_name.clone(),
                dtype: auto.dtype.clone(),
                ..Default::default()
            }),
            location_spec_groups: Vec::new(),
            default_query_type: QueryType::QtPrefixMatch as i32,
        });
        request.set_timeout(self.config.apply_timeout);
        let response =
            tokio::time::timeout(self.config.apply_timeout, client.register_instance(request))
                .await
                .map_err(|_| {
                    Status::deadline_exceeded("Tair KVCache RegisterInstance timed out")
                })??;
        check_header(response.into_inner().header.as_ref(), "RegisterInstance")?;
        self.instance_registered.store(true, Ordering::Release);
        Ok(())
    }
}

/// A [`KvIndexerBackend`] backed by a Tair KVCache MetaService.
pub struct TairKVCacheIndexerBackend {
    inner: Arc<Inner>,
}

impl fmt::Debug for TairKVCacheIndexerBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TairKVCacheIndexerBackend")
            .field("instance_id", &self.inner.config.instance_id)
            .field("storage_type", &self.inner.config.storage_type)
            .field("report_tiers", &self.inner.config.report_tiers)
            .finish_non_exhaustive()
    }
}

impl TairKVCacheIndexerBackend {
    /// Connects lazily; a misconfigured endpoint is rejected up front (see
    /// [`parse_endpoint`]) rather than failing every RPC later.
    pub fn new(config: TairKVCacheBackendConfig) -> Result<Self, crate::InvalidEndpoint> {
        let endpoint = parse_endpoint(&config.endpoint)?;
        // A non-servable report tier (SSD) silently excludes every
        // spec-declaring worker; legacy workers are unaffected. Warn at
        // startup instead of letting the first fleet look empty.
        if config
            .report_tiers
            .iter()
            .any(|tier| !tier_in_mask(SERVABLE_TIER_MASK, *tier as i32))
        {
            tracing::warn!(
                report_tiers = ?config.report_tiers,
                "report tiers outside HBM/DRAM exclude every worker that declares a cache spec from prefix answers"
            );
        }
        let mediums = config.report_mediums();
        Ok(Self {
            inner: Arc::new(Inner {
                client: MetaServiceClient::new(endpoint.connect_lazy()),
                mediums,
                workers: RwLock::new(HashMap::new()),
                instance_registered: AtomicBool::new(false),
                trace: AtomicU64::new(0),
                config,
            }),
        })
    }

    /// Keeps one reporter alive per worker. The task holds a weak handle, so
    /// dropping the backend stops it; workers that stopped being answerable are
    /// skipped, since nothing is reported for them.
    fn spawn_heartbeat(&self, worker_id: String) {
        let Some(interval) = self.inner.config.heartbeat_interval else {
            return;
        };
        let inner = Arc::downgrade(&self.inner);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                let Some(inner) = inner.upgrade() else {
                    break;
                };
                let (eligible, registered) = {
                    let Ok(workers) = inner.workers_read() else {
                        break;
                    };
                    let Some(record) = workers.get(&worker_id) else {
                        break;
                    };
                    (record.tair_kvcache_eligible, record.node_registered)
                };
                if !eligible {
                    // Eligibility only narrows: an excluded worker never gets
                    // another report, so its heartbeat task has no work left.
                    break;
                }
                // A heartbeat that follows a forgotten registration (or a
                // failed first report) re-registers the reporter in passing.
                let mut events = Vec::new();
                if !registered {
                    events.push(node_register_event(&inner.mediums));
                }
                events.push(heartbeat_event());
                if let Err(error) = inner
                    .report_event_checked(&worker_id, events, inner.config.apply_timeout)
                    .await
                {
                    tracing::warn!(worker = %worker_id, %error, "Tair KVCache heartbeat failed");
                }
            }
        });
    }
}

#[tonic::async_trait]
impl KvIndexerBackend for TairKVCacheIndexerBackend {
    async fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status> {
        // '#' is the Tair KVCache location-id separator; a worker id carrying it would
        // corrupt every placement the reporter ever sends.
        if request.worker_id.contains('#') {
            return Err(Status::invalid_argument(
                "worker_id must not contain '#' (Tair KVCache location-id component)",
            ));
        }
        let (first_seen, node_registered, tair_kvcache_eligible) = {
            let mut workers =
                self.inner.workers.write().map_err(|_| {
                    Status::internal("Tair KVCache backend workers map lock poisoned")
                })?;
            match workers.entry(request.worker_id.clone()) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let record = entry.get_mut();
                    // The address is restated on every batch, so record it even
                    // when the batch translates to no events at all.
                    record.address = request.worker_address.clone();
                    (false, record.node_registered, record.tair_kvcache_eligible)
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(WorkerRecord {
                        address: request.worker_address.clone(),
                        node_registered: false,
                        tair_kvcache_eligible: true,
                    });
                    (true, false, true)
                }
            }
        };
        if first_seen {
            self.spawn_heartbeat(request.worker_id.clone());
        }
        self.inner.register_instance().await?;
        let batch_eligible =
            batch_is_tair_kvcache_eligible(&request, &self.inner.config.report_tiers)?;
        // Eligibility only ever narrows: once a worker is excluded it stays
        // excluded, because Tair KVCache already holds its earlier placements.
        if tair_kvcache_eligible && !batch_eligible {
            tracing::warn!(
                worker = %request.worker_id,
                "cache spec or component detail requires prefix rules Tair KVCache cannot evaluate; excluding the worker from prefix answers (fail closed)"
            );
            if let Ok(mut workers) = self.inner.workers.write() {
                if let Some(record) = workers.get_mut(&request.worker_id) {
                    record.tair_kvcache_eligible = false;
                }
            }
        }
        if !batch_eligible {
            // Nothing is reported for an excluded worker, so there is nothing
            // to forward — not even its node registration.
            return Ok(ApplyExternalKvBatchResponse::default());
        }
        let translated = translate_batch(
            &request,
            !node_registered,
            &self.inner.config.report_tiers,
            &self.inner.mediums,
            &format!(
                "{}://{}",
                storage_uri_scheme(self.inner.config.storage_type),
                reporter_uri_authority(&request.worker_address, &request.worker_id)
            ),
        );
        // One deadline for the whole batch: a slow Tair KVCache must not turn four
        // chunk RPCs into four full timeouts while the bridge's own channel
        // timeout expires underneath and drops the batch.
        let deadline = tokio::time::Instant::now() + self.inner.config.apply_timeout;
        for segment in translated.segments {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Err(Status::deadline_exceeded(
                    "Tair KVCache apply budget exhausted before all events were forwarded",
                ));
            }
            match segment {
                TranslatedSegment::Delta(events) => {
                    for chunk in events.chunks(MAX_EVENTS_PER_REPORT_EVENT) {
                        let remaining =
                            deadline.saturating_duration_since(tokio::time::Instant::now());
                        if remaining.is_zero() {
                            return Err(Status::deadline_exceeded(
                                "Tair KVCache apply budget exhausted before all chunks were forwarded",
                            ));
                        }
                        self.inner
                            .report_event_checked(&request.worker_id, chunk.to_vec(), remaining)
                            .await?;
                    }
                }
                TranslatedSegment::Clear { register } => {
                    // Synchronous by design: the fence must be committed before
                    // the actions that follow it are sent, or they would land
                    // on the generation this clear ends.
                    self.inner
                        .report_flush_snapshot(&request.worker_id, register, deadline)
                        .await?;
                }
            }
        }
        Ok(ApplyExternalKvBatchResponse::default())
    }

    async fn match_external_kv(
        &self,
        _request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        Err(Status::unimplemented(
            "the Tair KVCache backend answers prefix queries only: Tair KVCache has no per-worker per-block placement query (use MatchExternalKvPrefix)",
        ))
    }

    /// Answers via `GetHostCacheState`: per reporter host, the longest contiguous
    /// run of held blocks from block 0. Component-aware workers are excluded
    /// rather than over-reported, so the answer stays a lower bound. (The default
    /// path derives placements from `match_external_kv`, which has no equivalent
    /// here.)
    async fn match_external_kv_prefix(
        &self,
        request: MatchExternalKvPrefixRequest,
    ) -> Result<MatchExternalKvPrefixResponse, Status> {
        let limit = prefix_limit(request.hashes.len(), request.max_blocks);
        let hashes: Vec<i64> = request.hashes.into_iter().take(limit).collect();
        if hashes.is_empty() {
            return Ok(MatchExternalKvPrefixResponse::default());
        }
        // A prefix query can be the first thing this backend does, and an
        // unregistered instance is a permanent fault the router turns into a
        // 503. Without `auto_register` that fault is a real config error.
        self.inner.register_instance().await?;
        let response = self.inner.get_host_cache_state(&hashes).await?;
        let blocks_read = hashes.len() as u32;
        let workers = self.inner.workers_read()?;
        let entries: Vec<(String, String, u32)> = response
            .hosts
            .into_iter()
            .filter_map(|host| {
                let record = workers.get(&host.host_ip_port)?;
                prefix_entry(host, blocks_read, record)
            })
            .collect();
        Ok(assemble_prefix_response(entries, blocks_read))
    }

    /// Hit counting is tied to `MatchExternalKv`, which this backend does not
    /// answer; the diagnostic counters stay empty rather than invented.
    async fn get_external_kv_hit_counts(
        &self,
        _request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        Ok(GetExternalKvHitCountsResponse::default())
    }
}

#[cfg(test)]
mod tests {
    /// The batch's delta events, in order. Most assertions are about what the
    /// translation produces, not where the request boundaries fell; the tests
    /// that care about the segmentation inspect `segments` directly.
    fn delta_events(batch: &TranslatedBatch) -> Vec<EventItem> {
        batch
            .segments
            .iter()
            .flat_map(|segment| match segment {
                TranslatedSegment::Delta(events) => events.clone(),
                TranslatedSegment::Clear { .. } => Vec::new(),
            })
            .collect()
    }

    use super::*;
    use crate::pb::{ExternalKvAction, MatchExternalKvPrefixRequest};

    fn report_action(tier: i32, hashes: &[i64], masks: &[u32]) -> ExternalKvAction {
        ExternalKvAction {
            r#type: ExternalKvActionType::ActionReport as i32,
            tier,
            hashes: hashes.to_vec(),
            component_masks: masks.to_vec(),
            block_sizes: Vec::new(),
            parent_block_hash: None,
        }
    }

    fn apply(
        worker: &str,
        address: &str,
        actions: Vec<ExternalKvAction>,
        cache_spec: Option<WorkerCacheSpec>,
    ) -> ApplyExternalKvBatchRequest {
        ApplyExternalKvBatchRequest {
            worker_id: worker.to_string(),
            seq: 1,
            actions,
            worker_address: address.to_string(),
            cache_spec,
        }
    }

    fn hbm() -> i32 {
        TierType::TierHbm as i32
    }

    fn dram() -> i32 {
        TierType::TierDram as i32
    }

    fn spec(components: u32, version: u32, full_tier_mask: u32) -> WorkerCacheSpec {
        WorkerCacheSpec {
            version,
            components,
            swa_window_tokens: 0,
            full_tier_mask,
            swa_tier_mask: 0,
            mamba_tier_mask: 0,
        }
    }

    #[test]
    fn uri_helpers_pick_scheme_and_authority() {
        assert_eq!(
            storage_uri_scheme(StorageType::StEventReportL1p5),
            "event_report"
        );
        assert_eq!(storage_uri_scheme(StorageType::StEventReportL2), "vineyard");
        assert_eq!(
            reporter_uri_authority("http://10.0.0.1:30000", "worker-0"),
            "10.0.0.1:30000"
        );
        assert_eq!(
            reporter_uri_authority("http://h1:1/some/path", "worker-0"),
            "h1:1"
        );
        assert_eq!(reporter_uri_authority("", "worker-0"), "worker-0");
        assert_eq!(reporter_uri_authority("no-scheme", "worker-0"), "worker-0");
        assert_eq!(reporter_uri_authority("http://", "worker-0"), "worker-0");
    }

    #[test]
    fn tier_medium_maps_every_valid_tier() {
        assert_eq!(tier_medium(TierType::TierHbm), Some("hbm"));
        assert_eq!(tier_medium(TierType::TierDram), Some("mem"));
        assert_eq!(tier_medium(TierType::TierSsd), Some("disk"));
        assert_eq!(tier_medium(TierType::TierUnknown), None);
    }

    #[test]
    fn component_specs_cover_legacy_and_components() {
        assert_eq!(component_spec_names(0), vec![SPEC_FULL]);
        assert_eq!(component_spec_names(COMPONENT_FULL), vec![SPEC_FULL]);
        assert_eq!(
            component_spec_names(COMPONENT_FULL | COMPONENT_SWA),
            vec![SPEC_FULL, SPEC_SWA]
        );
        assert_eq!(
            component_spec_names(COMPONENT_FULL | COMPONENT_SWA | COMPONENT_MAMBA),
            vec![SPEC_FULL, SPEC_SWA, SPEC_MAMBA]
        );
    }

    #[test]
    fn unknown_components_alone_fail_closed() {
        // A future component bit without FULL cannot serve full attention;
        // reporting it as whole-block would over-report.
        assert!(component_spec_names(1 << 3).is_empty());
        assert!(component_spec_names(COMPONENT_SWA).is_empty());
        // A future bit alongside FULL keeps the known components only.
        assert_eq!(
            component_spec_names(COMPONENT_FULL | (1 << 3)),
            vec![SPEC_FULL]
        );
    }

    #[test]
    fn translate_batch_reports_components_and_revoke() {
        let request = apply(
            "worker-0",
            "http://w0:30000",
            vec![
                report_action(
                    hbm(),
                    &[7, 8],
                    &[COMPONENT_FULL, COMPONENT_FULL | COMPONENT_SWA],
                ),
                ExternalKvAction {
                    r#type: ExternalKvActionType::ActionRevoke as i32,
                    tier: hbm(),
                    hashes: vec![9],
                    component_masks: Vec::new(),
                    block_sizes: Vec::new(),
                    parent_block_hash: None,
                },
            ],
            None,
        );
        let translated = translate_batch(
            &request,
            true,
            &[TierType::TierHbm],
            &["hbm".into()],
            "event_report://worker-0",
        );
        let events = delta_events(&translated);
        let EventItem {
            event_type,
            event_params: Some(event_item::EventParams::NodeRegister(register)),
        } = &events[0]
        else {
            panic!("first event must be NODE_REGISTER");
        };
        assert_eq!(*event_type, ReportEventType::EventNodeRegister as i32);
        assert_eq!(register.mediums, ["hbm"]);
        let EventItem {
            event_params: Some(event_item::EventParams::BlockAdd(add)),
            ..
        } = &events[1]
        else {
            panic!("expected BLOCK_ADD");
        };
        assert_eq!(add.block_key, "7");
        assert_eq!(add.medium, "hbm");
        assert_eq!(add.specs.len(), 1);
        assert_eq!(add.specs[0].name, SPEC_FULL);
        // Tair KVCache rejects spec URIs that are empty or lack the channel's scheme.
        assert_eq!(add.specs[0].uri, "event_report://worker-0/hbm");
        let EventItem {
            event_params: Some(event_item::EventParams::BlockAdd(add)),
            ..
        } = &events[2]
        else {
            panic!("expected BLOCK_ADD");
        };
        assert_eq!(
            add.specs
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>(),
            ["full", "swa"]
        );
        let EventItem {
            event_params: Some(event_item::EventParams::BlockDelete(delete)),
            ..
        } = &events[3]
        else {
            panic!("expected BLOCK_DELETE");
        };
        assert_eq!(delete.block_key, "9");
        assert_eq!(delete.spec_names, SPEC_NAMES);
    }

    #[test]
    fn translate_batch_keeps_the_clear_position() {
        let request = apply(
            "worker-0",
            "http://w0:30000",
            vec![
                report_action(dram(), &[1], &[COMPONENT_FULL]),
                ExternalKvAction {
                    r#type: ExternalKvActionType::ActionClearAllAtTier as i32,
                    tier: hbm(),
                    hashes: Vec::new(),
                    component_masks: Vec::new(),
                    block_sizes: Vec::new(),
                    parent_block_hash: None,
                },
            ],
            None,
        );
        // A pending node registration is still sent: it registers the reporter
        // even when nothing else in the batch is forwardable.
        let translated = translate_batch(
            &request,
            true,
            &[TierType::TierHbm],
            &["hbm".into()],
            "event_report://worker-0",
        );
        // The unreported DRAM action produces no delta at all; the clear is
        // its own segment, and the registration rides with it.
        assert_eq!(translated.segments.len(), 1);
        assert!(matches!(
            translated.segments[0],
            TranslatedSegment::Clear { register: true }
        ));
        assert!(delta_events(&translated).is_empty());
        // A later batch with nothing to forward sends nothing.
        let translated = translate_batch(
            &request,
            false,
            &[TierType::TierHbm],
            &["hbm".into()],
            "event_report://worker-0",
        );
        assert_eq!(translated.segments.len(), 1);
        assert!(matches!(
            translated.segments[0],
            TranslatedSegment::Clear { .. }
        ));
        assert!(delta_events(&translated).is_empty());
    }

    /// The pending registration rides the batch's first request whatever it is:
    /// left behind on a clear's snapshot it would arrive one request too late,
    /// and a forgotten reporter would fail that first delta and lose the batch.
    #[test]
    fn translate_batch_registers_on_the_first_request_of_a_batch() {
        let clear = || ExternalKvAction {
            r#type: ExternalKvActionType::ActionClearAllAtTier as i32,
            tier: hbm(),
            hashes: Vec::new(),
            component_masks: Vec::new(),
            block_sizes: Vec::new(),
            parent_block_hash: None,
        };
        let report = |hash: i64| report_action(hbm(), &[hash], &[COMPONENT_FULL]);

        // [REPORT, CLEAR]: the registration rides the delta, not the snapshot.
        let request = apply(
            "worker-0",
            "http://w0:30000",
            vec![report(1), clear()],
            None,
        );
        let translated = translate_batch(
            &request,
            true,
            &[TierType::TierHbm],
            &["hbm".into()],
            "event_report://worker-0",
        );
        let TranslatedSegment::Delta(events) = &translated.segments[0] else {
            panic!("the deltas come first");
        };
        assert_eq!(
            events[0].event_type,
            ReportEventType::EventNodeRegister as i32
        );
        assert!(matches!(
            translated.segments[1],
            TranslatedSegment::Clear { register: false }
        ));

        // [CLEAR, REPORT]: nothing precedes the clear, so it carries it.
        let request = apply(
            "worker-0",
            "http://w0:30000",
            vec![clear(), report(2)],
            None,
        );
        let translated = translate_batch(
            &request,
            true,
            &[TierType::TierHbm],
            &["hbm".into()],
            "event_report://worker-0",
        );
        assert!(matches!(
            translated.segments[0],
            TranslatedSegment::Clear { register: true }
        ));
    }

    /// A clear's position is its meaning: deltas before it belong to the
    /// generation it ends, deltas after it belong to the fresh one.
    #[test]
    fn translate_batch_orders_deltas_around_a_clear() {
        let clear = || ExternalKvAction {
            r#type: ExternalKvActionType::ActionClearAllAtTier as i32,
            tier: hbm(),
            hashes: Vec::new(),
            component_masks: Vec::new(),
            block_sizes: Vec::new(),
            parent_block_hash: None,
        };
        let report = |hash: i64| report_action(hbm(), &[hash], &[COMPONENT_FULL]);
        // [REPORT 1, CLEAR, REPORT 2, CLEAR]: the second clear emits no empty
        // delta of its own, and the last one is followed by nothing.
        let request = apply(
            "worker-0",
            "http://w0:30000",
            vec![report(1), clear(), report(2), clear()],
            None,
        );
        let translated = translate_batch(
            &request,
            false,
            &[TierType::TierHbm],
            &["hbm".into()],
            "event_report://worker-0",
        );
        let shape: Vec<&str> = translated
            .segments
            .iter()
            .map(|segment| match segment {
                TranslatedSegment::Delta(events) => {
                    assert_eq!(events.len(), 1, "one ADD per report action");
                    "delta"
                }
                TranslatedSegment::Clear { .. } => "clear",
            })
            .collect();
        assert_eq!(shape, ["delta", "clear", "delta", "clear"]);

        // A batch that opens with a clear has no leading delta to send.
        let request = apply(
            "worker-0",
            "http://w0:30000",
            vec![clear(), report(3)],
            None,
        );
        let translated = translate_batch(
            &request,
            false,
            &[TierType::TierHbm],
            &["hbm".into()],
            "event_report://worker-0",
        );
        let shape: Vec<&str> = translated
            .segments
            .iter()
            .map(|segment| match segment {
                TranslatedSegment::Delta(_) => "delta",
                TranslatedSegment::Clear { .. } => "clear",
            })
            .collect();
        assert_eq!(shape, ["clear", "delta"]);
    }

    #[test]
    fn translate_batch_skips_blocks_with_only_unknown_components() {
        let request = apply(
            "worker-0",
            "http://w0:30000",
            vec![report_action(hbm(), &[1, 2], &[1 << 3, 0])],
            None,
        );
        let translated = translate_batch(
            &request,
            false,
            &[TierType::TierHbm],
            &["hbm".into()],
            "event_report://worker-0",
        );
        let events = delta_events(&translated);
        assert_eq!(events.len(), 1);
        let EventItem {
            event_params: Some(event_item::EventParams::BlockAdd(add)),
            ..
        } = &events[0]
        else {
            panic!("expected BLOCK_ADD");
        };
        assert_eq!(add.block_key, "2");
    }

    #[test]
    fn negative_hash_keys_round_trip_as_signed_decimal() {
        // kv_indexer hashes are sfixed64 and may be negative; Tair KVCache block keys
        // are int64 parsed from decimal, so -1 must stay -1 on both sides.
        let request = apply(
            "worker-0",
            "",
            vec![report_action(hbm(), &[-1], &[0])],
            None,
        );
        let translated = translate_batch(
            &request,
            false,
            &[TierType::TierHbm],
            &["hbm".into()],
            "event_report://worker-0",
        );
        let events = delta_events(&translated);
        let EventItem {
            event_params: Some(event_item::EventParams::BlockAdd(add)),
            ..
        } = &events[0]
        else {
            panic!("expected BLOCK_ADD");
        };
        assert_eq!(add.block_key, "-1");
        assert_eq!(add.block_key.parse::<i64>(), Ok(-1));
    }

    #[test]
    fn legacy_workers_stay_eligible() {
        let request = apply(
            "worker-0",
            "http://w0:30000",
            vec![
                report_action(hbm(), &[1], &[]),
                report_action(dram(), &[2], &[]),
            ],
            None,
        );
        // DRAM is filtered from reporting but still validated as a tier.
        assert!(batch_is_tair_kvcache_eligible(&request, &[TierType::TierHbm]).unwrap());
    }

    #[test]
    fn component_detail_without_a_spec_makes_the_worker_ineligible() {
        // The rule engine fails such a worker closed (0); whole-block
        // matching would answer a prefix, so the worker must be excluded.
        let request = apply(
            "worker-0",
            "http://w0:30000",
            vec![report_action(hbm(), &[1], &[COMPONENT_FULL])],
            None,
        );
        assert!(!batch_is_tair_kvcache_eligible(&request, &[TierType::TierHbm]).unwrap());
    }

    #[test]
    fn swa_and_mamba_specs_make_the_worker_ineligible() {
        for components in [
            COMPONENT_FULL | COMPONENT_SWA,
            COMPONENT_FULL | COMPONENT_MAMBA,
        ] {
            let request = apply(
                "worker-0",
                "http://w0:30000",
                vec![report_action(hbm(), &[1], &[components])],
                Some(spec(components, 1, 1 << hbm())),
            );
            assert!(!batch_is_tair_kvcache_eligible(&request, &[TierType::TierHbm]).unwrap());
        }
    }

    #[test]
    fn full_only_specs_stay_eligible_within_servable_reported_tiers() {
        let masks = [COMPONENT_FULL];
        let request = apply(
            "worker-0",
            "http://w0:30000",
            vec![report_action(hbm(), &[1], &masks)],
            Some(spec(COMPONENT_FULL, 1, 1 << hbm())),
        );
        assert!(batch_is_tair_kvcache_eligible(&request, &[TierType::TierHbm]).unwrap());
        let request = apply(
            "worker-0",
            "http://w0:30000",
            vec![report_action(hbm(), &[1], &masks)],
            Some(spec(COMPONENT_FULL, 1, 1 << dram())),
        );
        assert!(!batch_is_tair_kvcache_eligible(&request, &[TierType::TierHbm]).unwrap());
        // Future spec versions and empty component sets: ineligible.
        for bad in [spec(COMPONENT_FULL, 2, 1 << hbm()), spec(0, 1, 1 << hbm())] {
            let request = apply(
                "worker-0",
                "http://w0:30000",
                vec![report_action(hbm(), &[1], &masks)],
                Some(bad),
            );
            assert!(!batch_is_tair_kvcache_eligible(&request, &[TierType::TierHbm]).unwrap());
        }
        // Legacy (mask-less) reports under a spec: the rule engine cannot
        // confirm FULL, so neither can whole-block matching.
        let request = apply(
            "worker-0",
            "http://w0:30000",
            vec![report_action(hbm(), &[1], &[])],
            Some(spec(COMPONENT_FULL, 1, 1 << hbm())),
        );
        assert!(!batch_is_tair_kvcache_eligible(&request, &[TierType::TierHbm]).unwrap());
        // A zero mask inside a component batch is the same hole: the bridge
        // encodes mask-less blocks as 0 next to component blocks, and the
        // rule engine fails the unconfirmed block closed.
        let request = apply(
            "worker-0",
            "http://w0:30000",
            vec![report_action(hbm(), &[1, 2], &[COMPONENT_FULL, 0])],
            Some(spec(COMPONENT_FULL, 1, 1 << hbm())),
        );
        assert!(!batch_is_tair_kvcache_eligible(&request, &[TierType::TierHbm]).unwrap());
    }

    #[test]
    fn eligibility_validation_rejects_malformed_actions() {
        // The service layer already validates these; the trait is public, so
        // a direct caller must get an error, not a panic.
        let mut action = report_action(hbm(), &[1, 2], &[COMPONENT_FULL]);
        action.component_masks = vec![COMPONENT_FULL]; // misaligned
        let request = apply("worker-0", "", vec![action], None);
        assert_eq!(
            batch_is_tair_kvcache_eligible(&request, &[TierType::TierHbm])
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );
        let mut action = report_action(hbm(), &[1], &[COMPONENT_FULL]);
        action.r#type = 99;
        let request = apply("worker-0", "", vec![action], None);
        assert!(batch_is_tair_kvcache_eligible(&request, &[TierType::TierHbm]).is_err());
    }

    #[test]
    fn prefix_entry_clamps_and_filters() {
        let host = |host: &str, local: i64| HostCacheMatch {
            host_ip_port: host.to_string(),
            local,
            p2p_1_fetch: 0,
            p2p_1_total_match: 0,
        };
        let record = |address: &str, eligible: bool| WorkerRecord {
            address: address.to_string(),
            node_registered: true,
            tair_kvcache_eligible: eligible,
        };
        assert_eq!(
            prefix_entry(host("w0", 3), 10, &record("http://w0", true)),
            Some(("w0".into(), "http://w0".into(), 3))
        );
        // A Tair KVCache over-report cannot exceed the query length.
        assert_eq!(
            prefix_entry(host("w0", 99), 10, &record("http://w0", true)).map(|e| e.2),
            Some(10)
        );
        assert_eq!(
            prefix_entry(host("w0", 0), 10, &record("http://w0", true)),
            None
        );
        assert_eq!(
            prefix_entry(host("w0", -1), 10, &record("http://w0", true)),
            None
        );
        assert_eq!(prefix_entry(host("w0", 5), 10, &record("", true)), None);
        assert_eq!(
            prefix_entry(host("w0", 5), 10, &record("http://w0", false)),
            None
        );
    }

    #[test]
    fn parse_storage_type_accepts_both_channels() {
        assert_eq!(parse_storage_type(None), Ok(StorageType::StEventReportL1p5));
        assert_eq!(
            parse_storage_type(Some("l1p5")),
            Ok(StorageType::StEventReportL1p5)
        );
        assert_eq!(
            parse_storage_type(Some("l2")),
            Ok(StorageType::StEventReportL2)
        );
        assert!(parse_storage_type(Some("l3")).is_err());
    }

    #[test]
    fn parse_report_tiers_defaults_to_hbm_and_dedups() {
        assert_eq!(parse_report_tiers(None), Ok(vec![TierType::TierHbm]));
        assert_eq!(
            parse_report_tiers(Some("hbm, dram ,HBM")),
            Ok(vec![TierType::TierHbm, TierType::TierDram])
        );
        assert!(parse_report_tiers(Some("hbm,nope")).is_err());
        assert!(parse_report_tiers(Some(" ")).is_err());
    }

    #[test]
    fn check_header_maps_tair_kvcache_codes() {
        let header = |code: ErrorCode| CommonResponseHeader {
            status: Some(crate::tair_kvcache_pb::Status {
                code: code as i32,
                message: "boom".into(),
            }),
            request_id: String::new(),
            tracer_result: String::new(),
        };
        assert!(check_header(Some(&header(ErrorCode::Ok)), "rpc").is_ok());
        let status = check_header(Some(&header(ErrorCode::InvalidArgument)), "rpc").unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("boom"));
        // Permanent configuration faults must not masquerade as transient
        // unavailability: the bridge exits on them instead of reconnecting
        // forever against a typo'd instance id.
        for code in [
            ErrorCode::InstanceNotExist,
            ErrorCode::Unsupported,
            ErrorCode::ReachMaxEntityCapacity,
            ErrorCode::DuplicateEntity,
        ] {
            let status = check_header(Some(&header(code)), "rpc").unwrap_err();
            assert_eq!(status.code(), tonic::Code::FailedPrecondition, "{code:?}");
        }
        // A reporter Tair KVCache forgot is an expected, self-healing event: the
        // bridge retries (transient), and the next report re-registers.
        let status = check_header(Some(&header(ErrorCode::NodeNotRegistered)), "rpc").unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unavailable);
        let status = check_header(Some(&header(ErrorCode::ServiceNotReady)), "rpc").unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unavailable);
        assert!(check_header(None, "rpc").is_err());
    }

    /// The config defaults must stay self-consistent: mediums derive from the
    /// default tier set.
    #[test]
    fn config_defaults_report_hbm_only() {
        let config = TairKVCacheBackendConfig::new("http://tair_kvcache:1", "inst");
        assert_eq!(config.storage_type, StorageType::StEventReportL1p5);
        assert_eq!(config.report_tiers, [TierType::TierHbm]);
        assert_eq!(config.auto_register, None);
        assert_eq!(config.report_mediums(), ["hbm"]);
    }

    /// The environment variables are the operator's entry point, and the one
    /// place a mistake is silent: a renamed or unenforced variable means the
    /// backend registers without the model deployment Tair KVCache demands, and every
    /// report then fails `INVALID_ARGUMENT` (the bridge exits). This is the
    /// only test in the crate that touches `KV_INDEXER_TAIR_KVCACHE_KVCM_*`, and no other
    /// test reads them, so one sequential test keeps the global environment to
    /// itself.
    #[test]
    fn from_env_assembles_and_validates_the_auto_registration() {
        const KEYS: [&str; 7] = [
            ENV_ENDPOINT,
            ENV_INSTANCE_ID,
            ENV_AUTO_REGISTER,
            ENV_INSTANCE_GROUP,
            ENV_BLOCK_SIZE,
            ENV_MODEL_NAME,
            ENV_MODEL_DTYPE,
        ];
        for key in KEYS {
            std::env::remove_var(key);
        }

        std::env::set_var(ENV_ENDPOINT, "http://127.0.0.1:50052");
        std::env::set_var(ENV_INSTANCE_ID, "inst-0");
        // Auto-registration is opt-in: without the flag the deployment identity
        // is not required, because the connector registered the instance.
        let config =
            TairKVCacheBackendConfig::from_env().expect("endpoint and instance are enough");
        assert_eq!(config.endpoint, "http://127.0.0.1:50052");
        assert_eq!(config.instance_id, "inst-0");
        assert_eq!(config.auto_register, None);

        std::env::set_var(ENV_AUTO_REGISTER, "1");
        assert_eq!(
            TairKVCacheBackendConfig::from_env().unwrap_err(),
            TairKVCacheConfigError::Missing(ENV_INSTANCE_GROUP)
        );
        std::env::set_var(ENV_INSTANCE_GROUP, "group-a");
        assert!(matches!(
            TairKVCacheBackendConfig::from_env().unwrap_err(),
            TairKVCacheConfigError::Invalid {
                key: ENV_BLOCK_SIZE,
                ..
            }
        ));
        std::env::set_var(ENV_BLOCK_SIZE, "0");
        assert!(matches!(
            TairKVCacheBackendConfig::from_env().unwrap_err(),
            TairKVCacheConfigError::Invalid {
                key: ENV_BLOCK_SIZE,
                ..
            }
        ));
        std::env::set_var(ENV_BLOCK_SIZE, "128");
        // Tair KVCache rejects a registration without a model name, a dtype, or a
        // positive block size, so each one has to be a hard requirement.
        assert_eq!(
            TairKVCacheBackendConfig::from_env().unwrap_err(),
            TairKVCacheConfigError::Missing(ENV_MODEL_NAME)
        );
        std::env::set_var(ENV_MODEL_NAME, "qwen3-8b");
        assert_eq!(
            TairKVCacheBackendConfig::from_env().unwrap_err(),
            TairKVCacheConfigError::Missing(ENV_MODEL_DTYPE)
        );
        std::env::set_var(ENV_MODEL_DTYPE, "FP8");

        let config = TairKVCacheBackendConfig::from_env().expect("complete environment");
        assert_eq!(
            config.auto_register,
            Some(TairKVCacheAutoRegister {
                instance_group: "group-a".to_string(),
                block_size: 128,
                model_name: "qwen3-8b".to_string(),
                dtype: "FP8".to_string(),
            })
        );

        for key in KEYS {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn prefix_request_truncation_uses_the_shared_helper() {
        // The override must truncate exactly like the default implementation.
        let request = MatchExternalKvPrefixRequest {
            hashes: vec![1; 10],
            max_blocks: 4,
        };
        let limit = prefix_limit(request.hashes.len(), request.max_blocks);
        assert_eq!(limit, 4);
    }
}
