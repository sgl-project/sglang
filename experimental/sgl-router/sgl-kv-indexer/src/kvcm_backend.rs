// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! KVCM backend: a [`KvIndexerBackend`] whose placement state lives in a Tair
//! KVCache Manager (KVCM) `MetaService` instead of this process.
//!
//! # Shape
//!
//! The ingestion side is unchanged: the same bridge turns each worker's ZMQ
//! KV-event stream into [`ApplyExternalKvBatch`] RPCs. This backend translates
//! those actions into KVCM `ReportEvent` mutations — REPORT becomes
//! `BLOCK_ADD` (one component spec per resident component), REVOKE becomes
//! `BLOCK_DELETE`, and the reporter identity is the indexer `worker_id`, so
//! `GetHostCacheState` answers map straight back onto workers.
//!
//! Query side: `MatchExternalKvPrefix` is answered by `GetHostCacheState`
//! (`QT_PREFIX_MATCH`), whose per-host result is the longest contiguous run of
//! held blocks from block 0 — the *legacy whole-block* quantity. That is not
//! the rule engine's answer for a component-aware worker, so a worker whose
//! cache spec requires SWA or MAMBA rules, or that reports components without
//! a spec, is excluded from the query answer (see [`batch_is_kvcm_eligible`]):
//! the answer stays a lower bound, never an over-report — except for the
//! stale placements a skipped `CLEAR_ALL_AT_TIER` leaves behind (documented
//! below). The per-block placement query (`MatchExternalKv`) has no KVCM
//! equivalent -- KVCM answers hosts, not per-block location lists -- so it
//! fails `UNIMPLEMENTED` and the prefix query is the only routing signal.
//!
//! # Deliberate limitations
//!
//! `CLEAR_ALL_AT_TIER` has no KVCM equivalent and is skipped with a warning;
//! host-dram placements are not reported by default; the instance is registered
//! only when auto-registration is configured; the reporter→address map is
//! process-local, so a restart loses the join and not the placements; and
//! `snapshot_required` marks a reporter lifecycle rather than a fault. The
//! README carries the operational contract and the failure mode behind each.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use tonic::transport::Channel;
use tonic::{Request, Status};

use crate::client::parse_endpoint;
use crate::kvcm_pb::meta_service_client::MetaServiceClient;
use crate::kvcm_pb::{
    event_item, BlockAddEventParams, BlockDeleteEventParams, CommonResponseHeader, ErrorCode,
    EventItem, GetHostCacheStateRequest, GetHostCacheStateResponse, HeartbeatEventParams,
    HostCacheMatch, LocationSpec, LocationSpecInfo, ModelDeployment, NodeRegisterEventParams,
    QueryType, RegisterInstanceRequest, ReportEventRequest, ReportEventType, StorageType,
};
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

/// Per-query deadline for `GetHostCacheState`. Half the router-side prefix
/// deadline (`client::DEFAULT_QUERY_DEADLINE`), because this budget starts
/// only after the service layer's admission checks: a full router-deadline
/// budget here means the indexer keeps scanning after the router gave up.
pub const DEFAULT_QUERY_TIMEOUT_MS: u64 = 50;
/// Deadline for mutation and registration RPCs, shared by every chunk of one
/// apply batch (see [`KvcmIndexerBackend::apply_external_kv_batch`]). Applies
/// are never shed by the service layer, so this bounds a hung KVCM from
/// stalling a bridge.
pub const DEFAULT_APPLY_TIMEOUT_MS: u64 = 5_000;
/// Default heartbeat cadence. The backend heartbeats every worker it has seen
/// for its own lifetime, so an idle worker's cache stays routable at the cost of
/// KVCM liveness never reclaiming a dead reporter -- the Router's health
/// intersection is what keeps stale entries unroutable.
pub const DEFAULT_HEARTBEAT_INTERVAL_MS: u64 = 30_000;
/// Events per `ReportEvent` RPC. A maximal apply batch (16,384 hashes) would
/// otherwise produce one multi-hundred-kilobyte message; chunking keeps each
/// RPC comfortably under transport ceilings while preserving event order.
const MAX_EVENTS_PER_REPORT_EVENT: usize = 4_096;

const ENV_ENDPOINT: &str = "KV_INDEXER_KVCM_ENDPOINT";
const ENV_INSTANCE_ID: &str = "KV_INDEXER_KVCM_INSTANCE_ID";
const ENV_STORAGE_TYPE: &str = "KV_INDEXER_KVCM_STORAGE_TYPE";
const ENV_REPORT_TIERS: &str = "KV_INDEXER_KVCM_REPORT_TIERS";
const ENV_AUTO_REGISTER: &str = "KV_INDEXER_KVCM_AUTO_REGISTER";
const ENV_INSTANCE_GROUP: &str = "KV_INDEXER_KVCM_INSTANCE_GROUP";
const ENV_BLOCK_SIZE: &str = "KV_INDEXER_KVCM_BLOCK_SIZE";
const ENV_MODEL_NAME: &str = "KV_INDEXER_KVCM_MODEL_NAME";
const ENV_MODEL_DTYPE: &str = "KV_INDEXER_KVCM_MODEL_DTYPE";
const ENV_QUERY_TIMEOUT_MS: &str = "KV_INDEXER_KVCM_QUERY_TIMEOUT_MS";
const ENV_APPLY_TIMEOUT_MS: &str = "KV_INDEXER_KVCM_APPLY_TIMEOUT_MS";
const ENV_HEARTBEAT_INTERVAL_MS: &str = "KV_INDEXER_KVCM_HEARTBEAT_INTERVAL_MS";

/// Location-spec names reported per component. They double as the
/// `LocationSpecInfo` names an auto-registered instance declares, and as the
/// `spec_names` a REVOKE deletes (superset removal: a revoke drops the whole
/// placement, whatever components it held).
pub const SPEC_FULL: &str = "full";
pub const SPEC_SWA: &str = "swa";
pub const SPEC_MAMBA: &str = "mamba";
const SPEC_NAMES: [&str; 3] = [SPEC_FULL, SPEC_SWA, SPEC_MAMBA];

/// A misconfigured [`KvcmBackendConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvcmConfigError {
    Missing(&'static str),
    Invalid { key: &'static str, reason: String },
}

impl KvcmConfigError {
    fn invalid(key: &'static str, reason: impl Into<String>) -> Self {
        Self::Invalid {
            key,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for KvcmConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(key) => write!(f, "{key} is required"),
            Self::Invalid { key, reason } => write!(f, "invalid {key}: {reason}"),
        }
    }
}

impl std::error::Error for KvcmConfigError {}

/// Instance registration for routing-only deployments (the KVCM connector
/// registers the instance otherwise, and a second registration with different
/// fields fails `DUPLICATE_ENTITY`).
///
/// KVCM refuses a registration that omits the model deployment or a positive
/// block size (`MetaServiceImpl::RegisterInstance`), so both are required here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvcmAutoRegister {
    pub instance_group: String,
    pub block_size: u32,
    /// The routed deployment's identity, as a KVCM `ModelDeployment`: the
    /// indexer knows the fleet's workers, not their model, so the operator
    /// states the name and dtype KVCM demands (a pre-registered instance
    /// accepts this registration only when its fields match; otherwise it
    /// fails `DUPLICATE_ENTITY`). Other `ModelDeployment` fields keep their
    /// KVCM default.
    pub model_name: String,
    pub dtype: String,
}

/// KVCM backend configuration.
#[derive(Debug, Clone)]
pub struct KvcmBackendConfig {
    /// gRPC endpoint of the KVCM MetaService, e.g. `http://10.0.0.1:50052`.
    pub endpoint: String,
    /// KVCM instance, one per model deployment across the routed fleet.
    pub instance_id: String,
    /// Event-report channel the deployment provisioned (`l1p5` or `l2`).
    pub storage_type: StorageType,
    /// Tiers whose events are forwarded to KVCM. Everything else is dropped.
    pub report_tiers: Vec<TierType>,
    /// Opt-in `RegisterInstance` before the first apply batch or query.
    pub auto_register: Option<KvcmAutoRegister>,
    pub query_timeout: Duration,
    pub apply_timeout: Duration,
    /// `None` disables heartbeats (KVCM liveness then reclaims idle workers).
    pub heartbeat_interval: Option<Duration>,
}

impl KvcmBackendConfig {
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

    /// Builds the configuration from `KV_INDEXER_KVCM_*` environment
    /// variables. `KV_INDEXER_KVCM_ENDPOINT` and
    /// `KV_INDEXER_KVCM_INSTANCE_ID` are required.
    pub fn from_env() -> Result<Self, KvcmConfigError> {
        let endpoint = env_nonempty(ENV_ENDPOINT).ok_or(KvcmConfigError::Missing(ENV_ENDPOINT))?;
        let instance_id =
            env_nonempty(ENV_INSTANCE_ID).ok_or(KvcmConfigError::Missing(ENV_INSTANCE_ID))?;
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
                    .ok_or(KvcmConfigError::Missing(ENV_INSTANCE_GROUP))?;
                let block_size = parse_u64(ENV_BLOCK_SIZE, 0)?;
                let block_size = u32::try_from(block_size)
                    .map_err(|_| KvcmConfigError::invalid(ENV_BLOCK_SIZE, "value exceeds u32"))?;
                if block_size == 0 {
                    return Err(KvcmConfigError::invalid(
                        ENV_BLOCK_SIZE,
                        "must be the fleet's block size; KVCM rejects a missing or zero value",
                    ));
                }
                config.auto_register = Some(KvcmAutoRegister {
                    instance_group,
                    block_size,
                    model_name: env_nonempty(ENV_MODEL_NAME)
                        .ok_or(KvcmConfigError::Missing(ENV_MODEL_NAME))?,
                    dtype: env_nonempty(ENV_MODEL_DTYPE)
                        .ok_or(KvcmConfigError::Missing(ENV_MODEL_DTYPE))?,
                });
            } else if flag != "0" && !flag.eq_ignore_ascii_case("false") {
                return Err(KvcmConfigError::invalid(
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

/// Medium string a tier reports under. Free-form in the KVCM contract; these
/// names match the vocabulary KVCM's own tools use (`hbm`/`mem`/`disk`).
fn tier_medium(tier: TierType) -> Option<&'static str> {
    match tier {
        TierType::TierHbm => Some("hbm"),
        TierType::TierDram => Some("mem"),
        TierType::TierSsd => Some("disk"),
        TierType::TierUnknown => None,
    }
}

fn parse_storage_type(raw: Option<&str>) -> Result<StorageType, KvcmConfigError> {
    match raw {
        None | Some("l1p5") => Ok(StorageType::StEventReportL1p5),
        Some("l2") => Ok(StorageType::StEventReportL2),
        Some(other) => Err(KvcmConfigError::invalid(
            ENV_STORAGE_TYPE,
            format!("expected `l1p5` or `l2`, got `{other}`"),
        )),
    }
}

fn parse_report_tiers(raw: Option<&str>) -> Result<Vec<TierType>, KvcmConfigError> {
    let raw = raw.unwrap_or("HBM");
    let mut tiers = Vec::new();
    for part in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let tier = match part.to_ascii_uppercase().as_str() {
            "HBM" => TierType::TierHbm,
            "DRAM" => TierType::TierDram,
            "SSD" => TierType::TierSsd,
            other => {
                return Err(KvcmConfigError::invalid(
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
        return Err(KvcmConfigError::invalid(
            ENV_REPORT_TIERS,
            "at least one tier is required",
        ));
    }
    Ok(tiers)
}

/// Parses a millisecond timeout, `default` when unset, capped at one day:
/// the apply path adds the value to an `Instant`, which panics on overflow.
fn parse_timeout_ms(key: &'static str, default: u64) -> Result<u64, KvcmConfigError> {
    let value = parse_u64(key, default)?;
    if value > 86_400_000 {
        return Err(KvcmConfigError::invalid(
            key,
            format!("must be at most 86400000 (one day), got {value}"),
        ));
    }
    Ok(value)
}

/// Parses a non-negative integer, `default` when unset; a `0` is meaningful
/// rather than an error (it disables heartbeats and means "no block size").
fn parse_u64(key: &'static str, default: u64) -> Result<u64, KvcmConfigError> {
    let Some(raw) = env_nonempty(key) else {
        return Ok(default);
    };
    raw.parse::<u64>().map_err(|_| {
        KvcmConfigError::invalid(key, format!("expected a non-negative integer, got `{raw}`"))
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

/// Resident component-spec names for a reported bitmask. Legacy (mask 0) is an
/// implicit whole-block store: the full component alone. A non-zero mask without
/// FULL yields no specs at all -- such a block cannot serve full attention, and
/// KVCM counts a held block by *any* spec present, so reporting it would
/// over-report. The caller drops it, matching the rule engine's fail-closed
/// rule that FULL is required on every matched block.
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

/// Whether one apply batch keeps its worker answerable by KVCM's whole-block
/// prefix match *without over-reporting* relative to the rule engine in
/// `service::compute_worker_prefix`.
///
/// The rule engine gates a component-aware worker on more than block presence
/// (SWA must cover a trailing window, MAMBA must sit on the boundary,
/// fail-closed detail without a spec) and `GetHostCacheState` answers none of
/// it, so a worker stays answerable only while every batch keeps it legacy
/// whole-block, or FULL-only within servable reported tiers with component
/// detail on every block.
fn batch_is_kvcm_eligible(
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
            // A legacy worker that reports component detail would fail the
            // rule engine closed; it is not answerable here either.
            None => {
                if !action.component_masks.is_empty() {
                    return Ok(false);
                }
            }
            Some(spec) => {
                // Every mask must confirm FULL. A non-empty array can still
                // carry 0 for a legacy block (the bridge encodes mask-less
                // blocks as 0 inside a component batch), and the rule engine
                // fails those closed: an unconfirmed block must not become a
                // reported whole-block placement.
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

/// URI scheme KVCM expects for a channel's spec URIs (l1p5 reports as
/// `event_report://`, l2 as `vineyard://`).
fn storage_uri_scheme(storage_type: StorageType) -> &'static str {
    match storage_type {
        StorageType::StEventReportL2 => "vineyard",
        _ => "event_report",
    }
}

/// The URI authority identifying the reporter: the worker address's
/// `host:port` when one is restated, else the worker id. KVCM validates spec
/// URIs structurally (scheme + non-empty rest); the data plane later resolves
/// the authority when fetching.
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
                // backend could have reported; KVCM treats missing names as no-ops.
                spec_names: SPEC_NAMES.iter().map(|name| name.to_string()).collect(),
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

/// Whether an event list mutates placements. KVCM answers `snapshot_required`
/// for a reporter with no committed snapshot version, and the request that
/// creates one is a delta request (see `CacheManager::refresh_snapshot_response`).
fn carries_delta(events: &[EventItem]) -> bool {
    events.iter().any(|event| {
        event.event_type == ReportEventType::EventBlockAdd as i32
            || event.event_type == ReportEventType::EventBlockDelete as i32
            || event.event_type == ReportEventType::EventBlockSnapshot as i32
    })
}

/// One apply batch, translated. Event order matches action order.
struct TranslatedBatch {
    events: Vec<EventItem>,
    /// `CLEAR_ALL_AT_TIER` actions with no KVCM equivalent (skipped, warned).
    skipped_clears: usize,
}

/// Pure translation of one validated apply batch into KVCM report events.
/// Actions on tiers outside `report_tiers` are dropped before translation;
/// `prepend_node_register` re-registers a reporter KVCM has forgotten (or
/// never accepted).
fn translate_batch(
    request: &ApplyExternalKvBatchRequest,
    prepend_node_register: bool,
    report_tiers: &[TierType],
    mediums: &[String],
    uri_prefix: &str,
) -> TranslatedBatch {
    let mut events = Vec::new();
    if prepend_node_register {
        events.push(node_register_event(mediums));
    }
    let mut skipped_clears = 0;
    for action in &request.actions {
        // The service layer already rejects unknown tiers and action types;
        // `batch_is_kvcm_eligible` re-checks both for direct callers.
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
                skipped_clears += 1;
            }
            Ok(ExternalKvActionType::ActionUnknown) | Err(_) => {}
        }
    }
    TranslatedBatch {
        events,
        skipped_clears,
    }
}

/// The outcome a KVCM response header carries.
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

/// Maps a KVCM response header into a `Result`. `INVALID_ARGUMENT` and
/// `DUPLICATE_ENTITY` stay loud contract errors; permanent configuration
/// faults (`INSTANCE_NOT_EXIST`, `UNSUPPORTED`, `REACH_MAX_ENTITY_CAPACITY`)
/// become `FAILED_PRECONDITION`, which the bridge classifies as permanent
/// instead of reconnecting forever on a typo'd instance id. Every other code
/// — including `NODE_NOT_REGISTERED`, an expected event for a reporter KVCM
/// forgot and the next report re-registers — is a transient availability
/// problem the caller retries or degrades on.
fn check_header(header: Option<&CommonResponseHeader>, what: &str) -> Result<(), Status> {
    match header_outcome(header) {
        HeaderOutcome::Ok => Ok(()),
        HeaderOutcome::Failed { code, message } => Err(match code {
            ErrorCode::InvalidArgument => {
                Status::invalid_argument(format!("KVCM rejected {what}: {message}"))
            }
            ErrorCode::DuplicateEntity => Status::failed_precondition(format!(
                "KVCM {what}: entity already registered with different configuration: {message}"
            )),
            ErrorCode::InstanceNotExist
            | ErrorCode::Unsupported
            | ErrorCode::ReachMaxEntityCapacity => Status::failed_precondition(format!(
                "KVCM {what} failed on a permanent configuration fault: {code:?}: {message}"
            )),
            code => Status::unavailable(format!("KVCM {what} failed: {code:?}: {message}")),
        }),
    }
}

/// `(worker_id, address, matched blocks)` for one KVCM host match: the worker
/// must be answerable (see [`batch_is_kvcm_eligible`]) with a non-empty
/// router-facing address, and the reported prefix is clamped to the query
/// length so a KVCM over-report cannot exceed the ask.
fn prefix_entry(
    host: HostCacheMatch,
    blocks_read: u32,
    record: &WorkerRecord,
) -> Option<(String, String, u32)> {
    if !record.kvcm_eligible || record.address.is_empty() {
        return None;
    }
    let matched = u32::try_from(host.local).ok()?.min(blocks_read);
    (matched > 0).then(|| (host.host_ip_port, record.address.clone(), matched))
}

/// Router-facing bookkeeping per worker, refreshed by every apply batch.
#[derive(Debug, Default, Clone)]
struct WorkerRecord {
    address: String,
    /// Whether a `NODE_REGISTER` for this reporter has been accepted: a first
    /// batch that fails must not burn the registration, and KVCM may forget the
    /// node after a liveness cleanup, so the flag flips only on a successful
    /// report carrying one.
    node_registered: bool,
    /// Sticky: once a batch makes the worker unanswerable by whole-block
    /// matching (see [`batch_is_kvcm_eligible`]), it stays excluded — the
    /// placements KVCM already holds cannot be un-reported.
    kvcm_eligible: bool,
}

struct Inner {
    client: MetaServiceClient<Channel>,
    config: KvcmBackendConfig,
    mediums: Vec<String>,
    /// Router-facing record per worker, refreshed by every apply batch: the
    /// query side resolves KVCM reporter ids back into addresses here.
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
            .map_err(|_| Status::internal("KVCM backend workers map lock poisoned"))
    }

    fn set_node_registered(&self, host: &str, registered: bool) {
        if let Ok(mut workers) = self.workers.write() {
            if let Some(record) = workers.get_mut(host) {
                record.node_registered = registered;
            }
        }
    }

    /// Forwards one `ReportEvent` and maintains the reporter's registration
    /// flag: a successful report carrying `NODE_REGISTER` marks it registered,
    /// and a `NODE_NOT_REGISTERED` failure resets it so the next report
    /// re-registers.
    async fn report_event_checked(
        &self,
        host: &str,
        events: Vec<EventItem>,
        timeout: Duration,
    ) -> Result<(), Status> {
        let registered_node = carries_node_register(&events);
        let has_delta = carries_delta(&events);
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
            .map_err(|_| Status::deadline_exceeded("KVCM ReportEvent timed out"))??;
        let response = response.into_inner();
        if let HeaderOutcome::Failed {
            code: ErrorCode::NodeNotRegistered,
            ..
        } = header_outcome(response.header.as_ref())
        {
            self.set_node_registered(host, false);
        }
        if response.snapshot_required {
            // A lifecycle start, not a fault: a request carrying deltas is
            // what creates the version KVCM is asking about, while a beat from
            // a reporter that never wrote one is only worth a debug line. This
            // backend holds no placement state and cannot reconcile either way.
            if has_delta {
                tracing::info!(
                    host,
                    "KVCM has no committed snapshot for this reporter; it stays delta-only"
                );
            } else {
                tracing::debug!(
                    host,
                    "KVCM still has no committed snapshot for this reporter"
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
                        "KVCM ReportEvent partially failed ({what}): item results {:?}",
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
        if registered_node {
            self.set_node_registered(host, true);
        }
        Ok(())
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
        .map_err(|_| Status::deadline_exceeded("KVCM GetHostCacheState timed out"))??;
        let response = response.into_inner();
        check_header(response.header.as_ref(), "GetHostCacheState")?;
        Ok(response)
    }

    /// One-shot `RegisterInstance` when auto-registration is configured.
    ///
    /// It runs on the apply budget, not a caller's: this is a setup call rather
    /// than a data-plane answer. Concurrent first calls (a cold start) may each
    /// send one; KVCM is idempotent for identical fields, and after the first
    /// success the flag short-circuits the rest.
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
                .map_err(|_| Status::deadline_exceeded("KVCM RegisterInstance timed out"))??;
        check_header(response.into_inner().header.as_ref(), "RegisterInstance")?;
        self.instance_registered.store(true, Ordering::Release);
        Ok(())
    }
}

/// A [`KvIndexerBackend`] backed by a KVCM MetaService.
pub struct KvcmIndexerBackend {
    inner: Arc<Inner>,
}

impl fmt::Debug for KvcmIndexerBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KvcmIndexerBackend")
            .field("instance_id", &self.inner.config.instance_id)
            .field("storage_type", &self.inner.config.storage_type)
            .field("report_tiers", &self.inner.config.report_tiers)
            .finish_non_exhaustive()
    }
}

impl KvcmIndexerBackend {
    /// Connects lazily; a misconfigured endpoint is rejected up front (see
    /// [`parse_endpoint`]) rather than failing every RPC later.
    pub fn new(config: KvcmBackendConfig) -> Result<Self, crate::InvalidEndpoint> {
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
    /// dropping the backend stops the heartbeats; workers that stopped being
    /// answerable (component rules) are skipped, since nothing is reported
    /// for them and no answer should come back either.
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
                    (record.kvcm_eligible, record.node_registered)
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
                    tracing::warn!(worker = %worker_id, %error, "KVCM heartbeat failed");
                }
            }
        });
    }
}

#[tonic::async_trait]
impl KvIndexerBackend for KvcmIndexerBackend {
    async fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status> {
        // '#' is the KVCM location-id separator; a worker id carrying it would
        // corrupt every placement the reporter ever sends.
        if request.worker_id.contains('#') {
            return Err(Status::invalid_argument(
                "worker_id must not contain '#' (KVCM location-id component)",
            ));
        }
        let (first_seen, node_registered, kvcm_eligible) = {
            let mut workers = self
                .inner
                .workers
                .write()
                .map_err(|_| Status::internal("KVCM backend workers map lock poisoned"))?;
            match workers.entry(request.worker_id.clone()) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let record = entry.get_mut();
                    // The address is restated on every batch, so record it even
                    // when the batch translates to no events at all.
                    record.address = request.worker_address.clone();
                    (false, record.node_registered, record.kvcm_eligible)
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(WorkerRecord {
                        address: request.worker_address.clone(),
                        node_registered: false,
                        kvcm_eligible: true,
                    });
                    (true, false, true)
                }
            }
        };
        if first_seen {
            self.spawn_heartbeat(request.worker_id.clone());
        }
        self.inner.register_instance().await?;
        let batch_eligible = batch_is_kvcm_eligible(&request, &self.inner.config.report_tiers)?;
        // Eligibility only ever narrows: once a worker is excluded it stays
        // excluded, because KVCM already holds its earlier placements.
        if kvcm_eligible && !batch_eligible {
            tracing::warn!(
                worker = %request.worker_id,
                "cache spec or component detail requires prefix rules KVCM cannot evaluate; excluding the worker from prefix answers (fail closed)"
            );
            if let Ok(mut workers) = self.inner.workers.write() {
                if let Some(record) = workers.get_mut(&request.worker_id) {
                    record.kvcm_eligible = false;
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
        if translated.skipped_clears > 0 {
            tracing::warn!(
                worker = %request.worker_id,
                count = translated.skipped_clears,
                "CLEAR_ALL_AT_TIER has no KVCM equivalent; stale placements remain until KVCM capacity eviction removes them"
            );
        }
        // One deadline for the whole batch: a slow KVCM must not turn four
        // chunk RPCs into four full timeouts while the bridge's own channel
        // timeout expires underneath and drops the batch.
        let deadline = tokio::time::Instant::now() + self.inner.config.apply_timeout;
        for chunk in translated.events.chunks(MAX_EVENTS_PER_REPORT_EVENT) {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Err(Status::deadline_exceeded(
                    "KVCM apply budget exhausted before all chunks were forwarded",
                ));
            }
            self.inner
                .report_event_checked(&request.worker_id, chunk.to_vec(), remaining)
                .await?;
        }
        Ok(ApplyExternalKvBatchResponse::default())
    }

    async fn match_external_kv(
        &self,
        _request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        Err(Status::unimplemented(
            "the KVCM backend answers prefix queries only: KVCM has no per-worker per-block placement query (use MatchExternalKvPrefix)",
        ))
    }

    /// Answers via `GetHostCacheState`: per reporter host, the longest
    /// contiguous run of held blocks from block 0 — the legacy whole-block
    /// quantity. Component-aware workers the rule engine would gate on SWA
    /// windows or MAMBA boundaries are excluded rather than over-reported, so
    /// the answer stays a lower bound like the default implementation's. (The
    /// default path itself is unusable here: it derives placements from
    /// `match_external_kv`, which has no KVCM equivalent.)
    async fn match_external_kv_prefix(
        &self,
        request: MatchExternalKvPrefixRequest,
    ) -> Result<MatchExternalKvPrefixResponse, Status> {
        let limit = prefix_limit(request.hashes.len(), request.max_blocks);
        let hashes: Vec<i64> = request.hashes.into_iter().take(limit).collect();
        if hashes.is_empty() {
            return Ok(MatchExternalKvPrefixResponse::default());
        }
        // A prefix query can be the first thing this backend ever does, and an
        // unregistered instance is a permanent fault the router turns into a
        // 503 for every request. Registration needs nothing from a batch, so
        // the query path ensures it too; without `auto_register` the fault
        // stays, and is then a real configuration error.
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
        assert_eq!(translated.skipped_clears, 0);
        let EventItem {
            event_type,
            event_params: Some(event_item::EventParams::NodeRegister(register)),
        } = &translated.events[0]
        else {
            panic!("first event must be NODE_REGISTER");
        };
        assert_eq!(*event_type, ReportEventType::EventNodeRegister as i32);
        assert_eq!(register.mediums, ["hbm"]);
        let EventItem {
            event_params: Some(event_item::EventParams::BlockAdd(add)),
            ..
        } = &translated.events[1]
        else {
            panic!("expected BLOCK_ADD");
        };
        assert_eq!(add.block_key, "7");
        assert_eq!(add.medium, "hbm");
        assert_eq!(add.specs.len(), 1);
        assert_eq!(add.specs[0].name, SPEC_FULL);
        // KVCM rejects spec URIs that are empty or lack the channel's scheme.
        assert_eq!(add.specs[0].uri, "event_report://worker-0/hbm");
        let EventItem {
            event_params: Some(event_item::EventParams::BlockAdd(add)),
            ..
        } = &translated.events[2]
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
        } = &translated.events[3]
        else {
            panic!("expected BLOCK_DELETE");
        };
        assert_eq!(delete.block_key, "9");
        assert_eq!(delete.spec_names, SPEC_NAMES);
    }

    #[test]
    fn translate_batch_drops_unreported_tiers_and_counts_clears() {
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
        assert_eq!(translated.events.len(), 1);
        assert_eq!(translated.skipped_clears, 1);
        // A later batch with nothing to forward sends nothing.
        let translated = translate_batch(
            &request,
            false,
            &[TierType::TierHbm],
            &["hbm".into()],
            "event_report://worker-0",
        );
        assert!(translated.events.is_empty());
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
        assert_eq!(translated.events.len(), 1);
        let EventItem {
            event_params: Some(event_item::EventParams::BlockAdd(add)),
            ..
        } = &translated.events[0]
        else {
            panic!("expected BLOCK_ADD");
        };
        assert_eq!(add.block_key, "2");
    }

    #[test]
    fn negative_hash_keys_round_trip_as_signed_decimal() {
        // kv_indexer hashes are sfixed64 and may be negative; KVCM block keys
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
        let EventItem {
            event_params: Some(event_item::EventParams::BlockAdd(add)),
            ..
        } = &translated.events[0]
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
        assert!(batch_is_kvcm_eligible(&request, &[TierType::TierHbm]).unwrap());
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
        assert!(!batch_is_kvcm_eligible(&request, &[TierType::TierHbm]).unwrap());
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
            assert!(!batch_is_kvcm_eligible(&request, &[TierType::TierHbm]).unwrap());
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
        assert!(batch_is_kvcm_eligible(&request, &[TierType::TierHbm]).unwrap());
        let request = apply(
            "worker-0",
            "http://w0:30000",
            vec![report_action(hbm(), &[1], &masks)],
            Some(spec(COMPONENT_FULL, 1, 1 << dram())),
        );
        assert!(!batch_is_kvcm_eligible(&request, &[TierType::TierHbm]).unwrap());
        // Future spec versions and empty component sets: ineligible.
        for bad in [spec(COMPONENT_FULL, 2, 1 << hbm()), spec(0, 1, 1 << hbm())] {
            let request = apply(
                "worker-0",
                "http://w0:30000",
                vec![report_action(hbm(), &[1], &masks)],
                Some(bad),
            );
            assert!(!batch_is_kvcm_eligible(&request, &[TierType::TierHbm]).unwrap());
        }
        // Legacy (mask-less) reports under a spec: the rule engine cannot
        // confirm FULL, so neither can whole-block matching.
        let request = apply(
            "worker-0",
            "http://w0:30000",
            vec![report_action(hbm(), &[1], &[])],
            Some(spec(COMPONENT_FULL, 1, 1 << hbm())),
        );
        assert!(!batch_is_kvcm_eligible(&request, &[TierType::TierHbm]).unwrap());
        // A zero mask inside a component batch is the same hole: the bridge
        // encodes mask-less blocks as 0 next to component blocks, and the
        // rule engine fails the unconfirmed block closed.
        let request = apply(
            "worker-0",
            "http://w0:30000",
            vec![report_action(hbm(), &[1, 2], &[COMPONENT_FULL, 0])],
            Some(spec(COMPONENT_FULL, 1, 1 << hbm())),
        );
        assert!(!batch_is_kvcm_eligible(&request, &[TierType::TierHbm]).unwrap());
    }

    #[test]
    fn eligibility_validation_rejects_malformed_actions() {
        // The service layer already validates these; the trait is public, so
        // a direct caller must get an error, not a panic.
        let mut action = report_action(hbm(), &[1, 2], &[COMPONENT_FULL]);
        action.component_masks = vec![COMPONENT_FULL]; // misaligned
        let request = apply("worker-0", "", vec![action], None);
        assert_eq!(
            batch_is_kvcm_eligible(&request, &[TierType::TierHbm])
                .unwrap_err()
                .code(),
            tonic::Code::InvalidArgument
        );
        let mut action = report_action(hbm(), &[1], &[COMPONENT_FULL]);
        action.r#type = 99;
        let request = apply("worker-0", "", vec![action], None);
        assert!(batch_is_kvcm_eligible(&request, &[TierType::TierHbm]).is_err());
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
            kvcm_eligible: eligible,
        };
        assert_eq!(
            prefix_entry(host("w0", 3), 10, &record("http://w0", true)),
            Some(("w0".into(), "http://w0".into(), 3))
        );
        // A KVCM over-report cannot exceed the query length.
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
    fn check_header_maps_kvcm_codes() {
        let header = |code: ErrorCode| CommonResponseHeader {
            status: Some(crate::kvcm_pb::Status {
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
        // A reporter KVCM forgot is an expected, self-healing event: the
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
        let config = KvcmBackendConfig::new("http://kvcm:1", "inst");
        assert_eq!(config.storage_type, StorageType::StEventReportL1p5);
        assert_eq!(config.report_tiers, [TierType::TierHbm]);
        assert_eq!(config.auto_register, None);
        assert_eq!(config.report_mediums(), ["hbm"]);
    }

    /// The environment variables are the operator's entry point, and the one
    /// place a mistake is silent: a renamed or unenforced variable means the
    /// backend registers without the model deployment KVCM demands, and every
    /// report then fails `INVALID_ARGUMENT` (the bridge exits). This is the
    /// only test in the crate that touches `KV_INDEXER_KVCM_*`, and no other
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
        let config = KvcmBackendConfig::from_env().expect("endpoint and instance are enough");
        assert_eq!(config.endpoint, "http://127.0.0.1:50052");
        assert_eq!(config.instance_id, "inst-0");
        assert_eq!(config.auto_register, None);

        std::env::set_var(ENV_AUTO_REGISTER, "1");
        assert_eq!(
            KvcmBackendConfig::from_env().unwrap_err(),
            KvcmConfigError::Missing(ENV_INSTANCE_GROUP)
        );
        std::env::set_var(ENV_INSTANCE_GROUP, "group-a");
        assert!(matches!(
            KvcmBackendConfig::from_env().unwrap_err(),
            KvcmConfigError::Invalid {
                key: ENV_BLOCK_SIZE,
                ..
            }
        ));
        std::env::set_var(ENV_BLOCK_SIZE, "0");
        assert!(matches!(
            KvcmBackendConfig::from_env().unwrap_err(),
            KvcmConfigError::Invalid {
                key: ENV_BLOCK_SIZE,
                ..
            }
        ));
        std::env::set_var(ENV_BLOCK_SIZE, "128");
        // KVCM rejects a registration without a model name, a dtype, or a
        // positive block size, so each one has to be a hard requirement.
        assert_eq!(
            KvcmBackendConfig::from_env().unwrap_err(),
            KvcmConfigError::Missing(ENV_MODEL_NAME)
        );
        std::env::set_var(ENV_MODEL_NAME, "qwen3-8b");
        assert_eq!(
            KvcmBackendConfig::from_env().unwrap_err(),
            KvcmConfigError::Missing(ENV_MODEL_DTYPE)
        );
        std::env::set_var(ENV_MODEL_DTYPE, "FP8");

        let config = KvcmBackendConfig::from_env().expect("complete environment");
        assert_eq!(
            config.auto_register,
            Some(KvcmAutoRegister {
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
