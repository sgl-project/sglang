// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Live integration tests for the Tair KVCache backend against a real Tair
//! KVCache Manager (KVCM). `tests/test_tair_kvcache_live/setup_tair_kvcache_manager.py`
//! builds and provisions one, the env file it writes points the backend at it,
//! and `--ignored` runs this suite; the script's docstring carries the commands.
//!
//! The tests are `#[ignore]`d, so a plain `cargo test` never touches the
//! network, and a run that explicitly asks for them without the harness fails
//! rather than skipping: `N passed` then always means the manager was driven.
//! Unlike `tair_kvcache_contract.rs` (our translation against a mock), these
//! pin the *manager's* behavior -- accepted events, liveness reclamation,
//! `GetHostCacheState` answers -- and use the vendored client directly where
//! our own mapping would hide that answer. The harness provisions deliberately
//! tight liveness (2s timeout, 2s grace), so every backend here sets an
//! explicit heartbeat cadence: a silent reporter's placements are *deleted*,
//! not just hidden.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sgl_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ExternalKvAction, ExternalKvActionType,
    MatchExternalKvPrefixRequest, MatchExternalKvPrefixResponse, TierType,
};
use sgl_kv_indexer::tair_kvcache_pb::event_item::EventParams;
use sgl_kv_indexer::tair_kvcache_pb::meta_service_client::MetaServiceClient;
use sgl_kv_indexer::tair_kvcache_pb::{
    block_mask, event_item, BlockAddEventParams, BlockDeleteEventParams, BlockMask,
    BlockSnapshotEventParams, CacheLocation, CommonResponse, CommonResponseHeader, ErrorCode,
    EventItem, FinishWriteCacheRequest, GetCacheMetaRequest, GetCacheMetaResponse,
    GetHostCacheStateRequest, GetHostCacheStateResponse, HeartbeatEventParams, LocationSpec,
    LocationSpecInfo, ModelDeployment, NodeRegisterEventParams, QueryType, RegisterInstanceRequest,
    RegisterInstanceResponse, ReportEventRequest, ReportEventResponse, ReportEventType,
    StartWriteCacheRequest, StartWriteCacheResponse, StorageType,
};
use sgl_kv_indexer::{
    KvIndexerBackend, TairKVCacheAutoRegister, TairKVCacheBackendConfig, TairKVCacheIndexerBackend,
};
use tonic::transport::Channel;

const ENV_ENDPOINT: &str = "TAIR_KVCACHE_LIVE_ENDPOINT";
const ENV_INSTANCE_ID: &str = "TAIR_KVCACHE_LIVE_INSTANCE_ID";
const ENV_INSTANCE_GROUP: &str = "TAIR_KVCACHE_LIVE_INSTANCE_GROUP";
const ENV_MODEL_NAME: &str = "TAIR_KVCACHE_LIVE_MODEL_NAME";
const ENV_MODEL_DTYPE: &str = "TAIR_KVCACHE_LIVE_MODEL_DTYPE";
/// Written by the harness, so a failure names the workdir actually used.
const ENV_WORKDIR: &str = "TAIR_KVCACHE_LIVE_WORKDIR";
const DEFAULT_LIVE_WORKDIR: &str = "/tmp/tair_kvcache-live";

/// The harness provisions a 2s heartbeat timeout: 300ms stays well inside it.
const HEARTBEAT: Duration = Duration::from_millis(300);
/// Generous per-RPC budgets: these tests assert Tair KVCache's answers, not scheduling.
const RPC_TIMEOUT: Duration = Duration::from_secs(5);

/// The harness configuration, or `None` when the live tests are not requested.
struct LiveEnv {
    endpoint: String,
    /// Instance registered by `setup_tair_kvcache_manager.py` with the `full`/`swa`/`mamba`
    /// spec names; tests that exercise the no-auto-register path report here.
    instance_id: String,
    instance_group: String,
    /// The model deployment that instance was registered with: Tair KVCache refuses a
    /// registration without one, and a re-registration whose fields differ.
    model_name: String,
    dtype: String,
}

fn nonempty_env(key: &str) -> Option<String> {
    let value = std::env::var(key).ok()?.trim().to_string();
    (!value.is_empty()).then_some(value)
}

/// Whether this run explicitly asked for ignored tests (`--ignored` or
/// `--include-ignored`). A requested-but-skipped suite would report the same
/// `ok. N passed` as one that verified something, so an explicit request with
/// no harness fails; only an implicit run skips silently.
fn live_suite_requested() -> bool {
    std::env::args()
        .skip(1)
        .any(|arg| arg == "--ignored" || arg == "--include-ignored")
}

fn live_env() -> Option<LiveEnv> {
    let Some(endpoint) = nonempty_env(ENV_ENDPOINT) else {
        assert!(
            !live_suite_requested(),
            "--ignored was requested but {ENV_ENDPOINT} is not set, so the live Tair KVCache suite would verify nothing. \
             Start the manager and source its environment first:\n  \
             python3 sgl-kv-indexer/tests/test_tair_kvcache_live/setup_tair_kvcache_manager.py start --workdir /tmp/tair_kvcache-live\n  \
             source /tmp/tair_kvcache-live/tair_kvcache_live.env"
        );
        return None;
    };
    let required = |key: &str| {
        nonempty_env(key).unwrap_or_else(|| {
            panic!("{ENV_ENDPOINT} is set but {key} is not; source the harness env file")
        })
    };
    let env = LiveEnv {
        endpoint,
        instance_id: required(ENV_INSTANCE_ID),
        instance_group: required(ENV_INSTANCE_GROUP),
        model_name: required(ENV_MODEL_NAME),
        dtype: required(ENV_MODEL_DTYPE),
    };
    // A manager that is gone must read as an environment failure with a next
    // step, not as bare connection errors; the printed commands name the
    // workdir this run actually used.
    if let Err(reason) = probe_endpoint(&env.endpoint) {
        let workdir =
            std::env::var(ENV_WORKDIR).unwrap_or_else(|_| DEFAULT_LIVE_WORKDIR.to_string());
        panic!(
            "{} is not reachable ({reason}); check the manager with \
             `python3 sgl-kv-indexer/tests/test_tair_kvcache_live/setup_tair_kvcache_manager.py status --workdir {workdir}` \
             and restart it with `start`",
            env.endpoint
        );
    }
    Some(env)
}

/// A TCP connect tells "running" from "gone" without an RPC or a runtime.
fn probe_endpoint(endpoint: &str) -> Result<(), String> {
    use std::net::ToSocketAddrs;

    let Some(host_port) = endpoint
        .strip_prefix("http://")
        .map(|rest| rest.trim_end_matches('/'))
    else {
        // Not a form this probe understands; the first RPC reports it instead.
        return Ok(());
    };
    let addr = host_port
        .to_socket_addrs()
        .map_err(|error| format!("{host_port} is not resolvable: {error}"))?
        .next()
        .ok_or_else(|| format!("{host_port} resolves to no address"))?;
    std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(2))
        .map(|_| ())
        .map_err(|error| format!("{addr}: {error}"))
}

/// Skips when the harness is absent and this run did not ask for the suite.
macro_rules! live_env {
    () => {
        match live_env() {
            Some(env) => env,
            None => {
                eprintln!("{ENV_ENDPOINT} is not set; skipping live Tair KVCache test");
                return;
            }
        }
    };
}

/// Instances are never deleted, so every test takes a fresh id.
fn unique(prefix: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    format!(
        "kvci-live-{prefix}-{}-{}",
        std::process::id(),
        counter + millis
    )
}

fn backend(
    env: &LiveEnv,
    instance_id: &str,
    heartbeat: Option<Duration>,
) -> TairKVCacheIndexerBackend {
    let mut config = TairKVCacheBackendConfig::new(env.endpoint.clone(), instance_id);
    config.query_timeout = RPC_TIMEOUT;
    config.apply_timeout = RPC_TIMEOUT;
    config.heartbeat_interval = heartbeat;
    TairKVCacheIndexerBackend::new(config).expect("the harness endpoint is a valid http endpoint")
}

fn auto_registering_backend(
    env: &LiveEnv,
    instance_id: &str,
    block_size: u32,
) -> TairKVCacheIndexerBackend {
    let mut config = TairKVCacheBackendConfig::new(env.endpoint.clone(), instance_id);
    config.query_timeout = RPC_TIMEOUT;
    config.apply_timeout = RPC_TIMEOUT;
    config.heartbeat_interval = Some(HEARTBEAT);
    config.auto_register = Some(TairKVCacheAutoRegister {
        instance_group: env.instance_group.clone(),
        block_size,
        model_name: env.model_name.clone(),
        dtype: env.dtype.clone(),
    });
    TairKVCacheIndexerBackend::new(config).expect("the harness endpoint is a valid http endpoint")
}

/// Sets process environment variables for one test and restores them on drop.
struct EnvGuard(Vec<(String, Option<String>)>);

impl EnvGuard {
    fn set(pairs: &[(&str, &str)]) -> Self {
        let saved = pairs
            .iter()
            .map(|(key, value)| {
                let previous = std::env::var(key).ok();
                std::env::set_var(key, value);
                (key.to_string(), previous)
            })
            .collect();
        Self(saved)
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, previous) in &self.0 {
            match previous {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
}

/// A legacy (whole-block, mask-less) worker reporting `hashes` on HBM.
fn report_batch(
    worker: &str,
    address: &str,
    seq: u64,
    hashes: &[i64],
) -> ApplyExternalKvBatchRequest {
    ApplyExternalKvBatchRequest {
        worker_id: worker.to_string(),
        seq,
        actions: vec![ExternalKvAction {
            r#type: ExternalKvActionType::ActionReport as i32,
            tier: TierType::TierHbm as i32,
            hashes: hashes.to_vec(),
            component_masks: Vec::new(),
            block_sizes: Vec::new(),
            parent_block_hash: None,
        }],
        worker_address: address.to_string(),
        cache_spec: None,
    }
}

fn revoke_batch(
    worker: &str,
    address: &str,
    seq: u64,
    hashes: &[i64],
) -> ApplyExternalKvBatchRequest {
    let mut batch = report_batch(worker, address, seq, hashes);
    batch.actions[0].r#type = ExternalKvActionType::ActionRevoke as i32;
    batch
}

/// The wire form of a clear: an empty complete snapshot, alone in its request.
fn empty_snapshot_event() -> EventItem {
    EventItem {
        event_type: ReportEventType::EventBlockSnapshot as i32,
        event_params: Some(event_item::EventParams::BlockSnapshot(
            BlockSnapshotEventParams {
                blocks: Vec::new(),
                ..Default::default()
            },
        )),
    }
}

fn clear_action() -> ExternalKvAction {
    ExternalKvAction {
        r#type: ExternalKvActionType::ActionClearAllAtTier as i32,
        tier: TierType::TierHbm as i32,
        hashes: Vec::new(),
        component_masks: Vec::new(),
        block_sizes: Vec::new(),
        parent_block_hash: None,
    }
}

fn clear_batch(worker: &str, address: &str, seq: u64) -> ApplyExternalKvBatchRequest {
    ApplyExternalKvBatchRequest {
        worker_id: worker.to_string(),
        seq,
        actions: vec![clear_action()],
        worker_address: address.to_string(),
        cache_spec: None,
    }
}

async fn prefix(
    backend: &TairKVCacheIndexerBackend,
    hashes: &[i64],
    max_blocks: u32,
) -> MatchExternalKvPrefixResponse {
    backend
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: hashes.to_vec(),
            max_blocks,
        })
        .await
        .expect("prefix query succeeds")
}

/// `address → matched prefix` for the answer, so assertions read as sets.
fn prefixes_by_address(response: &MatchExternalKvPrefixResponse) -> HashMap<String, u32> {
    response
        .matches
        .iter()
        .map(|entry| (entry.worker_address.clone(), entry.matched_prefix_blocks))
        .collect()
}

// --- Raw MetaService helpers ------------------------------------------------

/// Talks to the manager directly, for what our backend hides or maps away.
struct RawKvcm {
    client: MetaServiceClient<Channel>,
    instance_id: String,
    trace: AtomicU64,
}

fn header_code(header: Option<&CommonResponseHeader>) -> ErrorCode {
    header
        .and_then(|header| header.status.as_ref())
        .and_then(|status| ErrorCode::try_from(status.code).ok())
        .unwrap_or(ErrorCode::UnknownError)
}

impl RawKvcm {
    async fn connect(
        env: &LiveEnv,
        instance_id: &str,
        spec_names: &[&str],
        block_size: i32,
    ) -> Self {
        let client = MetaServiceClient::connect(env.endpoint.clone())
            .await
            .expect("Tair KVCache gRPC endpoint is reachable");
        let mut raw = Self {
            client,
            instance_id: instance_id.to_string(),
            trace: AtomicU64::new(0),
        };
        let request = RegisterInstanceRequest {
            trace_id: raw.next_trace("register"),
            instance_group: env.instance_group.clone(),
            instance_id: instance_id.to_string(),
            block_size,
            location_spec_infos: spec_names
                .iter()
                .map(|name| LocationSpecInfo {
                    name: name.to_string(),
                    size: 0,
                })
                .collect(),
            model_deployment: Some(ModelDeployment {
                model_name: env.model_name.clone(),
                dtype: env.dtype.clone(),
                ..Default::default()
            }),
            location_spec_groups: Vec::new(),
            default_query_type: QueryType::QtPrefixMatch as i32,
        };
        let response: RegisterInstanceResponse = raw
            .client
            .register_instance(request)
            .await
            .expect("RegisterInstance RPC completes")
            .into_inner();
        assert_eq!(
            header_code(response.header.as_ref()),
            ErrorCode::Ok,
            "RegisterInstance must succeed for a fresh instance"
        );
        raw
    }

    fn next_trace(&self, what: &str) -> String {
        format!(
            "kvci-live-{}-{}",
            what,
            self.trace.fetch_add(1, Ordering::Relaxed)
        )
    }

    async fn report(
        &mut self,
        host: &str,
        storage: StorageType,
        events: Vec<EventItem>,
    ) -> ReportEventResponse {
        let request = ReportEventRequest {
            trace_id: self.next_trace("apply"),
            instance_id: self.instance_id.clone(),
            host_ip_port: host.to_string(),
            events,
            storage_type: storage as i32,
        };
        self.client
            .report_event(request)
            .await
            .expect("ReportEvent RPC completes")
            .into_inner()
    }

    /// A data-plane write session: the L3 (Tair KVCache pool) side of the user's model.
    /// `write_timeout_seconds` must be positive or the manager drops the session
    /// before `finish_write_cache` can use it.
    async fn start_write_cache(
        &mut self,
        keys: &[i64],
        write_timeout_seconds: i32,
    ) -> StartWriteCacheResponse {
        let request = StartWriteCacheRequest {
            trace_id: self.next_trace("start-write"),
            instance_id: self.instance_id.clone(),
            block_keys: keys.to_vec(),
            token_ids: Vec::new(),
            location_spec_group_names: Vec::new(),
            write_timeout_seconds,
            min_replica_count: 0,
        };
        let response = self
            .client
            .start_write_cache(request)
            .await
            .expect("StartWriteCache RPC completes")
            .into_inner();
        assert_eq!(
            header_code(response.header.as_ref()),
            ErrorCode::Ok,
            "StartWriteCache must succeed"
        );
        response
    }

    /// Completes a write session. `success_blocks` marks the blocks that really
    /// landed, as a prefix of the requested keys (`BlockMask::offset` covers
    /// `[0, offset)`); a block outside the mask is deleted, not served.
    async fn finish_write_cache(
        &mut self,
        session: &str,
        success_blocks: i32,
        locations: Vec<CacheLocation>,
    ) -> CommonResponse {
        let request = FinishWriteCacheRequest {
            trace_id: self.next_trace("finish-write"),
            instance_id: self.instance_id.clone(),
            write_session_id: session.to_string(),
            success_blocks: Some(BlockMask {
                info: Some(block_mask::Info::Offset(success_blocks)),
            }),
            locations,
        };
        self.client
            .finish_write_cache(request)
            .await
            .expect("FinishWriteCache RPC completes")
            .into_inner()
    }

    /// The serving locations per key: the only way to tell "Tair KVCache holds this
    /// data-plane block" from "the write was dropped".
    async fn cache_meta(&mut self, keys: &[i64]) -> GetCacheMetaResponse {
        let request = GetCacheMetaRequest {
            trace_id: self.next_trace("cache-meta"),
            instance_id: self.instance_id.clone(),
            block_keys: keys.to_vec(),
            token_ids: Vec::new(),
            block_mask: None,
            detail_level: 0,
        };
        let response = self
            .client
            .get_cache_meta(request)
            .await
            .expect("GetCacheMeta RPC completes")
            .into_inner();
        assert_eq!(
            header_code(response.header.as_ref()),
            ErrorCode::Ok,
            "GetCacheMeta must succeed"
        );
        response
    }

    async fn query(
        &mut self,
        hashes: &[i64],
        medium: &[&str],
        p2p_host_count: i32,
    ) -> GetHostCacheStateResponse {
        let request = GetHostCacheStateRequest {
            trace_id: self.next_trace("query"),
            instance_id: self.instance_id.clone(),
            query_type: QueryType::QtPrefixMatch as i32,
            block_cache_keys: hashes.to_vec(),
            medium: medium.iter().map(|m| m.to_string()).collect(),
            p2p_host_count,
        };
        let response = self
            .client
            .get_host_cache_state(request)
            .await
            .expect("GetHostCacheState RPC completes")
            .into_inner();
        assert_eq!(
            header_code(response.header.as_ref()),
            ErrorCode::Ok,
            "GetHostCacheState must succeed"
        );
        response
    }
}

/// The reporter's URI convention the manager requires to be structurally valid.
fn spec_uri(host: &str, medium: &str) -> String {
    format!("event_report://{host}/{medium}")
}

fn node_register(mediums: &[&str]) -> EventItem {
    EventItem {
        event_type: ReportEventType::EventNodeRegister as i32,
        event_params: Some(EventParams::NodeRegister(NodeRegisterEventParams {
            mediums: mediums.iter().map(|m| m.to_string()).collect(),
        })),
    }
}

fn block_add(block_key: i64, medium: &str, host: &str, spec_names: &[&str]) -> EventItem {
    EventItem {
        event_type: ReportEventType::EventBlockAdd as i32,
        event_params: Some(EventParams::BlockAdd(BlockAddEventParams {
            block_key: block_key.to_string(),
            uri: String::new(),
            medium: medium.to_string(),
            specs: spec_names
                .iter()
                .map(|name| LocationSpec {
                    name: name.to_string(),
                    uri: spec_uri(host, medium),
                })
                .collect(),
        })),
    }
}

fn block_delete(block_key: i64, medium: &str, spec_names: &[&str]) -> EventItem {
    EventItem {
        event_type: ReportEventType::EventBlockDelete as i32,
        event_params: Some(EventParams::BlockDelete(BlockDeleteEventParams {
            block_key: block_key.to_string(),
            medium: medium.to_string(),
            spec_names: spec_names.iter().map(|name| name.to_string()).collect(),
        })),
    }
}

fn heartbeat() -> EventItem {
    EventItem {
        event_type: ReportEventType::EventHeartbeat as i32,
        event_params: Some(EventParams::Heartbeat(HeartbeatEventParams::default())),
    }
}

fn assert_ok(response: &ReportEventResponse, what: &str) {
    assert_eq!(
        header_code(response.header.as_ref()),
        ErrorCode::Ok,
        "{what}: item_results={:?} extra_info={}",
        response.item_results,
        response.extra_info
    );
}

fn locals(response: &GetHostCacheStateResponse) -> HashMap<String, i64> {
    response
        .hosts
        .iter()
        .map(|host| (host.host_ip_port.clone(), host.local))
        .collect()
}

// --- Ingestion: what Tair KVCache accepts ------------------------------------------

/// The operator-facing entry point, end to end: `from_env` assembles an
/// auto-registering backend a real manager accepts. The other tests build the
/// config programmatically, so this is what catches a renamed or unenforced
/// variable, whose failure mode is a rejected registration (the bridge exits).
#[tokio::test]
#[ignore]
async fn environment_configuration_registers_and_reports() {
    let env = live_env!();
    let instance = unique("env");
    // The variable names are spelled out here rather than imported: a typo in
    // the constant is exactly what this test exists to catch.
    let _guard = EnvGuard::set(&[
        (
            "KV_INDEXER_TAIR_KVCACHE_KVCM_ENDPOINT",
            env.endpoint.as_str(),
        ),
        (
            "KV_INDEXER_TAIR_KVCACHE_KVCM_INSTANCE_ID",
            instance.as_str(),
        ),
        ("KV_INDEXER_TAIR_KVCACHE_KVCM_AUTO_REGISTER", "1"),
        (
            "KV_INDEXER_TAIR_KVCACHE_KVCM_INSTANCE_GROUP",
            env.instance_group.as_str(),
        ),
        // A positive block size and the model deployment are mandatory for
        // Tair KVCache's RegisterInstance.
        ("KV_INDEXER_TAIR_KVCACHE_KVCM_BLOCK_SIZE", "128"),
        (
            "KV_INDEXER_TAIR_KVCACHE_KVCM_MODEL_NAME",
            env.model_name.as_str(),
        ),
        (
            "KV_INDEXER_TAIR_KVCACHE_KVCM_MODEL_DTYPE",
            env.dtype.as_str(),
        ),
        // The harness provisions 2s liveness, so the configured cadence matters.
        ("KV_INDEXER_TAIR_KVCACHE_KVCM_HEARTBEAT_INTERVAL_MS", "300"),
    ]);
    let backend = TairKVCacheIndexerBackend::new(
        TairKVCacheBackendConfig::from_env().expect("the environment assembles a config"),
    )
    .expect("the harness endpoint is a valid http endpoint");
    let worker = format!("{instance}-w0");
    let address = "http://127.0.0.1:30014";

    backend
        .apply_external_kv_batch(report_batch(&worker, address, 1, &[181, 182]))
        .await
        .expect("an environment-configured backend registers and reports");
    assert_eq!(
        prefixes_by_address(&prefix(&backend, &[181, 182], 0).await),
        HashMap::from([(address.to_string(), 2)])
    );
}

/// Auto-registration end to end: `RegisterInstance`, then reports landing.
#[tokio::test]
#[ignore]
async fn auto_registration_reports_and_answers_prefixes() {
    let env = live_env!();
    let instance = unique("auto");
    let backend = auto_registering_backend(&env, &instance, 128);
    let w0 = format!("{instance}-w0");
    let w1 = format!("{instance}-w1");

    backend
        .apply_external_kv_batch(report_batch(
            &w0,
            "http://127.0.0.1:30001",
            1,
            &[11, 12, 13],
        ))
        .await
        .expect("auto-registered instance accepts a first report");
    backend
        .apply_external_kv_batch(report_batch(&w1, "http://127.0.0.1:30002", 1, &[11]))
        .await
        .expect("a second reporter joins the same instance");

    let answer = prefix(&backend, &[11, 12, 13], 0).await;
    assert_eq!(
        prefixes_by_address(&answer),
        HashMap::from([
            ("http://127.0.0.1:30001".to_string(), 3),
            ("http://127.0.0.1:30002".to_string(), 1),
        ]),
        "Tair KVCache answers the local prefix per reporter"
    );
    assert_eq!(answer.best_prefix_blocks, 3);
    assert_eq!(answer.blocks_read, 3);
    // The indexer's own contract: longest prefix first.
    assert_eq!(
        answer
            .matches
            .iter()
            .map(|entry| entry.matched_prefix_blocks)
            .collect::<Vec<_>>(),
        [3, 1]
    );

    // Tair KVCache's `local` is bounded by the query, so a longer report cannot
    // over-report; the backend's clamp stays a safety net, not a fixup.
    let answer = prefix(&backend, &[11, 12], 0).await;
    assert_eq!(
        prefixes_by_address(&answer),
        HashMap::from([
            ("http://127.0.0.1:30001".to_string(), 2),
            ("http://127.0.0.1:30002".to_string(), 1),
        ])
    );
    assert_eq!(answer.blocks_read, 2);
}

/// The Router's first request is a prefix query on an instance no worker
/// has reported to yet: a permanent fault for the query path, not transient.
#[tokio::test]
#[ignore]
async fn a_query_before_any_report_registers_the_instance() {
    let env = live_env!();
    let instance = unique("earlyquery");
    let keys = [9101, 9102, 9103];

    // No apply has run, so Tair KVCache has never heard of this instance.
    let registrar = auto_registering_backend(&env, &instance, 128);
    let answer = prefix(&registrar, &keys, 0).await;
    assert!(
        answer.matches.is_empty(),
        "a fresh instance holds nothing: {answer:?}"
    );
    assert_eq!(answer.blocks_read, 3);

    // The registration reached Tair KVCache rather than only flipping our own flag: a
    // second backend, which must not register anything, can query the same
    // instance. Under the old order this is where INSTANCE_NOT_EXIST appeared.
    let observer = backend(&env, &instance, None);
    let answer = prefix(&observer, &keys, 0).await;
    assert!(answer.matches.is_empty());

    // A deployment that configured no auto-registration keeps the fault: a
    // typo'd instance id is a configuration error, not a transient condition.
    let stranger = backend(&env, &unique("nodice"), None);
    let error = stranger
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: keys.to_vec(),
            max_blocks: 0,
        })
        .await
        .expect_err("an unknown instance is a permanent fault");
    assert_eq!(error.code(), tonic::Code::FailedPrecondition);

    // And the instance the query registered is usable, not just present.
    let address = "http://127.0.0.1:31031";
    registrar
        .apply_external_kv_batch(report_batch(&format!("{instance}-w0"), address, 1, &keys))
        .await
        .expect("the registered instance accepts a report");
    let answer = prefix(&registrar, &keys, 0).await;
    assert_eq!(
        prefixes_by_address(&answer),
        HashMap::from([(address.to_string(), 3)])
    );
}

/// `max_blocks` truncates the ask before it reaches Tair KVCache.
#[tokio::test]
#[ignore]
async fn max_blocks_truncates_the_query() {
    let env = live_env!();
    let instance = unique("truncate");
    let backend = auto_registering_backend(&env, &instance, 128);
    let worker = format!("{instance}-w0");

    backend
        .apply_external_kv_batch(report_batch(
            &worker,
            "http://127.0.0.1:30003",
            1,
            &[21, 22, 23],
        ))
        .await
        .expect("report succeeds");

    let answer = prefix(&backend, &[21, 22, 23], 2).await;
    assert_eq!(answer.blocks_read, 2);
    assert_eq!(
        prefixes_by_address(&answer),
        HashMap::from([("http://127.0.0.1:30003".to_string(), 2)])
    );
}

/// The deployment-registered path: no `AUTO_REGISTER`, the instance exists.
#[tokio::test]
#[ignore]
async fn reports_into_an_instance_registered_by_the_deployment() {
    let env = live_env!();
    let backend = backend(&env, &env.instance_id, Some(HEARTBEAT));
    let worker = unique("external");

    backend
        .apply_external_kv_batch(report_batch(
            &worker,
            "http://127.0.0.1:30004",
            1,
            &[31, 32],
        ))
        .await
        .expect("a pre-registered instance accepts our spec names without auto-registration");

    let answer = prefix(&backend, &[31, 32], 0).await;
    assert_eq!(
        prefixes_by_address(&answer),
        HashMap::from([("http://127.0.0.1:30004".to_string(), 2)])
    );
}

/// Identical re-registration is a no-op; a field mismatch is permanent.
#[tokio::test]
#[ignore]
async fn re_registration_is_idempotent_and_mismatches_are_permanent() {
    let env = live_env!();
    let identical = auto_registering_backend(&env, &env.instance_id, 128);
    identical
        .apply_external_kv_batch(report_batch(
            &unique("idempotent"),
            "http://127.0.0.1:30005",
            1,
            &[41],
        ))
        .await
        .expect("identical re-registration is idempotent");

    let mismatched = auto_registering_backend(&env, &env.instance_id, 256);
    let error = mismatched
        .apply_external_kv_batch(report_batch(
            &unique("mismatch"),
            "http://127.0.0.1:30006",
            1,
            &[42],
        ))
        .await
        .expect_err("a field mismatch is a permanent configuration fault");
    assert_eq!(error.code(), tonic::Code::FailedPrecondition);
    assert!(
        error.message().contains("already registered"),
        "unexpected message: {}",
        error.message()
    );
}

/// `BLOCK_ADD` must name declared specs; an undeclared one fails the report.
#[tokio::test]
#[ignore]
async fn reports_into_an_instance_without_our_spec_names_fail() {
    let env = live_env!();
    let instance = unique("specs");
    let mut raw = RawKvcm::connect(&env, &instance, &["tp0"], 128).await;
    let backend = backend(&env, &instance, Some(HEARTBEAT));

    let error = backend
        .apply_external_kv_batch(report_batch(
            &format!("{instance}-w0"),
            "http://127.0.0.1:30007",
            1,
            &[51],
        ))
        .await
        .expect_err("an instance without the `full` spec name rejects BLOCK_ADD");
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(
        error
            .message()
            .contains("Tair KVCache rejected ReportEvent"),
        "unexpected message: {}",
        error.message()
    );

    // The rejection names no reporter, so nothing was indexed.
    let answer = raw.query(&[51], &["hbm"], 0).await;
    assert!(answer.hosts.is_empty(), "rejected blocks must not index");
}

/// An unknown or forgotten reporter is a permanent query-path fault.
#[tokio::test]
#[ignore]
async fn unknown_instance_is_a_permanent_fault() {
    let env = live_env!();
    let backend = backend(&env, &unique("missing"), None);
    let worker = unique("unknown");

    let error = backend
        .apply_external_kv_batch(report_batch(&worker, "http://127.0.0.1:30008", 1, &[61]))
        .await
        .expect_err("ReportEvent for an unknown instance fails");
    assert_eq!(error.code(), tonic::Code::FailedPrecondition, "{error}");

    let error = backend
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: vec![61],
            max_blocks: 0,
        })
        .await
        .expect_err("GetHostCacheState for an unknown instance fails");
    assert_eq!(error.code(), tonic::Code::FailedPrecondition, "{error}");
}

/// `BLOCK_DELETE` must name registered specs; our REVOKE sends them all.
#[tokio::test]
#[ignore]
async fn block_delete_requires_registered_spec_names() {
    let env = live_env!();
    let instance = unique("delete");
    let mut raw = RawKvcm::connect(&env, &instance, &["full", "swa", "mamba"], 128).await;
    let host = "10.5.0.1:8080";

    assert_ok(
        &raw.report(
            host,
            StorageType::StEventReportL1p5,
            vec![
                node_register(&["hbm"]),
                block_add(161, "hbm", host, &["full"]),
                block_add(162, "hbm", host, &["full"]),
            ],
        )
        .await,
        "initial report",
    );

    let rejected = raw
        .report(
            host,
            StorageType::StEventReportL1p5,
            vec![block_delete(161, "hbm", &["not_a_spec"])],
        )
        .await;
    assert_eq!(
        header_code(rejected.header.as_ref()),
        ErrorCode::InvalidArgument,
        "an unregistered spec name is rejected outright"
    );
    assert_eq!(
        locals(&raw.query(&[161, 162], &["hbm"], 0).await),
        HashMap::from([(host.to_string(), 2)]),
        "a rejected delete must not remove anything"
    );

    // The backend's superset removal: this reporter only ever stored `full`,
    // and naming the other components too is a no-op for them.
    assert_ok(
        &raw.report(
            host,
            StorageType::StEventReportL1p5,
            vec![block_delete(161, "hbm", &["full", "swa", "mamba"])],
        )
        .await,
        "superset delete",
    );
    assert_eq!(
        locals(&raw.query(&[162], &["hbm"], 0).await),
        HashMap::from([(host.to_string(), 1)])
    );
    assert!(
        raw.query(&[161], &["hbm"], 0).await.hosts.is_empty(),
        "the deleted block is gone"
    );
}

/// A repeated `NODE_REGISTER` is idempotent, and earlier placements stay.
#[tokio::test]
#[ignore]
async fn node_register_resend_is_idempotent() {
    let env = live_env!();
    let instance = unique("register");
    let mut raw = RawKvcm::connect(&env, &instance, &["full", "swa", "mamba"], 128).await;
    let host = "10.1.2.3:8080";

    let first = raw
        .report(
            host,
            StorageType::StEventReportL1p5,
            vec![
                node_register(&["hbm"]),
                block_add(71, "hbm", host, &["full"]),
                block_add(72, "hbm", host, &["full"]),
            ],
        )
        .await;
    assert_ok(&first, "initial registration and report");
    assert!(first.item_results.is_empty(), "no per-item failures");

    let again = raw
        .report(
            host,
            StorageType::StEventReportL1p5,
            vec![
                node_register(&["hbm"]),
                block_add(73, "hbm", host, &["full"]),
            ],
        )
        .await;
    assert_ok(&again, "repeated NODE_REGISTER");

    let answer = raw.query(&[71, 72, 73], &["hbm"], 0).await;
    assert_eq!(
        locals(&answer),
        HashMap::from([(host.to_string(), 3)]),
        "re-registration must not invalidate earlier placements"
    );
}

/// REVOKE maps to `BLOCK_DELETE`, which removes exactly that block.
#[tokio::test]
#[ignore]
async fn revokes_remove_single_blocks() {
    let env = live_env!();
    let instance = unique("revoke");
    let backend = auto_registering_backend(&env, &instance, 128);
    let worker = format!("{instance}-w0");
    let address = "http://127.0.0.1:30009";

    backend
        .apply_external_kv_batch(report_batch(&worker, address, 1, &[81, 82, 83]))
        .await
        .expect("report succeeds");
    backend
        .apply_external_kv_batch(revoke_batch(&worker, address, 2, &[82]))
        .await
        .expect("revoke succeeds");

    let answer = prefix(&backend, &[81, 82, 83], 0).await;
    assert_eq!(
        prefixes_by_address(&answer),
        HashMap::from([(address.to_string(), 1)]),
        "the revoked block ends the prefix"
    );
}

// --- Liveness ---------------------------------------------------------------

/// Heartbeats keep an idle reporter routable; silence lets it be reclaimed.
#[tokio::test]
#[ignore]
async fn heartbeats_keep_an_idle_reporter_alive() {
    let env = live_env!();
    let instance = unique("alive");
    let backend = auto_registering_backend(&env, &instance, 128);
    let worker = format!("{instance}-w0");
    let address = "http://127.0.0.1:30010";

    backend
        .apply_external_kv_batch(report_batch(&worker, address, 1, &[91, 92, 93]))
        .await
        .expect("report succeeds");

    // Longer than the harness heartbeat timeout + cleanup grace (2s + 2s).
    tokio::time::sleep(Duration::from_secs(7)).await;

    let answer = prefix(&backend, &[91, 92, 93], 0).await;
    assert_eq!(
        prefixes_by_address(&answer),
        HashMap::from([(address.to_string(), 3)]),
        "a heartbeating reporter keeps its placements"
    );
}

/// A silent reporter is reclaimed -- placements deleted, not hidden -- and its
/// next report re-registers it. The rejection is a transient
/// `NodeNotRegistered`, which is what makes the self-healing work.
#[tokio::test]
#[ignore]
async fn liveness_reclaims_a_silent_reporter_and_the_retry_self_heals() {
    let env = live_env!();
    let instance = unique("reclaim");
    // The instance is registered directly so that no heartbeating backend ever
    // owns the reporter: silence is what this test needs.
    let _raw = RawKvcm::connect(&env, &instance, &["full", "swa", "mamba"], 128).await;
    let silent = backend(&env, &instance, None);
    let worker = format!("{instance}-w0");
    let address = "http://127.0.0.1:30011";

    silent
        .apply_external_kv_batch(report_batch(&worker, address, 1, &[101, 102, 103]))
        .await
        .expect("first report registers the reporter and its blocks");
    assert_eq!(
        prefix(&silent, &[101, 102, 103], 0)
            .await
            .best_prefix_blocks,
        3
    );

    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let answer = prefix(&silent, &[101, 102, 103], 0).await;
        if answer.matches.is_empty() {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "Tair KVCache never stopped answering the silent reporter: {:?}",
            answer.matches
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    // Becoming unroutable is immediate; the tombstone that rejects plain
    // reports appears once `cleanup_grace_ms` has passed. A revoke probe writes
    // nothing the sweep below could interleave with.
    let mut seq = 2;
    let error = loop {
        match silent
            .apply_external_kv_batch(revoke_batch(&worker, address, seq, &[104, 105, 106]))
            .await
        {
            Err(error) => break error,
            Ok(_) => {
                seq += 1;
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "Tair KVCache never unregistered the silent reporter"
                );
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
        }
    };
    assert_eq!(error.code(), tonic::Code::Unavailable, "{error}");
    assert!(
        error.message().contains("NodeNotRegistered"),
        "unexpected message: {}",
        error.message()
    );

    // The reporter resumes: the backend saw `NODE_NOT_REGISTERED` and prepends
    // the registration that clears the tombstone. Whether the manager keeps
    // *those* blocks is a separate race, handled below.
    silent
        .apply_external_kv_batch(report_batch(&worker, address, seq, &[104, 105, 106]))
        .await
        .expect("the retry re-registers the reporter and succeeds");

    // The sweep's generation guard only covers pre-lease writes, so a
    // re-report can still be swept: converge on the answer instead of
    // sleeping (this backend heartbeats, keeping liveness out of it).
    let resumed = backend(&env, &instance, Some(HEARTBEAT));
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        resumed
            .apply_external_kv_batch(report_batch(&worker, address, seq, &[104, 105, 106]))
            .await
            .expect("a re-report after the tombstone cleared is accepted");
        if prefix(&resumed, &[104, 105, 106], 0)
            .await
            .best_prefix_blocks
            == 3
        {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the reporter never became routable again after the tombstone cleared"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    // Reclaimed blocks stay reclaimed: only what was re-reported is indexed.
    assert!(prefix(&resumed, &[101, 102, 103], 0)
        .await
        .matches
        .is_empty());
}

// --- Query semantics --------------------------------------------------------

/// A reporter without a router-facing address is excluded from the answer.
#[tokio::test]
#[ignore]
async fn reporters_without_a_router_address_are_excluded() {
    let env = live_env!();
    let instance = unique("noaddr");
    let backend = auto_registering_backend(&env, &instance, 128);
    let worker = format!("{instance}-w0");

    // An empty address falls back to the worker id as the URI authority, so the
    // report is still valid; only the answer drops the reporter.
    backend
        .apply_external_kv_batch(report_batch(&worker, "", 1, &[111, 112]))
        .await
        .expect("a report without an address is still accepted by Tair KVCache");

    let answer = prefix(&backend, &[111, 112], 0).await;
    assert!(
        answer.matches.is_empty(),
        "an unroutable reporter must not be answered: {:?}",
        answer.matches
    );

    // Tair KVCache keeps the placement under the reporter id; the filter is ours, not
    // Tair KVCache's, so a router that learned the address later would see it again.
    let mut raw = RawKvcm::connect(&env, &instance, &["full", "swa", "mamba"], 128).await;
    assert_eq!(
        locals(&raw.query(&[111, 112], &["hbm"], 0).await),
        HashMap::from([(worker, 2)])
    );
}

/// The raw `p2p_host_count = 0` answer, i.e. the local fast path:
/// `p2p_1_fetch == 0`, `p2p_1_total_match == local`, and only reporters that
/// hold the first block appear at all.
#[tokio::test]
#[ignore]
async fn zero_p2p_host_count_answers_local_prefixes_only() {
    let env = live_env!();
    let instance = unique("p2pzero");
    let mut raw = RawKvcm::connect(&env, &instance, &["full", "swa", "mamba"], 128).await;
    let holder = "10.2.0.1:8080";
    let short = "10.2.0.2:8080";
    let late = "10.2.0.3:8080";

    for (host, keys) in [
        (holder, &[121i64, 122, 123][..]),
        (short, &[121][..]),
        (late, &[122][..]),
    ] {
        let mut events = vec![node_register(&["hbm"])];
        events.extend(
            keys.iter()
                .map(|key| block_add(*key, "hbm", host, &["full"])),
        );
        assert_ok(
            &raw.report(host, StorageType::StEventReportL1p5, events)
                .await,
            "report",
        );
    }

    let answer = raw.query(&[121, 122, 123], &["hbm"], 0).await;
    assert_eq!(
        locals(&answer),
        HashMap::from([(holder.to_string(), 3), (short.to_string(), 1),]),
        "only reporters holding the first block are candidates, with local>0"
    );
    for host in &answer.hosts {
        assert_eq!(host.p2p_1_fetch, 0, "p2p_host_count=0 computes no fetch");
        assert_eq!(
            host.p2p_1_total_match, host.local,
            "p2p_host_count=0 leaves the local prefix unchanged"
        );
    }
}

/// The per-worker answer reflects only what workers reported over ZMQ, not what
/// the data plane wrote: the manager attributes a data-plane block to the
/// *storage* that holds it (`nfs_01`), so an L3 hit can only be served by
/// fetching, never by routing to a worker.
#[tokio::test]
#[ignore]
async fn l3_locations_do_not_answer_per_worker_prefixes() {
    let env = live_env!();
    let instance = unique("l3");
    let mut raw = RawKvcm::connect(&env, &instance, &["full", "swa", "mamba"], 128).await;
    let worker = format!("{instance}-w0");
    let address = "http://127.0.0.1:30041";
    let l1_key = 9401;
    let l3_key = 9402;

    // L1 arm: the block arrives the way this backend's own ingestion does.
    let backend = backend(&env, &instance, None);
    backend
        .apply_external_kv_batch(report_batch(&worker, address, 1, &[l1_key]))
        .await
        .expect("a legacy whole-block report is accepted");

    // L3 arm: a data-plane write into the instance group's data-plane storage.
    // A positive write timeout is required, and the success mask covers the
    // keys that landed, so a single-key write marks `[0, 1)`.
    let started = raw.start_write_cache(&[l3_key], 60).await;
    assert_eq!(started.locations.len(), 1, "one location for one key");
    let finished = raw
        .finish_write_cache(&started.write_session_id, 1, started.locations)
        .await;
    assert_eq!(
        header_code(finished.header.as_ref()),
        ErrorCode::Ok,
        "FinishWriteCache must succeed"
    );

    // The write really landed: without this the "not in the answer" assertions
    // below would hold for the wrong reason (nothing to answer with).
    let meta = raw.cache_meta(&[l3_key]).await;
    assert!(
        meta.locations.iter().any(|location| location
            .location_specs
            .iter()
            .any(|spec| !spec.uri.is_empty())),
        "the L3 block must have serving locations: {:?}",
        meta.locations
    );
    assert!(
        meta.metas.iter().any(|meta| meta.contains("CLS_SERVING")),
        "the L3 block must be serving: {:?}",
        meta.metas
    );

    // Tair KVCache answers the L3 block as the storage that holds it, and only when the
    // medium filter does not exclude it. `nfs_01` is a storage, not a worker:
    // no routing address can come out of this arm.
    let unfiltered = raw.query(&[l3_key], &[], 0).await;
    assert_eq!(
        locals(&unfiltered),
        HashMap::from([("nfs_01".to_string(), 1)]),
        "the storage itself is the holder when nothing is filtered"
    );
    assert!(
        raw.query(&[l3_key], &["hbm"], 0).await.hosts.is_empty(),
        "the hbm filter the backend uses excludes the L3 location"
    );

    // Through the trait the Router calls: the reported block answers with the
    // worker's routing address, the pool block answers nothing.
    let answer = prefix(&backend, &[l1_key], 0).await;
    assert_eq!(
        prefixes_by_address(&answer),
        HashMap::from([(address.to_string(), 1)]),
        "the L1 fact is served"
    );
    let answer = prefix(&backend, &[l3_key], 0).await;
    assert!(
        answer.matches.is_empty(),
        "an L3-only block must not name any worker: {:?}",
        answer.matches
    );
}

/// The reported medium decides which placements a query sees.
#[tokio::test]
#[ignore]
async fn medium_filters_select_the_reported_medium() {
    let env = live_env!();
    let instance = unique("medium");
    let mut raw = RawKvcm::connect(&env, &instance, &["full", "swa", "mamba"], 128).await;
    let host = "10.3.0.1:8080";

    assert_ok(
        &raw.report(
            host,
            StorageType::StEventReportL1p5,
            vec![
                node_register(&["hbm"]),
                block_add(131, "hbm", host, &["full"]),
            ],
        )
        .await,
        "hbm report",
    );

    assert_eq!(
        locals(&raw.query(&[131], &["hbm"], 0).await),
        HashMap::from([(host.to_string(), 1)])
    );
    assert!(
        raw.query(&[131], &["mem"], 0).await.hosts.is_empty(),
        "a mem-filtered query must not see an hbm placement"
    );
    assert!(
        raw.query(&[131], &[], 0).await.hosts.len() == 1,
        "an empty medium filter considers every medium"
    );
}

/// `snapshot_required` marks a reporter lifecycle, not an error: the manager asks
/// for a first snapshot on the request that creates a reporter's generation,
/// and the flag is advisory -- the delta still lands.
#[tokio::test]
#[ignore]
async fn snapshot_required_marks_the_start_of_a_reporter_lifecycle() {
    let env = live_env!();
    let instance = unique("snapshot");
    let mut raw = RawKvcm::connect(&env, &instance, &["full", "swa", "mamba"], 128).await;
    let host = "10.4.0.1:8080";

    let registered = raw
        .report(
            host,
            StorageType::StEventReportL1p5,
            vec![node_register(&["hbm"])],
        )
        .await;
    assert_ok(&registered, "registration only");
    assert!(
        registered.snapshot_required,
        "a reporter with no committed version is asked for a snapshot"
    );
    assert!(registered.committed_snapshot_version.is_empty());

    let first_delta = raw
        .report(
            host,
            StorageType::StEventReportL1p5,
            vec![block_add(141, "hbm", host, &["full"])],
        )
        .await;
    assert_ok(&first_delta, "first delta");
    assert!(
        !first_delta.committed_snapshot_version.is_empty(),
        "a delta commits a snapshot version"
    );
    assert!(
        first_delta.snapshot_required,
        "the request that created the version still asks for a snapshot"
    );

    for (what, events) in [
        ("second delta", vec![block_add(142, "hbm", host, &["full"])]),
        ("heartbeat only", vec![heartbeat()]),
    ] {
        let response = raw
            .report(host, StorageType::StEventReportL1p5, events)
            .await;
        assert_ok(&response, what);
        assert!(
            !response.snapshot_required,
            "{what}: an established lifecycle is not asked for a snapshot"
        );
    }
}

/// The L2 channel accepts `vineyard://` spec URIs and answers the same way.
#[tokio::test]
#[ignore]
async fn l2_channel_accepts_vineyard_reports() {
    let env = live_env!();
    let instance = unique("l2");
    let mut config = TairKVCacheBackendConfig::new(env.endpoint.clone(), instance.clone());
    config.storage_type = StorageType::StEventReportL2;
    config.query_timeout = RPC_TIMEOUT;
    config.apply_timeout = RPC_TIMEOUT;
    config.heartbeat_interval = Some(HEARTBEAT);
    config.auto_register = Some(TairKVCacheAutoRegister {
        instance_group: env.instance_group.clone(),
        block_size: 128,
        model_name: env.model_name.clone(),
        dtype: env.dtype.clone(),
    });
    let backend = TairKVCacheIndexerBackend::new(config)
        .expect("the harness endpoint is a valid http endpoint");
    let worker = format!("{instance}-w0");
    let address = "http://127.0.0.1:30013";

    backend
        .apply_external_kv_batch(report_batch(&worker, address, 1, &[171, 172]))
        .await
        .expect("the L2 channel accepts vineyard:// spec URIs");
    assert_eq!(
        prefixes_by_address(&prefix(&backend, &[171, 172], 0).await),
        HashMap::from([(address.to_string(), 2)])
    );
}

/// `CLEAR_ALL_AT_TIER` reaches the manager as an empty `BLOCK_SNAPSHOT`: the
/// reporter owns no blocks afterwards, so the prefix answer goes with it. This
/// is the harness's flush gap closing (it used to leave 100% of placements).
#[tokio::test]
#[ignore]
async fn a_clear_removes_every_placement_of_the_reporter() {
    let env = live_env!();
    let instance = unique("clear");
    let backend = auto_registering_backend(&env, &instance, 128);
    let worker = format!("{instance}-w0");
    let address = "http://127.0.0.1:30012";

    backend
        .apply_external_kv_batch(report_batch(&worker, address, 1, &[151, 152, 153]))
        .await
        .expect("report succeeds");
    assert_eq!(
        prefixes_by_address(&prefix(&backend, &[151, 152, 153], 0).await),
        HashMap::from([(address.to_string(), 3)]),
        "the blocks are placed before the clear"
    );

    backend
        .apply_external_kv_batch(clear_batch(&worker, address, 2))
        .await
        .expect("an empty snapshot commits");

    let answer = prefix(&backend, &[151, 152, 153], 0).await;
    assert_eq!(
        prefixes_by_address(&answer),
        HashMap::new(),
        "the reporter owns nothing after the clear"
    );
}

/// A report after the clear lands on the snapshot's fresh generation: visible
/// immediately, while the pre-clear blocks stay gone. The manager stamps the new
/// token server-side; the client never carries one.
#[tokio::test]
#[ignore]
async fn a_report_after_a_clear_is_immediately_visible() {
    let env = live_env!();
    let instance = unique("clear-then-report");
    let backend = auto_registering_backend(&env, &instance, 128);
    let worker = format!("{instance}-w0");
    let address = "http://127.0.0.1:30012";

    backend
        .apply_external_kv_batch(report_batch(&worker, address, 1, &[161, 162]))
        .await
        .expect("report succeeds");
    backend
        .apply_external_kv_batch(clear_batch(&worker, address, 2))
        .await
        .expect("an empty snapshot commits");
    backend
        .apply_external_kv_batch(report_batch(&worker, address, 3, &[163]))
        .await
        .expect("the post-clear report succeeds");

    assert_eq!(
        prefixes_by_address(&prefix(&backend, &[161], 0).await),
        HashMap::new(),
        "blocks reported before the clear stay fenced off"
    );
    assert_eq!(
        prefixes_by_address(&prefix(&backend, &[163], 0).await),
        HashMap::from([(address.to_string(), 1)]),
        "the block reported after the clear is visible"
    );
}

/// A report-then-clear batch arriving after liveness reclaimed the reporter: the
/// registration has to ride the batch's first request. Left for the snapshot,
/// the manager would reject the unregistered deltas with the tombstone, the
/// batch would be lost (the bridge does not replay), and the registration would
/// never be reached.
#[tokio::test]
#[ignore]
async fn a_report_then_clear_batch_gets_through_a_reclaimed_reporter() {
    let env = live_env!();
    let instance = unique("forgotten-clear");
    // Registered directly so that no heartbeating backend owns the reporter:
    // silence is what this test needs (as in the liveness case).
    let _raw = RawKvcm::connect(&env, &instance, &["full", "swa", "mamba"], 128).await;
    let silent = backend(&env, &instance, None);
    let worker = format!("{instance}-w0");
    let address = "http://127.0.0.1:30014";

    silent
        .apply_external_kv_batch(report_batch(&worker, address, 1, &[201, 202]))
        .await
        .expect("the first report registers the reporter");

    // Wait for the tombstone, probing with a revoke so the wait writes nothing
    // the metadata sweep could interleave with.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    let mut seq = 2;
    loop {
        match silent
            .apply_external_kv_batch(revoke_batch(&worker, address, seq, &[999]))
            .await
        {
            Err(error) => {
                assert!(
                    error.message().contains("NodeNotRegistered"),
                    "unexpected error before the tombstone: {error}"
                );
                break;
            }
            Ok(_) => {
                seq += 1;
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "Tair KVCache never unregistered the silent reporter"
                );
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
        }
    }

    // The shape under test: deltas first, then the clear, in one batch.
    let mut mixed = report_batch(&worker, address, seq, &[203, 204]);
    mixed.actions.push(clear_action());
    silent
        .apply_external_kv_batch(mixed)
        .await
        .expect("the registration rides the first request, so the batch gets through");

    // The clear committed: this reporter is claimed to own nothing. (Whether
    // the two deltas survive the concurrent sweep is a race the metadata side
    // owns; the fence hides them either way, which is the point of the clear.)
    assert_eq!(
        prefixes_by_address(&prefix(&silent, &[203, 204], 0).await),
        HashMap::new(),
        "the clear takes effect even on the batch that re-registered the reporter"
    );
}

/// The manager rate-limits snapshots per (instance, channel, reporter), 30s by
/// default. A second flush inside that window is dropped with a warning rather
/// than failing the batch, which would cost the deltas riding with it.
#[tokio::test]
#[ignore]
async fn a_rate_limited_clear_degrades_to_a_warning() {
    let env = live_env!();
    let instance = unique("clear-rate-limited");
    let backend = auto_registering_backend(&env, &instance, 128);
    let worker = format!("{instance}-w0");
    let address = "http://127.0.0.1:30012";

    backend
        .apply_external_kv_batch(report_batch(&worker, address, 1, &[171, 172]))
        .await
        .expect("report succeeds");
    backend
        .apply_external_kv_batch(clear_batch(&worker, address, 2))
        .await
        .expect("the first clear commits");

    // Prove the cooldown is live, so the second clear really takes the
    // degradation path: a raw empty snapshot is answered SNAPSHOT_RATE_LIMITED
    // with the remaining cooldown (a failed snapshot does not extend it).
    let mut raw = RawKvcm::connect(&env, &instance, &["full", "swa", "mamba"], 128).await;
    let probe = raw
        .report(
            &worker,
            StorageType::StEventReportL1p5,
            vec![empty_snapshot_event()],
        )
        .await;
    assert_eq!(
        header_code(probe.header.as_ref()),
        ErrorCode::SnapshotRateLimited,
        "the snapshot cooldown is live"
    );
    assert!(
        probe.retry_after_ms > 0,
        "Tair KVCache reports the cooldown"
    );

    backend
        .apply_external_kv_batch(clear_batch(&worker, address, 3))
        .await
        .expect("a rate-limited flush is dropped, not failed");

    assert_eq!(
        prefixes_by_address(&prefix(&backend, &[171, 172], 0).await),
        HashMap::new(),
        "the first clear's fence still stands"
    );
}
