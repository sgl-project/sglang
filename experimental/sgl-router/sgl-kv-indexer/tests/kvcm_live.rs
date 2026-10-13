// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Live integration tests for the KVCM backend against a *real* Tair KVCache
//! Manager (KVCM) MetaService, driven through `tests/kvcm_live/setup_kvcm.py`.
//!
//!     # in the tair-kvcache checkout
//!     bazelisk build //kv_cache_manager:kv_cache_manager_bin
//!     # in experimental/sgl-router
//!     python3 sgl-kv-indexer/tests/kvcm_live/setup_kvcm.py start --workdir /tmp/kvcm-live
//!     source /tmp/kvcm-live/kvcm_live.env
//!     cargo test -p sgl-kv-indexer --test kvcm_live -- --ignored
//!     python3 sgl-kv-indexer/tests/kvcm_live/setup_kvcm.py stop --workdir /tmp/kvcm-live
//!
//! Two guards keep these tests out of every other run: they are `#[ignore]`d,
//! and each one returns early unless `KVCM_LIVE_ENDPOINT` is set. Without the
//! harness `cargo test -p sgl-kv-indexer` never touches the network.
//!
//! Unlike `tests/kvcm_contract.rs` (which pins our translation against a mock),
//! these tests pin *KVCM's* behavior: which events it accepts, what its liveness
//! does to a silent reporter, and what `GetHostCacheState` answers. Where a test
//! needs KVCM's raw answer rather than our mapping of it, it talks to the
//! MetaService directly with the vendored client.
//!
//! The harness provisions deliberately tight liveness (2s heartbeat timeout,
//! 2s cleanup grace), so every backend here sets an explicit heartbeat cadence
//! and the reclamation tests finish in seconds. A production deployment
//! provisioned the same way would need the same care: a silent reporter's
//! placements are *deleted*, not just hidden.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sgl_kv_indexer::kvcm_pb::event_item::EventParams;
use sgl_kv_indexer::kvcm_pb::meta_service_client::MetaServiceClient;
use sgl_kv_indexer::kvcm_pb::{
    BlockAddEventParams, BlockDeleteEventParams, CommonResponseHeader, ErrorCode, EventItem,
    GetHostCacheStateRequest, GetHostCacheStateResponse, HeartbeatEventParams, LocationSpec,
    LocationSpecInfo, ModelDeployment, NodeRegisterEventParams, QueryType, RegisterInstanceRequest,
    RegisterInstanceResponse, ReportEventRequest, ReportEventResponse, ReportEventType,
    StorageType,
};
use sgl_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ExternalKvAction, ExternalKvActionType,
    MatchExternalKvPrefixRequest, MatchExternalKvPrefixResponse, TierType,
};
use sgl_kv_indexer::{KvIndexerBackend, KvcmAutoRegister, KvcmBackendConfig, KvcmIndexerBackend};
use tonic::transport::Channel;

const ENV_ENDPOINT: &str = "KVCM_LIVE_ENDPOINT";
const ENV_INSTANCE_ID: &str = "KVCM_LIVE_INSTANCE_ID";
const ENV_INSTANCE_GROUP: &str = "KVCM_LIVE_INSTANCE_GROUP";
const ENV_MODEL_NAME: &str = "KVCM_LIVE_MODEL_NAME";
const ENV_MODEL_DTYPE: &str = "KVCM_LIVE_MODEL_DTYPE";

/// The harness provisions a 2s heartbeat timeout; 300ms keeps every reporter
/// alive with room for a slow test step.
const HEARTBEAT: Duration = Duration::from_millis(300);
/// Generous per-RPC budgets: these tests assert KVCM's answers, not scheduling.
const RPC_TIMEOUT: Duration = Duration::from_secs(5);

/// The harness configuration, or `None` when the live tests are not requested.
struct LiveEnv {
    endpoint: String,
    /// Instance registered by `setup_kvcm.py` with the `full`/`swa`/`mamba`
    /// spec names; tests that exercise the no-auto-register path report here.
    instance_id: String,
    instance_group: String,
    /// The model deployment that instance was registered with: KVCM refuses a
    /// registration without one, and a re-registration whose fields differ.
    model_name: String,
    dtype: String,
}

fn nonempty_env(key: &str) -> Option<String> {
    let value = std::env::var(key).ok()?.trim().to_string();
    (!value.is_empty()).then_some(value)
}

fn live_env() -> Option<LiveEnv> {
    let endpoint = nonempty_env(ENV_ENDPOINT)?;
    let required = |key: &str| {
        nonempty_env(key).unwrap_or_else(|| {
            panic!("{ENV_ENDPOINT} is set but {key} is not; source the harness env file")
        })
    };
    Some(LiveEnv {
        endpoint,
        instance_id: required(ENV_INSTANCE_ID),
        instance_group: required(ENV_INSTANCE_GROUP),
        model_name: required(ENV_MODEL_NAME),
        dtype: required(ENV_MODEL_DTYPE),
    })
}

/// Skips the test (rather than failing) when the harness is not running.
macro_rules! live_env {
    () => {
        match live_env() {
            Some(env) => env,
            None => {
                eprintln!("{ENV_ENDPOINT} is not set; skipping live KVCM test");
                return;
            }
        }
    };
}

/// KVCM instances are created, never deleted, so every test takes a fresh id:
/// runs are independent and repeatable against one long-lived manager.
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

fn backend(env: &LiveEnv, instance_id: &str, heartbeat: Option<Duration>) -> KvcmIndexerBackend {
    let mut config = KvcmBackendConfig::new(env.endpoint.clone(), instance_id);
    config.query_timeout = RPC_TIMEOUT;
    config.apply_timeout = RPC_TIMEOUT;
    config.heartbeat_interval = heartbeat;
    KvcmIndexerBackend::new(config).expect("the harness endpoint is a valid http endpoint")
}

fn auto_registering_backend(
    env: &LiveEnv,
    instance_id: &str,
    block_size: u32,
) -> KvcmIndexerBackend {
    let mut config = KvcmBackendConfig::new(env.endpoint.clone(), instance_id);
    config.query_timeout = RPC_TIMEOUT;
    config.apply_timeout = RPC_TIMEOUT;
    config.heartbeat_interval = Some(HEARTBEAT);
    config.auto_register = Some(KvcmAutoRegister {
        instance_group: env.instance_group.clone(),
        block_size,
        model_name: env.model_name.clone(),
        dtype: env.dtype.clone(),
    });
    KvcmIndexerBackend::new(config).expect("the harness endpoint is a valid http endpoint")
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

fn clear_batch(worker: &str, address: &str, seq: u64) -> ApplyExternalKvBatchRequest {
    ApplyExternalKvBatchRequest {
        worker_id: worker.to_string(),
        seq,
        actions: vec![ExternalKvAction {
            r#type: ExternalKvActionType::ActionClearAllAtTier as i32,
            tier: TierType::TierHbm as i32,
            hashes: Vec::new(),
            component_masks: Vec::new(),
            block_sizes: Vec::new(),
            parent_block_hash: None,
        }],
        worker_address: address.to_string(),
        cache_spec: None,
    }
}

async fn prefix(
    backend: &KvcmIndexerBackend,
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

/// Talks to KVCM directly, for behavior our backend hides or maps away
/// (snapshot flags, raw `p2p_1_*` fields, medium filtering).
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
            .expect("KVCM gRPC endpoint is reachable");
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

/// The reporter's URI convention (channel scheme + reporter authority +
/// medium), which KVCM requires to be structurally valid.
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

// --- Ingestion: what KVCM accepts ------------------------------------------

/// The auto-registration path end to end: `RegisterInstance` with `full`/
/// `swa`/`mamba` `size: 0` specs, then reports landing as prefixes.
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
        "KVCM answers the local prefix per reporter"
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

    // KVCM's `local` is bounded by the query, so a longer report cannot
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

/// `max_blocks` truncates the ask before it reaches KVCM.
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

/// The deployment-registered path: no `AUTO_REGISTER`, an instance the harness
/// (standing in for the KVCM connector) already registered with our spec names.
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

/// Re-registering an existing instance with identical fields is a no-op; any
/// field mismatch is a permanent fault. The harness registers this instance
/// with exactly what `KvcmAutoRegister` sends, so the first must hold.
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

/// `BLOCK_ADD` specs must name specs the instance declared: the harness
/// instance declares `full`/`swa`/`mamba`, an instance declaring anything else
/// rejects every report (the bridge exits instead of retrying forever).
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
        error.message().contains("KVCM rejected ReportEvent"),
        "unexpected message: {}",
        error.message()
    );

    // The rejection names no reporter, so nothing was indexed.
    let answer = raw.query(&[51], &["hbm"], 0).await;
    assert!(answer.hosts.is_empty(), "rejected blocks must not index");
}

/// A reporter KVCM never saw, or forgot, is a permanent fault for the query
/// path (the bridge's retry is what re-registers it).
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

/// `BLOCK_DELETE` must name registered specs. Our REVOKE sends every component
/// name as a superset removal, which is only legal because the instance
/// declares them all.
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

/// Re-sending `NODE_REGISTER` for an existing reporter is accepted, and the
/// placements reported before it stay readable (KVCM merges the node instead of
/// replacing it).
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

/// The boundary the README documents: heartbeats keep an idle reporter's
/// placements routable, silence lets KVCM delete them.
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

/// A silent reporter is reclaimed — placements deleted, not hidden — and its
/// next report fails `NODE_NOT_REGISTERED` until a `NODE_REGISTER` clears the
/// tombstone. That is exactly the sequence the backend's self-heal handles:
/// one rejected batch, then a successful retry carrying the registration.
#[tokio::test]
#[ignore]
async fn liveness_reclaims_a_silent_reporter_and_the_retry_self_heals() {
    let env = live_env!();
    let instance = unique("reclaim");
    // The instance is registered directly so that no heartbeating backend ever
    // owns the reporter: silence is what this test needs.
    let _raw = RawKvcm::connect(&env, &instance, &["full", "swa", "mamba"], 128).await;
    let backend = backend(&env, &instance, None);
    let worker = format!("{instance}-w0");
    let address = "http://127.0.0.1:30011";

    backend
        .apply_external_kv_batch(report_batch(&worker, address, 1, &[101, 102, 103]))
        .await
        .expect("first report registers the reporter and its blocks");
    assert_eq!(
        prefix(&backend, &[101, 102, 103], 0)
            .await
            .best_prefix_blocks,
        3
    );

    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let answer = prefix(&backend, &[101, 102, 103], 0).await;
        if answer.matches.is_empty() {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "KVCM never stopped answering the silent reporter: {:?}",
            answer.matches
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    // Becoming unroutable is immediate; the tombstone that rejects plain
    // reports only appears once `cleanup_grace_ms` has passed and KVCM has
    // unregistered the reporter. Probe with a revoke so the wait writes
    // nothing the sweep below could interleave with.
    let mut seq = 2;
    let error = loop {
        match backend
            .apply_external_kv_batch(revoke_batch(&worker, address, seq, &[104, 105, 106]))
            .await
        {
            Err(error) => break error,
            Ok(_) => {
                seq += 1;
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "KVCM never unregistered the silent reporter"
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

    // KVCM dispatches the metadata sweep in the same breath as the unregister,
    // and its generation guard only covers writes that landed before the sweep
    // took its lease. A real bridge reconnects with a delay, so this test waits
    // the same way instead of racing a sweep that deletes by location.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // The bridge retries the batch; the backend now prepends the registration,
    // which clears the tombstone.
    backend
        .apply_external_kv_batch(report_batch(&worker, address, seq, &[104, 105, 106]))
        .await
        .expect("the retry re-registers the reporter and succeeds");
    assert_eq!(
        prefix(&backend, &[104, 105, 106], 0)
            .await
            .best_prefix_blocks,
        3
    );
    // Reclaimed blocks stay reclaimed: the retry indexes only what it re-reports.
    assert!(prefix(&backend, &[101, 102, 103], 0)
        .await
        .matches
        .is_empty());
}

// --- Query semantics --------------------------------------------------------

/// Reporters with no router-facing address are excluded: KVCM knows the host,
/// the Router cannot route to it.
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
        .expect("a report without an address is still accepted by KVCM");

    let answer = prefix(&backend, &[111, 112], 0).await;
    assert!(
        answer.matches.is_empty(),
        "an unroutable reporter must not be answered: {:?}",
        answer.matches
    );

    // KVCM keeps the placement under the reporter id; the filter is ours, not
    // KVCM's, so a router that learned the address later would see it again.
    let mut raw = RawKvcm::connect(&env, &instance, &["full", "swa", "mamba"], 128).await;
    assert_eq!(
        locals(&raw.query(&[111, 112], &["hbm"], 0).await),
        HashMap::from([(worker, 2)])
    );
}

/// The raw `p2p_host_count = 0` answer, i.e. the fast path:
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

/// The reported medium decides which placements a query sees: the backend
/// reports HBM and queries HBM, so a DRAM-filtered query is empty.
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

/// `snapshot_required` marks a reporter lifecycle, not an error: KVCM asks for
/// a reconciliation snapshot while the reporter has no committed snapshot
/// version, and the first delta of a lifecycle is what creates that version —
/// the request that creates it carries the flag too
/// (`CacheManager::refresh_snapshot_response`: `committed.empty() ||
/// created_generation`). A delta-only reporter therefore sees it exactly once
/// per lifecycle.
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

/// The other event-report channel: an L2 deployment reports `vineyard://` spec
/// URIs and answers the same way (and only on the L2 channel). The router-side
/// P2P dimension is out of scope here — this only pins the channel's plumbing.
#[tokio::test]
#[ignore]
async fn l2_channel_accepts_vineyard_reports() {
    let env = live_env!();
    let instance = unique("l2");
    let mut config = KvcmBackendConfig::new(env.endpoint.clone(), instance.clone());
    config.storage_type = StorageType::StEventReportL2;
    config.query_timeout = RPC_TIMEOUT;
    config.apply_timeout = RPC_TIMEOUT;
    config.heartbeat_interval = Some(HEARTBEAT);
    config.auto_register = Some(KvcmAutoRegister {
        instance_group: env.instance_group.clone(),
        block_size: 128,
        model_name: env.model_name.clone(),
        dtype: env.dtype.clone(),
    });
    let backend =
        KvcmIndexerBackend::new(config).expect("the harness endpoint is a valid http endpoint");
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

/// `CLEAR_ALL_AT_TIER` has no KVCM equivalent, so the backend drops it and the
/// cleared blocks stay indexed. Pinned here so the documented limitation fails
/// loudly if KVCM ever grows an equivalent.
#[tokio::test]
#[ignore]
async fn cleared_tiers_leave_stale_placements() {
    let env = live_env!();
    let instance = unique("clear");
    let backend = auto_registering_backend(&env, &instance, 128);
    let worker = format!("{instance}-w0");
    let address = "http://127.0.0.1:30012";

    backend
        .apply_external_kv_batch(report_batch(&worker, address, 1, &[151, 152, 153]))
        .await
        .expect("report succeeds");
    backend
        .apply_external_kv_batch(clear_batch(&worker, address, 2))
        .await
        .expect("a clear is skipped, not failed");

    let answer = prefix(&backend, &[151, 152, 153], 0).await;
    assert_eq!(
        prefixes_by_address(&answer),
        HashMap::from([(address.to_string(), 3)]),
        "skipped clears leave the stale placements indexed"
    );
}
