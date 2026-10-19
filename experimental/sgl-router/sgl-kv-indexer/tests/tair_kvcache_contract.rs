// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Contract tests for the Tair KVCache backend: drive
//! [`TairKVCacheIndexerBackend`] (and, in one case, the whole `KVIndexer` service)
//! against an in-process mock `MetaService` over real gRPC. The mock records every
//! request, pinning the translation layer -- actions to events, host answers to
//! prefix matches -- not the manager's own matching.

#[path = "common/kv.rs"]
mod test_kv;
#[path = "common/net.rs"]
mod test_net;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tonic::transport::Server;
use tonic::{Code, Request, Response, Status};

use sgl_kv_indexer::pb::kv_indexer_client::KvIndexerClient;
use sgl_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ExternalKvActionType, GetExternalKvHitCountsRequest,
    MatchExternalKvPrefixRequest, MatchExternalKvRequest,
};
use sgl_kv_indexer::tair_kvcache_pb::meta_service_server::{MetaService, MetaServiceServer};
use sgl_kv_indexer::tair_kvcache_pb::{
    CommonResponse, CommonResponseHeader, ErrorCode, FinishWriteCacheRequest,
    GetCacheLocationLenRequest, GetCacheLocationLenResponse, GetCacheLocationRequest,
    GetCacheLocationResponse, GetCacheLocationsByBackendRequest,
    GetCacheLocationsByBackendResponse, GetCacheMetaRequest, GetCacheMetaResponse,
    GetClusterInfoRequest, GetClusterInfoResponse, GetHostCacheStateRequest,
    GetHostCacheStateResponse, GetInstanceInfoRequest, GetInstanceInfoResponse, HostCacheMatch,
    QueryType, RegisterInstanceRequest, RegisterInstanceResponse, RemoveCacheRequest,
    ReportEventRequest, ReportEventResponse, StartWriteCacheRequest, StartWriteCacheResponse,
    Status as TairKVCacheStatus, StorageType, TrimCacheRequest,
};
use sgl_kv_indexer::{
    server_builder, KvIndexerBackend, KvIndexerService, TairKVCacheBackendConfig,
    TairKVCacheIndexerBackend, COMPONENT_FULL,
};
use test_kv::{action, apply_request, component_report, dram, hbm};
use test_net::free_addr;

/// Everything the mock Tair KVCache observed, plus the canned answers it serves.
#[derive(Default)]
struct MockState {
    registrations: Vec<RegisterInstanceRequest>,
    reports: Vec<ReportEventRequest>,
    queries: Vec<GetHostCacheStateRequest>,
    /// Canned answer for the next `GetHostCacheState`.
    hosts: Vec<HostCacheMatch>,
    report_error: Option<(ErrorCode, String)>,
    query_error: Option<(ErrorCode, String)>,
    register_error: Option<(ErrorCode, String)>,
    /// Canned partial-failure `item_results` for `ReportEvent`.
    report_item_results: Vec<i32>,
    report_snapshot_required: bool,
    /// Canned per-snapshot failures, one popped per snapshot request (empty means
    /// "commit"). A failed snapshot answers with the code in the header and
    /// `item_results`, plus `retry_after_ms` when rate-limited.
    snapshot_failures: Vec<ErrorCode>,
    /// Spec names per registered instance, as `RegisterInstance` declared
    /// them: Tair KVCache checks every reported name against this set.
    registered_specs: HashMap<String, Vec<String>>,
}

impl MockState {
    fn lock(state: &Arc<Mutex<Self>>) -> std::sync::MutexGuard<'_, Self> {
        state.lock().unwrap_or_else(|error| error.into_inner())
    }

    /// The spec names the manager would accept for this instance.
    fn declared_specs(&self, instance_id: &str) -> Vec<String> {
        self.registered_specs
            .get(instance_id)
            .cloned()
            .unwrap_or_else(|| {
                DEPLOYMENT_SPEC_NAMES
                    .iter()
                    .map(|name| name.to_string())
                    .collect()
            })
    }
}

fn ok_header() -> CommonResponseHeader {
    CommonResponseHeader {
        status: Some(TairKVCacheStatus {
            code: ErrorCode::Ok as i32,
            message: String::new(),
        }),
        request_id: String::new(),
        tracer_result: String::new(),
    }
}

fn err_header(code: ErrorCode, message: &str) -> CommonResponseHeader {
    CommonResponseHeader {
        status: Some(TairKVCacheStatus {
            code: code as i32,
            message: message.to_string(),
        }),
        request_id: String::new(),
        tracer_result: String::new(),
    }
}

/// The spec names Tair KVCache accepts for an instance the mock never saw registered:
/// the connector's deployment (see the live suite's `full`/`swa`/`mamba`
/// instance). A registration replaces this set for that instance.
const DEPLOYMENT_SPEC_NAMES: [&str; 3] = ["full", "swa", "mamba"];

/// How Tair KVCache answers a rejected request: the whole request or nothing, or a
/// per-event `item_results` entry.
enum ReportViolation {
    /// Request-level checks run before any event is inspected and answer with
    /// a header error (`CacheManager::ReportEvent`).
    Request(String),
    /// Per-event checks answer per item.
    Items(String),
}

/// The Tair KVCache acceptance rules the backend's generated events must satisfy
/// (mirrors cache_manager.cc: BLOCK_ADD specs carry a channel-scheme URI and
/// name a spec the instance declared, BLOCK_DELETE names a declared spec,
/// NODE_REGISTER declares mediums). Returns the first violation's reason.
fn validate_report_events(
    request: &ReportEventRequest,
    registered: &[String],
) -> Option<ReportViolation> {
    use sgl_kv_indexer::tair_kvcache_pb::event_item::EventParams;
    use sgl_kv_indexer::tair_kvcache_pb::ReportEventType as T;
    // `CacheManager::ReportEvent`'s request-level snapshot rules: a snapshot
    // travels alone, at most one per request; empty blocks are legal.
    let snapshots = request
        .events
        .iter()
        .filter(|event| event.event_type == T::EventBlockSnapshot as i32)
        .count();
    let deltas = request
        .events
        .iter()
        .filter(|event| {
            event.event_type == T::EventBlockAdd as i32
                || event.event_type == T::EventBlockDelete as i32
        })
        .count();
    if snapshots > 0 && deltas > 0 {
        return Some(ReportViolation::Request(
            "snapshot and delta mutations must use separate ReportEvent requests".into(),
        ));
    }
    if snapshots > 1 {
        return Some(ReportViolation::Request(
            "a ReportEvent request may contain only one complete snapshot".into(),
        ));
    }
    let expected_scheme = match StorageType::try_from(request.storage_type) {
        Ok(StorageType::StEventReportL2) => "vineyard://",
        _ => "event_report://",
    };
    let declared = |name: &str| registered.iter().any(|spec| spec == name);
    for event in &request.events {
        match T::try_from(event.event_type) {
            Ok(T::EventBlockAdd) => {
                let Some(EventParams::BlockAdd(add)) = event.event_params.as_ref() else {
                    continue;
                };
                if add.specs.is_empty() {
                    return Some(ReportViolation::Items("BLOCK_ADD without specs".into()));
                }
                for spec in &add.specs {
                    if !spec.uri.starts_with(expected_scheme) {
                        return Some(ReportViolation::Items(format!(
                            "BLOCK_ADD spec `{}` URI `{}` lacks channel scheme `{expected_scheme}`",
                            spec.name, spec.uri
                        )));
                    }
                    if !declared(&spec.name) {
                        return Some(ReportViolation::Items(format!(
                            "BLOCK_ADD names undeclared spec `{}`",
                            spec.name
                        )));
                    }
                }
            }
            Ok(T::EventBlockDelete) => {
                let Some(EventParams::BlockDelete(delete)) = event.event_params.as_ref() else {
                    continue;
                };
                if delete.spec_names.is_empty() {
                    return Some(ReportViolation::Items(
                        "BLOCK_DELETE without spec_names".into(),
                    ));
                }
                for name in &delete.spec_names {
                    if !declared(name) {
                        return Some(ReportViolation::Items(format!(
                            "BLOCK_DELETE names undeclared spec `{name}`"
                        )));
                    }
                }
            }
            Ok(T::EventNodeRegister) => {
                let Some(EventParams::NodeRegister(register)) = event.event_params.as_ref() else {
                    continue;
                };
                if register.mediums.is_empty() {
                    return Some(ReportViolation::Items(
                        "NODE_REGISTER without mediums".into(),
                    ));
                }
            }
            _ => {}
        }
    }
    None
}

/// An in-process Tair KVCache MetaService that records the three RPCs the backend
/// uses and refuses everything else.
struct MockTairKVCacheManager {
    state: Arc<Mutex<MockState>>,
}

/// The registration fields Tair KVCache requires of every `RegisterInstance`
/// (`MetaServiceImpl::RegisterInstance`), so that a registration the mock
/// accepts is one a real manager accepts too.
fn validate_registration(request: &RegisterInstanceRequest) -> Option<String> {
    if request.instance_group.is_empty() {
        return Some("RegisterInstance without instance_group".into());
    }
    if request.instance_id.is_empty() {
        return Some("RegisterInstance without instance_id".into());
    }
    if request.block_size <= 0 {
        return Some(format!(
            "RegisterInstance with block_size {}",
            request.block_size
        ));
    }
    let Some(deployment) = request.model_deployment.as_ref() else {
        return Some("RegisterInstance without a model_deployment".into());
    };
    if deployment.model_name.is_empty() {
        return Some("model_deployment without model_name".into());
    }
    if deployment.dtype.is_empty() {
        return Some("model_deployment without dtype".into());
    }
    if request.location_spec_infos.is_empty() {
        return Some("RegisterInstance without location_spec_infos".into());
    }
    if request
        .location_spec_infos
        .iter()
        .any(|spec| spec.name.is_empty())
    {
        return Some("location_spec_infos with an empty name".into());
    }
    None
}

#[tonic::async_trait]
impl MetaService for MockTairKVCacheManager {
    async fn register_instance(
        &self,
        request: Request<RegisterInstanceRequest>,
    ) -> Result<Response<RegisterInstanceResponse>, Status> {
        let request = request.into_inner();
        let violation = validate_registration(&request);
        let error = {
            let mut state = MockState::lock(&self.state);
            let observed = state.register_error.clone();
            let rejected = observed.is_some() || violation.is_some();
            // Spec names resolve against the instance's registration, but only
            // an accepted registration creates one; a rejected registration
            // must keep failing reports like the real manager.
            if !rejected {
                state.registered_specs.insert(
                    request.instance_id.clone(),
                    request
                        .location_spec_infos
                        .iter()
                        .map(|spec| spec.name.clone())
                        .collect(),
                );
            }
            state.registrations.push(request);
            observed
        };
        let header = match (error, violation) {
            (Some((code, message)), _) => err_header(code, &message),
            (None, None) => ok_header(),
            (None, Some(reason)) => err_header(ErrorCode::InvalidArgument, &reason),
        };
        Ok(Response::new(RegisterInstanceResponse {
            header: Some(header),
            storage_configs: String::new(),
            extra_info: String::new(),
        }))
    }

    async fn report_event(
        &self,
        request: Request<ReportEventRequest>,
    ) -> Result<Response<ReportEventResponse>, Status> {
        let request = request.into_inner();
        // Tair KVCache's own acceptance rules (cache_manager.cc
        // ValidateEventLocationSpec): invalid events fail loudly here, and
        // spec names resolve against the instance's registration.
        let (violation, error, snapshot_failure, item_results, snapshot_required) = {
            let mut state = MockState::lock(&self.state);
            let declared = state.declared_specs(&request.instance_id);
            let violation = validate_report_events(&request, &declared);
            // Scripted snapshot outcomes, one per snapshot request.
            let is_snapshot = request.events.iter().any(|event| {
                event.event_type
                    == sgl_kv_indexer::tair_kvcache_pb::ReportEventType::EventBlockSnapshot as i32
            });
            let snapshot_failure = if is_snapshot && !state.snapshot_failures.is_empty() {
                Some(state.snapshot_failures.remove(0))
            } else {
                None
            };
            let item_results = std::mem::take(&mut state.report_item_results);
            let snapshot_required = state.report_snapshot_required;
            let error = state.report_error.clone();
            state.reports.push(request);
            (
                violation,
                error,
                snapshot_failure,
                item_results,
                snapshot_required,
            )
        };
        let header = match error {
            Some((code, message)) => err_header(code, &message),
            None => ok_header(),
        };
        if let Some(code) = snapshot_failure {
            return Ok(Response::new(ReportEventResponse {
                header: Some(err_header(code, "canned snapshot failure")),
                item_results: vec![code as i32],
                committed_snapshot_version: String::new(),
                retry_after_ms: if code == ErrorCode::SnapshotRateLimited {
                    30_000
                } else {
                    0
                },
                snapshot_required,
                extra_info: String::new(),
            }));
        }
        if let Some(violation) = violation {
            let (header, item_results, extra_info) = match violation {
                ReportViolation::Request(reason) => (
                    err_header(ErrorCode::InvalidArgument, &reason),
                    Vec::new(),
                    String::new(),
                ),
                ReportViolation::Items(reason) => {
                    (header, vec![ErrorCode::InvalidArgument as i32], reason)
                }
            };
            return Ok(Response::new(ReportEventResponse {
                header: Some(header),
                item_results,
                committed_snapshot_version: String::new(),
                retry_after_ms: 0,
                snapshot_required,
                extra_info,
            }));
        }
        Ok(Response::new(ReportEventResponse {
            header: Some(header),
            item_results,
            committed_snapshot_version: String::new(),
            retry_after_ms: 0,
            snapshot_required,
            extra_info: String::new(),
        }))
    }

    async fn get_host_cache_state(
        &self,
        request: Request<GetHostCacheStateRequest>,
    ) -> Result<Response<GetHostCacheStateResponse>, Status> {
        let request = request.into_inner();
        let (hosts, error) = {
            let mut state = MockState::lock(&self.state);
            state.queries.push(request);
            (state.hosts.clone(), state.query_error.clone())
        };
        let header = match error {
            Some((code, message)) => err_header(code, &message),
            None => ok_header(),
        };
        Ok(Response::new(GetHostCacheStateResponse {
            header: Some(header),
            hosts,
        }))
    }

    async fn get_instance_info(
        &self,
        _request: Request<GetInstanceInfoRequest>,
    ) -> Result<Response<GetInstanceInfoResponse>, Status> {
        Err(Status::unimplemented("not used by the KV Indexer"))
    }

    async fn get_cache_meta(
        &self,
        _request: Request<GetCacheMetaRequest>,
    ) -> Result<Response<GetCacheMetaResponse>, Status> {
        Err(Status::unimplemented("not used by the KV Indexer"))
    }

    async fn get_cache_location(
        &self,
        _request: Request<GetCacheLocationRequest>,
    ) -> Result<Response<GetCacheLocationResponse>, Status> {
        Err(Status::unimplemented("not used by the KV Indexer"))
    }

    async fn get_cache_location_len(
        &self,
        _request: Request<GetCacheLocationLenRequest>,
    ) -> Result<Response<GetCacheLocationLenResponse>, Status> {
        Err(Status::unimplemented("not used by the KV Indexer"))
    }

    async fn start_write_cache(
        &self,
        _request: Request<StartWriteCacheRequest>,
    ) -> Result<Response<StartWriteCacheResponse>, Status> {
        Err(Status::unimplemented("not used by the KV Indexer"))
    }

    async fn finish_write_cache(
        &self,
        _request: Request<FinishWriteCacheRequest>,
    ) -> Result<Response<CommonResponse>, Status> {
        Err(Status::unimplemented("not used by the KV Indexer"))
    }

    async fn remove_cache(
        &self,
        _request: Request<RemoveCacheRequest>,
    ) -> Result<Response<CommonResponse>, Status> {
        Err(Status::unimplemented("not used by the KV Indexer"))
    }

    async fn trim_cache(
        &self,
        _request: Request<TrimCacheRequest>,
    ) -> Result<Response<CommonResponse>, Status> {
        Err(Status::unimplemented("not used by the KV Indexer"))
    }

    async fn get_cluster_info(
        &self,
        _request: Request<GetClusterInfoRequest>,
    ) -> Result<Response<GetClusterInfoResponse>, Status> {
        Err(Status::unimplemented("not used by the KV Indexer"))
    }

    async fn get_cache_locations_by_backend(
        &self,
        _request: Request<GetCacheLocationsByBackendRequest>,
    ) -> Result<Response<GetCacheLocationsByBackendResponse>, Status> {
        Err(Status::unimplemented("not used by the KV Indexer"))
    }
}

async fn start_mock_tair_kvcache_manager(state: Arc<Mutex<MockState>>) -> SocketAddr {
    let addr = free_addr();
    tokio::spawn(async move {
        Server::builder()
            .add_service(MetaServiceServer::new(MockTairKVCacheManager { state }))
            .serve(addr)
            .await
            .expect("mock Tair KVCache serve");
    });
    addr
}

fn backend_config(addr: SocketAddr) -> TairKVCacheBackendConfig {
    let mut config = TairKVCacheBackendConfig::new(format!("http://{addr}"), "test-instance");
    // Generous timeouts: these tests assert translation, not scheduling.
    config.query_timeout = Duration::from_secs(2);
    config.apply_timeout = Duration::from_secs(2);
    config.heartbeat_interval = None;
    config
}

async fn start_backend() -> (TairKVCacheIndexerBackend, Arc<Mutex<MockState>>) {
    let state = Arc::new(Mutex::new(MockState::default()));
    let addr = start_mock_tair_kvcache_manager(Arc::clone(&state)).await;
    let backend = TairKVCacheIndexerBackend::new(backend_config(addr)).expect("valid endpoint");
    (backend, state)
}

#[tokio::test]
async fn apply_translates_report_revoke_and_registers_node() {
    let (backend, state) = start_backend().await;
    let mut batch = apply_request(
        "worker-0",
        "http://w0:30000",
        1,
        vec![
            component_report(hbm(), &[1, 2], &[COMPONENT_FULL, COMPONENT_FULL], &[16, 16]),
            action(ExternalKvActionType::ActionRevoke, hbm(), &[3]),
        ],
    );
    // A FULL-only spec keeps the worker whole-block answerable (eligible).
    batch.cache_spec = Some(sgl_kv_indexer::pb::WorkerCacheSpec {
        version: 1,
        components: COMPONENT_FULL,
        swa_window_tokens: 0,
        full_tier_mask: 1 << hbm(),
        swa_tier_mask: 0,
        mamba_tier_mask: 0,
    });
    backend
        .apply_external_kv_batch(batch)
        .await
        .expect("apply succeeds");

    let state = MockState::lock(&state);
    assert_eq!(state.registrations.len(), 0, "no auto-register by default");
    assert_eq!(state.reports.len(), 1);
    let report = &state.reports[0];
    assert_eq!(report.instance_id, "test-instance");
    assert_eq!(report.host_ip_port, "worker-0");
    assert_eq!(report.storage_type, StorageType::StEventReportL1p5 as i32);
    assert_eq!(report.events.len(), 4);

    use sgl_kv_indexer::tair_kvcache_pb::event_item::EventParams;
    use sgl_kv_indexer::tair_kvcache_pb::ReportEventType;
    assert_eq!(
        report.events[0].event_type,
        ReportEventType::EventNodeRegister as i32
    );
    let EventParams::NodeRegister(register) = report.events[0].event_params.as_ref().unwrap()
    else {
        panic!("first event must be NODE_REGISTER");
    };
    assert_eq!(register.mediums, ["hbm"]);

    let EventParams::BlockAdd(add) = report.events[1].event_params.as_ref().unwrap() else {
        panic!("expected BLOCK_ADD");
    };
    assert_eq!(add.block_key, "1");
    assert_eq!(add.medium, "hbm");
    assert_eq!(add.specs.len(), 1);
    assert_eq!(add.specs[0].name, "full");
    // The reporter address authority, with the channel's scheme.
    assert_eq!(add.specs[0].uri, "event_report://w0:30000/hbm");

    let EventParams::BlockAdd(add) = report.events[2].event_params.as_ref().unwrap() else {
        panic!("expected BLOCK_ADD");
    };
    // An eligible worker is whole-block answerable: every block carries `full`.
    assert_eq!(
        add.specs
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>(),
        ["full"]
    );

    let EventParams::BlockDelete(delete) = report.events[3].event_params.as_ref().unwrap() else {
        panic!("expected BLOCK_DELETE");
    };
    assert_eq!(delete.block_key, "3");
    assert_eq!(delete.spec_names, ["full", "swa", "mamba"]);
}

#[tokio::test]
async fn apply_clears_with_an_empty_snapshot_of_its_own() {
    use sgl_kv_indexer::tair_kvcache_pb::event_item::EventParams;
    use sgl_kv_indexer::tair_kvcache_pb::ReportEventType;
    let (backend, state) = start_backend().await;
    // DRAM is outside the default report tiers, so its REPORT is dropped; the
    // CLEAR still goes, and the registration rides the snapshot.
    backend
        .apply_external_kv_batch(apply_request(
            "worker-0",
            "http://w0:30000",
            1,
            vec![
                action(ExternalKvActionType::ActionReport, dram(), &[1]),
                action(ExternalKvActionType::ActionClearAllAtTier, hbm(), &[]),
            ],
        ))
        .await
        .expect("apply succeeds");
    {
        let state = MockState::lock(&state);
        assert_eq!(state.reports.len(), 1);
        let request = &state.reports[0];
        assert_eq!(request.storage_type, StorageType::StEventReportL1p5 as i32);
        assert_eq!(request.events.len(), 2);
        assert_eq!(
            request.events[0].event_type,
            ReportEventType::EventNodeRegister as i32
        );
        let EventParams::BlockSnapshot(snapshot) = request.events[1]
            .event_params
            .as_ref()
            .expect("snapshot carries params")
        else {
            panic!("expected BLOCK_SNAPSHOT");
        };
        assert!(
            snapshot.blocks.is_empty(),
            "a clear is an empty complete snapshot"
        );
    }

    // A later batch with nothing forwardable sends no RPC at all.
    backend
        .apply_external_kv_batch(apply_request(
            "worker-0",
            "http://w0:30000",
            2,
            vec![action(ExternalKvActionType::ActionReport, dram(), &[2])],
        ))
        .await
        .expect("apply succeeds");
    assert_eq!(MockState::lock(&state).reports.len(), 1);
}

/// A pending registration rides the batch's *first* request. For a first-seen
/// worker whose batch reports blocks and then clears, that is the delta: a
/// reporter Tair KVCache has forgotten would fail the unregistered delta and lose the
/// whole batch otherwise, and the registration on the snapshot would never be
/// reached (the review's wire probe: `apply-1 node_register_count=0`, then
/// `apply-2 node_register_count=1`).
/// One request as the Tair KVCache access log would phrase it.
fn wire_shape(request: &ReportEventRequest) -> String {
    use sgl_kv_indexer::tair_kvcache_pb::ReportEventType as T;
    let count = |kind: T| {
        request
            .events
            .iter()
            .filter(|event| event.event_type == kind as i32)
            .count()
    };
    format!(
        "node_register={} block_add={} block_delete={} block_snapshot={}",
        count(T::EventNodeRegister),
        count(T::EventBlockAdd),
        count(T::EventBlockDelete),
        count(T::EventBlockSnapshot),
    )
}

#[tokio::test]
async fn a_first_batch_that_reports_then_clears_registers_on_its_first_request() {
    use sgl_kv_indexer::tair_kvcache_pb::ReportEventType;
    let (backend, state) = start_backend().await;
    backend
        .apply_external_kv_batch(apply_request(
            "worker-0",
            "http://w0:30000",
            1,
            vec![
                action(ExternalKvActionType::ActionReport, hbm(), &[1]),
                action(ExternalKvActionType::ActionClearAllAtTier, hbm(), &[]),
            ],
        ))
        .await
        .expect("apply succeeds");

    let reports = MockState::lock(&state).reports.clone();
    let shape: Vec<String> = reports.iter().map(wire_shape).collect();
    assert_eq!(
        shape,
        [
            "node_register=1 block_add=1 block_delete=0 block_snapshot=0",
            "node_register=0 block_add=0 block_delete=0 block_snapshot=1",
        ],
        "the registration rides the first request; the snapshot goes alone"
    );
    assert_eq!(
        reports[0].events[0].event_type,
        ReportEventType::EventNodeRegister as i32
    );
    assert_eq!(
        reports[0].events[1].event_type,
        ReportEventType::EventBlockAdd as i32
    );
}

/// A flush is best-effort: Tair KVCache's rate limit (30s by default) cannot be waited
/// out inside an apply budget, and failing the batch would cost more than a
/// missed flush -- the bridge reconnects and does not replay, so every delta
/// that rode in the same batch would be lost too.
#[tokio::test]
async fn a_rate_limited_clear_succeeds_without_retrying() {
    use sgl_kv_indexer::tair_kvcache_pb::ReportEventType;
    let (backend, state) = start_backend().await;
    MockState::lock(&state).snapshot_failures = vec![ErrorCode::SnapshotRateLimited];
    backend
        .apply_external_kv_batch(apply_request(
            "worker-0",
            "http://w0:30000",
            1,
            vec![action(
                ExternalKvActionType::ActionClearAllAtTier,
                hbm(),
                &[],
            )],
        ))
        .await
        .expect("a rate-limited flush must not fail the batch");

    let state = MockState::lock(&state);
    let snapshots = state
        .reports
        .iter()
        .filter(|request| {
            request
                .events
                .iter()
                .any(|event| event.event_type == ReportEventType::EventBlockSnapshot as i32)
        })
        .count();
    assert_eq!(snapshots, 1, "the cooldown is not retried inside the batch");
}

/// Tair KVCache forgets a reporter it has not heard from; the flush re-registers it in
/// the retry rather than leaving the clear to the next batch.
#[tokio::test]
async fn a_clear_registers_a_forgotten_reporter_and_retries() {
    use sgl_kv_indexer::tair_kvcache_pb::ReportEventType;
    let (backend, state) = start_backend().await;
    // The reporter is known (it just reported blocks), so the flush request
    // itself carries no registration -- only the retry does.
    backend
        .apply_external_kv_batch(apply_request(
            "worker-0",
            "http://w0:30000",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &[1])],
        ))
        .await
        .expect("the report registers the reporter");
    MockState::lock(&state).snapshot_failures = vec![ErrorCode::NodeNotRegistered];
    backend
        .apply_external_kv_batch(apply_request(
            "worker-0",
            "http://w0:30000",
            2,
            vec![action(
                ExternalKvActionType::ActionClearAllAtTier,
                hbm(),
                &[],
            )],
        ))
        .await
        .expect("the flush re-registers and succeeds");

    let state = MockState::lock(&state);
    let snapshots: Vec<&ReportEventRequest> = state
        .reports
        .iter()
        .filter(|request| {
            request
                .events
                .iter()
                .any(|event| event.event_type == ReportEventType::EventBlockSnapshot as i32)
        })
        .collect();
    assert_eq!(snapshots.len(), 2, "one rejected attempt, one retry");
    assert!(
        !snapshots[0]
            .events
            .iter()
            .any(|event| event.event_type == ReportEventType::EventNodeRegister as i32),
        "the first attempt has nothing to register with"
    );
    assert_eq!(
        snapshots[1].events[0].event_type,
        ReportEventType::EventNodeRegister as i32,
        "the retry carries a fresh NODE_REGISTER"
    );
}

/// A concurrent snapshot is a transient race, not a failure: the complete
/// snapshot is re-sent after a short backoff.
#[tokio::test]
async fn a_snapshot_in_progress_clear_is_retried_until_it_commits() {
    use sgl_kv_indexer::tair_kvcache_pb::ReportEventType;
    let (backend, state) = start_backend().await;
    MockState::lock(&state).snapshot_failures =
        vec![ErrorCode::SnapshotInProgress, ErrorCode::SnapshotInProgress];
    backend
        .apply_external_kv_batch(apply_request(
            "worker-0",
            "http://w0:30000",
            1,
            vec![action(
                ExternalKvActionType::ActionClearAllAtTier,
                hbm(),
                &[],
            )],
        ))
        .await
        .expect("the third attempt commits");

    let state = MockState::lock(&state);
    let snapshots = state
        .reports
        .iter()
        .filter(|request| {
            request
                .events
                .iter()
                .any(|event| event.event_type == ReportEventType::EventBlockSnapshot as i32)
        })
        .count();
    assert_eq!(snapshots, 3, "two in-progress answers, then a commit");
}

/// The mock's snapshot rules are Tair KVCache's, so the empty snapshot the flush
/// relies on stays accepted and a mixed request fails loudly here instead of
/// only against a real manager (`CacheManager::ReportEvent`).
#[tokio::test]
async fn mock_snapshot_accepts_empty_and_rejects_mixed_requests() {
    use sgl_kv_indexer::tair_kvcache_pb::event_item::EventParams;
    use sgl_kv_indexer::tair_kvcache_pb::meta_service_client::MetaServiceClient;
    use sgl_kv_indexer::tair_kvcache_pb::{
        BlockAddEventParams, BlockSnapshotEventParams, EventItem, ReportEventType,
    };

    let state = Arc::new(Mutex::new(MockState::default()));
    let addr = start_mock_tair_kvcache_manager(Arc::clone(&state)).await;
    // The mock's address is handed back before its task binds, so the first
    // connect can race the listener.
    let mut client = None;
    for _ in 0..50 {
        match MetaServiceClient::connect(format!("http://{addr}")).await {
            Ok(connected) => {
                client = Some(connected);
                break;
            }
            Err(_) => tokio::time::sleep(Duration::from_millis(20)).await,
        }
    }
    let mut client = client.expect("the mock is reachable");

    let snapshot = || EventItem {
        event_type: ReportEventType::EventBlockSnapshot as i32,
        event_params: Some(EventParams::BlockSnapshot(BlockSnapshotEventParams {
            blocks: Vec::new(),
            ..Default::default()
        })),
    };
    let add = |hash: i64| EventItem {
        event_type: ReportEventType::EventBlockAdd as i32,
        event_params: Some(EventParams::BlockAdd(BlockAddEventParams {
            block_key: hash.to_string(),
            medium: "hbm".into(),
            specs: Vec::new(),
            ..Default::default()
        })),
    };
    let request = |events: Vec<EventItem>| ReportEventRequest {
        trace_id: "mock-rules".into(),
        instance_id: "test-instance".into(),
        host_ip_port: "worker-0".into(),
        events,
        storage_type: StorageType::StEventReportL1p5 as i32,
    };
    let code_of = |response: &ReportEventResponse| {
        ErrorCode::try_from(
            response
                .header
                .as_ref()
                .and_then(|header| header.status.as_ref())
                .expect("every response carries a status")
                .code,
        )
        .expect("the mock answers known codes")
    };

    let answer = client
        .report_event(request(vec![snapshot()]))
        .await
        .expect("the mock answers")
        .into_inner();
    assert_eq!(
        code_of(&answer),
        ErrorCode::Ok,
        "an empty snapshot is legal"
    );

    let answer = client
        .report_event(request(vec![snapshot(), add(7)]))
        .await
        .expect("the mock answers")
        .into_inner();
    assert_eq!(
        code_of(&answer),
        ErrorCode::InvalidArgument,
        "a snapshot mixed with deltas is a request-level rejection"
    );

    let answer = client
        .report_event(request(vec![snapshot(), snapshot()]))
        .await
        .expect("the mock answers")
        .into_inner();
    assert_eq!(code_of(&answer), ErrorCode::InvalidArgument);
}

#[tokio::test]
async fn apply_maps_tair_kvcache_error_to_unavailable() {
    let (backend, state) = start_backend().await;
    MockState::lock(&state).report_error = Some((ErrorCode::InternalError, "boom".into()));
    let error = backend
        .apply_external_kv_batch(apply_request(
            "worker-0",
            "http://w0:30000",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &[1])],
        ))
        .await
        .expect_err("Tair KVCache error must surface");
    assert_eq!(error.code(), Code::Unavailable);
    assert!(error.message().contains("boom"));
}

#[tokio::test]
async fn apply_rejects_worker_id_with_hash_char() {
    let (backend, state) = start_backend().await;
    let error = backend
        .apply_external_kv_batch(apply_request(
            "worker#0",
            "http://w0:30000",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &[1])],
        ))
        .await
        .expect_err("'#' corrupts Tair KVCache location ids");
    assert_eq!(error.code(), Code::InvalidArgument);
    assert!(MockState::lock(&state).reports.is_empty());
}

/// Sets process environment variables for one test and restores them on drop.
/// The environment is global, so this file keeps its use to the single test
/// below, which sets every variable it needs before building the config.
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

/// The environment is the entry point operators use, and the one place a
/// mistake is silent: a renamed or unenforced variable produces a registration
/// Tair KVCache rejects, which makes the bridge exit rather than misbehave visibly.
/// Drive the whole chain -- environment -> `TairKVCacheBackendConfig::from_env` ->
/// `RegisterInstance` -> wire -- so no link can drift unnoticed.
#[tokio::test]
async fn environment_configuration_reaches_the_wire() {
    let state = Arc::new(Mutex::new(MockState::default()));
    let addr = start_mock_tair_kvcache_manager(Arc::clone(&state)).await;
    // Spelled out rather than imported: a typo in the constant is what this
    // test exists to catch.
    let endpoint = format!("http://{addr}");
    let _guard = EnvGuard::set(&[
        ("KV_INDEXER_TAIR_KVCACHE_KVCM_ENDPOINT", endpoint.as_str()),
        ("KV_INDEXER_TAIR_KVCACHE_KVCM_INSTANCE_ID", "env-instance"),
        ("KV_INDEXER_TAIR_KVCACHE_KVCM_AUTO_REGISTER", "1"),
        ("KV_INDEXER_TAIR_KVCACHE_KVCM_INSTANCE_GROUP", "group-env"),
        ("KV_INDEXER_TAIR_KVCACHE_KVCM_BLOCK_SIZE", "16"),
        ("KV_INDEXER_TAIR_KVCACHE_KVCM_MODEL_NAME", "qwen3-8b"),
        ("KV_INDEXER_TAIR_KVCACHE_KVCM_MODEL_DTYPE", "FP8"),
    ]);
    let mut config =
        sgl_kv_indexer::TairKVCacheBackendConfig::from_env().expect("complete environment");
    // The tests assert translation, not scheduling.
    config.query_timeout = Duration::from_secs(2);
    config.apply_timeout = Duration::from_secs(2);
    config.heartbeat_interval = None;
    let backend = TairKVCacheIndexerBackend::new(config).expect("valid endpoint");

    backend
        .apply_external_kv_batch(apply_request(
            "worker-0",
            "http://w0:30000",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &[7])],
        ))
        .await
        .expect("apply succeeds");

    let state = MockState::lock(&state);
    let registration = state.registrations.first().expect("auto-registered");
    assert_eq!(registration.instance_id, "env-instance");
    assert_eq!(registration.instance_group, "group-env");
    assert_eq!(registration.block_size, 16);
    let deployment = registration
        .model_deployment
        .as_ref()
        .expect("Tair KVCache rejects a registration without a model deployment");
    assert_eq!(deployment.model_name, "qwen3-8b");
    assert_eq!(deployment.dtype, "FP8");
    assert_eq!(
        registration
            .location_spec_infos
            .iter()
            .map(|spec| spec.name.as_str())
            .collect::<Vec<_>>(),
        ["full", "swa", "mamba"]
    );
}

#[tokio::test]
async fn a_prefix_query_registers_an_auto_registering_instance_first() {
    // No apply batch has registered the instance yet. The manager answers
    // INSTANCE_NOT_EXIST, which is permanent, so the query path must register
    // ahead of the query it is answering.
    let state = Arc::new(Mutex::new(MockState::default()));
    let addr = start_mock_tair_kvcache_manager(Arc::clone(&state)).await;
    let mut config = backend_config(addr);
    config.auto_register = Some(sgl_kv_indexer::TairKVCacheAutoRegister {
        instance_group: "group-a".into(),
        block_size: 16,
        model_name: "qwen3-8b".into(),
        dtype: "FP8".into(),
    });
    let backend = TairKVCacheIndexerBackend::new(config).expect("valid endpoint");

    backend
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: vec![1, 2, 3],
            max_blocks: 0,
        })
        .await
        .expect("a query before any report succeeds");
    {
        let state = MockState::lock(&state);
        assert_eq!(state.registrations.len(), 1, "the query registered");
        assert_eq!(state.queries.len(), 1, "and still asked");
        assert_eq!(state.queries[0].instance_id, "test-instance");
    }

    // The second query does not re-register.
    backend
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: vec![4],
            max_blocks: 0,
        })
        .await
        .expect("query succeeds");
    assert_eq!(
        MockState::lock(&state).registrations.len(),
        1,
        "registration is once per backend, not per query"
    );

    // Without auto-registration the instance stays missing: a configuration error,
    // not a transient condition to paper over.
    let observer_state = Arc::new(Mutex::new(MockState::default()));
    let observer_addr = start_mock_tair_kvcache_manager(Arc::clone(&observer_state)).await;
    MockState::lock(&observer_state).query_error =
        Some((ErrorCode::InstanceNotExist, "no such instance".into()));
    let observer =
        TairKVCacheIndexerBackend::new(backend_config(observer_addr)).expect("valid endpoint");
    let error = observer
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: vec![1],
            max_blocks: 0,
        })
        .await
        .expect_err("an unregistered instance is still a permanent fault");
    assert_eq!(error.code(), tonic::Code::FailedPrecondition);
    assert!(
        MockState::lock(&observer_state).registrations.is_empty(),
        "no registration was invented for a deployment that configured none"
    );
}

#[tokio::test]
async fn auto_register_registers_once_with_spec_names() {
    let state = Arc::new(Mutex::new(MockState::default()));
    let addr = start_mock_tair_kvcache_manager(Arc::clone(&state)).await;
    let mut config = backend_config(addr);
    config.auto_register = Some(sgl_kv_indexer::TairKVCacheAutoRegister {
        instance_group: "group-a".into(),
        block_size: 16,
        model_name: "qwen3-8b".into(),
        dtype: "FP8".into(),
    });
    let backend = TairKVCacheIndexerBackend::new(config).expect("valid endpoint");

    for seq in 1..=2 {
        backend
            .apply_external_kv_batch(apply_request(
                "worker-0",
                "http://w0:30000",
                seq,
                vec![action(
                    ExternalKvActionType::ActionReport,
                    hbm(),
                    &[seq as i64],
                )],
            ))
            .await
            .expect("apply succeeds");
    }
    let state = MockState::lock(&state);
    assert_eq!(state.registrations.len(), 1, "register exactly once");
    let registration = &state.registrations[0];
    assert_eq!(registration.instance_id, "test-instance");
    assert_eq!(registration.instance_group, "group-a");
    assert_eq!(registration.block_size, 16);
    assert_eq!(
        registration
            .location_spec_infos
            .iter()
            .map(|spec| spec.name.as_str())
            .collect::<Vec<_>>(),
        ["full", "swa", "mamba"]
    );
    assert_eq!(
        registration.default_query_type,
        QueryType::QtPrefixMatch as i32
    );
    // Tair KVCache rejects a registration without a model deployment and a positive
    // block size, so both must be on the wire.
    let deployment = registration
        .model_deployment
        .as_ref()
        .expect("registration carries the model deployment Tair KVCache requires");
    assert_eq!(deployment.model_name, "qwen3-8b");
    assert_eq!(deployment.dtype, "FP8");
    // Both batches were forwarded after the registration.
    assert_eq!(state.reports.len(), 2);
}

#[tokio::test]
async fn prefix_maps_hosts_to_routable_addresses_sorted() {
    let (backend, state) = start_backend().await;
    // Empty-action batches still record each worker's routing address.
    for (worker, address) in [
        ("worker-0", "http://w0:30000"),
        ("worker-1", "http://w1:30000"),
        ("worker-2", ""),
    ] {
        backend
            .apply_external_kv_batch(apply_request(worker, address, 0, Vec::new()))
            .await
            .expect("address-only apply succeeds");
    }
    let hashes: Vec<i64> = (1..=10).collect();
    {
        let mut state = MockState::lock(&state);
        state.hosts = vec![
            host_match("worker-0", 3),
            host_match("worker-1", 7),
            host_match("ghost", 9),
            host_match("worker-2", 5),
        ];
    }

    let response = backend
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: hashes.clone(),
            max_blocks: 0,
        })
        .await
        .expect("prefix query succeeds");

    // Sorted by matched blocks; the unknown host and the address-less worker
    // cannot be routed and are excluded.
    assert_eq!(response.best_prefix_blocks, 7);
    assert_eq!(response.blocks_read, 10);
    assert_eq!(response.matches.len(), 2);
    assert_eq!(response.matches[0].worker_address, "http://w1:30000");
    assert_eq!(response.matches[0].matched_prefix_blocks, 7);
    assert_eq!(response.matches[1].worker_address, "http://w0:30000");
    assert_eq!(response.matches[1].matched_prefix_blocks, 3);

    let state = MockState::lock(&state);
    assert_eq!(state.queries.len(), 1);
    let query = &state.queries[0];
    assert_eq!(query.instance_id, "test-instance");
    assert_eq!(query.query_type, QueryType::QtPrefixMatch as i32);
    assert_eq!(query.block_cache_keys, hashes);
    assert_eq!(query.medium, ["hbm"]);
    assert_eq!(query.p2p_host_count, 0);
}

#[tokio::test]
async fn prefix_truncates_to_max_blocks() {
    let (backend, state) = start_backend().await;
    backend
        .apply_external_kv_batch(apply_request("worker-0", "http://w0:30000", 0, Vec::new()))
        .await
        .expect("address-only apply succeeds");
    MockState::lock(&state).hosts = vec![host_match("worker-0", 4)];

    let response = backend
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: vec![1; 10],
            max_blocks: 4,
        })
        .await
        .expect("prefix query succeeds");

    assert_eq!(response.blocks_read, 4);
    assert_eq!(response.matches.len(), 1);
    assert_eq!(response.matches[0].matched_prefix_blocks, 4);
    assert_eq!(
        MockState::lock(&state).queries[0].block_cache_keys,
        vec![1; 4]
    );
}

#[tokio::test]
async fn prefix_clamps_oversized_local_answer() {
    let (backend, state) = start_backend().await;
    backend
        .apply_external_kv_batch(apply_request("worker-0", "http://w0:30000", 0, Vec::new()))
        .await
        .expect("address-only apply succeeds");
    MockState::lock(&state).hosts = vec![host_match("worker-0", 99)];

    let response = backend
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: vec![1; 10],
            max_blocks: 0,
        })
        .await
        .expect("prefix query succeeds");
    // A Tair KVCache over-report cannot claim more blocks than the query carried.
    assert_eq!(response.matches[0].matched_prefix_blocks, 10);
}

#[tokio::test]
async fn prefix_maps_tair_kvcache_unready_to_unavailable() {
    let (backend, state) = start_backend().await;
    MockState::lock(&state).query_error = Some((ErrorCode::ServiceNotReady, "starting".into()));
    let error = backend
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: vec![1, 2, 3],
            max_blocks: 0,
        })
        .await
        .expect_err("Tair KVCache error must surface");
    assert_eq!(error.code(), Code::Unavailable);
    assert!(error.message().contains("starting"));
}

#[tokio::test]
async fn match_external_kv_is_unimplemented() {
    let (backend, _state) = start_backend().await;
    let error = backend
        .match_external_kv(MatchExternalKvRequest {
            hashes: vec![1],
            count_as_hit: false,
        })
        .await
        .expect_err("no Tair KVCache equivalent exists");
    assert_eq!(error.code(), Code::Unimplemented);
}

#[tokio::test]
async fn hit_counts_stay_empty() {
    let (backend, _state) = start_backend().await;
    let response = backend
        .get_external_kv_hit_counts(GetExternalKvHitCountsRequest { hashes: vec![1] })
        .await
        .expect("diagnostic RPC still answers");
    assert!(response.entries.is_empty());
}

#[tokio::test]
async fn heartbeats_keep_reporters_alive() {
    let state = Arc::new(Mutex::new(MockState::default()));
    let addr = start_mock_tair_kvcache_manager(Arc::clone(&state)).await;
    let mut config = backend_config(addr);
    config.heartbeat_interval = Some(Duration::from_millis(20));
    let backend = TairKVCacheIndexerBackend::new(config).expect("valid endpoint");

    backend
        .apply_external_kv_batch(apply_request(
            "worker-0",
            "http://w0:30000",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &[1])],
        ))
        .await
        .expect("apply succeeds");
    tokio::time::sleep(Duration::from_millis(150)).await;

    let heartbeats = MockState::lock(&state)
        .reports
        .iter()
        .filter(|report| {
            report.host_ip_port == "worker-0"
                && report.events.iter().any(|event| {
                    event.event_type
                        == sgl_kv_indexer::tair_kvcache_pb::ReportEventType::EventHeartbeat as i32
                })
        })
        .count();
    assert!(heartbeats >= 1, "expected at least one heartbeat report");
}

/// The backend behind the real `KVIndexer` service: validation, admission, and
/// the wire contract of the router-facing RPCs stay unchanged.
#[tokio::test]
async fn tair_kvcache_backend_over_real_indexer_grpc() {
    let mock_state = Arc::new(Mutex::new(MockState::default()));
    let mock_addr = start_mock_tair_kvcache_manager(Arc::clone(&mock_state)).await;
    let backend =
        TairKVCacheIndexerBackend::new(backend_config(mock_addr)).expect("valid endpoint");

    let svc = KvIndexerService::new(backend).into_server();
    let addr = free_addr();
    tokio::spawn(async move {
        server_builder()
            .add_service(svc)
            .serve(addr)
            .await
            .expect("indexer serve");
    });
    let endpoint = format!("http://{addr}");
    let mut client = loop {
        if let Ok(client) = KvIndexerClient::connect(endpoint.clone()).await {
            break client;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    };

    client
        .apply_external_kv_batch(ApplyExternalKvBatchRequest {
            worker_id: "worker-0".into(),
            seq: 1,
            actions: vec![action(ExternalKvActionType::ActionReport, hbm(), &[1, 2])],
            worker_address: "http://w0:30000".into(),
            cache_spec: None,
        })
        .await
        .expect("apply over gRPC succeeds");

    MockState::lock(&mock_state).hosts = vec![host_match("worker-0", 2)];
    let response = client
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: vec![1, 2, 3],
            max_blocks: 0,
        })
        .await
        .expect("prefix over gRPC succeeds");
    let response = response.into_inner();
    assert_eq!(response.best_prefix_blocks, 2);
    assert_eq!(response.matches.len(), 1);
    assert_eq!(response.matches[0].worker_address, "http://w0:30000");
    assert_eq!(response.matches[0].matched_prefix_blocks, 2);

    // The per-block query keeps its honest UNIMPLEMENTED over the wire, so a
    // caller cannot mistake it for "no worker holds these blocks".
    let error = client
        .match_external_kv(MatchExternalKvRequest {
            hashes: vec![1],
            count_as_hit: false,
        })
        .await
        .expect_err("no Tair KVCache equivalent exists");
    assert_eq!(error.code(), Code::Unimplemented);

    let counts = client
        .get_external_kv_hit_counts(GetExternalKvHitCountsRequest { hashes: vec![1] })
        .await
        .expect("hit counts answer over gRPC");
    assert!(counts.into_inner().entries.is_empty());
}

fn host_match(host: &str, local: i64) -> HostCacheMatch {
    HostCacheMatch {
        host_ip_port: host.to_string(),
        local,
        p2p_1_fetch: 0,
        p2p_1_total_match: 0,
    }
}

/// B1 regression: a worker whose cache spec requires SWA or MAMBA rules is
/// excluded — nothing is reported for it, and no host answer for it survives
/// the join — because whole-block matching would over-report its prefix.
#[tokio::test]
async fn component_workers_are_excluded_from_prefix_answers() {
    let (backend, state) = start_backend().await;
    let swa_spec = sgl_kv_indexer::pb::WorkerCacheSpec {
        version: 1,
        components: sgl_kv_indexer::COMPONENT_FULL | sgl_kv_indexer::COMPONENT_SWA,
        swa_window_tokens: 100,
        full_tier_mask: 1 << hbm(),
        swa_tier_mask: 1 << hbm(),
        mamba_tier_mask: 0,
    };
    let mut batch = apply_request(
        "worker-0",
        "http://w0:30000",
        1,
        vec![action(ExternalKvActionType::ActionReport, hbm(), &[1, 2])],
    );
    batch.cache_spec = Some(swa_spec);
    backend
        .apply_external_kv_batch(batch)
        .await
        .expect("apply succeeds without forwarding");

    // Nothing was reported: an excluded worker's placements are not even sent.
    assert!(
        MockState::lock(&state).reports.is_empty(),
        "excluded workers must not be reported"
    );

    // A legacy worker stays answerable, and the query still works for it.
    backend
        .apply_external_kv_batch(apply_request(
            "worker-1",
            "http://w1:30000",
            1,
            vec![action(
                ExternalKvActionType::ActionReport,
                hbm(),
                &[1, 2, 3],
            )],
        ))
        .await
        .expect("legacy worker applies");
    {
        let mut state = MockState::lock(&state);
        state.hosts = vec![host_match("worker-0", 3), host_match("worker-1", 2)];
    }
    let response = backend
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: vec![1, 2, 3],
            max_blocks: 0,
        })
        .await
        .expect("prefix query succeeds");
    assert_eq!(response.matches.len(), 1, "only the legacy worker answers");
    assert_eq!(response.matches[0].worker_address, "http://w1:30000");
}

/// A component batch's zero masks carry the same hole as legacy reports: the
/// bridge encodes mask-less blocks as 0 inside a component batch, and the rule
/// engine fails the unconfirmed block closed. The whole worker must be
/// excluded, not just the block.
#[tokio::test]
async fn zero_mask_inside_a_component_batch_excludes_the_worker() {
    let (backend, state) = start_backend().await;
    let mut batch = apply_request(
        "worker-0",
        "http://w0:30000",
        1,
        vec![component_report(
            hbm(),
            &[1, 2],
            &[COMPONENT_FULL, 0],
            &[16, 16],
        )],
    );
    batch.cache_spec = Some(sgl_kv_indexer::pb::WorkerCacheSpec {
        version: 1,
        components: COMPONENT_FULL,
        swa_window_tokens: 0,
        full_tier_mask: 1 << hbm(),
        swa_tier_mask: 0,
        mamba_tier_mask: 0,
    });
    backend
        .apply_external_kv_batch(batch)
        .await
        .expect("apply succeeds");
    assert!(
        MockState::lock(&state).reports.is_empty(),
        "the unconfirmed block must not be reported as a whole-block placement"
    );
}

/// Eligibility is sticky: a worker that was reported while eligible is
/// excluded from answers the moment a later batch makes it ineligible, even
/// though Tair KVCache still holds its earlier placements.
#[tokio::test]
async fn eligibility_is_sticky_across_batches() {
    let (backend, state) = start_backend().await;
    // Batch 1: eligible legacy report, forwarded.
    backend
        .apply_external_kv_batch(apply_request(
            "worker-0",
            "http://w0:30000",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &[1, 2])],
        ))
        .await
        .expect("first batch forwards");
    assert_eq!(MockState::lock(&state).reports.len(), 1);

    // Batch 2: component detail under the same (absent) spec → ineligible.
    backend
        .apply_external_kv_batch(apply_request(
            "worker-0",
            "http://w0:30000",
            2,
            vec![component_report(hbm(), &[3], &[COMPONENT_FULL], &[16])],
        ))
        .await
        .expect("second batch is accepted");

    MockState::lock(&state).hosts = vec![host_match("worker-0", 2)];
    let response = backend
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: vec![1, 2],
            max_blocks: 0,
        })
        .await
        .expect("prefix query succeeds");
    assert!(
        response.matches.is_empty(),
        "Tair KVCache still holds the placements, but the worker must not be answered"
    );
}

/// A heartbeat that fails with `NODE_NOT_REGISTERED` re-registers on the next
/// tick: the failure is transient, and the flag reset drives the retry.
#[tokio::test]
async fn heartbeats_re_register_forgotten_reporters() {
    let state = Arc::new(Mutex::new(MockState::default()));
    let addr = start_mock_tair_kvcache_manager(Arc::clone(&state)).await;
    let mut config = backend_config(addr);
    config.heartbeat_interval = Some(Duration::from_millis(20));
    let backend = TairKVCacheIndexerBackend::new(config).expect("valid endpoint");

    // Register and report once, then make Tair KVCache forget the reporter.
    backend
        .apply_external_kv_batch(apply_request(
            "worker-0",
            "http://w0:30000",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &[1])],
        ))
        .await
        .expect("apply succeeds");
    MockState::lock(&state).report_error = Some((ErrorCode::NodeNotRegistered, "forgotten".into()));
    tokio::time::sleep(Duration::from_millis(150)).await;
    MockState::lock(&state).report_error = None;
    tokio::time::sleep(Duration::from_millis(150)).await;

    // After recovery, some heartbeat report must carry NODE_REGISTER again.
    let re_registered = MockState::lock(&state).reports.iter().any(|report| {
        report.host_ip_port == "worker-0"
            && report.events.iter().any(|event| {
                event.event_type
                    == sgl_kv_indexer::tair_kvcache_pb::ReportEventType::EventNodeRegister as i32
                    && report.events.iter().any(|e| {
                        e.event_type
                            == sgl_kv_indexer::tair_kvcache_pb::ReportEventType::EventHeartbeat
                                as i32
                    })
            })
    });
    assert!(
        re_registered,
        "a heartbeat after the reset must carry NODE_REGISTER"
    );
}

/// B2 regression: a `NODE_REGISTER` is not consumed by a failed batch, and a
/// reporter Tair KVCache forgot is re-registered by the next successful one.
#[tokio::test]
async fn node_register_is_resent_after_tair_kvcache_forgets_the_reporter() {
    let (backend, state) = start_backend().await;
    let batch = |seq: u64| {
        apply_request(
            "worker-0",
            "http://w0:30000",
            seq,
            vec![action(
                ExternalKvActionType::ActionReport,
                hbm(),
                &[seq as i64],
            )],
        )
    };

    // Phase 1: successful register+apply.
    backend
        .apply_external_kv_batch(batch(1))
        .await
        .expect("first apply succeeds");
    // Phase 2: Tair KVCache forgot the node; the batch fails loudly.
    MockState::lock(&state).report_error = Some((ErrorCode::NodeNotRegistered, "forgotten".into()));
    assert!(backend.apply_external_kv_batch(batch(2)).await.is_err());
    // Phase 3: recovery — the reporter must be re-registered.
    MockState::lock(&state).report_error = None;
    backend
        .apply_external_kv_batch(batch(3))
        .await
        .expect("third apply succeeds");

    let reports = {
        let state = MockState::lock(&state);
        std::mem::take(&mut state.reports.clone())
    };
    assert_eq!(reports.len(), 3);
    let has_register = |report: &ReportEventRequest| {
        report.events.iter().any(|event| {
            event.event_type
                == sgl_kv_indexer::tair_kvcache_pb::ReportEventType::EventNodeRegister as i32
        })
    };
    assert!(has_register(&reports[0]), "first batch registers the node");
    assert!(
        !has_register(&reports[1]),
        "a registered reporter does not re-register"
    );
    assert!(
        has_register(&reports[2]),
        "a reporter Tair KVCache forgot is re-registered"
    );
}

#[tokio::test]
async fn large_batches_are_chunked_in_order_under_one_budget() {
    let (backend, state) = start_backend().await;
    let hashes: Vec<i64> = (1..=5_000).collect();
    backend
        .apply_external_kv_batch(apply_request(
            "worker-0",
            "http://w0:30000",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &hashes)],
        ))
        .await
        .expect("large apply succeeds");

    let reports = {
        let state = MockState::lock(&state);
        std::mem::take(&mut state.reports.clone())
    };
    assert_eq!(reports.len(), 2, "5001 events split at 4096");
    // Chunk 1: NODE_REGISTER then the first 4095 blocks; chunk 2: the rest.
    assert_eq!(reports[0].events.len(), 4_096);
    assert_eq!(reports[1].events.len(), 905);
    assert_eq!(
        reports[0].events[0].event_type,
        sgl_kv_indexer::tair_kvcache_pb::ReportEventType::EventNodeRegister as i32
    );
    let keys: Vec<&str> = reports
        .iter()
        .flat_map(|report| report.events.iter())
        .filter_map(|event| match event.event_params.as_ref()? {
            sgl_kv_indexer::tair_kvcache_pb::event_item::EventParams::BlockAdd(add) => {
                Some(add.block_key.as_str())
            }
            _ => None,
        })
        .collect();
    let expected: Vec<String> = hashes.iter().map(|h| h.to_string()).collect();
    assert_eq!(
        keys,
        expected.iter().map(String::as_str).collect::<Vec<_>>()
    );
}

/// The spec URI scheme follows the event-report channel (vineyard for l2),
/// and a worker with no restated address falls back to its id.
#[tokio::test]
async fn spec_uris_follow_the_channel_and_reporter_identity() {
    let state = Arc::new(Mutex::new(MockState::default()));
    let addr = start_mock_tair_kvcache_manager(Arc::clone(&state)).await;
    let mut config = backend_config(addr);
    config.storage_type = StorageType::StEventReportL2;
    let backend = TairKVCacheIndexerBackend::new(config).expect("valid endpoint");

    // With an address: vineyard://<authority>/<medium>.
    backend
        .apply_external_kv_batch(apply_request(
            "worker-0",
            "http://10.0.0.1:30000",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &[1])],
        ))
        .await
        .expect("apply succeeds");
    // Without one: vineyard://<worker id>/<medium>.
    backend
        .apply_external_kv_batch(apply_request(
            "worker-1",
            "",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &[1])],
        ))
        .await
        .expect("address-less apply succeeds");

    let reports = MockState::lock(&state).reports.clone();
    assert_eq!(reports.len(), 2);
    use sgl_kv_indexer::tair_kvcache_pb::event_item::EventParams;
    let uris = |report: &ReportEventRequest| {
        report
            .events
            .iter()
            .filter_map(|event| match event.event_params.as_ref()? {
                EventParams::BlockAdd(add) => Some(add.specs[0].uri.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(uris(&reports[0]), ["vineyard://10.0.0.1:30000/hbm"]);
    assert_eq!(uris(&reports[1]), ["vineyard://worker-1/hbm"]);
    // The mock's channel rule accepted both shapes.
    let declared: Vec<String> = DEPLOYMENT_SPEC_NAMES
        .iter()
        .map(|name| name.to_string())
        .collect();
    assert!(
        reports
            .iter()
            .all(|report| validate_report_events(report, &declared).is_none()),
        "generated events must satisfy Tair KVCache acceptance rules"
    );
}

#[tokio::test]
async fn partial_item_results_fail_the_apply() {
    let (backend, state) = start_backend().await;
    MockState::lock(&state).report_item_results = vec![0, 0];
    let error = backend
        .apply_external_kv_batch(apply_request(
            "worker-0",
            "http://w0:30000",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &[1, 2])],
        ))
        .await
        .expect_err("a non-empty item_results list means partial failure");
    assert_eq!(error.code(), tonic::Code::Internal);
    assert!(error.message().contains("partially failed"));
}

#[tokio::test]
async fn snapshot_required_is_advisory() {
    let (backend, state) = start_backend().await;
    MockState::lock(&state).report_snapshot_required = true;
    backend
        .apply_external_kv_batch(apply_request(
            "worker-0",
            "http://w0:30000",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &[1])],
        ))
        .await
        .expect("an advisory snapshot request must not fail the batch");
}

#[tokio::test]
async fn permanent_configuration_faults_are_failed_precondition() {
    let (backend, state) = start_backend().await;
    MockState::lock(&state).report_error =
        Some((ErrorCode::InstanceNotExist, "no such instance".into()));
    let error = backend
        .apply_external_kv_batch(apply_request(
            "worker-0",
            "http://w0:30000",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &[1])],
        ))
        .await
        .expect_err("instance fault must surface");
    // Not Unavailable: the bridge must exit on a permanent fault instead of
    // reconnecting forever with a silently empty index.
    assert_eq!(error.code(), tonic::Code::FailedPrecondition);
}

/// Dropping the backend stops every heartbeat task, and each worker gets its
/// own.
#[tokio::test]
async fn heartbeats_stop_when_the_backend_drops() {
    let state = Arc::new(Mutex::new(MockState::default()));
    let addr = start_mock_tair_kvcache_manager(Arc::clone(&state)).await;
    let mut config = backend_config(addr);
    config.heartbeat_interval = Some(Duration::from_millis(20));
    let backend = TairKVCacheIndexerBackend::new(config).expect("valid endpoint");
    for worker in ["worker-0", "worker-1"] {
        backend
            .apply_external_kv_batch(apply_request(worker, "http://w", 1, Vec::new()))
            .await
            .expect("address-only apply succeeds");
    }
    tokio::time::sleep(Duration::from_millis(120)).await;
    let heartbeats_for = |worker: &str| {
        MockState::lock(&state)
            .reports
            .iter()
            .filter(|report| {
                report.host_ip_port == worker
                    && report.events.iter().any(|event| {
                        event.event_type
                            == sgl_kv_indexer::tair_kvcache_pb::ReportEventType::EventHeartbeat
                                as i32
                    })
            })
            .count()
    };
    assert!(heartbeats_for("worker-0") >= 1, "worker-0 heartbeats");
    assert!(heartbeats_for("worker-1") >= 1, "worker-1 heartbeats");

    let before = MockState::lock(&state).reports.len();
    drop(backend);
    tokio::time::sleep(Duration::from_millis(120)).await;
    let after = MockState::lock(&state).reports.len();
    assert_eq!(after, before, "no reports after the backend is dropped");
}
