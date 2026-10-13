// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! KVCM backend contract tests: drive [`KvcmIndexerBackend`] (and, in one
//! case, the full `KVIndexer` service in front of it) against an in-process
//! mock KVCM `MetaService` over real gRPC. The mock records every request, so
//! these tests pin the translation layer — indexer actions to KVCM events,
//! KVCM host answers to indexer prefix matches — not KVCM's own matching.

#[path = "common/kv.rs"]
mod test_kv;
#[path = "common/net.rs"]
mod test_net;

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tonic::transport::Server;
use tonic::{Code, Request, Response, Status};

use sgl_kv_indexer::kvcm_pb::meta_service_server::{MetaService, MetaServiceServer};
use sgl_kv_indexer::kvcm_pb::{
    CommonResponse, CommonResponseHeader, ErrorCode, FinishWriteCacheRequest,
    GetCacheLocationLenRequest, GetCacheLocationLenResponse, GetCacheLocationRequest,
    GetCacheLocationResponse, GetCacheLocationsByBackendRequest,
    GetCacheLocationsByBackendResponse, GetCacheMetaRequest, GetCacheMetaResponse,
    GetClusterInfoRequest, GetClusterInfoResponse, GetHostCacheStateRequest,
    GetHostCacheStateResponse, GetInstanceInfoRequest, GetInstanceInfoResponse, HostCacheMatch,
    QueryType, RegisterInstanceRequest, RegisterInstanceResponse, RemoveCacheRequest,
    ReportEventRequest, ReportEventResponse, StartWriteCacheRequest, StartWriteCacheResponse,
    Status as KvcmStatus, StorageType, TrimCacheRequest,
};
use sgl_kv_indexer::pb::kv_indexer_client::KvIndexerClient;
use sgl_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ExternalKvActionType, GetExternalKvHitCountsRequest,
    MatchExternalKvPrefixRequest, MatchExternalKvRequest,
};
use sgl_kv_indexer::{
    server_builder, KvIndexerBackend, KvIndexerService, KvcmBackendConfig, KvcmIndexerBackend,
    COMPONENT_FULL,
};
use test_kv::{action, apply_request, component_report, dram, hbm};
use test_net::free_addr;

/// Everything the mock KVCM observed, plus the canned answers it serves.
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
}

impl MockState {
    fn lock(state: &Arc<Mutex<Self>>) -> std::sync::MutexGuard<'_, Self> {
        state.lock().unwrap_or_else(|error| error.into_inner())
    }
}

fn ok_header() -> CommonResponseHeader {
    CommonResponseHeader {
        status: Some(KvcmStatus {
            code: ErrorCode::Ok as i32,
            message: String::new(),
        }),
        request_id: String::new(),
        tracer_result: String::new(),
    }
}

fn err_header(code: ErrorCode, message: &str) -> CommonResponseHeader {
    CommonResponseHeader {
        status: Some(KvcmStatus {
            code: code as i32,
            message: message.to_string(),
        }),
        request_id: String::new(),
        tracer_result: String::new(),
    }
}

/// The KVCM acceptance rules the backend's generated events must satisfy
/// (mirrors cache_manager.cc: BLOCK_ADD specs carry a channel-scheme URI,
/// BLOCK_DELETE names a registered spec, NODE_REGISTER declares mediums).
/// Returns the first violation's reason.
fn validate_report_events(request: &ReportEventRequest) -> Option<String> {
    use sgl_kv_indexer::kvcm_pb::event_item::EventParams;
    use sgl_kv_indexer::kvcm_pb::ReportEventType as T;
    let expected_scheme = match StorageType::try_from(request.storage_type) {
        Ok(StorageType::StEventReportL2) => "vineyard://",
        _ => "event_report://",
    };
    for event in &request.events {
        match T::try_from(event.event_type) {
            Ok(T::EventBlockAdd) => {
                let Some(EventParams::BlockAdd(add)) = event.event_params.as_ref() else {
                    continue;
                };
                if add.specs.is_empty() {
                    return Some("BLOCK_ADD without specs".into());
                }
                for spec in &add.specs {
                    if !spec.uri.starts_with(expected_scheme) {
                        return Some(format!(
                            "BLOCK_ADD spec `{}` URI `{}` lacks channel scheme `{expected_scheme}`",
                            spec.name, spec.uri
                        ));
                    }
                }
            }
            Ok(T::EventBlockDelete) => {
                let Some(EventParams::BlockDelete(delete)) = event.event_params.as_ref() else {
                    continue;
                };
                if delete.spec_names.is_empty() {
                    return Some("BLOCK_DELETE without spec_names".into());
                }
                const REGISTERED: [&str; 3] = ["full", "swa", "mamba"];
                for name in &delete.spec_names {
                    if !REGISTERED.contains(&name.as_str()) {
                        return Some(format!("BLOCK_DELETE names unregistered spec `{name}`"));
                    }
                }
            }
            Ok(T::EventNodeRegister) => {
                let Some(EventParams::NodeRegister(register)) = event.event_params.as_ref() else {
                    continue;
                };
                if register.mediums.is_empty() {
                    return Some("NODE_REGISTER without mediums".into());
                }
            }
            _ => {}
        }
    }
    None
}

/// An in-process KVCM MetaService that records the three RPCs the backend
/// uses and refuses everything else.
struct MockKvcm {
    state: Arc<Mutex<MockState>>,
}

/// The registration fields KVCM requires of every `RegisterInstance`
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
impl MetaService for MockKvcm {
    async fn register_instance(
        &self,
        request: Request<RegisterInstanceRequest>,
    ) -> Result<Response<RegisterInstanceResponse>, Status> {
        let request = request.into_inner();
        let violation = validate_registration(&request);
        let error = {
            let mut state = MockState::lock(&self.state);
            state.registrations.push(request);
            state.register_error.clone()
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
        // KVCM's own acceptance rules, so a regression to structurally
        // invalid events fails loudly here instead of only against a real
        // server (see cache_manager.cc ValidateEventLocationSpec).
        let violation = validate_report_events(&request);
        let error = {
            let mut state = MockState::lock(&self.state);
            state.reports.push(request);
            state.report_error.clone()
        };
        let header = match error {
            Some((code, message)) => err_header(code, &message),
            None => ok_header(),
        };
        let (mut item_results, snapshot_required) = {
            let mut state = MockState::lock(&self.state);
            (
                std::mem::take(&mut state.report_item_results),
                state.report_snapshot_required,
            )
        };
        if let Some(reason) = violation {
            item_results = vec![ErrorCode::InvalidArgument as i32];
            return Ok(Response::new(ReportEventResponse {
                header: Some(header),
                item_results,
                committed_snapshot_version: String::new(),
                retry_after_ms: 0,
                snapshot_required,
                extra_info: reason,
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

async fn start_mock_kvcm(state: Arc<Mutex<MockState>>) -> SocketAddr {
    let addr = free_addr();
    tokio::spawn(async move {
        Server::builder()
            .add_service(MetaServiceServer::new(MockKvcm { state }))
            .serve(addr)
            .await
            .expect("mock KVCM serve");
    });
    addr
}

fn backend_config(addr: SocketAddr) -> KvcmBackendConfig {
    let mut config = KvcmBackendConfig::new(format!("http://{addr}"), "test-instance");
    // Generous timeouts: these tests assert translation, not scheduling.
    config.query_timeout = Duration::from_secs(2);
    config.apply_timeout = Duration::from_secs(2);
    config.heartbeat_interval = None;
    config
}

async fn start_backend() -> (KvcmIndexerBackend, Arc<Mutex<MockState>>) {
    let state = Arc::new(Mutex::new(MockState::default()));
    let addr = start_mock_kvcm(Arc::clone(&state)).await;
    let backend = KvcmIndexerBackend::new(backend_config(addr)).expect("valid endpoint");
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

    use sgl_kv_indexer::kvcm_pb::event_item::EventParams;
    use sgl_kv_indexer::kvcm_pb::ReportEventType;
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
    // An eligible worker is whole-block answerable: every reported block
    // carries the `full` component. Multi-component naming is covered by the
    // translation unit tests; on the wire it only appears for excluded
    // workers, which are never forwarded.
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
async fn apply_skips_unreported_tiers_and_clears() {
    let (backend, state) = start_backend().await;
    // DRAM is outside the default report tiers; CLEAR has no KVCM equivalent.
    // A first-seen worker still registers itself on its reported mediums.
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
        assert_eq!(state.reports[0].events.len(), 1);
        assert_eq!(
            state.reports[0].events[0].event_type,
            sgl_kv_indexer::kvcm_pb::ReportEventType::EventNodeRegister as i32
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

#[tokio::test]
async fn apply_maps_kvcm_error_to_unavailable() {
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
        .expect_err("KVCM error must surface");
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
        .expect_err("'#' corrupts KVCM location ids");
    assert_eq!(error.code(), Code::InvalidArgument);
    assert!(MockState::lock(&state).reports.is_empty());
}

#[tokio::test]
async fn auto_register_registers_once_with_spec_names() {
    let state = Arc::new(Mutex::new(MockState::default()));
    let addr = start_mock_kvcm(Arc::clone(&state)).await;
    let mut config = backend_config(addr);
    config.auto_register = Some(sgl_kv_indexer::KvcmAutoRegister {
        instance_group: "group-a".into(),
        block_size: 16,
        model_name: "qwen3-8b".into(),
        dtype: "FP8".into(),
    });
    let backend = KvcmIndexerBackend::new(config).expect("valid endpoint");

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
    // KVCM rejects a registration without a model deployment and a positive
    // block size, so both must be on the wire.
    let deployment = registration
        .model_deployment
        .as_ref()
        .expect("registration carries the model deployment KVCM requires");
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
    // A KVCM over-report cannot claim more blocks than the query carried.
    assert_eq!(response.matches[0].matched_prefix_blocks, 10);
}

#[tokio::test]
async fn prefix_maps_kvcm_unready_to_unavailable() {
    let (backend, state) = start_backend().await;
    MockState::lock(&state).query_error = Some((ErrorCode::ServiceNotReady, "starting".into()));
    let error = backend
        .match_external_kv_prefix(MatchExternalKvPrefixRequest {
            hashes: vec![1, 2, 3],
            max_blocks: 0,
        })
        .await
        .expect_err("KVCM error must surface");
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
        .expect_err("no KVCM equivalent exists");
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
    let addr = start_mock_kvcm(Arc::clone(&state)).await;
    let mut config = backend_config(addr);
    config.heartbeat_interval = Some(Duration::from_millis(20));
    let backend = KvcmIndexerBackend::new(config).expect("valid endpoint");

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
                        == sgl_kv_indexer::kvcm_pb::ReportEventType::EventHeartbeat as i32
                })
        })
        .count();
    assert!(heartbeats >= 1, "expected at least one heartbeat report");
}

/// The backend behind the real `KVIndexer` service: validation, admission, and
/// the wire contract of the router-facing RPCs stay unchanged.
#[tokio::test]
async fn kvcm_backend_over_real_indexer_grpc() {
    let mock_state = Arc::new(Mutex::new(MockState::default()));
    let mock_addr = start_mock_kvcm(Arc::clone(&mock_state)).await;
    let backend = KvcmIndexerBackend::new(backend_config(mock_addr)).expect("valid endpoint");

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
        .expect_err("no KVCM equivalent exists");
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
/// though KVCM still holds its earlier placements.
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
        "KVCM still holds the placements, but the worker must not be answered"
    );
}

/// A heartbeat that fails with `NODE_NOT_REGISTERED` re-registers on the next
/// tick: the failure is transient, and the flag reset drives the retry.
#[tokio::test]
async fn heartbeats_re_register_forgotten_reporters() {
    let state = Arc::new(Mutex::new(MockState::default()));
    let addr = start_mock_kvcm(Arc::clone(&state)).await;
    let mut config = backend_config(addr);
    config.heartbeat_interval = Some(Duration::from_millis(20));
    let backend = KvcmIndexerBackend::new(config).expect("valid endpoint");

    // Register and report once, then make KVCM forget the reporter.
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
                    == sgl_kv_indexer::kvcm_pb::ReportEventType::EventNodeRegister as i32
                    && report.events.iter().any(|e| {
                        e.event_type
                            == sgl_kv_indexer::kvcm_pb::ReportEventType::EventHeartbeat as i32
                    })
            })
    });
    assert!(
        re_registered,
        "a heartbeat after the reset must carry NODE_REGISTER"
    );
}

/// B2 regression: a `NODE_REGISTER` is not consumed by a failed batch, and a
/// reporter KVCM forgot is re-registered by the next successful one.
#[tokio::test]
async fn node_register_is_resent_after_kvcm_forgets_the_reporter() {
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
    // Phase 2: KVCM forgot the node; the batch fails loudly.
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
            event.event_type == sgl_kv_indexer::kvcm_pb::ReportEventType::EventNodeRegister as i32
        })
    };
    assert!(has_register(&reports[0]), "first batch registers the node");
    assert!(
        !has_register(&reports[1]),
        "a registered reporter does not re-register"
    );
    assert!(
        has_register(&reports[2]),
        "a reporter KVCM forgot is re-registered"
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
        sgl_kv_indexer::kvcm_pb::ReportEventType::EventNodeRegister as i32
    );
    let keys: Vec<&str> = reports
        .iter()
        .flat_map(|report| report.events.iter())
        .filter_map(|event| match event.event_params.as_ref()? {
            sgl_kv_indexer::kvcm_pb::event_item::EventParams::BlockAdd(add) => {
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
    let addr = start_mock_kvcm(Arc::clone(&state)).await;
    let mut config = backend_config(addr);
    config.storage_type = StorageType::StEventReportL2;
    let backend = KvcmIndexerBackend::new(config).expect("valid endpoint");

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
    use sgl_kv_indexer::kvcm_pb::event_item::EventParams;
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
    assert!(
        reports
            .iter()
            .all(|report| validate_report_events(report).is_none()),
        "generated events must satisfy KVCM acceptance rules"
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
    let addr = start_mock_kvcm(Arc::clone(&state)).await;
    let mut config = backend_config(addr);
    config.heartbeat_interval = Some(Duration::from_millis(20));
    let backend = KvcmIndexerBackend::new(config).expect("valid endpoint");
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
                            == sgl_kv_indexer::kvcm_pb::ReportEventType::EventHeartbeat as i32
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
