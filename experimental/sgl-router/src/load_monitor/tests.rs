use super::*;
use crate::discovery::{ModelId, WorkerSpec};
use proto::load_monitor_service_server::{LoadMonitorService, LoadMonitorServiceServer};
use proto::{RegisterResponse, WorkerFrame, WorkerType};
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::{Request, Response};

/// Builds one Router worker for store and session tests.
fn test_worker(id: &str, mode: WorkerMode) -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(id.to_string()),
        url: format!("http://{id}:30000"),
        mode,
        model_ids: vec![ModelId("model".to_string())],
        bootstrap_port: None,
    }))
}

/// Maps a discovery mode to compatibility metadata used by test reports.
fn test_worker_type(mode: WorkerMode) -> WorkerType {
    match mode {
        WorkerMode::Plain => WorkerType::Regular,
        WorkerMode::Prefill => WorkerType::Prefill,
        WorkerMode::Decode => WorkerType::Decode,
    }
}

/// Builds a healthy report with one valid rank.
fn test_report(worker_addr: &str, source: &str, sequence: u64, mode: WorkerMode) -> LoadReport {
    LoadReport {
        source_instance_id: source.to_string(),
        sequence_id: sequence,
        report_time_unix_ms: 123,
        worker: Some(proto::Worker {
            worker_addr: worker_addr.to_string(),
            worker_type: test_worker_type(mode) as i32,
            model: Some("model".to_string()),
            zone: None,
        }),
        status: ReportStatus::Healthy as i32,
        last_error: None,
        ranks: vec![RankLoad {
            dp_rank: 0,
            snapshot_time_unix_ms: 123,
            num_running_reqs: 2,
            num_waiting_reqs: 3,
            num_waiting_uncached_tokens: 4,
            num_used_tokens: 20,
            num_total_tokens: 24,
            max_total_num_tokens: 100,
            max_running_requests: 10,
            token_usage: 0.2,
            gen_throughput: 5.0,
            cache_hit_rate: 0.5,
            utilization: 0.7,
        }],
    }
}

/// Creates an enabled in-memory monitor using the supplied reporter port.
fn test_monitor(port: u16) -> LoadMonitor {
    LoadMonitor::new(LoadMonitorConfig {
        enabled: true,
        reporter_port: NonZeroU16::new(port),
    })
}

/// Installs one Worker in the snapshot store without starting a network task.
fn install_worker(monitor: &LoadMonitor, worker: Arc<Worker>) -> WorkerId {
    let port = monitor.inner.config.reporter_port.unwrap();
    let target = WorkerTarget::from_worker(&worker, Some(port)).unwrap();
    let id = target.id.clone();
    monitor.update_store(HashMap::from([(id.clone(), target)]));
    id
}

/// Starts a direct test session for one already-installed Worker.
fn begin_test_session(monitor: &LoadMonitor, worker_id: &WorkerId) -> u64 {
    monitor.begin_session(worker_id).unwrap()
}

/// Disabled snapshots preserve the documented empty state.
#[test]
fn disabled_snapshot_is_empty() {
    let snapshot = LoadMonitor::disabled().snapshot();
    assert!(!snapshot.enabled);
    assert_eq!(snapshot.version, 0);
    assert!(snapshot.captured_at.is_none());
    assert!(snapshot.workers.is_empty());
}

/// Reporter endpoints replace the inference port and preserve IPv6 syntax.
#[test]
fn reporter_endpoint_uses_worker_host_and_fixed_port() {
    let port = NonZeroU16::new(31000).unwrap();
    assert_eq!(
        reporter_endpoint("http://worker.example:30000/v1", port).unwrap(),
        "http://worker.example:31000"
    );
    assert_eq!(
        reporter_endpoint("http://[2001:db8::1]:30000", port).unwrap(),
        "http://[2001:db8::1]:31000"
    );
}

/// Rank aggregation sums counters and generation throughput.
#[test]
fn aggregate_sums_rank_loads() {
    let first = validate_rank(&test_report("ignored", "s", 1, WorkerMode::Plain).ranks[0]).unwrap();
    let mut second = first.clone();
    second.dp_rank = 1;
    let aggregate = aggregate_ranks(&[first, second]);
    assert_eq!(aggregate.total_requests, 10);
    assert_eq!(aggregate.free_tokens, 160);
    assert_eq!(aggregate.available_slots, 16);
    assert_eq!(aggregate.gen_throughput, 10.0);
}

/// Invalid counts, capacity relations, duplicate ranks, and floats are rejected.
#[test]
fn rejects_invalid_rank_contract_categories() {
    let base = test_report("ignored", "source", 1, WorkerMode::Plain);
    let mut cases = Vec::new();

    let mut negative_count = base.clone();
    negative_count.ranks[0].num_running_reqs = -1;
    cases.push(("negative count", negative_count));

    let mut token_capacity = base.clone();
    token_capacity.ranks[0].num_used_tokens = 101;
    cases.push(("token capacity", token_capacity));

    let mut request_capacity = base.clone();
    request_capacity.ranks[0].num_running_reqs = 11;
    cases.push(("request capacity", request_capacity));

    let mut duplicate_rank = base.clone();
    duplicate_rank.ranks.push(duplicate_rank.ranks[0]);
    cases.push(("duplicate rank", duplicate_rank));

    let mut infinite_metric = base;
    infinite_metric.ranks[0].utilization = f64::INFINITY;
    cases.push(("infinite metric", infinite_metric));

    for (category, report) in cases {
        let error = validate_report(&report, SystemTime::now()).unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument, "{category}");
    }
}

/// Unreachable reports carry only an explanatory error and no rank set.
#[test]
fn validates_unreachable_report_shape() {
    let mut report = test_report("ignored", "s", 1, WorkerMode::Plain);
    report.status = ReportStatus::Unreachable as i32;
    report.last_error = Some("scheduler unavailable".to_string());
    assert!(validate_report(&report, SystemTime::now()).is_err());

    report.ranks.clear();
    assert!(validate_report(&report, SystemTime::now()).is_ok());
    report.last_error = None;
    assert!(validate_report(&report, SystemTime::now()).is_err());
}

/// Discovery binds report identity; compatibility `worker_addr` is ignored.
#[test]
fn report_metadata_cannot_select_a_different_worker() {
    let monitor = test_monitor(31000);
    let id = install_worker(&monitor, test_worker("worker", WorkerMode::Plain));
    let session = begin_test_session(&monitor, &id);
    monitor
        .apply_report(
            &id,
            session,
            test_report("http://attacker:9999", "source", 1, WorkerMode::Decode),
        )
        .unwrap();
    let snapshot = monitor.snapshot();
    assert_eq!(snapshot.workers[0].worker_id, "worker");
    assert_eq!(snapshot.workers[0].sequence_id, Some(1));
}

/// Duplicate sequences are ignored while a new source may restart at one.
#[test]
fn sequence_order_is_scoped_to_source_instance() {
    let monitor = test_monitor(31000);
    let id = install_worker(&monitor, test_worker("worker", WorkerMode::Plain));
    let session = begin_test_session(&monitor, &id);
    monitor
        .apply_report(
            &id,
            session,
            test_report("ignored", "old", 10, WorkerMode::Plain),
        )
        .unwrap();
    monitor
        .apply_report(
            &id,
            session,
            test_report("ignored", "old", 9, WorkerMode::Plain),
        )
        .unwrap();
    assert_eq!(monitor.snapshot().workers[0].sequence_id, Some(10));
    monitor
        .apply_report(
            &id,
            session,
            test_report("ignored", "new", 1, WorkerMode::Plain),
        )
        .unwrap();
    assert_eq!(monitor.snapshot().workers[0].sequence_id, Some(1));
}

/// Healthy reports with zero capacity retain diagnostics but are stale.
#[test]
fn zero_capacity_healthy_report_is_locally_stale() {
    let monitor = test_monitor(31000);
    let id = install_worker(&monitor, test_worker("worker", WorkerMode::Plain));
    let session = begin_test_session(&monitor, &id);
    let mut report = test_report("ignored", "source", 1, WorkerMode::Plain);
    report.ranks[0].num_running_reqs = 0;
    report.ranks[0].num_used_tokens = 0;
    report.ranks[0].max_running_requests = 0;
    report.ranks[0].max_total_num_tokens = 0;
    monitor.apply_report(&id, session, report).unwrap();
    let snapshot = monitor.snapshot();
    assert_eq!(snapshot.workers[0].freshness, Freshness::Stale);
    assert_eq!(
        snapshot.workers[0].last_error.as_deref(),
        Some("engine reported a rank with zero token or request capacity")
    );
}

/// An owned snapshot cannot change after a later report replaces the store.
#[test]
fn captured_snapshot_is_immutable_across_updates() {
    let monitor = test_monitor(31000);
    let id = install_worker(&monitor, test_worker("worker", WorkerMode::Plain));
    let session = begin_test_session(&monitor, &id);
    monitor
        .apply_report(
            &id,
            session,
            test_report("ignored", "source", 1, WorkerMode::Plain),
        )
        .unwrap();
    let first = monitor.snapshot();
    monitor
        .apply_report(
            &id,
            session,
            test_report("ignored", "source", 2, WorkerMode::Plain),
        )
        .unwrap();
    let second = monitor.snapshot();
    assert_eq!(first.workers[0].sequence_id, Some(1));
    assert_eq!(second.workers[0].sequence_id, Some(2));
    assert!(second.version > first.version);
}

#[derive(Clone)]
struct FakeReporter {
    report: LoadReport,
    event_tx: mpsc::Sender<&'static str>,
}

#[tonic::async_trait]
impl LoadMonitorService for FakeReporter {
    type MonitorStream = ReceiverStream<std::result::Result<WorkerFrame, Status>>;

    /// Acknowledges registration, publishes one report, and records controls.
    async fn monitor(
        &self,
        request: Request<tonic::Streaming<RouterFrame>>,
    ) -> std::result::Result<Response<Self::MonitorStream>, Status> {
        let mut inbound = request.into_inner();
        let first = inbound
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("missing register frame"))?;
        let register = match first.payload {
            Some(router_frame::Payload::Register(register)) => register,
            _ => return Err(Status::invalid_argument("first frame must register")),
        };
        assert!(!register.router_id.is_empty());
        assert_eq!(register.report_interval_ms, 1000);
        assert_eq!(register.lease_ttl_ms, 15000);
        self.event_tx.send("register").await.unwrap();

        let (response_tx, response_rx) = mpsc::channel(4);
        let report = self.report.clone();
        let event_tx = self.event_tx.clone();
        tokio::spawn(async move {
            response_tx
                .send(Ok(WorkerFrame {
                    payload: Some(worker_frame::Payload::Registered(RegisterResponse {
                        lease_ttl_ms: 200,
                        renew_after_ms: 50,
                    })),
                }))
                .await
                .unwrap();
            response_tx
                .send(Ok(WorkerFrame {
                    payload: Some(worker_frame::Payload::Report(report)),
                }))
                .await
                .unwrap();
            while let Ok(Some(frame)) = inbound.message().await {
                match frame.payload {
                    Some(router_frame::Payload::KeepAlive(_)) => {
                        let _ = event_tx.send("keep_alive").await;
                    }
                    Some(router_frame::Payload::Stop(_)) => {
                        let _ = event_tx.send("stop").await;
                        break;
                    }
                    _ => {}
                }
            }
        });
        Ok(Response::new(ReceiverStream::new(response_rx)))
    }
}

/// Runs a real h2c bidi stream from Router registration through load ingestion.
#[tokio::test]
async fn router_dials_worker_and_drives_bidi_monitor_session() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let reporter_addr = listener.local_addr().unwrap();
    let (event_tx, mut event_rx) = mpsc::channel(8);
    let service = FakeReporter {
        report: test_report("http://untrusted:9999", "engine", 1, WorkerMode::Decode),
        event_tx,
    };
    let server_cancel = CancellationToken::new();
    let server_cancel_task = server_cancel.clone();
    let server = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(LoadMonitorServiceServer::new(service))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
                server_cancel_task.cancelled().await;
            })
            .await
            .unwrap();
    });

    let monitor = test_monitor(reporter_addr.port());
    let worker = Arc::new(Worker::new(WorkerSpec {
        id: WorkerId("worker".to_string()),
        url: "http://127.0.0.1:30000".to_string(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("model".to_string())],
        bootstrap_port: None,
    }));
    monitor.reconcile(vec![worker]).await;

    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), event_rx.recv())
            .await
            .unwrap(),
        Some("register")
    );
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if monitor
                .snapshot()
                .workers
                .first()
                .is_some_and(|worker| worker.freshness == Freshness::Fresh)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(monitor.snapshot().workers[0].worker_id, "worker");
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), event_rx.recv())
            .await
            .unwrap(),
        Some("keep_alive")
    );

    monitor.shutdown().await;
    server_cancel.cancel();
    server.await.unwrap();
}

/// A reporter that accepts the gRPC stream but never sends the registration ack.
#[derive(Clone)]
struct SilentReporter;

#[tonic::async_trait]
impl LoadMonitorService for SilentReporter {
    type MonitorStream = ReceiverStream<std::result::Result<WorkerFrame, Status>>;

    /// Accepts the stream and returns an open response stream with no frames.
    async fn monitor(
        &self,
        _request: Request<tonic::Streaming<RouterFrame>>,
    ) -> std::result::Result<Response<Self::MonitorStream>, Status> {
        let (tx, rx) = mpsc::channel(1);
        // Keep the sender alive without producing any frame so the client's
        // first-message wait neither resolves nor sees EOF.
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(60)).await;
            drop(tx);
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

/// A silent Worker must fail registration by ack timeout.
#[tokio::test]
async fn registration_ack_timeout_fails_session() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let reporter_addr = listener.local_addr().unwrap();
    let server_cancel = CancellationToken::new();
    let server_cancel_task = server_cancel.clone();
    let server = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(LoadMonitorServiceServer::new(SilentReporter))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
                server_cancel_task.cancelled().await;
            })
            .await
            .unwrap();
    });

    let monitor = test_monitor(reporter_addr.port());
    let worker = Arc::new(Worker::new(WorkerSpec {
        id: WorkerId("worker".to_string()),
        url: "http://127.0.0.1:30000".to_string(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("model".to_string())],
        bootstrap_port: None,
    }));
    let target = WorkerTarget::from_worker(
        &worker,
        Some(NonZeroU16::new(reporter_addr.port()).unwrap()),
    )
    .unwrap();
    let cancel = CancellationToken::new();
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        monitor.monitor_once(&target, &cancel),
    )
    .await
    .expect("registration ack timeout must complete the session attempt");
    assert!(
        result.is_err(),
        "a silent Worker must fail registration by ack timeout"
    );
    assert!(result.unwrap_err().to_string().contains("ack"));

    monitor.shutdown().await;
    server_cancel.cancel();
    server.await.unwrap();
}

/// Unreachable reports age to Stale after the freshness window.
#[test]
fn unreachable_report_ages_to_stale() {
    let monitor = test_monitor(31000);
    let id = install_worker(&monitor, test_worker("worker", WorkerMode::Plain));
    let session = begin_test_session(&monitor, &id);
    let mut report = test_report("ignored", "source", 1, WorkerMode::Plain);
    report.status = ReportStatus::Unreachable as i32;
    report.last_error = Some("scheduler unavailable".to_string());
    report.ranks.clear();
    monitor.apply_report(&id, session, report).unwrap();
    assert_eq!(
        monitor.snapshot().workers[0].freshness,
        Freshness::Unreachable
    );

    {
        let mut store = monitor.inner.store.write();
        store
            .workers
            .get_mut(&id)
            .unwrap()
            .report
            .as_mut()
            .unwrap()
            .received_at = SystemTime::now() - STALE_AFTER - Duration::from_millis(1);
    }
    assert_eq!(monitor.snapshot().workers[0].freshness, Freshness::Stale);
}

/// A Worker's own `/server_info` reporter port wins over the global fallback.
#[test]
fn worker_advertised_reporter_port_wins_over_fallback() {
    let worker = test_worker("worker", WorkerMode::Plain);
    worker.set_reporter_port(Some(32000));
    let target = WorkerTarget::from_worker(&worker, Some(NonZeroU16::new(31000).unwrap())).unwrap();
    assert_eq!(target.reporter_endpoint, "http://worker:32000");
}

/// The global fallback port applies when the Worker advertises none.
#[test]
fn fallback_port_used_when_worker_advertises_none() {
    let worker = test_worker("worker", WorkerMode::Plain);
    let target = WorkerTarget::from_worker(&worker, Some(NonZeroU16::new(31000).unwrap())).unwrap();
    assert_eq!(target.reporter_endpoint, "http://worker:31000");
}

/// A Worker with no reporter port and no fallback is skipped.
#[tokio::test]
async fn worker_without_reporter_port_is_skipped() {
    let monitor = LoadMonitor::new(LoadMonitorConfig {
        enabled: true,
        reporter_port: None,
    });
    monitor
        .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
        .await;
    assert!(monitor.snapshot().workers.is_empty());
}
