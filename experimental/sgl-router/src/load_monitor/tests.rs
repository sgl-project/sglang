use super::*;
use crate::discovery::{ModelId, WorkerSpec};

/// Builds one Router worker for store and snapshot tests.
fn test_worker(id: &str, mode: WorkerMode) -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(id.to_string()),
        url: format!("http://{id}:30000"),
        mode,
        model_ids: vec![ModelId("model".to_string())],
        bootstrap_port: None,
    }))
}

/// Builds a healthy report with one valid rank.
fn test_report(origin: &str, source: &str, sequence: u64, mode: WorkerMode) -> LoadReport {
    LoadReport {
        source_instance_id: source.to_string(),
        sequence_id: sequence,
        report_time_unix_ms: 123,
        worker: Some(proto::Worker {
            worker_addr: origin.to_string(),
            worker_type: worker_type_for_mode(mode) as i32,
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

/// Creates an enabled in-memory monitor without starting network servers.
fn test_monitor() -> LoadMonitor {
    LoadMonitor::new_enabled(
        LoadMonitorConfig {
            enabled: true,
            bind_host: "127.0.0.1".to_string(),
            bind_port: 0,
            report_ip: Some("127.0.0.1".to_string()),
        },
        12345,
    )
    .unwrap()
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

/// Rank aggregation sums counters and generation throughput.
#[test]
fn aggregate_sums_rank_loads() {
    let first = validate_rank(&test_report("w:30000", "s", 1, WorkerMode::Plain).ranks[0]).unwrap();
    let mut second = first.clone();
    second.dp_rank = 1;
    let aggregate = aggregate_ranks(&[first, second]);
    assert_eq!(aggregate.total_requests, 10);
    assert_eq!(aggregate.free_tokens, 160);
    assert_eq!(aggregate.available_slots, 16);
    assert_eq!(aggregate.gen_throughput, 10.0);
}

/// Invalid counts, capacity relations, duplicate ranks, and floats are
/// rejected before any worker snapshot can be replaced.
#[test]
fn rejects_invalid_rank_contract_categories() {
    let base = test_report("worker:30000", "source", 1, WorkerMode::Plain);
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
        assert_eq!(
            error.code(),
            tonic::Code::InvalidArgument,
            "category {category}"
        );
    }
}

/// Negative engine timestamps are rejected even though Router receipt
/// time remains authoritative for freshness.
#[test]
fn rejects_negative_report_timestamp() {
    let mut report = test_report("w:30000", "s", 1, WorkerMode::Plain);
    report.report_time_unix_ms = -1;
    assert!(validate_report(&report, SystemTime::now()).is_err());
}

/// Unreachable reports carry only an explanatory error and no rank set.
#[test]
fn validates_unreachable_report_shape() {
    let mut report = test_report("w:30000", "s", 1, WorkerMode::Plain);
    report.status = ReportStatus::Unreachable as i32;
    report.last_error = Some("scheduler unavailable".to_string());
    assert!(validate_report(&report, SystemTime::now()).is_err());

    report.ranks.clear();
    assert!(validate_report(&report, SystemTime::now()).is_ok());
    report.last_error = None;
    assert!(validate_report(&report, SystemTime::now()).is_err());
}

/// Duplicate and out-of-order sequences leave the latest accepted report.
#[tokio::test]
async fn ignores_non_increasing_sequence_without_closing_stream() {
    let monitor = test_monitor();
    monitor
        .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
        .await;
    let binding = monitor
        .begin_stream(test_report("worker:30000", "source", 2, WorkerMode::Plain))
        .unwrap();
    monitor
        .apply_stream_report(
            &binding,
            test_report("worker:30000", "source", 1, WorkerMode::Plain),
        )
        .unwrap();
    assert_eq!(monitor.snapshot().workers[0].sequence_id, Some(2));
    monitor.stop_registrations().await;
}

/// Duplicate discovery origins remain visible but cannot bind a report
/// stream to an arbitrary WorkerId.
#[tokio::test]
async fn duplicate_origin_rejects_stream_binding() {
    let monitor = test_monitor();
    let first = test_worker("worker", WorkerMode::Plain);
    let second = Arc::new(Worker::new(WorkerSpec {
        id: WorkerId("worker-copy".to_string()),
        url: first.url.clone(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("model".to_string())],
        bootstrap_port: None,
    }));
    monitor.reconcile(vec![first, second]).await;

    let error = monitor
        .begin_stream(test_report("worker:30000", "source", 1, WorkerMode::Plain))
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(error.message().contains("duplicate worker origin"));
    assert_eq!(monitor.snapshot().workers.len(), 2);
    monitor.stop_registrations().await;
}

/// First-message lookup and subsequent identity checks reject unknown,
/// role-mismatched, or changing report streams.
#[tokio::test]
async fn stream_identity_is_bound_and_immutable() {
    let monitor = test_monitor();
    monitor
        .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
        .await;

    let unknown = test_report("unknown:30000", "source", 1, WorkerMode::Plain);
    assert!(monitor.begin_stream(unknown).is_err());
    let wrong_role = test_report("worker:30000", "source", 1, WorkerMode::Decode);
    assert!(monitor.begin_stream(wrong_role).is_err());

    let binding = monitor
        .begin_stream(test_report("worker:30000", "source", 1, WorkerMode::Plain))
        .unwrap();
    let changed_source = test_report("worker:30000", "other-source", 2, WorkerMode::Plain);
    assert!(monitor
        .apply_stream_report(&binding, changed_source)
        .is_err());
    let changed_origin = test_report("other:30000", "source", 2, WorkerMode::Plain);
    assert!(monitor
        .apply_stream_report(&binding, changed_origin)
        .is_err());
    monitor.end_stream(&binding);
    monitor.stop_registrations().await;
}

/// A new source takes ownership and permanently retires the previous one.
#[tokio::test]
async fn new_source_retires_old_source_until_worker_recreated() {
    let monitor = test_monitor();
    monitor
        .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
        .await;
    let old = monitor
        .begin_stream(test_report("worker:30000", "old", 1, WorkerMode::Plain))
        .unwrap();
    let _new = monitor
        .begin_stream(test_report("worker:30000", "new", 1, WorkerMode::Plain))
        .unwrap();
    assert!(monitor
        .apply_stream_report(
            &old,
            test_report("worker:30000", "old", 2, WorkerMode::Plain)
        )
        .is_err());
    assert!(monitor
        .begin_stream(test_report("worker:30000", "old", 3, WorkerMode::Plain))
        .is_err());
    monitor.stop_registrations().await;
}

/// Healthy reports with zero capacity retain diagnostics but are stale.
#[tokio::test]
async fn zero_capacity_healthy_report_is_locally_stale() {
    let monitor = test_monitor();
    monitor
        .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
        .await;
    let mut report = test_report("worker:30000", "source", 1, WorkerMode::Plain);
    report.ranks[0].num_running_reqs = 0;
    report.ranks[0].num_used_tokens = 0;
    report.ranks[0].max_running_requests = 0;
    report.ranks[0].max_total_num_tokens = 0;
    monitor.begin_stream(report).unwrap();
    let snapshot = monitor.snapshot();
    assert_eq!(snapshot.workers[0].freshness, Freshness::Stale);
    assert_eq!(snapshot.workers[0].ranks.len(), 1);
    assert_eq!(
        snapshot.workers[0].last_error.as_deref(),
        Some("engine reported a rank with zero token or request capacity")
    );
    monitor.stop_registrations().await;
}

/// Freshness expiration uses Router receipt time rather than engine time.
#[tokio::test]
async fn freshness_expires_from_router_receipt_time() {
    let monitor = test_monitor();
    monitor
        .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
        .await;
    monitor
        .begin_stream(test_report("worker:30000", "source", 1, WorkerMode::Plain))
        .unwrap();
    {
        let mut store = monitor.inner.store.write();
        store
            .workers
            .get_mut(&WorkerId("worker".to_string()))
            .unwrap()
            .report
            .as_mut()
            .unwrap()
            .received_at = SystemTime::now() - STALE_AFTER - Duration::from_millis(1);
    }
    assert_eq!(monitor.snapshot().workers[0].freshness, Freshness::Stale);
    monitor.stop_registrations().await;
}

/// An owned snapshot cannot change after a later report replaces the store.
#[tokio::test]
async fn captured_snapshot_is_immutable_across_updates() {
    let monitor = test_monitor();
    monitor
        .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
        .await;
    let binding = monitor
        .begin_stream(test_report("worker:30000", "source", 1, WorkerMode::Plain))
        .unwrap();
    let first = monitor.snapshot();
    monitor
        .apply_stream_report(
            &binding,
            test_report("worker:30000", "source", 2, WorkerMode::Plain),
        )
        .unwrap();
    let second = monitor.snapshot();
    assert_eq!(first.workers[0].sequence_id, Some(1));
    assert_eq!(second.workers[0].sequence_id, Some(2));
    assert!(second.version > first.version);
    monitor.stop_registrations().await;
}

/// Exercises actual HTTP registration, an ephemeral gRPC listener, stream
/// ingestion, and immutable snapshot publication without engine auth.
#[tokio::test]
async fn fake_engine_registration_and_grpc_report_form_complete_loop() {
    use axum::routing::post;
    use axum::{Json, Router};
    use tokio::sync::mpsc;

    let (registration_tx, mut registration_rx) = mpsc::channel(4);
    let app = Router::new().route(
        START_REPORTING_PATH,
        post(
            move |headers: axum::http::HeaderMap, Json(body): Json<serde_json::Value>| {
                let registration_tx = registration_tx.clone();
                async move {
                    registration_tx
                        .send((
                            body,
                            headers.contains_key(axum::http::header::AUTHORIZATION),
                        ))
                        .await
                        .unwrap();
                    axum::http::StatusCode::OK
                }
            },
        ),
    );
    let engine_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let engine_addr = engine_listener.local_addr().unwrap();
    let engine_server = tokio::spawn(async move {
        axum::serve(engine_listener, app).await.unwrap();
    });

    let config = LoadMonitorConfig {
        enabled: true,
        bind_host: "127.0.0.1".to_string(),
        bind_port: 0,
        report_ip: Some("127.0.0.1".to_string()),
    };
    let (monitor, grpc) = bind_and_serve(config).await.unwrap();
    let worker = Arc::new(Worker::new(WorkerSpec {
        id: WorkerId("worker".to_string()),
        url: format!("http://{engine_addr}"),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("model".to_string())],
        bootstrap_port: None,
    }));
    monitor.reconcile(vec![worker]).await;

    let (registration, has_authorization) =
        tokio::time::timeout(Duration::from_secs(3), registration_rx.recv())
            .await
            .unwrap()
            .unwrap();
    assert!(!has_authorization, "Router must not send a Bearer token");
    assert_eq!(registration["ip"], "127.0.0.1");
    assert_eq!(
        registration["port"].as_u64(),
        Some(grpc.local_addr().port() as u64)
    );
    assert_eq!(registration["report_interval_ms"].as_u64(), Some(1000));
    assert_eq!(registration["lease_ttl_ms"].as_u64(), Some(15000));

    let mut client = proto::load_monitor_service_client::LoadMonitorServiceClient::connect(
        format!("http://{}", grpc.local_addr()),
    )
    .await
    .unwrap();
    let report = test_report(
        &engine_addr.to_string(),
        "fake-engine",
        1,
        WorkerMode::Plain,
    );
    client
        .report(tokio_stream::iter(vec![report]))
        .await
        .unwrap();
    let snapshot = monitor.snapshot();
    assert_eq!(snapshot.workers[0].freshness, Freshness::Fresh);
    assert_eq!(
        snapshot.workers[0]
            .aggregate
            .as_ref()
            .unwrap()
            .total_requests,
        5
    );
    let renewal = tokio::time::timeout(Duration::from_secs(3), registration_rx.recv())
        .await
        .unwrap();
    assert!(renewal.is_some(), "Router must renew the reporting lease");

    grpc.shutdown(&monitor).await;
    engine_server.abort();
}

/// Retryable HTTP responses back off, terminal 4xx responses pause until
/// the next topology reconcile, and removal clears monitor state.
#[tokio::test]
async fn registration_retry_terminal_response_and_removal_reconcile() {
    use axum::routing::post;
    use axum::Router;
    use tokio::sync::mpsc;

    let attempts = Arc::new(AtomicUsize::new(0));
    let (attempt_tx, mut attempt_rx) = mpsc::channel(8);
    let attempts_for_handler = Arc::clone(&attempts);
    let app = Router::new().route(
        START_REPORTING_PATH,
        post(move || {
            let attempt_tx = attempt_tx.clone();
            let attempt = attempts_for_handler.fetch_add(1, Ordering::AcqRel) + 1;
            async move {
                attempt_tx.send(attempt).await.unwrap();
                match attempt {
                    1 => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    2 => axum::http::StatusCode::TOO_MANY_REQUESTS,
                    3 => axum::http::StatusCode::BAD_REQUEST,
                    _ => axum::http::StatusCode::OK,
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let engine_addr = listener.local_addr().unwrap();
    let engine_server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let monitor = LoadMonitor::new_enabled(
        LoadMonitorConfig {
            enabled: true,
            bind_host: "127.0.0.1".to_string(),
            bind_port: 0,
            report_ip: Some("127.0.0.1".to_string()),
        },
        3456,
    )
    .unwrap();
    let worker = Arc::new(Worker::new(WorkerSpec {
        id: WorkerId("worker".to_string()),
        url: format!("http://{engine_addr}"),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("model".to_string())],
        bootstrap_port: None,
    }));
    monitor.reconcile(vec![Arc::clone(&worker)]).await;

    for expected in 1..=3 {
        let actual = tokio::time::timeout(Duration::from_secs(3), attempt_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(actual, expected);
    }
    assert!(
        tokio::time::timeout(Duration::from_millis(1200), attempt_rx.recv())
            .await
            .is_err(),
        "terminal 4xx must pause registration until topology reconcile"
    );

    monitor.reconcile(vec![worker]).await;
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), attempt_rx.recv())
            .await
            .unwrap(),
        Some(4)
    );
    monitor.reconcile(Vec::new()).await;
    assert!(monitor.snapshot().workers.is_empty());

    monitor.stop_registrations().await;
    engine_server.abort();
}
