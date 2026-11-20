//! The frontend's life as its launcher sees it from outside: what fails a
//! start, what a graceful or forced drain serves and closes, and that a
//! shutdown returns with work still in flight.

use std::sync::Arc;

use sglang_api_types::api::v1 as proto;
use sglang_server::{
    DisaggregationMode, MediaError, MmData, MmProcessOutput, MmProcessor, TokenIds,
};
use sglang_server_test::{SchedulerlessServer, free_loopback_addrs};

use crate::common::{
    connect_grpc, grpc_ids_request, next_scheduler_rid, push_terminal, push_tokens, read_response,
    send_http, take_startup_freeze_gc, wait_drain_complete,
};

/// A registered `/server_info` state key that names a field it already
/// reports, or one registered twice, fails startup before anything runs.
#[test]
fn start_rejects_colliding_server_info_state_keys() {
    for (keys, problem) in [
        (
            ["package_state", "memory_usage"],
            r#""memory_usage" names a field /server_info already reports"#,
        ),
        (
            ["package_state", "package_state"],
            r#""package_state" is registered twice"#,
        ),
    ] {
        let [http_addr] = free_loopback_addrs();
        let error = SchedulerlessServer::builder(http_addr)
            .server_info_state_keys(keys.map(String::from).to_vec())
            .start()
            .err()
            .expect("startup must fail");
        assert!(error.contains(problem), "{error}");
    }
}

/// The graceful drain, as Python's SIGTERM handling: health answers 503
/// at once without probing the scheduler, an in-flight stream runs to its
/// `[DONE]`, a request arriving during the drain is still admitted and
/// served, and the listener closes only once both finished.
#[test]
fn drain_answers_503_and_serves_in_flight_requests_before_closing() {
    let [addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(addr)
        .start()
        .expect("start runtime");
    take_startup_freeze_gc(&rt);

    let streaming = send_http(
        addr,
        "POST",
        "/generate",
        r#"{"input_ids":[1,2,3],"stream":true,"sampling_params":{"max_new_tokens":8}}"#,
    );
    let streaming_rid = next_scheduler_rid(&rt);
    assert!(push_tokens(&rt, &streaming_rid, &[7]));

    // A router's idle keep-alive connection: answered, left open.
    let mut keep_alive = std::net::TcpStream::connect(addr).expect("connect");
    keep_alive
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    std::io::Write::write_all(
        &mut keep_alive,
        b"GET /model_info HTTP/1.1\r\nHost: t\r\n\r\n",
    )
    .unwrap();

    rt.begin_drain(false);
    for path in ["/health", "/health_generate"] {
        let response = read_response(send_http(addr, "GET", path, ""));
        assert!(response.starts_with("HTTP/1.1 503"), "{path}: {response}");
    }
    assert!(
        rt.next_request(std::time::Duration::from_millis(200))
            .is_none(),
        "a draining server must not send a health probe"
    );

    // Admitted during the drain, as Python admits it.
    let late = send_http(
        addr,
        "POST",
        "/generate",
        r#"{"input_ids":[4,5],"sampling_params":{"max_new_tokens":8}}"#,
    );
    let late_rid = next_scheduler_rid(&rt);

    assert!(push_terminal(&rt, &streaming_rid, &[8]));
    let stream = read_response(streaming);
    assert!(stream.starts_with("HTTP/1.1 200"), "{stream}");
    assert!(stream.contains("data: [DONE]"), "stream cut: {stream}");

    // The late request still holds the drain open.
    std::thread::sleep(std::time::Duration::from_millis(200));
    assert!(!rt.drain_complete());
    assert!(std::net::TcpStream::connect(addr).is_ok());

    assert!(push_terminal(&rt, &late_rid, &[9]));
    let response = read_response(late);
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert!(response.contains("[9]"), "{response}");

    wait_drain_complete(&rt);
    assert!(
        std::net::TcpStream::connect(addr).is_err(),
        "listener still open after the drain completed"
    );
    // The idle keep-alive connection did not hold the close open.
    let keep_alive = read_response(keep_alive);
    assert!(keep_alive.starts_with("HTTP/1.1 200"), "{keep_alive}");
    rt.shutdown();
}

/// A forced drain (`SGLANG_FORCE_SHUTDOWN`, or a failed last health probe)
/// does not wait: the listener closes with a request still in flight, and
/// that request is cut.
#[test]
fn forced_drain_closes_without_waiting() {
    let [addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(addr)
        .start()
        .expect("start runtime");
    take_startup_freeze_gc(&rt);
    let in_flight = send_http(
        addr,
        "POST",
        "/generate",
        r#"{"input_ids":[1,2,3],"stream":true,"sampling_params":{"max_new_tokens":8}}"#,
    );
    next_scheduler_rid(&rt);

    rt.begin_drain(true);
    wait_drain_complete(&rt);
    assert!(std::net::TcpStream::connect(addr).is_err());
    assert!(!read_response(in_flight).contains("[DONE]"));
    rt.shutdown();
}

/// A response that never finishes (here a control request no scheduler
/// answers, which the drain does not count) holds the graceful close
/// only for the graceful-close timeout, so the scheduler's shutdown still
/// goes out.
#[test]
fn graceful_close_is_bounded() {
    use std::time::{Duration, Instant};

    let timeout = Duration::from_millis(500);
    let [addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(addr)
        .rust_server_args(|args| args.graceful_close_timeout = timeout)
        .start()
        .expect("start runtime");
    take_startup_freeze_gc(&rt);
    let stuck = send_http(addr, "GET", "/server_info", "");
    assert!(rt.next_request(Duration::from_secs(5)).is_some());

    let begun = Instant::now();
    rt.begin_drain(false);
    std::thread::sleep(Duration::from_millis(200));
    assert!(
        !rt.drain_complete(),
        "the open connection was not waited for"
    );
    while !rt.drain_complete() {
        assert!(
            begun.elapsed() < timeout + Duration::from_secs(3),
            "the graceful close was not bounded"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    rt.shutdown();
    drop(stuck);
}

/// gRPC drains with HTTP: `HealthCheck` is UNAVAILABLE, an in-flight
/// `Generate` stream completes, and both listeners close afterwards.
#[test]
fn grpc_drains_with_http() {
    use std::time::Duration;

    let [http_addr, grpc_addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(http_addr)
        .grpc_addr(grpc_addr)
        .start()
        .expect("start runtime");
    take_startup_freeze_gc(&rt);
    let client_runtime = // Multi-threaded, so the client's connection task keeps answering
    // the server's HTTP/2 shutdown handshake between `block_on`s.
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let (mut client, mut stream) = client_runtime.block_on(async {
        let mut client = connect_grpc(grpc_addr).await;
        let stream = client
            .generate(grpc_ids_request(&[1, 2, 3], "draining"))
            .await
            .expect("Generate RPC")
            .into_inner();
        (client, stream)
    });
    let rid = next_scheduler_rid(&rt);

    rt.begin_drain(false);
    let health = client_runtime
        .block_on(client.health_check(proto::HealthCheckRequest {}))
        .expect_err("a draining server must not report healthy");
    assert_eq!(health.code(), tonic::Code::Unavailable);
    assert!(!rt.drain_complete());

    assert!(push_terminal(&rt, &rid, &[9, 10]));
    let item = client_runtime
        .block_on(async { tokio::time::timeout(Duration::from_secs(5), stream.message()).await })
        .expect("timed out waiting for the gRPC response")
        .expect("gRPC stream error")
        .expect("gRPC stream ended without a response");
    assert!(matches!(
        item.item,
        Some(proto::generate_stream_item::Item::Frame(_))
    ));
    // The client keeps its connection open: the graceful close must not
    // wait for an idle HTTP/2 connection.
    wait_drain_complete(&rt);
    assert!(std::net::TcpStream::connect(http_addr).is_err());
    assert!(std::net::TcpStream::connect(grpc_addr).is_err());
    drop((client, stream));
    rt.shutdown();
}

/// Regression: `request_shutdown` must actually stop the API server — it joins
/// the api thread once the listener closes, so the port stops accepting.
/// (Previously it set an unread flag and the port kept accepting.)
#[test]
fn request_shutdown_closes_listener() {
    let [addr] = free_loopback_addrs();
    // Bind is synchronous in `start`, so the port is already accepting.
    let rt = SchedulerlessServer::builder(addr)
        .start()
        .expect("start runtime");
    assert!(
        std::net::TcpStream::connect(addr).is_ok(),
        "server not listening on {addr} after start returned",
    );

    // Joins the api thread; the listener is closed by the time it returns.
    rt.shutdown();

    assert!(
        std::net::TcpStream::connect(addr).is_err(),
        "port still accepting connections after shutdown",
    );
}

/// A multimodal model admits no generation until its MM pool's processor
/// is installed, because admission runs the processor's media hooks; a
/// request sent earlier waits and is admitted once the pool starts. The
/// control routes serve from the start: `/health` answers 503, and the PD
/// bootstrap registry accepts a rank's registration, so other ranks can
/// register while the processor is still being built. A text model admits
/// at once, and a runtime shut down with a request still waiting exits.
#[test]
fn multimodal_admission_waits_for_the_mm_pool() {
    use std::io::{Read, Write};
    use std::time::{Duration, Instant};

    struct NoMedia;
    impl MmProcessor for NoMedia {
        fn process(
            &self,
            _input_ids: TokenIds,
            _mm: MmData,
        ) -> Result<MmProcessOutput, MediaError> {
            unreachable!("no request carries media")
        }
    }
    let start_prefill = |is_multimodal: bool| {
        let [addr] = free_loopback_addrs();
        let rt = SchedulerlessServer::builder(addr)
            .server_args(|args| {
                args.disaggregation_mode = DisaggregationMode::Prefill;
                args.model_config.is_multimodal = is_multimodal;
            })
            .start()
            .expect("start runtime");
        take_startup_freeze_gc(&rt);
        (rt, addr)
    };
    let send = |addr, method: &str, path: &str, body: &str| {
        let mut conn = std::net::TcpStream::connect(addr).expect("connect");
        write!(
            conn,
            "{method} {path} HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        conn
    };
    let response = |mut conn: std::net::TcpStream| {
        conn.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut response = String::new();
        conn.read_to_string(&mut response).unwrap();
        response
    };
    let generate = r#"{"input_ids":[1,2,3],"sampling_params":{"max_new_tokens":8}}"#;
    let registration = serde_json::json!({
        "attn_tp_size": 2, "attn_tp_rank": 1,
        "attn_cp_size": 1, "attn_cp_rank": 0,
        "attn_dp_size": 1, "attn_dp_rank": 0,
        "pp_size": 1, "pp_rank": 0,
        "system_dp_size": 1, "system_dp_rank": 0,
        "rank_ip": "10.0.0.1", "rank_port": 17000, "page_size": 64,
    })
    .to_string();

    let (rt, addr) = start_prefill(false);
    let _waiting = send(addr, "POST", "/generate", generate);
    assert!(
        rt.next_request(Duration::from_secs(2)).is_some(),
        "a text model did not admit the request",
    );
    rt.shutdown();

    let (rt, addr) = start_prefill(true);
    let waiting = send(addr, "POST", "/generate", generate);
    let registered = response(send(addr, "PUT", "/route", &registration));
    assert!(registered.starts_with("HTTP/1.1 200"), "{registered}");
    let health = response(send(addr, "GET", "/health", ""));
    assert!(health.starts_with("HTTP/1.1 503"), "{health}");
    assert!(
        rt.next_request(Duration::from_millis(300)).is_none(),
        "admitted before the MM pool ran",
    );
    rt.start_mm_workers(Arc::new(NoMedia), 1);
    assert!(
        rt.next_request(Duration::from_secs(2)).is_some(),
        "not admitted once the MM pool ran",
    );
    rt.shutdown();
    drop(waiting);

    let (rt, addr) = start_prefill(true);
    let _waiting = send(addr, "POST", "/generate", generate);
    std::thread::sleep(Duration::from_millis(300));
    let started = Instant::now();
    rt.shutdown();
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "shutdown took {:?} with a request waiting for the MM pool",
        started.elapsed(),
    );
    assert!(
        std::net::TcpStream::connect(addr).is_err(),
        "port still accepting connections after shutdown",
    );
}

/// Regression: shutdown must return promptly even with an in-flight
/// `/generate`.
#[test]
fn shutdown_returns_with_in_flight_request() {
    use std::io::Write;
    use std::time::{Duration, Instant};

    let [addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(addr)
        .start()
        .expect("start runtime");

    // Fire a request that will block (already-tokenized → valid → pushed to the
    // ring, then the handler awaits decode frames that never arrive).
    let mut conn = std::net::TcpStream::connect(addr).expect("connect");
    let body = r#"{"input_ids":[1,2,3],"stream":false,"sampling_params":{"max_new_tokens":8}}"#;
    let req = format!(
        "POST /generate HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    conn.write_all(req.as_bytes()).unwrap();
    conn.flush().unwrap();
    std::thread::sleep(Duration::from_millis(300)); // reach the blocked state

    let t = Instant::now();
    rt.shutdown();
    let elapsed = t.elapsed();
    assert!(
        elapsed < Duration::from_secs(3),
        "shutdown took {elapsed:?} with an in-flight request (deadlock?)",
    );
    drop(conn);
}

/// Regression: a port conflict must fail `start` (so the scheduler doesn't
/// advertise ready), not return an `Ok` runtime whose listener never binds.
#[test]
fn start_fails_on_port_conflict() {
    // Hold the port so the runtime's bind conflicts (EADDRINUSE).
    let hog = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = hog.local_addr().unwrap();

    let err = match SchedulerlessServer::builder(addr).start() {
        Ok(_) => panic!("bind conflict must fail startup, got Ok"),
        Err(e) => e,
    };
    assert!(err.contains("bind"), "error should mention bind: {err}");
}

/// A configured gRPC bind is part of startup, not a best-effort side task:
/// failure returns synchronously and releases the HTTP listener acquired
/// immediately before it.
#[test]
fn grpc_port_conflict_fails_startup_without_leaking_http_listener() {
    let grpc_hog = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let grpc_addr = grpc_hog.local_addr().unwrap();
    // Choose HTTP while the gRPC port remains occupied, guaranteeing that
    // the two addresses differ.
    let http_probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let http_addr = http_probe.local_addr().unwrap();
    drop(http_probe);

    let error = match SchedulerlessServer::builder(http_addr)
        .grpc_addr(grpc_addr)
        .start()
    {
        Ok(_) => panic!("gRPC bind conflict must fail startup"),
        Err(error) => error,
    };
    assert!(error.contains("gRPC listener"), "unexpected error: {error}");
    assert!(error.contains(&grpc_addr.to_string()));

    let rebound = std::net::TcpListener::bind(http_addr)
        .expect("failed startup must release its pre-bound HTTP listener");
    drop(rebound);
}

/// Tonic's unbounded graceful-drain API waits for open HTTP/2 connections.
/// Keep a generated response stream live to ensure the frontend's shared
/// shutdown remains bounded and cancels it instead.
#[test]
fn grpc_shutdown_returns_promptly_with_in_flight_generation() {
    use std::time::{Duration, Instant};

    let [http_addr, grpc_addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(http_addr)
        .grpc_addr(grpc_addr)
        .start()
        .expect("start runtime");
    take_startup_freeze_gc(&rt);
    let client_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let (_client, _stream) = client_runtime.block_on(async {
        let mut client = connect_grpc(grpc_addr).await;
        let stream = tokio::time::timeout(
            Duration::from_secs(5),
            client.generate(grpc_ids_request(&[1, 2, 3], "in-flight")),
        )
        .await
        .expect("timed out starting Generate RPC")
        .expect("Generate RPC")
        .into_inner();
        (client, stream)
    });
    assert!(rt.next_request(Duration::from_secs(2)).is_some());

    let started = Instant::now();
    rt.shutdown();
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "shutdown waited for an open gRPC stream: {:?}",
        started.elapsed()
    );
    assert!(std::net::TcpStream::connect(http_addr).is_err());
    assert!(std::net::TcpStream::connect(grpc_addr).is_err());
}
