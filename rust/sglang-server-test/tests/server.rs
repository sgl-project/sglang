//! The crate as its callers use it: a served frontend over real sockets,
//! with the test standing in for the scheduler.

use std::io::{Read, Write};
use std::net::SocketAddr;
use std::time::Duration;

use sglang_server_test::{ApiKeys, SchedulerlessServer, free_loopback_addrs};

const WAIT: Duration = Duration::from_secs(5);

/// Send one HTTP/1.1 request on a fresh connection and return the response.
fn http(addr: SocketAddr, method: &str, path: &str, headers: &str, body: &str) -> String {
    let mut conn = std::net::TcpStream::connect(addr).expect("connect");
    conn.set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    write!(
        conn,
        "{method} {path} HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n{headers}\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut response = String::new();
    conn.read_to_string(&mut response).expect("read response");
    response
}

fn take_freeze_gc(server: &SchedulerlessServer) {
    let first = server.next_request(WAIT).expect("no message after startup");
    assert_eq!(first.kind().as_deref(), Some("FreezeGCReq"));
}

/// A generation reaches the ring with its token buffer, and the output the
/// test pushes for it is the HTTP response.
#[test]
fn generation_round_trips_through_the_ring() {
    let [addr] = free_loopback_addrs();
    let server = SchedulerlessServer::builder(addr).start().expect("start");
    take_freeze_gc(&server);

    let client =
        std::thread::spawn(move || http(addr, "POST", "/generate", "", r#"{"input_ids":[1,2,3]}"#));
    let request = server.next_request(WAIT).expect("no generation");
    assert_eq!(request.kind().as_deref(), Some("TokenizedGenerateReqInput"));
    assert_eq!(request.buffer_shape("input_ids"), Some(&[3][..]));
    let rid = request.rid().expect("rid");
    let stop = serde_json::json!({"type": "stop"});
    server.push_output(&rid, &[9, 10], Some(stop)).unwrap();

    let response = client.join().unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert!(response.contains(r#""output_ids":[9,10]"#), "{response}");
    server.shutdown();
    assert!(std::net::TcpStream::connect(addr).is_err());
}

/// A scheduler error for a generation fails its HTTP request.
#[test]
fn scheduler_error_fails_the_request() {
    let [addr] = free_loopback_addrs();
    let server = SchedulerlessServer::builder(addr).start().expect("start");
    take_freeze_gc(&server);

    let client =
        std::thread::spawn(move || http(addr, "POST", "/generate", "", r#"{"input_ids":[1]}"#));
    let rid = server
        .next_request(WAIT)
        .and_then(|r| r.rid())
        .expect("rid");
    server.push_error(&rid, "scheduler exploded").unwrap();

    let response = client.join().unwrap();
    assert!(!response.starts_with("HTTP/1.1 200"), "{response}");
    assert!(response.contains("scheduler exploded"), "{response}");
}

/// Launch arguments set through the builder take effect, and a control
/// request is answered with the result the test pushes.
#[test]
fn control_request_is_answered_from_the_ring() {
    let [addr] = free_loopback_addrs();
    let server = SchedulerlessServer::builder(addr)
        .server_args(|args| args.api_keys = ApiKeys::new(None, Some("sk-admin".into())))
        .start()
        .expect("start");
    take_freeze_gc(&server);

    let body = r#"{"hicache_storage_backend":"file","hicache_write_policy":"write_through"}"#;
    let refused = http(addr, "PUT", "/hicache/storage-backend", "", body);
    assert!(!refused.starts_with("HTTP/1.1 200"), "{refused}");

    let client = std::thread::spawn(move || {
        let key = "Authorization: Bearer sk-admin\r\n";
        http(addr, "PUT", "/hicache/storage-backend", key, body)
    });
    let rid = server
        .next_request(WAIT)
        .and_then(|r| r.rid())
        .expect("rid");
    let result = serde_json::json!({"success": true, "message": "attached"});
    server.push_control_result(&rid, &result).unwrap();

    let response = client.join().unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    let (_, body) = response.split_once("\r\n\r\n").unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(body).unwrap(),
        result
    );
}

/// A graceful drain with nothing in flight ends and closes the listener.
#[test]
fn drain_ends_and_closes_the_listener() {
    let [addr, grpc_addr] = free_loopback_addrs();
    let server = SchedulerlessServer::builder(addr)
        .grpc_addr(grpc_addr)
        .start()
        .expect("start");
    assert!(std::net::TcpStream::connect(grpc_addr).is_ok());

    server.begin_drain(false);
    let deadline = std::time::Instant::now() + WAIT;
    while !server.drain_complete() {
        assert!(std::time::Instant::now() < deadline, "drain did not end");
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(std::net::TcpStream::connect(addr).is_err());
}
