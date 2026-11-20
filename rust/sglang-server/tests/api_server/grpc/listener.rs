//! The gRPC listener beside the HTTP one: it serves generation, shares the
//! HiCache state, keeps its message limit, and closes with HTTP.

use sglang_api_types::api::v1 as proto;
use sglang_server_test::{ApiKeys, SchedulerlessServer, free_loopback_addrs};

use crate::common::{
    connect_grpc, grpc_ids_request, next_scheduler_rid, push_terminal, take_startup_freeze_gc,
};

/// End to end over the configured socket: the generated api.v1 client
/// reaches the existing scheduler ring, and an existing scheduler frame is
/// translated back into a protobuf frame. Shutdown then closes both
/// transport listeners.
#[test]
fn configured_grpc_serves_generation_and_closes_with_http() {
    use std::time::Duration;

    let [http_addr, grpc_addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(http_addr)
        .grpc_addr(grpc_addr)
        .start()
        .expect("start runtime");
    take_startup_freeze_gc(&rt);
    assert!(std::net::TcpStream::connect(http_addr).is_ok());

    let client_runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut stream = client_runtime.block_on(async {
        let mut client = connect_grpc(grpc_addr).await;
        tokio::time::timeout(
            Duration::from_secs(5),
            client.generate(grpc_ids_request(&[1, 2, 3], "network-generation")),
        )
        .await
        .expect("timed out starting Generate RPC")
        .expect("Generate RPC")
        .into_inner()
    });

    let request = rt
        .next_request(Duration::from_secs(2))
        .expect("gRPC request did not reach the scheduler ring");
    assert_eq!(request.buffer_shape("input_ids"), Some(&[3][..]));
    let rid = request.rid().expect("request rid");
    assert!(push_terminal(&rt, &rid, &[9, 10]));

    let item = client_runtime
        .block_on(async { tokio::time::timeout(Duration::from_secs(2), stream.message()).await })
        .expect("timed out waiting for gRPC response")
        .expect("gRPC stream error")
        .expect("gRPC stream ended without a response");
    let Some(proto::generate_stream_item::Item::Frame(response)) = item.item else {
        panic!("expected a frame, got {item:?}");
    };
    assert_eq!(response.output_ids.map(|ids| ids.ids), Some(vec![9, 10]));
    let meta_info = response.meta_info.expect("frame meta_info");
    assert_eq!(meta_info.id, "network-generation");
    assert!(meta_info.finish_reason.is_some());

    rt.shutdown();
    assert!(std::net::TcpStream::connect(http_addr).is_err());
    assert!(std::net::TcpStream::connect(grpc_addr).is_err());
}

/// An HTTP attach reaches the scheduler ring; after a successful reply,
/// gRPC returns the updated HiCache configuration, verifying that both
/// transports share the same state.
#[test]
fn hicache_state_is_shared_between_http_and_grpc() {
    use std::io::{Read, Write};
    use std::time::Duration;

    let [http_addr, grpc_addr] = free_loopback_addrs();
    // Attach needs a configured admin key.
    let rt = SchedulerlessServer::builder(http_addr)
        .grpc_addr(grpc_addr)
        .server_args(|args| args.api_keys = ApiKeys::new(None, Some("sk-admin".into())))
        .start()
        .expect("start runtime");
    take_startup_freeze_gc(&rt);
    let request = std::thread::spawn(move || {
        let mut conn = std::net::TcpStream::connect(http_addr).unwrap();
        conn.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let body = r#"{"hicache_storage_backend":"file","hicache_write_policy":"write_through"}"#;
        write!(conn,
            "PUT /hicache/storage-backend HTTP/1.1\r\nHost: t\r\nAuthorization: Bearer sk-admin\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()
        ).unwrap();
        let mut response = String::new();
        conn.read_to_string(&mut response).unwrap();
        response
    });
    let rid = next_scheduler_rid(&rt);
    let result = serde_json::json!({"success": true, "message": "attached"});
    rt.push_control_result(&rid, &result).unwrap();
    let response = request.join().unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    let (_, body) = response.split_once("\r\n\r\n").unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(body).unwrap(),
        serde_json::json!({"success": true, "message": "attached"})
    );

    let client_runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let status = client_runtime.block_on(async {
        let mut client = connect_grpc(grpc_addr).await;
        tokio::time::timeout(
            Duration::from_secs(5),
            client.get_hi_cache_storage(proto::GetHiCacheStorageRequest {}),
        )
        .await
        .unwrap()
        .unwrap()
        .into_inner()
    });
    assert_eq!(
        status,
        proto::HiCacheStatus {
            hicache_storage_backend: Some("file".into()),
            hicache_write_policy: Some("write_through".into()),
            ..Default::default()
        }
    );
    rt.shutdown();
}

/// The Python-backed gRPC listener accepts bodies above Tonic's 4 MiB
/// default. Keep that transport policy on the Rust frontend; reaching
/// adapter validation proves the body decoded.
#[test]
fn grpc_listener_preserves_existing_large_message_limit() {
    use tonic::Code;

    let [http_addr, grpc_addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(http_addr)
        .grpc_addr(grpc_addr)
        .start()
        .expect("start runtime");
    let client_runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let error = client_runtime.block_on(async {
        let client = connect_grpc(grpc_addr).await;
        let mut client = client.max_encoding_message_size(64 * 1024 * 1024);
        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            client.generate(proto::GenerateRequest {
                // `text` and `input_ids` together is a normalization
                // error: an INVALID_ARGUMENT status proves the >4 MiB
                // protobuf reached the adapter instead of Tonic rejecting
                // it on size.
                text: Some(proto::StringOrList {
                    value: Some(proto::string_or_list::Value::One(
                        "x".repeat(5 * 1024 * 1024),
                    )),
                }),
                ..grpc_ids_request(&[1], "large")
            }),
        )
        .await
        .expect("timed out sending large Generate RPC")
        .expect_err("text and input_ids together must fail")
    });
    assert_eq!(error.code(), Code::InvalidArgument);

    rt.shutdown();
    assert!(std::net::TcpStream::connect(grpc_addr).is_err());
}
