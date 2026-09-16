use super::{
    DEFAULT_GRPC_MAX_MESSAGE_SIZE, openai_status_code, resolve_max_message_size,
    terminal_error_status,
};
use crate::bridge::TerminalError;
use std::collections::HashMap;
use tonic::Code;

// The metadata wire test starts a real server that reads SGLANG_TONIC_PAYLOAD.
// Keep that read separate from the environment override test below.
static PAYLOAD_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[test]
fn openai_status_code_uses_forwarded_status_when_present() {
    let meta_info = HashMap::from([(String::from("status_code"), String::from("429"))]);
    assert_eq!(openai_status_code(&meta_info, 200), 429);
}

#[test]
fn openai_status_code_falls_back_when_missing_or_invalid() {
    assert_eq!(openai_status_code(&HashMap::new(), 200), 200);

    let meta_info = HashMap::from([(String::from("status_code"), String::from("not-an-int"))]);
    assert_eq!(openai_status_code(&meta_info, 200), 200);
}

#[test]
fn terminal_error_status_maps_channel_full_to_resource_exhausted() {
    let status = terminal_error_status(TerminalError::ChannelFull {
        rid: "rid".to_string(),
    });

    assert_eq!(status.code(), Code::ResourceExhausted);
}

#[test]
fn terminal_error_status_maps_abort_to_cancelled() {
    let status = terminal_error_status(TerminalError::Aborted {
        rid: "rid".to_string(),
    });

    assert_eq!(status.code(), Code::Cancelled);
}

// SAFETY: env vars are process-global; bundle all SGLANG_TONIC_PAYLOAD cases into one
// serial test so they don't race each other under `cargo test`'s default parallelism.
#[test]
fn resolve_max_message_size_honors_env_var() {
    let _guard = PAYLOAD_ENV_LOCK.blocking_lock();
    const VAR: &str = "SGLANG_TONIC_PAYLOAD";

    // Unset → default.
    // SAFETY: single-threaded test mutating process env (see note above).
    unsafe {
        std::env::remove_var(VAR);
    }
    assert_eq!(resolve_max_message_size(), DEFAULT_GRPC_MAX_MESSAGE_SIZE);

    // Valid override → honored verbatim.
    unsafe {
        std::env::set_var(VAR, "1048576");
    }
    assert_eq!(resolve_max_message_size(), 1_048_576);

    // Invalid string → warn + fall back to default.
    unsafe {
        std::env::set_var(VAR, "not-a-number");
    }
    assert_eq!(resolve_max_message_size(), DEFAULT_GRPC_MAX_MESSAGE_SIZE);

    // Zero → treated as invalid, fall back to default.
    unsafe {
        std::env::set_var(VAR, "0");
    }
    assert_eq!(resolve_max_message_size(), DEFAULT_GRPC_MAX_MESSAGE_SIZE);

    unsafe {
        std::env::remove_var(VAR);
    }
}

#[tokio::test]
async fn follower_metadata_server_exposes_only_server_info_and_shuts_down() {
    use crate::proto::{GetServerInfoRequest, GetServerInfoResponse};
    use tonic::codec::ProstCodec;

    let _guard = PAYLOAD_ENV_LOCK.lock().await;

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    listener.set_nonblocking(true).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let server_info_json = serde_json::json!({
        "node_rank": 1,
        "nnodes": 2,
        "dp_size": 8,
        "kv_event_sources": [{
            "dp_rank": 4,
            "endpoint": "tcp://127.0.0.1:5557",
            "topic": "kv-events",
            "block_size": 16
        }]
    })
    .to_string();
    // No Python interpreter, RuntimeHandle, tokenizer, or PyBridge is created.
    let mut handle = crate::start_server_thread(
        runtime,
        listener,
        super::MetadataService {
            server_info_json: server_info_json.clone(),
        },
    )
    .unwrap();
    assert!(handle.is_alive());

    let channel = tonic::transport::Endpoint::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect_timeout(std::time::Duration::from_secs(5))
        .timeout(std::time::Duration::from_secs(5))
        .connect()
        .await
        .unwrap();
    let mut client = tonic::client::Grpc::new(channel);
    client.ready().await.unwrap();
    let response: tonic::Response<GetServerInfoResponse> = client
        .unary(
            tonic::Request::new(GetServerInfoRequest {}),
            tonic::codegen::http::uri::PathAndQuery::from_static(
                "/sglang.runtime.v1.SglangService/GetServerInfo",
            ),
            ProstCodec::default(),
        )
        .await
        .unwrap();
    assert_eq!(response.into_inner().json_info, server_info_json);

    // These cover generation (including streaming), model queries, health,
    // and controls. The follower must never advertise an inference frontend.
    for method in [
        "Generate",
        "ChatComplete",
        "GetModelInfo",
        "HealthCheck",
        "FlushCache",
        "Abort",
    ] {
        client.ready().await.unwrap();
        let result: Result<tonic::Response<GetServerInfoResponse>, _> = client
            .unary(
                tonic::Request::new(GetServerInfoRequest {}),
                format!("/sglang.runtime.v1.SglangService/{method}")
                    .parse()
                    .unwrap(),
                ProstCodec::default(),
            )
            .await;
        assert_eq!(result.unwrap_err().code(), Code::Unimplemented, "{method}");
    }
    drop(client);
    // Keep the client's runtime driving connection teardown while the server
    // joins its own thread during graceful shutdown.
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        tokio::task::spawn_blocking(move || {
            handle.shutdown();
            assert!(!handle.is_alive());
            handle.shutdown(); // Existing handle shutdown remains idempotent.
        }),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(std::net::TcpStream::connect(addr).is_err());
}
