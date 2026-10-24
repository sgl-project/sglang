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

#[tokio::test]
async fn api_and_admin_keys_match_http_permissions_before_dispatch() {
    use sglang_api_types::runtime::v1::{GetServerInfoRequest, GetServerInfoResponse};
    use tonic_prost::ProstCodec;

    let _guard = PAYLOAD_ENV_LOCK.lock().await;
    for (api_key, admin_api_key) in [
        (None, None),
        (Some("user-key"), None),
        (None, Some("admin-key")),
        (Some("user-key"), Some("admin-key")),
    ] {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let shutdown = std::sync::Arc::new(tokio::sync::Notify::new());
        let server_shutdown = shutdown.clone();
        let server = tokio::spawn(super::serve_grpc(
            listener,
            super::MetadataService {
                server_info_json: "{}".into(),
            },
            server_shutdown,
            None,
            crate::auth::ApiKeyAuth::new(
                api_key.map(str::to_owned),
                admin_api_key.map(str::to_owned),
            ),
        ));
        let channel = tonic::transport::Endpoint::from_shared(format!("http://{addr}"))
            .unwrap()
            .timeout(std::time::Duration::from_secs(5))
            .connect()
            .await
            .unwrap();
        let mut client = tonic::client::Grpc::new(channel);
        for (method, streaming, admin) in [
            ("GetServerInfo", false, false),
            ("GetModelInfo", false, false),
            ("HealthCheck", false, false),
            ("ListModels", false, false),
            ("GetLoad", false, false),
            ("TextEmbed", false, false),
            ("Embed", false, false),
            ("Classify", false, false),
            ("Tokenize", false, false),
            ("Detokenize", false, false),
            ("OpenAIEmbed", false, false),
            ("OpenAIClassify", false, false),
            ("Score", false, false),
            ("Rerank", false, false),
            ("PauseGeneration", false, true),
            ("ContinueGeneration", false, true),
            ("Abort", false, true),
            ("FlushCache", false, true),
            ("StartProfile", false, true),
            ("StopProfile", false, true),
            ("UpdateWeightsFromDisk", false, true),
            ("Generate", true, false),
            ("TextGenerate", true, false),
            ("ChatComplete", true, false),
            ("Complete", true, false),
            ("WatchEngineState", true, false),
        ] {
            for token in [None, Some("wrong"), Some("user-key"), Some("admin-key")] {
                let mut request = tonic::Request::new(GetServerInfoRequest {});
                if let Some(token) = token {
                    request
                        .metadata_mut()
                        .insert("authorization", format!("Bearer {token}").parse().unwrap());
                }
                let path = format!("/sglang.runtime.v1.SglangService/{method}")
                    .parse()
                    .unwrap();
                client.ready().await.unwrap();
                let result = if streaming {
                    client
                        .server_streaming::<_, GetServerInfoResponse, _>(
                            request,
                            path,
                            ProstCodec::default(),
                        )
                        .await
                        .map(|_| ())
                } else {
                    client
                        .unary::<_, GetServerInfoResponse, _>(request, path, ProstCodec::default())
                        .await
                        .map(|_| ())
                };
                let expected_key = if method == "HealthCheck" {
                    None
                } else if admin {
                    admin_api_key.or(api_key)
                } else {
                    api_key
                };
                if expected_key.is_some() && token != expected_key {
                    assert_eq!(
                        result.unwrap_err().code(),
                        Code::Unauthenticated,
                        "{method}"
                    );
                } else if method == "GetServerInfo" {
                    result.unwrap();
                } else {
                    // Authorized calls reach the metadata-only service's default handler.
                    assert_eq!(result.unwrap_err().code(), Code::Unimplemented, "{method}");
                }
            }
        }
        shutdown.notify_one();
        tokio::time::timeout(std::time::Duration::from_secs(5), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }
}

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
    use sglang_api_types::runtime::v1::{GetServerInfoRequest, GetServerInfoResponse};
    use tonic_prost::ProstCodec;

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
    // Exercise the metadata service without the Python-facing startup wrapper,
    // which requires libpython when linked into a standalone test executable.
    let shutdown = std::sync::Arc::new(tokio::sync::Notify::new());
    let server_shutdown = shutdown.clone();
    let service = super::MetadataService {
        server_info_json: server_info_json.clone(),
    };
    let handle = std::thread::spawn(move || {
        runtime
            .block_on(super::serve_grpc(
                listener,
                service,
                server_shutdown,
                None,
                crate::auth::ApiKeyAuth::new(None, None),
            ))
            .unwrap();
    });
    assert!(!handle.is_finished());

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
    // Keep the client's runtime driving connection teardown while joining the
    // server thread during graceful shutdown.
    shutdown.notify_one();
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        tokio::task::spawn_blocking(move || {
            handle.join().unwrap();
        }),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(std::net::TcpStream::connect(addr).is_err());
}
