use super::{
    DEFAULT_GRPC_MAX_MESSAGE_SIZE, openai_status_code, reflection_service,
    resolve_max_message_size, terminal_error_status,
};
use crate::bridge::TerminalError;
use std::collections::{BTreeSet, HashMap};
use std::net::SocketAddr;
use tokio::sync::oneshot;
use tokio_stream::{StreamExt, wrappers::TcpListenerStream};
use tonic::transport::{Endpoint, Server};
use tonic::{Code, Request};
use tonic_reflection::pb::v1::{
    ServerReflectionRequest, ServiceResponse, server_reflection_client::ServerReflectionClient,
    server_reflection_request::MessageRequest, server_reflection_response::MessageResponse,
};

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
async fn reflection_service_advertises_sglang_service() {
    let services = make_reflection_request(ServerReflectionRequest {
        host: String::new(),
        message_request: Some(MessageRequest::ListServices(String::new())),
    })
    .await;

    let MessageResponse::ListServicesResponse(services) = services else {
        panic!("expected ListServicesResponse");
    };
    let names = service_names(services.service);

    assert!(names.contains("grpc.reflection.v1.ServerReflection"));
    assert!(names.contains("sglang.runtime.v1.SglangService"));

    let descriptor = make_reflection_request(ServerReflectionRequest {
        host: String::new(),
        message_request: Some(MessageRequest::FileContainingSymbol(String::from(
            "sglang.runtime.v1.SglangService",
        ))),
    })
    .await;

    let MessageResponse::FileDescriptorResponse(descriptor) = descriptor else {
        panic!("expected FileDescriptorResponse");
    };
    assert!(!descriptor.file_descriptor_proto.is_empty());
}

fn service_names(services: Vec<ServiceResponse>) -> BTreeSet<String> {
    services.into_iter().map(|service| service.name).collect()
}

async fn make_reflection_request(request: ServerReflectionRequest) -> MessageResponse {
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let addr: SocketAddr = "127.0.0.1:0".parse().expect("parse reflection bind addr");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind reflection test listener");
    let endpoint = format!("http://{}", listener.local_addr().expect("local addr"));

    let server = tokio::spawn(async move {
        Server::builder()
            .add_service(reflection_service().expect("build reflection service"))
            .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                drop(shutdown_rx.await);
            })
            .await
            .expect("serve reflection test");
    });

    let channel = Endpoint::from_shared(endpoint)
        .expect("reflection endpoint")
        .connect()
        .await
        .expect("connect reflection client");
    let mut client = ServerReflectionClient::new(channel);
    let request = Request::new(tokio_stream::once(request));
    let mut response = client
        .server_reflection_info(request)
        .await
        .expect("reflection request")
        .into_inner();
    let message = response
        .next()
        .await
        .expect("reflection response")
        .expect("successful reflection response")
        .message_response
        .expect("reflection message response");

    shutdown_tx.send(()).expect("send reflection shutdown");
    server.await.expect("join reflection test server");

    message
}
