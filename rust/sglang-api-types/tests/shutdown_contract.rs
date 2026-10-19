use prost::Message;
use sglang_api_types::{api::v1 as api, runtime::v1 as runtime};

#[test]
fn shutdown_messages_are_empty_in_both_packages() {
    assert!(runtime::ShutdownRequest {}.encode_to_vec().is_empty());
    assert!(runtime::ShutdownResponse {}.encode_to_vec().is_empty());
    assert!(api::ShutdownRequest {}.encode_to_vec().is_empty());
    assert!(api::ShutdownResponse {}.encode_to_vec().is_empty());
    assert_eq!(
        serde_json::to_string(&api::ShutdownRequest {}).unwrap(),
        "{}"
    );
    assert_eq!(
        serde_json::to_string(&api::ShutdownResponse {}).unwrap(),
        "{}"
    );
}

#[test]
fn both_generated_clients_expose_shutdown() {
    let _ = |client: &mut runtime::sglang_service_client::SglangServiceClient<
        tonic::transport::Channel,
    >| {
        drop(client.shutdown(runtime::ShutdownRequest {}));
    };
    let _ = |client: &mut api::sglang_service_client::SglangServiceClient<
        tonic::transport::Channel,
    >| {
        drop(client.shutdown(api::ShutdownRequest {}));
    };
}
