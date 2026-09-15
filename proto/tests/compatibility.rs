use prost::Message;
use sglang_grpc_types::{
    proto::{sglang_service_client::SglangServiceClient, sglang_service_server::SERVICE_NAME},
    sglang::runtime::v1::GenerateRequest,
};

// A minimal view of the released runtime.v1 request. Decoding in both
// directions catches accidental changes to these public field numbers without
// coupling the test to Prost's generated source or field ordering.
#[derive(Clone, PartialEq, Message)]
struct ReleasedGenerateRequest {
    #[prost(int32, repeated, tag = "1")]
    input_ids: Vec<i32>,
    #[prost(bool, optional, tag = "3")]
    stream: Option<bool>,
    #[prost(string, optional, tag = "7")]
    rid: Option<String>,
}

#[test]
fn generate_request_preserves_released_wire_tags() {
    let current = GenerateRequest {
        input_ids: vec![1, 300, 42],
        stream: Some(true),
        rid: Some("request-7".to_owned()),
        ..Default::default()
    };

    let released = ReleasedGenerateRequest::decode(current.encode_to_vec().as_slice()).unwrap();
    assert_eq!(released.input_ids, current.input_ids);
    assert_eq!(released.stream, current.stream);
    assert_eq!(released.rid, current.rid);

    let decoded = GenerateRequest::decode(released.encode_to_vec().as_slice()).unwrap();
    assert_eq!(decoded.input_ids, current.input_ids);
    assert_eq!(decoded.stream, current.stream);
    assert_eq!(decoded.rid, current.rid);
}

#[test]
fn exports_the_canonical_client_and_service_identity() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<SglangServiceClient<tonic::transport::Channel>>();
    assert_eq!(SERVICE_NAME, "sglang.runtime.v1.SglangService");
}
