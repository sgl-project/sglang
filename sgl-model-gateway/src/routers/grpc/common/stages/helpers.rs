//! Common helper functions shared across stages

use std::sync::Arc;

use rand::Rng;
use tracing::debug;

use crate::{
    core::Worker, grpc_client::sglang_proto::DisaggregatedParams,
    routers::grpc::proto_wrapper::ProtoGenerateRequest,
};

/// Inject PD/EPD bootstrap metadata into a gRPC request
///
/// Used by both chat and generate request building stages when in PD/EPD mode.
/// Only SGLang supports PD (prefill/decode) and EPD (encode/prefill/decode) disaggregated modes.
///
/// For PD mode: only `prefill_worker` is provided
/// For EPD mode: both `prefill_worker` and `encode_worker` are provided
pub fn inject_bootstrap_metadata(
    request: &mut ProtoGenerateRequest,
    prefill_worker: &Arc<dyn Worker>,
    encode_worker: Option<&Arc<dyn Worker>>,
) {
    let hostname = prefill_worker.bootstrap_host();
    let bootstrap_port = prefill_worker.bootstrap_port().unwrap_or(8998);

    // Generate room ID for bootstrap
    let room_id = rand::rng().random_range(0..i32::MAX);

    // Create DisaggregatedParams with prefill worker metadata
    let mut disagg_params = DisaggregatedParams {
        bootstrap_host: hostname.to_string(),
        bootstrap_port: bootstrap_port as i32,
        bootstrap_room: room_id,
        encode_bootstrap_host: None,
        encode_bootstrap_port: None,
        encode_bootstrap_room: None,
    };

    // Add encode worker metadata if in EPD mode
    if let Some(encode) = encode_worker {
        let encode_hostname = encode.bootstrap_host();
        let encode_port = encode.bootstrap_port().unwrap_or(8998);
        let encode_room_id = rand::rng().random_range(0..i32::MAX);

        disagg_params.encode_bootstrap_host = Some(encode_hostname.to_string());
        disagg_params.encode_bootstrap_port = Some(encode_port as i32);
        disagg_params.encode_bootstrap_room = Some(encode_room_id);

        debug!(
            "Injected EPD bootstrap metadata: prefill(host={}, port={}, room={}), encode(host={}, port={}, room={})",
            hostname, bootstrap_port, room_id,
            encode_hostname, encode_port, encode_room_id
        );
    } else {
        debug!(
            "Injected PD bootstrap metadata: host={}, port={}, room={}",
            hostname, bootstrap_port, room_id
        );
    }

    // Inject metadata directly into SGLang request
    // (vLLM doesn't support PD/EPD mode, so this will panic if called with vLLM)
    let sglang_request = request.as_sglang_mut();
    sglang_request.disaggregated_params = Some(disagg_params);
}
