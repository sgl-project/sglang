//! Common helper functions shared across stages

use std::sync::Arc;

use rand::Rng;
use tracing::debug;

use crate::{
    core::Worker, grpc_client::sglang_proto::DisaggregatedParams,
    routers::grpc::proto_wrapper::ProtoGenerateRequest,
};

/// Inject PD bootstrap metadata into a gRPC request
///
/// Used by both chat and generate request building stages when in PD mode.
/// Only SGLang supports PD (prefill/decode) disaggregated mode.
pub fn inject_bootstrap_metadata(
    request: &mut ProtoGenerateRequest,
    prefill_worker: &Arc<dyn Worker>,
) {
    let hostname = prefill_worker.bootstrap_host();
    let bootstrap_port = prefill_worker.bootstrap_port().unwrap_or(8998);

    // Generate room ID for bootstrap
    let room_id = rand::rng().random_range(0..i32::MAX);

    // Create DisaggregatedParams
    let disagg_params = DisaggregatedParams {
        bootstrap_host: hostname.to_string(),
        bootstrap_port: bootstrap_port as i32,
        bootstrap_room: room_id,
    };

    // Inject metadata directly into SGLang request
    // (vLLM doesn't support PD mode, so this will panic if called with vLLM)
    let sglang_request = request.as_sglang_mut();
    sglang_request.disaggregated_params = Some(disagg_params);

    debug!(
        "Injected bootstrap metadata: host={}, port={}, room={}",
        hostname, bootstrap_port, room_id
    );
}
