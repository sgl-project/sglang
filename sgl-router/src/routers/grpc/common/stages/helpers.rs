//! Common helper functions shared across stages

use std::sync::Arc;

use proto::DisaggregatedParams;
use rand::Rng;
use tracing::debug;

use crate::{core::Worker, grpc_client::proto};

/// Inject PD bootstrap metadata into a gRPC request
///
/// Used by both chat and generate request building stages when in PD mode.
pub fn inject_bootstrap_metadata(
    request: &mut proto::GenerateRequest,
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

    // Inject metadata directly into request
    request.disaggregated_params = Some(disagg_params);

    debug!(
        "Injected bootstrap metadata: host={}, port={}, room={}",
        hostname, bootstrap_port, room_id
    );
}
