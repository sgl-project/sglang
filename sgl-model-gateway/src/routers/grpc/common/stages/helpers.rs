//! Common helper functions shared across stages

use std::sync::Arc;

use rand::Rng;
use smg_grpc_client::sglang_proto::DisaggregatedParams;
use tracing::debug;

use crate::{core::Worker, routers::grpc::proto_wrapper::ProtoGenerateRequest};

/// Inject PD bootstrap metadata into a gRPC request
///
/// Used by both chat and generate request building stages when in PD mode.
/// Only SGLang supports PD (prefill/decode) disaggregated mode.
pub(crate) fn inject_bootstrap_metadata(
    request: &mut ProtoGenerateRequest,
    prefill_worker: &Arc<dyn Worker>,
) {
    let hostname = prefill_worker.bootstrap_host();
    let bootstrap_port = prefill_worker.bootstrap_port().unwrap_or(8998);

    // In follow_bootstrap_room mode SGLang derives the prefill DP rank from
    // `bootstrap_room % dp_size`. Keep the room aligned with the logical worker
    // selected by the router.
    let room_id = match (prefill_worker.dp_rank(), prefill_worker.dp_size()) {
        (Some(rank), Some(size)) if rank < size => {
            match (i32::try_from(rank), i32::try_from(size)) {
                (Ok(rank), Ok(size)) if size > 0 => {
                    let bucket_count = ((i32::MAX - 1 - rank) / size) + 1;
                    rand::rng().random_range(0..bucket_count) * size + rank
                }
                _ => rand::rng().random_range(0..i32::MAX),
            }
        }
        _ => rand::rng().random_range(0..i32::MAX),
    };

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

#[cfg(test)]
mod tests {
    use smg_grpc_client::sglang_proto;

    use super::*;
    use crate::core::{DPAwareWorkerBuilder, WorkerType};

    #[test]
    fn bootstrap_room_targets_selected_prefill_rank() {
        let worker: Arc<dyn Worker> = Arc::new(
            DPAwareWorkerBuilder::new("grpc://prefill:30000", 2, 4)
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: Some(8998),
                })
                .build(),
        );

        for _ in 0..64 {
            let mut request =
                ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest::default()));
            inject_bootstrap_metadata(&mut request, &worker);
            let room = request
                .as_sglang()
                .disaggregated_params
                .as_ref()
                .unwrap()
                .bootstrap_room;
            assert_eq!(room % 4, 2);
        }
    }
}
