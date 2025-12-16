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
///
/// Note: For EPD mode, encode workers use HTTP REST API, not gRPC bootstrap.
/// The prefill↔decode KV cache transfer still uses bootstrap metadata.
pub fn inject_bootstrap_metadata(
    request: &mut ProtoGenerateRequest,
    prefill_worker: &Arc<dyn Worker>,
) {
    let hostname = prefill_worker.bootstrap_host();
    let bootstrap_port = prefill_worker.bootstrap_port().unwrap_or(8998);

    // Generate room ID for bootstrap
    let room_id = rand::rng().random_range(0..i32::MAX);

    // Create DisaggregatedParams with prefill worker metadata
    let disagg_params = DisaggregatedParams {
        bootstrap_host: hostname.to_string(),
        bootstrap_port: bootstrap_port as i32,
        bootstrap_room: room_id,
        // EPD encode bootstrap fields are not used - encode uses HTTP REST API
        encode_bootstrap_host: None,
        encode_bootstrap_port: None,
        encode_bootstrap_room: None,
    };

    debug!(
        "Injected PD bootstrap metadata: host={}, port={}, room={}",
        hostname, bootstrap_port, room_id
    );

    // Inject metadata directly into SGLang request
    // (vLLM doesn't support PD mode, so this will panic if called with vLLM)
    let sglang_request = request.as_sglang_mut();
    sglang_request.disaggregated_params = Some(disagg_params);
}

/// Extract multimodal item URLs from a proto request for EPD mode
///
/// Returns a vector of URL strings that can be:
/// - Image URLs (http/https)
/// - Video URLs
/// - Audio URLs
///
/// Note: Raw data (image_data, video_data, audio_data) are not extracted here
/// as they are bytes in the proto. The encode worker expects URLs or data URIs.
///
/// Returns empty vector if no multimodal inputs are present.
pub fn extract_multimodal_items(request: &ProtoGenerateRequest) -> Vec<String> {
    let mut items = Vec::new();

    match request {
        ProtoGenerateRequest::Sglang(req) => {
            if let Some(ref mm) = req.mm_inputs {
                // Add image URLs
                items.extend(mm.image_urls.iter().cloned());
                // Add video URLs
                items.extend(mm.video_urls.iter().cloned());
                // Add audio URLs
                items.extend(mm.audio_urls.iter().cloned());
                // Note: image_data, video_data, audio_data are bytes in proto,
                // not suitable for direct URL extraction
            }
        }
        ProtoGenerateRequest::Vllm(_) => {
            // vLLM doesn't support EPD mode
        }
    }

    items
}

/// Check if a proto request contains multimodal inputs
pub fn has_multimodal_inputs(request: &ProtoGenerateRequest) -> bool {
    match request {
        ProtoGenerateRequest::Sglang(req) => {
            if let Some(ref mm) = req.mm_inputs {
                !mm.image_urls.is_empty()
                    || !mm.video_urls.is_empty()
                    || !mm.audio_urls.is_empty()
                    || !mm.image_data.is_empty()
                    || !mm.video_data.is_empty()
                    || !mm.audio_data.is_empty()
            } else {
                false
            }
        }
        ProtoGenerateRequest::Vllm(_) => false,
    }
}

/// Clear multimodal inputs from a proto request
///
/// Used in EPD mode to clear mm_inputs from the prefill request
/// since the encode worker handles multimodal processing.
pub fn clear_multimodal_inputs(request: &mut ProtoGenerateRequest) {
    if let ProtoGenerateRequest::Sglang(req) = request {
        req.mm_inputs = None;
    }
}
