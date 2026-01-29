//! Common helper functions shared across stages

use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD, Engine};
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
/// Note: For EPD mode, encode workers use gRPC while the prefill↔decode KV cache
/// transfer still relies on bootstrap metadata.
pub(crate) fn inject_bootstrap_metadata(
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
/// - Data URIs (for inline base64 data)
///
/// Raw data bytes (image_data, video_data, audio_data) are converted back to
/// data URIs so the encode worker can process them uniformly.
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

                // Convert raw image data bytes back to data URIs
                // The encoder expects URLs or data URIs, not raw bytes
                for data in &mm.image_data {
                    if !data.is_empty() {
                        let base64_data = STANDARD.encode(data);
                        // Use generic image mime type; encoder will detect actual format
                        let data_uri = format!("data:image/png;base64,{}", base64_data);
                        items.push(data_uri);
                    }
                }

                // Convert raw video data bytes back to data URIs
                for data in &mm.video_data {
                    if !data.is_empty() {
                        let base64_data = STANDARD.encode(data);
                        let data_uri = format!("data:video/mp4;base64,{}", base64_data);
                        items.push(data_uri);
                    }
                }

                // Convert raw audio data bytes back to data URIs
                for data in &mm.audio_data {
                    if !data.is_empty() {
                        let base64_data = STANDARD.encode(data);
                        let data_uri = format!("data:audio/wav;base64,{}", base64_data);
                        items.push(data_uri);
                    }
                }
            }
        }
        ProtoGenerateRequest::Vllm(_) => {
            // vLLM doesn't support EPD mode
        }
    }

    items
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
