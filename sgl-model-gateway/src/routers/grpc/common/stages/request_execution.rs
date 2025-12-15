//! Request execution stage: Execute gRPC requests (single or dual dispatch)

use std::sync::atomic::{AtomicI32, Ordering};

use async_trait::async_trait;
use axum::response::Response;
use tracing::{debug, error, info_span, warn, Instrument};

use super::PipelineStage;
use crate::{
    core::WorkerType,
    grpc_client::sglang_proto as sglang,
    routers::{
        error,
        grpc::{
            context::{ClientSelection, ExecutionResult, LoadGuards, RequestContext, WorkerSelection},
            proto_wrapper::{ProtoGenerateRequest, ProtoStream},
        },
    },
};

/// Global counter for generating unique bootstrap room IDs
static BOOTSTRAP_ROOM_COUNTER: AtomicI32 = AtomicI32::new(0);

type StreamResult = Result<ProtoStream, Box<dyn std::error::Error + Send + Sync>>;

/// Request execution stage: Execute gRPC requests (single or dual dispatch)
pub struct RequestExecutionStage {
    mode: ExecutionMode,
}

#[derive(Debug, Clone, Copy)]
pub enum ExecutionMode {
    /// Regular mode: single worker execution
    Single,
    /// PD mode: dual dispatch to prefill + decode workers
    DualDispatch,
    /// EPD mode: encode + prefill + decode workers
    TripleDispatch,
}

impl RequestExecutionStage {
    pub fn new(mode: ExecutionMode) -> Self {
        Self { mode }
    }
}

#[async_trait]
impl PipelineStage for RequestExecutionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let proto_request = ctx.state.proto_request.take().ok_or_else(|| {
            error!(
                function = "RequestExecutionStage::execute",
                "Proto request not built"
            );
            error::internal_error("proto_request_not_built", "Proto request not built")
        })?;

        let clients = ctx.state.clients.as_mut().ok_or_else(|| {
            error!(
                function = "RequestExecutionStage::execute",
                "Client acquisition not completed"
            );
            error::internal_error(
                "client_acquisition_not_completed",
                "Client acquisition not completed",
            )
        })?;

        // Create load guards for worker load tracking (increment load when created)
        // They will be automatically dropped (and decrement load) when RequestContext is dropped
        let workers = ctx.state.workers.as_ref().ok_or_else(|| {
            error!(
                function = "RequestExecutionStage::execute",
                "Worker selection not completed"
            );
            error::internal_error(
                "worker_selection_not_completed",
                "Worker selection not completed",
            )
        })?;

        ctx.state.load_guards = Some(LoadGuards::from(workers));

        // Extract dispatch metadata for tracing span
        let request_id = ctx
            .state
            .dispatch
            .as_ref()
            .map(|d| d.request_id.as_str())
            .unwrap_or("unknown");
        let model = ctx
            .state
            .dispatch
            .as_ref()
            .map(|d| d.model.as_str())
            .unwrap_or("unknown");

        // Create OTEL span for gRPC request execution
        let span = info_span!(
            target: "sgl_model_gateway::otel-trace",
            "grpc_generate",
            request_id = %request_id,
            model = %model,
            mode = ?self.mode,
        );

        let result = async {
            match self.mode {
                ExecutionMode::Single => self.execute_single(proto_request, clients, workers).await,
                ExecutionMode::DualDispatch => {
                    self.execute_dual_dispatch(proto_request, clients, workers)
                        .await
                }
                ExecutionMode::TripleDispatch => {
                    let workers = ctx.state.workers.as_ref().ok_or_else(|| {
                        error!(
                            function = "RequestExecutionStage::execute",
                            "Worker selection not available for triple dispatch"
                        );
                        error::internal_error(
                            "worker_selection_not_available",
                            "Worker selection not available",
                        )
                    })?;
                    self.execute_triple_dispatch(proto_request, clients, workers)
                        .await
                }
            }
        }
        .instrument(span)
        .await?;

        // Store result in context for ResponseProcessingStage
        ctx.state.response.execution_result = Some(result);
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "RequestExecution"
    }
}

impl RequestExecutionStage {
    async fn execute_single(
        &self,
        proto_request: ProtoGenerateRequest,
        clients: &mut ClientSelection,
        workers: &WorkerSelection,
    ) -> Result<ExecutionResult, Response> {
        let client = clients.single_mut().ok_or_else(|| {
            error!(
                function = "execute_single",
                "Expected single client but got dual"
            );
            error::internal_error(
                "expected_single_client_got_dual",
                "Expected single client but got dual",
            )
        })?;

        let result = client.generate(proto_request).await;

        // Record circuit breaker outcome
        workers.record_outcome(result.is_ok());

        let stream = result.map_err(|e| {
            error!(
                function = "execute_single",
                error = %e,
                "Failed to start generation"
            );
            error::internal_error(
                "start_generation_failed",
                format!("Failed to start generation: {}", e),
            )
        })?;

        Ok(ExecutionResult::Single { stream })
    }

    async fn execute_dual_dispatch(
        &self,
        proto_request: ProtoGenerateRequest,
        clients: &mut ClientSelection,
        workers: &WorkerSelection,
    ) -> Result<ExecutionResult, Response> {
        let (prefill_client, decode_client) = clients.dual_mut().ok_or_else(|| {
            error!(
                function = "execute_dual_dispatch",
                "Expected dual clients but got single"
            );
            error::internal_error(
                "expected_dual_clients_got_single",
                "Expected dual clients but got single",
            )
        })?;

        let prefill_request = proto_request.clone_inner();
        let decode_request = proto_request;

        let (prefill_result, decode_result): (StreamResult, StreamResult) = tokio::join!(
            prefill_client.generate(prefill_request),
            decode_client.generate(decode_request)
        );

        // Record circuit breaker outcomes for each worker individually
        workers.record_dual_outcomes(prefill_result.is_ok(), decode_result.is_ok());

        // Handle prefill result
        let prefill_stream = match prefill_result {
            Ok(s) => s,
            Err(e) => {
                error!(
                    function = "execute_dual_dispatch",
                    error = %e,
                    "Prefill worker failed to start"
                );
                return Err(error::internal_error(
                    "prefill_worker_failed_to_start",
                    format!("Prefill worker failed to start: {}", e),
                ));
            }
        };

        // Handle decode result
        let decode_stream = match decode_result {
            Ok(s) => s,
            Err(e) => {
                error!(
                    function = "execute_dual_dispatch",
                    error = %e,
                    "Decode worker failed to start"
                );
                return Err(error::internal_error(
                    "decode_worker_failed_to_start",
                    format!("Decode worker failed to start: {}", e),
                ));
            }
        };

        Ok(ExecutionResult::Dual {
            prefill: prefill_stream,
            decode: Box::new(decode_stream),
        })
    }

    /// Execute triple dispatch for EPD (Encode-Prefill-Decode) mode
    ///
    /// For multimodal requests, the flow is:
    /// 1. Router splits the request: image data → Encoder, text → Prefill
    /// 2. All three workers start in parallel
    /// 3. Encoder computes embeddings and pushes to Prefill via bootstrap
    /// 4. Prefill merges embeddings with text and computes KV cache
    /// 5. Decode receives KV cache from Prefill and generates output
    ///
    /// For non-multimodal requests, Encoder receives an empty request but the flow
    /// still goes through all three workers for consistency.
    async fn execute_triple_dispatch(
        &self,
        proto_request: ProtoGenerateRequest,
        clients: &mut ClientSelection,
        workers: &WorkerSelection,
    ) -> Result<ExecutionResult, Response> {
        let (encode_client, prefill_client, decode_client) =
            clients.triple_mut().ok_or_else(|| {
                error!(
                    function = "execute_triple_dispatch",
                    "Expected triple clients but selection differed"
                );
                error::internal_error(
                    "expected_encode/prefill/decode_clients_but_selection_differed",
                    "Expected encode/prefill/decode clients but selection differed",
                )
            })?;

        let (encode_worker, prefill_worker, _decode_worker) =
            workers.triple().ok_or_else(|| {
                error!(
                    function = "execute_triple_dispatch",
                    "Expected triple workers but selection differed"
                );
                error::internal_error(
                    "expected_encode/prefill/decode_workers_but_selection_differed",
                    "Expected encode/prefill/decode workers but selection differed",
                )
            })?;

        // Get bootstrap info from workers
        let encode_bootstrap_host = encode_worker.bootstrap_host();
        let encode_bootstrap_port = match encode_worker.worker_type() {
            WorkerType::Encode { bootstrap_port } => bootstrap_port.unwrap_or(0) as i32,
            _ => {
                warn!(
                    "Encode worker has unexpected type: {:?}",
                    encode_worker.worker_type()
                );
                0
            }
        };

        let prefill_bootstrap_host = prefill_worker.bootstrap_host();
        let prefill_bootstrap_port = prefill_worker.bootstrap_port().unwrap_or(0) as i32;

        // Generate unique room ID for this request's bootstrap coordination
        let bootstrap_room = BOOTSTRAP_ROOM_COUNTER.fetch_add(1, Ordering::Relaxed);

        // Check if request has multimodal content
        let is_multimodal = Self::has_multimodal(&proto_request);

        debug!(
            encode_bootstrap_host = %encode_bootstrap_host,
            encode_bootstrap_port = %encode_bootstrap_port,
            prefill_bootstrap_host = %prefill_bootstrap_host,
            prefill_bootstrap_port = %prefill_bootstrap_port,
            bootstrap_room = %bootstrap_room,
            is_multimodal = %is_multimodal,
            "EPD dispatch: Setting up bootstrap metadata"
        );

        // Create specialized requests for each worker
        //
        // EPD Flow:
        // - Encoder: receives multimodal data, outputs embeddings to prefill
        // - Prefill: receives text tokens + encode bootstrap (to get embeddings),
        //            outputs KV cache to decode
        // - Decode:  receives prefill bootstrap (to get KV cache), generates output
        let encode_request = Self::build_encode_request(&proto_request, bootstrap_room);
        let prefill_request = Self::build_prefill_request_epd(
            &proto_request,
            prefill_bootstrap_host,
            prefill_bootstrap_port,
            bootstrap_room,
            encode_bootstrap_host,
            encode_bootstrap_port,
            bootstrap_room,
        );
        let decode_request = Self::build_decode_request_epd(
            &proto_request,
            prefill_bootstrap_host,
            prefill_bootstrap_port,
            bootstrap_room,
        );

        // Launch all three workers in parallel
        // The workers coordinate via their bootstrap connections:
        // - Encoder pushes embeddings to Prefill
        // - Prefill pushes KV cache to Decode
        let encode_future = encode_client.generate(encode_request);
        let prefill_future = prefill_client.generate(prefill_request);
        let decode_future = decode_client.generate(decode_request);

        let (encode_result, prefill_result, decode_result): (
            StreamResult,
            StreamResult,
            StreamResult,
        ) = tokio::join!(encode_future, prefill_future, decode_future);

        let encode_stream = match encode_result {
            Ok(s) => s,
            Err(e) => {
                error!(
                    function = "execute_triple_dispatch",
                    error = %e,
                    "Encode worker failed to start"
                );
                return Err(error::internal_error(
                    "encode_worker_failed_to_start",
                    format!("Encode worker failed to start: {}", e),
                ));
            }
        };

        let prefill_stream = match prefill_result {
            Ok(s) => s,
            Err(e) => {
                error!(
                    function = "execute_triple_dispatch",
                    error = %e,
                    "Prefill worker failed to start"
                );
                return Err(error::internal_error(
                    "prefill_worker_failed_to_start",
                    format!("Prefill worker failed to start: {}", e),
                ));
            }
        };

        let decode_stream = match decode_result {
            Ok(s) => s,
            Err(e) => {
                error!(
                    function = "execute_triple_dispatch",
                    error = %e,
                    "Decode worker failed to start"
                );
                return Err(error::internal_error(
                    "decode_worker_failed_to_start",
                    format!("Decode worker failed to start: {}", e),
                ));
            }
        };

        Ok(ExecutionResult::Triple {
            encode: Box::new(encode_stream),
            prefill: Box::new(prefill_stream),
            decode: Box::new(decode_stream),
        })
    }

    // ===== EPD Request Building Helpers =====

    /// Check if a request contains multimodal content (images, videos, audio)
    fn has_multimodal(proto_request: &ProtoGenerateRequest) -> bool {
        match proto_request {
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

    /// Build request for Encoder worker
    ///
    /// Encoder receives multimodal inputs and produces embeddings.
    fn build_encode_request(
        proto_request: &ProtoGenerateRequest,
        encode_bootstrap_room: i32,
    ) -> ProtoGenerateRequest {
        match proto_request {
            ProtoGenerateRequest::Sglang(req) => {
                let mut encoder_req = (**req).clone();
                // Keep multimodal inputs for encoder
                // Clear tokenized input - encoder only processes multimodal
                encoder_req.tokenized = Some(sglang::TokenizedInput {
                    original_text: String::new(),
                    input_ids: Vec::new(),
                });
                // Set room ID so encoder knows this is its assigned room
                if encoder_req.disaggregated_params.is_none() {
                    encoder_req.disaggregated_params = Some(sglang::DisaggregatedParams::default());
                }
                if let Some(ref mut params) = encoder_req.disaggregated_params {
                    params.encode_bootstrap_room = Some(encode_bootstrap_room);
                }
                ProtoGenerateRequest::Sglang(Box::new(encoder_req))
            }
            ProtoGenerateRequest::Vllm(_) => {
                panic!("EPD mode not supported for vLLM")
            }
        }
    }

    /// Build request for Prefill worker in EPD mode
    ///
    /// Prefill receives text tokens and waits for embeddings from Encoder.
    fn build_prefill_request_epd(
        proto_request: &ProtoGenerateRequest,
        prefill_bootstrap_host: &str,
        prefill_bootstrap_port: i32,
        prefill_bootstrap_room: i32,
        encode_bootstrap_host: &str,
        encode_bootstrap_port: i32,
        encode_bootstrap_room: i32,
    ) -> ProtoGenerateRequest {
        match proto_request {
            ProtoGenerateRequest::Sglang(req) => {
                let mut prefill_req = (**req).clone();
                // Clear multimodal inputs - encoder processes those
                prefill_req.mm_inputs = None;
                // Set disaggregated params
                prefill_req.disaggregated_params = Some(sglang::DisaggregatedParams {
                    // Prefill's bootstrap info (for decode to connect)
                    bootstrap_host: prefill_bootstrap_host.to_string(),
                    bootstrap_port: prefill_bootstrap_port,
                    bootstrap_room: prefill_bootstrap_room,
                    // Encode bootstrap info (for prefill to receive embeddings)
                    encode_bootstrap_host: Some(encode_bootstrap_host.to_string()),
                    encode_bootstrap_port: Some(encode_bootstrap_port),
                    encode_bootstrap_room: Some(encode_bootstrap_room),
                });
                ProtoGenerateRequest::Sglang(Box::new(prefill_req))
            }
            ProtoGenerateRequest::Vllm(_) => {
                panic!("EPD mode not supported for vLLM")
            }
        }
    }

    /// Build request for Decode worker in EPD mode
    ///
    /// Decode receives KV cache from Prefill.
    fn build_decode_request_epd(
        proto_request: &ProtoGenerateRequest,
        prefill_bootstrap_host: &str,
        prefill_bootstrap_port: i32,
        prefill_bootstrap_room: i32,
    ) -> ProtoGenerateRequest {
        match proto_request {
            ProtoGenerateRequest::Sglang(req) => {
                let mut decode_req = (**req).clone();
                // Clear multimodal inputs - not needed for decode
                decode_req.mm_inputs = None;
                // Clear tokenized input - decode receives context from prefill
                decode_req.tokenized = Some(sglang::TokenizedInput {
                    original_text: String::new(),
                    input_ids: Vec::new(),
                });
                // Set prefill bootstrap info for decode to receive KV cache
                decode_req.disaggregated_params = Some(sglang::DisaggregatedParams {
                    bootstrap_host: prefill_bootstrap_host.to_string(),
                    bootstrap_port: prefill_bootstrap_port,
                    bootstrap_room: prefill_bootstrap_room,
                    encode_bootstrap_host: None,
                    encode_bootstrap_port: None,
                    encode_bootstrap_room: None,
                });
                ProtoGenerateRequest::Sglang(Box::new(decode_req))
            }
            ProtoGenerateRequest::Vllm(_) => {
                panic!("EPD mode not supported for vLLM")
            }
        }
    }
}
