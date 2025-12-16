//! Request execution stage: Execute gRPC requests (single or dual dispatch)

use async_trait::async_trait;
use axum::response::Response;
use tracing::{debug, error, info_span, warn, Instrument};

use super::{helpers, EncodeHttpClient, EncodeRequest, PipelineStage};
use crate::routers::{
    error,
    grpc::{
        context::{ClientSelection, ExecutionResult, LoadGuards, RequestContext, WorkerSelection},
        proto_wrapper::{ProtoGenerateRequest, ProtoStream},
    },
};

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

        // For triple dispatch, we need workers from context
        let workers = ctx.state.workers.as_ref();

        let result = async {
            match self.mode {
                ExecutionMode::Single => self.execute_single(proto_request, clients, workers).await,
                ExecutionMode::DualDispatch => {
                    self.execute_dual_dispatch(proto_request, clients, workers)
                        .await
                }
                ExecutionMode::TripleDispatch => {
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
    /// EPD Flow with HTTP Encode + gRPC Prefill/Decode:
    /// 1. Extract multimodal items from request
    /// 2. Call encode worker HTTP `/encode` endpoint (wait for ack)
    /// 3. Encode worker processes images and sends embeddings to prefill via ZMQ
    /// 4. Send gRPC requests to prefill and decode workers in parallel
    /// 5. Prefill receives embeddings via ZMQ, computes KV cache, sends to decode
    /// 6. Decode generates output tokens
    ///
    /// Note: Encode uses HTTP REST API, prefill/decode use gRPC.
    /// Returns ExecutionResult::Dual since encode doesn't produce a gRPC stream.
    async fn execute_triple_dispatch(
        &self,
        mut proto_request: ProtoGenerateRequest,
        clients: &mut ClientSelection,
        workers: Option<&WorkerSelection>,
    ) -> Result<ExecutionResult, Response> {
        let (encode_url, prefill_client, decode_client) =
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

        // Get prefill worker for bootstrap host info
        let prefill_worker = workers.and_then(|w| w.prefill_worker()).ok_or_else(|| {
            error!(
                function = "execute_triple_dispatch",
                "Prefill worker not found in context"
            );
            error::internal_error(
                "prefill_worker_not_found",
                "Prefill worker not found in context",
            )
        })?;

        // Generate request ID
        let request_id = uuid::Uuid::new_v4().to_string();

        // Extract multimodal items for encode worker
        let mm_items = helpers::extract_multimodal_items(&proto_request);

        if mm_items.is_empty() {
            // No multimodal content - fall back to PD mode
            warn!(
                request_id = %request_id,
                "EPD mode but no multimodal content - falling back to PD mode"
            );
            // Clear mm_inputs and proceed with PD-like dual dispatch
            helpers::clear_multimodal_inputs(&mut proto_request);
            return self
                .execute_dual_dispatch_with_clients(proto_request, prefill_client, decode_client)
                .await;
        }

        debug!(
            request_id = %request_id,
            mm_items_count = mm_items.len(),
            encode_url = %encode_url,
            "EPD dispatch: calling encode worker HTTP endpoint"
        );

        // Step 1: Call encode worker HTTP /encode endpoint
        let encode_client = EncodeHttpClient::new();
        let encode_request = EncodeRequest {
            mm_items,
            req_id: request_id.clone(),
            num_parts: 1, // Single encode worker for now
            part_idx: 0,
            prefill_host: prefill_worker.bootstrap_host().to_string(),
            embedding_port: None, // Let runtime allocate ZMQ ports dynamically
        };

        // Wait for encode to complete (embeddings will be sent to prefill via ZMQ)
        if let Err(e) = encode_client.encode(encode_url, encode_request).await {
            error!(
                request_id = %request_id,
                error = %e,
                "Encode worker HTTP call failed"
            );
            return Err(error::internal_error(
                "encode_worker_http_failed",
                format!("Encode worker HTTP call failed: {}", e),
            ));
        }

        debug!(
            request_id = %request_id,
            "Encode HTTP call completed - embeddings being sent via ZMQ"
        );

        // Step 2: Clear multimodal inputs from prefill request
        // (encode worker handles multimodal, prefill receives embeddings via ZMQ)
        helpers::clear_multimodal_inputs(&mut proto_request);

        // Step 3: Mark request as waiting for image embeddings
        // The prefill scheduler will wait for embeddings from encode via ZMQ
        Self::set_wait_for_image(&mut proto_request, true);

        // Step 4: Execute prefill and decode in parallel via gRPC
        self.execute_dual_dispatch_with_clients(proto_request, prefill_client, decode_client)
            .await
    }

    /// Execute dual dispatch with explicit client references
    /// Used by both PD mode and EPD mode (after encode completes)
    async fn execute_dual_dispatch_with_clients(
        &self,
        proto_request: ProtoGenerateRequest,
        prefill_client: &mut crate::routers::grpc::client::GrpcClient,
        decode_client: &mut crate::routers::grpc::client::GrpcClient,
    ) -> Result<ExecutionResult, Response> {
        let prefill_request = proto_request.clone_inner();
        let decode_request = proto_request;

        let (prefill_result, decode_result): (StreamResult, StreamResult) = tokio::join!(
            prefill_client.generate(prefill_request),
            decode_client.generate(decode_request)
        );

        // Handle prefill result
        let prefill_stream = match prefill_result {
            Ok(s) => s,
            Err(e) => {
                error!(
                    function = "execute_dual_dispatch_with_clients",
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
                    function = "execute_dual_dispatch_with_clients",
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

    /// Set the need_wait_for_image flag on a proto request
    ///
    /// This tells the prefill scheduler to wait for image embeddings
    /// from the encode worker via ZMQ before processing.
    fn set_wait_for_image(proto_request: &mut ProtoGenerateRequest, wait: bool) {
        if let ProtoGenerateRequest::Sglang(req) = proto_request {
            req.need_wait_for_image = Some(wait);
        }
    }
}
