//! Request execution stage: Execute gRPC requests (single or dual dispatch)

use async_trait::async_trait;
use axum::response::Response;
use tracing::{debug, error, info_span, warn, Instrument};

use super::{helpers, PipelineStage};
use crate::{
    grpc_client::encoder_proto,
    routers::{
        error,
        grpc::{
            context::{
                ClientSelection, ExecutionResult, LoadGuards, RequestContext, WorkerSelection,
            },
            proto_wrapper::{
                ProtoEmbedRequest, ProtoEmbedResponseVariant, ProtoGenerateRequest, ProtoRequest,
                ProtoStream,
            },
        },
    },
};

type StreamResult = Result<ProtoStream, Box<dyn std::error::Error + Send + Sync>>;

/// Request execution stage: Execute gRPC requests (single or dual dispatch)
pub(crate) struct RequestExecutionStage {
    mode: ExecutionMode,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ExecutionMode {
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

        ctx.state.load_guards = Some(LoadGuards::new(workers, ctx.input.headers.as_ref()));

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
            target: "smg::otel-trace",
            "grpc_generate",
            request_id = %request_id,
            model = %model,
            mode = ?self.mode,
        );

        let result = async {
            match proto_request {
                ProtoRequest::Generate(req) => match self.mode {
                    ExecutionMode::Single => self.execute_single(req, clients, workers).await,
                    ExecutionMode::DualDispatch => {
                        self.execute_dual_dispatch(req, clients, workers).await
                    }
                    ExecutionMode::TripleDispatch => {
                        self.execute_triple_dispatch(req, clients, Some(workers))
                            .await
                    }
                },
                ProtoRequest::Embed(req) => self.execute_single_embed(req, clients).await,
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

    async fn execute_single_embed(
        &self,
        proto_request: ProtoEmbedRequest,
        clients: &mut ClientSelection,
    ) -> Result<ExecutionResult, Response> {
        let client = clients.single_mut().ok_or_else(|| {
            error!(
                function = "execute_single_embed",
                "Expected single client but got dual"
            );
            error::internal_error(
                "expected_single_client_got_dual",
                "Expected single client but got dual",
            )
        })?;

        let response = client.embed(proto_request).await.map_err(|e| {
            error!(
                function = "execute_single_embed",
                error = %e,
                "Failed to start embedding"
            );
            error::internal_error(
                "start_embedding_failed",
                format!("Failed to start embedding: {}", e),
            )
        })?;

        match response.into_response() {
            ProtoEmbedResponseVariant::Complete(complete) => {
                Ok(ExecutionResult::Embedding { response: complete })
            }
            ProtoEmbedResponseVariant::Error(e) => {
                error!(
                    function = "execute_single_embed",
                    error = %e.message(),
                    "Embedding execution failed"
                );
                Err(error::internal_error(
                    "embedding_execution_failed",
                    e.message().to_string(),
                ))
            }
            ProtoEmbedResponseVariant::None => {
                error!(
                    function = "execute_single_embed",
                    "Embedding execution returned no response"
                );
                Err(error::internal_error(
                    "embedding_no_response",
                    "Embedding execution returned no response",
                ))
            }
        }
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

        let result =
            Self::execute_prefill_decode_grpc(proto_request, prefill_client, decode_client).await;

        // Record circuit breaker outcomes for each worker
        workers.record_dual_outcomes(result.is_ok(), result.is_ok());

        result
    }

    /// Execute triple dispatch for EPD (Encode-Prefill-Decode) mode
    ///
    /// EPD Flow with gRPC Encode + gRPC Prefill/Decode:
    /// 1. Extract multimodal items from request
    /// 2. Call encode worker gRPC Encode() RPC (wait for ack)
    /// 3. Encode worker processes images and sends embeddings to prefill via ZMQ
    /// 4. Send gRPC requests to prefill and decode workers in parallel
    /// 5. Prefill receives embeddings via ZMQ, computes KV cache, sends to decode
    /// 6. Decode generates output tokens
    ///
    /// Returns ExecutionResult::Dual since encode doesn't produce a streaming response.
    async fn execute_triple_dispatch(
        &self,
        mut proto_request: ProtoGenerateRequest,
        clients: &mut ClientSelection,
        workers: Option<&WorkerSelection>,
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

        let request_id = proto_request.request_id().to_string();

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
            return Self::execute_prefill_decode_grpc(proto_request, prefill_client, decode_client)
                .await;
        }

        debug!(
            request_id = %request_id,
            mm_items_count = mm_items.len(),
            "EPD dispatch: calling encode worker gRPC endpoint"
        );

        // Step 1: Call encode worker gRPC Encode() RPC
        let encode_request = encoder_proto::EncodeRequest {
            mm_items,
            req_id: request_id.clone(),
            num_parts: 1, // Single encode worker for now
            part_idx: 0,
            prefill_host: prefill_worker.bootstrap_host().to_string(),
            embedding_port: vec![], // Let runtime allocate ZMQ ports dynamically
        };

        if let Err(e) = encode_client.encode(encode_request).await {
            error!(
                request_id = %request_id,
                error = %e,
                "Encode worker gRPC call failed"
            );
            if let Some(w) = workers {
                w.record_triple_outcomes(false, true, true);
            }
            return Err(error::internal_error(
                "encode_worker_grpc_failed",
                format!("Encode worker gRPC call failed: {}", e),
            ));
        }

        debug!(
            request_id = %request_id,
            "Encode gRPC call completed - embeddings being sent via ZMQ"
        );

        // Step 2: Clear multimodal inputs from prefill request
        helpers::clear_multimodal_inputs(&mut proto_request);

        // Step 3: Mark request as waiting for image embeddings
        Self::set_wait_for_image(&mut proto_request, true);

        // Step 4: Execute prefill and decode in parallel via gRPC
        let result =
            Self::execute_prefill_decode_grpc(proto_request, prefill_client, decode_client).await;

        // Record circuit breaker outcomes for all workers
        if let Some(w) = workers {
            // Encode succeeded (we got here), prefill/decode success based on result
            w.record_triple_outcomes(true, result.is_ok(), result.is_ok());
        }

        result
    }

    /// Execute gRPC requests to prefill and decode workers in parallel
    ///
    /// Shared logic for PD mode and EPD mode (after encode completes).
    async fn execute_prefill_decode_grpc(
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
        let prefill_stream = prefill_result.map_err(|e| {
            error!(
                function = "execute_prefill_decode_grpc",
                error = %e,
                "Prefill worker failed to start"
            );
            error::internal_error(
                "prefill_worker_failed_to_start",
                format!("Prefill worker failed to start: {}", e),
            )
        })?;

        // Handle decode result
        let decode_stream = decode_result.map_err(|e| {
            error!(
                function = "execute_prefill_decode_grpc",
                error = %e,
                "Decode worker failed to start"
            );
            error::internal_error(
                "decode_worker_failed_to_start",
                format!("Decode worker failed to start: {}", e),
            )
        })?;

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
