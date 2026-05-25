use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use pyo3::PyErr;
use pyo3::Python;
use pyo3::exceptions::{PyTypeError, PyValueError};
use tokio::sync::{Notify, mpsc::Receiver};
use tokio::time::{Duration, timeout};
use tokio_stream::Stream;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

use crate::bridge::{PyBridge, ResponseChunk, TerminalError};
use crate::proto;
use crate::utils::{
    build_classify_dict, build_embed_dict, build_generate_dict, build_text_embed_dict,
    build_text_generate_dict, extract_model_path,
};

pub struct SglangServiceImpl {
    pub bridge: Arc<PyBridge>,
    pub response_timeout: Duration,
}

type StreamResult<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;
pub const DEFAULT_RESPONSE_TIMEOUT_SECS: u64 = 300;

/// 64 MiB — leaves headroom for multimodal inputs and OpenAI JSON pass-through bodies,
/// well above tonic's 4 MiB decode default.
pub const DEFAULT_GRPC_MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024;

/// Resolve the per-message size cap (bytes) applied to the Tonic encoder/decoder.
//
// TODO(grpc-args): promote SGLANG_TONIC_PAYLOAD to a proper `--grpc-max-message-size`
// server argument once the launcher PR (3/4) wires server args through.
fn resolve_max_message_size() -> usize {
    match std::env::var("SGLANG_TONIC_PAYLOAD") {
        Ok(raw) => match raw.parse::<usize>() {
            Ok(n) if n > 0 => {
                tracing::info!(
                    bytes = n,
                    "Using SGLANG_TONIC_PAYLOAD override for gRPC max message size"
                );
                n
            }
            _ => {
                tracing::warn!(
                    value = %raw,
                    default = DEFAULT_GRPC_MAX_MESSAGE_SIZE,
                    "Ignoring invalid SGLANG_TONIC_PAYLOAD; using default"
                );
                DEFAULT_GRPC_MAX_MESSAGE_SIZE
            }
        },
        Err(_) => DEFAULT_GRPC_MAX_MESSAGE_SIZE,
    }
}

/// Classify a bridge `PyErr` into the right gRPC `Status`.
///
/// `PyValueError` / `PyTypeError` mean the client sent bad input — surface as
/// `INVALID_ARGUMENT` so callers can distinguish them from server failures.
/// Everything else (typically `PyRuntimeError`, but also Python tracebacks
/// from inside the tokenizer manager) maps to `INTERNAL`.
fn pyerr_to_status(err: PyErr, context: &str) -> Status {
    let is_client_error = Python::with_gil(|py| {
        err.is_instance_of::<PyValueError>(py) || err.is_instance_of::<PyTypeError>(py)
    });
    let msg = format!("{}: {}", context, err);
    if is_client_error {
        Status::invalid_argument(msg)
    } else {
        Status::internal(msg)
    }
}

async fn recv_chunk_with_timeout(
    receiver: &mut Receiver<ResponseChunk>,
    response_timeout: Duration,
    timeout_message: impl FnOnce() -> String,
) -> Result<Option<ResponseChunk>, Status> {
    timeout(response_timeout, receiver.recv())
        .await
        .map_err(|_| Status::deadline_exceeded(timeout_message()))
}

struct RequestAbortGuard {
    bridge: Arc<PyBridge>,
    rid: String,
    armed: bool,
}

impl RequestAbortGuard {
    fn new(bridge: Arc<PyBridge>, rid: impl Into<String>) -> Self {
        Self {
            bridge,
            rid: rid.into(),
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }

    fn abort_now(&mut self) {
        if self.armed {
            self.armed = false;
            spawn_abort(self.bridge.clone(), self.rid.clone());
        }
    }
}

impl Drop for RequestAbortGuard {
    fn drop(&mut self) {
        if self.armed {
            // Dropping a response stream means the client stopped consuming; propagate
            // cancellation to Python without blocking the Tokio worker.
            spawn_abort(self.bridge.clone(), self.rid.clone());
        }
    }
}

fn spawn_abort(bridge: Arc<PyBridge>, rid: String) {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            let _ = handle.spawn_blocking(move || {
                let _ = bridge.abort(&rid, false);
            });
        }
        Err(_) => {
            tracing::warn!(
                rid,
                "Skipping gRPC request abort because no Tokio runtime is available"
            );
        }
    }
}

async fn recv_terminal_chunk_for_request(
    bridge: &Arc<PyBridge>,
    rid: &str,
    receiver: &mut Receiver<ResponseChunk>,
    response_timeout: Duration,
) -> Result<ResponseChunk, Status> {
    let mut abort_guard = RequestAbortGuard::new(bridge.clone(), rid.to_string());

    match recv_chunk_with_timeout(receiver, response_timeout, || {
        format!("Request timed out after {}s", response_timeout.as_secs())
    })
    .await
    {
        Ok(Some(ResponseChunk::Data(_))) => {
            tracing::warn!(
                rid,
                "Unary gRPC response received non-terminal Data chunk; expected Finished"
            );
            abort_guard.abort_now();
            Err(Status::internal(
                "Unary response protocol violation: expected Finished, got Data",
            ))
        }
        Ok(Some(chunk @ (ResponseChunk::Finished(_) | ResponseChunk::Error(_)))) => {
            abort_guard.disarm();
            Ok(chunk)
        }
        Ok(None) => {
            let (status, should_abort) = closed_stream_status(bridge, rid);
            if should_abort {
                abort_guard.abort_now();
            } else {
                abort_guard.disarm();
            }
            Err(status)
        }
        Err(status) => {
            if status.code() == tonic::Code::DeadlineExceeded {
                abort_guard.abort_now();
            } else {
                abort_guard.disarm();
            }
            Err(status)
        }
    }
}

fn closed_stream_status(bridge: &Arc<PyBridge>, rid: &str) -> (Status, bool) {
    if let Some(error) = bridge.take_terminal_error(rid) {
        (terminal_error_status(error), false)
    } else {
        (
            Status::internal("gRPC response stream closed before a terminal response"),
            true,
        )
    }
}

fn terminal_error_status(error: TerminalError) -> Status {
    let message = error.message();
    match error {
        TerminalError::ChannelFull { .. } => Status::resource_exhausted(message),
        TerminalError::ClientDisconnected { .. } | TerminalError::Aborted { .. } => {
            Status::cancelled(message)
        }
    }
}

fn openai_status_code(meta_info: &HashMap<String, String>, default: i32) -> i32 {
    meta_info
        .get("status_code")
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(default)
}

#[tonic::async_trait]
impl proto::sglang_service_server::SglangService for SglangServiceImpl {
    // --- SGLang-native RPCs: TextGenerate / Generate ---

    type TextGenerateStream = StreamResult<proto::TextGenerateResponse>;

    async fn text_generate(
        &self,
        request: Request<proto::TextGenerateRequest>,
    ) -> Result<Response<Self::TextGenerateStream>, Status> {
        let req = request.into_inner();
        let rid = req
            .rid
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let req_dict = build_text_generate_dict(&rid, &req);

        let mut receiver = self
            .bridge
            .submit_request(&rid, "generate", req_dict)
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let bridge = self.bridge.clone();
        let rid_clone = rid.clone();
        let response_timeout = self.response_timeout;

        let stream = async_stream::stream! {
            let mut abort_guard = RequestAbortGuard::new(bridge.clone(), rid_clone.clone());
            loop {
                match recv_chunk_with_timeout(&mut receiver, response_timeout, || "Stream chunk timed out".to_string()).await {
                    Ok(Some(ResponseChunk::Data(data))) => {
                        yield Ok(proto::TextGenerateResponse {
                            text: data.text.unwrap_or_default(),
                            meta_info: data.meta_info,
                            finished: false,
                        });
                    }
                    Ok(Some(ResponseChunk::Finished(data))) => {
                        abort_guard.disarm();
                        yield Ok(proto::TextGenerateResponse {
                            text: data.text.unwrap_or_default(),
                            meta_info: data.meta_info,
                            finished: true,
                        });
                        break;
                    }
                    Ok(Some(ResponseChunk::Error(msg))) => {
                        abort_guard.disarm();
                        yield Err(Status::internal(msg));
                        break;
                    }
                    Ok(None) => {
                        let (status, should_abort) = closed_stream_status(&bridge, &rid_clone);
                        if should_abort {
                            abort_guard.abort_now();
                        } else {
                            abort_guard.disarm();
                        }
                        yield Err(status);
                        break;
                    }
                    Err(status) => {
                        abort_guard.abort_now();
                        yield Err(status);
                        break;
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    type GenerateStream = StreamResult<proto::GenerateResponse>;

    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let req = request.into_inner();
        let rid = req
            .rid
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let req_dict = build_generate_dict(&rid, &req);

        let mut receiver = self
            .bridge
            .submit_request(&rid, "generate", req_dict)
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let bridge = self.bridge.clone();
        let rid_clone = rid.clone();
        let response_timeout = self.response_timeout;

        let stream = async_stream::stream! {
            let mut abort_guard = RequestAbortGuard::new(bridge.clone(), rid_clone.clone());
            loop {
                match recv_chunk_with_timeout(&mut receiver, response_timeout, || "Stream chunk timed out".to_string()).await {
                    Ok(Some(ResponseChunk::Data(data))) => {
                        yield Ok(proto::GenerateResponse {
                            output_ids: data.output_ids.unwrap_or_default(),
                            meta_info: data.meta_info,
                            finished: false,
                        });
                    }
                    Ok(Some(ResponseChunk::Finished(data))) => {
                        abort_guard.disarm();
                        yield Ok(proto::GenerateResponse {
                            output_ids: data.output_ids.unwrap_or_default(),
                            meta_info: data.meta_info,
                            finished: true,
                        });
                        break;
                    }
                    Ok(Some(ResponseChunk::Error(msg))) => {
                        abort_guard.disarm();
                        yield Err(Status::internal(msg));
                        break;
                    }
                    Ok(None) => {
                        let (status, should_abort) = closed_stream_status(&bridge, &rid_clone);
                        if should_abort {
                            abort_guard.abort_now();
                        } else {
                            abort_guard.disarm();
                        }
                        yield Err(status);
                        break;
                    }
                    Err(status) => {
                        abort_guard.abort_now();
                        yield Err(status);
                        break;
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    // --- SGLang-native RPCs: Embed (text / tokenized) ---

    async fn text_embed(
        &self,
        request: Request<proto::TextEmbedRequest>,
    ) -> Result<Response<proto::TextEmbedResponse>, Status> {
        let req = request.into_inner();
        let rid = req
            .rid
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let req_dict = build_text_embed_dict(&rid, &req);

        let mut receiver = self
            .bridge
            .submit_request(&rid, "embed", req_dict)
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let chunk = recv_terminal_chunk_for_request(
            &self.bridge,
            &rid,
            &mut receiver,
            self.response_timeout,
        )
        .await?;

        match chunk {
            ResponseChunk::Data(data) | ResponseChunk::Finished(data) => {
                Ok(Response::new(proto::TextEmbedResponse {
                    embedding: data.embedding.unwrap_or_default(),
                    meta_info: data.meta_info,
                }))
            }
            ResponseChunk::Error(msg) => Err(Status::internal(msg)),
        }
    }

    async fn embed(
        &self,
        request: Request<proto::EmbedRequest>,
    ) -> Result<Response<proto::EmbedResponse>, Status> {
        let req = request.into_inner();
        let rid = req
            .rid
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let req_dict = build_embed_dict(&rid, &req);

        let mut receiver = self
            .bridge
            .submit_request(&rid, "embed", req_dict)
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let chunk = recv_terminal_chunk_for_request(
            &self.bridge,
            &rid,
            &mut receiver,
            self.response_timeout,
        )
        .await?;

        match chunk {
            ResponseChunk::Data(data) | ResponseChunk::Finished(data) => {
                Ok(Response::new(proto::EmbedResponse {
                    embedding: data.embedding.unwrap_or_default(),
                    meta_info: data.meta_info,
                }))
            }
            ResponseChunk::Error(msg) => Err(Status::internal(msg)),
        }
    }

    // --- SGLang-native RPCs: Classify ---

    async fn classify(
        &self,
        request: Request<proto::ClassifyRequest>,
    ) -> Result<Response<proto::ClassifyResponse>, Status> {
        let req = request.into_inner();
        if req.text.is_empty() && req.input_ids.is_empty() {
            return Err(Status::invalid_argument(
                "Classify requires either text or input_ids",
            ));
        }
        let rid = req
            .rid
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let req_dict = build_classify_dict(&rid, &req);

        let mut receiver = self
            .bridge
            .submit_request(&rid, "embed", req_dict)
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let chunk = recv_terminal_chunk_for_request(
            &self.bridge,
            &rid,
            &mut receiver,
            self.response_timeout,
        )
        .await?;

        match chunk {
            ResponseChunk::Data(data) | ResponseChunk::Finished(data) => {
                Ok(Response::new(proto::ClassifyResponse {
                    embedding: data.embedding.unwrap_or_default(),
                    meta_info: data.meta_info,
                }))
            }
            ResponseChunk::Error(msg) => Err(Status::internal(msg)),
        }
    }

    // --- SGLang-native RPCs: Tokenize / Detokenize (Rust-native with fallback) ---

    async fn tokenize(
        &self,
        request: Request<proto::TokenizeRequest>,
    ) -> Result<Response<proto::TokenizeResponse>, Status> {
        let req = request.into_inner();
        let add_special = req.add_special_tokens.unwrap_or(true);

        // Try Rust-native tokenizer first (no GIL)
        if let Some(tok) = self.bridge.rust_tokenizer() {
            let tokens = tok
                .encode(&req.text, add_special)
                .map_err(Status::internal)?;
            let count = tokens.len() as i32;
            return Ok(Response::new(proto::TokenizeResponse {
                tokens: tokens.iter().map(|&t| t as i32).collect(),
                count,
                max_model_len: self.bridge.context_len(),
                input_text: req.text,
            }));
        }

        // Fallback to Python
        let json_str = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            let text = req.text.clone();
            move || bridge.tokenize_py(&text, add_special)
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| pyerr_to_status(e, "Tokenize failed"))?;

        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::TokenizeResponse {
            tokens: v["tokens"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_i64().map(|n| n as i32))
                        .collect()
                })
                .unwrap_or_default(),
            count: v["count"].as_i64().unwrap_or(0) as i32,
            max_model_len: self.bridge.context_len(),
            input_text: req.text,
        }))
    }

    async fn detokenize(
        &self,
        request: Request<proto::DetokenizeRequest>,
    ) -> Result<Response<proto::DetokenizeResponse>, Status> {
        let req = request.into_inner();
        if req.tokens.iter().any(|&token| token < 0) {
            return Err(Status::invalid_argument(
                "Detokenize tokens must be non-negative",
            ));
        }

        // Try Rust-native tokenizer first (no GIL)
        if let Some(tok) = self.bridge.rust_tokenizer() {
            let ids: Vec<u32> = req.tokens.iter().map(|&t| t as u32).collect();
            let text = tok.decode(&ids, true).map_err(Status::internal)?;
            return Ok(Response::new(proto::DetokenizeResponse { text }));
        }

        // Fallback to Python
        let json_str = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            let tokens = req.tokens;
            move || bridge.detokenize_py(tokens)
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| pyerr_to_status(e, "Detokenize failed"))?;

        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::DetokenizeResponse {
            text: v["text"].as_str().unwrap_or("").to_string(),
        }))
    }

    // --- SGLang-native RPCs: Info / control ---

    async fn health_check(
        &self,
        _request: Request<proto::HealthCheckRequest>,
    ) -> Result<Response<proto::HealthCheckResponse>, Status> {
        let healthy = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            move || bridge.health_check()
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| pyerr_to_status(e, "Health check failed"))?;

        Ok(Response::new(proto::HealthCheckResponse { healthy }))
    }

    async fn get_model_info(
        &self,
        _request: Request<proto::GetModelInfoRequest>,
    ) -> Result<Response<proto::GetModelInfoResponse>, Status> {
        let json_info = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            move || bridge.get_model_info()
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| pyerr_to_status(e, "Failed to get model info"))?;

        Ok(Response::new(proto::GetModelInfoResponse {
            model_path: extract_model_path(&json_info),
            json_info,
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<proto::GetServerInfoRequest>,
    ) -> Result<Response<proto::GetServerInfoResponse>, Status> {
        let json_info = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            move || bridge.get_server_info()
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| pyerr_to_status(e, "Failed to get server info"))?;

        Ok(Response::new(proto::GetServerInfoResponse { json_info }))
    }

    async fn list_models(
        &self,
        _request: Request<proto::ListModelsRequest>,
    ) -> Result<Response<proto::ListModelsResponse>, Status> {
        let json_str = tokio::task::spawn_blocking({
            let bridge = self.bridge.clone();
            move || bridge.list_models()
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
        .map_err(|e| pyerr_to_status(e, "Failed to list models"))?;

        let models_arr: Vec<serde_json::Value> = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse models JSON: {}", e)))?;

        let models = models_arr
            .iter()
            .map(|m| proto::ModelCard {
                id: m["id"].as_str().unwrap_or("").to_string(),
                root: m["root"].as_str().unwrap_or("").to_string(),
                parent: m.get("parent").and_then(|v| v.as_str()).map(String::from),
                max_model_len: m
                    .get("max_model_len")
                    .and_then(|v| v.as_i64())
                    .map(|n| n as i32),
            })
            .collect();

        Ok(Response::new(proto::ListModelsResponse { models }))
    }

    async fn get_load(
        &self,
        request: Request<proto::GetLoadRequest>,
    ) -> Result<Response<proto::GetLoadResponse>, Status> {
        let req = request.into_inner();
        let rid = uuid::Uuid::new_v4().to_string();
        let receiver = self
            .bridge
            .submit_get_load(&rid, req.dp_rank)
            .map_err(|e| pyerr_to_status(e, "Failed to get load"))?;

        let json_info =
            recv_json_response(&self.bridge, &rid, receiver, self.response_timeout).await?;
        Ok(Response::new(proto::GetLoadResponse { json_info }))
    }

    async fn abort(
        &self,
        request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        let req = request.into_inner();
        if !req.abort_all && req.rid.trim().is_empty() {
            return Err(Status::invalid_argument(
                "Abort requires a non-empty rid unless abort_all is true",
            ));
        }
        if req.abort_all {
            tracing::warn!(
                "Received abort_all over gRPC; this endpoint must only be exposed to trusted clients"
            );
        }
        self.bridge
            .abort(&req.rid, req.abort_all)
            .map_err(|e| pyerr_to_status(e, "Failed to abort"))?;

        Ok(Response::new(proto::AbortResponse { success: true }))
    }

    async fn flush_cache(
        &self,
        _request: Request<proto::FlushCacheRequest>,
    ) -> Result<Response<proto::FlushCacheResponse>, Status> {
        let rid = uuid::Uuid::new_v4().to_string();
        let receiver = self
            .bridge
            .submit_flush_cache(&rid)
            .map_err(|e| pyerr_to_status(e, "Failed to flush cache"))?;

        let json_str =
            recv_json_response(&self.bridge, &rid, receiver, self.response_timeout).await?;
        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::FlushCacheResponse {
            success: v["success"].as_bool().unwrap_or(false),
            message: v["message"].as_str().unwrap_or("").to_string(),
        }))
    }

    async fn pause_generation(
        &self,
        request: Request<proto::PauseGenerationRequest>,
    ) -> Result<Response<proto::PauseGenerationResponse>, Status> {
        let req = request.into_inner();
        let rid = uuid::Uuid::new_v4().to_string();
        let receiver = self
            .bridge
            .submit_pause_generation(&rid, &req.mode)
            .map_err(|e| pyerr_to_status(e, "Failed to pause generation"))?;

        let json_str =
            recv_json_response(&self.bridge, &rid, receiver, self.response_timeout).await?;
        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::PauseGenerationResponse {
            message: v["message"].as_str().unwrap_or("").to_string(),
        }))
    }

    async fn continue_generation(
        &self,
        _request: Request<proto::ContinueGenerationRequest>,
    ) -> Result<Response<proto::ContinueGenerationResponse>, Status> {
        let rid = uuid::Uuid::new_v4().to_string();
        let receiver = self
            .bridge
            .submit_continue_generation(&rid)
            .map_err(|e| pyerr_to_status(e, "Failed to continue generation"))?;

        let json_str =
            recv_json_response(&self.bridge, &rid, receiver, self.response_timeout).await?;
        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::ContinueGenerationResponse {
            message: v["message"].as_str().unwrap_or("").to_string(),
        }))
    }

    // --- OpenAI-compatible RPCs (JSON pass-through) ---

    type ChatCompleteStream = StreamResult<proto::OpenAiStreamChunk>;

    async fn chat_complete(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<Self::ChatCompleteStream>, Status> {
        self.openai_streaming_rpc(request, "submit_openai_chat")
            .await
    }

    type CompleteStream = StreamResult<proto::OpenAiStreamChunk>;

    async fn complete(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<Self::CompleteStream>, Status> {
        self.openai_streaming_rpc(request, "submit_openai_complete")
            .await
    }

    async fn open_ai_embed(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        self.openai_unary_rpc(request, "submit_openai_embed").await
    }

    async fn open_ai_classify(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        self.openai_unary_rpc(request, "submit_openai_classify")
            .await
    }

    async fn score(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        self.openai_unary_rpc(request, "submit_openai_score").await
    }

    async fn rerank(
        &self,
        request: Request<proto::OpenAiRequest>,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        self.openai_unary_rpc(request, "submit_openai_rerank").await
    }

    // --- Admin RPCs ---

    async fn start_profile(
        &self,
        request: Request<proto::StartProfileRequest>,
    ) -> Result<Response<proto::StartProfileResponse>, Status> {
        let req = request.into_inner();
        let rid = uuid::Uuid::new_v4().to_string();
        let receiver = self
            .bridge
            .submit_start_profile(&rid, req.output_dir.as_deref())
            .map_err(|e| pyerr_to_status(e, "Failed to start profile"))?;

        let json_str =
            recv_json_response(&self.bridge, &rid, receiver, self.response_timeout).await?;
        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::StartProfileResponse {
            message: v["message"].as_str().unwrap_or("").to_string(),
        }))
    }

    async fn stop_profile(
        &self,
        _request: Request<proto::StopProfileRequest>,
    ) -> Result<Response<proto::StopProfileResponse>, Status> {
        let rid = uuid::Uuid::new_v4().to_string();
        let receiver = self
            .bridge
            .submit_stop_profile(&rid)
            .map_err(|e| pyerr_to_status(e, "Failed to stop profile"))?;

        let json_str =
            recv_json_response(&self.bridge, &rid, receiver, self.response_timeout).await?;
        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::StopProfileResponse {
            message: v["message"].as_str().unwrap_or("").to_string(),
        }))
    }

    async fn update_weights_from_disk(
        &self,
        request: Request<proto::UpdateWeightsRequest>,
    ) -> Result<Response<proto::UpdateWeightsResponse>, Status> {
        let req = request.into_inner();
        let rid = uuid::Uuid::new_v4().to_string();
        let receiver = self
            .bridge
            .submit_update_weights(&rid, &req.model_path, req.load_format.as_deref())
            .map_err(|e| pyerr_to_status(e, "Failed to update weights"))?;

        let json_str =
            recv_json_response(&self.bridge, &rid, receiver, self.response_timeout).await?;
        let v: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| Status::internal(format!("Failed to parse JSON response: {}", e)))?;
        Ok(Response::new(proto::UpdateWeightsResponse {
            success: v["success"].as_bool().unwrap_or(false),
            message: v["message"].as_str().unwrap_or("").to_string(),
        }))
    }
}

// Helper methods for OpenAI pass-through RPCs.
impl SglangServiceImpl {
    async fn openai_streaming_rpc(
        &self,
        request: Request<proto::OpenAiRequest>,
        method_name: &str,
    ) -> Result<Response<StreamResult<proto::OpenAiStreamChunk>>, Status> {
        let req = request.into_inner();
        let rid = uuid::Uuid::new_v4().to_string();

        let mut receiver = self
            .bridge
            .submit_openai(&rid, method_name, &req.json_body, &req.trace_headers)
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let bridge = self.bridge.clone();
        let rid_clone = rid.clone();
        let response_timeout = self.response_timeout;

        let stream = async_stream::stream! {
            let mut abort_guard = RequestAbortGuard::new(bridge.clone(), rid_clone.clone());
            loop {
                match recv_chunk_with_timeout(&mut receiver, response_timeout, || "Stream chunk timed out".to_string()).await {
                    Ok(Some(ResponseChunk::Data(data))) => {
                        yield Ok(proto::OpenAiStreamChunk {
                            json_chunk: data.json_bytes.unwrap_or_default(),
                            finished: false,
                        });
                    }
                    Ok(Some(ResponseChunk::Finished(data))) => {
                        let bytes = data.json_bytes.unwrap_or_default();
                        abort_guard.disarm();
                        yield Ok(proto::OpenAiStreamChunk {
                            json_chunk: bytes,
                            finished: true,
                        });
                        break;
                    }
                    Ok(Some(ResponseChunk::Error(msg))) => {
                        abort_guard.disarm();
                        yield Err(Status::internal(msg));
                        break;
                    }
                    Ok(None) => {
                        let (status, should_abort) = closed_stream_status(&bridge, &rid_clone);
                        if should_abort {
                            abort_guard.abort_now();
                        } else {
                            abort_guard.disarm();
                        }
                        yield Err(status);
                        break;
                    }
                    Err(status) => {
                        abort_guard.abort_now();
                        yield Err(status);
                        break;
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    async fn openai_unary_rpc(
        &self,
        request: Request<proto::OpenAiRequest>,
        method_name: &str,
    ) -> Result<Response<proto::OpenAiResponse>, Status> {
        let req = request.into_inner();
        let rid = uuid::Uuid::new_v4().to_string();

        let mut receiver = self
            .bridge
            .submit_openai(&rid, method_name, &req.json_body, &req.trace_headers)
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

        let chunk = recv_terminal_chunk_for_request(
            &self.bridge,
            &rid,
            &mut receiver,
            self.response_timeout,
        )
        .await?;

        match chunk {
            ResponseChunk::Data(data) | ResponseChunk::Finished(data) => {
                Ok(Response::new(proto::OpenAiResponse {
                    json_body: data.json_bytes.unwrap_or_default(),
                    status_code: openai_status_code(&data.meta_info, 200),
                }))
            }
            ResponseChunk::Error(msg) => {
                let error_json = serde_json::json!({"error": {"message": msg}});
                Ok(Response::new(proto::OpenAiResponse {
                    json_body: error_json.to_string().into_bytes(),
                    status_code: 500,
                }))
            }
        }
    }
}

/// Receive a single JSON response from the bridge channel.
async fn recv_json_response(
    bridge: &Arc<PyBridge>,
    rid: &str,
    mut receiver: Receiver<ResponseChunk>,
    response_timeout: Duration,
) -> Result<String, Status> {
    let chunk =
        recv_terminal_chunk_for_request(bridge, rid, &mut receiver, response_timeout).await?;

    match chunk {
        ResponseChunk::Data(data) | ResponseChunk::Finished(data) => {
            let bytes = data.json_bytes.unwrap_or_default();
            String::from_utf8(bytes)
                .map_err(|e| Status::internal(format!("Invalid UTF-8 in response: {}", e)))
        }
        ResponseChunk::Error(msg) => Err(Status::internal(msg)),
    }
}

/// Start the Tonic gRPC server on the given address.
//
// TODO(grpc-auth): this listener is currently unauthenticated. Before exposing
// it in any default deploy path, gate it with the same API-key / admin-key
// checks the HTTP server applies (see issue tracking gRPC auth parity).
pub async fn run_grpc_server(
    listener: std::net::TcpListener,
    bridge: Arc<PyBridge>,
    shutdown: Arc<Notify>,
    response_timeout: Duration,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr = listener.local_addr()?;
    let listener = tokio::net::TcpListener::from_std(listener)?;
    let service = SglangServiceImpl {
        bridge,
        response_timeout,
    };

    let max_message_size = resolve_max_message_size();
    let svc = proto::sglang_service_server::SglangServiceServer::new(service)
        .max_decoding_message_size(max_message_size)
        .max_encoding_message_size(max_message_size);

    tracing::info!("gRPC server listening on {}", addr);

    tonic::transport::Server::builder()
        .add_service(svc)
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
            shutdown.notified().await;
            tracing::info!("gRPC server shutting down");
        })
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests;
