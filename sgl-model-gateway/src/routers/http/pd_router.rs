use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use bytes::Bytes;
use futures_util::StreamExt;
use memchr::memmem;
use reqwest::Client;
use serde_json::{json, Value};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, warn};

use super::pd_types::api_path;
use crate::{
    config::types::RetryConfig,
    core::{
        is_retryable_status, HashRing, RetryExecutor, Worker, WorkerLoadGuard, WorkerRegistry,
        WorkerType, UNKNOWN_MODEL_ID,
    },
    observability::{
        events::{self, Event},
        metrics::{bool_to_static_str, metrics_labels, Metrics},
        otel_trace::inject_trace_context_http,
    },
    policies::{LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo},
    protocols::{
        classify::ClassifyRequest, common::StringOrArray, completion::CompletionRequest,
        embedding::EmbeddingRequest, rerank::RerankRequest,
    },
    routers::{
        error,
        grpc::utils::{error_type_from_status, route_to_endpoint},
        header_utils,
        http::routing_view::{validate_extensions_in_value, ChatRoutingView, GenerateRoutingView},
        RouterTrait,
    },
};

#[derive(Debug)]
pub struct PDRouter {
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub client: Client,
    pub retry_config: RetryConfig,
    pub api_key: Option<String>,
    pub enable_igw: bool,
}

#[derive(Clone)]
struct PDRequestContext<'a> {
    route: &'static str,
    batch_size: Option<usize>,
    is_stream: bool,
    return_logprob: bool,
    /// SGLang RL extension. When true and `is_stream` is false, PD
    /// merges `routed_experts` from the prefill response into the
    /// decode response (base64 prefix-suffix concat).
    return_routed_experts: bool,
    request_text: Option<String>,
    model_id: Option<&'a str>,
    headers: Option<HeaderMap>,
}

impl PDRequestContext<'_> {
    /// Whether the response merge step needs to walk the prefill JSON.
    /// Logprobs and routed_experts both live there.
    fn needs_prefill_json_merge(&self) -> bool {
        !self.is_stream && (self.return_logprob || self.return_routed_experts)
    }
}

impl PDRouter {
    async fn proxy_to_first_prefill_worker(
        &self,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let workers = self.worker_registry.get_prefill_workers();
        let first_worker_url = workers.first().map(|w| w.url().to_string());

        if let Some(worker_url) = first_worker_url {
            self.proxy_to_worker(worker_url, endpoint, headers).await
        } else {
            error::service_unavailable("no_prefill_servers", "No prefill servers available")
        }
    }

    async fn proxy_to_worker(
        &self,
        worker_url: String,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let url = format!("{}/{}", worker_url, endpoint);
        let mut request_builder = self.client.get(&url);

        if let Some(headers) = headers {
            for (name, value) in headers {
                request_builder = request_builder.header(name, value);
            }
        }

        match request_builder.send().await {
            Ok(res) if res.status().is_success() => {
                let response_headers = header_utils::preserve_response_headers(res.headers());

                match res.bytes().await {
                    Ok(body) => {
                        let mut response = Response::new(Body::from(body));
                        *response.status_mut() = StatusCode::OK;
                        *response.headers_mut() = response_headers;
                        response
                    }
                    Err(e) => {
                        error!("Failed to read response body: {}", e);
                        error::internal_error(
                            "read_response_body_failed",
                            format!("Failed to read response body: {}", e),
                        )
                    }
                }
            }
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                // Use the status code to determine which error function to use
                match status {
                    StatusCode::BAD_REQUEST => error::bad_request(
                        "server_bad_request",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::NOT_FOUND => error::not_found(
                        "server_not_found",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                        "server_internal_error",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                        "server_unavailable",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::BAD_GATEWAY => error::bad_gateway(
                        "server_bad_gateway",
                        format!("Server returned status: {}", res.status()),
                    ),
                    _ => error::internal_error(
                        "server_error",
                        format!("Server returned status: {}", res.status()),
                    ),
                }
            }
            Err(e) => {
                error!("Failed to proxy request server: {}", e);
                error::internal_error(
                    "proxy_request_failed",
                    format!("Failed to proxy request: {}", e),
                )
            }
        }
    }

    pub async fn new(ctx: &Arc<crate::app_context::AppContext>) -> Result<Self, String> {
        Ok(PDRouter {
            worker_registry: Arc::clone(&ctx.worker_registry),
            policy_registry: Arc::clone(&ctx.policy_registry),
            client: ctx.client.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            api_key: ctx.router_config.api_key.clone(),
            enable_igw: ctx.router_config.enable_igw,
        })
    }

    fn handle_server_selection_error(error: String) -> Response {
        error!("Failed to select PD pair error={}", error);
        error::service_unavailable(
            "server_selection_failed",
            format!("No available servers: {}", error),
        )
    }

    fn handle_serialization_error(error: impl std::fmt::Display) -> Response {
        error!("Failed to serialize request error={}", error);
        error::internal_error("serialization_failed", "Failed to serialize request")
    }

    // Batch-size derivation lives in the free functions
    // `generate_batch_size_from_value` / `chat_batch_size_from_value`
    // below so PD can read them off the same `Value` it parses for
    // bootstrap injection.

    fn get_completion_batch_size(req: &CompletionRequest) -> Option<usize> {
        if let StringOrArray::Array(arr) = &req.prompt {
            if !arr.is_empty() {
                return Some(arr.len());
            }
        }
        None
    }

    // Static key strings to avoid per-request allocations
    const BOOTSTRAP_HOST_KEY: &'static str = "bootstrap_host";
    const BOOTSTRAP_PORT_KEY: &'static str = "bootstrap_port";
    const BOOTSTRAP_ROOM_KEY: &'static str = "bootstrap_room";

    fn inject_bootstrap_into_value(
        mut original: Value,
        prefill_worker: &dyn Worker,
        batch_size: Option<usize>,
    ) -> Result<Value, String> {
        let obj = original
            .as_object_mut()
            .ok_or_else(|| "Request must be a JSON object".to_string())?;

        if let Some(n) = batch_size {
            let mut hosts = Vec::with_capacity(n);
            let mut ports = Vec::with_capacity(n);
            let mut rooms = Vec::with_capacity(n);
            for _ in 0..n {
                hosts.push(prefill_worker.bootstrap_host());
                ports.push(prefill_worker.bootstrap_port());
                rooms.push(super::pd_types::generate_room_id());
            }
            // Use static string keys to avoid per-request allocations
            obj.insert(
                Self::BOOTSTRAP_HOST_KEY.to_string(),
                Value::Array(hosts.into_iter().map(Value::from).collect()),
            );
            obj.insert(
                Self::BOOTSTRAP_PORT_KEY.to_string(),
                Value::Array(
                    ports
                        .into_iter()
                        .map(|p| match p {
                            Some(v) => Value::from(v),
                            None => Value::Null,
                        })
                        .collect(),
                ),
            );
            obj.insert(
                Self::BOOTSTRAP_ROOM_KEY.to_string(),
                Value::Array(rooms.into_iter().map(Value::from).collect()),
            );
        } else {
            // Use static string keys to avoid per-request allocations
            obj.insert(
                Self::BOOTSTRAP_HOST_KEY.to_string(),
                Value::from(prefill_worker.bootstrap_host()),
            );
            obj.insert(
                Self::BOOTSTRAP_PORT_KEY.to_string(),
                match prefill_worker.bootstrap_port() {
                    Some(v) => Value::from(v),
                    None => Value::Null,
                },
            );
            obj.insert(
                Self::BOOTSTRAP_ROOM_KEY.to_string(),
                Value::from(super::pd_types::generate_room_id()),
            );
        }
        Ok(original)
    }

    async fn execute_dual_dispatch(
        &self,
        headers: Option<&HeaderMap>,
        original_request: Value,
        context: PDRequestContext<'_>,
    ) -> Response {
        let start_time = Instant::now();

        let route = context.route;
        let model = context.model_id.unwrap_or(UNKNOWN_MODEL_ID);
        let endpoint = route_to_endpoint(route);

        // Record request start (Layer 2)
        record_pd_request_start(&context);
        // Each retry needs a fresh Value because `inject_bootstrap_into_value`
        // mutates in place. The serde round-trip lives at the typed call
        // sites (`route_completion`, `route_rerank`) once before this
        // dispatch; Value callers (`route_generate`, `route_chat`) hand
        // their parsed body straight in. Wrapping in `Arc` only saves
        // an extra `Value` clone at dispatch entry.
        let shared_request = Arc::new(original_request);
        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            {
                move |attempt: u32| {
                    let shared_request = Arc::clone(&shared_request);
                    let context = context.clone();
                    async move {
                        let (prefill, decode) = match self
                            .select_pd_pair(
                                context.request_text.as_deref(),
                                context.model_id,
                                context.headers.as_ref(),
                            )
                            .await
                        {
                            Ok(pair) => pair,
                            Err(e) => {
                                return Self::handle_server_selection_error(e);
                            }
                        };

                        debug!(
                            "PD retry attempt {} using prefill={} decode={}",
                            attempt,
                            prefill.url(),
                            decode.url()
                        );

                        // Deep-clone the pre-serialized Value once per
                        // retry; bootstrap injection mutates and
                        // consumes it.
                        let mut json_request = (*shared_request).clone();

                        json_request = match Self::inject_bootstrap_into_value(
                            json_request,
                            prefill.as_ref(),
                            context.batch_size,
                        ) {
                            Ok(v) => v,
                            Err(e) => return Self::handle_serialization_error(e),
                        };

                        let response = self
                            .execute_dual_dispatch_internal(
                                headers,
                                json_request,
                                context,
                                Arc::clone(&prefill),
                                Arc::clone(&decode),
                                start_time,
                            )
                            .await;

                        let status = response.status();
                        let not_error = status.is_success() || status.is_client_error();
                        prefill.record_outcome(not_error);
                        decode.record_outcome(not_error);

                        // Record worker errors for server errors (5xx)
                        if status.is_server_error() {
                            let error_type = error_type_from_status(status);
                            Metrics::record_worker_error(
                                metrics_labels::WORKER_PREFILL,
                                metrics_labels::CONNECTION_HTTP,
                                error_type,
                            );
                            Metrics::record_worker_error(
                                metrics_labels::WORKER_DECODE,
                                metrics_labels::CONNECTION_HTTP,
                                error_type,
                            );
                        }

                        response
                    }
                }
            },
            |res, _attempt| is_retryable_status(res.status()),
            |delay, attempt| {
                // Layer 3 worker metrics (PD mode uses both prefill and decode workers)
                Metrics::record_worker_retry(metrics_labels::WORKER_PREFILL, endpoint);
                Metrics::record_worker_retry(metrics_labels::WORKER_DECODE, endpoint);
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_PREFILL, endpoint);
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_DECODE, endpoint);
            },
        )
        .await;

        // Record Layer 2 metrics
        let duration = start_time.elapsed();
        if response.status().is_success() {
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else if !is_retryable_status(response.status()) {
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }

        response
    }

    async fn handle_decode_error_response(
        &self,
        res: reqwest::Response,
        context: &PDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
    ) -> Response {
        let status = res.status();

        if context.is_stream {
            // Handle streaming error response
            let response_headers = header_utils::preserve_response_headers(res.headers());
            let error_payload = match res.bytes().await {
                Ok(error_body) => {
                    if let Ok(error_json) = serde_json::from_slice::<Value>(&error_body) {
                        json!({ "message": error_json, "status": status.as_u16() })
                    } else {
                        json!({ "message": String::from_utf8_lossy(&error_body).to_string(), "status": status.as_u16() })
                    }
                }
                Err(e) => {
                    json!({ "message": format!("Decode server error: {}", e), "status": status.as_u16() })
                }
            };

            let sse_data = format!(
                "data: {{'error': {}}}",
                serde_json::to_string(&error_payload).unwrap_or_default()
            );
            let error_stream = tokio_stream::once(Ok(axum::body::Bytes::from(sse_data)));

            let decode_url = decode.url().to_string();
            self.create_streaming_response(
                error_stream,
                status,
                None,
                context.return_logprob,
                Some(decode_url),
                Some(response_headers),
                prefill,
                decode,
            )
        } else {
            // Handle non-streaming error response
            match res.bytes().await {
                Ok(error_body) => {
                    // Try to parse error message from body, fallback to status-based error
                    let error_message = if let Ok(error_json) =
                        serde_json::from_slice::<Value>(&error_body)
                    {
                        if let Some(msg) = error_json
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                        {
                            msg.to_string()
                        } else if let Some(msg) = error_json.get("message").and_then(|m| m.as_str())
                        {
                            msg.to_string()
                        } else {
                            String::from_utf8_lossy(&error_body).to_string()
                        }
                    } else {
                        String::from_utf8_lossy(&error_body).to_string()
                    };

                    let status_code = StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match status_code {
                        StatusCode::BAD_REQUEST => {
                            error::bad_request("decode_bad_request", error_message)
                        }
                        StatusCode::NOT_FOUND => {
                            error::not_found("decode_not_found", error_message)
                        }
                        StatusCode::INTERNAL_SERVER_ERROR => {
                            error::internal_error("decode_internal_error", error_message)
                        }
                        StatusCode::SERVICE_UNAVAILABLE => {
                            error::service_unavailable("decode_unavailable", error_message)
                        }
                        StatusCode::BAD_GATEWAY => {
                            error::bad_gateway("decode_bad_gateway", error_message)
                        }
                        _ => error::internal_error("decode_error", error_message),
                    }
                }
                Err(e) => {
                    let error_message = format!("Decode server error: {}", e);
                    let status_code = StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match status_code {
                        StatusCode::BAD_REQUEST => {
                            error::bad_request("decode_read_failed", error_message)
                        }
                        StatusCode::NOT_FOUND => {
                            error::not_found("decode_read_failed", error_message)
                        }
                        StatusCode::INTERNAL_SERVER_ERROR => {
                            error::internal_error("decode_read_failed", error_message)
                        }
                        StatusCode::SERVICE_UNAVAILABLE => {
                            error::service_unavailable("decode_read_failed", error_message)
                        }
                        StatusCode::BAD_GATEWAY => {
                            error::bad_gateway("decode_read_failed", error_message)
                        }
                        _ => error::internal_error("decode_read_failed", error_message),
                    }
                }
            }
        }
    }

    // Internal method that performs the actual dual dispatch (without retry logic)
    async fn execute_dual_dispatch_internal(
        &self,
        headers: Option<&HeaderMap>,
        json_request: Value,
        context: PDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        _start_time: Instant,
    ) -> Response {
        // For non-streaming: use guard for automatic load management
        // For streaming: load will be managed in create_streaming_response
        let _prefill_guard =
            (!context.is_stream).then(|| WorkerLoadGuard::new(prefill.clone(), headers));
        let _decode_guard =
            (!context.is_stream).then(|| WorkerLoadGuard::new(decode.clone(), headers));

        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        let headers = Some(&headers_with_trace);

        // Build both requests
        let prefill_request = self.build_post_with_headers(
            &self.client,
            prefill.url(),
            context.route,
            &json_request,
            headers,
            false,
        );
        let decode_request = self.build_post_with_headers(
            &self.client,
            decode.url(),
            context.route,
            &json_request,
            headers,
            false,
        );

        // Send both requests concurrently and wait for both
        // Note: Using borrowed references avoids heap allocation
        events::RequestPDSentEvent {
            prefill_url: prefill.url(),
            decode_url: decode.url(),
        }
        .emit();

        let (prefill_result, decode_result) =
            tokio::join!(prefill_request.send(), decode_request.send());

        events::RequestReceivedEvent {}.emit();

        // Process decode response
        match decode_result {
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                debug!("Decode response status: {}", status);

                if !status.is_success() {
                    error!(
                        "Decode server returned error status decode_url={} status={}",
                        decode.url(),
                        status
                    );

                    return self
                        .handle_decode_error_response(res, &context, prefill, decode)
                        .await;
                }

                // Process prefill response. We need the prefill body
                // when *any* prefill-side merge applies (logprobs or
                // routed_experts).
                let needs_prefill_json = context.needs_prefill_json_merge();
                let prefill_body = match self
                    .process_prefill_response(prefill_result, prefill.url(), needs_prefill_json)
                    .await
                {
                    Ok((_, body)) => body,
                    Err(error_response) => return error_response,
                };

                if context.is_stream {
                    // Streaming response
                    let prefill_logprobs = if context.return_logprob {
                        prefill_body
                            .as_ref()
                            .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                            .and_then(|json| {
                                json.pointer("/meta_info/input_token_logprobs").cloned()
                            })
                    } else {
                        None
                    };

                    let response_headers = header_utils::preserve_response_headers(res.headers());

                    self.create_streaming_response(
                        res.bytes_stream(),
                        status,
                        prefill_logprobs,
                        context.return_logprob,
                        None,
                        Some(response_headers),
                        prefill,
                        decode,
                    )
                } else {
                    // Non-streaming response
                    if context.needs_prefill_json_merge() {
                        self.process_non_streaming_response(
                            res,
                            status,
                            context.return_logprob,
                            context.return_routed_experts,
                            prefill_body,
                        )
                        .await
                    } else {
                        // Direct passthrough when no logprobs needed
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(decode_body) => {
                                let mut response = Response::new(Body::from(decode_body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => {
                                error!("Failed to read decode response: {}", e);
                                error::internal_error(
                                    "read_response_failed",
                                    "Failed to read response",
                                )
                            }
                        }
                    }
                }
            }
            Err(e) => {
                error!(
                    decode_url = %decode.url(),
                    error = %e,
                    "Decode request failed"
                );
                error::bad_gateway("decode_server_error", format!("Decode server error: {}", e))
            }
        }
    }

    fn policies_need_request_text(&self) -> bool {
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        prefill_policy.needs_request_text() || decode_policy.needs_request_text()
    }

    async fn select_pd_pair(
        &self,
        request_text: Option<&str>,
        model_id: Option<&str>,
        headers: Option<&HeaderMap>,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), String> {
        let effective_model_id = if !self.enable_igw { None } else { model_id };

        debug!(
            "Selecting PD pair: enable_igw={}, model_id={:?}, effective_model_id={:?}",
            self.enable_igw, model_id, effective_model_id
        );

        let prefill_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model(model)
                .iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }))
                .cloned()
                .collect()
        } else {
            self.worker_registry.get_prefill_workers()
        };

        let decode_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model(model)
                .iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Decode))
                .cloned()
                .collect()
        } else {
            self.worker_registry.get_decode_workers()
        };

        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        // Get cached hash ring for consistent hashing
        let hash_ring = self
            .worker_registry
            .get_hash_ring(effective_model_id.unwrap_or(UNKNOWN_MODEL_ID));

        let prefill = Self::pick_worker_by_policy_arc(
            &prefill_workers,
            &*prefill_policy,
            request_text,
            headers,
            hash_ring.clone(),
            "prefill",
        )
        .await?;

        let decode = Self::pick_worker_by_policy_arc(
            &decode_workers,
            &*decode_policy,
            request_text,
            headers,
            hash_ring,
            "decode",
        )
        .await?;

        // Record worker selection metrics (Layer 3)
        let model = model_id.unwrap_or(UNKNOWN_MODEL_ID);
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            metrics_labels::CONNECTION_HTTP,
            model,
            prefill_policy.name(),
        );
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            metrics_labels::CONNECTION_HTTP,
            model,
            decode_policy.name(),
        );

        Ok((prefill, decode))
    }

    async fn pick_worker_by_policy_arc(
        workers: &[Arc<dyn Worker>],
        policy: &dyn LoadBalancingPolicy,
        request_text: Option<&str>,
        headers: Option<&HeaderMap>,
        hash_ring: Option<Arc<HashRing>>,
        worker_type: &str,
    ) -> Result<Arc<dyn Worker>, String> {
        if workers.is_empty() {
            return Err(format!(
                "No {} workers available. Please check if {} servers are configured and healthy.",
                worker_type, worker_type
            ));
        }

        let available_workers: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        if available_workers.is_empty() {
            return Err(format!(
                "No available {} workers (all circuits open or unhealthy)",
                worker_type
            ));
        }

        let selected_idx = policy
            .select_worker(
                &available_workers,
                &SelectWorkerInfo {
                    request_text,
                    tokens: None, // HTTP doesn't have tokens, use gRPC for PrefixHash
                    headers,
                    hash_ring,
                },
            )
            .await
            .ok_or_else(|| {
                format!(
                    "Policy {} failed to select a {} worker",
                    policy.name(),
                    worker_type
                )
            })?;

        Ok(available_workers[selected_idx].clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn create_streaming_response(
        &self,
        stream: impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
        status: StatusCode,
        prefill_logprobs: Option<Value>,
        return_logprob: bool,
        decode_url: Option<String>,
        headers: Option<HeaderMap>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
    ) -> Response {
        use crate::core::AttachedBody;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            futures_util::pin_mut!(stream);
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        let is_done = memmem::find(&chunk, b"data: [DONE]").is_some();

                        let result = if return_logprob && prefill_logprobs.is_some() {
                            Self::merge_streaming_logprobs(prefill_logprobs.clone(), &chunk)
                                .unwrap_or(chunk)
                        } else {
                            chunk
                        };

                        if tx.send(Ok(result)).is_err() {
                            break;
                        }

                        if is_done {
                            break;
                        }
                    }
                    Err(e) => {
                        if let Some(ref url) = decode_url {
                            error!("Stream error from decode server {}: {}", url, e);
                        }
                        let _ = tx.send(Err(format!("Stream error: {}", e)));
                        break;
                    }
                }
            }
        });

        let stream = UnboundedReceiverStream::new(rx);
        let body = Body::from_stream(stream);

        let guards = vec![
            WorkerLoadGuard::new(prefill, headers.as_ref()),
            WorkerLoadGuard::new(decode, headers.as_ref()),
        ];

        let mut response = Response::new(body);
        *response.status_mut() = status;

        let mut response_headers = headers.unwrap_or_default();
        response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        *response.headers_mut() = response_headers;

        AttachedBody::wrap_response(response, guards)
    }

    // Helper to process non-streaming decode response with logprob merging
    async fn process_non_streaming_response(
        &self,
        res: reqwest::Response,
        status: StatusCode,
        return_logprob: bool,
        return_routed_experts: bool,
        prefill_body: Option<bytes::Bytes>,
    ) -> Response {
        let response = res.bytes().await;
        let decode_body = match response {
            Ok(decode_body) => decode_body,
            Err(e) => {
                error!("Failed to read decode response: {}", e);
                return error::internal_error("read_response_failed", "Failed to read response");
            }
        };

        if !return_logprob && !return_routed_experts {
            return (status, decode_body).into_response();
        }

        let Some(prefill_body) = prefill_body else {
            return (status, decode_body).into_response();
        };

        // Parse both responses to walk into them for merging logprobs
        // and/or routed_experts.
        let (Ok(prefill_json), Ok(mut decode_json)) = (
            serde_json::from_slice::<Value>(&prefill_body),
            serde_json::from_slice::<Value>(&decode_body),
        ) else {
            warn!("Failed to parse responses for prefill/decode merging");
            return (status, decode_body).into_response();
        };

        if return_logprob {
            Self::merge_logprobs_in_json(&prefill_json, &mut decode_json);
        }
        if return_routed_experts {
            Self::merge_routed_experts(&prefill_json, &mut decode_json);
        }

        // Return merged response
        match serde_json::to_vec(&decode_json) {
            Ok(body) => (status, body).into_response(),
            Err(e) => {
                error!("Failed to serialize merged response: {}", e);
                (status, decode_body).into_response()
            }
        }
    }

    // Helper to process prefill response and extract body if needed for logprobs
    async fn process_prefill_response(
        &self,
        prefill_result: Result<reqwest::Response, reqwest::Error>,
        prefill_url: &str,
        return_logprob: bool,
    ) -> Result<(StatusCode, Option<bytes::Bytes>), Response> {
        // Check prefill result first - it's critical for disaggregated mode
        let prefill_response = match prefill_result {
            Ok(response) => response,
            Err(e) => {
                error!(
                    "Prefill server failed (CRITICAL) prefill_url={} error={}. Decode will timeout without prefill KV cache.",
                    prefill_url,
                    e
                );

                // Return error immediately - don't wait for decode to timeout
                return Err(error::bad_gateway(
                    "prefill_server_error",
                    format!(
                        "Prefill server error: {}. This will cause decode timeout.",
                        e
                    ),
                ));
            }
        };

        let prefill_status = StatusCode::from_u16(prefill_response.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        // Check if prefill succeeded
        if !prefill_status.is_success() {
            // Get error body from prefill
            let error_msg = prefill_response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown prefill error".to_string());

            error!(
                "Prefill server returned error status prefill_url={} status={} body={}",
                prefill_url, prefill_status, error_msg
            );

            // Map prefill_status to appropriate error function
            let error_response = match prefill_status {
                StatusCode::BAD_REQUEST => error::bad_request(
                    "prefill_bad_request",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::NOT_FOUND => error::not_found(
                    "prefill_not_found",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                    "prefill_internal_error",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                    "prefill_unavailable",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::BAD_GATEWAY => error::bad_gateway(
                    "prefill_bad_gateway",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                _ => error::internal_error(
                    "prefill_error",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
            };
            return Err(error_response);
        }

        // Read prefill body if needed for logprob merging
        let prefill_body = if return_logprob {
            match prefill_response.bytes().await {
                Ok(body) => Some(body),
                Err(e) => {
                    warn!("Failed to read prefill response body for logprobs: {}", e);
                    None
                }
            }
        } else {
            // For non-logprob requests, just consume the response without storing
            debug!("Consuming prefill response body (non-logprob request)");
            match prefill_response.bytes().await {
                Ok(_) => debug!("Prefill response consumed successfully"),
                Err(e) => warn!("Error consuming prefill response: {}", e),
            }
            None
        };

        Ok((prefill_status, prefill_body))
    }

    fn build_post_with_headers(
        &self,
        client: &Client,
        url: &str,
        route: &'static str,
        json_request: &Value,
        headers: Option<&HeaderMap>,
        connection_close: bool,
    ) -> reqwest::RequestBuilder {
        let mut request = client.post(api_path(url, route)).json(json_request);
        if connection_close {
            request = request.header("Connection", "close");
        }
        if let Some(headers) = headers {
            for (name, value) in headers.iter() {
                if header_utils::should_forward_request_header(name.as_str()) {
                    if let Ok(val) = value.to_str() {
                        request = request.header(name, val);
                    }
                }
            }
        }
        request
    }

    /// Merge `routed_experts` from prefill into the decode response.
    ///
    /// SGLang's RL extension ships expert ids as a base64-packed byte
    /// string. The decode worker echoes the prefill's leading bytes
    /// (first `prefill_len` bytes of decode are the same as prefill)
    /// and appends per-token expert ids in the suffix. To produce the
    /// merged stream the gateway concatenates `prefill_bytes ++
    /// decode_bytes[prefill_len..]` and writes that back into
    /// `decode_json["routed_experts"]`. Returns `true` when a merge
    /// happened, `false` if either side was missing or unparsable.
    ///
    /// Caller (`merge_prefill_json`) walks two known envelopes for
    /// this field: `meta_info.routed_experts` on `/generate`
    /// responses, and `sglext.routed_experts` on OpenAI-compatible
    /// responses (the `sglext` envelope is a typed contract upstream
    /// — see `python/sglang/srt/entrypoints/openai/protocol.py::SglExt`).
    fn merge_routed_experts_in_json(prefill_json: &Value, decode_json: &mut Value) -> bool {
        let (Some(prefill_routed_experts), Some(decode_routed_experts)) = (
            prefill_json.get("routed_experts").and_then(Value::as_str),
            decode_json.get("routed_experts").and_then(Value::as_str),
        ) else {
            return false;
        };

        // Base64 decode failures here mean the client asked for merged
        // routed_experts but will receive only decode-side payload.
        // Log as error so data-integrity events show up in dashboards;
        // leave decode_json untouched.
        let prefill_bytes = match BASE64_STANDARD.decode(prefill_routed_experts) {
            Ok(b) => b,
            Err(error) => {
                error!("routed_experts_decode_failed (prefill): merge skipped: {error}");
                return false;
            }
        };
        let decode_bytes = match BASE64_STANDARD.decode(decode_routed_experts) {
            Ok(b) => b,
            Err(error) => {
                error!("routed_experts_decode_failed (decode): merge skipped: {error}");
                return false;
            }
        };

        let suffix = decode_bytes.get(prefill_bytes.len()..).unwrap_or_default();
        let mut merged = Vec::with_capacity(prefill_bytes.len() + suffix.len());
        merged.extend_from_slice(&prefill_bytes);
        merged.extend_from_slice(suffix);

        if let Some(slot) = decode_json.get_mut("routed_experts") {
            *slot = Value::String(BASE64_STANDARD.encode(&merged));
            true
        } else {
            false
        }
    }

    /// Walk both possible response shapes (chat-completions
    /// `sglext.routed_experts` and `/generate` `meta_info.routed_experts`)
    /// and merge whichever applies.
    fn merge_routed_experts(prefill_json: &Value, decode_json: &mut Value) -> bool {
        let mut merged_any = false;
        // chat: response.sglext.routed_experts
        if let Some(prefill_sglext) = prefill_json.get("sglext") {
            if let Some(decode_sglext) = decode_json.get_mut("sglext") {
                merged_any |= Self::merge_routed_experts_in_json(prefill_sglext, decode_sglext);
            }
        }
        // generate: response.meta_info.routed_experts
        if let Some(prefill_meta) = prefill_json.get("meta_info") {
            if let Some(decode_meta) = decode_json.get_mut("meta_info") {
                merged_any |= Self::merge_routed_experts_in_json(prefill_meta, decode_meta);
            }
        }
        merged_any
    }

    // Helper to merge logprobs from prefill and decode responses
    // Optimized to avoid double cloning by taking ownership of decode array
    fn merge_logprobs_in_json(prefill_json: &Value, decode_json: &mut Value) -> bool {
        if let (Some(prefill_meta), Some(decode_meta)) = (
            prefill_json.get("meta_info"),
            decode_json.get_mut("meta_info"),
        ) {
            if let (Some(prefill_logprobs), Some(decode_logprobs)) = (
                prefill_meta.get("input_token_logprobs"),
                decode_meta.get_mut("input_token_logprobs"),
            ) {
                if let Some(prefill_arr) = prefill_logprobs.as_array() {
                    // Take ownership of decode array to avoid cloning it
                    let decode_arr = std::mem::take(decode_logprobs);
                    if let Value::Array(decode_vec) = decode_arr {
                        // Pre-allocate merged array with exact capacity
                        let mut merged = Vec::with_capacity(prefill_arr.len() + decode_vec.len());
                        merged.extend(prefill_arr.iter().cloned());
                        merged.extend(decode_vec);
                        decode_meta["input_token_logprobs"] = Value::Array(merged);
                        return true;
                    }
                }
            }
        }
        false
    }

    // Simple helper to merge logprobs in streaming responses
    // Optimized to reduce allocations in the merge path
    fn merge_streaming_logprobs(
        prefill_logprobs: Option<Value>,
        decode_chunk: &[u8],
    ) -> Result<bytes::Bytes, ()> {
        // Skip non-data chunks
        let chunk_str = std::str::from_utf8(decode_chunk).map_err(|_| ())?;
        if !chunk_str.starts_with("data: ") || chunk_str.contains("[DONE]") {
            return Err(());
        }

        // Parse JSON from chunk
        let json_str = chunk_str.trim_start_matches("data: ").trim();
        let mut decode_json: Value = serde_json::from_str(json_str).map_err(|_| ())?;

        // Merge prefill logprobs if available
        if let Some(ref p_logprobs) = prefill_logprobs {
            if let Some(meta) = decode_json.get_mut("meta_info") {
                if let Some(d_logprobs) = meta.get_mut("input_token_logprobs") {
                    if let Some(p_arr) = p_logprobs.as_array() {
                        // Take ownership of decode array to avoid cloning it
                        let decode_arr = std::mem::take(d_logprobs);
                        if let Value::Array(d_vec) = decode_arr {
                            // Pre-allocate merged array with exact capacity
                            let mut merged = Vec::with_capacity(p_arr.len() + d_vec.len());
                            merged.extend(p_arr.iter().cloned());
                            merged.extend(d_vec);
                            *d_logprobs = Value::Array(merged);
                        }
                    }
                }
            }
        }

        // Re-serialize
        let merged_str = format!(
            "data: {}\n\n",
            serde_json::to_string(&decode_json).unwrap_or_default()
        );
        Ok(bytes::Bytes::from(merged_str))
    }
}

#[async_trait]
impl RouterTrait for PDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Note: This endpoint actually causes the model to generate tokens, so we only test one pair

        // Select a random worker pair using the policy
        let (prefill, decode) = match self.select_pd_pair(None, None, None).await {
            Ok(pair) => pair,
            Err(e) => {
                return error::service_unavailable(
                    "no_healthy_worker_pair",
                    format!("No healthy worker pair available: {}", e),
                );
            }
        };

        let prefill_url = format!("{}/health_generate", prefill.url());
        let (prefill_result, decode_result) = tokio::join!(
            self.client.get(&prefill_url).send(),
            self.client
                .get(format!("{}/health_generate", decode.url()))
                .send()
        );

        // Check results
        let mut errors = Vec::new();

        match prefill_result {
            Ok(res) if res.status().is_success() => {
                debug!(
                    "Health generate passed for prefill server: {}",
                    prefill.url()
                );
            }
            Ok(res) => {
                errors.push(format!(
                    "Prefill {} returned status {}",
                    prefill.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Prefill {} error: {}", prefill.url(), e));
            }
        }

        match decode_result {
            Ok(res) if res.status().is_success() => {
                debug!("Health generate passed for decode server: {}", decode.url());
            }
            Ok(res) => {
                errors.push(format!(
                    "Decode {} returned status {}",
                    decode.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Decode {} error: {}", decode.url(), e));
            }
        }

        if errors.is_empty() {
            (
                StatusCode::OK,
                format!(
                    "Health generate passed on selected pair: prefill={}, decode={}",
                    prefill.url(),
                    decode.url()
                ),
            )
                .into_response()
        } else {
            error::service_unavailable(
                "health_generate_failed",
                format!("Health generate failed: {:?}", errors),
            )
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // Get info from the first decode server to match sglang's server info format
        // Note: We use decode workers for server info to match expected format
        self.proxy_to_first_prefill_worker("server_info", None)
            .await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("v1/models", Some(headers))
            .await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("model_info", Some(headers))
            .await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &Bytes,
        model_id: Option<&str>,
    ) -> Response {
        // PD has to mutate the body (bootstrap_host/port/room
        // injection per attempt), so we parse to `Value` either way.
        // Routing-decision fields are read off the same `Value` so
        // there's no second `from_slice` over the bytes.
        let request_json: Value = match serde_json::from_slice(body) {
            Ok(v) => v,
            Err(e) => {
                return error::bad_request("json_parse_error", format!("Invalid JSON data: {e}"));
            }
        };

        // Preserve the structured-400 contract: SGLang extension
        // fields with the wrong type get the named
        // `invalid_sglang_extension` code instead of being silently
        // forwarded. The view's typed Deserialize would have caught
        // this; with a permissive Value parse we re-check explicitly.
        if let Some(field) = validate_extensions_in_value::<GenerateRoutingView>(&request_json) {
            return error::bad_request(
                "invalid_sglang_extension",
                format!("Invalid SGLang extension field `{field}`"),
            );
        }

        let is_stream = json_bool(&request_json, "stream").unwrap_or(false);
        let return_logprob = json_bool(&request_json, "return_logprob").unwrap_or(false);
        let return_routed_experts =
            json_bool(&request_json, "return_routed_experts").unwrap_or(false);

        if is_stream && return_routed_experts {
            return error::bad_request(
                "streaming_routed_experts_unsupported",
                "return_routed_experts is not supported with stream=true on PD mode \
                 (the streaming SSE path does not merge routed_experts across \
                 prefill/decode); send the request with stream=false to receive \
                 merged routed_experts",
            );
        }

        // Stash `body_model` and `batch_size` as owned values so we
        // can move `request_json` into the retry loop without leaving
        // a borrow alive in `PDRequestContext`.
        let body_model_owned = request_json
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string);
        let batch_size = generate_batch_size_from_value(&request_json);

        let request_text = self
            .policies_need_request_text()
            .then(|| {
                request_json
                    .get("text")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .flatten();

        let effective_model = model_id.or(body_model_owned.as_deref());
        let context = PDRequestContext {
            route: "/generate",
            batch_size,
            is_stream,
            return_logprob,
            return_routed_experts,
            request_text,
            model_id: effective_model,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, request_json, context)
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &Bytes,
        model_id: Option<&str>,
    ) -> Response {
        let request_json: Value = match serde_json::from_slice(body) {
            Ok(v) => v,
            Err(e) => {
                return error::bad_request("json_parse_error", format!("Invalid JSON data: {e}"));
            }
        };

        if let Some(field) = validate_extensions_in_value::<ChatRoutingView>(&request_json) {
            return error::bad_request(
                "invalid_sglang_extension",
                format!("Invalid SGLang extension field `{field}`"),
            );
        }

        let is_stream = json_bool(&request_json, "stream").unwrap_or(false);
        let return_logprob = json_bool(&request_json, "logprobs").unwrap_or(false);
        let return_routed_experts =
            json_bool(&request_json, "return_routed_experts").unwrap_or(false);

        if is_stream && return_routed_experts {
            return error::bad_request(
                "streaming_routed_experts_unsupported",
                "return_routed_experts is not supported with stream=true on PD mode \
                 (the streaming SSE path does not merge routed_experts across \
                 prefill/decode); send the request with stream=false to receive \
                 merged routed_experts",
            );
        }

        // See `route_generate`: stash owned scalars so we can move
        // `request_json` into the retry loop.
        let body_model_owned = request_json
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string);
        let batch_size = chat_batch_size_from_value(&request_json);

        // Skip the walk when no active policy reports
        // `needs_request_text == true` (cache-aware is the canonical
        // consumer today; the gate keeps the non-cache-aware path at
        // a single body parse).
        let request_text = self
            .policies_need_request_text()
            .then(|| extract_chat_request_text(&request_json))
            .flatten();

        let effective_model = model_id.or(body_model_owned.as_deref());
        let context = PDRequestContext {
            route: "/v1/chat/completions",
            batch_size,
            is_stream,
            return_logprob,
            return_routed_experts,
            request_text,
            model_id: effective_model,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, request_json, context)
            .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs.is_some();

        let request_text = if self.policies_need_request_text() {
            match &body.prompt {
                StringOrArray::String(s) => Some(s.clone()),
                StringOrArray::Array(v) => v.first().map(|s| s.to_string()),
            }
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_completion_batch_size(body);

        let context = PDRequestContext {
            route: "/v1/completions",
            batch_size,
            is_stream,
            return_logprob,
            return_routed_experts: false,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        // Pre-serialize once outside the retry loop so each retry only
        // pays a `Value::clone`, not a serde round-trip. Record the
        // start metric on failure so the failure counts toward the
        // same dispatch row `execute_dual_dispatch` would have recorded.
        let request_json = match serde_json::to_value(body) {
            Ok(v) => v,
            Err(e) => {
                record_pd_request_start(&context);
                return Self::handle_serialization_error(e);
            }
        };
        self.execute_dual_dispatch(headers, request_json, context)
            .await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Extract text for cache-aware routing
        let req_text = if self.policies_need_request_text() {
            Some(body.query.clone())
        } else {
            None
        };

        let context = PDRequestContext {
            route: "/v1/rerank",
            batch_size: None,
            is_stream: false,
            return_logprob: false,
            return_routed_experts: false,
            request_text: req_text,
            model_id,
            headers: headers.cloned(),
        };

        // Pre-serialize once outside the retry loop; on failure still
        // record the start metric so the request counter doesn't drift
        // below the error counter for this dispatch.
        let request_json = match serde_json::to_value(body) {
            Ok(v) => v,
            Err(e) => {
                record_pd_request_start(&context);
                return Self::handle_serialization_error(e);
            }
        };
        self.execute_dual_dispatch(headers, request_json, context)
            .await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        let _ = (headers, body, model_id);
        warn!("PD mode does not support /v1/embeddings; returning bad request");
        error::bad_request(
            "pd_unsupported_embeddings",
            "PD mode does not support /v1/embeddings",
        )
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        let _ = (headers, body, model_id);
        warn!("PD mode does not support /v1/classify; returning bad request");
        error::bad_request(
            "pd_unsupported_classify",
            "PD mode does not support /v1/classify",
        )
    }

    fn router_type(&self) -> &'static str {
        "pd"
    }
}

/// Read a top-level JSON boolean field by name. Used by the PD
/// router to pull routing flags (`stream`, `return_logprob`,
/// `return_routed_experts`) off the `Value` it already parsed for
/// bootstrap injection.
fn json_bool(value: &Value, key: &str) -> Option<bool> {
    value.get(key)?.as_bool()
}

/// Record the "request start" metric for the PD path. Lives here as
/// a free function so the typed call sites can invoke it on their
/// pre-serialization error paths (before `execute_dual_dispatch`
/// would otherwise have recorded it), keeping the request counter
/// from diverging from the error counter on the rare typed
/// `serde_json::to_value` failure.
fn record_pd_request_start(context: &PDRequestContext<'_>) {
    let model = context.model_id.unwrap_or(UNKNOWN_MODEL_ID);
    let endpoint = route_to_endpoint(context.route);
    Metrics::record_router_request(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_PD,
        metrics_labels::CONNECTION_HTTP,
        model,
        endpoint,
        bool_to_static_str(context.is_stream),
    );
}

/// Pull a stable cache-aware routing prefix out of an already-parsed
/// chat body, walking the FULL `messages` array. Cache-aware (and any
/// other policy reporting `needs_request_text() == true`) needs every
/// turn of the conversation in order to reuse the previous turn's KV
/// cache on the same worker — first-message-only collapses every
/// conversation that shares a system prompt onto one worker.
///
/// Per message:
/// - `content` as string → append the string verbatim;
/// - `content` as array of parts → append text-typed parts in order
///   and skip image/audio/etc. (image URLs are not stable prefixes
///   and add noise to the hash).
///
/// Messages are joined by `\n` so adjacent turns with the same raw
/// text do not silently merge. Roles are preserved as a leading
/// `{role}:` tag so a user turn cannot impersonate the system one;
/// a malformed message with no `role` field still parses but uses an
/// empty tag (`:`). Returns `None` if no text was found anywhere
/// (e.g. an all-image conversation has no useful prefix).
pub(crate) fn extract_chat_request_text(json: &Value) -> Option<String> {
    let messages = json.get("messages")?.as_array()?;
    let mut out = String::new();
    for msg in messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("");
        let Some(content) = msg.get("content") else {
            continue;
        };
        let chunk = match content {
            Value::String(s) => s.clone(),
            Value::Array(parts) => {
                let mut buf = String::new();
                for part in parts {
                    let is_text = part.get("type").and_then(Value::as_str) == Some("text");
                    if !is_text {
                        continue;
                    }
                    if let Some(t) = part.get("text").and_then(Value::as_str) {
                        buf.push_str(t);
                    }
                }
                buf
            }
            _ => continue,
        };
        if chunk.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(role);
        out.push(':');
        out.push_str(&chunk);
    }
    (!out.is_empty()).then_some(out)
}

/// Mirrors `GenerateRoutingView::batch_size` but reads from a raw
/// `Value` so PD doesn't have to re-construct the typed view. SGLang's
/// `input_ids` is either `Vec<u32>` (single — its length is the token
/// count, NOT the batch size) or `Vec<Vec<u32>>` (batch — outer length
/// IS the batch size); disambiguate by the type of the first element.
fn generate_batch_size_from_value(value: &Value) -> Option<usize> {
    let arr = value.get("input_ids")?.as_array()?;
    match arr.first() {
        Some(Value::Array(_)) => Some(arr.len()),
        _ => None,
    }
}

/// Mirrors `ChatRoutingView::batch_size`: `n > 1` triggers PD batch
/// estimation, anything else is single-shot.
fn chat_batch_size_from_value(value: &Value) -> Option<usize> {
    match value.get("n")?.as_u64()? {
        n if n > 1 => Some(n as usize),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    fn create_test_pd_router() -> PDRouter {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin));

        PDRouter {
            worker_registry,
            policy_registry,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            api_key: Some("test_api_key".to_string()),
            enable_igw: false,
        }
    }

    /// Mock policy that records the `request_text` it sees on every
    /// `select_worker` call, then returns `None` so the PD dispatch
    /// short-circuits before any HTTP is attempted.
    ///
    /// `needs_request_text() == true` triggers PD's walker gate so
    /// `extract_chat_request_text` runs and its output flows into
    /// `SelectWorkerInfo::request_text`.
    #[derive(Debug, Default)]
    struct RecordingPolicy {
        seen_request_text: Mutex<Vec<Option<String>>>,
    }

    #[async_trait]
    impl crate::policies::LoadBalancingPolicy for RecordingPolicy {
        async fn select_worker(
            &self,
            _workers: &[Arc<dyn Worker>],
            info: &crate::policies::SelectWorkerInfo<'_>,
        ) -> Option<usize> {
            self.seen_request_text
                .lock()
                .unwrap()
                .push(info.request_text.map(str::to_string));
            None
        }

        fn name(&self) -> &'static str {
            "test-recording"
        }

        fn needs_request_text(&self) -> bool {
            true
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn create_test_worker(url: String, worker_type: WorkerType, healthy: bool) -> Box<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .build();
        worker.set_healthy(healthy);
        Box::new(worker)
    }

    #[tokio::test]
    async fn test_select_healthy_prefill_worker() {
        let router = create_test_pd_router();

        let healthy_worker = create_test_worker(
            "http://healthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let unhealthy_worker = create_test_worker(
            "http://unhealthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            false,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(unhealthy_worker));
        router.worker_registry.register(Arc::from(healthy_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let result = router.select_pd_pair(None, None, None).await;

        assert!(result.is_ok());
        let (prefill, _decode) = result.unwrap();

        assert_eq!(prefill.url(), "http://healthy");
        assert!(prefill.is_healthy());
    }

    #[tokio::test]
    async fn test_empty_worker_lists() {
        let router = create_test_pd_router();

        let result = router.select_pd_pair(None, None, None).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No prefill workers available"));
    }

    /// Wire-up regression: PD's `route_chat` must run the full-array
    /// walker AND pass the result into the policy as `request_text`.
    /// This test replaces the prefill+decode policies with a recorder
    /// that returns `None` from `select_worker` (so the dispatch
    /// short-circuits before HTTP) and asserts the recorded text is
    /// the walker's role-tagged multi-message output. A regression
    /// that dropped `request_text` between `route_chat` and the
    /// policy would leave the recorded text as `None`.
    #[tokio::test]
    async fn route_chat_passes_walker_output_to_policy() {
        let router = create_test_pd_router();

        let recording = Arc::new(RecordingPolicy::default());
        router.policy_registry.set_prefill_policy(recording.clone());
        router.policy_registry.set_decode_policy(recording.clone());

        // Register one healthy prefill + one healthy decode so the
        // policy is actually invoked (empty worker lists fail before
        // ever calling select_worker).
        router
            .worker_registry
            .register(Arc::from(create_test_worker(
                "http://prefill".to_string(),
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
                true,
            )));
        router
            .worker_registry
            .register(Arc::from(create_test_worker(
                "http://decode".to_string(),
                WorkerType::Decode,
                true,
            )));

        let body = Bytes::from(
            serde_json::json!({
                "model": "mock-model",
                "messages": [
                    {"role": "system", "content": "be concise"},
                    {"role": "user", "content": "hello"}
                ],
                "stream": false,
            })
            .to_string(),
        );

        // Response will be a server-selection 5xx because the
        // recording policy returns None — that's fine. We only care
        // that the policy was called with the walker's output.
        let _ = router.route_chat(None, &body, None).await;

        let seen = recording.seen_request_text.lock().unwrap().clone();
        assert!(
            !seen.is_empty(),
            "policy must have been invoked at least once"
        );
        let first = seen[0]
            .as_ref()
            .expect("policy must see Some(request_text); got None means walker output was dropped");
        let expected = "system:be concise\nuser:hello";
        assert_eq!(
            first, expected,
            "policy must see the walker's role-tagged multi-message output"
        );
    }

    #[test]
    fn test_worker_load_metrics() {
        let prefill_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        ));
        let decode_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://decode".to_string(),
            WorkerType::Decode,
            true,
        ));

        let _prefill_guard = WorkerLoadGuard::new(prefill_worker.clone(), None);
        let _decode_guard = WorkerLoadGuard::new(decode_worker.clone(), None);

        assert_eq!(prefill_worker.load(), 1);
        assert_eq!(decode_worker.load(), 1);

        drop(_prefill_guard);
        drop(_decode_guard);

        assert_eq!(prefill_worker.load(), 0);
        assert_eq!(decode_worker.load(), 0);
    }

    #[tokio::test]
    async fn test_streaming_load_tracking() {
        use futures_util::StreamExt;
        use tokio::time::{sleep, Duration};

        let router = create_test_pd_router();

        let prefill_worker = create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(prefill_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let prefill_workers = router.worker_registry.get_prefill_workers();
        let decode_workers = router.worker_registry.get_decode_workers();

        let prefill_ref = prefill_workers[0].clone();
        let decode_ref = decode_workers[0].clone();

        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream = UnboundedReceiverStream::new(rx);

        {
            let response = router.create_streaming_response(
                stream.map(Ok),
                StatusCode::OK,
                None,
                false,
                None,
                None,
                prefill_ref.clone(),
                decode_ref.clone(),
            );

            // Guards are now attached to response body, so load should be 1
            assert_eq!(prefill_ref.load(), 1);
            assert_eq!(decode_ref.load(), 1);

            tx.send(bytes::Bytes::from("test data")).unwrap();

            sleep(Duration::from_millis(10)).await;

            // Load still 1 while response body exists
            assert_eq!(prefill_ref.load(), 1);
            assert_eq!(decode_ref.load(), 1);

            drop(tx);

            // Response (and its body with guards) dropped here
            drop(response);
        }

        // Guards dropped when response dropped
        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);
    }

    /// Cache-aware policies hash on a stable text prefix per request.
    /// The walker concatenates every message's text content so turn N+1
    /// of a conversation shares a stable prefix with turn N — that's
    /// what lets the prefix trie route subsequent turns to the worker
    /// that already holds the KV cache.
    mod extract_chat_request_text {
        use serde_json::json;

        use super::super::extract_chat_request_text;

        #[test]
        fn single_user_string_message() {
            let body = json!({
                "messages": [{"role": "user", "content": "deep learning is"}]
            });
            assert_eq!(
                extract_chat_request_text(&body),
                Some("user:deep learning is".to_string())
            );
        }

        #[test]
        fn multi_turn_grows_prefix() {
            // A 4-message conversation (system + user + assistant + user).
            // The walker concatenates every message so the cache-aware
            // prefix trie keeps matching across turns.
            let body = json!({
                "messages": [
                    {"role": "system", "content": "be concise"},
                    {"role": "user", "content": "what is rust?"},
                    {"role": "assistant", "content": "a systems lang"},
                    {"role": "user", "content": "and ownership?"}
                ]
            });
            assert_eq!(
                extract_chat_request_text(&body),
                Some(
                    "system:be concise\n\
                     user:what is rust?\n\
                     assistant:a systems lang\n\
                     user:and ownership?"
                        .to_string()
                )
            );
        }

        #[test]
        fn turn_n_strictly_extends_turn_n_minus_1_prefix() {
            // The whole point of the walker: turn N's routing text
            // must be a prefix of turn N+1's, so the cache-aware trie
            // picks the same worker.
            let turn1 = json!({
                "messages": [
                    {"role": "system", "content": "be concise"},
                    {"role": "user", "content": "what is rust?"}
                ]
            });
            let turn2 = json!({
                "messages": [
                    {"role": "system", "content": "be concise"},
                    {"role": "user", "content": "what is rust?"},
                    {"role": "assistant", "content": "a systems lang"},
                    {"role": "user", "content": "and ownership?"}
                ]
            });
            let t1 = extract_chat_request_text(&turn1).unwrap();
            let t2 = extract_chat_request_text(&turn2).unwrap();
            assert!(
                t2.starts_with(&t1),
                "turn N+1 routing text must extend turn N's; got {t1:?} vs {t2:?}"
            );
        }

        #[test]
        fn role_tag_disambiguates_same_text() {
            // Without the role prefix, an assistant echoing the user
            // would yield identical text. With the tag, the two turns
            // produce distinct routing prefixes that hash to separate
            // trie entries.
            let body = json!({
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hello"}
                ]
            });
            assert_eq!(
                extract_chat_request_text(&body),
                Some("user:hello\nassistant:hello".to_string())
            );
        }

        #[test]
        fn multimodal_parts_keep_text_drop_images() {
            // System prompt mixes text + image parts; image is ignored
            // for hashing (no stable prefix), text is preserved.
            let body = json!({
                "messages": [{
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "you are "},
                        {"type": "text", "text": "helpful"},
                        {"type": "image_url", "image_url": {"url": "ignored"}}
                    ]
                }]
            });
            assert_eq!(
                extract_chat_request_text(&body),
                Some("system:you are helpful".to_string())
            );
        }

        #[test]
        fn all_image_message_is_skipped() {
            // A message whose content is purely images contributes
            // nothing to the prefix (and must not split the role tag
            // off into a dangling `role:` token).
            let body = json!({
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": "x"}}
                    ]},
                    {"role": "user", "content": "describe please"}
                ]
            });
            assert_eq!(
                extract_chat_request_text(&body),
                Some("user:describe please".to_string())
            );
        }

        #[test]
        fn missing_messages_yields_none() {
            let body = json!({"model": "x"});
            assert_eq!(extract_chat_request_text(&body), None);
        }

        #[test]
        fn empty_messages_yields_none() {
            let body = json!({"messages": []});
            assert_eq!(extract_chat_request_text(&body), None);
        }

        #[test]
        fn all_messages_textless_yields_none() {
            let body = json!({
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": "x"}}
                    ]}
                ]
            });
            assert_eq!(extract_chat_request_text(&body), None);
        }
    }
}
