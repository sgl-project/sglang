use super::*;

impl PDRouter {
    pub(super) fn worker_endpoint_url(worker: &dyn Worker, endpoint: &str) -> String {
        api_path(worker.base_url(), endpoint)
    }

    pub(super) async fn proxy_to_first_prefill_worker(
        &self,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let workers = self.worker_registry.get_prefill_workers();

        if let Some(worker) = workers.first() {
            self.proxy_to_worker(worker.as_ref(), endpoint, headers)
                .await
        } else {
            error::service_unavailable("no_prefill_servers", "No prefill servers available")
        }
    }

    pub(super) async fn proxy_to_worker(
        &self,
        worker: &dyn Worker,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let url = Self::worker_endpoint_url(worker, endpoint);
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
            rendezvous_gate: Arc::new(Mutex::new(())),
            active_item_permits: Arc::new(tokio::sync::Semaphore::new(PD_ACTIVE_ITEM_CAPACITY)),
        })
    }

    pub(super) fn handle_server_selection_error(error: String) -> Response {
        error!("Failed to select PD pair error={}", error);
        error::service_unavailable(
            "server_selection_failed",
            format!("No available servers: {}", error),
        )
    }

    pub(super) fn handle_serialization_error(error: impl std::fmt::Display) -> Response {
        error!("Failed to serialize request error={}", error);
        error::internal_error("serialization_failed", "Failed to serialize request")
    }

    pub(super) fn get_generate_batch_size(req: &GenerateRequest) -> Option<usize> {
        // GenerateRequest doesn't support batch via arrays, only via input_ids
        if let Some(InputIds::Batch(batches)) = &req.input_ids {
            if !batches.is_empty() {
                return Some(batches.len());
            }
        }
        None
    }

    pub(super) fn get_chat_batch_size(req: &ChatCompletionRequest) -> Option<usize> {
        if let Some(n) = req.n {
            if n > 1 {
                return Some(n as usize);
            }
        }
        None
    }

    pub(super) fn get_completion_batch_size(req: &CompletionRequest) -> Option<usize> {
        if let StringOrArray::Array(arr) = &req.prompt {
            if !arr.is_empty() {
                return Some(arr.len());
            }
        }
        None
    }

    const DISAGG_PREFILL_DP_RANK_KEY: &'static str = "disagg_prefill_dp_rank";

    pub(super) fn inject_prefill_dp_rank_for_decode<'a>(
        decode_request: Cow<'a, Value>,
        prefill_worker: &dyn Worker,
    ) -> Result<Cow<'a, Value>, String> {
        let Some(prefill_dp_rank) = prefill_worker.dp_rank() else {
            return Ok(decode_request);
        };

        let mut decode_request = decode_request.into_owned();
        let Some(obj) = decode_request.as_object_mut() else {
            return Err(
                "Failed to insert disagg_prefill_dp_rank because request body is not an object"
                    .to_string(),
            );
        };

        obj.insert(
            Self::DISAGG_PREFILL_DP_RANK_KEY.to_string(),
            Value::from(prefill_dp_rank as u64),
        );
        Ok(Cow::Owned(decode_request))
    }

    pub(super) async fn prepare_worker_request<'a>(
        route: &'static str,
        worker: &dyn Worker,
        json_request: Cow<'a, Value>,
    ) -> Result<PreparedWorkerRequest<'a>, String> {
        let body = if worker.is_dp_aware() {
            Cow::Owned(
                worker
                    .prepare_request(json_request.into_owned())
                    .await
                    .map_err(|err| {
                        format!(
                            "Failed to prepare request for worker {}: {}",
                            worker.url(),
                            err
                        )
                    })?,
            )
        } else {
            json_request
        };

        Ok(PreparedWorkerRequest {
            endpoint_url: Self::worker_endpoint_url(worker, route),
            body,
        })
    }

    pub(super) async fn prepare_pd_worker_requests<'a>(
        route: &'static str,
        json_request: &'a Value,
        prefill: &dyn Worker,
        decode: &dyn Worker,
    ) -> Result<(PreparedWorkerRequest<'a>, PreparedWorkerRequest<'a>), String> {
        let prefill_request =
            Self::prepare_worker_request(route, prefill, Cow::Borrowed(json_request)).await?;
        let decode_json_request =
            Self::inject_prefill_dp_rank_for_decode(Cow::Borrowed(json_request), prefill)?;
        let decode_request =
            Self::prepare_worker_request(route, decode, decode_json_request).await?;

        Ok((prefill_request, decode_request))
    }

    pub(super) fn policies_need_request_text(&self) -> bool {
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        prefill_policy.needs_request_text() || decode_policy.needs_request_text()
    }

    /// Builds the text used for cache-aware routing of a chat request.
    ///
    /// This must reflect the *full* conversation (system prompt, prior turns,
    /// the current message and tool context) so that KV-cache prefix matching
    /// routes to the worker that actually shares the most prefix. Using only the
    /// first message ignores the conversation history that drives KV reuse in
    /// multi-turn chats. See https://github.com/sgl-project/sglang/issues/26263.
    ///
    /// Returns `None` when the conversation has no text to route on, preserving
    /// the prior behavior of not feeding an empty key into prefix matching.
    pub(super) fn build_chat_request_text(body: &ChatCompletionRequest) -> Option<String> {
        // `extract_text_for_routing` walks every message (system, prior turns,
        // current message, tool content) and is the same routing text the regular
        // (non-PD) router uses, keeping cache-aware routing consistent across both.
        let text = body.extract_text_for_routing();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }

    pub(super) async fn select_pd_pair(
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

    pub(super) async fn pick_worker_by_policy_arc(
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
}
