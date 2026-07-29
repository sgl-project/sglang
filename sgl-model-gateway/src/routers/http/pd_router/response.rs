use super::*;

impl PDRouter {
    // Streaming ownership needs both backend identities and both abort guards.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn create_streaming_response(
        &self,
        stream: impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
        status: StatusCode,
        prefill_logprobs: Option<Value>,
        return_logprob: bool,
        headers: Option<HeaderMap>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
    ) -> Response {
        use crate::core::AttachedBody;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        // Uses select! to race stream.next() against tx.closed() so that
        // when the client disconnects the upstream HTTP connection is dropped
        // promptly, allowing the engine to abort the request.
        // `biased;` drains a ready upstream chunk before observing client
        // disconnect, so a chunk already produced by reqwest reaches the
        // client (and the logprob merger) before we tear the loop down.
        //
        // The upstream stream is wrapped in `BreakerTrackedStream` so the
        // decode worker's circuit breaker is updated once on drop: success
        // on clean completion (`[DONE]` sentinel or `None`), failure on
        // stream error, neither on client disconnect. PD's pre-PR semantics
        // treated 4xx (client error) as not-a-worker-fault, so we only
        // pre-mark the wrapper as Errored on 5xx — `handle_decode_error_response`
        // synthesizes a single-chunk SSE error envelope that would otherwise
        // stream cleanly to None and record a spurious success.
        let mut tracked =
            BreakerTrackedStream::new(stream, Arc::clone(&decode), decode.url().to_string());
        if !(status.is_success() || status.is_client_error()) {
            tracked.mark_errored();
        }
        let decode_for_log = decode.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    chunk_result = tracked.next() => {
                        match chunk_result {
                            Some(Ok(chunk)) => {
                                let is_done = memmem::find(&chunk, b"data: [DONE]").is_some();

                                let result = if return_logprob && prefill_logprobs.is_some() {
                                    Self::merge_streaming_logprobs(prefill_logprobs.clone(), &chunk)
                                        .unwrap_or(chunk)
                                } else {
                                    chunk
                                };

                                // Mark the wrapper completed before the client
                                // send: upstream finished cleanly regardless of
                                // whether the client is still listening, and
                                // the worker deserves the success tick either
                                // way. `mark_completed` is a no-op once Errored
                                // is set, so the synthetic-error path is unaffected.
                                if is_done {
                                    tracked.mark_completed();
                                }

                                if tx.send(Ok(result)).is_err() {
                                    tracing::debug!(
                                        "Receiver dropped (likely client disconnect), \
                                        cancelling upstream PD stream"
                                    );
                                    break;
                                }

                                if is_done {
                                    break;
                                }
                            }
                            Some(Err(e)) => {
                                // BreakerTrackedStream already logged the error
                                // and marked the terminal state as Errored so
                                // the worker's circuit breaker will tick on drop.
                                let _ = tx.send(Err(format!("Stream error: {}", e)));
                                break;
                            }
                            None => break,
                        }
                    }
                    _ = tx.closed() => {
                        tracing::info!(
                            "Client disconnected, cancelling upstream PD stream from {}",
                            decode_for_log.url()
                        );
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
        // The body may be a synthetic SSE envelope around an upstream unary
        // error, so its original byte count is no longer valid. Hyper will
        // choose the correct framing for this stream.
        response_headers.remove(CONTENT_LENGTH);
        response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        *response.headers_mut() = response_headers;

        AttachedBody::wrap_response(response, guards)
    }

    // Helper to process non-streaming decode response with logprob merging
    pub(super) async fn process_non_streaming_response(
        &self,
        res: reqwest::Response,
        status: StatusCode,
        return_logprob: bool,
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

        if !return_logprob {
            return (status, decode_body).into_response();
        }

        let Some(prefill_body) = prefill_body else {
            return (status, decode_body).into_response();
        };

        // Merge logprobs from prefill and decode
        let (Ok(prefill_json), Ok(mut decode_json)) = (
            serde_json::from_slice::<Value>(&prefill_body),
            serde_json::from_slice::<Value>(&decode_body),
        ) else {
            warn!("Failed to parse responses for logprob merging");
            return (status, decode_body).into_response();
        };

        Self::merge_logprobs_in_json(&prefill_json, &mut decode_json);

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
    pub(super) async fn process_prefill_response(
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
        let response_headers = header_utils::preserve_response_headers(prefill_response.headers());

        // Check if prefill succeeded
        if !prefill_status.is_success() {
            // Get error body from prefill
            let error_body = prefill_response.bytes().await.unwrap_or_default();

            error!(
                "Prefill server returned error status prefill_url={} status={}",
                prefill_url, prefill_status
            );
            if raw_generate::has_typed_error(&response_headers) {
                let mut response = Response::new(Body::from(error_body));
                *response.status_mut() = prefill_status;
                *response.headers_mut() = response_headers;
                return Err(response);
            }
            let error_msg = String::from_utf8_lossy(&error_body).to_string();

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

    pub(super) fn build_post_with_headers(
        &self,
        client: &Client,
        endpoint_url: &str,
        json_request: &Value,
        headers: Option<&HeaderMap>,
        connection_close: bool,
    ) -> reqwest::RequestBuilder {
        let mut request = client.post(endpoint_url).json(json_request);
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

    // Helper to merge logprobs from prefill and decode responses
    // Optimized to avoid double cloning by taking ownership of decode array
    pub(super) fn merge_logprobs_in_json(prefill_json: &Value, decode_json: &mut Value) -> bool {
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
    pub(super) fn merge_streaming_logprobs(
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
