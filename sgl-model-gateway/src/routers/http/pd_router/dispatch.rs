use super::*;

impl PDRouter {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn execute_dual_dispatch<T: Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        original_request: &T,
        context: PDRequestContext<'_>,
    ) -> Response {
        let start_time = Instant::now();

        let route = context.route;
        let model = context.model_id.unwrap_or(UNKNOWN_MODEL_ID);
        let endpoint = route_to_endpoint(route);

        // Record request start (Layer 2)
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_PD,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(context.is_stream),
        );
        // Clone request once outside the retry loop, then use Arc to share across attempts
        // This avoids O(retries) clones by sharing the same data
        let shared_request = Arc::new(original_request.clone());
        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            {
                move |attempt: u32| {
                    // Clone Arc (cheap reference count increment) instead of cloning the entire request
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

                        let mut json_request = match serde_json::to_value(shared_request.as_ref()) {
                            Ok(v) => v,
                            Err(e) => return Self::handle_serialization_error(e),
                        };

                        json_request = match raw_generate::inject_bootstrap(
                            json_request,
                            prefill.as_ref(),
                            context.batch_size,
                            context.route == "/generate",
                        ) {
                            Ok(v) => v,
                            Err(e) => return Self::handle_serialization_error(e),
                        };

                        let item_count = context.batch_size.unwrap_or(1);
                        if item_count > PD_ACTIVE_ITEM_CAPACITY {
                            return raw_generate::pd_error_response(
                                raw_generate::PdRawError::RequestInvalid,
                            );
                        }
                        let item_permits = Arc::clone(&self.active_item_permits)
                            .acquire_many_owned(item_count as u32)
                            .await
                            .expect("PD item admission semaphore is never closed");

                        let ctx_is_stream = context.is_stream;
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
                        let response = AttachedBody::wrap_response(response, item_permits);

                        let status = response.status();
                        let outcomes_already_recorded = response
                            .extensions()
                            .get::<BreakerOutcomesRecorded>()
                            .is_some();
                        if !outcomes_already_recorded {
                            let not_error = status.is_success() || status.is_client_error();
                            // Prefill is always non-streaming and fully read before
                            // we get here, so its outcome is final.
                            prefill.record_outcome(not_error);
                            // Decode for a streaming request is still mid-flight at
                            // this point; the `BreakerTrackedStream` wrapped around
                            // its byte stream records the outcome on drop. Skip the
                            // eager success record to avoid masking "200-then-broken"
                            // decode workers.
                            if !ctx_is_stream {
                                decode.record_outcome(not_error);
                            }
                        }

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
            |res, _attempt| raw_generate::response_retryable(res),
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
        } else if !raw_generate::response_retryable(&response) {
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

    pub(super) async fn handle_decode_error_response(
        &self,
        res: reqwest::Response,
        context: &PDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
    ) -> Response {
        if raw_generate::has_typed_error(res.headers()) {
            return self
                .handle_typed_pd_error_response(res, context.is_stream, prefill, decode)
                .await;
        }
        let status = res.status();
        let response_headers = header_utils::preserve_response_headers(res.headers());

        if context.is_stream {
            // Handle streaming error response
            let error_body = res.bytes().await;
            let error_payload = match error_body {
                Ok(error_body) => match serde_json::from_slice::<Value>(&error_body) {
                    Ok(error_json) => {
                        json!({ "message": error_json, "status": status.as_u16() })
                    }
                    Err(parse_err) => {
                        let body_text = String::from_utf8_lossy(&error_body).to_string();
                        let preview: String = body_text.chars().take(256).collect();
                        tracing::warn!(
                            "Failed to parse decode error body as JSON from {}: {} \
                             (status={}, body preview: {:?})",
                            decode.url(),
                            parse_err,
                            status.as_u16(),
                            preview
                        );
                        json!({ "message": body_text, "status": status.as_u16() })
                    }
                },
                Err(e) => {
                    json!({ "message": format!("Decode server error: {}", e), "status": status.as_u16() })
                }
            };

            let sse_data = format!(
                "data: {{'error': {}}}",
                serde_json::to_string(&error_payload).unwrap_or_default()
            );
            let error_stream = tokio_stream::once(Ok(axum::body::Bytes::from(sse_data)));

            self.create_streaming_response(
                error_stream,
                status,
                None,
                context.return_logprob,
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

    pub(super) async fn handle_typed_pd_error_response(
        &self,
        res: reqwest::Response,
        is_stream: bool,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
    ) -> Response {
        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let headers = header_utils::preserve_response_headers(res.headers());
        let body = res.bytes().await.unwrap_or_default();
        if !is_stream {
            let mut response = Response::new(Body::from(body));
            *response.status_mut() = status;
            *response.headers_mut() = headers;
            return response;
        }

        let payload = format!(
            "data: {}\n\ndata: [DONE]\n\n",
            String::from_utf8_lossy(&body)
        );
        let stream = tokio_stream::once(Ok(axum::body::Bytes::from(payload)));
        self.create_streaming_response(stream, status, None, false, Some(headers), prefill, decode)
    }

    // Internal method that performs the actual dual dispatch (without retry logic)
    pub(super) async fn execute_dual_dispatch_internal(
        &self,
        headers: Option<&HeaderMap>,
        json_request: Value,
        context: PDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        _start_time: Instant,
    ) -> Response {
        // The frozen PD control protocol performs one synchronous Room
        // rendezvous at a time on a shared peer session. Independent HTTP
        // connections can otherwise deliver concurrent requests to P and D in
        // different orders. Hold this gate only until Prefill confirms that
        // the paired rendezvous and transfer completed; Decode generation and
        // streaming continue concurrently after that point.
        let rendezvous_guard = self.rendezvous_gate.lock().await;

        // For non-streaming: use guard for automatic load management
        // For streaming: load will be managed in create_streaming_response
        let _prefill_guard =
            (!context.is_stream).then(|| WorkerLoadGuard::new(prefill.clone(), headers));
        let _decode_guard =
            (!context.is_stream).then(|| WorkerLoadGuard::new(decode.clone(), headers));

        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        let headers = Some(&headers_with_trace);

        let (prepared_prefill, prepared_decode) = match Self::prepare_pd_worker_requests(
            context.route,
            &json_request,
            prefill.as_ref(),
            decode.as_ref(),
        )
        .await
        {
            Ok(requests) => requests,
            Err(e) => {
                error!("Failed to prepare PD worker requests: {}", e);
                return error::internal_error("pd_request_preparation_failed", e);
            }
        };

        // Build both requests
        let prefill_request = self.build_post_with_headers(
            &self.client,
            &prepared_prefill.endpoint_url,
            &prepared_prefill.body,
            headers,
            false,
        );
        let decode_request = self.build_post_with_headers(
            &self.client,
            &prepared_decode.endpoint_url,
            &prepared_decode.body,
            headers,
            false,
        );

        // Run both in this handler task (not a detached tokio::spawn) so a client
        // disconnect cancels the pending decode request too, keeping the
        // upstream-cancel behavior from #19524.
        events::RequestPDSentEvent {
            prefill_url: prefill.url(),
            decode_url: decode.url(),
        }
        .emit();

        let prefill_fut = prefill_request.send();
        let decode_fut = decode_request.send();
        tokio::pin!(prefill_fut);
        tokio::pin!(decode_fut);

        // Poll both until prefill resolves; decode normally resolves later, but
        // may resolve first if it rejects the request outright.
        let prefill_result;
        let mut decode_early: Option<Result<reqwest::Response, reqwest::Error>> = None;
        loop {
            tokio::select! {
                biased;
                pr = &mut prefill_fut => {
                    prefill_result = pr;
                    break;
                }
                dr = &mut decode_fut, if decode_early.is_none() => {
                    let decode_failed = match &dr {
                        Ok(response) => !response.status().is_success(),
                        Err(_) => true,
                    };
                    if decode_failed {
                        warn!(
                            "Decode failed first, aborting paired prefill request decode_url={} prefill_url={}",
                            decode.url(),
                            prefill.url()
                        );
                        let mut response = match dr {
                            Ok(response) => {
                                let status = StatusCode::from_u16(response.status().as_u16())
                                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                                decode.record_outcome(status.is_client_error());
                                self.handle_decode_error_response(
                                    response,
                                    &context,
                                    Arc::clone(&prefill),
                                    Arc::clone(&decode),
                                )
                                .await
                            }
                            Err(error) => {
                                decode.record_outcome(false);
                                warn!(
                                    "Decode request failed decode_url={} error={}",
                                    decode.url(),
                                    error
                                );
                                error::bad_gateway(
                                    "decode_server_error",
                                    "Decode server request failed",
                                )
                            }
                        };
                        response.extensions_mut().insert(BreakerOutcomesRecorded);
                        return response;
                    }
                    decode_early = Some(dr);
                }
            }
        }

        // Decode can't generate without prefill's KV, so any prefill failure
        // (non-2xx / transport error) dooms the paired decode request, which would
        // otherwise block in WaitingForInput until the 300s disaggregation
        // timeout. Drop the decode future to close its connection; the decode
        // engine then detects the disconnect and aborts the request in ~4-8s.
        let prefill_failed = match &prefill_result {
            Ok(resp) => !resp.status().is_success(),
            Err(_) => true,
        };

        if prefill_failed {
            warn!(
                "Prefill failed, aborting paired decode request decode_url={} prefill_url={}",
                decode.url(),
                prefill.url()
            );

            // Tick prefill by its real status (4xx = client fault). Don't record
            // decode: it was cancelled due to a prefill fault, not its own, so a
            // prefill error storm can't trip healthy decode breakers.
            let prefill_ok = match &prefill_result {
                Ok(r) => r.status().is_client_error(),
                Err(_) => false,
            };
            prefill.record_outcome(prefill_ok);

            // Status-faithful error shaping (4xx forwarded, transport/5xx -> 502).
            // Typed PD failures use the same unary/SSE terminal shape regardless
            // of which paired side resolves first.
            let mut response = match prefill_result {
                Ok(response) if raw_generate::has_typed_error(response.headers()) => {
                    self.handle_typed_pd_error_response(
                        response,
                        context.is_stream,
                        Arc::clone(&prefill),
                        Arc::clone(&decode),
                    )
                    .await
                }
                result => match self
                    .process_prefill_response(result, prefill.url(), false)
                    .await
                {
                    Err(error_response) => error_response,
                    Ok(_) => error::bad_gateway(
                        "prefill_server_error",
                        "Prefill reported failure but returned a success response".to_string(),
                    ),
                },
            };
            response.extensions_mut().insert(BreakerOutcomesRecorded);
            return response;
        }

        // A successful `reqwest::send` only proves that Prefill returned HTTP
        // headers. The Scheduler clears its Rust handle immediately before it
        // publishes the response body, so drain that body while still holding
        // the rendezvous gate. Releasing on headers allows Decode to finish and
        // the outer item permit to recycle before the Prefill slot is clear,
        // creating a transient 33rd handle at the frozen capacity of 32.
        let prefill_body = match self
            .process_prefill_response(prefill_result, prefill.url(), context.return_logprob)
            .await
        {
            Ok((_, body)) => body,
            Err(error_response) => return error_response,
        };
        drop(rendezvous_guard);

        // Prefill ok: take decode's result, awaiting it if still pending.
        let decode_result = match decode_early {
            Some(dr) => dr,
            None => (&mut decode_fut).await,
        };

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

                    // Per-worker breaker attribution before the synthetic 5xx
                    // response takes over. Prefill ran concurrently in the
                    // `tokio::join!`: tick it based on its actual response
                    // status, not on the decode-driven failure. For
                    // non-streaming the response carries no tracked stream
                    // so record decode's outcome here too — but treat 4xx
                    // as a client fault rather than a worker fault, matching
                    // the legacy outer-dispatcher rule and the streaming
                    // `BreakerTrackedStream` pre-mark in
                    // `create_streaming_response`. For streaming
                    // `handle_decode_error_response` wraps the synthetic
                    // error SSE in a `BreakerTrackedStream` that ticks
                    // decode on drop, so skip to avoid double-counting.
                    // Mark the response so the outer dispatcher skips its
                    // status-derived `record_outcome`.
                    prefill.record_outcome(true);
                    if !context.is_stream {
                        let decode_ok = status.is_success() || status.is_client_error();
                        decode.record_outcome(decode_ok);
                    }

                    let mut response = self
                        .handle_decode_error_response(res, &context, prefill, decode)
                        .await;
                    response.extensions_mut().insert(BreakerOutcomesRecorded);
                    return response;
                }

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
                        Some(response_headers),
                        prefill,
                        decode,
                    )
                } else {
                    // Non-streaming response
                    if context.return_logprob {
                        self.process_non_streaming_response(
                            res,
                            status,
                            context.return_logprob,
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
                // Decode failed at TCP/transport level. No tracked
                // stream will ever wrap a response (streaming path) and
                // we shortcut past the outer non-streaming
                // `record_outcome` too — so record decode failure
                // directly. Prefill ran concurrently in the
                // `tokio::join!`: record its real per-worker outcome
                // (success on a 2xx/4xx send, failure on transport
                // error) so the decode-driven 502 doesn't penalise a
                // healthy prefill. Mark the response so the outer
                // dispatcher skips its status-derived `record_outcome`
                // and we don't double-count.
                decode.record_outcome(false);
                prefill.record_outcome(true);

                let mut response = error::bad_gateway(
                    "decode_server_error",
                    format!("Decode server error: {}", e),
                );
                response.extensions_mut().insert(BreakerOutcomesRecorded);
                response
            }
        }
    }
}
