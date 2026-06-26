// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Anthropic `/v1/messages` passthrough route.
//!
//! Forwards the Anthropic request body to a selected SGLang worker at
//! `/v1/messages` without translating to/from OpenAI chat completions — the
//! worker natively serves `/v1/messages`. The router only needs `model` and
//! `stream` from the body for worker selection and buffered-vs-SSE routing.
//!
//! Deliberately does NOT replicate chat.rs's `input_ids` forwarding, PD
//! bootstrap injection, or decode-peer resolution (see design.md). It DOES
//! register active-load + hold the per-worker LoadGuard so load-aware
//! policies (`power_of_two`, `cache_aware_zmq`) see accurate in-flight
//! counts — without this the LB scheme would route on stale load.

use crate::discovery::{ModelId, WorkerMode};
use crate::policies::registry::{PdPoolResolver, PdResolveError};
use crate::policies::SelectionContext;
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::{RequestOutcome, WorkerModeLabel};
use crate::workers::LoadGuard;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Response};
use bytes::Bytes;
use serde::Deserialize;
use std::sync::Arc;

/// Per-route body cap, mirroring chat. Same rationale: bound heap allocation
/// before forwarding while accommodating long contexts.
pub const MAX_MESSAGES_BODY_BYTES: usize = 5 << 20;

/// Minimal probe: `model` selects the worker, `stream` picks buffered vs SSE.
/// The worker is authoritative for the full Anthropic schema. `#[serde(default)]`
/// keeps it tolerant of optional fields — only `model` is required.
#[derive(Debug, Deserialize)]
struct MessagesProbe {
    #[serde(default)]
    stream: Option<bool>,
    model: Option<String>,
}

fn parse_probe(body: &Bytes) -> Result<MessagesProbe, ApiError> {
    serde_json::from_slice(body)
        .map_err(|_| ApiError::BadRequest("invalid request: body must be a JSON object".into()))
}

/// POST /v1/messages — select a worker via the per-model policy and proxy the
/// raw Anthropic body to `<worker>/v1/messages`.
pub async fn messages(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response<Body> {
    match messages_inner(State(ctx), headers, body).await {
        Ok(resp) => resp,
        Err(e) => anthropic_error_response(e),
    }
}

/// Map a router-originated `ApiError` to an Anthropic Messages error envelope
/// `{"type":"error","error":{"type":...,"message":...}}` so Anthropic SDK clients
/// parse router-side failures (missing `model`, PD-reject, no healthy worker,
/// breaker open, stale). Worker-originated errors are forwarded verbatim by the
/// proxy and are already Anthropic-shaped, so they never reach here.
///
/// `message` comes from `ApiError::client_message()` — the same sanitized string
/// the OpenAI envelope uses, so no worker URL / anyhow chain leaks.
fn anthropic_error_response(e: ApiError) -> Response<Body> {
    use serde::Serialize;
    #[derive(Serialize)]
    struct AnthropicErr {
        #[serde(rename = "type")]
        typ: &'static str,
        message: String,
    }
    #[derive(Serialize)]
    struct Envelope {
        #[serde(rename = "type")]
        typ: &'static str,
        error: AnthropicErr,
    }
    let status = e.status_code();
    let typ = match status.as_u16() {
        400 => "invalid_request_error",
        404 => "not_found_error",
        503 => "overloaded_error",
        _ => "api_error",
    };
    let message = e.client_message();
    let body = serde_json::to_vec(&Envelope {
        typ: "error",
        error: AnthropicErr { typ, message },
    })
    .unwrap_or_else(|_| {
        b"{\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"internal error\"}}"
            .to_vec()
    });
    let mut r = Response::new(Body::from(body));
    *r.status_mut() = status;
    r.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/json"),
    );
    r
}

async fn messages_inner(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let start = std::time::Instant::now();
    let probe = parse_probe(&body)?;
    let streaming = probe.stream.unwrap_or(false);
    let model_str = probe
        .model
        .ok_or_else(|| ApiError::BadRequest("missing `model` field".into()))?;
    let model_id = ModelId(model_str.clone());

    // Same candidate set as chat (prefill pool for PD; full set for plain).
    let resolver = PdPoolResolver::new(Arc::clone(&ctx.registry));
    let workers = resolver
        .prefill_candidates(&model_id)
        .map_err(|e| match e {
            PdResolveError::NoHealthyWorkers => ApiError::NoHealthyWorkers {
                model: model_str.clone(),
            },
            PdResolveError::NoPrefillWorkersAvailable => ApiError::NoPrefillWorkersAvailable {
                model: model_str.clone(),
            },
            PdResolveError::NoDecodeWorkersAvailable => ApiError::NoDecodeWorkersAvailable {
                model: model_str.clone(),
            },
        })?;

    let policy = ctx
        .policies
        .get(&model_id)
        .ok_or_else(|| ApiError::ModelNotFound(model_str.clone()))?;

    // Deliberately do NOT produce routing tokens for /v1/messages.
    //
    // The cache-aware-zmq policy hashes the request to find a worker that
    // already holds the prefix in its KV cache. For chat completions the
    // router's chat-encoder tokenization matches the engine's cached blocks.
    // For Anthropic bodies that is NOT true: the worker's Anthropic serving
    // path folds the top-level `system` prompt into the actual prompt before
    // tokenizing, so hashing only `messages` (which `request_tokens_for`
    // would read) would route two requests with identical `messages` but
    // different `system` to the same cached worker — a false locality hit.
    // Rather than replicate the engine's `system`-folding exactly, we skip
    // router-side tokenization here: cache_aware_zmq falls back to min-load
    // for /v1/messages, and the worker tokenizes the body itself (we never
    // forward input_ids). Correct and safe; costs cache affinity on this
    // route, which is the honest trade-off until the engine exposes its
    // Anthropic block hashes.
    let request_tokens: Option<crate::policies::RequestTokens> = None;

    let routing_key = ctx
        .config
        .model
        .sticky
        .as_ref()
        .and_then(|s| headers.get(s.header_name.as_str()))
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty());
    // Pass NO body to the selection context. With `request_tokens = None`
    // AND no body, CacheAwareZmqPolicy::select cannot tokenize (its fallback
    // reads the body) and falls back to min-load — which is exactly the honest
    // behavior we want for /v1/messages (see the request_tokens comment).
    // Passing the body would let the policy tokenize `messages` and reintroduce
    // the false-locality bug (identical `messages`, different `system`).
    // `routing_key` is still honored by the sticky policy (it reads headers, not body).
    let selection_ctx = SelectionContext::with_routing_key(&model_id, None, routing_key)
        .with_request_tokens(request_tokens.as_ref().map(|t| t.ids.as_slice()));
    let worker =
        policy
            .select(&workers, &selection_ctx)
            .ok_or_else(|| ApiError::PolicySelectionFailed {
                model: model_str.clone(),
            })?;

    // PD-disaggregated mode: this passthrough only forwards to a single worker
    // and does NOT replicate chat.rs's decode-peer resolution + bootstrap body
    // injection (see design.md). In PD mode the final response comes from the
    // decode side, so silently forwarding to a prefill worker would hang. Fail
    // explicitly instead of degrading silently. Plain mode is unaffected.
    if worker.mode() != WorkerMode::Plain {
        return Err(ApiError::BadRequest(
            "/v1/messages passthrough does not support PD-disaggregated mode yet; use /v1/chat/completions".into(),
        ));
    }

    // Hold the per-worker in-flight guard + register active load so load-aware
    // policies see this request. prefill_load uses the real token count when we
    // tokenized, else the byte heuristic (same as chat).
    let guard = worker.load_guard();
    let prefill_load = request_tokens
        .as_ref()
        .map(|t| t.ids.len().max(1))
        .unwrap_or_else(|| crate::server::routes::chat::estimate_prefill_tokens(&body));
    let active_guard =
        ctx.active_load
            .register(worker.id.clone(), worker.url.clone(), prefill_load, 0);
    let stale_token = active_guard.cancel_token().clone();
    let metrics_worker_url = worker.url.clone();
    let metrics_model = model_str.clone();
    let metrics_mode = match worker.mode() {
        WorkerMode::Prefill => WorkerModeLabel::Prefill,
        WorkerMode::Decode => WorkerModeLabel::Decode,
        WorkerMode::Plain => WorkerModeLabel::Plain,
    };

    let result = if streaming {
        // ponytail: skip chat's TTFT hook + streaming-duration RAII guard —
        // v1 measures header-time only; end-to-end streaming latency is a
        // follow-up. Guards move into stream_guards so load stays accurate.
        let stream_guards: Box<dyn Send + 'static> = Box::new((guard, active_guard));
        let fetch = ctx.proxy.forward_streaming_to(
            &worker.url,
            &worker.breaker,
            "/v1/messages",
            &headers,
            body,
            Some(stream_guards),
            None,
        );
        tokio::select! {
            biased;
            r = fetch => r,
            _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
        }
    } else {
        let _holds: (LoadGuard, _) = (guard, active_guard);
        let fetch =
            ctx.proxy
                .forward_json_to(&worker.url, &worker.breaker, "/v1/messages", &headers, body);
        tokio::select! {
            biased;
            r = fetch => r,
            _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
        }
    };

    let outcome = match &result {
        Ok(_) => RequestOutcome::Success,
        Err(ApiError::StaleRequestExpired { .. }) => {
            ctx.metrics
                .record_stale_request(crate::server::metrics::StaleRequestOutcome::Expired);
            RequestOutcome::Cancelled
        }
        Err(_) => RequestOutcome::Error,
    };
    ctx.metrics
        .record_request(&metrics_worker_url, &metrics_model, metrics_mode, outcome);

    let elapsed = start.elapsed();
    if !streaming {
        ctx.metrics
            .observe_request_duration(&metrics_model, elapsed.as_secs_f64());
    }
    let http_status = match &result {
        Ok(resp) => resp.status().as_u16(),
        Err(e) => e.status_code().as_u16(),
    };
    ctx.metrics.record_response(http_status);
    let request_id = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-");
    tracing::info!(
        request_id = %request_id,
        method = "POST",
        path = "/v1/messages",
        model = %metrics_model,
        worker = %metrics_worker_url,
        outcome = match outcome {
            RequestOutcome::Success => "success",
            RequestOutcome::Error => "error",
            RequestOutcome::Cancelled => "cancelled",
        },
        http_status,
        stream = streaming,
        latency_ms = elapsed.as_millis() as u64,
        "messages",
    );
    result
}

#[cfg(test)]
mod tests {
    use super::parse_probe;
    use bytes::Bytes;

    #[test]
    fn probe_reads_stream_and_model() {
        let b = Bytes::from(r#"{"model":"glm","stream":true,"messages":[]}"#);
        let p = parse_probe(&b).unwrap();
        assert_eq!(p.stream, Some(true));
        assert_eq!(p.model.as_deref(), Some("glm"));
    }

    #[test]
    fn probe_stream_defaults_to_false() {
        let b = Bytes::from(r#"{"model":"glm","messages":[]}"#);
        let p = parse_probe(&b).unwrap();
        assert_eq!(p.stream, None);
        assert_eq!(p.model.as_deref(), Some("glm"));
    }

    #[test]
    fn probe_rejects_non_object() {
        let b = Bytes::from(b"\"hi\"".as_ref());
        assert!(parse_probe(&b).is_err(), "string body must not parse");
    }

    #[test]
    fn probe_allows_anthropic_shape_without_stream() {
        // Real Anthropic body has system/messages/max_tokens; only model is required here.
        let b = Bytes::from(
            r#"{"model":"claude-3","max_tokens":256,"system":"s","messages":[{"role":"user","content":"hi"}]}"#,
        );
        let p = parse_probe(&b).unwrap();
        assert_eq!(p.model.as_deref(), Some("claude-3"));
        assert_eq!(p.stream, None);
    }
}
