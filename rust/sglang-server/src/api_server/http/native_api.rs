//! The native SGLang data-plane HTTP handlers: `/generate` (unary JSON, a
//! batch array, or SSE `data: {json}` … `[DONE]`, byte-compatible with Python
//! `http_server.generate_request`) and `/health` + `/health_generate` (which
//! round-trip a 1-token generate probe). The transport-neutral halves live in
//! `api_server::core::{generate, health}`; these handlers only extract, pick the
//! response shape, and encode.

use std::sync::Arc;

use http::StatusCode;

use super::app::AppState;
use super::encode::sse_encode;
use super::plumbing::{HttpResponse, json_response, read_json, status_response};
use crate::api_server::core::generate::{
    GeneratePlan, drain_unary, generate_start, generation_event_stream,
};
use crate::api_server::core::health::{HealthStatus, health_probe};
use crate::utils::response::{error_response, error_value};

/// native api error response: unary → `code` plus the JSON `body`,
/// streaming → 200 with one SSE error frame + `[DONE]`.
pub(super) fn native_error(code: StatusCode, message: &str, stream: bool) -> HttpResponse {
    error_response(code, error_value(code.as_u16(), message), stream)
}

/// `GET /health_generate` — 200 if the response heartbeat advances within
/// `timeout` (from `SGLANG_HEALTH_CHECK_TIMEOUT`, frozen at router build),
/// else 503. (`/health` shares this handler when its env gate is on.)
pub(super) async fn health_generate(
    state: Arc<AppState>,
    timeout: std::time::Duration,
) -> HttpResponse {
    match health_probe(&state, timeout).await {
        Ok(HealthStatus::Alive) => status_response(StatusCode::OK),
        Ok(HealthStatus::Stalled) => status_response(StatusCode::SERVICE_UNAVAILABLE),
        Err(e) => native_error(e.http_status(), &e.message, false),
    }
}

/// `POST /generate` — the native generation endpoint. The body parses into
/// the schema-generated `GenerateRequest` (proto/sglang/api/v1 is the wire
/// contract) and converts into the internal fan-out input; [`generate_start`]
/// validates, fans out, and submits; this handler only picks the response
/// shape: SSE stream, one unary JSON object, or (batch) a JSON array.
///
/// The body is extracted as a `Result` so a deserialization failure is answered
/// with **400** (Python's status for a bad request) carrying serde's field-level
/// message, instead of axum's default 422.
pub(super) async fn generate<B: http_body::Body>(
    state: Arc<AppState>,
    req: http::Request<B>,
) -> HttpResponse {
    let body = match read_json::<sglang_api_types::api::v1::GenerateRequest, _>(req).await {
        Ok(body) => body,
        // A body that fails to parse has no readable `stream` flag, so this one
        // can only answer unary — as Python's does (FastAPI rejects before its
        // handler runs).
        Err(rejection) => {
            return native_error(StatusCode::BAD_REQUEST, &rejection.body_text, false);
        }
    };
    let stream = body.stream_or_default();
    let body = crate::message::convert::generate_body(body);
    let plan = match generate_start(&state, body).await {
        Ok(plan) => plan,
        // Answer an error raised *before* anything was submitted, in the shape
        // the client asked for.
        Err(e) => return native_error(e.http_status(), &e.message, stream),
    };
    let GeneratePlan {
        receivers,
        mut guard,
        is_batch,
        incremental,
    } = plan;

    if stream {
        // A single request is a 1-element batch without the `index` field — the
        // same multiplexed stream serves both, so the frame/abort/truncation
        // logic lives in one place. `guard` moves into the stream so a client
        // disconnect aborts what's unfinished.
        sse_encode(generation_event_stream(
            receivers,
            guard,
            incremental,
            is_batch,
        ))
    } else if !is_batch {
        // `into_requests` guarantees exactly one payload for a non-batch body.
        let (rid_str, mut rx, timing) = receivers
            .into_iter()
            .next()
            .expect("into_requests yields >=1 payload");
        // Unary: fold to the terminal, respond once. Disarm only on a real terminal
        // (a truncation leaves the guard armed so the scheduler work is aborted).
        let unary = drain_unary(&mut rx, rid_str.client_facing(), timing).await;
        if unary.terminal {
            guard.disarm(&rid_str);
        }
        let status = StatusCode::from_u16(unary.code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        json_response(status, &unary.body)
    } else {
        // Unary batch: poll every item concurrently, as Python's gather does.
        // `join_all` preserves input order for the final JSON array, while each
        // drain observes its own terminal output promptly (important for
        // per-item e2e_latency). A failed item is its own `{ "error": … }`
        // entry; the batch response is 200.
        let drained = futures::future::join_all(receivers.into_iter().map(
            |(rid_str, mut rx, request_timing)| async move {
                let client_rid = rid_str.client_facing().to_owned();
                let unary = drain_unary(&mut rx, &client_rid, request_timing).await;
                (rid_str, unary)
            },
        ))
        .await;
        let mut results = Vec::with_capacity(drained.len());
        for (rid_str, unary) in drained {
            if unary.terminal {
                guard.disarm(&rid_str);
            }
            results.push(unary.body);
        }
        json_response(StatusCode::OK, &serde_json::Value::Array(results))
    }
}
