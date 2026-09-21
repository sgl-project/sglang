//! The native SGLang data-plane HTTP handlers: `/generate` (unary JSON, a
//! batch array, or SSE `data: {json}` … `[DONE]`, byte-compatible with Python
//! `http_server.generate_request`) and `/health` + `/health_generate` (which
//! round-trip a 1-token generate probe). The transport-neutral halves live in
//! `api_server::core::{generate, health}`; these handlers only extract, pick the
//! response shape, and encode.

use std::sync::Arc;

use http::StatusCode;

use super::app::AppState;
use super::response::error_response;
use super::response::sse_encode;
use super::response::{HttpResponse, bytes_response, read_json, status_response};
use crate::api_server::core::frame::{error_value, frame_typed};
use crate::api_server::core::generate::{
    UnaryDrainPolicy, UnaryOutcome, add_e2e_latency, drain_plan_unary, generate_start,
    generation_event_stream,
};
use crate::api_server::core::health::{HealthStatus, health_probe};

/// native api error response: unary → `code` plus the JSON `body`,
/// streaming → 200 with one SSE error frame + `[DONE]`.
pub(super) fn native_error(code: StatusCode, message: &str, stream: bool) -> HttpResponse {
    error_response(code, error_value(code.as_u16(), message), stream)
}

/// One rendered unary response: the HTTP status code, the serialized result or
/// `{"error": …}` JSON body, and whether a real terminal item arrived (`false`
/// = truncation; the caller keeps the abort guard armed).
struct NativeUnary {
    code: u16,
    body: String,
}

/// The HTTP JSON rendering of a folded [`UnaryOutcome`].
fn render_unary(outcome: UnaryOutcome, rid: &str) -> NativeUnary {
    match outcome {
        UnaryOutcome::Complete(out, timing) => {
            let mut frame = frame_typed(&out, rid);
            add_e2e_latency(&mut frame, &timing);
            NativeUnary {
                code: 200,
                body: serde_json::to_string(&frame).expect("a generated frame always serializes"),
            }
        }
        UnaryOutcome::Error { code, message } => NativeUnary {
            code,
            body: error_value(code, &message).to_string(),
        },
        UnaryOutcome::Truncated => NativeUnary {
            code: 500,
            body: error_value(500, "response truncated before completion").to_string(),
        },
    }
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
    let is_batch = plan.is_batch;
    if stream {
        // A single request is a 1-element batch without the `index` field — the
        // same multiplexed stream serves both, so the frame/abort/truncation
        // logic lives in one place. `guard` moves into the stream so a client
        // disconnect aborts what's unfinished.
        sse_encode(generation_event_stream(plan))
    } else {
        let drained = match drain_plan_unary(plan, UnaryDrainPolicy::PerItem).await {
            Ok(drained) => drained,
            Err(e) => return native_error(e.http_status(), &e.message, false),
        };
        let mut rendered = drained
            .into_iter()
            .map(|(rid, outcome)| render_unary(outcome, rid.client_facing()));
        if !is_batch {
            // `into_requests` guarantees exactly one payload for a non-batch body.
            let unary = rendered.next().expect("into_requests yields one payload");
            let status =
                StatusCode::from_u16(unary.code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            bytes_response(status, "application/json", unary.body.into_bytes())
        } else {
            // The shared per-item drain preserves input order; each item keeps its
            // own terminal/error shape while the batch itself remains HTTP 200.
            let results: Vec<_> = rendered.map(|unary| unary.body).collect();
            bytes_response(
                StatusCode::OK,
                "application/json",
                format!("[{}]", results.join(",")).into_bytes(),
            )
        }
    }
}
