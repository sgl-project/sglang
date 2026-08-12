//! Request submission into the ingress pipeline, shared by every endpoint
//! module: mint the client-visible rid (uuid hex, Python-parity), build the
//! `Request`, and hand it to the TM with an egress receiver for the response.

use std::convert::Infallible;

use axum::{
    Json,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use tokio::sync::mpsc;

use super::AppState;
use super::frame::error_value;
use crate::fsm::RequestState;
use crate::ids::Rid;
use crate::message::{EgressItem, EgressSink, Request, RequestKind};
use crate::tokenizer_manager::TmEvent;

/// Submit one request; returns the rid, its hashed routing key, and the egress
/// receiver. Every request arrives with its final rid — a generate request from
/// `into_requests` (or the `HEALTH_CHECK_<uuid>` the health probe sets), a
/// control request from its constructor — so this only echoes it back.
pub(super) async fn submit(
    state: &AppState,
    kind: RequestKind,
    // `stream`: the client is reading an SSE stream, so it expects 200 plus an
    // error frame rather than a 4xx — same rule `pre_submit_error` applies
    // everywhere else.
    stream: bool,
) -> Result<(Rid, mpsc::Receiver<EgressItem>), Response> {
    let rid = match &kind {
        // Generate rids are already final: `GenerateBody::into_requests` normalized the
        // client's, or minted one. Control requests have no client-facing rid.
        RequestKind::Generate(g) => g.rid.clone(),
        RequestKind::Control(c) => c.rid().into(),
        // Internal service call — no client-facing rid; mint a fresh one.
        RequestKind::Detokenize { .. } => Rid::new(),
    };
    // Two in-flight requests can name the same client rid, but they cannot share a
    // `Rid`: `into_requests` built each through `Rid::from_client`, which appends a
    // uniquifier. So nothing here needs to check for a collision — the detok table
    // key is unique by construction, and `client_facing` restores what the client
    // sent for `meta_info.id`.
    // Async-aware send so a full TM inbox yields (backpressure) instead of parking
    // a thread; Err only when the inbox is closed (shutdown).
    let (tx, rx) = mpsc::channel::<EgressItem>(state.egress_buf);
    let request = Request {
        rid: rid.clone(),
        state: RequestState::Received,
        sink: EgressSink::Local(tx),
        kind,
    };
    match state.senders.tm.send_async(TmEvent::Ingress(request)).await {
        Ok(()) => Ok((rid, rx)),
        // `SendError` has a single meaning — the channel is disconnected.
        Err(_) => {
            tracing::error!(%rid, "tm inbox closed; request rejected");
            // Return 503 so the client can retry.
            Err(pre_submit_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "service unavailable",
                stream,
            ))
        }
    }
}

/// Shape an error that occurs *before* (or instead of) a successful submit into a
/// client response. Two parity points with Python's `generate_request`. The body
/// is the same `{"error": {...}}` object every other path emits — not bare text,
/// which a client parsing JSON chokes on. And a streaming request gets 200 plus
/// one SSE error frame and `[DONE]`, not a 4xx: the client has already committed
/// to reading a stream, and Python answers it inside `stream_results()`.
pub(super) fn pre_submit_error(code: StatusCode, message: &str, stream: bool) -> Response {
    let body = error_value(code.as_u16(), message);
    if !stream {
        return (code, Json(body)).into_response();
    }
    sse_error_response(body)
}

/// A 200 SSE response carrying one error frame + `[DONE]` — how a stream the
/// client is already committed to reading reports a failure. Shared by every
/// endpoint family: the native API via [`pre_submit_error`] and the OpenAI
/// frontend's `openai_error_response`.
pub(super) fn sse_error_response(body: serde_json::Value) -> Response {
    let frames = [body.to_string(), "[DONE]".to_string()];
    Sse::new(futures::stream::iter(
        frames.map(|data| Ok::<_, Infallible>(Event::default().data(data))),
    ))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unary pre-submit errors are a 4xx/5xx with a JSON `{"error":...}` body;
    /// streaming ones are 200 + an SSE error frame + `[DONE]`, because Python
    /// answers from inside `stream_results()` once the stream is committed.
    #[tokio::test]
    async fn pre_submit_errors_match_python_shape() {
        let unary = pre_submit_error(StatusCode::BAD_REQUEST, "bad input", false);
        assert_eq!(unary.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(unary.into_body(), 64 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).expect("JSON body");
        assert_eq!(v["error"]["message"], "bad input");
        assert_eq!(v["error"]["code"], 400);

        let streamed = pre_submit_error(StatusCode::BAD_REQUEST, "bad input", true);
        assert_eq!(
            streamed.status(),
            StatusCode::OK,
            "the stream itself is 200"
        );
        let body = axum::body::to_bytes(streamed.into_body(), 64 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(
            text.contains(r#""code":400"#),
            "carries the status in-band: {text}"
        );
        assert!(
            text.trim_end().ends_with("data: [DONE]"),
            "terminated: {text}"
        );
    }
}
