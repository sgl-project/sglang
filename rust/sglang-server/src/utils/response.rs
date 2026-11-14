//! Shared HTTP error-response shaping for the api-server endpoint modules.
//!
//! Two mechanics live here; the WIRE SHAPES stay owned by their endpoints:
//! the native `{"error": {"message", "code"}}` body (Python
//! `http_server.generate_request` parity) built by [`error_value`] and formed
//! by [`json_error`], and the SSE variant [`sse_error_response`] used by any
//! endpoint family (native and OpenAI alike — the caller supplies its own
//! body shape). The OpenAI error payload and the PD bootstrap registry's
//! plain-text bodies are protocol-owned and deliberately not unified here.

use std::convert::Infallible;

use axum::{
    Json,
    http::{HeaderValue, StatusCode, header::CONTENT_TYPE},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use futures::Stream;

/// The streaming content type Python's `StreamingResponse` sends: a `text/*`
/// media type gets its charset, so clients decode the frames as UTF-8.
pub const SSE_CONTENT_TYPE: &str = "text/event-stream; charset=utf-8";

/// An SSE response over `events`, with Python's content type.
pub fn sse_response(
    events: impl Stream<Item = Result<Event, Infallible>> + Send + 'static,
) -> Response {
    let mut response = Sse::new(events).into_response();
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(SSE_CONTENT_TYPE));
    response
}

/// The native error body — the same `{"error": {...}}` object every native
/// path emits, not bare text, which a client parsing JSON chokes on.
pub fn error_value(code: u16, message: &str) -> serde_json::Value {
    serde_json::json!({ "error": { "message": message, "code": code } })
}

/// Unary native-shape error response: `code` + [`error_value`] body.
pub fn json_error(code: StatusCode, message: &str) -> Response {
    error_response(code, error_value(code.as_u16(), message), false)
}

/// Form an error in the shape the client committed to: unary → `code` plus
/// the JSON `body`; streaming → 200 with one SSE error frame + `[DONE]` (the
/// client is already reading a stream — Python answers in-stream too, from
/// `stream_results()`). The `body` is caller-shaped: native [`error_value`]
/// or the OpenAI error payload.
pub fn error_response(code: StatusCode, body: serde_json::Value, stream: bool) -> Response {
    if !stream {
        return (code, Json(body)).into_response();
    }
    sse_error_response(body)
}

/// A 200 SSE response carrying one error frame + `[DONE]` — how a stream the
/// client is already committed to reading reports a failure. Shared by every
/// endpoint family: the native API and the OpenAI
/// frontend's `openai_error_response`.
pub fn sse_error_response(body: serde_json::Value) -> Response {
    let frames = [body.to_string(), "[DONE]".to_string()];
    sse_response(futures::stream::iter(
        frames.map(|data| Ok::<_, Infallible>(Event::default().data(data))),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python-parity pins for the native error shapes (`generate_request`):
    /// unary errors are a 4xx/5xx with a JSON `{"error": ...}` body; streaming
    /// ones are 200 + one SSE error frame + `[DONE]`, because Python answers
    /// from inside `stream_results()` once the stream is committed.
    #[tokio::test]
    async fn error_responses_match_python_shape() {
        let unary = error_response(
            StatusCode::BAD_REQUEST,
            error_value(400, "bad input"),
            false,
        );
        assert_eq!(unary.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(unary.into_body(), 64 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).expect("JSON body");
        assert_eq!(v["error"]["message"], "bad input");
        assert_eq!(v["error"]["code"], 400);

        let streamed = error_response(StatusCode::BAD_REQUEST, error_value(400, "bad input"), true);
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
