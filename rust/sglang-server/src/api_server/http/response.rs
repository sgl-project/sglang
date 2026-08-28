//! The hand-built HTTP plumbing the transport runs on: the body type, the
//! response builders, JSON extraction, and SSE framing — over `http` +
//! `http-body` + tower directly. The client-visible bytes (extractor
//! rejection texts, SSE headers, error statuses) reproduce the previous
//! axum edge exactly; `http_edge_wire_contract` pins them.

use bytes::Bytes;
use futures::StreamExt;
use http::StatusCode;
use http_body_util::BodyExt;

use crate::api_server::core::frame::error_value;

/// One boxed body type for every response: full buffers and SSE streams
/// alike (unsync-boxed, the same shape axum's own Body wrapped). Infallible
/// error: our streams never fail, they end.
pub(crate) type HttpBody =
    http_body_util::combinators::UnsyncBoxBody<Bytes, std::convert::Infallible>;

pub(crate) type HttpResponse = http::Response<HttpBody>;

pub(crate) fn full(bytes: impl Into<Bytes>) -> HttpBody {
    http_body_util::Full::new(bytes.into())
        .map_err(|never| match never {})
        .boxed_unsync()
}

pub(crate) fn empty() -> HttpBody {
    http_body_util::Empty::new()
        .map_err(|never| match never {})
        .boxed_unsync()
}

pub(crate) fn status_response(status: http::StatusCode) -> HttpResponse {
    http::Response::builder()
        .status(status)
        .body(empty())
        .expect("static response builds")
}

/// `(status, Json(value))` for a typed body: serde_json::to_vec directly,
/// byte-identical to axum's `Json` responder.
pub(crate) fn json_typed_response<T: serde::Serialize>(
    status: http::StatusCode,
    value: &T,
) -> HttpResponse {
    bytes_response(
        status,
        "application/json",
        serde_json::to_vec(value).unwrap_or_default(),
    )
}

/// `(status, Json(value))`: the shape axum's `Json` responder produced.
pub(crate) fn json_response(status: http::StatusCode, value: &serde_json::Value) -> HttpResponse {
    bytes_response(
        status,
        "application/json",
        serde_json::to_vec(value).unwrap_or_default(),
    )
}

pub(crate) fn bytes_response(
    status: http::StatusCode,
    content_type: &'static str,
    body: Vec<u8>,
) -> HttpResponse {
    http::Response::builder()
        .status(status)
        .header(http::header::CONTENT_TYPE, content_type)
        .body(full(body))
        .expect("static response builds")
}

/// Plain-text response (the `(StatusCode, String)` responder shape:
/// `text/plain; charset=utf-8`).
pub(crate) fn text_response(status: http::StatusCode, body: String) -> HttpResponse {
    bytes_response(status, "text/plain; charset=utf-8", body.into_bytes())
}

/// Known path, wrong method: axum's bare 405 — empty body, comma-joined
/// `Allow` (no spaces).
pub(crate) fn method_not_allowed(allow: &'static str) -> HttpResponse {
    http::Response::builder()
        .status(http::StatusCode::METHOD_NOT_ALLOWED)
        .header(http::header::ALLOW, allow)
        .body(empty())
        .expect("static response builds")
}

/// Wrap a stream of complete JSON frame payloads as the SSE response the
/// native and OpenAI endpoints share: one `data: {json}` event per frame,
/// terminated by the `data: [DONE]` sentinel. The sentinel is transport
/// framing, so it is appended here — core streams never yield it.
pub(super) fn sse_encode(
    frames: impl futures::Stream<Item = String> + Send + 'static,
) -> HttpResponse {
    sse_response(frames.chain(futures::stream::once(async { "[DONE]".to_string() })))
}

/// A committed SSE response: 200 + the headers axum's `Sse` set, each item
/// framed as one `data: {payload}\n\n` event.
pub(crate) fn sse_response(
    frames: impl futures::Stream<Item = String> + Send + 'static,
) -> HttpResponse {
    let body = http_body_util::StreamBody::new(frames.map(|payload| {
        let mut event = String::with_capacity(payload.len() + 8);
        event.push_str("data: ");
        event.push_str(&payload);
        event.push_str("\n\n");
        Ok::<_, std::convert::Infallible>(http_body::Frame::data(Bytes::from(event)))
    }));
    http::Response::builder()
        .status(http::StatusCode::OK)
        .header(http::header::CONTENT_TYPE, "text/event-stream")
        .header(http::header::CACHE_CONTROL, "no-cache")
        .body(HttpBody::new(body))
        .expect("static response builds")
}

/// `application/json` (or `application/*+json`), the check axum's `Json`
/// extractor applied.
fn json_content_type(headers: &http::HeaderMap) -> bool {
    let Some(content_type) = headers.get(http::header::CONTENT_TYPE) else {
        return false;
    };
    let Ok(content_type) = content_type.to_str() else {
        return false;
    };
    let Ok(mime) = content_type.parse::<mime::Mime>() else {
        return false;
    };
    mime.type_() == "application"
        && (mime.subtype() == "json" || mime.suffix().is_some_and(|name| name == "json"))
}

/// Unary native-shape error response: `code` + [`error_value`] body.
pub fn json_error(code: StatusCode, message: &str) -> HttpResponse {
    error_response(code, error_value(code.as_u16(), message), false)
}

/// Form an error in the shape the client committed to: unary → `code` plus
/// the JSON `body`; streaming → 200 with one SSE error frame + `[DONE]` (the
/// client is already reading a stream — Python answers in-stream too, from
/// `stream_results()`). The `body` is caller-shaped: native [`error_value`]
/// or the OpenAI error payload.
pub fn error_response(code: StatusCode, body: serde_json::Value, stream: bool) -> HttpResponse {
    if !stream {
        return json_response(code, &body);
    }
    sse_error_response(body)
}

/// A 200 SSE response carrying one error frame + `[DONE]` — how a stream the
/// client is already committed to reading reports a failure. Shared by every
/// endpoint family: the native API and the OpenAI
/// frontend's `openai_error_response`.
pub fn sse_error_response(body: serde_json::Value) -> HttpResponse {
    sse_response(futures::stream::iter([
        body.to_string(),
        "[DONE]".to_string(),
    ]))
}

/// axum-`JsonRejection` parity: the exact `body_text` clients saw, plus the
/// default status each rejection carried (415 for a missing JSON content
/// type, 422 for a type mismatch, 400 for malformed JSON). Handlers that
/// always answered 400 keep doing so by using only `body_text`; the PD
/// bootstrap routes answer with the default status like their bare `Json`
/// extractors did.
#[derive(Debug)]
pub(crate) struct JsonRejection {
    pub(crate) status: http::StatusCode,
    pub(crate) body_text: String,
}

/// Buffer and parse a JSON request body; `Err` carries the exact rejection
/// axum's `Json` extractor produced: serde_path_to_error paths, position
/// suffixes, and the trailing-content check included. No body-size cap,
/// matching the previous `DefaultBodyLimit::disable()`.
pub(crate) async fn read_json<T: serde::de::DeserializeOwned, B>(
    req: http::Request<B>,
) -> Result<T, JsonRejection>
where
    B: http_body::Body,
{
    if !json_content_type(req.headers()) {
        return Err(JsonRejection {
            status: http::StatusCode::UNSUPPORTED_MEDIA_TYPE,
            body_text: "Expected request with `Content-Type: application/json`".to_string(),
        });
    }
    let Ok(collected) = req.into_body().collect().await else {
        return Err(JsonRejection {
            status: http::StatusCode::BAD_REQUEST,
            body_text: "Failed to buffer the request body".to_string(),
        });
    };
    let bytes = collected.to_bytes();

    let syntax = |err: String| JsonRejection {
        status: http::StatusCode::BAD_REQUEST,
        body_text: format!("Failed to parse the request body as JSON: {err}"),
    };
    let mut deserializer = serde_json::Deserializer::from_slice(&bytes);
    let value: T = match serde_path_to_error::deserialize(&mut deserializer) {
        Ok(value) => value,
        Err(err) => {
            return Err(match err.inner().classify() {
                serde_json::error::Category::Data => JsonRejection {
                    status: http::StatusCode::UNPROCESSABLE_ENTITY,
                    body_text: format!(
                        "Failed to deserialize the JSON body into the target type: {err}"
                    ),
                },
                _ => syntax(err.to_string()),
            });
        }
    };
    deserializer.end().map_err(|err| syntax(err.to_string()))?;
    Ok(value)
}

impl crate::api_server::layers::RejectionBody for HttpBody {
    fn empty() -> Self {
        empty()
    }
    fn from_static(bytes: &'static [u8]) -> Self {
        full(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json_req(
        body: &str,
        content_type: Option<&str>,
    ) -> http::Request<http_body_util::Full<Bytes>> {
        let mut b = http::Request::builder().uri("/x");
        if let Some(ct) = content_type {
            b = b.header(http::header::CONTENT_TYPE, ct);
        }
        b.body(http_body_util::Full::new(Bytes::from(body.to_owned())))
            .unwrap()
    }

    /// The three rejection texts, byte-matching axum's `Json` extractor.
    #[tokio::test]
    async fn rejection_texts_match_axum() {
        #[derive(Debug, serde::Deserialize)]
        struct T {
            #[allow(dead_code)]
            model: String,
        }
        let err = read_json::<T, _>(json_req(r#"{"model": }"#, Some("application/json")))
            .await
            .unwrap_err();
        assert_eq!(err.status, http::StatusCode::BAD_REQUEST);
        assert_eq!(
            err.body_text,
            "Failed to parse the request body as JSON: model: expected value at line 1 column 11"
        );
        let err = read_json::<T, _>(json_req(r#"{"model": 3}"#, Some("application/json")))
            .await
            .unwrap_err();
        assert_eq!(err.status, http::StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            err.body_text,
            "Failed to deserialize the JSON body into the target type: model: invalid type: integer `3`, expected a string at line 1 column 11"
        );
        let err = read_json::<T, _>(json_req("{}", None)).await.unwrap_err();
        assert_eq!(err.status, http::StatusCode::UNSUPPORTED_MEDIA_TYPE);
        assert_eq!(
            err.body_text,
            "Expected request with `Content-Type: application/json`"
        );
        // Trailing content after a valid document is a syntax rejection.
        let err = read_json::<T, _>(json_req(
            r#"{"model": "m"} extra"#,
            Some("application/json"),
        ))
        .await
        .unwrap_err();
        assert!(
            err.body_text
                .starts_with("Failed to parse the request body as JSON: trailing characters"),
            "{}",
            err.body_text
        );
        // application/*+json passes the content-type gate.
        assert!(
            read_json::<T, _>(json_req(r#"{"model": "m"}"#, Some("application/ld+json")))
                .await
                .is_ok()
        );
    }

    /// SSE framing: the exact bytes and headers `sse_encode` shipped under
    /// axum (data-prefixed events; text/event-stream + no-cache).
    #[tokio::test]
    async fn sse_bytes_and_headers() {
        let res = sse_response(futures::stream::iter(vec![
            "{\"a\":1}".to_string(),
            "[DONE]".to_string(),
        ]));
        assert_eq!(
            res.headers().get(http::header::CONTENT_TYPE).unwrap(),
            "text/event-stream"
        );
        assert_eq!(
            res.headers().get(http::header::CACHE_CONTROL).unwrap(),
            "no-cache"
        );
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&bytes[..], b"data: {\"a\":1}\n\ndata: [DONE]\n\n");
    }

    /// The SSE wire shape: each frame is one `data:` event and the stream is
    /// terminated by `data: [DONE]` — appended by the encoder, byte-compatible
    /// with the Python server's streaming responses.
    #[tokio::test]
    async fn sse_encode_appends_done_sentinel() {
        let resp = sse_encode(futures::stream::iter(vec![
            "{\"text\":\"a\"}".to_string(),
            "{\"text\":\"b\"}".to_string(),
        ]));
        assert_eq!(resp.status(), http::StatusCode::OK);
        let body = http_body_util::BodyExt::collect(resp.into_body())
            .await
            .unwrap()
            .to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert_eq!(
            text,
            "data: {\"text\":\"a\"}\n\ndata: {\"text\":\"b\"}\n\ndata: [DONE]\n\n"
        );
    }

    #[tokio::test]
    async fn error_responses_match_python_shape() {
        let unary = error_response(
            StatusCode::BAD_REQUEST,
            error_value(400, "bad input"),
            false,
        );
        assert_eq!(unary.status(), StatusCode::BAD_REQUEST);
        let body = http_body_util::BodyExt::collect(unary.into_body())
            .await
            .unwrap()
            .to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&body).expect("JSON body");
        assert_eq!(v["error"]["message"], "bad input");
        assert_eq!(v["error"]["code"], 400);

        let streamed = error_response(StatusCode::BAD_REQUEST, error_value(400, "bad input"), true);
        assert_eq!(
            streamed.status(),
            StatusCode::OK,
            "the stream itself is 200"
        );
        let body = http_body_util::BodyExt::collect(streamed.into_body())
            .await
            .unwrap()
            .to_bytes();
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
