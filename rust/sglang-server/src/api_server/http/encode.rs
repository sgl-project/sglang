//! HTTP-side stream encoding: core event streams become SSE responses here.

use futures::StreamExt;

use super::plumbing::{HttpResponse, sse_response};

/// Wrap a stream of complete JSON frame payloads as the SSE response the
/// native and OpenAI endpoints share: one `data: {json}` event per frame,
/// terminated by the `data: [DONE]` sentinel. The sentinel is transport
/// framing, so it is appended here — core streams never yield it.
pub(super) fn sse_encode(
    frames: impl futures::Stream<Item = String> + Send + 'static,
) -> HttpResponse {
    sse_response(frames.chain(futures::stream::once(async { "[DONE]".to_string() })))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
