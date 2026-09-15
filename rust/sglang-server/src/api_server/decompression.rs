//! The Python HTTP server's opt-in `x-body-compressed: zstd` contract.

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::Request;
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::{Next, from_fn};
use axum::response::{IntoResponse, Response};

pub(super) fn apply(router: Router, enabled: bool) -> Router {
    if enabled {
        router.layer(from_fn(decompress))
    } else {
        router
    }
}

async fn decompress(request: Request, next: Next) -> Response {
    let Some(method) = request.headers().get("x-body-compressed") else {
        return next.run(request).await;
    };
    if method != "zstd" {
        let method: String = method
            .as_bytes()
            .iter()
            .map(|&byte| char::from(byte))
            .collect();
        let quote = if method.contains('\'') && !method.contains('"') {
            '"'
        } else {
            '\''
        };
        let escaped = method
            .replace('\\', "\\\\")
            .replace(quote, &format!("\\{quote}"));
        return (
            StatusCode::BAD_REQUEST,
            format!("unsupported x-body-compressed {quote}{escaped}{quote}; supported: ['zstd']"),
        )
            .into_response();
    }
    let (mut parts, body) = request.into_parts();
    // The Python middleware and ordinary generation endpoint have no body limit.
    let Ok(compressed) = to_bytes(body, usize::MAX).await else {
        return (StatusCode::BAD_REQUEST, "decompress failed").into_response();
    };
    let result =
        tokio::task::spawn_blocking(move || zstd::stream::decode_all(compressed.as_ref())).await;
    let body = match result {
        Ok(Ok(body)) => body,
        error => {
            tracing::warn!(?error, "request body decompress failed");
            return (StatusCode::BAD_REQUEST, "decompress failed").into_response();
        }
    };
    parts.headers.remove("x-body-compressed");
    parts
        .headers
        .insert(header::CONTENT_LENGTH, HeaderValue::from(body.len()));
    next.run(Request::from_parts(parts, Body::from(body))).await
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use axum::http::Method;
    use axum::routing::post;
    use tower::ServiceExt;

    // Produced by Python zstandard.ZstdCompressor().compress, as sent by clients.
    pub(crate) fn python_zstd_request() -> Vec<u8> {
        let hex = "28b52ffd2039c901007b22696e7075745f696473223a5b31315d2c2273616d706c696e675f706172616d73223a7b226d61785f6e65775f746f6b656e73223a337d7d";
        hex.as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }

    async fn echo(request: Request) -> Response {
        assert!(!request.headers().contains_key("x-body-compressed"));
        let length = request.headers()[header::CONTENT_LENGTH]
            .to_str()
            .unwrap()
            .parse::<usize>()
            .unwrap();
        let body = to_bytes(request.into_body(), usize::MAX).await.unwrap();
        assert_eq!(length, body.len());
        assert_eq!(
            body.as_ref(),
            br#"{"input_ids":[11],"sampling_params":{"max_new_tokens":3}}"#
        );
        body.into_response()
    }

    #[tokio::test]
    async fn python_compressed_post_and_put_rewrite_the_body_and_headers() {
        let app = apply(Router::new().route("/generate", post(echo).put(echo)), true);
        for method in [Method::POST, Method::PUT] {
            let body = python_zstd_request();
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri("/generate")
                        .header("x-body-compressed", "zstd")
                        .header(header::CONTENT_LENGTH, body.len())
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
    }

    #[tokio::test]
    async fn invalid_compression_returns_the_python_error_before_streaming() {
        let app = apply(
            Router::new().fallback(|| async { StatusCode::IM_A_TEAPOT }),
            true,
        );
        for (method, body, expected) in [
            (
                "gzip",
                b"data".as_slice(),
                "unsupported x-body-compressed 'gzip'; supported: ['zstd']",
            ),
            ("zstd", b"{\"stream\":true}".as_slice(), "decompress failed"),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri("/generate")
                        .header("x-body-compressed", method)
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert_eq!(
                to_bytes(response.into_body(), usize::MAX).await.unwrap(),
                expected
            );
        }
    }

    #[tokio::test]
    async fn disabled_decompression_and_missing_header_pass_through() {
        for (enabled, compression) in [(false, Some("zstd")), (true, None)] {
            let app = apply(
                Router::new().fallback(|request: Request| async move {
                    to_bytes(request.into_body(), usize::MAX).await.unwrap()
                }),
                enabled,
            );
            let mut request = Request::builder().method(Method::POST).uri("/generate");
            if let Some(method) = compression {
                request = request.header("x-body-compressed", method);
            }
            let response = app
                .oneshot(request.body(Body::from("untouched")).unwrap())
                .await
                .unwrap();
            assert_eq!(
                to_bytes(response.into_body(), usize::MAX).await.unwrap(),
                "untouched"
            );
        }
    }
}
