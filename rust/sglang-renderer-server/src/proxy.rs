//! Streaming HTTP fallback to the native Rust server.

use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, Request, Response, StatusCode, header};
use axum::response::IntoResponse;

#[derive(Clone)]
pub(super) struct RustServerProxy {
    client: reqwest::Client,
    upstream_url: String,
}

impl RustServerProxy {
    pub(super) fn new(upstream_url: String) -> Result<Self, String> {
        let upstream_url = upstream_url.trim_end_matches('/').to_owned();
        reqwest::Url::parse(&upstream_url)
            .map_err(|error| format!("invalid proxy upstream {upstream_url:?}: {error}"))?;
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| format!("building Rust-server proxy client failed: {error}"))?;
        Ok(Self {
            client,
            upstream_url,
        })
    }

    pub(super) async fn forward(&self, request: Request<Body>) -> Response<Body> {
        let (mut parts, body) = request.into_parts();
        strip_hop_by_hop_headers(&mut parts.headers);
        // Let the client set Host for the upstream origin.
        parts.headers.remove(header::HOST);
        let path = parts
            .uri
            .path_and_query()
            .map_or("/", axum::http::uri::PathAndQuery::as_str);
        let upstream = format!("{}{path}", self.upstream_url);
        let response = self
            .client
            .request(parts.method, upstream)
            .headers(parts.headers)
            .body(reqwest::Body::wrap_stream(body.into_data_stream()))
            .send()
            .await;
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                tracing::error!(%error, "Rust-server proxy request failed");
                return (StatusCode::BAD_GATEWAY, "Rust server unavailable").into_response();
            }
        };

        let status = response.status();
        let mut headers = response.headers().clone();
        strip_hop_by_hop_headers(&mut headers);
        let mut builder = Response::builder().status(status);
        *builder
            .headers_mut()
            .expect("response builder must expose headers") = headers;
        builder
            .body(Body::from_stream(response.bytes_stream()))
            .unwrap_or_else(|error| {
                tracing::error!(%error, "building Rust-server proxy response failed");
                (StatusCode::BAD_GATEWAY, "Invalid Rust server response").into_response()
            })
    }
}

fn strip_hop_by_hop_headers(headers: &mut HeaderMap) {
    let connection_headers = headers
        .get(header::CONNECTION)
        .and_then(|value| value.to_str().ok())
        .into_iter()
        .flat_map(|value| value.split(','))
        .filter_map(|name| HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect::<Vec<_>>();
    for name in connection_headers {
        headers.remove(name);
    }
    for name in [
        header::CONNECTION,
        header::HeaderName::from_static("keep-alive"),
        header::PROXY_AUTHENTICATE,
        header::PROXY_AUTHORIZATION,
        header::TE,
        header::TRAILER,
        header::TRANSFER_ENCODING,
        header::UPGRADE,
    ] {
        headers.remove(name);
    }
}
