//! Renderer process state and HTTP listener.

use std::net::SocketAddr;
use std::sync::Arc;

use crate::{DynamoTokenizer, RendererConfig, RendererService, TextTokenizer, load_tokenizer};
use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, Request, Response, StatusCode, header};
use axum::response::IntoResponse;

use crate::engine::HttpGenerateClient;
use crate::openai::{OpenAIHttpFrontend, hosted_routes, render_only_routes, standalone_routes};

#[derive(Clone, Debug)]
pub struct RendererRuntimeConfig {
    pub http_addr: SocketAddr,
    pub http_workers: usize,
    pub tokenizer_workers: usize,
    pub queue_capacity: usize,
    /// Optional SGLang engine origin. When absent, inference routes are not mounted.
    pub engine_url: Option<String>,
    /// Proxy routes not owned by the renderer to `engine_url`.
    pub proxy_unhandled_routes: bool,
    pub renderer: RendererConfig,
}

pub async fn serve(config: RendererRuntimeConfig) -> Result<(), String> {
    let mode = match (&config.engine_url, config.proxy_unhandled_routes) {
        (None, false) => "render-only",
        (Some(_), false) => "serving",
        (Some(_), true) => "hosted",
        (None, true) => return Err("proxy_unhandled_routes requires engine_url".to_string()),
    };
    let tokenizer_without_specials = load_tokenizer(
        (!config.renderer.tokenizer_path.is_empty())
            .then_some(config.renderer.tokenizer_path.as_str()),
        config.renderer.revision.as_deref(),
        false,
    )?;
    let tokenizer_with_specials = load_tokenizer(
        (!config.renderer.tokenizer_path.is_empty())
            .then_some(config.renderer.tokenizer_path.as_str()),
        config.renderer.revision.as_deref(),
        true,
    )?;
    let encode_tokenizer: Arc<dyn TextTokenizer> = Arc::new(DynamoTokenizer::new(
        tokenizer_without_specials.clone(),
        tokenizer_with_specials,
    ));
    let renderer = Arc::new(RendererService::with_tokenizer(
        config.renderer,
        encode_tokenizer,
        config.tokenizer_workers,
        config.queue_capacity,
    ));
    let app = match (config.engine_url, config.proxy_unhandled_routes) {
        (None, false) => render_only_routes(renderer),
        (Some(engine_url), false) => {
            let generate_client = HttpGenerateClient::new(engine_url, tokenizer_without_specials)?;
            standalone_routes(OpenAIHttpFrontend::new(renderer, generate_client))
        }
        (Some(engine_url), true) => {
            let generate_client = HttpGenerateClient::new(&engine_url, tokenizer_without_specials)?;
            hosted_routes(
                OpenAIHttpFrontend::new(renderer, generate_client),
                engine_url,
            )?
        }
        (None, true) => unreachable!("runtime topology was validated above"),
    };
    let listener = tokio::net::TcpListener::bind(config.http_addr)
        .await
        .map_err(|error| format!("binding renderer on {} failed: {error}", config.http_addr))?;
    tracing::info!(address = %config.http_addr, mode, "renderer listening");
    axum::serve(listener, app.into_make_service())
        .with_graceful_shutdown(async {
            if let Err(error) = tokio::signal::ctrl_c().await {
                tracing::error!(%error, "installing renderer shutdown signal failed");
            }
        })
        .await
        .map_err(|error| format!("renderer HTTP server failed: {error}"))
}

#[derive(Clone)]
pub(crate) struct RustServerProxy {
    client: reqwest::Client,
    upstream_url: String,
}

impl RustServerProxy {
    pub(crate) fn new(upstream_url: String) -> Result<Self, String> {
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

    pub(crate) async fn forward(&self, request: Request<Body>) -> Response<Body> {
        let (mut parts, body) = request.into_parts();
        strip_hop_by_hop_headers(&mut parts.headers, true);
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
        strip_hop_by_hop_headers(&mut headers, false);
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

fn strip_hop_by_hop_headers(headers: &mut HeaderMap, request: bool) {
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
    if request {
        headers.remove(header::HOST);
    }
}
