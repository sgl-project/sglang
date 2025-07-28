use axum::{extract::Request, http::HeaderValue, response::Response};
use std::sync::Arc;
use std::time::Instant;
use tower::{Layer, Service};
use tower_http::trace::{MakeSpan, OnRequest, OnResponse, TraceLayer};
use tracing::{field::Empty, info_span, Span};

/// Generate OpenAI-compatible request ID based on endpoint
fn generate_request_id(path: &str) -> String {
    let prefix = if path.contains("/chat/completions") {
        "chatcmpl-"
    } else if path.contains("/completions") {
        "cmpl-"
    } else if path.contains("/generate") {
        "gnt-"
    } else {
        "req-"
    };

    // Generate a random string similar to OpenAI's format
    let random_part: String = (0..24)
        .map(|_| {
            let chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
            chars
                .chars()
                .nth(rand::random::<usize>() % chars.len())
                .unwrap()
        })
        .collect();

    format!("{}{}", prefix, random_part)
}

/// Extension type for storing request ID
#[derive(Clone, Debug)]
pub struct RequestId(pub String);

/// Tower Layer for request ID middleware
#[derive(Clone)]
pub struct RequestIdLayer {
    headers: Arc<Vec<String>>,
}

impl RequestIdLayer {
    pub fn new(headers: Vec<String>) -> Self {
        Self {
            headers: Arc::new(headers),
        }
    }
}

impl<S> Layer<S> for RequestIdLayer {
    type Service = RequestIdMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestIdMiddleware {
            inner,
            headers: self.headers.clone(),
        }
    }
}

/// Tower Service for request ID middleware
#[derive(Clone)]
pub struct RequestIdMiddleware<S> {
    inner: S,
    headers: Arc<Vec<String>>,
}

impl<S> Service<Request> for RequestIdMiddleware<S>
where
    S: Service<Request, Response = Response> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: Request) -> Self::Future {
        let headers = self.headers.clone();

        // Extract request ID from headers or generate new one
        let mut request_id = None;

        for header_name in headers.iter() {
            if let Some(header_value) = req.headers().get(header_name) {
                if let Ok(value) = header_value.to_str() {
                    request_id = Some(value.to_string());
                    break;
                }
            }
        }

        let request_id = request_id.unwrap_or_else(|| generate_request_id(req.uri().path()));

        // Insert request ID into request extensions
        req.extensions_mut().insert(RequestId(request_id.clone()));

        // Call the inner service
        let future = self.inner.call(req);

        Box::pin(async move {
            let mut response = future.await?;

            // Add request ID to response headers
            response.headers_mut().insert(
                "x-request-id",
                HeaderValue::from_str(&request_id)
                    .unwrap_or_else(|_| HeaderValue::from_static("invalid-request-id")),
            );

            Ok(response)
        })
    }
}

// ============= Logging Middleware =============

/// Custom span maker that includes request ID
#[derive(Clone, Debug)]
pub struct RequestSpan;

impl<B> MakeSpan<B> for RequestSpan {
    fn make_span(&mut self, request: &Request<B>) -> Span {
        // Extract request ID from extensions
        let request_id = request
            .extensions()
            .get::<RequestId>()
            .map(|id| id.0.as_str())
            .unwrap_or("unknown");

        info_span!(
            "http_request",
            method = %request.method(),
            uri = %request.uri(),
            version = ?request.version(),
            request_id = %request_id,
            status_code = Empty,
            latency = Empty,
            error = Empty,
        )
    }
}

/// Custom on_request handler
#[derive(Clone, Debug)]
pub struct RequestLogger;

impl<B> OnRequest<B> for RequestLogger {
    fn on_request(&mut self, _request: &Request<B>, span: &Span) {
        let _enter = span.enter();
        tracing::info!(
            target: "sglang_router_rs::request",
            "started processing request"
        );
    }
}

/// Custom on_response handler
#[derive(Clone, Debug)]
pub struct ResponseLogger {
    _start_time: Instant,
}

impl Default for ResponseLogger {
    fn default() -> Self {
        Self {
            _start_time: Instant::now(),
        }
    }
}

impl<B> OnResponse<B> for ResponseLogger {
    fn on_response(self, response: &Response<B>, latency: std::time::Duration, span: &Span) {
        let status = response.status();

        span.record("status_code", status.as_u16());
        span.record("latency", format!("{:?}", latency));

        let _enter = span.enter();

        if status.is_server_error() {
            tracing::error!(
                target: "sglang_router_rs::response",
                status = %status,
                latency = ?latency,
                "request failed with server error"
            );
        } else if status.is_client_error() {
            tracing::warn!(
                target: "sglang_router_rs::response",
                status = %status,
                latency = ?latency,
                "request failed with client error"
            );
        } else {
            tracing::info!(
                target: "sglang_router_rs::response",
                status = %status,
                latency = ?latency,
                "finished processing request"
            );
        }
    }
}

/// Create a configured TraceLayer for HTTP logging with request ID support
pub fn create_logging_layer() -> TraceLayer<
    tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>,
    RequestSpan,
    RequestLogger,
    ResponseLogger,
> {
    TraceLayer::new_for_http()
        .make_span_with(RequestSpan)
        .on_request(RequestLogger)
        .on_response(ResponseLogger::default())
}

/// Structured logging data for requests
#[derive(Debug, serde::Serialize)]
pub struct RequestLogEntry {
    pub timestamp: String,
    pub request_id: String,
    pub method: String,
    pub uri: String,
    pub status: u16,
    pub latency_ms: u64,
    pub user_agent: Option<String>,
    pub remote_addr: Option<String>,
    pub error: Option<String>,
}

/// Log a request with structured data
pub fn log_request(entry: RequestLogEntry) {
    if entry.status >= 500 {
        tracing::error!(
            target: "sglang_router_rs::http",
            request_id = %entry.request_id,
            method = %entry.method,
            uri = %entry.uri,
            status = entry.status,
            latency_ms = entry.latency_ms,
            user_agent = ?entry.user_agent,
            remote_addr = ?entry.remote_addr,
            error = ?entry.error,
            "HTTP request failed"
        );
    } else if entry.status >= 400 {
        tracing::warn!(
            target: "sglang_router_rs::http",
            request_id = %entry.request_id,
            method = %entry.method,
            uri = %entry.uri,
            status = entry.status,
            latency_ms = entry.latency_ms,
            user_agent = ?entry.user_agent,
            remote_addr = ?entry.remote_addr,
            "HTTP request client error"
        );
    } else {
        tracing::info!(
            target: "sglang_router_rs::http",
            request_id = %entry.request_id,
            method = %entry.method,
            uri = %entry.uri,
            status = entry.status,
            latency_ms = entry.latency_ms,
            user_agent = ?entry.user_agent,
            remote_addr = ?entry.remote_addr,
            "HTTP request completed"
        );
    }
}
