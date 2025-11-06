use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use rand::Rng;
use subtle::ConstantTimeEq;
use tokio::sync::{mpsc, oneshot};
use tower::{Layer, Service};
use tower_http::trace::{MakeSpan, OnRequest, OnResponse, TraceLayer};
use tracing::{debug, error, field::Empty, info, info_span, warn, Span};

pub use crate::core::token_bucket::TokenBucket;
use crate::{metrics::RouterMetrics, server::AppState};

#[derive(Clone)]
pub struct AuthConfig {
    pub api_key: Option<String>,
}

/// Middleware to validate Bearer token against configured API key
/// Only active when router has an API key configured
pub async fn auth_middleware(
    State(auth_config): State<AuthConfig>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    if let Some(expected_key) = &auth_config.api_key {
        // Extract Authorization header
        let auth_header = request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|h| h.to_str().ok());

        match auth_header {
            Some(header_value) if header_value.starts_with("Bearer ") => {
                let token = &header_value[7..]; // Skip "Bearer "
                                                // Use constant-time comparison to prevent timing attacks
                let token_bytes = token.as_bytes();
                let expected_bytes = expected_key.as_bytes();

                // Check if lengths match first (this is not constant-time but necessary)
                if token_bytes.len() != expected_bytes.len() {
                    return Err(StatusCode::UNAUTHORIZED);
                }

                // Constant-time comparison of the actual values
                if token_bytes.ct_eq(expected_bytes).unwrap_u8() != 1 {
                    return Err(StatusCode::UNAUTHORIZED);
                }
            }
            _ => return Err(StatusCode::UNAUTHORIZED),
        }
    }

    Ok(next.run(request).await)
}

/// Generate OpenAI-compatible request ID based on endpoint
fn generate_request_id(path: &str) -> String {
    let prefix = if path.contains("/chat/completions") {
        "chatcmpl-"
    } else if path.contains("/completions") {
        "cmpl-"
    } else if path.contains("/generate") {
        "gnt-"
    } else if path.contains("/responses") {
        "resp-"
    } else {
        "req-"
    };

    // Generate a random string similar to OpenAI's format
    let chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut rng = rand::rng();
    let random_part: String = (0..24)
        .map(|_| {
            let idx = rng.random_range(0..chars.len());
            chars.chars().nth(idx).unwrap()
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

        // Insert request ID into request extensions for other middleware/handlers to use
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

/// Custom span maker that includes request ID
#[derive(Clone, Debug)]
pub struct RequestSpan;

impl<B> MakeSpan<B> for RequestSpan {
    fn make_span(&mut self, request: &Request<B>) -> Span {
        // Don't try to extract request ID here - it won't be available yet
        // The RequestIdLayer runs after TraceLayer creates the span
        info_span!(
            "http_request",
            method = %request.method(),
            uri = %request.uri(),
            version = ?request.version(),
            request_id = Empty,  // Will be set later
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
    fn on_request(&mut self, request: &Request<B>, span: &Span) {
        let _enter = span.enter();

        // Try to get the request ID from extensions
        // This will work if RequestIdLayer has already run
        if let Some(request_id) = request.extensions().get::<RequestId>() {
            span.record("request_id", request_id.0.as_str());
        }

        // Log the request start
        info!(
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
    fn on_response(self, response: &Response<B>, latency: Duration, span: &Span) {
        let status = response.status();

        // Record these in the span for structured logging/observability tools
        span.record("status_code", status.as_u16());
        span.record("latency", format!("{:?}", latency));

        // Log the response completion
        let _enter = span.enter();
        if status.is_server_error() {
            error!(
                target: "sglang_router_rs::response",
                "request failed with server error"
            );
        } else if status.is_client_error() {
            warn!(
                target: "sglang_router_rs::response",
                "request failed with client error"
            );
        } else {
            info!(
                target: "sglang_router_rs::response",
                "finished processing request"
            );
        }
    }
}

/// Create a configured TraceLayer for HTTP logging
/// Note: Actual request/response logging with request IDs is done in RequestIdService
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

/// Request queue entry
pub struct QueuedRequest {
    /// Time when the request was queued
    queued_at: Instant,
    /// Channel to send the permit back when acquired
    permit_tx: oneshot::Sender<Result<(), StatusCode>>,
}

/// Queue metrics for monitoring
#[derive(Debug, Default)]
pub struct QueueMetrics {
    pub total_queued: AtomicU64,
    pub current_queued: AtomicU64,
    pub total_timeout: AtomicU64,
    pub total_rejected: AtomicU64,
}

/// Queue processor that handles queued requests
pub struct QueueProcessor {
    token_bucket: Arc<TokenBucket>,
    queue_rx: mpsc::Receiver<QueuedRequest>,
    queue_timeout: Duration,
}

impl QueueProcessor {
    pub fn new(
        token_bucket: Arc<TokenBucket>,
        queue_rx: mpsc::Receiver<QueuedRequest>,
        queue_timeout: Duration,
    ) -> Self {
        Self {
            token_bucket,
            queue_rx,
            queue_timeout,
        }
    }

    pub async fn run(mut self) {
        info!("Starting concurrency queue processor");

        // Process requests in a single task to reduce overhead
        while let Some(queued) = self.queue_rx.recv().await {
            // Check timeout immediately
            let elapsed = queued.queued_at.elapsed();
            if elapsed >= self.queue_timeout {
                warn!("Request already timed out in queue");
                let _ = queued.permit_tx.send(Err(StatusCode::REQUEST_TIMEOUT));
                continue;
            }

            let remaining_timeout = self.queue_timeout - elapsed;

            // Try to acquire token for this request
            if self.token_bucket.try_acquire(1.0).await.is_ok() {
                // Got token immediately
                debug!("Queue: acquired token immediately for queued request");
                let _ = queued.permit_tx.send(Ok(()));
            } else {
                // Need to wait for token
                let token_bucket = self.token_bucket.clone();

                // Spawn task only when we actually need to wait
                tokio::spawn(async move {
                    if token_bucket
                        .acquire_timeout(1.0, remaining_timeout)
                        .await
                        .is_ok()
                    {
                        debug!("Queue: acquired token after waiting");
                        let _ = queued.permit_tx.send(Ok(()));
                    } else {
                        warn!("Queue: request timed out waiting for token");
                        let _ = queued.permit_tx.send(Err(StatusCode::REQUEST_TIMEOUT));
                    }
                });
            }
        }

        warn!("Concurrency queue processor shutting down");
    }
}

/// State for the concurrency limiter
pub struct ConcurrencyLimiter {
    pub queue_tx: Option<mpsc::Sender<QueuedRequest>>,
}

impl ConcurrencyLimiter {
    /// Create new concurrency limiter with optional queue
    pub fn new(
        token_bucket: Option<Arc<TokenBucket>>,
        queue_size: usize,
        queue_timeout: Duration,
    ) -> (Self, Option<QueueProcessor>) {
        match (token_bucket, queue_size) {
            (None, _) => (Self { queue_tx: None }, None),
            (Some(bucket), size) if size > 0 => {
                let (queue_tx, queue_rx) = mpsc::channel(size);
                let processor = QueueProcessor::new(bucket, queue_rx, queue_timeout);
                (
                    Self {
                        queue_tx: Some(queue_tx),
                    },
                    Some(processor),
                )
            }
            (Some(_), _) => (Self { queue_tx: None }, None),
        }
    }
}

/// Middleware function for concurrency limiting with optional queuing
pub async fn concurrency_limit_middleware(
    State(app_state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let token_bucket = match &app_state.context.rate_limiter {
        Some(bucket) => bucket.clone(),
        None => {
            // Rate limiting disabled, pass through immediately
            return next.run(request).await;
        }
    };

    // Static counter for embeddings queue size
    static EMBEDDINGS_QUEUE_SIZE: AtomicU64 = AtomicU64::new(0);

    // Identify if this is an embeddings request based on path
    let is_embeddings = request.uri().path().contains("/v1/embeddings");

    // Try to acquire token immediately
    if token_bucket.try_acquire(1.0).await.is_ok() {
        debug!("Acquired token immediately");
        let response = next.run(request).await;

        // Return the token to the bucket
        token_bucket.return_tokens(1.0).await;

        response
    } else {
        // No tokens available, try to queue if enabled
        if let Some(queue_tx) = &app_state.concurrency_queue_tx {
            debug!("No tokens available, attempting to queue request");

            // Create a channel for the token response
            let (permit_tx, permit_rx) = oneshot::channel();

            let queued = QueuedRequest {
                queued_at: Instant::now(),
                permit_tx,
            };

            // Try to send to queue
            match queue_tx.try_send(queued) {
                Ok(_) => {
                    // On successful enqueue, update embeddings queue gauge if applicable
                    if is_embeddings {
                        let new_val = EMBEDDINGS_QUEUE_SIZE.fetch_add(1, Ordering::Relaxed) + 1;
                        RouterMetrics::set_embeddings_queue_size(new_val as usize);
                    }

                    // Wait for token from queue processor
                    match permit_rx.await {
                        Ok(Ok(())) => {
                            debug!("Acquired token from queue");
                            // Dequeue for embeddings
                            if is_embeddings {
                                let new_val =
                                    EMBEDDINGS_QUEUE_SIZE.fetch_sub(1, Ordering::Relaxed) - 1;
                                RouterMetrics::set_embeddings_queue_size(new_val as usize);
                            }

                            let response = next.run(request).await;

                            // Return the token to the bucket
                            token_bucket.return_tokens(1.0).await;

                            response
                        }
                        Ok(Err(status)) => {
                            warn!("Queue returned error status: {}", status);
                            // Dequeue for embeddings on error
                            if is_embeddings {
                                let new_val =
                                    EMBEDDINGS_QUEUE_SIZE.fetch_sub(1, Ordering::Relaxed) - 1;
                                RouterMetrics::set_embeddings_queue_size(new_val as usize);
                            }
                            status.into_response()
                        }
                        Err(_) => {
                            error!("Queue response channel closed");
                            // Dequeue for embeddings on channel error
                            if is_embeddings {
                                let new_val =
                                    EMBEDDINGS_QUEUE_SIZE.fetch_sub(1, Ordering::Relaxed) - 1;
                                RouterMetrics::set_embeddings_queue_size(new_val as usize);
                            }
                            StatusCode::INTERNAL_SERVER_ERROR.into_response()
                        }
                    }
                }
                Err(_) => {
                    warn!("Request queue is full, returning 429");
                    StatusCode::TOO_MANY_REQUESTS.into_response()
                }
            }
        } else {
            warn!("No tokens available and queuing is disabled, returning 429");
            StatusCode::TOO_MANY_REQUESTS.into_response()
        }
    }
}
