use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use http_body::Frame;
use rand::Rng;
use serde_json::json;
use subtle::ConstantTimeEq;
use tower::{Layer, Service};
use tower_http::trace::{MakeSpan, OnRequest, OnResponse, TraceLayer};
use tracing::{debug, error, field::Empty, info, info_span, warn, Span};

pub use crate::core::token_bucket::TokenBucket;
use crate::{
    core::{AdmissionError, AdmissionLease, AdmissionSnapshot},
    observability::{
        inflight_tracker::InFlightRequestTracker,
        metrics::{method_to_static_str, metrics_labels, AdmissionDecision, Metrics},
    },
    routers::error::extract_error_code_from_response,
    server::AppState,
    wasm::{
        module::{MiddlewareAttachPoint, WasmModuleAttachPoint},
        spec::{
            apply_modify_action_to_headers, build_wasm_headers_from_axum_headers,
            smg::gateway::middleware_types::{
                Action, Request as WasmRequest, Response as WasmResponse,
            },
        },
        types::WasmComponentInput,
    },
};

/// A stable rejection reason for Router admission control.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionRejection {
    QueueFull,
    QueueTimeout { waited: Duration },
    RateLimited,
    GlobalRateLimited,
    ShuttingDown,
    BackgroundAdmissionUnsupported,
}

impl From<AdmissionError> for AdmissionRejection {
    fn from(error: AdmissionError) -> Self {
        match error {
            AdmissionError::QueueFull => Self::QueueFull,
            AdmissionError::QueueTimeout { waited } => Self::QueueTimeout { waited },
            AdmissionError::ShuttingDown => Self::ShuttingDown,
        }
    }
}

impl AdmissionRejection {
    pub const fn decision(self) -> &'static str {
        match self {
            Self::QueueFull => "queue_full",
            Self::QueueTimeout { .. } => "queue_timeout",
            Self::RateLimited => "rate_limited",
            Self::GlobalRateLimited => "global_rate_limited",
            Self::ShuttingDown => "shutting_down",
            Self::BackgroundAdmissionUnsupported => "background_admission_unsupported",
        }
    }

    const fn status(self) -> StatusCode {
        match self {
            Self::QueueFull | Self::RateLimited | Self::GlobalRateLimited => {
                StatusCode::TOO_MANY_REQUESTS
            }
            Self::QueueTimeout { .. } => StatusCode::REQUEST_TIMEOUT,
            Self::ShuttingDown | Self::BackgroundAdmissionUnsupported => {
                StatusCode::SERVICE_UNAVAILABLE
            }
        }
    }

    const fn message(self) -> &'static str {
        match self {
            Self::QueueFull => "Router admission queue is full",
            Self::QueueTimeout { .. } => "Request exceeded router queue deadline",
            Self::RateLimited => "Router local rate limit exceeded",
            Self::GlobalRateLimited => "Router global rate limit exceeded",
            Self::ShuttingDown => "Router is shutting down",
            Self::BackgroundAdmissionUnsupported => "Background admission is unsupported",
        }
    }

    fn waited_ms(self) -> u64 {
        match self {
            Self::QueueTimeout { waited } => u64::try_from(waited.as_millis()).unwrap_or(u64::MAX),
            _ => 0,
        }
    }
}

/// Runtime values attached to a structured admission rejection log.
#[derive(Clone, Copy, Debug)]
pub struct AdmissionErrorContext<'a> {
    pub request_id: &'a str,
    pub max_concurrent_requests: usize,
    pub queue_size: usize,
    pub queue_timeout_secs: u64,
    pub snapshot: AdmissionSnapshot,
}

/// Builds the stable JSON response and structured log for an admission rejection.
pub fn admission_error_response(
    rejection: AdmissionRejection,
    context: AdmissionErrorContext<'_>,
) -> Response {
    let decision = rejection.decision();
    let waited_ms = rejection.waited_ms();
    warn!(
        request_id = context.request_id,
        decision,
        max_concurrent_requests = context.max_concurrent_requests,
        queue_size = context.queue_size,
        queue_timeout_secs = context.queue_timeout_secs,
        inflight = context.snapshot.inflight,
        queued = context.snapshot.queued,
        waited_ms,
        "request rejected by Router admission control"
    );

    let error = match rejection {
        AdmissionRejection::QueueFull => json!({
            "error": {
                "code": decision,
                "message": rejection.message(),
                "max_concurrent_requests": context.max_concurrent_requests,
                "queue_size": context.queue_size,
            }
        }),
        AdmissionRejection::QueueTimeout { .. } => json!({
            "error": {
                "code": decision,
                "message": rejection.message(),
                "waited_ms": waited_ms,
            }
        }),
        AdmissionRejection::RateLimited
        | AdmissionRejection::GlobalRateLimited
        | AdmissionRejection::ShuttingDown
        | AdmissionRejection::BackgroundAdmissionUnsupported => json!({
            "error": {
                "code": decision,
                "message": rejection.message(),
            }
        }),
    };

    (rejection.status(), Json(error)).into_response()
}

/// A body wrapper that holds an admission lease until the body is dropped.
///
/// Keeping the lease in the response body ensures streaming requests remain admitted for the
/// entire lifetime of the response, including when the client disconnects before completion.
pub struct PermitGuardBody {
    inner: Body,
    _lease: AdmissionLease,
}

impl PermitGuardBody {
    pub fn new(inner: Body, lease: AdmissionLease) -> Self {
        Self {
            inner,
            _lease: lease,
        }
    }
}

impl http_body::Body for PermitGuardBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        Pin::new(&mut this.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

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

/// Alphanumeric characters for request ID generation (as bytes for O(1) indexing)
const REQUEST_ID_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

/// Generate OpenAI-compatible request ID based on endpoint.
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
    // Use byte array indexing (O(1)) instead of chars().nth() (O(n))
    let mut rng = rand::rng();
    let random_part: String = (0..24)
        .map(|_| {
            let idx = rng.random_range(0..REQUEST_ID_CHARS.len());
            REQUEST_ID_CHARS[idx] as char
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
    type Future =
        Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
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
            target: "smg::otel-trace",
            "http_request",
            method = %request.method(),
            uri = %request.uri(),
            version = ?request.version(),
            request_id = Empty,  // Will be set later
            status_code = Empty,
            latency = Empty,
            error = Empty,
            module = "smg"
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

        let method = method_to_static_str(request.method().as_str());
        let path = normalize_path_for_metrics(request.uri().path());
        Metrics::record_http_request(method, &path);

        // Log the request start
        info!(
            target: "smg::request",
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
        let status_code = status.as_u16();

        let error_code = extract_error_code_from_response(response);

        // Layer 1: HTTP metrics
        Metrics::record_http_response(status_code, error_code);

        // Record these in the span for structured logging/observability tools
        span.record("status_code", status_code);
        // Use microseconds as integer to avoid format! string allocation
        span.record("latency", latency.as_micros() as u64);

        // Log the response completion
        let _enter = span.enter();
        if status.is_server_error() {
            error!(
                target: "smg::response",
                "request failed with server error"
            );
        } else if status.is_client_error() {
            warn!(
                target: "smg::response",
                "request failed with client error"
            );
        } else {
            info!(
                target: "smg::response",
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

/// Applies the inference-only traffic gates in their required order:
/// mesh/global QPS, local QPS, then lifecycle-owned admission.
///
/// Authentication and WASM middleware are composed outside this guard by
/// `server::build_app`, so either can short-circuit without consuming limiter
/// state.
pub async fn inference_guard_middleware(
    State(app_state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let request_id = request
        .extensions()
        .get::<RequestId>()
        .map(|request_id| request_id.0.as_str())
        .unwrap_or("unknown")
        .to_string();

    let error_context = |snapshot: AdmissionSnapshot| AdmissionErrorContext {
        request_id: &request_id,
        max_concurrent_requests: app_state
            .context
            .router_config
            .max_concurrent_requests
            .max(0) as usize,
        queue_size: app_state.context.router_config.queue_size,
        queue_timeout_secs: app_state.context.router_config.queue_timeout_secs,
        snapshot,
    };
    let admission_snapshot = || {
        app_state
            .context
            .admission_limiter
            .as_ref()
            .map(|limiter| limiter.snapshot())
            .unwrap_or(AdmissionSnapshot {
                inflight: 0,
                queued: 0,
            })
    };

    if let Some(sync_manager) = &app_state.mesh_sync_manager {
        let (is_exceeded, current_count, limit) = sync_manager.check_global_rate_limit();
        if is_exceeded {
            debug!(
                "Global rate limit exceeded: {}/{} req/s",
                current_count, limit
            );
            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
            Metrics::record_admission_decision(AdmissionDecision::GlobalRateLimited);
            return admission_error_response(
                AdmissionRejection::GlobalRateLimited,
                error_context(admission_snapshot()),
            );
        }
    }

    if let Some(rate_limiter) = &app_state.context.rate_limiter {
        if rate_limiter.try_acquire(1.0).await.is_err() {
            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
            Metrics::record_admission_decision(AdmissionDecision::RateLimited);
            return admission_error_response(
                AdmissionRejection::RateLimited,
                error_context(admission_snapshot()),
            );
        }
        Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_ALLOWED);
    }

    let lease = if let Some(admission_limiter) = &app_state.context.admission_limiter {
        match admission_limiter.acquire().await {
            Ok(lease) => Some(lease),
            Err(error) => {
                return admission_error_response(
                    AdmissionRejection::from(error),
                    error_context(admission_limiter.snapshot()),
                );
            }
        }
    } else {
        None
    };

    let response = next.run(request).await;
    match lease {
        Some(lease) => {
            let (parts, body) = response.into_parts();
            Response::from_parts(parts, Body::new(PermitGuardBody::new(body, lease)))
        }
        None => response,
    }
}

// ============================================================================
// HTTP Metrics Layer (Layer 1: SMG metrics)
// ============================================================================

/// Global counter for active HTTP connections (handlers currently executing)
static ACTIVE_HTTP_CONNECTIONS: AtomicU64 = AtomicU64::new(0);

/// Tower Layer for HTTP metrics collection (SMG Layer 1 metrics)
#[derive(Clone)]
pub struct HttpMetricsLayer {
    tracker: Arc<InFlightRequestTracker>,
}

impl HttpMetricsLayer {
    pub fn new(tracker: Arc<InFlightRequestTracker>) -> Self {
        Self { tracker }
    }
}

impl<S> Layer<S> for HttpMetricsLayer {
    type Service = HttpMetricsMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        HttpMetricsMiddleware {
            inner,
            in_flight_request_tracker: self.tracker.clone(),
        }
    }
}

/// Tower Service for HTTP metrics collection
#[derive(Clone)]
pub struct HttpMetricsMiddleware<S> {
    inner: S,
    in_flight_request_tracker: Arc<InFlightRequestTracker>,
}

impl<S> Service<Request> for HttpMetricsMiddleware<S>
where
    S: Service<Request, Response = Response> + Send + Clone + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future =
        Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request) -> Self::Future {
        // Convert method to static string to avoid allocation
        let method = method_to_static_str(req.method().as_str());
        let path = normalize_path_for_metrics(req.uri().path());
        let start = Instant::now();

        let mut inner = self.inner.clone();
        let in_flight_request_tracker = self.in_flight_request_tracker.clone();

        Box::pin(async move {
            // Increment inside async block - ensures no leak if future is dropped before polling
            let active = ACTIVE_HTTP_CONNECTIONS.fetch_add(1, Ordering::Relaxed) + 1;
            Metrics::set_http_connections_active(active as usize);

            let guard = in_flight_request_tracker.track();

            // Capture result before decrementing to ensure decrement happens on error too
            let result = inner.call(req).await;

            drop(guard);

            // Always decrement, regardless of success or failure
            let active = ACTIVE_HTTP_CONNECTIONS.fetch_sub(1, Ordering::Relaxed) - 1;
            Metrics::set_http_connections_active(active as usize);

            let response = result?;

            let duration = start.elapsed();
            Metrics::record_http_duration(method, &path, duration);

            Ok(response)
        })
    }
}

/// Normalize path for metrics to avoid high cardinality.
/// Replaces dynamic segments (IDs, UUIDs) with `{id}` placeholder.
/// Only allocates when normalization is needed; uses single-pass with byte offsets.
fn normalize_path_for_metrics(path: &str) -> String {
    let bytes = path.as_bytes();
    let mut segment_start = 0;
    let mut segment_idx = 0;
    let mut result: Option<String> = None;

    for (pos, &b) in bytes.iter().enumerate() {
        if b == b'/' || pos == bytes.len() - 1 {
            // Determine segment end (include last char if not a slash)
            let segment_end = if b == b'/' { pos } else { pos + 1 };
            let segment = &path[segment_start..segment_end];

            // Check segments after index 2 for dynamic IDs
            if segment_idx > 2 && !segment.is_empty() && is_dynamic_id(segment) {
                // Initialize result with everything before this segment
                let result = result.get_or_insert_with(|| {
                    let mut s = String::with_capacity(path.len());
                    s.push_str(&path[..segment_start]);
                    s
                });
                result.push_str("{id}");
            } else if let Some(ref mut r) = result {
                // Already normalizing, append this segment as-is
                r.push_str(segment);
            }

            // Add slash after segment (except at end)
            if b == b'/' {
                if let Some(ref mut r) = result {
                    r.push('/');
                }
                segment_start = pos + 1;
                segment_idx += 1;
            }
        }
    }

    result.unwrap_or_else(|| path.to_owned())
}

/// Check if segment looks like a dynamic ID (prefixed ID, UUID, or numeric).
#[inline]
fn is_dynamic_id(s: &str) -> bool {
    // Prefixed IDs: resp_xxx, chatcmpl_xxx (len > 10 with underscore)
    if s.len() > 10 && s.contains('_') {
        return true;
    }
    // UUIDs: 32+ hex chars with dashes
    if s.len() >= 32 && s.bytes().all(|b| b.is_ascii_hexdigit() || b == b'-') {
        return true;
    }
    // Numeric IDs
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

pub async fn wasm_middleware(
    State(app_state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    // Check if WASM is enabled
    if !app_state.context.router_config.enable_wasm {
        return Ok(next.run(request).await);
    }

    // Get WASM manager
    let wasm_manager = match &app_state.context.wasm_manager {
        Some(manager) => manager,
        None => {
            return Ok(next.run(request).await);
        }
    };

    // Get request ID from extensions or generate one
    let request_id = request
        .extensions()
        .get::<RequestId>()
        .map(|r| r.0.clone())
        .unwrap_or_else(|| generate_request_id(request.uri().path()));

    // ===== OnRequest Phase =====
    let on_request_attach_point =
        WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnRequest);

    let modules_on_request =
        match wasm_manager.get_modules_by_attach_point(on_request_attach_point.clone()) {
            Ok(modules) => modules,
            Err(e) => {
                error!("Failed to get WASM modules for OnRequest: {}", e);
                return Ok(next.run(request).await);
            }
        };

    let response = if modules_on_request.is_empty() {
        next.run(request).await
    } else {
        // Extract request body once before processing modules
        let method = request.method().clone();
        let uri = request.uri().clone();
        let mut headers = request.headers().clone();
        let max_body_size = wasm_manager.get_max_body_size();
        let body_bytes = match axum::body::to_bytes(request.into_body(), max_body_size).await {
            Ok(bytes) => bytes.to_vec(),
            Err(e) => {
                error!("Failed to read request body: {}", e);
                // Create a minimal request with empty body for error recovery
                let error_request = Request::builder()
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap_or_else(|_| Request::new(Body::empty()));
                return Ok(next.run(error_request).await);
            }
        };

        // Process each OnRequest module
        let mut modified_body = body_bytes;

        // Pre-compute strings once before the loop to avoid repeated allocations
        let method_str = method.to_string();
        let path_str = uri.path().to_string();
        let query_str = uri.query().unwrap_or("").to_string();

        for module in modules_on_request {
            // Build WebAssembly request from collected data
            let wasm_headers = build_wasm_headers_from_axum_headers(&headers);
            let wasm_request = WasmRequest {
                method: method_str.clone(),
                path: path_str.clone(),
                query: query_str.clone(),
                headers: wasm_headers,
                body: modified_body.clone(),
                request_id: request_id.clone(),
                now_epoch_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_else(|_| Duration::from_millis(0))
                    .as_millis() as u64,
            };

            // Execute WASM component
            let action = match wasm_manager
                .execute_module_for_attach_point(
                    &module,
                    on_request_attach_point.clone(),
                    WasmComponentInput::MiddlewareRequest(wasm_request),
                )
                .await
            {
                Some(action) => action,
                None => continue, // Continue to next module on error
            };

            // Process action
            match action {
                Action::Continue => {}
                Action::Reject(status) => {
                    return Err(StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_REQUEST));
                }
                Action::Modify(modify) => {
                    // Apply modifications to headers and body
                    apply_modify_action_to_headers(&mut headers, &modify);
                    if let Some(body_bytes) = modify.body_replace {
                        modified_body = body_bytes;
                    }
                }
            }
        }

        // Reconstruct request with modifications
        let mut final_request = Request::builder()
            .method(method)
            .uri(uri)
            .body(Body::from(modified_body))
            .unwrap_or_else(|_| Request::new(Body::empty()));
        *final_request.headers_mut() = headers;

        // Continue with request processing
        next.run(final_request).await
    };

    // ===== OnResponse Phase =====
    let on_response_attach_point =
        WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnResponse);

    let modules_on_response =
        match wasm_manager.get_modules_by_attach_point(on_response_attach_point.clone()) {
            Ok(modules) => modules,
            Err(e) => {
                error!("Failed to get WASM modules for OnResponse: {}", e);
                return Ok(response);
            }
        };
    if modules_on_response.is_empty() {
        return Ok(response);
    }
    // Extract response data once before processing modules
    let mut status = response.status();
    let mut headers = response.headers().clone();
    let max_body_size = wasm_manager.get_max_body_size();
    let mut body_bytes = match axum::body::to_bytes(response.into_body(), max_body_size).await {
        Ok(bytes) => bytes.to_vec(),
        Err(e) => {
            error!("Failed to read response body: {}", e);
            // Create a minimal response with empty body for error recovery
            let error_response = Response::builder()
                .status(status)
                .body(Body::empty())
                .unwrap_or_else(|_| Response::new(Body::empty()));
            return Ok(error_response);
        }
    };

    // Process each OnResponse module
    for module in modules_on_response {
        // Build WebAssembly response from collected data
        let wasm_headers = build_wasm_headers_from_axum_headers(&headers);
        let wasm_response = WasmResponse {
            status: status.as_u16(),
            headers: wasm_headers,
            body: body_bytes.clone(),
        };

        // Execute WASM component
        let action = match wasm_manager
            .execute_module_for_attach_point(
                &module,
                on_response_attach_point.clone(),
                WasmComponentInput::MiddlewareResponse(wasm_response),
            )
            .await
        {
            Some(action) => action,
            None => continue, // Continue to next module on error
        };

        // Process action - apply modifications incrementally
        match action {
            Action::Continue => {
                // Continue to next module
            }
            Action::Reject(status_code) => {
                // Override response status
                status = StatusCode::from_u16(status_code).unwrap_or(StatusCode::BAD_REQUEST);
                // Return immediately with current state
                let final_response = Response::builder()
                    .status(status)
                    .body(Body::from(body_bytes))
                    .unwrap_or_else(|_| Response::new(Body::empty()));
                let mut final_response = final_response;
                *final_response.headers_mut() = headers;
                return Ok(final_response);
            }
            Action::Modify(modify) => {
                // Apply status modification
                if let Some(new_status) = modify.status {
                    status = StatusCode::from_u16(new_status).unwrap_or(status);
                }
                // Apply headers modifications
                apply_modify_action_to_headers(&mut headers, &modify);
                // Apply body_replace
                if let Some(new_body) = modify.body_replace {
                    body_bytes = new_body;
                }
            }
        }
    }

    // Reconstruct final response with all modifications
    let final_response = Response::builder()
        .status(status)
        .body(Body::from(body_bytes))
        .unwrap_or_else(|_| Response::new(Body::empty()));
    let mut final_response = final_response;
    *final_response.headers_mut() = headers;
    Ok(final_response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_path_no_ids() {
        // Common API paths should pass through unchanged
        assert_eq!(
            normalize_path_for_metrics("/v1/chat/completions"),
            "/v1/chat/completions"
        );
        assert_eq!(
            normalize_path_for_metrics("/v1/completions"),
            "/v1/completions"
        );
        assert_eq!(normalize_path_for_metrics("/v1/models"), "/v1/models");
        assert_eq!(normalize_path_for_metrics("/health"), "/health");
    }

    #[test]
    fn test_normalize_path_with_prefixed_id() {
        // Prefixed IDs (resp_xxx, chatcmpl_xxx) should be normalized
        assert_eq!(
            normalize_path_for_metrics("/v1/responses/resp_abc123def456"),
            "/v1/responses/{id}"
        );
        assert_eq!(
            normalize_path_for_metrics("/v1/chat/completions/chatcmpl_abc123xyz"),
            "/v1/chat/completions/{id}"
        );
    }

    #[test]
    fn test_normalize_path_with_uuid() {
        assert_eq!(
            normalize_path_for_metrics("/v1/responses/550e8400-e29b-41d4-a716-446655440000"),
            "/v1/responses/{id}"
        );
    }

    #[test]
    fn test_normalize_path_with_numeric_id() {
        assert_eq!(
            normalize_path_for_metrics("/v1/workers/12345"),
            "/v1/workers/{id}"
        );
    }

    #[test]
    fn test_is_dynamic_id() {
        // Prefixed IDs
        assert!(is_dynamic_id("resp_abc123def"));
        assert!(is_dynamic_id("chatcmpl_xyz789"));
        assert!(!is_dynamic_id("short_id")); // Too short

        // UUIDs
        assert!(is_dynamic_id("550e8400-e29b-41d4-a716-446655440000"));
        assert!(is_dynamic_id("550e8400e29b41d4a716446655440000")); // No dashes

        // Numeric
        assert!(is_dynamic_id("12345"));
        assert!(!is_dynamic_id("")); // Empty

        // Regular words
        assert!(!is_dynamic_id("completions"));
        assert!(!is_dynamic_id("chat"));
    }
}
