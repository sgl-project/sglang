use std::{
    borrow::Cow,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use dashmap::DashMap;
use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder};
use once_cell::sync::Lazy;

// =============================================================================
// STRING INTERNING
// =============================================================================
//
// Dynamic strings (model_id, worker URLs, paths) are interned to avoid repeated
// heap allocations. The interner uses Arc<str> which is cheap to clone and
// allows the metrics crate to store references without repeated allocations.
//
// Performance characteristics:
// - First occurrence: One allocation + DashMap insert
// - Subsequent occurrences: DashMap lookup + Arc::clone (very cheap)
// - Memory: Strings are never freed (acceptable for bounded label cardinality)

/// Global string interner for metric labels.
/// Uses DashMap for lock-free concurrent access.
static STRING_INTERNER: Lazy<DashMap<String, Arc<str>>> = Lazy::new(DashMap::new);

/// Intern a string, returning a cheaply-cloneable Arc<str>.
///
/// This function is designed for high-throughput scenarios where the same
/// strings (model IDs, worker URLs) appear repeatedly. The first call allocates,
/// subsequent calls just clone the Arc (very cheap - just a ref count increment).
pub(crate) fn intern_string(s: &str) -> Arc<str> {
    // Fast path: check if already interned
    if let Some(entry) = STRING_INTERNER.get(s) {
        return Arc::clone(entry.value());
    }

    // Slow path: intern the string
    // Use entry API to avoid TOCTOU race
    STRING_INTERNER
        .entry(s.to_string())
        .or_insert_with(|| Arc::from(s))
        .clone()
}

#[allow(dead_code)]
pub(crate) fn interner_size() -> usize {
    STRING_INTERNER.len()
}

// =============================================================================
// STATIC STRING CONSTANTS
// =============================================================================

/// Static string constants for boolean labels to avoid allocations.
pub const STREAMING_TRUE: &str = "true";
pub const STREAMING_FALSE: &str = "false";

pub const fn bool_to_static_str(b: bool) -> &'static str {
    if b {
        STREAMING_TRUE
    } else {
        STREAMING_FALSE
    }
}

/// Static lookup table for common HTTP status codes to avoid allocations.
/// Returns a static string for known codes, or None for unknown codes.
#[inline]
pub fn status_code_to_static_str(code: u16) -> Option<&'static str> {
    // Using a match with explicit arms is faster than a lookup table for this size
    match code {
        200 => Some("200"),
        201 => Some("201"),
        204 => Some("204"),
        400 => Some("400"),
        401 => Some("401"),
        403 => Some("403"),
        404 => Some("404"),
        408 => Some("408"),
        422 => Some("422"),
        429 => Some("429"),
        500 => Some("500"),
        502 => Some("502"),
        503 => Some("503"),
        504 => Some("504"),
        _ => None,
    }
}

/// Static HTTP method strings to avoid allocations on every request.
pub(crate) mod http_methods {
    pub const GET: &str = "GET";
    pub const POST: &str = "POST";
    pub const PUT: &str = "PUT";
    pub const DELETE: &str = "DELETE";
    pub const PATCH: &str = "PATCH";
    pub const HEAD: &str = "HEAD";
    pub const OPTIONS: &str = "OPTIONS";
}

/// Convert HTTP method to static string. Returns the method as-is for unknown methods.
#[inline]
pub fn method_to_static_str(method: &str) -> &'static str {
    match method {
        "GET" => http_methods::GET,
        "POST" => http_methods::POST,
        "PUT" => http_methods::PUT,
        "DELETE" => http_methods::DELETE,
        "PATCH" => http_methods::PATCH,
        "HEAD" => http_methods::HEAD,
        "OPTIONS" => http_methods::OPTIONS,
        // For unknown methods, we return a static "OTHER" to avoid allocation
        // This is acceptable since unknown methods are rare in practice
        _ => "OTHER",
    }
}

/// Get status code as Cow - static for common codes, allocated for rare ones.
#[inline]
pub fn status_code_to_cow(code: u16) -> Cow<'static, str> {
    match status_code_to_static_str(code) {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(code.to_string()),
    }
}

#[derive(Debug, Clone)]
pub struct PrometheusConfig {
    pub port: u16,
    pub host: String,
    pub duration_buckets: Option<Vec<f64>>,
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            port: 29000,
            host: "0.0.0.0".to_string(),
            duration_buckets: None,
        }
    }
}

pub(crate) fn init_metrics() {
    // Layer 1: HTTP metrics
    describe_counter!(
        "smg_http_requests_total",
        "Total HTTP requests by method and path"
    );
    describe_histogram!(
        "smg_http_request_duration_seconds",
        "HTTP request duration by method and path"
    );
    describe_gauge!(
        "smg_http_inflight_request_age_count",
        "In-flight HTTP requests per age bucket (gt < age <= le, non-cumulative)"
    );
    describe_counter!(
        "smg_http_responses_total",
        "Total HTTP responses by status_code and error_code"
    );
    describe_gauge!(
        "smg_http_connections_active",
        "Currently active HTTP connections"
    );
    describe_counter!(
        "smg_http_rate_limit_total",
        "Rate limiting decisions by result (allowed/rejected)"
    );

    // Layer 2: Router metrics
    describe_counter!(
        "smg_router_requests_total",
        "Total routed requests by router_type, backend_type, connection_mode, model, endpoint, streaming"
    );
    describe_histogram!(
        "smg_router_request_duration_seconds",
        "Router request duration by router_type, backend_type, connection_mode, model, endpoint"
    );
    describe_counter!(
        "smg_router_request_errors_total",
        "Router errors by router_type, backend_type, connection_mode, model, endpoint, error_type"
    );
    describe_histogram!(
        "smg_router_stage_duration_seconds",
        "Pipeline stage duration by router_type and stage (gRPC only)"
    );
    describe_counter!(
        "smg_router_upstream_responses_total",
        "Upstream backend HTTP responses by router_type, status_code, error_code"
    );

    // Layer 2: Router inference metrics (gRPC only)
    describe_histogram!(
        "smg_router_ttft_seconds",
        "Time to first token by router_type, backend_type, model, endpoint (gRPC only)"
    );
    describe_histogram!(
        "smg_router_tpot_seconds",
        "Time per output token by router_type, backend_type, model, endpoint (gRPC only)"
    );
    describe_counter!(
        "smg_router_tokens_total",
        "Total tokens processed by router_type, backend_type, model, endpoint, token_type (gRPC only)"
    );
    describe_histogram!(
        "smg_router_generation_duration_seconds",
        "Total generation time by router_type, backend_type, model, endpoint (gRPC only)"
    );

    // Layer 3: Worker metrics
    describe_gauge!(
        "smg_worker_pool_size",
        "Current worker pool size by worker_type, connection_mode, model"
    );
    describe_gauge!(
        "smg_worker_connections_active",
        "Active connections to workers by worker_type, connection_mode"
    );
    describe_gauge!(
        "smg_worker_requests_active",
        "Currently running requests per worker"
    );
    describe_gauge!(
        "smg_worker_health",
        "Worker health status (1=healthy, 0=unhealthy)"
    );
    describe_counter!(
        "smg_worker_health_checks_total",
        "Health check results by worker_type and result"
    );
    describe_counter!(
        "smg_worker_selection_total",
        "Worker selection events by worker_type, connection_mode, model, policy"
    );
    describe_counter!(
        "smg_worker_errors_total",
        "Worker-level errors by worker_type, connection_mode, error_type"
    );
    describe_gauge!(
        "smg_manual_policy_cache_entries",
        "Number of routing entries in manual policy cache"
    );

    // Layer 3: Worker resilience metrics (circuit breaker)
    describe_gauge!(
        "smg_worker_cb_state",
        "Circuit breaker state per worker (0=closed, 1=open, 2=half_open)"
    );
    describe_counter!(
        "smg_worker_cb_transitions_total",
        "Circuit breaker state transitions by worker, from, to"
    );
    describe_counter!(
        "smg_worker_cb_outcomes_total",
        "Circuit breaker outcomes by worker and outcome (success/failure)"
    );
    describe_gauge!(
        "smg_worker_cb_consecutive_failures",
        "Current consecutive failure count per worker"
    );
    describe_gauge!(
        "smg_worker_cb_consecutive_successes",
        "Current consecutive success count per worker"
    );

    // Layer 3: Worker resilience metrics (retry)
    describe_counter!(
        "smg_worker_retries_total",
        "Total retry attempts by worker_type and endpoint"
    );
    describe_counter!(
        "smg_worker_retries_exhausted_total",
        "Requests that exhausted all retries by worker_type and endpoint"
    );
    describe_histogram!(
        "smg_worker_retry_backoff_seconds",
        "Retry backoff duration by attempt number"
    );

    // Layer 4: Discovery metrics
    describe_counter!(
        "smg_discovery_registrations_total",
        "Worker registration attempts by source and result"
    );
    describe_counter!(
        "smg_discovery_deregistrations_total",
        "Worker deregistration events by source and reason"
    );
    describe_histogram!(
        "smg_discovery_sync_duration_seconds",
        "Discovery sync duration by source"
    );
    describe_gauge!(
        "smg_discovery_workers_discovered",
        "Workers known via discovery by source"
    );

    // Layer 5: MCP metrics
    describe_counter!(
        "smg_mcp_tool_calls_total",
        "Total MCP tool invocations by model, tool_name, result"
    );
    describe_histogram!(
        "smg_mcp_tool_duration_seconds",
        "MCP tool execution duration by model, tool_name"
    );
    describe_gauge!("smg_mcp_servers_active", "Active MCP server connections");
    describe_counter!(
        "smg_mcp_tool_iterations_total",
        "Tool loop iterations in Responses API by model"
    );

    // Layer 6: Database metrics
    describe_counter!(
        "smg_db_operations_total",
        "Total database operations by storage_type, operation, result"
    );
    describe_histogram!(
        "smg_db_operation_duration_seconds",
        "Database operation duration by storage_type, operation"
    );
    describe_gauge!(
        "smg_db_connections_active",
        "Active database connections by storage_type"
    );
    describe_counter!("smg_db_items_stored", "Total items stored by storage_type");
}

pub fn start_prometheus(config: PrometheusConfig) {
    init_metrics();

    let duration_matcher = Matcher::Suffix(String::from("duration_seconds"));
    let duration_bucket: Vec<f64> = config.duration_buckets.unwrap_or_else(|| {
        vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
            60.0, 90.0, 120.0, 180.0, 240.0,
        ]
    });

    let ip_addr: IpAddr = config
        .host
        .parse()
        .unwrap_or(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));
    let socket_addr = SocketAddr::new(ip_addr, config.port);

    PrometheusBuilder::new()
        .with_http_listener(socket_addr)
        .upkeep_timeout(Duration::from_secs(5 * 60))
        .set_buckets_for_metric(duration_matcher, &duration_bucket)
        .expect("failed to set duration bucket")
        .install()
        .expect("failed to install Prometheus metrics exporter");
}

/// Label constants for consistent metric labeling
pub mod metrics_labels {
    // Router types
    pub const ROUTER_OPENAI: &str = "openai";
    pub const ROUTER_HTTP: &str = "http";
    pub const ROUTER_GRPC: &str = "grpc";

    // Backend types
    pub const BACKEND_REGULAR: &str = "regular";
    pub const BACKEND_PD: &str = "pd";
    pub const BACKEND_EXTERNAL: &str = "external";
    pub const BACKEND_HARMONY: &str = "harmony";

    // Connection modes
    pub const CONNECTION_HTTP: &str = "http";
    pub const CONNECTION_GRPC: &str = "grpc";

    // Endpoints
    pub const ENDPOINT_CHAT: &str = "chat";
    pub const ENDPOINT_GENERATE: &str = "generate";
    pub const ENDPOINT_RESPONSES: &str = "responses";
    pub const ENDPOINT_COMPLETIONS: &str = "completions";
    pub const ENDPOINT_RERANK: &str = "rerank";
    pub const ENDPOINT_EMBEDDINGS: &str = "embeddings";
    pub const ENDPOINT_CLASSIFY: &str = "classify";

    // Worker types
    pub const WORKER_REGULAR: &str = "regular";
    pub const WORKER_PREFILL: &str = "prefill";
    pub const WORKER_DECODE: &str = "decode";
    pub const WORKER_HTTP: &str = "http";
    pub const WORKER_GRPC: &str = "grpc";

    // Token types
    pub const TOKEN_INPUT: &str = "input";
    pub const TOKEN_OUTPUT: &str = "output";

    // Storage types
    pub const STORAGE_RESPONSE: &str = "response";
    pub const STORAGE_CONVERSATION: &str = "conversation";
    pub const STORAGE_CONVERSATION_ITEM: &str = "conversation_item";

    // Database operations
    pub const DB_OP_GET: &str = "get";
    pub const DB_OP_PUT: &str = "put";
    pub const DB_OP_DELETE: &str = "delete";
    pub const DB_OP_LIST: &str = "list";

    // Result types
    pub const RESULT_SUCCESS: &str = "success";
    pub const RESULT_ERROR: &str = "error";
    pub const RESULT_TIMEOUT: &str = "timeout";
    pub const RESULT_NOT_FOUND: &str = "not_found";

    // Discovery sources
    pub const DISCOVERY_STATIC: &str = "static";
    pub const DISCOVERY_KUBERNETES: &str = "kubernetes";
    pub const DISCOVERY_CONSUL: &str = "consul";
    pub const DISCOVERY_MANUAL: &str = "manual";

    // Discovery registration results
    pub const REGISTRATION_SUCCESS: &str = "success";
    pub const REGISTRATION_FAILED: &str = "failed";
    pub const REGISTRATION_DUPLICATE: &str = "duplicate";
    pub const DEREGISTRATION_POD_DELETED: &str = "pod_deleted";

    // Rate limit results
    pub const RATE_LIMIT_ALLOWED: &str = "allowed";
    pub const RATE_LIMIT_REJECTED: &str = "rejected";

    // Circuit breaker states
    pub const CB_CLOSED: &str = "closed";
    pub const CB_OPEN: &str = "open";
    pub const CB_HALF_OPEN: &str = "half_open";

    // Circuit breaker outcomes
    pub const CB_SUCCESS: &str = "success";
    pub const CB_FAILURE: &str = "failure";

    // Router error types
    pub const ERROR_NO_WORKERS: &str = "no_workers";
    pub const ERROR_TIMEOUT: &str = "timeout";
    pub const ERROR_BACKEND: &str = "backend_error";
    pub const ERROR_VALIDATION: &str = "validation_error";
    pub const ERROR_INTERNAL: &str = "internal_error";
}

/// SMG Metrics helper struct for the new layered metrics architecture.
///
/// Design principles for low overhead:
/// - Dynamic labels use string interning (single allocation per unique value)
/// - Static labels use the metrics crate's internal caching
pub struct Metrics;

/// Parameters for recording streaming metrics.
pub struct StreamingMetricsParams<'a> {
    /// Router type label (e.g., "grpc", "http")
    pub router_type: &'static str,
    /// Backend type label (e.g., "regular", "pd")
    pub backend_type: &'static str,
    /// Model identifier (will be converted to owned String for metrics)
    pub model_id: &'a str,
    /// Endpoint label (e.g., "chat", "generate")
    pub endpoint: &'static str,
    /// Time to first token (None if no tokens were generated)
    pub ttft: Option<Duration>,
    /// Total generation time
    pub generation_duration: Duration,
    /// Input token count (None for endpoints that don't track this)
    pub input_tokens: Option<u64>,
    /// Output token count
    pub output_tokens: u64,
}

impl Metrics {
    /// Record an HTTP request.
    /// Here we want a metric to directly reflect user's experience ("I am sending a request")
    /// when viewing the router as a blackbox, and is bumped immediately when the request arrives.
    pub fn record_http_request(method: &'static str, path: &str) {
        let path_interned = intern_string(path);
        counter!(
            "smg_http_requests_total",
            "method" => method,
            "path" => path_interned,
        )
        .increment(1);
    }

    /// Record HTTP request duration.
    /// For best performance, pass static strings for method.
    pub fn record_http_duration(method: &'static str, path: &str, duration: Duration) {
        let path_interned = intern_string(path);
        histogram!(
            "smg_http_request_duration_seconds",
            "method" => method,
            "path" => path_interned
        )
        .record(duration.as_secs_f64());
    }

    pub fn set_inflight_request_age_count(gt: &'static str, le: &'static str, count: usize) {
        gauge!(
            "smg_http_inflight_request_age_count",
            "gt" => gt,
            "le" => le
        )
        .set(count as f64);
    }

    /// Set active HTTP connections count
    pub fn set_http_connections_active(count: usize) {
        gauge!("smg_http_connections_active").set(count as f64);
    }

    /// Record HTTP response.
    pub fn record_http_response(status_code: u16, error_code: &str) {
        let status_str: Cow<'static, str> = status_code_to_cow(status_code);
        let error_interned = intern_string(error_code);
        counter!(
            "smg_http_responses_total",
            "status_code" => status_str,
            "error_code" => error_interned
        )
        .increment(1);
    }

    /// Record rate limit decision.
    pub fn record_http_rate_limit(result: &'static str) {
        counter!(
            "smg_http_rate_limit_total",
            "result" => result
        )
        .increment(1);
    }

    // ========================================================================
    // Layer 2: Router metrics
    // ========================================================================

    /// Record a routed request.
    ///
    /// Uses string interning for model_id to avoid repeated allocations.
    ///
    /// # Arguments
    /// * `streaming` - Use `bool_to_static_str(request.stream)` or the constants
    pub fn record_router_request(
        router_type: &'static str,
        backend_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        endpoint: &'static str,
        streaming: &'static str,
    ) {
        let model = intern_string(model_id);
        counter!(
            "smg_router_requests_total",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "connection_mode" => connection_mode,
            "model" => model,
            "endpoint" => endpoint,
            "streaming" => streaming
        )
        .increment(1);
    }

    /// Record router request duration.
    /// Uses string interning for model_id.
    pub fn record_router_duration(
        router_type: &'static str,
        backend_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        endpoint: &'static str,
        duration: Duration,
    ) {
        let model = intern_string(model_id);
        histogram!(
            "smg_router_request_duration_seconds",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "connection_mode" => connection_mode,
            "model" => model,
            "endpoint" => endpoint
        )
        .record(duration.as_secs_f64());
    }

    /// Record a router error.
    /// Uses string interning for model_id.
    pub fn record_router_error(
        router_type: &'static str,
        backend_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        endpoint: &'static str,
        error_type: &'static str,
    ) {
        let model = intern_string(model_id);
        counter!(
            "smg_router_request_errors_total",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "connection_mode" => connection_mode,
            "model" => model,
            "endpoint" => endpoint,
            "error_type" => error_type
        )
        .increment(1);
    }

    /// Record pipeline stage duration (gRPC only).
    /// All labels are static, so this is very fast.
    pub fn record_router_stage_duration(
        router_type: &'static str,
        stage: &'static str,
        duration: Duration,
    ) {
        histogram!(
            "smg_router_stage_duration_seconds",
            "router_type" => router_type,
            "stage" => stage
        )
        .record(duration.as_secs_f64());
    }

    /// Record upstream backend response.
    /// Uses static strings for common status codes and interning for error_code.
    pub fn record_router_upstream_response(
        router_type: &'static str,
        status_code: u16,
        error_code: &str,
    ) {
        let status_str: Cow<'static, str> = status_code_to_cow(status_code);
        let error_interned = intern_string(error_code);
        counter!(
            "smg_router_upstream_responses_total",
            "router_type" => router_type,
            "status_code" => status_str,
            "error_code" => error_interned
        )
        .increment(1);
    }

    // ========================================================================
    // Layer 2: Router inference metrics (gRPC only)
    // ========================================================================

    /// Record time to first token.
    /// Uses string interning for model_id.
    pub fn record_router_ttft(
        router_type: &'static str,
        backend_type: &'static str,
        model_id: &str,
        endpoint: &'static str,
        duration: Duration,
    ) {
        let model = intern_string(model_id);
        histogram!(
            "smg_router_ttft_seconds",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint
        )
        .record(duration.as_secs_f64());
    }

    /// Record time per output token
    pub fn record_router_tpot(
        router_type: &'static str,
        backend_type: &'static str,
        model_id: &str,
        endpoint: &'static str,
        duration: Duration,
    ) {
        let model = intern_string(model_id);
        histogram!(
            "smg_router_tpot_seconds",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint
        )
        .record(duration.as_secs_f64());
    }

    /// Record tokens processed
    pub fn record_router_tokens(
        router_type: &'static str,
        backend_type: &'static str,
        model_id: &str,
        endpoint: &'static str,
        token_type: &'static str,
        count: u64,
    ) {
        let model = intern_string(model_id);
        counter!(
            "smg_router_tokens_total",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint,
            "token_type" => token_type
        )
        .increment(count);
    }

    /// Record total generation duration.
    /// Uses string interning for model_id.
    pub fn record_router_generation_duration(
        router_type: &'static str,
        backend_type: &'static str,
        model_id: &str,
        endpoint: &'static str,
        duration: Duration,
    ) {
        let model = intern_string(model_id);
        histogram!(
            "smg_router_generation_duration_seconds",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint
        )
        .record(duration.as_secs_f64());
    }

    /// Record all streaming metrics in a single batch call.
    ///
    /// This consolidates TTFT, TPOT, generation duration, and token metrics
    /// into one function, handling TPOT calculation internally.
    pub fn record_streaming_metrics(params: StreamingMetricsParams<'_>) {
        let StreamingMetricsParams {
            router_type,
            backend_type,
            model_id,
            endpoint,
            ttft,
            generation_duration,
            input_tokens,
            output_tokens,
        } = params;

        // Intern model string once - Arc::clone is just a ref count increment
        let model = intern_string(model_id);

        // TTFT and TPOT (only if we have a first token time)
        if let Some(ttft_duration) = ttft {
            histogram!(
                "smg_router_ttft_seconds",
                "router_type" => router_type,
                "backend_type" => backend_type,
                "model" => Arc::clone(&model),
                "endpoint" => endpoint
            )
            .record(ttft_duration.as_secs_f64());

            // TPOT - only meaningful with >1 output token
            if output_tokens > 1 {
                let time_after_first = generation_duration.saturating_sub(ttft_duration);
                let tpot = time_after_first / (output_tokens as u32 - 1);
                histogram!(
                    "smg_router_tpot_seconds",
                    "router_type" => router_type,
                    "backend_type" => backend_type,
                    "model" => Arc::clone(&model),
                    "endpoint" => endpoint
                )
                .record(tpot.as_secs_f64());
            }
        }

        // Generation duration (always recorded)
        histogram!(
            "smg_router_generation_duration_seconds",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => Arc::clone(&model),
            "endpoint" => endpoint
        )
        .record(generation_duration.as_secs_f64());

        // Input tokens (if available)
        if let Some(input) = input_tokens {
            counter!(
                "smg_router_tokens_total",
                "router_type" => router_type,
                "backend_type" => backend_type,
                "model" => Arc::clone(&model),
                "endpoint" => endpoint,
                "token_type" => metrics_labels::TOKEN_INPUT
            )
            .increment(input);
        }

        // Output tokens (always recorded - move model on final use)
        counter!(
            "smg_router_tokens_total",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint,
            "token_type" => metrics_labels::TOKEN_OUTPUT
        )
        .increment(output_tokens);
    }

    // ========================================================================
    // Layer 3: Worker metrics
    // ========================================================================

    /// Set worker pool size
    pub fn set_worker_pool_size(
        worker_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        size: usize,
    ) {
        let model = intern_string(model_id);
        gauge!(
            "smg_worker_pool_size",
            "worker_type" => worker_type,
            "connection_mode" => connection_mode,
            "model" => model
        )
        .set(size as f64);
    }

    /// Set active worker connections
    pub fn set_worker_connections_active(
        worker_type: &'static str,
        connection_mode: &'static str,
        count: usize,
    ) {
        gauge!(
            "smg_worker_connections_active",
            "worker_type" => worker_type,
            "connection_mode" => connection_mode
        )
        .set(count as f64);
    }

    /// Record health check result
    pub fn record_worker_health_check(worker_type: &'static str, result: &'static str) {
        counter!(
            "smg_worker_health_checks_total",
            "worker_type" => worker_type,
            "result" => result
        )
        .increment(1);
    }

    /// Record worker selection
    pub fn record_worker_selection(
        worker_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        policy: &'static str,
    ) {
        let model = intern_string(model_id);
        counter!(
            "smg_worker_selection_total",
            "worker_type" => worker_type,
            "connection_mode" => connection_mode,
            "model" => model,
            "policy" => policy
        )
        .increment(1);
    }

    /// Record worker error
    pub fn record_worker_error(
        worker_type: &'static str,
        connection_mode: &'static str,
        error_type: &'static str,
    ) {
        counter!(
            "smg_worker_errors_total",
            "worker_type" => worker_type,
            "connection_mode" => connection_mode,
            "error_type" => error_type
        )
        .increment(1);
    }

    /// Record manual policy execution branch for routing decisions
    pub fn record_worker_manual_policy_branch(branch: &'static str) {
        counter!(
            "smg_manual_policy_branch_total",
            "branch" => branch
        )
        .increment(1);
    }

    /// Set manual policy cache entries count
    pub fn set_manual_policy_cache_entries(count: usize) {
        gauge!("smg_manual_policy_cache_entries").set(count as f64);
    }

    /// Record consistent hashing policy execution branch for routing decisions
    pub fn record_worker_consistent_hashing_policy_branch(branch: &'static str) {
        counter!(
            "smg_consistent_hashing_policy_branch_total",
            "branch" => branch
        )
        .increment(1);
    }

    /// Record prefix hash policy execution branch for routing decisions
    pub fn record_worker_prefix_hash_policy_branch(branch: &'static str) {
        counter!(
            "smg_prefix_hash_policy_branch_total",
            "branch" => branch
        )
        .increment(1);
    }

    /// Set running requests per worker
    pub fn set_worker_requests_active(worker: &str, count: usize) {
        let worker_interned = intern_string(worker);
        gauge!(
            "smg_worker_requests_active",
            "worker" => worker_interned
        )
        .set(count as f64);
    }

    /// Set worker health status
    pub fn set_worker_health(worker_url: &str, healthy: bool) {
        let worker_interned = intern_string(worker_url);
        gauge!(
            "smg_worker_health",
            "worker" => worker_interned
        )
        .set(if healthy { 1.0 } else { 0.0 });
    }

    // ========================================================================
    // Layer 3: Worker resilience metrics (circuit breaker)
    // ========================================================================

    /// Set circuit breaker state (0=closed, 1=open, 2=half_open)
    pub fn set_worker_cb_state(worker: &str, state_code: u8) {
        let worker_interned = intern_string(worker);
        gauge!(
            "smg_worker_cb_state",
            "worker" => worker_interned
        )
        .set(state_code as f64);
    }

    /// Record circuit breaker state transition
    pub fn record_worker_cb_transition(worker: &str, from: &'static str, to: &'static str) {
        let worker_interned = intern_string(worker);
        counter!(
            "smg_worker_cb_transitions_total",
            "worker" => worker_interned,
            "from" => from,
            "to" => to
        )
        .increment(1);
    }

    /// Record circuit breaker outcome
    pub fn record_worker_cb_outcome(worker: &str, outcome: &'static str) {
        let worker_interned = intern_string(worker);
        counter!(
            "smg_worker_cb_outcomes_total",
            "worker" => worker_interned,
            "outcome" => outcome
        )
        .increment(1);
    }

    /// Set circuit breaker consecutive failures
    pub fn set_worker_cb_consecutive_failures(worker: &str, count: u32) {
        let worker_interned = intern_string(worker);
        gauge!(
            "smg_worker_cb_consecutive_failures",
            "worker" => worker_interned
        )
        .set(count as f64);
    }

    /// Set circuit breaker consecutive successes
    pub fn set_worker_cb_consecutive_successes(worker: &str, count: u32) {
        let worker_interned = intern_string(worker);
        gauge!(
            "smg_worker_cb_consecutive_successes",
            "worker" => worker_interned
        )
        .set(count as f64);
    }

    // ========================================================================
    // Layer 3: Worker resilience metrics (retry)
    // ========================================================================

    /// Record retry attempt
    pub fn record_worker_retry(worker_type: &'static str, endpoint: &'static str) {
        counter!(
            "smg_worker_retries_total",
            "worker_type" => worker_type,
            "endpoint" => endpoint
        )
        .increment(1);
    }

    /// Record retries exhausted
    pub fn record_worker_retries_exhausted(worker_type: &'static str, endpoint: &'static str) {
        counter!(
            "smg_worker_retries_exhausted_total",
            "worker_type" => worker_type,
            "endpoint" => endpoint
        )
        .increment(1);
    }

    /// Record retry backoff duration.
    pub fn record_worker_retry_backoff(attempt: u32, duration: Duration) {
        let attempt_str: Cow<'static, str> = match attempt {
            1 => Cow::Borrowed("1"),
            2 => Cow::Borrowed("2"),
            3 => Cow::Borrowed("3"),
            4 => Cow::Borrowed("4"),
            5 => Cow::Borrowed("5"),
            _ => Cow::Owned(attempt.to_string()),
        };
        histogram!(
            "smg_worker_retry_backoff_seconds",
            "attempt" => attempt_str
        )
        .record(duration.as_secs_f64());
    }

    // ========================================================================
    // Layer 4: Discovery metrics
    // ========================================================================

    /// Record worker registration attempt
    pub fn record_discovery_registration(source: &'static str, result: &'static str) {
        counter!(
            "smg_discovery_registrations_total",
            "source" => source,
            "result" => result
        )
        .increment(1);
    }

    /// Record worker deregistration
    pub fn record_discovery_deregistration(source: &'static str, reason: &'static str) {
        counter!(
            "smg_discovery_deregistrations_total",
            "source" => source,
            "reason" => reason
        )
        .increment(1);
    }

    /// Record discovery sync duration
    pub fn record_discovery_sync_duration(source: &'static str, duration: Duration) {
        histogram!(
            "smg_discovery_sync_duration_seconds",
            "source" => source
        )
        .record(duration.as_secs_f64());
    }

    /// Set workers discovered count
    pub fn set_discovery_workers_discovered(source: &'static str, count: usize) {
        gauge!(
            "smg_discovery_workers_discovered",
            "source" => source
        )
        .set(count as f64);
    }

    // ========================================================================
    // Layer 5: MCP metrics
    // ========================================================================

    /// Record MCP tool call
    pub fn record_mcp_tool_call(model_id: &str, tool_name: &str, result: &'static str) {
        let model = intern_string(model_id);
        let tool = intern_string(tool_name);
        counter!(
            "smg_mcp_tool_calls_total",
            "model" => model,
            "tool_name" => tool,
            "result" => result
        )
        .increment(1);
    }

    /// Record MCP tool execution duration
    pub fn record_mcp_tool_duration(model_id: &str, tool_name: &str, duration: Duration) {
        let model = intern_string(model_id);
        let tool = intern_string(tool_name);
        histogram!(
            "smg_mcp_tool_duration_seconds",
            "model" => model,
            "tool_name" => tool
        )
        .record(duration.as_secs_f64());
    }

    /// Set active MCP servers count
    pub fn set_mcp_servers_active(count: usize) {
        gauge!("smg_mcp_servers_active").set(count as f64);
    }

    /// Record MCP tool loop iteration
    pub fn record_mcp_tool_iteration(model_id: &str) {
        let model = intern_string(model_id);
        counter!(
            "smg_mcp_tool_iterations_total",
            "model" => model
        )
        .increment(1);
    }

    // ========================================================================
    // Layer 6: Database metrics
    // ========================================================================

    /// Record database operation
    pub fn record_db_operation(
        storage_type: &'static str,
        operation: &'static str,
        result: &'static str,
    ) {
        counter!(
            "smg_db_operations_total",
            "storage_type" => storage_type,
            "operation" => operation,
            "result" => result
        )
        .increment(1);
    }

    /// Record database operation duration
    pub fn record_db_operation_duration(
        storage_type: &'static str,
        operation: &'static str,
        duration: Duration,
    ) {
        histogram!(
            "smg_db_operation_duration_seconds",
            "storage_type" => storage_type,
            "operation" => operation
        )
        .record(duration.as_secs_f64());
    }

    /// Set active database connections
    pub fn set_db_connections_active(storage_type: &'static str, count: usize) {
        gauge!(
            "smg_db_connections_active",
            "storage_type" => storage_type
        )
        .set(count as f64);
    }

    /// Record item stored
    pub fn increment_db_items_stored(storage_type: &'static str) {
        counter!(
            "smg_db_items_stored",
            "storage_type" => storage_type
        )
        .increment(1);
    }

    // ========================================================================
    // Worker cleanup
    // ========================================================================

    pub fn remove_worker_metrics(worker_url: &str) {
        // Intern once, clone (cheap) for each metric
        let worker = intern_string(worker_url);

        gauge!("smg_worker_cb_consecutive_failures", "worker" => Arc::clone(&worker)).set(0.0);
        gauge!("smg_worker_cb_consecutive_successes", "worker" => Arc::clone(&worker)).set(0.0);
        gauge!("smg_worker_requests_active", "worker" => Arc::clone(&worker)).set(0.0);

        // Zero for these metrics have special valid meaning, thus we set to -1 temporarily
        // (and will remove them completely after https://github.com/metrics-rs/metrics/issues/653)
        gauge!("smg_worker_cb_state", "worker" => Arc::clone(&worker)).set(-1.0);
        gauge!("smg_worker_health", "worker" => worker).set(-1.0);
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use super::*;

    #[test]
    fn test_prometheus_config_default() {
        let config = PrometheusConfig::default();
        assert_eq!(config.port, 29000);
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_prometheus_config_custom() {
        let config = PrometheusConfig {
            port: 8080,
            host: "127.0.0.1".to_string(),
            duration_buckets: None,
        };
        assert_eq!(config.port, 8080);
        assert_eq!(config.host, "127.0.0.1");
    }

    #[test]
    fn test_prometheus_config_clone() {
        let config = PrometheusConfig {
            port: 9090,
            host: "192.168.1.1".to_string(),
            duration_buckets: None,
        };
        let cloned = config.clone();
        assert_eq!(cloned.port, config.port);
        assert_eq!(cloned.host, config.host);
    }

    #[test]
    fn test_valid_ipv4_parsing() {
        let test_cases = vec!["127.0.0.1", "192.168.1.1", "0.0.0.0"];

        for ip_str in test_cases {
            let config = PrometheusConfig {
                port: 29000,
                host: ip_str.to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            assert!(matches!(ip_addr, IpAddr::V4(_)));
        }
    }

    #[test]
    fn test_valid_ipv6_parsing() {
        let test_cases = vec!["::1", "2001:db8::1", "::"];

        for ip_str in test_cases {
            let config = PrometheusConfig {
                port: 29000,
                host: ip_str.to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            assert!(matches!(ip_addr, IpAddr::V6(_)));
        }
    }

    #[test]
    fn test_invalid_ip_parsing() {
        let test_cases = vec!["invalid", "256.256.256.256", "hostname"];

        for ip_str in test_cases {
            let config = PrometheusConfig {
                port: 29000,
                host: ip_str.to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config
                .host
                .parse()
                .unwrap_or(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));

            assert_eq!(ip_addr, IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));
        }
    }

    #[test]
    fn test_socket_addr_creation() {
        let test_cases = vec![("127.0.0.1", 8080), ("0.0.0.0", 29000), ("::1", 9090)];

        for (host, port) in test_cases {
            let config = PrometheusConfig {
                port,
                host: host.to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            let socket_addr = SocketAddr::new(ip_addr, config.port);

            assert_eq!(socket_addr.port(), port);
            assert_eq!(socket_addr.ip().to_string(), host);
        }
    }

    #[test]
    fn test_socket_addr_with_different_ports() {
        let ports = vec![0, 80, 8080, 65535];

        for port in ports {
            let config = PrometheusConfig {
                port,
                host: "127.0.0.1".to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            let socket_addr = SocketAddr::new(ip_addr, config.port);

            assert_eq!(socket_addr.port(), port);
        }
    }

    #[test]
    fn test_duration_bucket_coverage() {
        let test_cases: [(f64, &str); 7] = [
            (0.0005, "sub-millisecond"),
            (0.005, "5ms"),
            (0.05, "50ms"),
            (1.0, "1s"),
            (10.0, "10s"),
            (60.0, "1m"),
            (240.0, "4m"),
        ];

        let buckets: [f64; 20] = [
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
            60.0, 90.0, 120.0, 180.0, 240.0,
        ];

        for (duration, label) in test_cases {
            let bucket_found = buckets
                .iter()
                .any(|&b| (b - duration).abs() < 0.0001 || b > duration);
            assert!(bucket_found, "No bucket found for {} ({})", duration, label);
        }
    }

    #[test]
    fn test_duration_suffix_matcher() {
        let matcher = Matcher::Suffix(String::from("duration_seconds"));

        let _matching_metrics = [
            "request_duration_seconds",
            "response_duration_seconds",
            "smg_request_duration_seconds",
        ];

        let _non_matching_metrics = ["duration_total", "duration_seconds_total", "other_metric"];

        match matcher {
            Matcher::Suffix(suffix) => assert_eq!(suffix, "duration_seconds"),
            _ => panic!("Expected Suffix matcher"),
        }
    }

    #[test]
    fn test_prometheus_builder_configuration() {
        let _config = PrometheusConfig::default();

        let duration_matcher = Matcher::Suffix(String::from("duration_seconds"));
        let duration_bucket = [
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
            60.0, 90.0, 120.0, 180.0, 240.0,
        ];

        assert_eq!(duration_bucket.len(), 20);

        match duration_matcher {
            Matcher::Suffix(s) => assert_eq!(s, "duration_seconds"),
            _ => panic!("Expected Suffix matcher"),
        }
    }

    #[test]
    fn test_upkeep_timeout_duration() {
        let timeout = Duration::from_secs(5 * 60);
        assert_eq!(timeout.as_secs(), 300);
    }

    #[test]
    fn test_custom_buckets_for_different_metrics() {
        let request_buckets = [0.001, 0.01, 0.1, 1.0, 10.0];
        let generate_buckets = [0.1, 0.5, 1.0, 5.0, 30.0, 60.0];

        assert_eq!(request_buckets.len(), 5);
        assert_eq!(generate_buckets.len(), 6);

        for i in 1..request_buckets.len() {
            assert!(request_buckets[i] > request_buckets[i - 1]);
        }

        for i in 1..generate_buckets.len() {
            assert!(generate_buckets[i] > generate_buckets[i - 1]);
        }
    }

    #[test]
    fn test_port_already_in_use() {
        let port = 29123;

        if let Ok(_listener) = TcpListener::bind(("127.0.0.1", port)) {
            let config = PrometheusConfig {
                port,
                host: "127.0.0.1".to_string(),
                duration_buckets: None,
            };

            assert_eq!(config.port, port);
        }
    }

    #[test]
    fn test_metrics_endpoint_accessibility() {
        let config = PrometheusConfig {
            port: 29000,
            host: "127.0.0.1".to_string(),
            duration_buckets: None,
        };

        let ip_addr: IpAddr = config.host.parse().unwrap();
        let socket_addr = SocketAddr::new(ip_addr, config.port);

        assert_eq!(socket_addr.to_string(), "127.0.0.1:29000");
    }

    // ========================================================================
    // String interning tests
    // ========================================================================

    #[test]
    fn test_intern_string_returns_same_arc() {
        let s1 = intern_string("test_model");
        let s2 = intern_string("test_model");

        // Should return the same Arc (pointer equality)
        assert!(Arc::ptr_eq(&s1, &s2));
        assert_eq!(&*s1, "test_model");
    }

    #[test]
    fn test_intern_string_different_strings() {
        let s1 = intern_string("model_a");
        let s2 = intern_string("model_b");

        // Different strings should have different Arcs
        assert!(!Arc::ptr_eq(&s1, &s2));
        assert_eq!(&*s1, "model_a");
        assert_eq!(&*s2, "model_b");
    }

    #[test]
    fn test_intern_string_empty() {
        let s1 = intern_string("");
        let s2 = intern_string("");

        assert!(Arc::ptr_eq(&s1, &s2));
        assert_eq!(&*s1, "");
    }

    #[test]
    fn test_interner_size_grows() {
        let initial_size = interner_size();

        // Intern some unique strings
        let unique = format!("unique_test_string_{}", initial_size);
        intern_string(&unique);

        assert!(interner_size() > initial_size);
    }

    #[test]
    fn test_bool_to_static_str() {
        assert_eq!(bool_to_static_str(true), "true");
        assert_eq!(bool_to_static_str(false), "false");
    }

    #[test]
    fn test_status_code_to_static_str() {
        // Common codes should return static strings
        assert_eq!(status_code_to_static_str(200), Some("200"));
        assert_eq!(status_code_to_static_str(404), Some("404"));
        assert_eq!(status_code_to_static_str(500), Some("500"));

        // Uncommon codes should return None
        assert_eq!(status_code_to_static_str(418), None);
        assert_eq!(status_code_to_static_str(999), None);
    }

    #[test]
    fn test_status_code_to_cow() {
        // Common codes should be borrowed
        let cow_200 = status_code_to_cow(200);
        assert!(matches!(cow_200, Cow::Borrowed(_)));
        assert_eq!(cow_200, "200");

        // Uncommon codes should be owned
        let cow_418 = status_code_to_cow(418);
        assert!(matches!(cow_418, Cow::Owned(_)));
        assert_eq!(cow_418, "418");
    }

    #[test]
    fn test_method_to_static_str() {
        assert_eq!(method_to_static_str("GET"), "GET");
        assert_eq!(method_to_static_str("POST"), "POST");
        assert_eq!(method_to_static_str("UNKNOWN"), "OTHER");
    }
}
