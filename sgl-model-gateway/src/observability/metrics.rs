use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    time::Duration,
};

use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder};

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

pub fn init_metrics() {
    describe_counter!(
        "sgl_router_requests_total",
        "Total number of requests by route and method"
    );
    describe_histogram!(
        "sgl_router_request_duration_seconds",
        "Request duration in seconds"
    );
    describe_counter!(
        "sgl_router_request_errors_total",
        "Total number of request errors by route and error type"
    );
    describe_counter!(
        "sgl_router_attempt_http_responses_total",
        "Total number of upstream engine HTTP responses by status code"
    );
    describe_counter!(
        "sgl_router_retries_total",
        "Total number of request retries by route"
    );
    describe_histogram!(
        "sgl_router_retry_backoff_duration_seconds",
        "Backoff duration in seconds by attempt index"
    );
    describe_counter!(
        "sgl_router_retries_exhausted_total",
        "Total number of requests that exhausted retries by route"
    );

    describe_gauge!(
        "sgl_router_cb_state",
        "Circuit breaker state per worker (0=closed, 1=open, 2=half_open)"
    );
    describe_counter!(
        "sgl_router_cb_state_transitions_total",
        "Total number of circuit breaker state transitions by worker"
    );
    describe_counter!(
        "sgl_router_cb_outcomes_total",
        "Total number of circuit breaker outcomes by worker and outcome type (success/failure)"
    );
    describe_gauge!(
        "sgl_router_cb_consecutive_failures",
        "Current consecutive failure count per worker circuit breaker"
    );
    describe_gauge!(
        "sgl_router_cb_consecutive_successes",
        "Current consecutive success count per worker circuit breaker"
    );

    describe_counter!(
        "sgl_router_discovery_watcher_errors_total",
        "Total number of Kubernetes watcher errors"
    );
    describe_counter!(
        "sgl_router_discovery_watcher_restarts_total",
        "Total number of Kubernetes watcher restarts"
    );

    describe_gauge!(
        "sgl_router_active_workers",
        "Number of currently active workers"
    );
    describe_gauge!(
        "sgl_router_worker_health",
        "Worker health status (1=healthy, 0=unhealthy)"
    );
    describe_counter!(
        "sgl_router_processed_requests_total",
        "Total requests processed by each worker"
    );

    describe_gauge!(
        "sgl_router_job_queue_depth",
        "Current number of pending jobs in the queue"
    );
    describe_histogram!(
        "sgl_router_job_duration_seconds",
        "Job processing duration in seconds by job type"
    );
    describe_counter!(
        "sgl_router_job_success_total",
        "Total successful job completions by job type"
    );
    describe_counter!(
        "sgl_router_job_failure_total",
        "Total failed job completions by job type"
    );
    describe_counter!(
        "sgl_router_job_queue_full_total",
        "Total number of jobs rejected due to queue full"
    );
    describe_counter!(
        "sgl_router_job_shutdown_rejected_total",
        "Total number of jobs rejected due to shutdown"
    );

    describe_counter!(
        "sgl_router_policy_decisions_total",
        "Total routing policy decisions by policy and worker"
    );
    describe_counter!("sgl_router_cache_hits_total", "Total cache hits");
    describe_counter!("sgl_router_cache_misses_total", "Total cache misses");
    describe_gauge!(
        "sgl_router_tree_size",
        "Current tree size for cache-aware routing"
    );
    describe_counter!(
        "sgl_router_load_balancing_events_total",
        "Total load balancing trigger events"
    );
    describe_gauge!("sgl_router_max_load", "Maximum worker load");
    describe_gauge!("sgl_router_min_load", "Minimum worker load");

    describe_counter!("sgl_router_pd_requests_total", "Total PD requests by route");
    describe_counter!(
        "sgl_router_pd_prefill_requests_total",
        "Total prefill requests per worker"
    );
    describe_counter!(
        "sgl_router_pd_decode_requests_total",
        "Total decode requests per worker"
    );
    describe_counter!(
        "sgl_router_pd_errors_total",
        "Total PD errors by error type"
    );
    describe_counter!(
        "sgl_router_pd_prefill_errors_total",
        "Total prefill server errors"
    );
    describe_counter!(
        "sgl_router_pd_decode_errors_total",
        "Total decode server errors"
    );
    describe_counter!(
        "sgl_router_pd_stream_errors_total",
        "Total streaming errors per worker"
    );
    describe_histogram!(
        "sgl_router_pd_request_duration_seconds",
        "PD request duration by route"
    );

    describe_counter!(
        "sgl_router_discovery_updates_total",
        "Total service discovery update events"
    );
    describe_gauge!(
        "sgl_router_discovery_workers_added",
        "Number of workers added in last discovery update"
    );
    describe_gauge!(
        "sgl_router_discovery_workers_removed",
        "Number of workers removed in last discovery update"
    );

    describe_histogram!(
        "sgl_router_generate_duration_seconds",
        "Generate request duration"
    );

    describe_counter!("sgl_router_embeddings_total", "Total embedding requests");
    describe_histogram!(
        "sgl_router_embeddings_duration_seconds",
        "Embedding request duration"
    );
    describe_counter!(
        "sgl_router_embeddings_errors_total",
        "Embedding request errors"
    );
    describe_gauge!("sgl_router_embeddings_queue_size", "Embedding queue size");

    describe_gauge!(
        "sgl_router_running_requests",
        "Number of running requests per worker"
    );

    describe_counter!(
        "sgl_router_http_requests_total",
        "Total number of HTTP requests"
    );
    describe_counter!(
        "sgl_router_http_responses_total",
        "Total number of HTTP responses by status code and error code"
    );

    // ========================================================================
    // SMG Metrics (new layered architecture)
    // ========================================================================

    // Layer 1: HTTP metrics
    describe_counter!(
        "smg_http_requests_total",
        "Total HTTP requests by method, path, and status"
    );
    describe_histogram!(
        "smg_http_request_duration_seconds",
        "HTTP request duration by method and path"
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

pub struct RouterMetrics;

impl RouterMetrics {
    pub fn record_request(route: &'static str) {
        counter!("sgl_router_requests_total",
            "route" => route
        )
        .increment(1);
    }

    pub fn record_request_duration(duration: Duration) {
        histogram!("sgl_router_request_duration_seconds").record(duration.as_secs_f64());
    }

    pub fn record_request_error(route: &'static str, error_type: &'static str) {
        counter!("sgl_router_request_errors_total",
            "route" => route,
            "error_type" => error_type
        )
        .increment(1);
    }

    // TODO unify metric names
    pub fn record_attempt_http_response(route: &'static str, status_code: u16, error_code: &str) {
        counter!("sgl_router_attempt_http_responses_total",
            "route" => route,
            "status_code" => status_code.to_string(),
            "error_code" => error_code.to_string()
        )
        .increment(1);
    }

    pub fn record_retry(route: &'static str) {
        counter!("sgl_router_retries_total",
            "route" => route
        )
        .increment(1);
    }

    pub fn record_retry_backoff_duration(duration: Duration, attempt: u32) {
        histogram!("sgl_router_retry_backoff_duration_seconds",
            "attempt" => attempt.to_string()
        )
        .record(duration.as_secs_f64());
    }

    pub fn record_retries_exhausted(route: &'static str) {
        counter!("sgl_router_retries_exhausted_total",
            "route" => route
        )
        .increment(1);
    }

    pub fn set_worker_health(worker_url: &str, healthy: bool) {
        gauge!("sgl_router_worker_health",
            "worker" => worker_url.to_string()
        )
        .set(if healthy { 1.0 } else { 0.0 });
    }

    pub fn set_active_workers(count: usize) {
        gauge!("sgl_router_active_workers").set(count as f64);
    }

    pub fn record_processed_request(worker_url: &str) {
        counter!("sgl_router_processed_requests_total",
            "worker" => worker_url.to_string()
        )
        .increment(1);
    }

    pub fn record_policy_decision(policy: &'static str, worker: &str) {
        counter!("sgl_router_policy_decisions_total",
            "policy" => policy,
            "worker" => worker.to_string()
        )
        .increment(1);
    }

    pub fn record_cache_hit() {
        counter!("sgl_router_cache_hits_total").increment(1);
    }

    pub fn record_cache_miss() {
        counter!("sgl_router_cache_misses_total").increment(1);
    }

    pub fn set_tree_size(worker: &str, size: usize) {
        gauge!("sgl_router_tree_size",
            "worker" => worker.to_string()
        )
        .set(size as f64);
    }

    pub fn record_load_balancing_event() {
        counter!("sgl_router_load_balancing_events_total").increment(1);
    }

    pub fn set_load_range(max_load: usize, min_load: usize) {
        gauge!("sgl_router_max_load").set(max_load as f64);
        gauge!("sgl_router_min_load").set(min_load as f64);
    }

    pub fn record_pd_request(route: &'static str) {
        counter!("sgl_router_pd_requests_total",
            "route" => route
        )
        .increment(1);
    }

    pub fn record_pd_request_duration(route: &'static str, duration: Duration) {
        histogram!("sgl_router_pd_request_duration_seconds",
            "route" => route
        )
        .record(duration.as_secs_f64());
    }

    pub fn record_pd_prefill_request(worker: &str) {
        counter!("sgl_router_pd_prefill_requests_total",
            "worker" => worker.to_string()
        )
        .increment(1);
    }

    pub fn record_pd_decode_request(worker: &str) {
        counter!("sgl_router_pd_decode_requests_total",
            "worker" => worker.to_string()
        )
        .increment(1);
    }

    pub fn record_pd_error(error_type: &'static str) {
        counter!("sgl_router_pd_errors_total",
            "error_type" => error_type
        )
        .increment(1);
    }

    pub fn record_pd_prefill_error(worker: &str) {
        counter!("sgl_router_pd_prefill_errors_total",
            "worker" => worker.to_string()
        )
        .increment(1);
    }

    pub fn record_pd_decode_error(worker: &str) {
        counter!("sgl_router_pd_decode_errors_total",
            "worker" => worker.to_string()
        )
        .increment(1);
    }

    pub fn record_pd_stream_error(worker: &str) {
        counter!("sgl_router_pd_stream_errors_total",
            "worker" => worker.to_string()
        )
        .increment(1);
    }

    pub fn record_discovery_update(added: usize, removed: usize) {
        counter!("sgl_router_discovery_updates_total").increment(1);
        gauge!("sgl_router_discovery_workers_added").set(added as f64);
        gauge!("sgl_router_discovery_workers_removed").set(removed as f64);
    }

    pub fn record_generate_duration(duration: Duration) {
        histogram!("sgl_router_generate_duration_seconds").record(duration.as_secs_f64());
    }

    pub fn record_embeddings_request() {
        counter!("sgl_router_embeddings_total").increment(1);
    }

    pub fn record_embeddings_duration(duration: Duration) {
        histogram!("sgl_router_embeddings_duration_seconds").record(duration.as_secs_f64());
    }

    pub fn record_embeddings_error(error_type: &str) {
        counter!(
            "sgl_router_embeddings_errors_total",
            "error_type" => error_type.to_string()
        )
        .increment(1);
    }

    pub fn set_embeddings_queue_size(size: usize) {
        gauge!("sgl_router_embeddings_queue_size").set(size as f64);
    }

    pub fn record_classify_request() {
        counter!("sgl_router_classify_total").increment(1);
    }

    pub fn record_classify_duration(duration: Duration) {
        histogram!("sgl_router_classify_duration_seconds").record(duration.as_secs_f64());
    }

    pub fn record_classify_error(error_type: &str) {
        counter!(
            "sgl_router_classify_errors_total",
            "error_type" => error_type.to_string()
        )
        .increment(1);
    }

    pub fn set_classify_queue_size(size: usize) {
        gauge!("sgl_router_classify_queue_size").set(size as f64);
    }

    pub fn set_running_requests(worker: &str, count: usize) {
        gauge!("sgl_router_running_requests",
            "worker" => worker.to_string()
        )
        .set(count as f64);
    }

    pub fn set_cb_state(worker: &str, state_code: u8) {
        gauge!("sgl_router_cb_state",
            "worker" => worker.to_string()
        )
        .set(state_code as f64);
    }

    pub fn record_cb_state_transition(worker: &str, from: &'static str, to: &'static str) {
        counter!("sgl_router_cb_state_transitions_total",
            "worker" => worker.to_string(),
            "from" => from,
            "to" => to
        )
        .increment(1);
    }

    pub fn record_cb_outcome(worker: &str, outcome: &'static str) {
        counter!("sgl_router_cb_outcomes_total",
            "worker" => worker.to_string(),
            "outcome" => outcome
        )
        .increment(1);
    }

    pub fn set_cb_consecutive_failures(worker: &str, count: u32) {
        gauge!("sgl_router_cb_consecutive_failures",
            "worker" => worker.to_string()
        )
        .set(count as f64);
    }

    pub fn set_cb_consecutive_successes(worker: &str, count: u32) {
        gauge!("sgl_router_cb_consecutive_successes",
            "worker" => worker.to_string()
        )
        .set(count as f64);
    }

    pub fn record_discovery_watcher_error() {
        counter!("sgl_router_discovery_watcher_errors_total").increment(1);
    }

    pub fn record_discovery_watcher_restart() {
        counter!("sgl_router_discovery_watcher_restarts_total").increment(1);
    }

    // TODO delete the metrics (instead of setting them to zero)
    pub fn remove_worker_metrics(worker_url: &str) {
        gauge!("sgl_router_cb_consecutive_failures","worker" => worker_url.to_string()).set(0.0);
        gauge!("sgl_router_cb_consecutive_successes","worker" => worker_url.to_string()).set(0.0);
        gauge!("sgl_router_running_requests","worker" => worker_url.to_string()).set(0.0);
        gauge!("sgl_router_tree_size","worker" => worker_url.to_string()).set(0.0);

        // Zero for these metrics have special valid meaning, thus we set to -1 temporarily
        // (and will remove them completely after https://github.com/metrics-rs/metrics/issues/653)
        gauge!("sgl_router_cb_state","worker" => worker_url.to_string()).set(-1.0);
        gauge!("sgl_router_worker_health","worker" => worker_url.to_string()).set(-1.0);
    }

    pub fn set_job_queue_depth(depth: usize) {
        gauge!("sgl_router_job_queue_depth").set(depth as f64);
    }

    pub fn record_job_duration(job_type: &'static str, duration: Duration) {
        histogram!("sgl_router_job_duration_seconds",
            "job_type" => job_type
        )
        .record(duration.as_secs_f64());
    }

    pub fn record_job_success(job_type: &'static str) {
        counter!("sgl_router_job_success_total",
            "job_type" => job_type
        )
        .increment(1);
    }

    pub fn record_job_failure(job_type: &'static str) {
        counter!("sgl_router_job_failure_total",
            "job_type" => job_type
        )
        .increment(1);
    }

    pub fn record_job_queue_full() {
        counter!("sgl_router_job_queue_full_total").increment(1);
    }

    pub fn record_job_shutdown_rejected() {
        counter!("sgl_router_job_shutdown_rejected_total").increment(1);
    }

    // This is different from the following:
    // * sgl_router_requests_total: bump when a request is handled and response is to be returned, thus very different from this.
    // * sgl_router_processed_requests_total: bump when routing decision is made.
    // Here we want a metric to directly reflect user's experience ("I am sending a request")
    // when viewing the router as a blackbox, and is bumped immediately when the request arrives.
    // TODO: add route name
    pub fn record_http_request() {
        counter!("sgl_router_http_requests_total").increment(1);
    }

    pub fn record_http_status_code(status_code: u16, error_code: &str) {
        counter!("sgl_router_http_responses_total",
            "status_code" => status_code.to_string(),
            "error_code" => error_code.to_string()
        )
        .increment(1);
    }
}

// ============================================================================
// SMG Metrics - New layered architecture
// ============================================================================

/// Label constants for consistent metric labeling
pub mod smg_labels {
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

    // Deregistration reasons
    pub const DEREGISTRATION_HEALTH_CHECK_FAILED: &str = "health_check_failed";
    pub const DEREGISTRATION_TIMEOUT: &str = "timeout";
    pub const DEREGISTRATION_MANUAL: &str = "manual";
    pub const DEREGISTRATION_SHUTDOWN: &str = "shutdown";
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

    // Pipeline stages (gRPC router)
    pub const STAGE_PREPARATION: &str = "preparation";
    pub const STAGE_WORKER_SELECTION: &str = "worker_selection";
    pub const STAGE_CLIENT_ACQUISITION: &str = "client_acquisition";
    pub const STAGE_REQUEST_BUILDING: &str = "request_building";
    pub const STAGE_DISPATCH_METADATA: &str = "dispatch_metadata";
    pub const STAGE_REQUEST_EXECUTION: &str = "request_execution";
    pub const STAGE_RESPONSE_PROCESSING: &str = "response_processing";
}

/// SMG Metrics helper struct for the new layered metrics architecture
pub struct SmgMetrics;

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

impl SmgMetrics {
    /// Record an HTTP request
    pub fn record_http_request(method: &str, path: &str, status_class: &str) {
        counter!(
            "smg_http_requests_total",
            "method" => method.to_string(),
            "path" => path.to_string(),
            "status" => status_class.to_string()
        )
        .increment(1);
    }

    /// Record HTTP request duration
    pub fn record_http_duration(method: &str, path: &str, duration: Duration) {
        histogram!(
            "smg_http_request_duration_seconds",
            "method" => method.to_string(),
            "path" => path.to_string()
        )
        .record(duration.as_secs_f64());
    }

    /// Set active HTTP connections count
    pub fn set_http_connections_active(count: usize) {
        gauge!("smg_http_connections_active").set(count as f64);
    }

    /// Record HTTP response
    pub fn record_http_response(status_code: u16, error_code: &str) {
        counter!(
            "smg_http_responses_total",
            "status_code" => status_code.to_string(),
            "error_code" => error_code.to_string()
        )
        .increment(1);
    }

    /// Record rate limit decision
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

    /// Record a routed request
    pub fn record_router_request(
        router_type: &'static str,
        backend_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        endpoint: &'static str,
        streaming: bool,
    ) {
        counter!(
            "smg_router_requests_total",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "connection_mode" => connection_mode,
            "model" => model_id.to_string(),
            "endpoint" => endpoint,
            "streaming" => streaming.to_string()
        )
        .increment(1);
    }

    /// Record router request duration
    pub fn record_router_duration(
        router_type: &'static str,
        backend_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        endpoint: &'static str,
        duration: Duration,
    ) {
        histogram!(
            "smg_router_request_duration_seconds",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "connection_mode" => connection_mode,
            "model" => model_id.to_string(),
            "endpoint" => endpoint
        )
        .record(duration.as_secs_f64());
    }

    /// Record a router error
    pub fn record_router_error(
        router_type: &'static str,
        backend_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        endpoint: &'static str,
        error_type: &'static str,
    ) {
        counter!(
            "smg_router_request_errors_total",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "connection_mode" => connection_mode,
            "model" => model_id.to_string(),
            "endpoint" => endpoint,
            "error_type" => error_type
        )
        .increment(1);
    }

    /// Record pipeline stage duration (gRPC only)
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

    /// Record upstream backend response
    pub fn record_router_upstream_response(
        router_type: &'static str,
        status_code: u16,
        error_code: &str,
    ) {
        counter!(
            "smg_router_upstream_responses_total",
            "router_type" => router_type,
            "status_code" => status_code.to_string(),
            "error_code" => error_code.to_string()
        )
        .increment(1);
    }

    // ========================================================================
    // Layer 2: Router inference metrics (gRPC only)
    // ========================================================================

    /// Record time to first token
    pub fn record_router_ttft(
        router_type: &'static str,
        backend_type: &'static str,
        model_id: &str,
        endpoint: &'static str,
        duration: Duration,
    ) {
        histogram!(
            "smg_router_ttft_seconds",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model_id.to_string(),
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
        histogram!(
            "smg_router_tpot_seconds",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model_id.to_string(),
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
        counter!(
            "smg_router_tokens_total",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model_id.to_string(),
            "endpoint" => endpoint,
            "token_type" => token_type
        )
        .increment(count);
    }

    /// Record total generation duration
    pub fn record_router_generation_duration(
        router_type: &'static str,
        backend_type: &'static str,
        model_id: &str,
        endpoint: &'static str,
        duration: Duration,
    ) {
        histogram!(
            "smg_router_generation_duration_seconds",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model_id.to_string(),
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
        // metrics-rs requires owned strings for dynamic labels (uses Cow<'static, str>).
        // We allocate once and clone for each metric - unavoidable with this API.
        let model = model_id.to_string();

        // TTFT and TPOT (only if we have a first token time)
        if let Some(ttft_duration) = ttft {
            histogram!(
                "smg_router_ttft_seconds",
                "router_type" => router_type,
                "backend_type" => backend_type,
                "model" => model.clone(),
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
                    "model" => model.clone(),
                    "endpoint" => endpoint
                )
                .record(tpot.as_secs_f64());
            }
        }

        // Generation duration
        histogram!(
            "smg_router_generation_duration_seconds",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model.clone(),
            "endpoint" => endpoint
        )
        .record(generation_duration.as_secs_f64());

        // Input tokens (if available)
        if let Some(input) = input_tokens {
            counter!(
                "smg_router_tokens_total",
                "router_type" => router_type,
                "backend_type" => backend_type,
                "model" => model.clone(),
                "endpoint" => endpoint,
                "token_type" => smg_labels::TOKEN_INPUT
            )
            .increment(input);
        }

        // Output tokens
        counter!(
            "smg_router_tokens_total",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint,
            "token_type" => smg_labels::TOKEN_OUTPUT
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
        gauge!(
            "smg_worker_pool_size",
            "worker_type" => worker_type,
            "connection_mode" => connection_mode,
            "model" => model_id.to_string()
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
        counter!(
            "smg_worker_selection_total",
            "worker_type" => worker_type,
            "connection_mode" => connection_mode,
            "model" => model_id.to_string(),
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

    /// Set running requests per worker
    pub fn set_worker_requests_active(worker: &str, count: usize) {
        gauge!(
            "smg_worker_requests_active",
            "worker" => worker.to_string()
        )
        .set(count as f64);
    }

    // ========================================================================
    // Layer 3: Worker resilience metrics (circuit breaker)
    // ========================================================================

    /// Set circuit breaker state (0=closed, 1=open, 2=half_open)
    pub fn set_worker_cb_state(worker: &str, state_code: u8) {
        gauge!(
            "smg_worker_cb_state",
            "worker" => worker.to_string()
        )
        .set(state_code as f64);
    }

    /// Record circuit breaker state transition
    pub fn record_worker_cb_transition(worker: &str, from: &'static str, to: &'static str) {
        counter!(
            "smg_worker_cb_transitions_total",
            "worker" => worker.to_string(),
            "from" => from,
            "to" => to
        )
        .increment(1);
    }

    /// Record circuit breaker outcome
    pub fn record_worker_cb_outcome(worker: &str, outcome: &'static str) {
        counter!(
            "smg_worker_cb_outcomes_total",
            "worker" => worker.to_string(),
            "outcome" => outcome
        )
        .increment(1);
    }

    /// Set circuit breaker consecutive failures
    pub fn set_worker_cb_consecutive_failures(worker: &str, count: u32) {
        gauge!(
            "smg_worker_cb_consecutive_failures",
            "worker" => worker.to_string()
        )
        .set(count as f64);
    }

    /// Set circuit breaker consecutive successes
    pub fn set_worker_cb_consecutive_successes(worker: &str, count: u32) {
        gauge!(
            "smg_worker_cb_consecutive_successes",
            "worker" => worker.to_string()
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

    /// Record retry backoff duration
    pub fn record_worker_retry_backoff(attempt: u32, duration: Duration) {
        histogram!(
            "smg_worker_retry_backoff_seconds",
            "attempt" => attempt.to_string()
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
        counter!(
            "smg_mcp_tool_calls_total",
            "model" => model_id.to_string(),
            "tool_name" => tool_name.to_string(),
            "result" => result
        )
        .increment(1);
    }

    /// Record MCP tool execution duration
    pub fn record_mcp_tool_duration(model_id: &str, tool_name: &str, duration: Duration) {
        histogram!(
            "smg_mcp_tool_duration_seconds",
            "model" => model_id.to_string(),
            "tool_name" => tool_name.to_string()
        )
        .record(duration.as_secs_f64());
    }

    /// Set active MCP servers count
    pub fn set_mcp_servers_active(count: usize) {
        gauge!("smg_mcp_servers_active").set(count as f64);
    }

    /// Record MCP tool loop iteration
    pub fn record_mcp_tool_iteration(model_id: &str) {
        counter!(
            "smg_mcp_tool_iterations_total",
            "model" => model_id.to_string()
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
            "sgl_router_request_duration_seconds",
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
    fn test_metrics_static_methods() {
        RouterMetrics::record_request("/generate");
        RouterMetrics::record_request_duration(Duration::from_millis(100));
        RouterMetrics::record_request_error("/generate", "timeout");
        RouterMetrics::record_retry("/generate");

        RouterMetrics::set_worker_health("http://worker1", true);
        RouterMetrics::record_processed_request("http://worker1");

        RouterMetrics::record_policy_decision("random", "http://worker1");
        RouterMetrics::record_cache_hit();
        RouterMetrics::record_cache_miss();
        RouterMetrics::set_tree_size("http://worker1", 1000);
        RouterMetrics::record_load_balancing_event();
        RouterMetrics::set_load_range(20, 5);

        RouterMetrics::record_pd_request("/v1/chat/completions");
        RouterMetrics::record_pd_request_duration("/v1/chat/completions", Duration::from_secs(1));
        RouterMetrics::record_pd_prefill_request("http://prefill1");
        RouterMetrics::record_pd_decode_request("http://decode1");
        RouterMetrics::record_pd_error("invalid_request");
        RouterMetrics::record_pd_prefill_error("http://prefill1");
        RouterMetrics::record_pd_decode_error("http://decode1");
        RouterMetrics::record_pd_stream_error("http://decode1");

        RouterMetrics::record_discovery_update(3, 1);
        RouterMetrics::record_generate_duration(Duration::from_secs(2));
        RouterMetrics::set_running_requests("http://worker1", 15);
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

    #[test]
    fn test_concurrent_metric_updates() {
        use std::{
            sync::{
                atomic::{AtomicBool, Ordering},
                Arc,
            },
            thread,
        };

        let done = Arc::new(AtomicBool::new(false));
        let mut handles = vec![];

        for i in 0..3 {
            let done_clone = done.clone();
            let handle = thread::spawn(move || {
                let worker = format!("http://worker{}", i);
                while !done_clone.load(Ordering::Relaxed) {
                    RouterMetrics::record_processed_request(&worker);
                    thread::sleep(Duration::from_millis(1));
                }
            });
            handles.push(handle);
        }

        thread::sleep(Duration::from_millis(10));
        done.store(true, Ordering::Relaxed);

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_empty_string_metrics() {
        RouterMetrics::record_request("");
        RouterMetrics::set_worker_health("", true);
        RouterMetrics::record_policy_decision("", "");
    }

    #[test]
    fn test_very_long_metric_labels() {
        let long_label = "a".repeat(1000);

        RouterMetrics::record_request("/very_long_test_route");
        RouterMetrics::set_worker_health(&long_label, false);
    }

    #[test]
    fn test_special_characters_in_labels() {
        let special_labels = [
            "test/with/slashes",
            "test-with-dashes",
            "test_with_underscores",
            "test.with.dots",
            "test:with:colons",
        ];

        for label in special_labels {
            RouterMetrics::record_request(label);
            RouterMetrics::set_worker_health(label, true);
        }
    }

    #[test]
    fn test_extreme_metric_values() {
        RouterMetrics::record_request_duration(Duration::from_nanos(1));
        RouterMetrics::record_request_duration(Duration::from_secs(86400));
    }
}
