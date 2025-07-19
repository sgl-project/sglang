use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct PrometheusConfig {
    pub port: u16,
    pub host: String,
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            port: 29000,
            host: "0.0.0.0".to_string(),
        }
    }
}

pub fn init_metrics() {
    // Request metrics
    describe_counter!(
        "sgl_router_requests_total",
        "Total number of requests by route and method"
    );
    describe_histogram!(
        "sgl_router_request_duration_seconds",
        "Request duration in seconds by route"
    );
    describe_counter!(
        "sgl_router_request_errors_total",
        "Total number of request errors by route and error type"
    );
    describe_counter!(
        "sgl_router_retries_total",
        "Total number of request retries by route"
    );

    // Worker metrics
    describe_gauge!(
        "sgl_router_active_workers",
        "Number of currently active workers"
    );
    describe_gauge!(
        "sgl_router_worker_health",
        "Worker health status (1=healthy, 0=unhealthy)"
    );
    describe_gauge!("sgl_router_worker_load", "Current load on each worker");
    describe_counter!(
        "sgl_router_processed_requests_total",
        "Total requests processed by each worker"
    );

    // Policy metrics
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

    // PD-specific metrics
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

    // Service discovery metrics
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

    // Generate request specific metrics
    describe_histogram!(
        "sgl_router_generate_duration_seconds",
        "Generate request duration"
    );

    // Running requests gauge for cache-aware policy
    describe_gauge!(
        "sgl_router_running_requests",
        "Number of running requests per worker"
    );
}

pub fn start_prometheus(config: PrometheusConfig) {
    // Initialize metric descriptions
    init_metrics();

    let duration_matcher = Matcher::Suffix(String::from("duration"));
    let duration_bucket = [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
        60.0, 90.0, 120.0, 180.0, 240.0,
    ];

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
    // Request metrics
    pub fn record_request(route: &str) {
        counter!("sgl_router_requests_total",
            "route" => route.to_string()
        )
        .increment(1);
    }

    pub fn record_request_duration(route: &str, duration: Duration) {
        histogram!("sgl_router_request_duration_seconds",
            "route" => route.to_string()
        )
        .record(duration.as_secs_f64());
    }

    pub fn record_request_error(route: &str, error_type: &str) {
        counter!("sgl_router_request_errors_total",
            "route" => route.to_string(),
            "error_type" => error_type.to_string()
        )
        .increment(1);
    }

    pub fn record_retry(route: &str) {
        counter!("sgl_router_retries_total",
            "route" => route.to_string()
        )
        .increment(1);
    }

    // Worker metrics
    pub fn set_active_workers(count: usize) {
        gauge!("sgl_router_active_workers").set(count as f64);
    }

    pub fn set_worker_health(worker_url: &str, healthy: bool) {
        gauge!("sgl_router_worker_health",
            "worker" => worker_url.to_string()
        )
        .set(if healthy { 1.0 } else { 0.0 });
    }

    pub fn set_worker_load(worker_url: &str, load: usize) {
        gauge!("sgl_router_worker_load",
            "worker" => worker_url.to_string()
        )
        .set(load as f64);
    }

    pub fn record_processed_request(worker_url: &str) {
        counter!("sgl_router_processed_requests_total",
            "worker" => worker_url.to_string()
        )
        .increment(1);
    }

    // Policy metrics
    pub fn record_policy_decision(policy: &str, worker: &str) {
        counter!("sgl_router_policy_decisions_total",
            "policy" => policy.to_string(),
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

    // PD-specific metrics
    pub fn record_pd_request(route: &str) {
        counter!("sgl_router_pd_requests_total",
            "route" => route.to_string()
        )
        .increment(1);
    }

    pub fn record_pd_request_duration(route: &str, duration: Duration) {
        histogram!("sgl_router_pd_request_duration_seconds",
            "route" => route.to_string()
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

    pub fn record_pd_error(error_type: &str) {
        counter!("sgl_router_pd_errors_total",
            "error_type" => error_type.to_string()
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

    // Service discovery metrics
    pub fn record_discovery_update(added: usize, removed: usize) {
        counter!("sgl_router_discovery_updates_total").increment(1);
        gauge!("sgl_router_discovery_workers_added").set(added as f64);
        gauge!("sgl_router_discovery_workers_removed").set(removed as f64);
    }

    // Generate request metrics
    pub fn record_generate_duration(duration: Duration) {
        histogram!("sgl_router_generate_duration_seconds").record(duration.as_secs_f64());
    }

    // Running requests for cache-aware policy
    pub fn set_running_requests(worker: &str, count: usize) {
        gauge!("sgl_router_running_requests",
            "worker" => worker.to_string()
        )
        .set(count as f64);
    }
}
