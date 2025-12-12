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
        "sgl_router_upstream_http_responses_total",
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

    describe_histogram!(
        "sgl_tokenizer_encode_duration_seconds",
        "Time to encode text to tokens"
    );
    describe_histogram!(
        "sgl_tokenizer_decode_duration_seconds",
        "Time to decode tokens to text"
    );
    describe_histogram!(
        "sgl_tokenizer_encode_batch_duration_seconds",
        "Time to encode a batch of texts"
    );
    describe_counter!(
        "sgl_tokenizer_encode_requests_total",
        "Total number of encode requests by tokenizer type"
    );
    describe_counter!(
        "sgl_tokenizer_decode_requests_total",
        "Total number of decode requests by tokenizer type"
    );
    describe_counter!(
        "sgl_tokenizer_encode_errors_total",
        "Total number of encode errors by error type"
    );
    describe_counter!(
        "sgl_tokenizer_decode_errors_total",
        "Total number of decode errors by error type"
    );
    describe_histogram!(
        "sgl_tokenizer_tokens_per_encode",
        "Number of tokens produced per encode operation"
    );
    describe_histogram!(
        "sgl_tokenizer_chars_per_encode",
        "Number of characters in input text per encode"
    );
    describe_histogram!(
        "sgl_tokenizer_tokens_per_decode",
        "Number of tokens decoded per operation"
    );
    describe_gauge!(
        "sgl_tokenizer_vocab_size",
        "Vocabulary size of the loaded tokenizer"
    );

    describe_counter!(
        "sgl_tokenizer_stop_sequences_detected_total",
        "Total stop sequences detected by type"
    );
    describe_counter!(
        "sgl_tokenizer_partial_matches_total",
        "Total partial stop sequence matches (jailed text)"
    );
    describe_histogram!(
        "sgl_tokenizer_stop_detection_duration_seconds",
        "Time to check for stop sequences per token"
    );

    describe_counter!(
        "sgl_tokenizer_stream_tokens_total",
        "Total tokens processed in streaming decode"
    );
    describe_counter!(
        "sgl_tokenizer_stream_incomplete_utf8_total",
        "Total incomplete UTF-8 sequences detected"
    );
    describe_histogram!(
        "sgl_tokenizer_stream_step_duration_seconds",
        "Time per streaming decode step"
    );

    describe_counter!(
        "sgl_tokenizer_factory_loads_total",
        "Total tokenizer loads by file type"
    );
    describe_counter!(
        "sgl_tokenizer_factory_errors_total",
        "Total tokenizer loading errors by type"
    );
    describe_histogram!(
        "sgl_tokenizer_factory_load_duration_seconds",
        "Time to load and initialize tokenizer"
    );

    describe_counter!(
        "sgl_router_http_requests_total",
        "Total number of HTTP requests"
    );
    describe_counter!(
        "sgl_router_http_responses_total",
        "Total number of HTTP responses by status code"
    );
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

pub struct TokenizerMetrics;

impl RouterMetrics {
    pub fn record_request(route: &str) {
        counter!("sgl_router_requests_total",
            "route" => route.to_string()
        )
        .increment(1);
    }

    pub fn record_request_duration(duration: Duration) {
        histogram!("sgl_router_request_duration_seconds").record(duration.as_secs_f64());
    }

    pub fn record_request_error(route: &str, error_type: &str) {
        counter!("sgl_router_request_errors_total",
            "route" => route.to_string(),
            "error_type" => error_type.to_string()
        )
        .increment(1);
    }

    // TODO unify metric names
    pub fn record_upstream_http_response(route: &str, status_code: u16) {
        counter!("sgl_router_upstream_http_responses_total",
            "route" => route.to_string(),
            "status_code" => status_code.to_string()
        )
        .increment(1);
    }

    pub fn record_retry(route: &str) {
        counter!("sgl_router_retries_total",
            "route" => route.to_string()
        )
        .increment(1);
    }

    pub fn record_retry_backoff_duration(duration: Duration, attempt: u32) {
        histogram!("sgl_router_retry_backoff_duration_seconds",
            "attempt" => attempt.to_string()
        )
        .record(duration.as_secs_f64());
    }

    pub fn record_retries_exhausted(route: &str) {
        counter!("sgl_router_retries_exhausted_total",
            "route" => route.to_string()
        )
        .increment(1);
    }

    pub fn set_worker_health(worker_url: &str, healthy: bool) {
        gauge!("sgl_router_worker_health",
            "worker" => worker_url.to_string()
        )
        .set(if healthy { 1.0 } else { 0.0 });
    }

    pub fn record_processed_request(worker_url: &str) {
        counter!("sgl_router_processed_requests_total",
            "worker" => worker_url.to_string()
        )
        .increment(1);
    }

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

    pub fn record_cb_state_transition(worker: &str, from: &str, to: &str) {
        counter!("sgl_router_cb_state_transitions_total",
            "worker" => worker.to_string(),
            "from" => from.to_string(),
            "to" => to.to_string()
        )
        .increment(1);
    }

    pub fn record_cb_outcome(worker: &str, outcome: &str) {
        counter!("sgl_router_cb_outcomes_total",
            "worker" => worker.to_string(),
            "outcome" => outcome.to_string()
        )
        .increment(1);
    }

    // TODO delete the metrics (instead of setting them to zero)
    pub fn remove_worker_metrics(worker_url: &str) {
        gauge!("sgl_router_cb_state","worker" => worker_url.to_string()).set(0.0);
        gauge!("sgl_router_worker_health","worker" => worker_url.to_string()).set(0.0);
        gauge!("sgl_router_running_requests","worker" => worker_url.to_string()).set(0.0);
        gauge!("sgl_router_tree_size","worker" => worker_url.to_string()).set(0.0);
    }

    pub fn set_job_queue_depth(depth: usize) {
        gauge!("sgl_router_job_queue_depth").set(depth as f64);
    }

    pub fn record_job_duration(job_type: &str, duration: Duration) {
        histogram!("sgl_router_job_duration_seconds",
            "job_type" => job_type.to_string()
        )
        .record(duration.as_secs_f64());
    }

    pub fn record_job_success(job_type: &str) {
        counter!("sgl_router_job_success_total",
            "job_type" => job_type.to_string()
        )
        .increment(1);
    }

    pub fn record_job_failure(job_type: &str) {
        counter!("sgl_router_job_failure_total",
            "job_type" => job_type.to_string()
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

    pub fn record_http_status_code(status_code: u16) {
        counter!("sgl_router_http_responses_total",
            "status_code" => status_code.to_string()
        )
        .increment(1);
    }
}

impl TokenizerMetrics {
    pub fn record_encode_request(tokenizer_type: &str) {
        counter!("sgl_tokenizer_encode_requests_total",
            "tokenizer_type" => tokenizer_type.to_string()
        )
        .increment(1);
    }

    pub fn record_encode_duration(duration: Duration) {
        histogram!("sgl_tokenizer_encode_duration_seconds").record(duration.as_secs_f64());
    }

    pub fn record_encode_error(error_type: &str) {
        counter!("sgl_tokenizer_encode_errors_total",
            "error_type" => error_type.to_string()
        )
        .increment(1);
    }

    pub fn record_tokens_per_encode(token_count: usize) {
        histogram!("sgl_tokenizer_tokens_per_encode").record(token_count as f64);
    }

    pub fn record_chars_per_encode(char_count: usize) {
        histogram!("sgl_tokenizer_chars_per_encode").record(char_count as f64);
    }

    pub fn record_decode_request(tokenizer_type: &str) {
        counter!("sgl_tokenizer_decode_requests_total",
            "tokenizer_type" => tokenizer_type.to_string()
        )
        .increment(1);
    }

    pub fn record_decode_duration(duration: Duration) {
        histogram!("sgl_tokenizer_decode_duration_seconds").record(duration.as_secs_f64());
    }

    pub fn record_decode_error(error_type: &str) {
        counter!("sgl_tokenizer_decode_errors_total",
            "error_type" => error_type.to_string()
        )
        .increment(1);
    }

    pub fn record_tokens_per_decode(token_count: usize) {
        histogram!("sgl_tokenizer_tokens_per_decode").record(token_count as f64);
    }

    pub fn record_encode_batch_duration(duration: Duration, batch_size: usize) {
        histogram!("sgl_tokenizer_encode_batch_duration_seconds",
            "batch_size" => batch_size.to_string()
        )
        .record(duration.as_secs_f64());
    }

    pub fn record_stop_sequence_detected(stop_type: &str) {
        counter!("sgl_tokenizer_stop_sequences_detected_total",
            "type" => stop_type.to_string()
        )
        .increment(1);
    }

    pub fn record_partial_match() {
        counter!("sgl_tokenizer_partial_matches_total").increment(1);
    }

    pub fn record_stop_detection_duration(duration: Duration) {
        histogram!("sgl_tokenizer_stop_detection_duration_seconds").record(duration.as_secs_f64());
    }

    pub fn record_stream_token() {
        counter!("sgl_tokenizer_stream_tokens_total").increment(1);
    }

    pub fn record_incomplete_utf8() {
        counter!("sgl_tokenizer_stream_incomplete_utf8_total").increment(1);
    }

    pub fn record_stream_step_duration(duration: Duration) {
        histogram!("sgl_tokenizer_stream_step_duration_seconds").record(duration.as_secs_f64());
    }

    pub fn record_factory_load(file_type: &str) {
        counter!("sgl_tokenizer_factory_loads_total",
            "file_type" => file_type.to_string()
        )
        .increment(1);
    }

    pub fn record_factory_error(error_type: &str) {
        counter!("sgl_tokenizer_factory_errors_total",
            "error_type" => error_type.to_string()
        )
        .increment(1);
    }

    pub fn record_factory_load_duration(duration: Duration) {
        histogram!("sgl_tokenizer_factory_load_duration_seconds").record(duration.as_secs_f64());
    }

    pub fn set_vocab_size(tokenizer_type: &str, size: usize) {
        gauge!("sgl_tokenizer_vocab_size",
            "tokenizer_type" => tokenizer_type.to_string()
        )
        .set(size as f64);
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
    fn test_tokenizer_metrics_static_methods() {
        TokenizerMetrics::record_encode_request("huggingface");
        TokenizerMetrics::record_encode_duration(Duration::from_millis(10));
        TokenizerMetrics::record_encode_error("invalid_input");
        TokenizerMetrics::record_tokens_per_encode(100);
        TokenizerMetrics::record_chars_per_encode(500);

        TokenizerMetrics::record_decode_request("huggingface");
        TokenizerMetrics::record_decode_duration(Duration::from_millis(5));
        TokenizerMetrics::record_decode_error("invalid_tokens");
        TokenizerMetrics::record_tokens_per_decode(50);

        TokenizerMetrics::record_encode_batch_duration(Duration::from_millis(100), 10);

        TokenizerMetrics::record_stop_sequence_detected("token");
        TokenizerMetrics::record_stop_sequence_detected("string");
        TokenizerMetrics::record_partial_match();
        TokenizerMetrics::record_stop_detection_duration(Duration::from_micros(100));

        TokenizerMetrics::record_stream_token();
        TokenizerMetrics::record_incomplete_utf8();
        TokenizerMetrics::record_stream_step_duration(Duration::from_micros(50));

        TokenizerMetrics::record_factory_load("json");
        TokenizerMetrics::record_factory_error("unsupported_format");
        TokenizerMetrics::record_factory_load_duration(Duration::from_millis(200));

        TokenizerMetrics::set_vocab_size("huggingface", 50000);
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

        RouterMetrics::record_request(&long_label);
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
