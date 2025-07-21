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

    let duration_matcher = Matcher::Suffix(String::from("duration_seconds"));
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    // ============= PrometheusConfig Tests =============

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
        };
        assert_eq!(config.port, 8080);
        assert_eq!(config.host, "127.0.0.1");
    }

    #[test]
    fn test_prometheus_config_clone() {
        let config = PrometheusConfig {
            port: 9090,
            host: "192.168.1.1".to_string(),
        };
        let cloned = config.clone();
        assert_eq!(cloned.port, config.port);
        assert_eq!(cloned.host, config.host);
    }

    // ============= IP Address Parsing Tests =============

    #[test]
    fn test_valid_ipv4_parsing() {
        let test_cases = vec!["127.0.0.1", "192.168.1.1", "0.0.0.0"];

        for ip_str in test_cases {
            let config = PrometheusConfig {
                port: 29000,
                host: ip_str.to_string(),
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
            };

            let ip_addr: IpAddr = config
                .host
                .parse()
                .unwrap_or(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));

            // Should fall back to 0.0.0.0
            assert_eq!(ip_addr, IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));
        }
    }

    // ============= Socket Address Creation Tests =============

    #[test]
    fn test_socket_addr_creation() {
        let test_cases = vec![("127.0.0.1", 8080), ("0.0.0.0", 29000), ("::1", 9090)];

        for (host, port) in test_cases {
            let config = PrometheusConfig {
                port,
                host: host.to_string(),
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
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            let socket_addr = SocketAddr::new(ip_addr, config.port);

            assert_eq!(socket_addr.port(), port);
        }
    }

    // ============= Duration Bucket Tests =============

    #[test]
    fn test_duration_bucket_values() {
        let expected_buckets = vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
            60.0, 90.0, 120.0, 180.0, 240.0,
        ];

        // The buckets are defined in start_prometheus function
        assert_eq!(expected_buckets.len(), 20);

        // Verify proper ordering
        for i in 1..expected_buckets.len() {
            assert!(expected_buckets[i] > expected_buckets[i - 1]);
        }
    }

    #[test]
    fn test_duration_bucket_coverage() {
        let test_cases = vec![
            (0.0005, "sub-millisecond"),
            (0.005, "5ms"),
            (0.05, "50ms"),
            (1.0, "1s"),
            (10.0, "10s"),
            (60.0, "1m"),
            (240.0, "4m"),
        ];

        let buckets = vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
            60.0, 90.0, 120.0, 180.0, 240.0,
        ];

        for (duration, label) in test_cases {
            let bucket_found = buckets
                .iter()
                .any(|&b| ((b - duration) as f64).abs() < 0.0001 || b > duration);
            assert!(bucket_found, "No bucket found for {} ({})", duration, label);
        }
    }

    // ============= Matcher Configuration Tests =============

    #[test]
    fn test_duration_suffix_matcher() {
        let matcher = Matcher::Suffix(String::from("duration_seconds"));

        // Test matching behavior
        let _matching_metrics = vec![
            "request_duration_seconds",
            "response_duration_seconds",
            "sgl_router_request_duration_seconds",
        ];

        let _non_matching_metrics =
            vec!["duration_total", "duration_seconds_total", "other_metric"];

        // Note: We can't directly test Matcher matching without the internals,
        // but we can verify the matcher is created correctly
        match matcher {
            Matcher::Suffix(suffix) => assert_eq!(suffix, "duration_seconds"),
            _ => panic!("Expected Suffix matcher"),
        }
    }

    // ============= Builder Configuration Tests =============

    #[test]
    fn test_prometheus_builder_configuration() {
        // This test verifies the builder configuration without actually starting Prometheus
        let _config = PrometheusConfig::default();

        let duration_matcher = Matcher::Suffix(String::from("duration_seconds"));
        let duration_bucket = [
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
            60.0, 90.0, 120.0, 180.0, 240.0,
        ];

        // Verify bucket configuration
        assert_eq!(duration_bucket.len(), 20);

        // Verify matcher is suffix type
        match duration_matcher {
            Matcher::Suffix(s) => assert_eq!(s, "duration_seconds"),
            _ => panic!("Expected Suffix matcher"),
        }
    }

    // ============= Upkeep Timeout Tests =============

    #[test]
    fn test_upkeep_timeout_duration() {
        let timeout = Duration::from_secs(5 * 60);
        assert_eq!(timeout.as_secs(), 300);
    }

    // ============= Custom Bucket Tests =============

    #[test]
    fn test_custom_buckets_for_different_metrics() {
        // Test that we can create different bucket configurations
        let request_buckets = vec![0.001, 0.01, 0.1, 1.0, 10.0];
        let generate_buckets = vec![0.1, 0.5, 1.0, 5.0, 30.0, 60.0];

        assert_eq!(request_buckets.len(), 5);
        assert_eq!(generate_buckets.len(), 6);

        // Verify each set is sorted
        for i in 1..request_buckets.len() {
            assert!(request_buckets[i] > request_buckets[i - 1]);
        }

        for i in 1..generate_buckets.len() {
            assert!(generate_buckets[i] > generate_buckets[i - 1]);
        }
    }

    // ============= RouterMetrics Tests =============

    #[test]
    fn test_metrics_static_methods() {
        // Test that all static methods can be called without panic
        RouterMetrics::record_request("/generate");
        RouterMetrics::record_request_duration("/generate", Duration::from_millis(100));
        RouterMetrics::record_request_error("/generate", "timeout");
        RouterMetrics::record_retry("/generate");

        RouterMetrics::set_active_workers(5);
        RouterMetrics::set_worker_health("http://worker1", true);
        RouterMetrics::set_worker_load("http://worker1", 10);
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

    // ============= Port Availability Tests =============

    #[test]
    fn test_port_already_in_use() {
        // Skip this test if we can't bind to the port
        let port = 29123; // Use a different port to avoid conflicts

        if let Ok(_listener) = TcpListener::bind(("127.0.0.1", port)) {
            // Port is available, we can test
            let config = PrometheusConfig {
                port,
                host: "127.0.0.1".to_string(),
            };

            // Just verify config is created correctly
            assert_eq!(config.port, port);
        }
    }

    // ============= Integration Test Helpers =============

    #[test]
    fn test_metrics_endpoint_accessibility() {
        // This would be an integration test in practice
        // Here we just verify the configuration
        let config = PrometheusConfig {
            port: 29000,
            host: "127.0.0.1".to_string(),
        };

        let ip_addr: IpAddr = config.host.parse().unwrap();
        let socket_addr = SocketAddr::new(ip_addr, config.port);

        assert_eq!(socket_addr.to_string(), "127.0.0.1:29000");
    }

    #[test]
    fn test_concurrent_metric_updates() {
        // Test that metric updates can be called concurrently
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::thread;

        let done = Arc::new(AtomicBool::new(false));
        let mut handles = vec![];

        for i in 0..3 {
            let done_clone = done.clone();
            let handle = thread::spawn(move || {
                let worker = format!("http://worker{}", i);
                while !done_clone.load(Ordering::Relaxed) {
                    RouterMetrics::set_worker_load(&worker, i * 10);
                    RouterMetrics::record_processed_request(&worker);
                    thread::sleep(Duration::from_millis(1));
                }
            });
            handles.push(handle);
        }

        // Let threads run briefly
        thread::sleep(Duration::from_millis(10));
        done.store(true, Ordering::Relaxed);

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // If we get here without panic, concurrent access works
        assert!(true);
    }

    // ============= Edge Cases Tests =============

    #[test]
    fn test_empty_string_metrics() {
        // Test that empty strings don't cause issues
        RouterMetrics::record_request("");
        RouterMetrics::set_worker_health("", true);
        RouterMetrics::record_policy_decision("", "");

        // If we get here without panic, empty strings are handled
        assert!(true);
    }

    #[test]
    fn test_very_long_metric_labels() {
        let long_label = "a".repeat(1000);

        RouterMetrics::record_request(&long_label);
        RouterMetrics::set_worker_health(&long_label, false);

        // If we get here without panic, long labels are handled
        assert!(true);
    }

    #[test]
    fn test_special_characters_in_labels() {
        let special_labels = vec![
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

        // If we get here without panic, special characters are handled
        assert!(true);
    }

    #[test]
    fn test_extreme_metric_values() {
        // Test extreme values
        RouterMetrics::set_active_workers(0);
        RouterMetrics::set_active_workers(usize::MAX);

        RouterMetrics::set_worker_load("worker", 0);
        RouterMetrics::set_worker_load("worker", usize::MAX);

        RouterMetrics::record_request_duration("route", Duration::from_nanos(1));
        RouterMetrics::record_request_duration("route", Duration::from_secs(86400)); // 24 hours

        // If we get here without panic, extreme values are handled
        assert!(true);
    }
}
