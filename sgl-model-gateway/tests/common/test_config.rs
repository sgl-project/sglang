//! Test configuration builders to reduce duplication across tests
//!
//! Provides pre-configured RouterConfig and MockWorkerConfig builders
//! for common test scenarios.

use smg::config::{
    CircuitBreakerConfig, ManualAssignmentMode, PolicyConfig, RetryConfig, RouterConfig,
};

use super::mock_worker::{HealthStatus, MockWorkerConfig, WorkerType};

/// Default test configuration values
pub mod defaults {
    pub const HOST: &str = "127.0.0.1";
    pub const MAX_PAYLOAD_SIZE: usize = 256 * 1024 * 1024; // 256MB
    pub const REQUEST_TIMEOUT_SECS: u64 = 600;
    pub const WORKER_STARTUP_TIMEOUT_SECS: u64 = 5;
    pub const WORKER_STARTUP_CHECK_INTERVAL_SECS: u64 = 1;
    pub const MAX_CONCURRENT_REQUESTS: i32 = 64;
    pub const QUEUE_TIMEOUT_SECS: u64 = 60;
}

/// Builder for common test RouterConfig patterns
pub struct TestRouterConfig;

impl TestRouterConfig {
    /// Create a basic round-robin config for routing tests
    pub fn round_robin(port: u16) -> RouterConfig {
        RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host(defaults::HOST)
            .port(port)
            .max_payload_size(defaults::MAX_PAYLOAD_SIZE)
            .request_timeout_secs(defaults::REQUEST_TIMEOUT_SECS)
            .worker_startup_timeout_secs(defaults::WORKER_STARTUP_TIMEOUT_SECS)
            .worker_startup_check_interval_secs(defaults::WORKER_STARTUP_CHECK_INTERVAL_SECS)
            .max_concurrent_requests(defaults::MAX_CONCURRENT_REQUESTS)
            .queue_timeout_secs(defaults::QUEUE_TIMEOUT_SECS)
            .build_unchecked()
    }

    /// Create a random load balancing config
    pub fn random(port: u16) -> RouterConfig {
        RouterConfig::builder()
            .regular_mode(vec![])
            .random_policy()
            .host(defaults::HOST)
            .port(port)
            .max_payload_size(defaults::MAX_PAYLOAD_SIZE)
            .request_timeout_secs(defaults::REQUEST_TIMEOUT_SECS)
            .worker_startup_timeout_secs(defaults::WORKER_STARTUP_TIMEOUT_SECS)
            .worker_startup_check_interval_secs(defaults::WORKER_STARTUP_CHECK_INTERVAL_SECS)
            .max_concurrent_requests(defaults::MAX_CONCURRENT_REQUESTS)
            .queue_timeout_secs(defaults::QUEUE_TIMEOUT_SECS)
            .build_unchecked()
    }

    /// Create a cache-aware config for routing tests
    pub fn cache_aware(port: u16) -> RouterConfig {
        RouterConfig::builder()
            .regular_mode(vec![])
            .cache_aware_policy(
                0.5,  // cache_threshold
                32,   // balance_abs_threshold
                1.5,  // balance_rel_threshold
                60,   // eviction_interval_secs
                1000, // max_tree_size
            )
            .host(defaults::HOST)
            .port(port)
            .max_payload_size(defaults::MAX_PAYLOAD_SIZE)
            .request_timeout_secs(defaults::REQUEST_TIMEOUT_SECS)
            .worker_startup_timeout_secs(defaults::WORKER_STARTUP_TIMEOUT_SECS)
            .worker_startup_check_interval_secs(defaults::WORKER_STARTUP_CHECK_INTERVAL_SECS)
            .max_concurrent_requests(defaults::MAX_CONCURRENT_REQUESTS)
            .queue_timeout_secs(defaults::QUEUE_TIMEOUT_SECS)
            .build_unchecked()
    }

    /// Create a power-of-two config
    pub fn power_of_two(port: u16) -> RouterConfig {
        RouterConfig::builder()
            .regular_mode(vec![])
            .power_of_two_policy(5) // load_check_interval_secs
            .host(defaults::HOST)
            .port(port)
            .max_payload_size(defaults::MAX_PAYLOAD_SIZE)
            .request_timeout_secs(defaults::REQUEST_TIMEOUT_SECS)
            .worker_startup_timeout_secs(defaults::WORKER_STARTUP_TIMEOUT_SECS)
            .worker_startup_check_interval_secs(defaults::WORKER_STARTUP_CHECK_INTERVAL_SECS)
            .max_concurrent_requests(defaults::MAX_CONCURRENT_REQUESTS)
            .queue_timeout_secs(defaults::QUEUE_TIMEOUT_SECS)
            .build_unchecked()
    }

    /// Create a manual routing config (for sticky routing tests)
    pub fn manual(port: u16) -> RouterConfig {
        Self::manual_with_mode(port, ManualAssignmentMode::Random)
    }

    /// Create a manual routing config with min_group assignment mode
    pub fn manual_min_group(port: u16) -> RouterConfig {
        Self::manual_with_mode(port, ManualAssignmentMode::MinGroup)
    }

    /// Create a manual routing config with specified assignment mode
    pub fn manual_with_mode(port: u16, assignment_mode: ManualAssignmentMode) -> RouterConfig {
        RouterConfig::builder()
            .regular_mode(vec![])
            .policy(PolicyConfig::Manual {
                eviction_interval_secs: 60,
                max_idle_secs: 3600,
                assignment_mode,
            })
            .host(defaults::HOST)
            .port(port)
            .max_payload_size(defaults::MAX_PAYLOAD_SIZE)
            .request_timeout_secs(defaults::REQUEST_TIMEOUT_SECS)
            .worker_startup_timeout_secs(defaults::WORKER_STARTUP_TIMEOUT_SECS)
            .worker_startup_check_interval_secs(defaults::WORKER_STARTUP_CHECK_INTERVAL_SECS)
            .max_concurrent_requests(defaults::MAX_CONCURRENT_REQUESTS)
            .queue_timeout_secs(defaults::QUEUE_TIMEOUT_SECS)
            .build_unchecked()
    }

    /// Create a config with custom concurrent request limit (for rate limiting tests)
    pub fn with_concurrency(port: u16, max_concurrent: i32) -> RouterConfig {
        RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host(defaults::HOST)
            .port(port)
            .max_payload_size(defaults::MAX_PAYLOAD_SIZE)
            .request_timeout_secs(defaults::REQUEST_TIMEOUT_SECS)
            .worker_startup_timeout_secs(defaults::WORKER_STARTUP_TIMEOUT_SECS)
            .worker_startup_check_interval_secs(defaults::WORKER_STARTUP_CHECK_INTERVAL_SECS)
            .max_concurrent_requests(max_concurrent)
            .queue_timeout_secs(defaults::QUEUE_TIMEOUT_SECS)
            .build_unchecked()
    }

    /// Create a config with custom payload size limit
    pub fn with_payload_limit(port: u16, max_payload_size: usize) -> RouterConfig {
        RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host(defaults::HOST)
            .port(port)
            .max_payload_size(max_payload_size)
            .request_timeout_secs(defaults::REQUEST_TIMEOUT_SECS)
            .worker_startup_timeout_secs(defaults::WORKER_STARTUP_TIMEOUT_SECS)
            .worker_startup_check_interval_secs(defaults::WORKER_STARTUP_CHECK_INTERVAL_SECS)
            .max_concurrent_requests(defaults::MAX_CONCURRENT_REQUESTS)
            .queue_timeout_secs(defaults::QUEUE_TIMEOUT_SECS)
            .build_unchecked()
    }

    /// Create a config with short timeouts (for timeout/retry tests)
    pub fn with_short_timeouts(port: u16) -> RouterConfig {
        RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host(defaults::HOST)
            .port(port)
            .max_payload_size(defaults::MAX_PAYLOAD_SIZE)
            .request_timeout_secs(5)
            .worker_startup_timeout_secs(2)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(defaults::MAX_CONCURRENT_REQUESTS)
            .queue_timeout_secs(5)
            .build_unchecked()
    }

    /// Create a round-robin config with retry settings
    pub fn round_robin_with_retry(port: u16, retry_config: RetryConfig) -> RouterConfig {
        RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host(defaults::HOST)
            .port(port)
            .max_payload_size(defaults::MAX_PAYLOAD_SIZE)
            .request_timeout_secs(defaults::REQUEST_TIMEOUT_SECS)
            .worker_startup_timeout_secs(defaults::WORKER_STARTUP_TIMEOUT_SECS)
            .worker_startup_check_interval_secs(defaults::WORKER_STARTUP_CHECK_INTERVAL_SECS)
            .max_concurrent_requests(defaults::MAX_CONCURRENT_REQUESTS)
            .queue_timeout_secs(defaults::QUEUE_TIMEOUT_SECS)
            .retry_config(retry_config)
            .build_unchecked()
    }

    /// Create a round-robin config with circuit breaker
    pub fn round_robin_with_circuit_breaker(
        port: u16,
        circuit_breaker: CircuitBreakerConfig,
    ) -> RouterConfig {
        RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host(defaults::HOST)
            .port(port)
            .max_payload_size(defaults::MAX_PAYLOAD_SIZE)
            .request_timeout_secs(defaults::REQUEST_TIMEOUT_SECS)
            .worker_startup_timeout_secs(defaults::WORKER_STARTUP_TIMEOUT_SECS)
            .worker_startup_check_interval_secs(defaults::WORKER_STARTUP_CHECK_INTERVAL_SECS)
            .max_concurrent_requests(defaults::MAX_CONCURRENT_REQUESTS)
            .queue_timeout_secs(defaults::QUEUE_TIMEOUT_SECS)
            .circuit_breaker_config(circuit_breaker)
            .build_unchecked()
    }

    /// Create a round-robin config with both retry and circuit breaker
    pub fn round_robin_with_reliability(
        port: u16,
        retry_config: RetryConfig,
        circuit_breaker: CircuitBreakerConfig,
    ) -> RouterConfig {
        RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host(defaults::HOST)
            .port(port)
            .max_payload_size(defaults::MAX_PAYLOAD_SIZE)
            .request_timeout_secs(defaults::REQUEST_TIMEOUT_SECS)
            .worker_startup_timeout_secs(defaults::WORKER_STARTUP_TIMEOUT_SECS)
            .worker_startup_check_interval_secs(defaults::WORKER_STARTUP_CHECK_INTERVAL_SECS)
            .max_concurrent_requests(defaults::MAX_CONCURRENT_REQUESTS)
            .queue_timeout_secs(defaults::QUEUE_TIMEOUT_SECS)
            .retry_config(retry_config)
            .circuit_breaker_config(circuit_breaker)
            .build_unchecked()
    }
}

/// Builder for common MockWorkerConfig patterns
pub struct TestWorkerConfig;

impl TestWorkerConfig {
    /// Create a healthy worker config
    pub fn healthy(port: u16) -> MockWorkerConfig {
        MockWorkerConfig {
            port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }
    }

    /// Create multiple healthy workers with sequential ports
    pub fn healthy_workers(start_port: u16, count: u16) -> Vec<MockWorkerConfig> {
        (0..count).map(|i| Self::healthy(start_port + i)).collect()
    }

    /// Create an unhealthy worker config
    pub fn unhealthy(port: u16) -> MockWorkerConfig {
        MockWorkerConfig {
            port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Unhealthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }
    }

    /// Create a slow worker config (for timeout tests)
    pub fn slow(port: u16, delay_ms: u64) -> MockWorkerConfig {
        MockWorkerConfig {
            port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: delay_ms,
            fail_rate: 0.0,
        }
    }

    /// Create multiple slow workers with sequential ports
    pub fn slow_workers(start_port: u16, count: u16, delay_ms: u64) -> Vec<MockWorkerConfig> {
        (0..count)
            .map(|i| Self::slow(start_port + i, delay_ms))
            .collect()
    }

    /// Create a flaky worker config (for retry/fault tolerance tests)
    pub fn flaky(port: u16, fail_rate: f32) -> MockWorkerConfig {
        MockWorkerConfig {
            port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate,
        }
    }

    /// Create a decode worker config (for PD routing tests)
    pub fn decode(port: u16) -> MockWorkerConfig {
        MockWorkerConfig {
            port,
            worker_type: WorkerType::Decode,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }
    }

    /// Create a prefill worker config (for PD routing tests)
    pub fn prefill(port: u16) -> MockWorkerConfig {
        MockWorkerConfig {
            port,
            worker_type: WorkerType::Prefill,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_robin_config() {
        let config = TestRouterConfig::round_robin(3000);
        assert_eq!(config.port, 3000);
    }

    #[test]
    fn test_healthy_workers() {
        let workers = TestWorkerConfig::healthy_workers(8000, 3);
        assert_eq!(workers.len(), 3);
        assert_eq!(workers[0].port, 8000);
        assert_eq!(workers[1].port, 8001);
        assert_eq!(workers[2].port, 8002);
    }
}
