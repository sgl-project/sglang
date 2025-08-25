use std::time::Duration;

#[derive(Debug, Clone)]
pub struct MCPConfig {
    pub connection_timeout_ms: u64,
    pub execution_timeout_ms: u64,
    pub max_concurrent_tools: usize,
    pub enable_caching: bool,
    pub cache_ttl_ms: u64,
    pub retry_attempts: u32,
    pub retry_delay_ms: u64,
}

impl MCPConfig {
    pub fn dev_mode() -> Self {
        Self {
            connection_timeout_ms: 5000,
            execution_timeout_ms: 30000,
            max_concurrent_tools: 5,
            enable_caching: false,
            cache_ttl_ms: 60000,
            retry_attempts: 2,
            retry_delay_ms: 1000,
        }
    }

    pub fn production() -> Self {
        Self {
            connection_timeout_ms: 10000,
            execution_timeout_ms: 60000,
            max_concurrent_tools: 10,
            enable_caching: true,
            cache_ttl_ms: 300000,
            retry_attempts: 3,
            retry_delay_ms: 2000,
        }
    }

    pub fn connection_timeout(&self) -> Duration {
        Duration::from_millis(self.connection_timeout_ms)
    }

    pub fn execution_timeout(&self) -> Duration {
        Duration::from_millis(self.execution_timeout_ms)
    }

    pub fn retry_delay(&self) -> Duration {
        Duration::from_millis(self.retry_delay_ms)
    }
}
