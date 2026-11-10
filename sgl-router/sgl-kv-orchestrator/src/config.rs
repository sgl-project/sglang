use crate::session_index::SessionLimits;
use serde::Deserialize;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrchestrationConfig {
    /// Trusted single-tenant caller, never accepted from a request body.
    pub caller_id: String,
    /// Required on lifecycle and control routes. Supply through a protected config file.
    pub control_bearer: String,
    pub queue_capacity: usize,
    pub max_operations: usize,
    pub max_workers: usize,
    pub max_engine_bytes: usize,
    pub rpc_timeout_ms: u64,
    pub poll_ms: u64,
    pub operation_ttl_ms: u64,
    pub limits: SessionLimits,
}
impl std::fmt::Debug for OrchestrationConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrchestrationConfig")
            .field("caller_id", &self.caller_id)
            .field("queue_capacity", &self.queue_capacity)
            .field("max_operations", &self.max_operations)
            .finish_non_exhaustive()
    }
}
impl OrchestrationConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.caller_id.is_empty()
                && self.caller_id.len() <= 128
                && !self.control_bearer.is_empty(),
            "orchestration caller/bearer required"
        );
        anyhow::ensure!(
            self.queue_capacity > 0
                && self.max_operations > 0
                && self.max_workers > 0
                && self.max_engine_bytes > 0
                && self.rpc_timeout_ms > 0
                && self.poll_ms > 0
                && self.operation_ttl_ms > 0
                && self.operation_ttl_ms <= 86_400_000
                && self.rpc_timeout_ms <= 86_400_000
                && self.poll_ms <= self.operation_ttl_ms,
            "orchestration budgets must be positive"
        );
        anyhow::ensure!(
            self.limits.max_identifier_bytes >= 128 && self.limits.max_identifier_bytes <= 256,
            "orchestration identifiers require 128..256 bytes"
        );
        Ok(())
    }
}
