//! Optional single-owner Router orchestration. Normal inference uses try_send;
//! only the background actor performs discovery, Engine control and reconciliation.
mod client;
mod feedback;
mod runtime;
pub use runtime::{KvContext, Runtime};
use serde::Deserialize;
use sgl_kv_orchestrator::session_index::SessionLimits;

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
    pub initial_prefetch_ms_per_token: f64,
    pub lead_margin_ms: u64,
    pub max_resume_prefetch_ms: u64,
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
                && self.poll_ms <= self.operation_ttl_ms
                && self.lead_margin_ms <= 86_400_000
                && self.max_resume_prefetch_ms <= 86_400_000,
            "orchestration budgets must be positive"
        );
        anyhow::ensure!(
            self.initial_prefetch_ms_per_token.is_finite()
                && self.initial_prefetch_ms_per_token > 0.0,
            "initial prefetch cost must be finite and positive"
        );
        anyhow::ensure!(
            self.limits.max_identifier_bytes >= 128 && self.limits.max_identifier_bytes <= 256,
            "orchestration identifiers require 128..256 bytes"
        );
        Ok(())
    }
}

/// Adds recent scoped preparation evidence to cache-policy inputs. Existing
/// routing policies keep admission, affinity, load balancing and the final pick.
/// Counts are normalized to tokens because provider block sizes may differ.
pub fn merge_prepared(
    original: Option<&crate::state::kv_events::PrefixLookupResult>,
    prepared: &[(String, u64)],
    tokens: usize,
) -> Option<crate::state::kv_events::PrefixLookupResult> {
    use crate::state::kv_events::PrefixLookupResult;
    use sgl_kv_indexer::{PrefixMatch, PrefixOutcome};
    if prepared.is_empty() {
        return original.map(|signal| PrefixLookupResult {
            outcome: signal.outcome.clone(),
            query_blocks: signal.query_blocks,
            pending_winners: signal.pending_winners.clone(),
            block_hashes: signal.block_hashes.clone(),
        });
    }
    let mut depths = std::collections::BTreeMap::<String, u32>::new();
    if let Some(signal) = original {
        if let PrefixOutcome::Matched { matches, .. } = &signal.outcome {
            for entry in matches {
                let n = (entry.matched_prefix_blocks as u64).saturating_mul(tokens as u64)
                    / (signal.query_blocks.max(1) as u64);
                depths.insert(entry.address.clone(), n.min(tokens as u64) as u32);
            }
        }
    }
    for (url, end) in prepared {
        let depth = depths.entry(url.clone()).or_default();
        *depth = (*depth).max((*end).min(tokens as u64) as u32);
    }
    if depths.is_empty() {
        return None;
    }
    let best_prefix_blocks = *depths.values().max().expect("nonempty");
    Some(PrefixLookupResult {
        outcome: PrefixOutcome::Matched {
            best_prefix_blocks,
            matches: depths
                .into_iter()
                .map(|(address, matched_prefix_blocks)| PrefixMatch {
                    worker_id: address.clone(),
                    address,
                    matched_prefix_blocks,
                })
                .collect(),
        },
        query_blocks: tokens,
        pending_winners: original.map_or_else(Vec::new, |s| s.pending_winners.clone()),
        block_hashes: original.and_then(|s| s.block_hashes.clone()),
    })
}
