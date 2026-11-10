//! Adapter from the Router's worker registry to session KV orchestration.
use std::sync::Arc;

use reqwest::header::HeaderValue;
pub use sgl_kv_orchestrator::{KvContext, OrchestrationConfig, Runtime};

use crate::{
    discovery::{ModelId, WorkerMode},
    workers::WorkerRegistry,
};

pub(crate) fn start(
    config: OrchestrationConfig,
    registry: Arc<WorkerRegistry>,
    model: ModelId,
    auth: Option<HeaderValue>,
) -> Arc<Runtime> {
    Runtime::start(
        config,
        move || {
            let mut workers = registry.healthy_workers_for(&model);
            workers.retain(|w| matches!(w.mode(), WorkerMode::Plain | WorkerMode::Prefill));
            workers.sort_by(|a, b| {
                a.router_inflight_load()
                    .cmp(&b.router_inflight_load())
                    .then_with(|| a.id.0.cmp(&b.id.0))
            });
            workers.into_iter().map(|w| w.url.clone()).collect()
        },
        auth,
    )
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
