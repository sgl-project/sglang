// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Prefill on the decode worker that already holds the prompt's prefix.
//!
//! In multi-turn traffic the decode worker that served the previous turn holds
//! that turn's prompt and output in its radix cache, so the next turn leaves
//! only a short suffix to prefill. Sending it through a prefill worker costs a
//! bootstrap handshake and a KV transfer for a few hundred tokens of compute.
//! With `--pd-local-prefill-max-uncached-tokens N`, a request whose uncached
//! suffix on some decode worker is at most N tokens goes to that worker alone,
//! marked `disagg_local_prefill`, and the worker prefills the suffix itself.

use super::preparation::PreparedRequest;
use crate::discovery::WorkerMode;
use crate::policies::registry::PdPoolResolver;
use crate::policies::PrefixLookupResult;
use crate::server::app_context::AppContext;
use crate::server::metrics::PdLocalPrefillOutcome;
use crate::workers::Worker;
use sgl_kv_indexer::{PrefixMatch, PrefixOutcome};
use std::sync::Arc;

/// The decode worker to send `request` to alone, if one holds enough of its
/// prompt. Records the decision in `sgl_router_pd_local_prefill_total`.
pub(super) fn pick_decode_worker(
    ctx: &AppContext,
    request: &PreparedRequest,
    resolver: &PdPoolResolver,
    prefix: Option<&PrefixLookupResult>,
) -> Option<Arc<Worker>> {
    let max_uncached_tokens = ctx.config.proxy.local_prefill_max_uncached_tokens?;
    // A batch has no single prompt to match, and fan-out items each need a
    // bootstrap room of their own.
    if request.fans_out || request.input_token_count != request.sequence_token_count {
        return None;
    }
    let block_size = u64::from(ctx.block_size_oracle.get()?);
    let matches: &[PrefixMatch] = match prefix.map(|signal| &signal.outcome) {
        Some(PrefixOutcome::Matched { matches, .. }) => matches,
        _ => &[],
    };
    let decode_workers = resolver
        .decode_candidates(&request.model)
        .unwrap_or_default();
    let picked = pick(&LocalPrefillInputs {
        max_uncached_tokens,
        prompt_tokens: request.sequence_token_count as u64,
        block_size,
        matches,
        decode_workers: &decode_workers,
    });
    ctx.metrics.record_pd_local_prefill(match &picked {
        Ok(_) => PdLocalPrefillOutcome::Routed,
        Err(outcome) => *outcome,
    });
    picked.ok()
}

pub(super) struct LocalPrefillInputs<'a> {
    pub max_uncached_tokens: u64,
    /// The prompt's length; a prefix match never covers more than this.
    pub prompt_tokens: u64,
    /// Tokens per KV block, so a match depth in blocks becomes tokens.
    pub block_size: u64,
    /// Prefix depths by worker address, from the local KV-event tree.
    pub matches: &'a [PrefixMatch],
    /// The healthy decode pool; addresses are compared with worker URLs.
    pub decode_workers: &'a [Arc<Worker>],
}

/// The decode worker holding the deepest prefix whose uncached suffix is within
/// the limit, or why none qualifies; a worker whose breaker refuses does not count.
pub(super) fn pick(inputs: &LocalPrefillInputs<'_>) -> Result<Arc<Worker>, PdLocalPrefillOutcome> {
    let mut best: Option<(u64, &Arc<Worker>)> = None;
    let mut over_limit = false;
    for found in inputs.matches {
        let Some(worker) = inputs.decode_workers.iter().find(|worker| {
            worker.mode() == WorkerMode::Decode
                && worker.url == found.address
                && worker.breaker.would_allow()
        }) else {
            continue;
        };
        let cached = u64::from(found.matched_prefix_blocks)
            .saturating_mul(inputs.block_size)
            .min(inputs.prompt_tokens);
        if inputs.prompt_tokens - cached > inputs.max_uncached_tokens {
            over_limit = true;
            continue;
        }
        if best.is_none_or(|(best_cached, _)| cached > best_cached) {
            best = Some((cached, worker));
        }
    }
    match best {
        Some((_, worker)) => Ok(Arc::clone(worker)),
        None if over_limit => Err(PdLocalPrefillOutcome::UncachedOverLimit),
        None => Err(PdLocalPrefillOutcome::NoPrefix),
    }
}

#[cfg(test)]
mod tests {
    use super::{pick, LocalPrefillInputs};
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::server::metrics::PdLocalPrefillOutcome;
    use crate::workers::Worker;
    use sgl_kv_indexer::PrefixMatch;
    use std::sync::Arc;

    fn worker(id: &str, mode: WorkerMode) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode,
            model_ids: vec![ModelId("model".into())],
            ..Default::default()
        }))
    }

    fn found(id: &str, blocks: u32) -> PrefixMatch {
        PrefixMatch {
            address: format!("http://{id}:30000"),
            matched_prefix_blocks: blocks,
            worker_id: id.into(),
        }
    }

    #[test]
    fn picks_the_deepest_decode_prefix_within_the_limit() {
        // The prefill worker holds more of the prompt, but only a decode worker
        // can prefill locally; among decode workers the deeper match wins.
        let prefill = worker("p", WorkerMode::Prefill);
        let shallow = worker("d1", WorkerMode::Decode);
        let deep = worker("d2", WorkerMode::Decode);
        let matches = [found("p", 10), found("d1", 6), found("d2", 8)];
        let picked = pick(&LocalPrefillInputs {
            max_uncached_tokens: 200,
            prompt_tokens: 1000,
            block_size: 100,
            matches: &matches,
            decode_workers: &[prefill, Arc::clone(&shallow), Arc::clone(&deep)],
        })
        .expect("d2 leaves 200 uncached tokens");
        assert_eq!(picked.id, deep.id);
    }

    #[test]
    fn a_suffix_over_the_limit_reports_over_limit_not_no_prefix() {
        let decode = worker("d1", WorkerMode::Decode);
        let matches = [found("d1", 6)];
        let outcome = pick(&LocalPrefillInputs {
            max_uncached_tokens: 300,
            prompt_tokens: 1000,
            block_size: 100,
            matches: &matches,
            decode_workers: &[decode],
        })
        .unwrap_err();
        assert_eq!(outcome, PdLocalPrefillOutcome::UncachedOverLimit);
    }

    #[test]
    fn a_prefix_held_only_by_other_workers_is_no_prefix() {
        // A prefill worker's match, or a decode worker that left the pool,
        // cannot serve a local prefill.
        let prefill = worker("p", WorkerMode::Prefill);
        let decode = worker("d1", WorkerMode::Decode);
        let matches = [found("p", 10), found("gone", 10)];
        let outcome = pick(&LocalPrefillInputs {
            max_uncached_tokens: 1000,
            prompt_tokens: 1000,
            block_size: 100,
            matches: &matches,
            decode_workers: &[prefill, decode],
        })
        .unwrap_err();
        assert_eq!(outcome, PdLocalPrefillOutcome::NoPrefix);
    }

    #[test]
    fn a_match_deeper_than_the_prompt_counts_as_fully_cached() {
        // The KV tree reports whole blocks; a depth past the prompt's end must
        // not underflow the uncached count.
        let decode = worker("d1", WorkerMode::Decode);
        let matches = [found("d1", 11)];
        let picked = pick(&LocalPrefillInputs {
            max_uncached_tokens: 0,
            prompt_tokens: 1050,
            block_size: 100,
            matches: &matches,
            decode_workers: &[Arc::clone(&decode)],
        })
        .expect("fully cached");
        assert_eq!(picked.id, decode.id);
    }
}
