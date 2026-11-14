// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::WorkerMode;
use crate::policies::dp_rank::request_rank_load;
use crate::policies::scoring::ScoringPolicy;
use crate::policies::SelectionContext;
use crate::workers::Worker;
use std::sync::Arc;

/// Prefers the worker whose heavier phase, prefill or decode, stays lightest
/// once the request joins, by the tokens this router has in flight on it.
#[derive(Debug, Default)]
pub struct ShapeAwarePolicy;

impl ShapeAwarePolicy {
    pub fn new() -> Self {
        Self
    }
}

impl ScoringPolicy for ShapeAwarePolicy {
    /// `1.0` for the smallest post-admission bottleneck down to `0.0` for the
    /// largest, min-max scaled to the CURRENT fleet like `LoadBasedPolicy`.
    fn scores(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Vec<f32> {
        let request = request_rank_load(
            ctx.input_tokens().unwrap_or(0) as usize,
            ctx.output_tokens(),
        );
        let bottlenecks: Vec<u64> = workers
            .iter()
            .map(|worker| {
                let load = worker.router_inflight_tokens();
                // A prefill worker only computes the prompt; decode owns the output.
                let output = if worker.mode() == WorkerMode::Prefill {
                    0
                } else {
                    request.output_tokens
                };
                load.input_tokens
                    .saturating_add(request.input_tokens)
                    .max(load.output_tokens.saturating_add(output))
            })
            .collect();
        let lo = bottlenecks.iter().min().copied().unwrap_or(0);
        let span = (bottlenecks.iter().max().copied().unwrap_or(0) - lo) as f32;
        // `max(1.0)` is exact: a zero span means every `b - lo` is zero too.
        bottlenecks
            .into_iter()
            .map(|b| 1.0 - (b - lo) as f32 / span.max(1.0))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerSpec};
    use crate::workers::RankLoad;

    fn worker(id: &str, mode: WorkerMode) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode,
            model_ids: vec![ModelId("tiny".into())],
            ..Default::default()
        }))
    }

    fn load(input_tokens: u64, output_tokens: u64) -> RankLoad {
        RankLoad {
            input_tokens,
            output_tokens,
        }
    }

    fn scores(workers: &[Arc<Worker>], input: u64, output: Option<u64>) -> Vec<f32> {
        let m = ModelId("tiny".into());
        let ctx = SelectionContext::new(&m, None)
            .with_input_tokens(input)
            .with_output_tokens(output);
        ShapeAwarePolicy::new().scores(workers, &ctx)
    }

    #[test]
    fn prefers_the_lightest_post_admission_bottleneck() {
        let ws = [
            worker("a", WorkerMode::Plain),
            worker("b", WorkerMode::Plain),
            worker("c", WorkerMode::Plain),
        ];
        let _held = [
            ws[0].load_guard().with_tokens(load(100, 10)),
            ws[1].load_guard().with_tokens(load(20, 90)),
            ws[2].load_guard().with_tokens(load(60, 50)),
        ];
        // Bottlenecks 120, 110 and 80 once a 20/20 request joins.
        let s = scores(&ws, 20, Some(20));
        assert!(s[2] > s[1] && s[1] > s[0], "{s:?}");
        // A decode-heavy request avoids the worker already deepest in decode.
        let s = scores(&ws, 1, Some(80));
        assert!(s[0] > s[1] && s[0] > s[2], "{s:?}");
    }

    #[test]
    fn prefill_worker_is_not_charged_the_output() {
        let ws = [
            worker("p", WorkerMode::Prefill),
            worker("q", WorkerMode::Prefill),
        ];
        let _held = ws[0].load_guard().with_tokens(load(50, 0));
        // The output budget would dominate both workers if it were charged.
        let s = scores(&ws, 10, Some(1_000));
        assert!(s[1] > s[0], "{s:?}");
    }

    #[test]
    fn tokens_are_released_when_the_guard_drops() {
        let w = worker("a", WorkerMode::Plain);
        drop(w.load_guard().with_tokens(load(500, 500)));
        assert_eq!(w.router_inflight_tokens(), load(0, 0));
        assert_eq!(w.router_inflight_load(), 0);
    }

    #[test]
    fn equal_load_scores_equal() {
        let ws = [
            worker("a", WorkerMode::Plain),
            worker("b", WorkerMode::Plain),
        ];
        assert_eq!(scores(&ws, 5, None), vec![1.0, 1.0]);
    }
}
