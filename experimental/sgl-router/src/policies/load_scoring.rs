// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::load_monitor::AggregateLoad;

/// Scores one fresh Engine load aggregate for routing and admission.
///
/// This scorer is intentionally independent of routing and request admission
/// execution so its inputs, normalization, and outputs can be reviewed in
/// isolation.
#[derive(Debug, Default)]
pub struct EngineLoadScorer;

impl EngineLoadScorer {
    /// Returns normalized pressure in `[0, 1]`.
    ///
    /// The input signals are normalized as follows:
    ///
    /// ```text
    /// request_pressure = clamp(total_requests / max_running_requests, 0, 1)
    ///
    /// throughput = max(prefill_throughput, gen_throughput)
    /// queue_delay = waiting_uncached_tokens / throughput
    /// queue_pressure = queue_delay / (1 + queue_delay)
    ///
    /// pressure = clamp(
    ///     max(request_pressure, queue_pressure, max_rank_token_usage),
    ///     0,
    ///     1,
    /// )
    /// ```
    ///
    /// `queue_delay / (1 + queue_delay)` maps a non-negative delay to
    /// `[0, 1)` without introducing a tuning threshold. Queued work with zero
    /// throughput is treated as full pressure. Taking the maximum prevents a
    /// saturated request, queue, or KV-cache dimension from being hidden by
    /// healthier dimensions.
    pub fn pressure(load: &AggregateLoad) -> f64 {
        let request_pressure = ratio(load.total_requests, load.max_running_requests);
        let queue_pressure = if load.num_waiting_uncached_tokens == 0 {
            0.0
        } else {
            let throughput = load.prefill_throughput.max(load.gen_throughput);
            if throughput > 0.0 {
                let delay = load.num_waiting_uncached_tokens as f64 / throughput;
                delay / (1.0 + delay)
            } else {
                1.0
            }
        };
        load.max_rank_token_usage
            .max(request_pressure)
            .max(queue_pressure)
            .clamp(0.0, 1.0)
    }

    /// Relative probability available to a pressure-aware routing policy.
    pub fn routing_weight(load: &AggregateLoad) -> f64 {
        1.0 - Self::pressure(load)
    }

    /// Router-side concurrent requests allowed beyond the captured Engine
    /// state. This is a policy output only; it does not mutate worker state.
    pub fn admission_quota(load: &AggregateLoad) -> usize {
        let weight = Self::routing_weight(load);
        if load.available_slots == 0 || weight <= f64::EPSILON {
            return 0;
        }
        let quota = (load.available_slots as f64 * weight).floor() as u64;
        usize::try_from(quota.max(1)).unwrap_or(usize::MAX)
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        f64::from(numerator > 0)
    } else {
        (numerator as f64 / denominator as f64).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_uses_queue_kv_and_throughput() {
        let load = AggregateLoad {
            total_requests: 20,
            max_running_requests: 100,
            num_waiting_uncached_tokens: 100,
            prefill_throughput: 100.0,
            max_rank_token_usage: 0.4,
            ..AggregateLoad::default()
        };
        assert_eq!(EngineLoadScorer::pressure(&load), 0.5);

        let faster = AggregateLoad {
            prefill_throughput: 900.0,
            ..load.clone()
        };
        assert_eq!(EngineLoadScorer::pressure(&faster), 0.4);
    }

    #[test]
    fn no_throughput_with_queued_work_is_full_pressure() {
        let load = AggregateLoad {
            num_waiting_uncached_tokens: 1,
            max_running_requests: 100,
            available_slots: 100,
            ..AggregateLoad::default()
        };
        assert_eq!(EngineLoadScorer::pressure(&load), 1.0);
        assert_eq!(EngineLoadScorer::admission_quota(&load), 0);
    }

    #[test]
    fn pressure_reduces_weight_and_quota() {
        let cool = AggregateLoad {
            max_running_requests: 100,
            available_slots: 100,
            max_rank_token_usage: 0.1,
            ..AggregateLoad::default()
        };
        let hot = AggregateLoad {
            max_rank_token_usage: 0.9,
            ..cool.clone()
        };
        assert_eq!(EngineLoadScorer::routing_weight(&cool), 0.9);
        assert!((EngineLoadScorer::routing_weight(&hot) - 0.1).abs() < f64::EPSILON);
        assert_eq!(EngineLoadScorer::admission_quota(&cool), 90);
        assert_eq!(EngineLoadScorer::admission_quota(&hot), 9);
    }

    #[test]
    fn full_request_pressure_sheds_new_work() {
        let load = AggregateLoad {
            total_requests: 100,
            max_running_requests: 100,
            available_slots: 10,
            ..AggregateLoad::default()
        };
        assert_eq!(EngineLoadScorer::pressure(&load), 1.0);
        assert_eq!(EngineLoadScorer::routing_weight(&load), 0.0);
        assert_eq!(EngineLoadScorer::admission_quota(&load), 0);
    }
}
