//! Bounded preparation cost samples. This estimates complete Engine preparation,
//! including admission and publication; it does not claim pure storage I/O time.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct PreparationConfig {
    pub max_groups: usize,
    pub sample_window: usize,
    pub min_samples: usize,
    pub quantile: f64,
    pub max_lead_ms: u64,
    /// Separate, finite L2 claim after lifecycle preparation; zero disables it.
    pub ready_pin_ttl_ms: u64,
    /// Conservative token reservation; overlapping jobs may be charged twice.
    pub auto_pin_max_tokens: u64,
    /// Avoid a host demotion immediately followed by its preparation.
    pub min_cold_residency_ms: u64,
}

impl Default for PreparationConfig {
    fn default() -> Self {
        Self {
            max_groups: 128,
            sample_window: 32,
            min_samples: 4,
            quantile: 0.9,
            max_lead_ms: 10_000,
            ready_pin_ttl_ms: 0,
            auto_pin_max_tokens: 65_536,
            min_cold_residency_ms: 3_000,
        }
    }
}

impl PreparationConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !(1..=4096).contains(&self.max_groups)
            || !(1..=1024).contains(&self.sample_window)
            || self.min_samples == 0
            || self.min_samples > self.sample_window
            || !self.quantile.is_finite()
            || !(0.0..=1.0).contains(&self.quantile)
            || !(1..=86_400_000).contains(&self.max_lead_ms)
            || self.ready_pin_ttl_ms > 60_000
            || self.auto_pin_max_tokens > 1 << 24
            || (self.ready_pin_ttl_ms > 0 && self.auto_pin_max_tokens == 0)
            || self.min_cold_residency_ms > 86_400_000
        {
            return Err("invalid preparation sample or lead budget");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Estimate {
    pub preparation_ms: u64,
    pub lead_ms: u64,
    pub samples: usize,
    pub incomplete_samples: usize,
}

#[derive(Default)]
struct Samples {
    values: VecDeque<(f64, bool)>,
    sequence: u64,
}

pub struct PreparationCosts {
    config: PreparationConfig,
    groups: BTreeMap<(String, u32), Samples>,
    sequence: u64,
}

impl PreparationCosts {
    pub fn new(config: PreparationConfig) -> Result<Self, &'static str> {
        config.validate()?;
        Ok(Self {
            config,
            groups: BTreeMap::new(),
            sequence: 0,
        })
    }

    fn key(owner: &str, tokens: u64) -> (String, u32) {
        // The caller includes the owner epoch and compatibility in owner.
        // Nearby sizes share samples; long and short prefixes do not.
        (owner.into(), 64 - tokens.max(1).leading_zeros())
    }

    pub fn observe(&mut self, owner: &str, tokens: u64, elapsed_ms: f64) {
        self.record(owner, tokens, elapsed_ms, true);
    }

    /// An incomplete operation supplies a lower bound, never a zero-cost hit.
    pub fn observe_incomplete(&mut self, owner: &str, tokens: u64, elapsed_ms: f64) {
        self.record(owner, tokens, elapsed_ms, false);
    }

    fn record(&mut self, owner: &str, tokens: u64, elapsed_ms: f64, complete: bool) {
        if tokens == 0 || !elapsed_ms.is_finite() || elapsed_ms <= 0.0 {
            return;
        }
        let key = Self::key(owner, tokens);
        if !self.groups.contains_key(&key) && self.groups.len() >= self.config.max_groups {
            if let Some(oldest) = self
                .groups
                .iter()
                .min_by_key(|(_, samples)| samples.sequence)
                .map(|(key, _)| key.clone())
            {
                self.groups.remove(&oldest);
            }
        }
        self.sequence = self.sequence.wrapping_add(1);
        let samples = self.groups.entry(key).or_default();
        samples.sequence = self.sequence;
        if samples.values.len() == self.config.sample_window {
            samples.values.pop_front();
        }
        samples
            .values
            .push_back((elapsed_ms / tokens as f64, complete));
    }

    pub fn estimate(
        &self,
        owner: &str,
        tokens: u64,
        prior_ms_per_token: f64,
        margin_ms: u64,
    ) -> Estimate {
        let samples = self.groups.get(&Self::key(owner, tokens));
        let count = samples.map_or(0, |s| s.values.len());
        let rate = if count >= self.config.min_samples {
            let mut sorted: Vec<_> = samples.unwrap().values.iter().map(|s| s.0).collect();
            sorted.sort_by(f64::total_cmp);
            let index = ((sorted.len() - 1) as f64 * self.config.quantile).ceil() as usize;
            sorted[index]
        } else {
            prior_ms_per_token
        };
        let preparation_ms = (rate * tokens as f64).ceil() as u64;
        Estimate {
            preparation_ms,
            lead_ms: preparation_ms
                .saturating_add(margin_ms)
                .min(self.config.max_lead_ms),
            samples: count,
            incomplete_samples: samples.map_or(0, |s| s.values.iter().filter(|v| !v.1).count()),
        }
    }

    pub fn groups(&self) -> usize {
        self.groups.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn costs() -> PreparationCosts {
        PreparationCosts::new(PreparationConfig {
            max_groups: 2,
            sample_window: 3,
            min_samples: 2,
            max_lead_ms: 500,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn learns_tail_with_bounded_lead_and_window() {
        let mut model = costs();
        model.observe("worker:epoch:layout", 100, 100.0);
        assert_eq!(
            model
                .estimate("worker:epoch:layout", 100, 2.0, 20)
                .preparation_ms,
            200
        );
        model.observe("worker:epoch:layout", 100, 900.0);
        let estimate = model.estimate("worker:epoch:layout", 100, 2.0, 20);
        assert_eq!(
            (estimate.preparation_ms, estimate.lead_ms, estimate.samples),
            (900, 500, 2)
        );
        for elapsed in [10.0, 10.0, 10.0] {
            model.observe("worker:epoch:layout", 100, elapsed);
        }
        assert_eq!(
            model
                .estimate("worker:epoch:layout", 100, 2.0, 20)
                .preparation_ms,
            10
        );
    }

    #[test]
    fn separates_epoch_size_and_evicts_oldest_group() {
        let mut model = costs();
        for _ in 0..2 {
            model.observe("a:epoch1:layout", 100, 500.0);
        }
        assert_eq!(model.estimate("a:epoch2:layout", 100, 2.0, 0).samples, 0);
        assert_eq!(model.estimate("a:epoch1:layout", 1000, 2.0, 0).samples, 0);
        model.observe("b:epoch1:layout", 100, 10.0);
        model.observe("c:epoch1:layout", 100, 10.0);
        assert_eq!(model.groups(), 2);
        assert_eq!(model.estimate("a:epoch1:layout", 100, 2.0, 0).samples, 0);
    }

    #[test]
    fn ignores_missing_invalid_and_zero_cost_observations() {
        let mut model = costs();
        for elapsed in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            model.observe("a", 100, elapsed);
        }
        model.observe("a", 0, 1.0);
        assert_eq!(model.groups(), 0);
        let mut config = PreparationConfig {
            quantile: f64::NAN,
            ..Default::default()
        };
        assert!(config.validate().is_err());
        config.quantile = 0.9;
        config.min_samples = config.sample_window + 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn incomplete_preparation_keeps_its_elapsed_lower_bound() {
        let mut model = costs();
        model.observe("a", 100, 10.0);
        model.observe_incomplete("a", 100, 900.0);
        let estimate = model.estimate("a", 100, 1.0, 0);
        assert_eq!(estimate.preparation_ms, 900);
        assert_eq!(estimate.incomplete_samples, 1);
    }

    #[test]
    fn ready_pin_is_disabled_by_default_and_has_finite_capacity_and_ttl() {
        assert_eq!(PreparationConfig::default().ready_pin_ttl_ms, 0);
        for (ttl, tokens) in [(60_001, 1), (1, 0), (1, (1 << 24) + 1)] {
            let config = PreparationConfig {
                ready_pin_ttl_ms: ttl,
                auto_pin_max_tokens: tokens,
                ..Default::default()
            };
            assert!(config.validate().is_err());
        }
    }
}
