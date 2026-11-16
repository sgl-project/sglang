// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use sgl_router::buckets_reorg::{Bucket, BucketGroups, BucketResolver, EngineGroup, SloPreference};
use sgl_router::policies_reorg::power_of_two::PowerOfTwoPolicy;
use sgl_router::policies_reorg::PickError;
use sgl_router::state::load_monitor::engine_reported_load::EngineReportedLoadTable;

fn bucket(id: &str, max: u64, rank: u32, ttft: Option<u64>, tps: Option<f64>) -> Bucket {
    let mut bucket = Bucket::new(
        id,
        BucketGroups::Plain(EngineGroup::new(Arc::new(PowerOfTwoPolicy::new(
            EngineReportedLoadTable::new(),
        )))),
    );
    bucket.limits.max = Some(max);
    bucket.rank = rank;
    bucket.ttft_ms = ttft;
    bucket.tokens_per_second = tps;
    bucket
}

fn resolver() -> BucketResolver {
    BucketResolver::new(vec![
        bucket("both", 100, 9, Some(50), Some(100.0)),
        bucket("ttft", 10, 1, Some(50), Some(10.0)),
        bucket("tps", 10, 2, Some(100), Some(100.0)),
        bucket("neither", 8, 0, Some(100), Some(10.0)),
        bucket("missing", 10, 3, None, None),
        bucket("too-short", 1, 0, Some(1), Some(1000.0)),
    ])
    .unwrap()
}

fn ids(resolver: &BucketResolver, ttft: Option<u64>, tps: Option<f64>) -> Vec<&str> {
    resolver
        .resolve(5, Some(5), ttft, tps)
        .unwrap()
        .into_iter()
        .map(|b| b.id.as_str())
        .collect()
}

#[test]
fn slo_tiers_keep_length_constraints_and_capacity_rank_order() {
    let mut resolver = resolver();
    assert_eq!(
        ids(&resolver, Some(50), Some(100.0)),
        ["neither", "ttft", "tps", "missing", "both"]
    );
    resolver.ttft_slo = SloPreference::SloFirst;
    resolver.tps_slo = SloPreference::SloFirst;
    assert_eq!(
        ids(&resolver, Some(50), Some(100.0)),
        ["both", "ttft", "tps", "neither", "missing"]
    );
    resolver.ttft_slo = SloPreference::BestEffort;
    resolver.tps_slo = SloPreference::BestEffort;
    assert_eq!(
        ids(&resolver, Some(50), Some(100.0)),
        ["neither", "missing", "ttft", "tps", "both"]
    );
    resolver.ttft_slo = SloPreference::SloFirst;
    assert_eq!(
        ids(&resolver, Some(50), Some(100.0)),
        ["ttft", "neither", "missing", "both", "tps"]
    );
}

#[test]
fn absent_targets_are_neutral_and_each_preference_can_be_disabled() {
    let mut resolver = resolver();
    resolver.ttft_slo = SloPreference::SloFirst;
    resolver.tps_slo = SloPreference::BestEffort;
    assert_eq!(
        ids(&resolver, None, None),
        ["neither", "ttft", "tps", "missing", "both"]
    );
    assert_eq!(
        ids(&resolver, Some(50), None),
        ["ttft", "both", "neither", "tps", "missing"]
    );
    resolver.ttft_slo = SloPreference::Disabled;
    resolver.tps_slo = SloPreference::SloFirst;
    assert_eq!(
        ids(&resolver, Some(1), Some(100.0)),
        ["tps", "both", "neither", "ttft", "missing"]
    );
}

#[test]
fn invalid_enabled_targets_fail_and_invalid_estimates_do_not_match() {
    let mut resolver = resolver();
    resolver.ttft_slo = SloPreference::SloFirst;
    resolver.tps_slo = SloPreference::SloFirst;
    assert!(matches!(
        resolver.resolve(5, None, Some(0), None),
        Err(PickError::InvalidSignal(_))
    ));
    for tps in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            resolver.resolve(5, None, None, Some(tps)),
            Err(PickError::InvalidSignal(_))
        ));
    }
    resolver.buckets[0].ttft_ms = Some(0);
    resolver.buckets[0].tokens_per_second = Some(f64::INFINITY);
    assert_eq!(
        ids(&resolver, Some(50), Some(100.0)),
        ["ttft", "tps", "neither", "missing", "both"]
    );
    resolver.ttft_slo = SloPreference::Disabled;
    resolver.tps_slo = SloPreference::Disabled;
    assert!(resolver.resolve(5, None, Some(0), Some(f64::NAN)).is_ok());
}

#[test]
fn slo_preferences_never_relax_peak_capacity_or_input_range() {
    let mut resolver = resolver();
    resolver.ttft_slo = SloPreference::SloFirst;
    resolver.tps_slo = SloPreference::SloFirst;
    resolver.buckets[0].max_context_tokens = Some(6);
    resolver.buckets[1].limits.min = Some(6);
    let found = resolver.resolve(5, Some(7), Some(50), Some(100.0)).unwrap();
    assert_eq!(
        found.iter().map(|b| b.id.as_str()).collect::<Vec<_>>(),
        ["tps", "neither", "missing"]
    );
    assert!(matches!(
        resolver.resolve(5, Some(4), None, None),
        Err(PickError::InvalidSignal(_))
    ));
}
