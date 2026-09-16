// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Static buckets define candidate domains and fallback order. Worker scoring,
//! admission, and guards remain the responsibility of the P/D policies.

use sgl_router::config::{BucketConfig, BucketSpec, BucketStage, SloBucketPolicy};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::buckets::{BucketRequest, BucketSelector};
use sgl_router::policies::CacheCandidate;
use sgl_router::workers::Worker;
use std::sync::Arc;

fn worker(id: &str, mode: WorkerMode) -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(id.into()),
        url: format!("http://{id}:30000"),
        mode,
        model_ids: vec![ModelId("m".into())],
        bootstrap_port: None,
    }))
}

fn bucket(id: &str, stage: BucketStage, rank: u32, worker_ids: &[&str]) -> BucketSpec {
    BucketSpec {
        id: id.into(),
        stage,
        rank,
        worker_ids: worker_ids.iter().map(|id| (*id).into()).collect(),
        min_extend_tokens: None,
        max_extend_tokens: None,
        min_sequence_tokens: None,
        max_sequence_tokens: None,
        max_context_tokens: None,
        ttft_p95_at_capacity_ms: None,
        tps_p05_at_capacity: None,
        max_pending_prefill_tokens: None,
    }
}

#[test]
fn prefill_slo_first_tries_eligible_buckets_by_rank_before_degrading() {
    let fast = worker("fast", WorkerMode::Prefill);
    let cheap = worker("cheap", WorkerMode::Prefill);
    let mut cheap_bucket = bucket("cheap", BucketStage::Prefill, 10, &["cheap"]);
    cheap_bucket.ttft_p95_at_capacity_ms = Some(300);
    let mut fast_bucket = bucket("fast", BucketStage::Prefill, 20, &["fast"]);
    fast_bucket.ttft_p95_at_capacity_ms = Some(80);
    let selector = BucketSelector::new(Some(BucketConfig {
        buckets: vec![cheap_bucket, fast_bucket],
        ttft_slo_policy: SloBucketPolicy::SloFirst,
        tps_slo_policy: SloBucketPolicy::Disabled,
    }));

    let domains = selector.prefill_domains(
        &[cheap, fast],
        BucketRequest {
            input_tokens: 256,
            expected_peak_sequence_tokens: None,
            ttft_slo_ms: Some(100),
            tps_slo: None,
        },
    );

    assert_eq!(
        domains
            .iter()
            .map(|domain| domain.id.as_str())
            .collect::<Vec<_>>(),
        ["fast", "cheap"],
        "eligible buckets come first; non-eligible buckets are the explicit SLO-degraded fallback"
    );
}

#[test]
fn prefill_best_effort_tries_non_slo_bucket_before_reserved_slo_capacity() {
    let fast = worker("fast", WorkerMode::Prefill);
    let cheap = worker("cheap", WorkerMode::Prefill);
    let mut fast_bucket = bucket("fast", BucketStage::Prefill, 10, &["fast"]);
    fast_bucket.ttft_p95_at_capacity_ms = Some(80);
    let mut cheap_bucket = bucket("cheap", BucketStage::Prefill, 20, &["cheap"]);
    cheap_bucket.ttft_p95_at_capacity_ms = Some(300);
    let selector = BucketSelector::new(Some(BucketConfig {
        buckets: vec![fast_bucket, cheap_bucket],
        ttft_slo_policy: SloBucketPolicy::BestEffort,
        tps_slo_policy: SloBucketPolicy::Disabled,
    }));

    let domains = selector.prefill_domains(
        &[fast, cheap],
        BucketRequest {
            input_tokens: 256,
            expected_peak_sequence_tokens: None,
            ttft_slo_ms: Some(100),
            tps_slo: None,
        },
    );

    assert_eq!(
        domains
            .iter()
            .map(|domain| domain.id.as_str())
            .collect::<Vec<_>>(),
        ["cheap", "fast"],
        "best-effort tries non-SLO capacity first and retains the SLO tier as fallback"
    );
}

#[test]
fn prefill_domain_uses_full_input_as_extend_work_without_cache() {
    let short = worker("short", WorkerMode::Prefill);
    let long = worker("long", WorkerMode::Prefill);
    let mut short_bucket = bucket("p-short", BucketStage::Prefill, 10, &["short"]);
    short_bucket.max_extend_tokens = Some(64);
    let mut long_bucket = bucket("p-long", BucketStage::Prefill, 20, &["long"]);
    long_bucket.min_extend_tokens = Some(65);
    let selector = BucketSelector::new(Some(BucketConfig {
        buckets: vec![short_bucket, long_bucket],
        ttft_slo_policy: SloBucketPolicy::Disabled,
        tps_slo_policy: SloBucketPolicy::Disabled,
    }));

    let domains = selector.prefill_domains(
        &[short, long],
        BucketRequest {
            input_tokens: 256,
            expected_peak_sequence_tokens: None,
            ttft_slo_ms: None,
            tps_slo: None,
        },
    );

    assert_eq!(
        domains
            .iter()
            .map(|domain| domain.id.as_str())
            .collect::<Vec<_>>(),
        ["p-long"]
    );
}

#[test]
fn cache_candidate_uses_bucket_metadata_without_extend_range_filtering() {
    let cached = worker("cached", WorkerMode::Prefill);
    let mut cached_bucket = bucket("p-cached", BucketStage::Prefill, 10, &["cached"]);
    cached_bucket.max_extend_tokens = Some(8);
    cached_bucket.max_pending_prefill_tokens = Some(64);
    let selector = BucketSelector::new(Some(BucketConfig {
        buckets: vec![cached_bucket],
        ttft_slo_policy: SloBucketPolicy::Disabled,
        tps_slo_policy: SloBucketPolicy::Disabled,
    }));
    let candidate = CacheCandidate {
        worker: cached,
        matched_prefix_tokens: 128,
        uncached_tokens: 128,
        matched_prefix_blocks: 8,
        candidate_range_id: "global".into(),
        max_pending_prefill_tokens: None,
    };

    let prepared = selector
        .prepare_prefill_cache_candidate(
            candidate,
            BucketRequest {
                input_tokens: 256,
                expected_peak_sequence_tokens: None,
                ttft_slo_ms: None,
                tps_slo: None,
            },
        )
        .expect("extend range does not constrain global cache affinity");

    assert_eq!(prepared.candidate_range_id, "p-cached");
    assert_eq!(prepared.max_pending_prefill_tokens, Some(64));
}

#[test]
fn decode_bucket_uses_peak_sequence_length_then_tps_profile_and_rank() {
    let short = worker("short", WorkerMode::Decode);
    let long = worker("long", WorkerMode::Decode);
    let mut short_bucket = bucket("short", BucketStage::Decode, 10, &["short"]);
    short_bucket.max_sequence_tokens = Some(1_024);
    short_bucket.tps_p05_at_capacity = Some(80.0);
    let mut long_bucket = bucket("long", BucketStage::Decode, 20, &["long"]);
    long_bucket.max_sequence_tokens = Some(8_192);
    long_bucket.tps_p05_at_capacity = Some(40.0);
    let selector = BucketSelector::new(Some(BucketConfig {
        buckets: vec![short_bucket, long_bucket],
        ttft_slo_policy: SloBucketPolicy::Disabled,
        tps_slo_policy: SloBucketPolicy::SloFirst,
    }));

    let domains = selector.decode_domains(
        &[short, long],
        BucketRequest {
            input_tokens: 256,
            expected_peak_sequence_tokens: Some(900),
            ttft_slo_ms: None,
            tps_slo: Some(60.0),
        },
    );

    assert_eq!(domains.len(), 2);
    assert_eq!(domains[0].id, "short");
    assert_eq!(domains[1].id, "long");
}

#[test]
fn missing_bucket_configuration_keeps_the_global_domain() {
    let p = worker("p", WorkerMode::Prefill);
    let d = worker("d", WorkerMode::Decode);
    let selector = BucketSelector::new(None);
    let facts = BucketRequest {
        input_tokens: 128,
        expected_peak_sequence_tokens: Some(512),
        ttft_slo_ms: Some(100),
        tps_slo: Some(20.0),
    };

    let prefill = selector.prefill_domains(&[p], facts);
    let decode = selector.decode_domains(&[d], facts);

    assert_eq!(prefill.len(), 1);
    assert_eq!(prefill[0].id, "global");
    assert_eq!(decode.len(), 1);
    assert_eq!(decode[0].id, "global");
}

#[test]
fn prefill_only_bucket_configuration_keeps_the_global_decode_domain() {
    let p = worker("p", WorkerMode::Prefill);
    let d = worker("d", WorkerMode::Decode);
    let selector = BucketSelector::new(Some(BucketConfig {
        buckets: vec![bucket("p", BucketStage::Prefill, 10, &["p"])],
        ttft_slo_policy: SloBucketPolicy::Disabled,
        tps_slo_policy: SloBucketPolicy::Disabled,
    }));
    let facts = BucketRequest {
        input_tokens: 128,
        expected_peak_sequence_tokens: Some(512),
        ttft_slo_ms: None,
        tps_slo: None,
    };

    assert_eq!(selector.prefill_domains(&[p], facts)[0].id, "p");
    let decode = selector.decode_domains(&[d], facts);
    assert_eq!(decode.len(), 1);
    assert_eq!(decode[0].id, "global");
}

#[test]
fn decode_catch_all_still_rejects_input_beyond_runtime_context() {
    let d = worker("d", WorkerMode::Decode);
    let mut catch_all = bucket("d-catch-all", BucketStage::Decode, 10, &["d"]);
    catch_all.max_context_tokens = Some(1_024);
    let selector = BucketSelector::new(Some(BucketConfig {
        buckets: vec![catch_all],
        ttft_slo_policy: SloBucketPolicy::Disabled,
        tps_slo_policy: SloBucketPolicy::Disabled,
    }));

    let domains = selector.decode_domains(
        &[d],
        BucketRequest {
            input_tokens: 2_048,
            expected_peak_sequence_tokens: None,
            ttft_slo_ms: None,
            tps_slo: None,
        },
    );

    assert!(
        domains.is_empty(),
        "an unknown output budget does not erase the known input context requirement"
    );
}

#[test]
fn membership_index_preserves_exact_matching_and_fleet_order() {
    let workers: Vec<_> = (0..10)
        .map(|index| worker(&format!("w{index}"), WorkerMode::Prefill))
        .collect();
    let scan = bucket("scan", BucketStage::Prefill, 10, &["w3", "W3", "w1", "w1"]);
    let set = bucket(
        "set",
        BucketStage::Prefill,
        20,
        &[
            "w9", "w3", "w1", "w1", "W3", " w2", "absent-0", "absent-1", "absent-2",
        ],
    );
    let selector = BucketSelector::new(Some(BucketConfig {
        buckets: vec![scan, set],
        ttft_slo_policy: SloBucketPolicy::Disabled,
        tps_slo_policy: SloBucketPolicy::Disabled,
    }));
    let request = BucketRequest {
        input_tokens: 128,
        expected_peak_sequence_tokens: None,
        ttft_slo_ms: None,
        tps_slo: None,
    };

    let domains = selector.prefill_domains(&workers, request);
    let ids = |index: usize| {
        domains[index]
            .workers
            .iter()
            .map(|worker| worker.id.0.as_str())
            .collect::<Vec<_>>()
    };
    assert_eq!(ids(0), ["w1", "w3"]);
    assert_eq!(ids(1), ["w1", "w3", "w9"]);

    assert_eq!(
        selector
            .prefill_affinity_domain(&workers, &workers[9], request)
            .expect("w9 has a bucket affinity")
            .id,
        "set"
    );
}
