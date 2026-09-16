// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Observable contract for decode policies.
//!
//! Decode guards require complete, fresh native monitor samples. Short frames
//! fall back to local load and must not appear as monitor-backed decisions.

use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::admission::{resolve_decode, CandidateDomain, DecisionReason};
use sgl_router::policies::decode::{
    resolve_decode_with_capacity_fallback, DecodePolicy, DecodePowerOfTwoPolicy,
    DecodeSelectionContext, LegacyHostAffinityDecodePolicy,
};
use sgl_router::policies::engine_load::{EngineLoadSnapshot, NativeCacheWorkerLoad};
use sgl_router::policies::SelectionProposal;
use sgl_router::workers::Worker;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

fn worker(id: &str) -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(id.into()),
        url: format!("http://{id}:30000"),
        mode: WorkerMode::Decode,
        model_ids: vec![ModelId("m".into())],
        bootstrap_port: None,
    }))
}

fn snapshot(entries: &[(&Arc<Worker>, u64, u64, u64, u64)]) -> EngineLoadSnapshot {
    EngineLoadSnapshot::from_native_cache_workers(
        7,
        entries
            .iter()
            .map(|(worker, running, waiting, used, capacity)| {
                (
                    worker.url.clone(),
                    NativeCacheWorkerLoad {
                        num_running_reqs: *running,
                        num_waiting_reqs: *waiting,
                        num_waiting_uncached_tokens: *waiting,
                        num_used_tokens: *used,
                        num_total_tokens: *used,
                        max_total_num_tokens: *capacity,
                        max_running_requests: 64,
                        prefill_throughput_tokens_per_s: None,
                        estimated_prefill_queue_ms: None,
                        captured_at: Instant::now(),
                    },
                )
            })
            .collect::<HashMap<_, _>>(),
    )
}

#[test]
fn decode_p2_proposes_a_distinct_lower_pressure_primary_and_backup() {
    let busy = worker("busy");
    let idle = worker("idle");
    busy.active_requests.store(8, Ordering::Relaxed);
    idle.active_requests.store(1, Ordering::Relaxed);
    let domain = CandidateDomain::global_decode(&[Arc::clone(&busy), Arc::clone(&idle)]);
    let ctx = DecodeSelectionContext::new();

    let proposal = DecodePowerOfTwoPolicy::new()
        .propose(&domain, &ctx)
        .expect("two decode candidates must produce a proposal");

    assert_eq!(proposal.primary.id, idle.id);
    assert_eq!(
        proposal.backup.expect("P2 keeps the other sample").id,
        busy.id
    );
}

#[test]
fn legacy_host_affinity_remains_an_explicit_single_primary_compatibility_policy() {
    let same_host = worker("host-a");
    let other_host = worker("host-b");
    let domain = CandidateDomain::global_decode(&[Arc::clone(&same_host), other_host]);
    let ctx = DecodeSelectionContext::new().with_prefill_url("http://host-a:9999");

    let proposal = LegacyHostAffinityDecodePolicy
        .propose(&domain, &ctx)
        .expect("legacy policy selects one compatible decode worker");

    assert_eq!(proposal.primary.id, same_host.id);
    assert!(
        proposal.backup.is_none(),
        "legacy semantics do not invent a backup"
    );
}

#[test]
fn decode_admission_uses_backup_before_scanning_domain() {
    let primary = worker("primary");
    let backup = worker("backup");
    let fallback = worker("fallback");
    let domain = CandidateDomain::global_decode(&[
        Arc::clone(&primary),
        Arc::clone(&backup),
        Arc::clone(&fallback),
    ]);
    let loads = snapshot(&[
        (&primary, 4, 0, 950, 1_000),
        (&backup, 0, 0, 0, 1_000),
        (&fallback, 0, 0, 0, 1_000),
    ]);
    let proposal = SelectionProposal::with_backup(Arc::clone(&primary), Arc::clone(&backup));

    let decision =
        resolve_decode(&domain, &proposal, 64, &loads).expect("admitted backup must be selected");

    assert_eq!(decision.selected.id, backup.id);
    assert_eq!(decision.reason, DecisionReason::BackupPrimaryAdmission);
}

#[test]
fn decode_guard_can_escape_a_primary_to_lower_dynamic_pressure_backup() {
    let primary = worker("primary");
    let backup = worker("backup");
    let domain = CandidateDomain::global_decode(&[Arc::clone(&primary), Arc::clone(&backup)]);
    let loads = snapshot(&[(&primary, 3, 2, 900, 2_000), (&backup, 1, 0, 100, 2_000)]);
    let proposal = SelectionProposal::with_backup(Arc::clone(&primary), Arc::clone(&backup));

    let decision =
        resolve_decode(&domain, &proposal, 64, &loads).expect("both candidates are admitted");

    assert_eq!(decision.selected.id, backup.id);
    assert_eq!(decision.reason, DecisionReason::BackupPressureGuard);
}

#[test]
fn decode_all_capacity_rejected_falls_back_to_power_of_two_within_domain() {
    let primary = worker("primary");
    let backup = worker("backup");
    let workers = vec![Arc::clone(&primary), Arc::clone(&backup)];
    let domain = CandidateDomain::global_decode(&workers);
    let loads = snapshot(&[
        (&primary, 0, 0, 1_000, 1_000),
        (&backup, 0, 10, 1_000, 1_000),
    ]);
    let proposal = SelectionProposal::with_backup(Arc::clone(&primary), Arc::clone(&backup));

    let decision = resolve_decode_with_capacity_fallback(&domain, &proposal, 64, &loads)
        .expect("capacity exhaustion must degrade within the decode domain");

    assert_eq!(decision.selected.id, primary.id);
    assert_eq!(decision.reason, DecisionReason::CapacityFallbackPowerOfTwo);
}
