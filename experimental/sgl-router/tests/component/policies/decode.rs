// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Decode policy 的最小可观察契约。
//!
//! 这些测试只覆盖 #34608 实际发布的 running/waiting/KV/local active-load 输入；
//! 不把未发布的 transfer、decode queue 或 retraction 指标伪造成可用数据。

use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::admission::{resolve_decode, CandidateDomain, DecisionReason};
use sgl_router::policies::decode::{
    DecodePolicy, DecodePowerOfTwoPolicy, DecodeSelectionContext, LegacyHostAffinityDecodePolicy,
};
use sgl_router::policies::engine_load::{EngineLoadSnapshot, EngineWorkerLoad};
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
    EngineLoadSnapshot::from_workers(
        7,
        entries
            .iter()
            .map(|(worker, running, waiting, used, capacity)| {
                (
                    worker.url.clone(),
                    EngineWorkerLoad {
                        num_running_reqs: *running,
                        num_waiting_reqs: *waiting,
                        num_tokens: *used,
                        max_total_num_tokens: *capacity,
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
    assert_eq!(decision.reason, DecisionReason::BackupLoadComparison);
}
