// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::engine_load::LoadStat;
use sgl_router::policies::{
    CacheCandidate, CacheCandidateProposal, Policy, PolicyRegistry, PrefillProposal, ProposalKind,
    SelectionContext, SelectionProposal,
};
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{Worker, WorkerRegistry};
use tower::ServiceExt;

use crate::common::mock_worker::MockWorker;

#[derive(Debug)]
struct AdmissionProbePolicy {
    primary: Arc<Worker>,
    backup: Arc<Worker>,
    committed: Arc<Mutex<Option<String>>>,
}

impl Policy for AdmissionProbePolicy {
    fn select(&self, _: &[Arc<Worker>], _: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        panic!("chat routing must resolve the prefill proposal before selection")
    }

    fn propose(&self, _: &[Arc<Worker>], _: &SelectionContext<'_>) -> Option<SelectionProposal> {
        Some(
            SelectionProposal::with_backup(Arc::clone(&self.primary), Arc::clone(&self.backup))
                .with_kind(ProposalKind::SessionAffinity),
        )
    }

    fn uses_shared_prefill_admission(&self) -> bool {
        true
    }

    fn commit_prefill_selection(
        &self,
        _: &SelectionContext<'_>,
        _: ProposalKind,
        selected: &Arc<Worker>,
    ) {
        *self.committed.lock().unwrap() = Some(selected.id.0.clone());
    }
}

#[derive(Debug)]
struct EmptyPolicy;

impl Policy for EmptyPolicy {
    fn select(&self, _: &[Arc<Worker>], _: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        None
    }
}

#[derive(Debug)]
struct InvalidPairPolicy {
    outsider: Arc<Worker>,
}

impl Policy for InvalidPairPolicy {
    fn select(&self, _: &[Arc<Worker>], _: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        panic!("chat routing must use the invalid prefill proposal")
    }

    fn propose(&self, _: &[Arc<Worker>], _: &SelectionContext<'_>) -> Option<SelectionProposal> {
        Some(SelectionProposal::primary(Arc::clone(&self.outsider)))
    }

    fn uses_shared_prefill_admission(&self) -> bool {
        true
    }
}

#[derive(Debug)]
struct CacheCandidatesPolicy {
    worker: Arc<Worker>,
}

#[derive(Debug)]
struct SnapshotProbePolicy {
    worker: Arc<Worker>,
    needs_snapshot: bool,
    observed_snapshot: Arc<Mutex<Option<bool>>>,
}

impl Policy for SnapshotProbePolicy {
    fn select(&self, _: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        *self.observed_snapshot.lock().unwrap() = Some(ctx.load_snapshot().is_some());
        Some(Arc::clone(&self.worker))
    }

    fn needs_load_snapshot(&self) -> bool {
        self.needs_snapshot
    }
}

impl Policy for CacheCandidatesPolicy {
    fn select(&self, _: &[Arc<Worker>], _: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        panic!("chat routing must use the cache-candidate proposal")
    }

    fn propose_prefill(
        &self,
        _: &[Arc<Worker>],
        _: &SelectionContext<'_>,
    ) -> Option<PrefillProposal> {
        Some(PrefillProposal::CacheCandidates(CacheCandidateProposal {
            candidates: vec![CacheCandidate {
                worker: Arc::clone(&self.worker),
                matched_prefix_tokens: 1,
                uncached_tokens: 1,
                candidate_range_id: "global".into(),
                max_pending_prefill_tokens: None,
            }],
            cache_switch_margin_tokens: 0,
        }))
    }

    fn needs_load_snapshot(&self) -> bool {
        true
    }
}

fn config(policy: PolicyKind) -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            affinity: None,
            fused: None,
            eligibility: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    }
}

struct TestFixture {
    ctx: Arc<AppContext>,
    backends: Vec<MockWorker>,
    workers: Vec<Arc<Worker>>,
}

async fn fixture(
    policy_kind: PolicyKind,
    build_policy: impl FnOnce(&[Arc<Worker>]) -> Arc<dyn Policy>,
) -> TestFixture {
    let backends = vec![
        MockWorker::start(vec![]).await,
        MockWorker::start(vec![]).await,
    ];
    let cfg = config(policy_kind);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for (index, backend) in backends.iter().enumerate() {
        registry
            .add(WorkerSpec {
                id: WorkerId(if index == 0 { "primary" } else { "backup" }.into()),
                url: backend.url.clone(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: None,
            })
            .unwrap();
    }
    let registered = registry.workers_for(&ModelId("tiny".into()));
    let workers = ["primary", "backup"]
        .into_iter()
        .map(|id| {
            registered
                .iter()
                .find(|worker| worker.id.0 == id)
                .cloned()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let policies = Arc::new(PolicyRegistry::default());
    policies.insert(ModelId("tiny".into()), build_policy(&workers));
    let ctx = Arc::new(AppContext::new(
        cfg,
        tokenizers,
        Arc::new(Proxy::new(Duration::from_secs(5)).unwrap()),
        registry,
        policies,
    ));
    TestFixture {
        ctx,
        backends,
        workers,
    }
}

async fn send_chat(ctx: &Arc<AppContext>) -> StatusCode {
    build_router(Arc::clone(ctx))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&serde_json::json!({
                        "model": "tiny",
                        "messages": [{"role": "user", "content": "hi"}],
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap()
        .status()
}

#[tokio::test]
async fn chat_attaches_load_snapshot_only_when_the_policy_needs_it() {
    for needs_snapshot in [false, true] {
        let observed_snapshot = Arc::new(Mutex::new(None));
        let fixture = fixture(PolicyKind::RoundRobin, |workers| {
            Arc::new(SnapshotProbePolicy {
                worker: Arc::clone(&workers[0]),
                needs_snapshot,
                observed_snapshot: Arc::clone(&observed_snapshot),
            })
        })
        .await;

        assert_eq!(send_chat(&fixture.ctx).await, StatusCode::OK);
        assert_eq!(*observed_snapshot.lock().unwrap(), Some(needs_snapshot));
    }
}

fn assert_failure_metric(ctx: &AppContext, policy: &str, expected_reason: &str) {
    let metrics = ctx.metrics.render();
    assert!(metrics.contains(&format!(
        "sgl_router_policy_selection_failures_total{{policy=\"{policy}\",reason=\"{expected_reason}\"}} 1"
    )));
    for other in [
        "prefill_admission_exhausted",
        "cache_candidates_exhausted",
        "proposal_empty",
    ] {
        if other != expected_reason {
            assert!(
                !metrics.contains(&format!("reason=\"{other}\"")),
                "{metrics}"
            );
        }
    }
}

#[tokio::test]
async fn chat_commits_the_admitted_prefill_backup() {
    let committed = Arc::new(Mutex::new(None));
    let fixture = fixture(PolicyKind::SessionAware, |workers| {
        Arc::new(AdmissionProbePolicy {
            primary: Arc::clone(&workers[0]),
            backup: Arc::clone(&workers[1]),
            committed: Arc::clone(&committed),
        })
    })
    .await;
    fixture.ctx.engine_load.set(
        &fixture.workers[0].url,
        0,
        LoadStat {
            num_running_reqs: 1,
            num_waiting_reqs: 0,
            num_tokens: 100,
            max_total_num_tokens: 100,
        },
        Instant::now(),
    );

    assert_eq!(send_chat(&fixture.ctx).await, StatusCode::OK);
    assert!(fixture.backends[0]
        .captured
        .lock()
        .unwrap()
        .last_body
        .is_none());
    assert!(fixture.backends[1]
        .captured
        .lock()
        .unwrap()
        .last_body
        .is_some());
    assert_eq!(committed.lock().unwrap().as_deref(), Some("backup"));
    assert!(!fixture
        .ctx
        .metrics
        .render()
        .contains("sgl_router_policy_selection_failures_total{"));
}

#[tokio::test]
async fn capacity_exhaustion_does_not_return_503() {
    let committed = Arc::new(Mutex::new(None));
    let fixture = fixture(PolicyKind::SessionAware, |workers| {
        Arc::new(AdmissionProbePolicy {
            primary: Arc::clone(&workers[0]),
            backup: Arc::clone(&workers[1]),
            committed: Arc::clone(&committed),
        })
    })
    .await;
    for worker in &fixture.workers {
        fixture.ctx.engine_load.set(
            &worker.url,
            0,
            LoadStat {
                num_running_reqs: 0,
                num_waiting_reqs: 0,
                num_tokens: 100,
                max_total_num_tokens: 100,
            },
            Instant::now(),
        );
    }

    assert_eq!(send_chat(&fixture.ctx).await, StatusCode::OK);
    assert_eq!(
        fixture
            .backends
            .iter()
            .filter(|backend| backend.captured.lock().unwrap().last_body.is_some())
            .count(),
        1,
        "the request must be dispatched to exactly one legal backend"
    );
    assert!(matches!(
        committed.lock().unwrap().as_deref(),
        Some("primary" | "backup")
    ));
    assert!(!fixture
        .ctx
        .metrics
        .render()
        .contains("sgl_router_policy_selection_failures_total{"));
}

#[tokio::test]
async fn chat_records_proposal_empty() {
    let fixture = fixture(PolicyKind::SessionAware, |_| Arc::new(EmptyPolicy)).await;

    assert_eq!(
        send_chat(&fixture.ctx).await,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_failure_metric(&fixture.ctx, "session_aware", "proposal_empty");
}

#[tokio::test]
async fn chat_records_prefill_admission_exhausted_for_out_of_range_primary() {
    let outsider = Arc::new(Worker::new(WorkerSpec {
        id: WorkerId("outsider".into()),
        url: "http://outsider:30000".into(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
    }));
    let fixture = fixture(PolicyKind::SessionAware, |_| {
        Arc::new(InvalidPairPolicy {
            outsider: Arc::clone(&outsider),
        })
    })
    .await;

    assert_eq!(
        send_chat(&fixture.ctx).await,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_failure_metric(&fixture.ctx, "session_aware", "prefill_admission_exhausted");
}

#[tokio::test]
async fn chat_records_cache_candidates_exhausted() {
    let fixture = fixture(PolicyKind::CacheAware, |workers| {
        Arc::new(CacheCandidatesPolicy {
            worker: Arc::clone(&workers[0]),
        })
    })
    .await;
    fixture.ctx.engine_load.set(
        &fixture.workers[0].url,
        0,
        LoadStat {
            num_running_reqs: 1,
            num_waiting_reqs: 0,
            num_tokens: 100,
            max_total_num_tokens: 100,
        },
        Instant::now(),
    );

    assert_eq!(
        send_chat(&fixture.ctx).await,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_failure_metric(&fixture.ctx, "cache_aware", "cache_candidates_exhausted");
}
