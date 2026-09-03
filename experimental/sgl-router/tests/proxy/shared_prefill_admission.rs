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
    Policy, PolicyRegistry, ProposalKind, SelectionContext, SelectionProposal,
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

fn config() -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::SessionAware,
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

#[tokio::test]
async fn chat_commits_the_admitted_prefill_backup() {
    let primary = MockWorker::start(vec![]).await;
    let backup = MockWorker::start(vec![]).await;
    let cfg = config();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    registry
        .add(WorkerSpec {
            id: WorkerId("primary".into()),
            url: primary.url.clone(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        })
        .unwrap();
    registry
        .add(WorkerSpec {
            id: WorkerId("backup".into()),
            url: backup.url.clone(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        })
        .unwrap();
    let registered = registry.workers_for(&ModelId("tiny".into()));
    let primary_worker = registered
        .iter()
        .find(|worker| worker.id.0 == "primary")
        .cloned()
        .unwrap();
    let backup_worker = registered
        .iter()
        .find(|worker| worker.id.0 == "backup")
        .cloned()
        .unwrap();
    let committed = Arc::new(Mutex::new(None));
    let policies = Arc::new(PolicyRegistry::default());
    policies.insert(
        ModelId("tiny".into()),
        Arc::new(AdmissionProbePolicy {
            primary: primary_worker,
            backup: backup_worker,
            committed: Arc::clone(&committed),
        }),
    );
    let ctx = Arc::new(AppContext::new(
        cfg,
        tokenizers,
        Arc::new(Proxy::new(Duration::from_secs(5)).unwrap()),
        registry,
        policies,
    ));
    ctx.engine_load.set(
        &primary.url,
        0,
        LoadStat {
            num_running_reqs: 1,
            num_waiting_reqs: 0,
            num_tokens: 100,
            max_total_num_tokens: 100,
        },
        Instant::now(),
    );

    let response = build_router(ctx)
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
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert!(primary.captured.lock().unwrap().last_body.is_none());
    assert!(backup.captured.lock().unwrap().last_body.is_some());
    assert_eq!(committed.lock().unwrap().as_deref(), Some("backup"));
}
