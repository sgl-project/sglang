// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared router config for the cache-aware proxy tests.
//!
//! The model id contains `deepseek-v4` so the tokenizer registry auto-attaches the
//! built-in V4 chat formatter — the engine-equivalent path — with no template fixture.

use std::sync::Arc;
use std::time::Duration;

use sgl_router::config::{
    AffinityConfig, CacheAwareConfig, CachePrefixProvider, Config, DiscoveryBackend,
    InflightLoadConfig, ModelConfig, ObservabilityConfig, PolicyKind, ProxyConfig, ServerConfig,
    StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry;
use sgl_router::policies::PolicyRegistry;
use sgl_router::policies_reorg::factory::build_resolver;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::{AppContext, ChatRouting};
use sgl_router::state::kv_events::{
    BlockSizeOracle, HashTree, KvEventIndex, RadixTreePrefixProvider,
};
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{EngineProfile, WireProtocol, WorkerRegistry};
use sglang_processor::openai::OpenAiSettings;

use crate::common::mock_worker::MockWorker;

pub const MODEL: &str = "deepseek-v4-tiny";

/// A single-model native `cache_aware` router. Discovery is a placeholder because
/// every caller installs its own `WorkerRegistry`.
pub fn config() -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: MODEL.into(),
            tokenizer_path: Some("tests/fixtures/tiny_tokenizer.json".into()),
            disable_input_ids_forwarding: false,
            tokenizer: Default::default(),
            policy: PolicyKind::CacheAware,
            decode_policy: Default::default(),
            dp_aware: false,
            bucket_config: None,
            reorg_buckets: None,
            reorg_admission: Default::default(),
            circuit_breaker: None,
            cache_aware: Some(CacheAwareConfig::default()),
            affinity: None,
            sticky: None,
            fused: None,
            eligibility: None,
            sampling_overrides: Default::default(),
            default_chat_template_kwargs: Default::default(),
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        router_inflight_load: InflightLoadConfig::default(),
    }
}

/// [`config`] with prefixes from the local radix tree and a single cache candidate.
fn radix_config() -> Config {
    let mut cfg = config();
    cfg.model.cache_aware.as_mut().unwrap().prefix_provider = CachePrefixProvider::RadixTree;
    cfg.model.affinity = Some(AffinityConfig {
        cache_affinity_min_matched_tokens: Some(0),
        cache_candidate_min_workers: 1,
        cache_candidate_ratio: 1.0,
        cache_candidate_max_workers: 1,
        ..Default::default()
    });
    cfg
}

/// Workers registered as introspection would, with `openai` as their OpenAI-layer settings.
/// `openai[i]` is what worker `i` reports; missing entries report nothing.
fn registry_of(
    workers: &[(&MockWorker, WorkerMode)],
    openai: &[Option<Arc<OpenAiSettings>>],
) -> WorkerRegistry {
    let registry = WorkerRegistry::default();
    for (i, &(worker, mode)) in workers.iter().enumerate() {
        let spec = WorkerSpec {
            id: WorkerId(worker.url.clone()),
            url: worker.url.clone(),
            mode,
            model_ids: vec![ModelId(MODEL.into())],
            bootstrap_port: (mode == WorkerMode::Prefill).then_some(8997),
            ..Default::default()
        };
        let profile = EngineProfile {
            openai: openai.get(i).cloned().flatten(),
            ..WireProtocol::default().into()
        };
        registry.add_with_cb(spec, None, profile).unwrap();
    }
    registry
}

/// A cache-aware router over `workers` whose KV prefixes come from the local `tree`.
#[allow(dead_code)] // Only some test files route by a local radix tree.
pub fn radix_router(workers: &[(&MockWorker, WorkerMode)], tree: HashTree) -> axum::Router {
    radix_router_with(workers, tree, &[])
}

/// [`radix_router`] whose workers reported default OpenAI settings, so
/// OpenAI completions go through `/generate`.
#[allow(dead_code)] // Only some test files serve OpenAI through `/generate`.
pub fn openai_router(workers: &[(&MockWorker, WorkerMode)]) -> axum::Router {
    let settings: Vec<_> = workers.iter().map(|_| OpenAiSettings::default()).collect();
    openai_router_each(workers, settings)
}

/// [`openai_router`] whose worker `i` reports `settings[i]`.
#[allow(dead_code)] // Only some test files set engine OpenAI settings.
pub fn openai_router_each(
    workers: &[(&MockWorker, WorkerMode)],
    settings: Vec<OpenAiSettings>,
) -> axum::Router {
    let settings: Vec<_> = settings.into_iter().map(|s| Some(Arc::new(s))).collect();
    radix_router_with(workers, HashTree::new(), &settings)
}

fn radix_router_with(
    workers: &[(&MockWorker, WorkerMode)],
    tree: HashTree,
    openai: &[Option<Arc<OpenAiSettings>>],
) -> axum::Router {
    let cfg = radix_config();
    let (tree, oracle) = (Arc::new(tree), BlockSizeOracle::new());
    oracle.try_set(1).unwrap();
    let policies = build_registry(&cfg, Arc::clone(&tree), Arc::clone(&oracle)).unwrap();
    let mut ctx = AppContext::new(
        cfg.clone(),
        Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap()),
        Arc::new(Proxy::new(Duration::from_secs(5)).unwrap()),
        Arc::new(registry_of(workers, openai)),
        Arc::new(policies),
    );
    ctx.radix_tree_prefix_provider = Some(RadixTreePrefixProvider::new(tree, Arc::clone(&oracle)));
    ctx.block_size_oracle = oracle;
    build_router(Arc::new(ctx))
}

/// [`radix_router`] on the bucket-first (reorg) selection path, over `state`'s tree.
#[allow(dead_code)] // Only some test files route by a local radix tree.
pub fn reorg_radix_router(
    workers: &[(&MockWorker, WorkerMode)],
    state: &KvEventIndex,
) -> axum::Router {
    let cfg = radix_config();
    let oracle = state.block_size_oracle();
    oracle.try_set(1).unwrap();
    let (resolver, _) = build_resolver(&cfg.model, state, None).unwrap();
    let mut ctx = AppContext::new(
        cfg.clone(),
        Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap()),
        Arc::new(Proxy::new(Duration::from_secs(5)).unwrap()),
        Arc::new(registry_of(workers, &[])),
        Arc::new(PolicyRegistry::default()),
    );
    ctx.chat_routing = ChatRouting::Reorg([(ModelId(MODEL.into()), resolver)].into());
    ctx.radix_tree_prefix_provider = Some(RadixTreePrefixProvider::new(
        state.tree(),
        Arc::clone(&oracle),
    ));
    ctx.block_size_oracle = oracle;
    build_router(Arc::new(ctx))
}
