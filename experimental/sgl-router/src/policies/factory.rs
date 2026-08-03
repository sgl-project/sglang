// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::{Config, ModelConfig, PolicyKind};
use crate::discovery::ModelId;
use crate::policies::{
    cache_aware_zmq::CacheAwareZmqPolicy,
    kv_events::{BlockSizeOracle, HashTree},
    load_based::LoadBasedPolicy,
    power_of_two::PowerOfTwoChoicesPolicy,
    random::RandomPolicy,
    round_robin::RoundRobinPolicy,
    scoring::{
        admission::Overloaded, prefix_cache, prefix_cache::PrefixCachePolicy, FusedScorePolicy,
        Pipeline,
    },
    sticky::StickyPolicy,
    Policy, PolicyRegistry,
};
use crate::tokenizer::TokenizerRegistry;
use anyhow::{anyhow, Result};
use std::sync::Arc;
use std::time::Duration;

/// Build a dependency-free policy for use as the sticky-session fallback
/// (keyless requests + initial pin of a new key). `Cli::into_config`
/// validates `--sticky-fallback-policy` to one of these four, so the
/// `CacheAwareZmq`/`Sticky` arms are never reached in practice.
fn build_sticky_fallback(kind: PolicyKind) -> Arc<dyn Policy> {
    match kind {
        PolicyKind::RoundRobin => Arc::new(RoundRobinPolicy::new()),
        PolicyKind::Random => Arc::new(RandomPolicy::new()),
        PolicyKind::PowerOfTwo => Arc::new(PowerOfTwoChoicesPolicy::new()),
        PolicyKind::LoadBased => Arc::new(LoadBasedPolicy::new()),
        PolicyKind::CacheAwareZmq
        | PolicyKind::Overloaded
        | PolicyKind::Sticky
        | PolicyKind::PrefixCache
        | PolicyKind::FusedScore => {
            unreachable!("sticky fallback is validated to be dependency-free in Cli::into_config")
        }
    }
}

/// Construct a [`StickyPolicy`] from a model's `sticky` config (or
/// defaults). Shared by `build_policy` and the test shim so the duration
/// conversion + fallback wiring live in one place.
fn build_sticky(model: &ModelConfig) -> Arc<dyn Policy> {
    let s = model.sticky.clone().unwrap_or_default();
    Arc::new(StickyPolicy::new(
        Duration::from_secs(s.idle_secs),
        Duration::from_secs(s.eviction_interval_secs),
        build_sticky_fallback(s.fallback_policy),
    ))
}

/// Constructs a policy for one model.
pub fn build_policy(
    model: &ModelConfig,
    tree: Arc<HashTree>,
    tokenizers: Arc<TokenizerRegistry>,
    block_size_oracle: Arc<BlockSizeOracle>,
) -> Result<Arc<dyn Policy>> {
    let inner = build_kind(model.policy, model, &tree, &tokenizers, &block_size_oracle)?;
    let Some(elig) = model.eligibility.as_ref().filter(|e| !e.filters.is_empty()) else {
        return Ok(inner);
    };
    let mut filters = Vec::with_capacity(elig.filters.len());
    for &kind in &elig.filters {
        let f = build_kind(kind, model, &tree, &tokenizers, &block_size_oracle)?;
        if !f.can_filter() {
            return Err(anyhow!("--filter: `{kind}` imposes no eligibility rule"));
        }
        filters.push(f);
    }
    Ok(Arc::new(Pipeline::new(filters, inner)?))
}

/// Build one policy kind with the shared model dependencies.
fn build_kind(
    kind: PolicyKind,
    model: &ModelConfig,
    tree: &Arc<HashTree>,
    tokenizers: &Arc<TokenizerRegistry>,
    block_size_oracle: &Arc<BlockSizeOracle>,
) -> Result<Arc<dyn Policy>> {
    let (tree, tokenizers, block_size_oracle) = (
        Arc::clone(tree),
        Arc::clone(tokenizers),
        Arc::clone(block_size_oracle),
    );
    Ok(match kind {
        PolicyKind::RoundRobin => Arc::new(RoundRobinPolicy::new()),
        PolicyKind::Random => Arc::new(RandomPolicy::new()),
        PolicyKind::PowerOfTwo => Arc::new(PowerOfTwoChoicesPolicy::new()),
        PolicyKind::LoadBased => Arc::new(LoadBasedPolicy::new()),
        PolicyKind::CacheAwareZmq => {
            let cache_cfg = model.cache_aware.clone().unwrap_or_default();
            Arc::new(CacheAwareZmqPolicy::new(
                cache_cfg,
                tree,
                tokenizers,
                block_size_oracle,
            ))
        }
        PolicyKind::Sticky => build_sticky(model),
        PolicyKind::FusedScore => build_fused(model, &tree, &tokenizers, &block_size_oracle)?,
        PolicyKind::PrefixCache => {
            let p = PrefixCachePolicy::new(tree, block_size_oracle, prefix_cache::DEFAULT_WEIGHT);
            let share = (model.eligibility.as_ref()).and_then(|e| e.min_prefix_share);
            Arc::new(match share {
                Some(s) => p.with_min_share(s),
                None => p,
            })
        }
        PolicyKind::Overloaded => {
            let cap = (model.eligibility.as_ref())
                .and_then(|e| e.max_in_flight)
                .unwrap_or(usize::MAX);
            Arc::new(Overloaded::new(cap))
        }
    })
}

/// Builds `--policy fused_score`.
fn build_fused(
    model: &ModelConfig,
    tree: &Arc<HashTree>,
    tokenizers: &Arc<TokenizerRegistry>,
    oracle: &Arc<BlockSizeOracle>,
) -> Result<Arc<dyn Policy>> {
    let spec = model.fused.as_deref().unwrap_or_default();
    if spec.is_empty() {
        return Err(anyhow!(
            "--policy fused_score needs at least one --fuse term"
        ));
    }
    let mut terms: Vec<(Arc<dyn Policy>, Option<f32>)> = Vec::with_capacity(spec.len());
    for t in spec {
        let mut m = model.clone();
        (m.policy, m.fused) = (t.kind, None);
        m.eligibility = None;
        let inner = build_policy(
            &m,
            Arc::clone(tree),
            Arc::clone(tokenizers),
            Arc::clone(oracle),
        )?;
        if !inner.can_fuse() {
            return Err(anyhow!("--fuse: `{}` does not support fusion", t.kind));
        }
        terms.push((inner, t.weight));
    }
    Ok(Arc::new(FusedScorePolicy::new(terms)?))
}

/// Builds a policy with test defaults.
#[cfg(test)]
pub fn build_policy_kind_only(kind: PolicyKind) -> Result<Arc<dyn Policy>> {
    Ok(match kind {
        PolicyKind::RoundRobin => Arc::new(RoundRobinPolicy::new()),
        PolicyKind::Random => Arc::new(RandomPolicy::new()),
        PolicyKind::PowerOfTwo => Arc::new(PowerOfTwoChoicesPolicy::new()),
        PolicyKind::LoadBased => Arc::new(LoadBasedPolicy::new()),
        PolicyKind::CacheAwareZmq => Arc::new(CacheAwareZmqPolicy::new(
            crate::config::CacheAwareConfig::default(),
            Arc::new(HashTree::new()),
            Arc::new(TokenizerRegistry::default()),
            BlockSizeOracle::new(),
        )),
        PolicyKind::Sticky => {
            let s = crate::config::StickyConfig::default();
            Arc::new(StickyPolicy::new(
                Duration::from_secs(s.idle_secs),
                Duration::from_secs(s.eviction_interval_secs),
                build_sticky_fallback(s.fallback_policy),
            ))
        }
        PolicyKind::Overloaded => Arc::new(Overloaded::new(usize::MAX)),
        PolicyKind::PrefixCache => Arc::new(PrefixCachePolicy::new(
            Arc::new(HashTree::new()),
            BlockSizeOracle::new(),
            prefix_cache::DEFAULT_WEIGHT,
        )),
        PolicyKind::FusedScore => {
            return Err(anyhow!("--policy {kind} needs --fuse terms from the model"))
        }
    })
}

pub fn build_registry(
    cfg: &Config,
    tree: Arc<HashTree>,
    tokenizers: Arc<TokenizerRegistry>,
    block_size_oracle: Arc<BlockSizeOracle>,
) -> Result<PolicyRegistry> {
    let reg = PolicyRegistry::default();
    let m = &cfg.model;
    reg.insert(
        ModelId(m.id.clone()),
        build_policy(
            m,
            Arc::clone(&tree),
            Arc::clone(&tokenizers),
            Arc::clone(&block_size_oracle),
        )?,
    );
    Ok(reg)
}

/// Builds a registry with empty cache-aware dependencies.
pub fn build_registry_with_defaults(cfg: &Config) -> Result<PolicyRegistry> {
    build_registry(
        cfg,
        Arc::new(HashTree::new()),
        Arc::new(TokenizerRegistry::default()),
        BlockSizeOracle::new(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ProxyConfig, ServerConfig,
        StaticUrlsDiscoveryConfig,
    };

    use crate::config::{EligibilityConfig, PolicyKind};
    use crate::discovery::{WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::SelectionContext;
    use crate::workers::Worker;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("modelA".into())],
            bootstrap_port: None,
        }))
    }

    #[test]
    fn filter_overloaded_wires_through_the_factory() {
        let mut cfg = cfg_with_model("modelA", PolicyKind::LoadBased);
        cfg.model.eligibility = Some(EligibilityConfig {
            filters: vec![PolicyKind::Overloaded],
            max_in_flight: Some(2),
            min_prefix_share: None,
        });
        let reg = build_registry_with_defaults(&cfg).unwrap();
        let p = reg.get(&ModelId("modelA".into())).unwrap();

        let ws = vec![worker("w0"), worker("w1")];
        let model = ModelId("modelA".into());
        let ctx = SelectionContext::new(&model, None);
        let _one = ws[1].load_guard();
        assert_eq!(
            p.select(&ws, &ctx).unwrap().id,
            ws[0].id,
            "load still ranks"
        );

        let _fill: Vec<_> = (ws.iter())
            .flat_map(|w| (0..2).map(|_| w.load_guard()))
            .collect();
        assert!(
            p.select(&ws, &ctx).is_none(),
            "every worker is over the cap"
        );
    }

    #[test]
    fn a_filter_must_actually_constrain() {
        let mut cfg = cfg_with_model("modelA", PolicyKind::PrefixCache);
        let built = |c: &Config| {
            build_registry_with_defaults(c).map(|r| r.get(&ModelId("modelA".into())).unwrap())
        };
        assert!(
            !built(&cfg).unwrap().can_filter(),
            "no floor, so the term stays a pure preference",
        );

        cfg.model.eligibility = Some(EligibilityConfig {
            filters: vec![PolicyKind::PrefixCache],
            max_in_flight: None,
            min_prefix_share: Some(0.6),
        });
        assert!(
            built(&cfg).unwrap().needs_request_tokens(),
            "the floor reads the prompt"
        );

        cfg.model.eligibility = Some(EligibilityConfig {
            filters: vec![PolicyKind::RoundRobin],
            max_in_flight: None,
            min_prefix_share: None,
        });
        let err = built(&cfg).unwrap_err().to_string();
        assert!(
            err.contains("round_robin") && err.contains("no eligibility rule"),
            "{err}"
        );
    }

    fn cfg_with_model(id: &str, policy: PolicyKind) -> Config {
        Config {
            server: ServerConfig {
                host: "0".into(),
                port: 0,
            },
            observability: Default::default(),
            model: ModelConfig {
                id: id.into(),
                tokenizer_path: "/tmp/x".into(),
                policy,
                circuit_breaker: None,
                cache_aware: None,
                sticky: None,
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

    #[test]
    fn build_policy_kind_only_covers_all_variants() {
        for kind in [
            PolicyKind::RoundRobin,
            PolicyKind::Random,
            PolicyKind::PowerOfTwo,
            PolicyKind::LoadBased,
            PolicyKind::CacheAwareZmq,
            PolicyKind::Sticky,
            PolicyKind::PrefixCache,
        ] {
            assert!(build_policy_kind_only(kind).is_ok(), "{kind:?}");
        }
        assert!(build_policy_kind_only(PolicyKind::FusedScore).is_err());
    }

    #[test]
    fn prefix_cache_builds_standalone_and_is_fusable() {
        let p = build_policy(
            &cfg_with_model("m", PolicyKind::PrefixCache).model,
            Arc::new(HashTree::new()),
            Arc::new(TokenizerRegistry::default()),
            BlockSizeOracle::new(),
        )
        .expect("--policy prefix_cache must build");
        assert!(p.can_fuse(), "prefix_cache must be usable as a --fuse term");
    }

    #[test]
    fn fused_score_refuses_a_non_fusable_term_at_startup() {
        let mut cfg = cfg_with_model("m", PolicyKind::FusedScore);
        let term = |kind, weight| crate::config::FusedTerm { kind, weight };

        for weight in [None, Some(2.5)] {
            cfg.model.fused = Some(vec![term(PolicyKind::PrefixCache, weight)]);
            assert!(build_registry_with_defaults(&cfg).is_ok(), "{weight:?}");
        }

        cfg.model.fused = Some(vec![term(PolicyKind::RoundRobin, None)]);
        let err = build_registry_with_defaults(&cfg).unwrap_err().to_string();
        assert!(err.contains("round_robin"), "{err}");
        assert!(err.contains("does not support fusion"), "{err}");

        cfg.model.fused = Some(vec![]);
        assert!(build_registry_with_defaults(&cfg)
            .unwrap_err()
            .to_string()
            .contains("at least one --fuse term"));
    }

    #[test]
    fn fused_score_accepts_an_outer_eligibility_pipeline() {
        let mut cfg = cfg_with_model("modelA", PolicyKind::FusedScore);
        cfg.model.fused = Some(vec![crate::config::FusedTerm {
            kind: PolicyKind::LoadBased,
            weight: None,
        }]);
        cfg.model.eligibility = Some(EligibilityConfig {
            filters: vec![PolicyKind::Overloaded],
            max_in_flight: Some(0),
            min_prefix_share: None,
        });

        let policy = build_registry_with_defaults(&cfg)
            .expect("an outer filter must not make fused terms non-fusable")
            .get(&ModelId("modelA".into()))
            .unwrap();
        let workers = vec![worker("w0"), worker("w1")];
        let model = ModelId("modelA".into());
        assert!(policy
            .select(&workers, &SelectionContext::new(&model, None))
            .is_none());
    }

    #[test]
    fn registry_assigns_configured_model() {
        let cfg = cfg_with_model("qwen", PolicyKind::RoundRobin);
        let tree = Arc::new(HashTree::new());
        let tokenizers = Arc::new(TokenizerRegistry::default());
        let reg = build_registry(&cfg, tree, tokenizers, BlockSizeOracle::new()).unwrap();
        assert!(reg.get(&ModelId("qwen".into())).is_some());
        assert!(reg.get(&ModelId("missing".into())).is_none());
    }

    #[test]
    fn cache_aware_zmq_builds_via_factory() {
        let cfg = cfg_with_model("modelA", PolicyKind::CacheAwareZmq);
        let tree = Arc::new(HashTree::new());
        let tokenizers = Arc::new(TokenizerRegistry::default());
        let reg = build_registry(&cfg, tree, tokenizers, BlockSizeOracle::new()).unwrap();
        let p = reg.get(&ModelId("modelA".into())).unwrap();
        let dbg = format!("{p:?}");
        assert!(
            dbg.contains("CacheAwareZmqPolicy"),
            "expected CacheAwareZmqPolicy debug repr, got: {dbg}",
        );
    }

    #[test]
    fn load_based_builds_via_factory() {
        let cfg = cfg_with_model("modelA", PolicyKind::LoadBased);
        let tree = Arc::new(HashTree::new());
        let tokenizers = Arc::new(TokenizerRegistry::default());
        let reg = build_registry(&cfg, tree, tokenizers, BlockSizeOracle::new()).unwrap();
        let p = reg.get(&ModelId("modelA".into())).unwrap();
        let dbg = format!("{p:?}");
        assert!(
            dbg.contains("LoadBasedPolicy"),
            "expected LoadBasedPolicy debug repr, got: {dbg}",
        );
    }

    #[test]
    fn sticky_builds_via_factory() {
        let cfg = cfg_with_model("modelA", PolicyKind::Sticky);
        let tree = Arc::new(HashTree::new());
        let tokenizers = Arc::new(TokenizerRegistry::default());
        let reg = build_registry(&cfg, tree, tokenizers, BlockSizeOracle::new()).unwrap();
        let p = reg.get(&ModelId("modelA".into())).unwrap();
        let dbg = format!("{p:?}");
        assert!(
            dbg.contains("StickyPolicy"),
            "expected StickyPolicy debug repr, got: {dbg}",
        );
    }
}
