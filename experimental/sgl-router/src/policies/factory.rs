// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::{
    Config, FilterKind, ModelConfig, PolicyKind, ScoreTermKind, StickyFallbackKind,
};
use crate::discovery::ModelId;
use crate::policies::{
    cache_aware::CacheAwarePolicy,
    cache_aware_zmq::CacheAwareZmqPolicy,
    engine_load::EngineLoadTable,
    kv_events::{BlockSizeOracle, HashTree},
    load_based::LoadBasedPolicy,
    power_of_two::PowerOfTwoChoicesPolicy,
    random::RandomPolicy,
    round_robin::RoundRobinPolicy,
    scoring::{
        admission::Overloaded, prefix_cache, prefix_cache::PrefixCachePolicy, FusedScorePolicy,
        Pipeline, ScorePolicy,
    },
    session_aware::SessionAwarePolicy,
    sticky::StickyPolicy,
    Policy, PolicyRegistry,
};
use crate::tokenizer::TokenizerRegistry;
use anyhow::{anyhow, Result};
use std::sync::Arc;
use std::time::Duration;

/// Build a dependency-free policy for keyless sticky requests and new pins.
fn build_sticky_fallback(kind: StickyFallbackKind) -> Arc<dyn Policy> {
    match kind {
        StickyFallbackKind::RoundRobin => Arc::new(RoundRobinPolicy::new()),
        StickyFallbackKind::Random => Arc::new(RandomPolicy::new()),
        StickyFallbackKind::PowerOfTwo => Arc::new(PowerOfTwoChoicesPolicy::new()),
        StickyFallbackKind::LoadBased => Arc::new(LoadBasedPolicy::new()),
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
    engine_load: Arc<EngineLoadTable>,
) -> Result<Arc<dyn Policy>> {
    validate_eligibility(model)?;
    let inner = build_kind(
        model.policy,
        model,
        &tree,
        &tokenizers,
        &block_size_oracle,
        &engine_load,
    )?;
    let Some(elig) = model.eligibility.as_ref().filter(|e| !e.filters.is_empty()) else {
        return Ok(inner);
    };
    let mut filters = Vec::with_capacity(elig.filters.len());
    for &kind in &elig.filters {
        filters.push(build_filter(kind, model, &tree, &block_size_oracle));
    }
    Ok(Arc::new(Pipeline::new(filters, inner)?))
}

/// Reject configurations that can bypass CLI validation when constructed in code.
fn validate_eligibility(model: &ModelConfig) -> Result<()> {
    let Some(eligibility) = model.eligibility.as_ref().filter(|e| !e.filters.is_empty()) else {
        return Ok(());
    };

    if model.policy == PolicyKind::Sticky {
        return Err(anyhow!(
            "eligibility filters cannot be combined with sticky policy"
        ));
    }
    if eligibility.filters.contains(&FilterKind::Overloaded)
        && eligibility.max_in_flight.unwrap_or(0) == 0
    {
        return Err(anyhow!("max_in_flight must be greater than 0"));
    }

    Ok(())
}

/// Build one policy kind with the shared model dependencies.
fn build_kind(
    kind: PolicyKind,
    model: &ModelConfig,
    tree: &Arc<HashTree>,
    tokenizers: &Arc<TokenizerRegistry>,
    block_size_oracle: &Arc<BlockSizeOracle>,
    engine_load: &Arc<EngineLoadTable>,
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
                Arc::clone(engine_load),
            ))
        }
        PolicyKind::SessionAware => Arc::new(SessionAwarePolicy::new(
            model.affinity.clone().unwrap_or_default(),
        )),
        PolicyKind::CacheAware => Arc::new(CacheAwarePolicy::new(
            model.affinity.clone().unwrap_or_default(),
        )),
        PolicyKind::Sticky => build_sticky(model),
        PolicyKind::FusedScore => build_fused(model, &tree, &block_size_oracle)?,
        PolicyKind::ScorePolicy => build_score_policy(model, &tree, &block_size_oracle)?,
    })
}

/// Builds one hard admission filter with the shared model dependencies.
fn build_filter(
    kind: FilterKind,
    model: &ModelConfig,
    tree: &Arc<HashTree>,
    block_size_oracle: &Arc<BlockSizeOracle>,
) -> Arc<dyn Policy> {
    match kind {
        FilterKind::Overloaded => {
            let cap = (model.eligibility.as_ref())
                .and_then(|e| e.max_in_flight)
                .unwrap_or(usize::MAX);
            Arc::new(Overloaded::new(cap))
        }
        FilterKind::PrefixCache => {
            let share = (model.eligibility.as_ref())
                .and_then(|e| e.min_prefix_share)
                .unwrap_or(0.0);
            Arc::new(
                PrefixCachePolicy::new(
                    Arc::clone(tree),
                    Arc::clone(block_size_oracle),
                    prefix_cache::DEFAULT_WEIGHT,
                )
                .with_min_share(share),
            )
        }
    }
}

/// Builds one soft scoring term for `--policy fused_score`.
fn build_score(
    kind: ScoreTermKind,
    tree: &Arc<HashTree>,
    block_size_oracle: &Arc<BlockSizeOracle>,
) -> Arc<dyn Policy> {
    match kind {
        ScoreTermKind::Random => Arc::new(RandomPolicy::new()),
        ScoreTermKind::LoadBased => Arc::new(LoadBasedPolicy::new()),
        ScoreTermKind::PrefixCache => Arc::new(PrefixCachePolicy::new(
            Arc::clone(tree),
            Arc::clone(block_size_oracle),
            prefix_cache::DEFAULT_WEIGHT,
        )),
    }
}

/// Builds `--policy fused_score`.
fn build_fused(
    model: &ModelConfig,
    tree: &Arc<HashTree>,
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
        terms.push((build_score(t.kind, tree, oracle), t.weight));
    }
    Ok(Arc::new(FusedScorePolicy::new(terms)?))
}

/// Builds the top-level `score_policy`.
fn build_score_policy(
    model: &ModelConfig,
    tree: &Arc<HashTree>,
    oracle: &Arc<BlockSizeOracle>,
) -> Result<Arc<dyn Policy>> {
    Ok(Arc::new(ScorePolicy::new(build_fused(
        model, tree, oracle,
    )?)))
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
            EngineLoadTable::new(),
        )),
        PolicyKind::SessionAware => Arc::new(SessionAwarePolicy::new(
            crate::config::AffinityConfig::default(),
        )),
        PolicyKind::CacheAware => Arc::new(CacheAwarePolicy::new(
            crate::config::AffinityConfig::default(),
        )),
        PolicyKind::Sticky => {
            let s = crate::config::StickyConfig::default();
            Arc::new(StickyPolicy::new(
                Duration::from_secs(s.idle_secs),
                Duration::from_secs(s.eviction_interval_secs),
                build_sticky_fallback(s.fallback_policy),
            ))
        }
        PolicyKind::FusedScore | PolicyKind::ScorePolicy => {
            return Err(anyhow!("--policy {kind} needs --fuse terms from the model"))
        }
    })
}

pub fn build_registry(
    cfg: &Config,
    tree: Arc<HashTree>,
    tokenizers: Arc<TokenizerRegistry>,
    block_size_oracle: Arc<BlockSizeOracle>,
    engine_load: Arc<EngineLoadTable>,
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
            Arc::clone(&engine_load),
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
        EngineLoadTable::new(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ProxyConfig, ServerConfig,
        StaticUrlsDiscoveryConfig,
    };

    use crate::config::{
        EligibilityConfig, FilterKind, PolicyKind, ScoreTermKind, StickyFallbackKind,
    };
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
            filters: vec![FilterKind::Overloaded],
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
        let mut cfg = cfg_with_model("modelA", PolicyKind::LoadBased);
        let built = |c: &Config| {
            build_registry_with_defaults(c).map(|r| r.get(&ModelId("modelA".into())).unwrap())
        };
        assert!(
            !built(&cfg).unwrap().can_filter(),
            "no floor, so the term stays a pure preference",
        );

        cfg.model.eligibility = Some(EligibilityConfig {
            filters: vec![FilterKind::PrefixCache],
            max_in_flight: None,
            min_prefix_share: Some(0.6),
        });
        assert!(
            built(&cfg).unwrap().needs_request_tokens(),
            "the floor reads the prompt"
        );

        cfg.model.eligibility = Some(EligibilityConfig {
            filters: vec![FilterKind::PrefixCache],
            max_in_flight: None,
            min_prefix_share: Some(0.6),
        });
        let filter = build_filter(
            FilterKind::PrefixCache,
            &cfg.model,
            &Arc::new(HashTree::new()),
            &BlockSizeOracle::new(),
        );
        assert!(
            filter.can_filter(),
            "a configured prefix-cache floor is a filter",
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

    #[test]
    fn build_policy_kind_only_covers_all_variants() {
        for (kind, needs_load_snapshot) in [
            (PolicyKind::RoundRobin, false),
            (PolicyKind::Random, false),
            (PolicyKind::PowerOfTwo, true),
            (PolicyKind::LoadBased, true),
            (PolicyKind::CacheAwareZmq, true),
            (PolicyKind::SessionAware, true),
            (PolicyKind::CacheAware, true),
            (PolicyKind::Sticky, false),
        ] {
            let policy = build_policy_kind_only(kind).unwrap();
            assert_eq!(
                policy.needs_load_snapshot(),
                needs_load_snapshot,
                "{kind:?}"
            );
        }
        assert!(build_policy_kind_only(PolicyKind::FusedScore).is_err());
        assert!(build_policy_kind_only(PolicyKind::ScorePolicy).is_err());
    }

    #[test]
    fn prefix_cache_builds_as_a_score_term() {
        let p = build_score(
            ScoreTermKind::PrefixCache,
            &Arc::new(HashTree::new()),
            &BlockSizeOracle::new(),
        );
        assert!(p.can_fuse(), "prefix_cache must be usable as a --fuse term");
        assert!(!p.needs_load_snapshot());
    }

    #[test]
    fn fused_score_builds_score_terms_and_rejects_an_empty_config() {
        let mut cfg = cfg_with_model("m", PolicyKind::FusedScore);
        let term = |kind, weight| crate::config::FusedTerm { kind, weight };

        for weight in [None, Some(2.5)] {
            cfg.model.fused = Some(vec![term(ScoreTermKind::PrefixCache, weight)]);
            assert!(build_registry_with_defaults(&cfg).is_ok(), "{weight:?}");
        }

        cfg.model.fused = Some(vec![]);
        assert!(build_registry_with_defaults(&cfg)
            .unwrap_err()
            .to_string()
            .contains("at least one --fuse term"));
    }

    /// `score_policy` uses its own top-level factory branch.
    #[test]
    fn score_policy_builds_via_its_own_factory_branch() {
        let mut cfg = cfg_with_model("modelA", PolicyKind::ScorePolicy);
        cfg.model.fused = Some(vec![crate::config::FusedTerm {
            kind: ScoreTermKind::PrefixCache,
            weight: Some(2.0),
        }]);

        let registry = build_registry_with_defaults(&cfg).expect("score policy builds");
        let policy = registry
            .get(&ModelId("modelA".into()))
            .expect("configured model has a policy");
        assert!(
            policy.can_fuse(),
            "the score policy exposes score semantics"
        );
        assert!(
            policy.uses_shared_prefill_admission(),
            "top-level score_policy participates in the shared hard admission layer"
        );
    }

    #[test]
    fn score_policy_with_filter_builds_one_outer_pipeline() {
        let mut cfg = cfg_with_model("modelA", PolicyKind::ScorePolicy);
        cfg.model.fused = Some(vec![crate::config::FusedTerm {
            kind: ScoreTermKind::LoadBased,
            weight: Some(1.0),
        }]);
        cfg.model.eligibility = Some(EligibilityConfig {
            filters: vec![FilterKind::Overloaded],
            max_in_flight: Some(2),
            min_prefix_share: None,
        });

        let registry = build_registry_with_defaults(&cfg)
            .expect("a score policy keeps eligibility outside its scoring terms");
        let policy = registry
            .get(&ModelId("modelA".into()))
            .expect("configured model has a policy");
        assert!(policy.uses_shared_prefill_admission());
    }

    #[test]
    fn factory_rejects_missing_or_zero_overloaded_capacity() {
        for max_in_flight in [None, Some(0)] {
            let mut cfg = cfg_with_model("modelA", PolicyKind::FusedScore);
            cfg.model.fused = Some(vec![crate::config::FusedTerm {
                kind: ScoreTermKind::LoadBased,
                weight: None,
            }]);
            cfg.model.eligibility = Some(EligibilityConfig {
                filters: vec![FilterKind::Overloaded],
                max_in_flight,
                min_prefix_share: None,
            });

            let error = build_registry_with_defaults(&cfg)
                .expect_err("overloaded capacity must be positive at construction");
            let message = error.to_string();
            assert!(message.contains("max_in_flight must be greater than 0"));
        }
    }

    #[test]
    fn factory_rejects_sticky_eligibility_pipeline() {
        let mut cfg = cfg_with_model("modelA", PolicyKind::Sticky);
        cfg.model.eligibility = Some(EligibilityConfig {
            filters: vec![FilterKind::Overloaded],
            max_in_flight: Some(2),
            min_prefix_share: None,
        });

        let error = build_registry_with_defaults(&cfg)
            .expect_err("sticky assignments cannot be wrapped by eligibility filters");
        let message = error.to_string();
        assert!(message.contains("eligibility filters cannot be combined with sticky"));
    }

    #[test]
    fn registry_assigns_configured_model() {
        let cfg = cfg_with_model("qwen", PolicyKind::RoundRobin);
        let tree = Arc::new(HashTree::new());
        let tokenizers = Arc::new(TokenizerRegistry::default());
        let reg = build_registry(
            &cfg,
            tree,
            tokenizers,
            BlockSizeOracle::new(),
            EngineLoadTable::new(),
        )
        .unwrap();
        assert!(reg.get(&ModelId("qwen".into())).is_some());
        assert!(reg.get(&ModelId("missing".into())).is_none());
    }

    #[test]
    fn cache_aware_zmq_builds_via_factory() {
        let cfg = cfg_with_model("modelA", PolicyKind::CacheAwareZmq);
        let tree = Arc::new(HashTree::new());
        let tokenizers = Arc::new(TokenizerRegistry::default());
        let reg = build_registry(
            &cfg,
            tree,
            tokenizers,
            BlockSizeOracle::new(),
            EngineLoadTable::new(),
        )
        .unwrap();
        let p = reg.get(&ModelId("modelA".into())).unwrap();
        let dbg = format!("{p:?}");
        assert!(
            dbg.contains("CacheAwareZmqPolicy"),
            "expected CacheAwareZmqPolicy debug repr, got: {dbg}",
        );
        assert!(p.needs_load_snapshot());
    }

    #[test]
    fn load_based_builds_via_factory() {
        let cfg = cfg_with_model("modelA", PolicyKind::LoadBased);
        let tree = Arc::new(HashTree::new());
        let tokenizers = Arc::new(TokenizerRegistry::default());
        let reg = build_registry(
            &cfg,
            tree,
            tokenizers,
            BlockSizeOracle::new(),
            EngineLoadTable::new(),
        )
        .unwrap();
        let p = reg.get(&ModelId("modelA".into())).unwrap();
        let dbg = format!("{p:?}");
        assert!(
            dbg.contains("LoadBasedPolicy"),
            "expected LoadBasedPolicy debug repr, got: {dbg}",
        );
        assert!(p.needs_load_snapshot());
    }

    #[test]
    fn sticky_builds_via_factory() {
        let cfg = cfg_with_model("modelA", PolicyKind::Sticky);
        let tree = Arc::new(HashTree::new());
        let tokenizers = Arc::new(TokenizerRegistry::default());
        let reg = build_registry(
            &cfg,
            tree,
            tokenizers,
            BlockSizeOracle::new(),
            EngineLoadTable::new(),
        )
        .unwrap();
        let p = reg.get(&ModelId("modelA".into())).unwrap();
        let dbg = format!("{p:?}");
        assert!(
            dbg.contains("StickyPolicy"),
            "expected StickyPolicy debug repr, got: {dbg}",
        );
    }

    #[test]
    fn sticky_fallback_builder_covers_all_typed_choices() {
        let workers = vec![worker("w0"), worker("w1")];
        let model = ModelId("modelA".into());
        let ctx = SelectionContext::new(&model, None);
        for kind in [
            StickyFallbackKind::RoundRobin,
            StickyFallbackKind::Random,
            StickyFallbackKind::PowerOfTwo,
            StickyFallbackKind::LoadBased,
        ] {
            assert!(
                build_sticky_fallback(kind).select(&workers, &ctx).is_some(),
                "{kind:?}"
            );
        }
    }
}
