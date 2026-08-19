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

/// Construct a policy for a single model from its [`ModelConfig`] and the
/// process-shared `HashTree` + `TokenizerRegistry` + `BlockSizeOracle`.
///
/// The tree, tokenizer registry, and oracle are only consulted by the
/// cache-aware-zmq variant; other policies ignore them. Callers building
/// all policies for the same process pass the same instances to every
/// model.
///
/// Fallible because `--policy fused_score` can name a term that turns out not
/// to be fusable — a startup error rather than a request-time surprise.
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
    // Built by the SAME constructor a `--policy` would use, so a filter gets
    // the real dependencies and there is no second name table. Whether it can
    // constrain at all is asked of the BUILT policy, never matched on the kind.
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

/// One policy of the named kind, so a `--filter` entry and a `--policy` of the
/// same name are built the same way off the same config -- `prefix_cache` reads
/// its floor here whichever list named it.
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
            let cache_cfg = model.cache_aware.unwrap_or_default();
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
            // From the eligibility config even for a `--fuse` term, so the two
            // halves of one signal cannot be configured apart.
            let share = (model.eligibility.as_ref()).and_then(|e| e.min_prefix_share);
            Arc::new(match share {
                Some(s) => p.with_min_share(s),
                None => p,
            })
        }
        PolicyKind::Overloaded => {
            // Unreachable in production: the CLI rejects the filter and the
            // flag without each other. It keeps the test shim constructible.
            let cap = (model.eligibility.as_ref())
                .and_then(|e| e.max_in_flight)
                .unwrap_or(usize::MAX);
            Arc::new(Overloaded::new(cap))
        }
    })
}

/// Build the composer for `--policy fused_score`.
///
/// Each term is built by `build_policy` itself, so terms get the same
/// dependencies real policies get and there is no second name table. A term
/// naming `fused_score` recurses once into an empty spec and stops there.
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
        let inner = build_policy(
            &m,
            Arc::clone(tree),
            Arc::clone(tokenizers),
            Arc::clone(oracle),
        )?;
        // Fusability is asked of the BUILT policy via `can_fuse()`, never
        // matched on `kind`: a name list here would be a second source of
        // truth and would go stale the first time someone adds a policy.
        // `FusedScorePolicy::new` re-checks; this one exists only to name the
        // CLI term, which its `{p:?}` message cannot.
        if !inner.can_fuse() {
            return Err(anyhow!("--fuse: `{}` does not support fusion", t.kind));
        }
        terms.push((inner, t.weight));
    }
    Ok(Arc::new(FusedScorePolicy::new(terms)?))
}

/// Compatibility shim used by tests + non-cache-aware code paths. Builds
/// a policy without wiring the cache-aware dependencies; rejects
/// `CacheAwareZmq` to keep the call sites that don't have a `HashTree` /
/// `TokenizerRegistry` to hand from accidentally compiling.
#[cfg(test)]
pub fn build_policy_kind_only(kind: PolicyKind) -> Result<Arc<dyn Policy>> {
    Ok(match kind {
        PolicyKind::RoundRobin => Arc::new(RoundRobinPolicy::new()),
        PolicyKind::Random => Arc::new(RandomPolicy::new()),
        PolicyKind::PowerOfTwo => Arc::new(PowerOfTwoChoicesPolicy::new()),
        PolicyKind::LoadBased => Arc::new(LoadBasedPolicy::new()),
        PolicyKind::CacheAwareZmq => {
            // Provide an empty tree + empty tokenizer registry + fresh
            // oracle so the test policy is constructible. Production
            // callers go through `build_policy` with the real
            // process-shared instances.
            Arc::new(CacheAwareZmqPolicy::new(
                crate::config::CacheAwareConfig::default(),
                Arc::new(HashTree::new()),
                Arc::new(TokenizerRegistry::default()),
                BlockSizeOracle::new(),
            ))
        }
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
        // The only genuinely dependency-BOUND kind: its terms live on
        // `ModelConfig`, which this constructor by definition does not have.
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

/// Convenience for tests + non-cache-aware callers: builds a registry with
/// a fresh, empty `HashTree` and an empty `TokenizerRegistry`. The
/// cache-aware-zmq policy will then degrade to min-load (no tokenizer +
/// no worker-published block size → fallback) — which is exactly what
/// the legacy tests assume.
///
/// Production callers go through [`build_registry`] with the real
/// process-shared instances.
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

    /// End to end through the factory -- the in-crate tests build a
    /// `Pipeline` by hand, so an arm that assembled one out of the wrong parts
    /// would pass every one of them. Both directions asserted: under the cap
    /// the inner policy still ranks, over it the Hold surfaces as "did not
    /// route" rather than the least-bad worker.
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

    /// The floor is what makes `prefix_cache` a filter at all, and a named
    /// filter that constrains nothing must fail at STARTUP rather than install
    /// a stage that admits everybody.
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
        // Trivially total — the match is exhaustive over `PolicyKind`.
        for kind in [
            PolicyKind::RoundRobin,
            PolicyKind::Random,
            PolicyKind::PowerOfTwo,
            PolicyKind::LoadBased,
            PolicyKind::CacheAwareZmq,
            PolicyKind::Sticky,
            // INVERTED (was asserted is_err): the `not_wired_yet` refusal it
            // pinned was a temporary state, not the contract. PLAN requires
            // prefix_cache usable standalone via `--policy prefix_cache`.
            PolicyKind::PrefixCache,
        ] {
            assert!(build_policy_kind_only(kind).is_ok(), "{kind:?}");
        }
        // FusedScore stays refused here permanently, and for a different
        // reason: its terms live on `ModelConfig`, which this constructor
        // does not take. `build_policy` is the door for it.
        assert!(build_policy_kind_only(PolicyKind::FusedScore).is_err());
    }

    /// The standalone path PLAN requires, exercised through `build_policy`
    /// rather than the dependency-free constructor, so the real `tree` and
    /// `oracle` are the ones threaded in.
    #[test]
    fn prefix_cache_builds_standalone_and_is_fusable() {
        let p = build_policy(
            &cfg_with_model("m", PolicyKind::PrefixCache).model,
            Arc::new(HashTree::new()),
            Arc::new(TokenizerRegistry::default()),
            BlockSizeOracle::new(),
        )
        .expect("--policy prefix_cache must build");
        // Not decoration: a `--fuse prefix_cache=W` term is admitted by
        // exactly this flag, so standalone-but-unfusable would be a silent
        // half-wiring that the is_ok() above cannot see.
        assert!(p.can_fuse(), "prefix_cache must be usable as a --fuse term");
    }

    /// The refusal must land while the registry is being built, not on the
    /// first request that happens to route.
    #[test]
    fn fused_score_refuses_a_non_fusable_term_at_startup() {
        let mut cfg = cfg_with_model("m", PolicyKind::FusedScore);
        let term = |kind, weight| crate::config::FusedTerm { kind, weight };

        // Positive half, from W3's `e864459d8d`: without it a gate that
        // refused EVERY term would pass the negative half below. `Some(2.5)`
        // is the only end-to-end cover of the `--fuse name=weight` override.
        for weight in [None, Some(2.5)] {
            cfg.model.fused = Some(vec![term(PolicyKind::PrefixCache, weight)]);
            assert!(build_registry_with_defaults(&cfg).is_ok(), "{weight:?}");
        }

        cfg.model.fused = Some(vec![term(PolicyKind::RoundRobin, None)]);
        let err = build_registry_with_defaults(&cfg).unwrap_err().to_string();
        assert!(err.contains("round_robin"), "{err}");
        assert!(err.contains("does not support fusion"), "{err}");

        // And an empty spec is refused too, rather than summing nothing.
        cfg.model.fused = Some(vec![]);
        assert!(build_registry_with_defaults(&cfg)
            .unwrap_err()
            .to_string()
            .contains("at least one --fuse term"));
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
        // Down-cast probe via Debug — cheaper than carrying a type-tag
        // on the trait. Pinning the debug repr is fine because the field
        // name is part of the file's public test surface.
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
