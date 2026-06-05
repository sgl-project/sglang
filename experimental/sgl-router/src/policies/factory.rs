// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::{Config, ModelConfig, PolicyKind};
use crate::discovery::ModelId;
use crate::policies::{
    cache_aware_zmq::CacheAwareZmqPolicy,
    kv_events::{BlockSizeOracle, HashTree},
    power_of_two::PowerOfTwoChoicesPolicy,
    random::RandomPolicy,
    round_robin::RoundRobinPolicy,
    Policy, PolicyRegistry,
};
use crate::tokenizer::TokenizerRegistry;
use anyhow::Result;
use std::sync::Arc;

/// Construct a policy for a single model from its [`ModelConfig`] and the
/// process-shared `HashTree` + `TokenizerRegistry` + `BlockSizeOracle`.
///
/// The tree, tokenizer registry, and oracle are only consulted by the
/// cache-aware-zmq variant; other policies ignore them. Callers building
/// all policies for the same process pass the same instances to every
/// model.
pub fn build_policy(
    model: &ModelConfig,
    tree: Arc<HashTree>,
    tokenizers: Arc<TokenizerRegistry>,
    block_size_oracle: Arc<BlockSizeOracle>,
) -> Arc<dyn Policy> {
    match model.policy {
        PolicyKind::RoundRobin => Arc::new(RoundRobinPolicy::new()),
        PolicyKind::Random => Arc::new(RandomPolicy::new()),
        PolicyKind::PowerOfTwo => Arc::new(PowerOfTwoChoicesPolicy::new()),
        PolicyKind::CacheAwareZmq => {
            let cache_cfg = model.cache_aware.unwrap_or_default();
            Arc::new(CacheAwareZmqPolicy::new(
                cache_cfg,
                tree,
                tokenizers,
                block_size_oracle,
            ))
        }
    }
}

/// Compatibility shim used by tests + non-cache-aware code paths. Builds
/// a policy without wiring the cache-aware dependencies; rejects
/// `CacheAwareZmq` to keep the call sites that don't have a `HashTree` /
/// `TokenizerRegistry` to hand from accidentally compiling.
#[cfg(test)]
pub fn build_policy_kind_only(kind: PolicyKind) -> Arc<dyn Policy> {
    match kind {
        PolicyKind::RoundRobin => Arc::new(RoundRobinPolicy::new()),
        PolicyKind::Random => Arc::new(RandomPolicy::new()),
        PolicyKind::PowerOfTwo => Arc::new(PowerOfTwoChoicesPolicy::new()),
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
    }
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
        ),
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

    use crate::config::PolicyKind;

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
        let _ = build_policy_kind_only(PolicyKind::RoundRobin);
        let _ = build_policy_kind_only(PolicyKind::Random);
        let _ = build_policy_kind_only(PolicyKind::PowerOfTwo);
        let _ = build_policy_kind_only(PolicyKind::CacheAwareZmq);
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
}
