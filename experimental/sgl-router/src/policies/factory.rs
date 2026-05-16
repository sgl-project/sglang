// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::{Config, PolicyKind};
use crate::discovery::ModelId;
use crate::policies::{
    power_of_two::PowerOfTwoChoicesPolicy, random::RandomPolicy, round_robin::RoundRobinPolicy,
    Policy, PolicyRegistry,
};
use anyhow::Result;
use std::sync::Arc;

pub fn build_policy(kind: PolicyKind) -> Arc<dyn Policy> {
    match kind {
        PolicyKind::RoundRobin => Arc::new(RoundRobinPolicy::new()),
        PolicyKind::Random => Arc::new(RandomPolicy::new()),
        PolicyKind::PowerOfTwo => Arc::new(PowerOfTwoChoicesPolicy::new()),
    }
}

pub fn build_registry(cfg: &Config) -> Result<PolicyRegistry> {
    let reg = PolicyRegistry::default();
    for m in &cfg.models {
        reg.insert(ModelId(m.id.clone()), build_policy(m.policy));
    }
    Ok(reg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        Config, DiscoveryBackend, DiscoveryConfig, ModelConfig, ServerConfig,
        StaticFileDiscoveryConfig,
    };

    use crate::config::PolicyKind;

    fn cfg_with_models(policies: &[(&str, PolicyKind)]) -> Config {
        Config {
            server: ServerConfig {
                host: "0".into(),
                port: 0,
            },
            observability: Default::default(),
            models: policies
                .iter()
                .map(|(id, p)| ModelConfig {
                    id: (*id).into(),
                    tokenizer_path: "/tmp/x".into(),
                    policy: *p,
                    circuit_breaker: None,
                })
                .collect(),
            discovery: DiscoveryConfig {
                backend: DiscoveryBackend::StaticFile(StaticFileDiscoveryConfig {
                    path: "/tmp/w".into(),
                    poll_interval_ms: 200,
                }),
            },
        }
    }

    #[test]
    fn build_policy_covers_all_variants() {
        // Trivially total — the match is exhaustive over `PolicyKind`.
        let _ = build_policy(PolicyKind::RoundRobin);
        let _ = build_policy(PolicyKind::Random);
        let _ = build_policy(PolicyKind::PowerOfTwo);
    }

    #[test]
    fn registry_assigns_per_model() {
        let cfg = cfg_with_models(&[
            ("qwen", PolicyKind::RoundRobin),
            ("deepseek", PolicyKind::Random),
        ]);
        let reg = build_registry(&cfg).unwrap();
        assert!(reg.get(&ModelId("qwen".into())).is_some());
        assert!(reg.get(&ModelId("deepseek".into())).is_some());
        assert!(reg.get(&ModelId("missing".into())).is_none());
    }
}
