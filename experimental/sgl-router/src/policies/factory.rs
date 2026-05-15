// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::Config;
use crate::discovery::ModelId;
use crate::policies::{
    power_of_two::PowerOfTwoChoicesPolicy, random::RandomPolicy, round_robin::RoundRobinPolicy,
    Policy, PolicyRegistry,
};
use anyhow::{anyhow, Result};
use std::sync::Arc;

pub fn build_policy(name: &str) -> Result<Arc<dyn Policy>> {
    match name {
        "round_robin" => Ok(Arc::new(RoundRobinPolicy::new())),
        "random" => Ok(Arc::new(RandomPolicy::new())),
        "power_of_two" => Ok(Arc::new(PowerOfTwoChoicesPolicy::new())),
        other => Err(anyhow!("unknown policy: {other}")),
    }
}

pub fn build_registry(cfg: &Config) -> Result<PolicyRegistry> {
    let reg = PolicyRegistry::default();
    for m in &cfg.models {
        let p = build_policy(&m.policy)?;
        reg.insert(ModelId(m.id.clone()), p);
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

    fn cfg_with_models(policies: &[(&str, &str)]) -> Config {
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
                    policy: (*p).into(),
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
    fn builds_three_policies() {
        assert!(build_policy("round_robin").is_ok());
        assert!(build_policy("random").is_ok());
        assert!(build_policy("power_of_two").is_ok());
    }

    #[test]
    fn rejects_unknown() {
        let err = build_policy("not_a_policy").unwrap_err();
        assert!(err.to_string().contains("not_a_policy"));
    }

    #[test]
    fn registry_assigns_per_model() {
        let cfg = cfg_with_models(&[("qwen", "round_robin"), ("deepseek", "random")]);
        let reg = build_registry(&cfg).unwrap();
        assert!(reg.get(&ModelId("qwen".into())).is_some());
        assert!(reg.get(&ModelId("deepseek".into())).is_some());
        assert!(reg.get(&ModelId("missing".into())).is_none());
    }
}
