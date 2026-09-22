// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Launch configuration for complete buckets and their engine groups.

use anyhow::{ensure, Result};
use serde::Deserialize;
use std::collections::HashSet;

use crate::buckets_reorg::SloPreference;
use crate::policies_reorg::admission::AdmissionLimits;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReorgConfig {
    pub buckets: Vec<BucketConfig>,
    #[serde(default)]
    pub ttft_slo: SloPreference,
    #[serde(default)]
    pub tps_slo: SloPreference,
    #[serde(default)]
    pub session: SessionConfig,
    /// Omit to use the local radix tree. Shared by all cache-aware groups.
    pub kv_indexer: Option<IndexerConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BucketConfig {
    pub id: String,
    #[serde(default)]
    pub rank: u32,
    pub min_input_tokens: Option<u64>,
    pub max_input_tokens: Option<u64>,
    pub max_context_tokens: Option<u64>,
    pub ttft_ms: Option<u64>,
    pub tokens_per_second: Option<f64>,
    pub groups: GroupsConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum GroupsConfig {
    Plain {
        plain: GroupConfig,
    },
    Pd {
        prefill: GroupConfig,
        decode: GroupConfig,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupConfig {
    /// Omitted means all healthy workers for this model and role; [] means none.
    pub worker_ids: Option<Vec<String>>,
    pub policy: PolicyKind,
    #[serde(default)]
    pub admission: AdmissionLimits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyKind {
    PowerOfTwo,
    CacheAware,
    SessionAware,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SessionConfig {
    pub header: String,
    pub idle_secs: u64,
    pub eviction_interval_secs: u64,
}

impl Default for SessionConfig {
    fn default() -> Self {
        let defaults = super::AffinityConfig::default();
        Self {
            header: defaults.session_id_header,
            idle_secs: defaults.session_idle_secs,
            eviction_interval_secs: defaults.session_eviction_interval_secs,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexerConfig {
    pub url: String,
    #[serde(default = "default_query_timeout_ms")]
    pub query_timeout_ms: u64,
    #[serde(default = "default_query_max_inflight")]
    pub query_max_inflight: usize,
}

fn default_query_timeout_ms() -> u64 {
    100
}
fn default_query_max_inflight() -> usize {
    sgl_kv_indexer::DEFAULT_QUERY_MAX_INFLIGHT
}

impl ReorgConfig {
    pub fn uses(&self, policy: PolicyKind) -> bool {
        self.buckets.iter().any(|bucket| match &bucket.groups {
            GroupsConfig::Plain { plain } => plain.policy == policy,
            GroupsConfig::Pd { prefill, decode } => {
                prefill.policy == policy || decode.policy == policy
            }
        })
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(!self.buckets.is_empty(), "reorg buckets must be non-empty");
        let mut ids = HashSet::new();
        for bucket in &self.buckets {
            ensure!(
                !bucket.id.trim().is_empty() && ids.insert(&bucket.id),
                "reorg bucket IDs must be non-empty and unique: {:?}",
                bucket.id
            );
            ensure!(
                bucket
                    .min_input_tokens
                    .zip(bucket.max_input_tokens)
                    .is_none_or(|(min, max)| min <= max),
                "reorg bucket {:?}: min_input_tokens exceeds max_input_tokens",
                bucket.id
            );
            ensure!(
                bucket.max_context_tokens != Some(0),
                "reorg bucket {:?}: max_context_tokens must be positive",
                bucket.id
            );
            ensure!(
                bucket
                    .min_input_tokens
                    .zip(bucket.max_context_tokens)
                    .is_none_or(|(min, max)| min <= max),
                "reorg bucket {:?}: min_input_tokens exceeds max_context_tokens",
                bucket.id
            );
            ensure!(
                bucket.ttft_ms != Some(0),
                "reorg bucket {:?}: ttft_ms must be positive",
                bucket.id
            );
            ensure!(
                bucket
                    .tokens_per_second
                    .is_none_or(|v| v.is_finite() && v > 0.0),
                "reorg bucket {:?}: tokens_per_second must be finite and positive",
                bucket.id
            );
            let groups = match &bucket.groups {
                GroupsConfig::Plain { plain } => vec![plain],
                GroupsConfig::Pd { prefill, decode } => {
                    ensure!(
                        decode.policy != PolicyKind::CacheAware,
                        "reorg bucket {:?}: cache_aware cannot serve decode",
                        bucket.id
                    );
                    vec![prefill, decode]
                }
            };
            for group in groups {
                if let Some(workers) = &group.worker_ids {
                    let mut ids = HashSet::new();
                    ensure!(
                        workers
                            .iter()
                            .all(|id| !id.trim().is_empty() && ids.insert(id)),
                        "reorg bucket {:?}: worker IDs must be non-empty and unique within a group",
                        bucket.id
                    );
                }
            }
        }
        ensure!(
            self.session.idle_secs > 0 && self.session.eviction_interval_secs > 0,
            "reorg session timeouts must be positive"
        );
        self.session
            .header
            .parse::<axum::http::HeaderName>()
            .map_err(|e| anyhow::anyhow!("invalid reorg session header: {e}"))?;
        if let Some(indexer) = &self.kv_indexer {
            ensure!(
                self.uses(PolicyKind::CacheAware),
                "reorg kv_indexer requires a cache_aware group"
            );
            ensure!(
                indexer.query_timeout_ms > 0 && indexer.query_max_inflight > 0,
                "reorg KV indexer query limits must be positive"
            );
            let url = url::Url::parse(&indexer.url)?;
            ensure!(
                matches!(url.scheme(), "http" | "https") && url.host_str().is_some(),
                "reorg KV indexer URL must use http or https and include a host"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use serde_json::{json, Value};

    fn plain() -> Value {
        json!({"buckets": [{"id": "all", "groups": {
            "mode": "plain", "plain": {"policy": "power_of_two"}
        }}]})
    }

    #[test]
    fn cli_selects_reorg_and_leaves_legacy_as_default() {
        let args = [
            "router",
            "--model-id",
            "tiny",
            "--worker-urls",
            "http://localhost:30000",
        ];
        assert!(crate::config::Cli::try_parse_from(args)
            .unwrap()
            .into_config()
            .unwrap()
            .model
            .reorg
            .is_none());
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), plain().to_string()).unwrap();
        let mut reorg_args = args.to_vec();
        reorg_args.extend(["--reorg-config", file.path().to_str().unwrap()]);
        let cfg = crate::config::Cli::try_parse_from(&reorg_args)
            .unwrap()
            .into_config()
            .unwrap();
        assert!(cfg.model.reorg.unwrap().uses(PolicyKind::PowerOfTwo));
        for flags in [
            vec!["--policy", "power_of_two"],
            vec!["--decode-policy", "power_of_two"],
            vec!["--bucket-config", "unused.json"],
            vec!["--cache-candidate-ratio", "0.2"],
            vec!["--session-id-header", "x-session"],
            vec!["--filter", "overloaded", "--max-in-flight", "2"],
        ] {
            assert!(
                crate::config::Cli::try_parse_from(reorg_args.iter().copied().chain(flags))
                    .is_err()
            );
        }
    }

    #[test]
    fn rejects_malformed_or_unsupported_bucket_configuration() {
        let mut cases = vec![json!({"buckets": []})];
        for (pointer, value) in [
            ("/buckets/0/id", json!("")),
            ("/buckets/0/max_context_tokens", json!(0)),
            ("/buckets/0/ttft_ms", json!(0)),
            ("/buckets/0/tokens_per_second", json!(-1)),
            ("/buckets/0/groups/plain/worker_ids", json!(["w", "w"])),
            ("/buckets/0/groups/plain/policy", json!("round_robin")),
            (
                "/buckets/0/groups/plain/admission",
                json!({"max_running_request": 1}),
            ),
            ("/session", json!({"header": "invalid header"})),
            ("/session", json!({"idle_secs": 0})),
            ("/kv_indexer", json!({"url": "http://localhost:50051"})),
        ] {
            let mut value_to_check = plain();
            let (parent, key) = pointer.rsplit_once('/').unwrap();
            value_to_check
                .pointer_mut(parent)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .insert(key.into(), value);
            cases.push(value_to_check);
        }
        let mut inverted = plain();
        inverted["buckets"][0]["min_input_tokens"] = json!(10);
        inverted["buckets"][0]["max_input_tokens"] = json!(5);
        cases.push(inverted);
        let mut duplicate = plain();
        duplicate["buckets"]
            .as_array_mut()
            .unwrap()
            .push(plain()["buckets"][0].clone());
        cases.push(duplicate);
        cases.push(json!({"buckets": [{"id": "pd", "groups": {"mode": "pd", "prefill": {"policy": "power_of_two"}}}]}));
        cases.push(json!({"buckets": [{"id": "pd", "groups": {"mode": "pd", "prefill": {"policy": "power_of_two"}, "decode": {"policy": "cache_aware"}}}]}));
        for value in cases {
            let result = serde_json::from_value::<ReorgConfig>(value.clone())
                .map_err(anyhow::Error::from)
                .and_then(|cfg| cfg.validate());
            assert!(result.is_err(), "accepted {value}");
        }
    }
}
