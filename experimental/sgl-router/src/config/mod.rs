pub mod cli;
pub mod types;
pub use cli::Cli;
pub use types::*;

use anyhow::{anyhow, Result};

impl Config {
    /// Check invariants the type system and `clap` don't already enforce.
    /// Called by [`cli::Cli::into_config`] after assembling the `Config`
    /// from flags. Unknown policy names and `--cb-threshold 0` are
    /// rejected at parse time (`ValueEnum` / `NonZeroU32`); only the
    /// remaining value-level invariants are checked here.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.model.id.is_empty() {
            return Err(anyhow!("model id must be non-empty"));
        }
        if let Some(bucket_config) = self.model.bucket_config.as_ref() {
            validate_bucket_config(bucket_config)?;
        }
        match &self.discovery {
            DiscoveryBackend::StaticUrls(s) => {
                if s.urls.is_empty() {
                    return Err(anyhow!(
                        "discovery.static_urls.urls must be a non-empty list"
                    ));
                }
                // Validate every entry up front so typos surface at
                // startup with a precise diagnostic instead of as
                // per-worker introspect failures or as two registry
                // entries pointing at the same SGLang (trailing-slash
                // near-duplicates). Dedupe runs against a normalized
                // form (trimmed + trailing `/` stripped) so
                // `"http://x:30000"` and `"http://x:30000/"` collide.
                let mut seen = std::collections::HashSet::new();
                for raw in &s.urls {
                    let trimmed = raw.trim();
                    if trimmed.is_empty() {
                        return Err(anyhow!(
                            "discovery.static_urls.urls contains an empty or whitespace-only entry"
                        ));
                    }
                    let parsed = url::Url::parse(trimmed).map_err(|e| {
                        anyhow!("discovery.static_urls.urls entry {raw:?} is not a valid URL: {e}")
                    })?;
                    match parsed.scheme() {
                        "http" | "https" => {}
                        other => {
                            return Err(anyhow!(
                                "discovery.static_urls.urls entry {raw:?} has unsupported scheme {other:?}; only http and https are supported"
                            ));
                        }
                    }
                    let normalized = parsed.as_str().trim_end_matches('/').to_string();
                    if !seen.insert(normalized.clone()) {
                        return Err(anyhow!(
                            "discovery.static_urls.urls contains duplicate entry {raw:?} (normalized: {normalized:?})"
                        ));
                    }
                }
            }
            // K8s selector validity is resolved at construction time
            // (`resolve_mode` in `Cli::build_discovery`), so the stored
            // `K8sDiscoveryMode` is already valid here. Any namespace
            // (including empty, for a cluster-wide watch) is accepted.
            DiscoveryBackend::K8s(_) => {}
        }
        Ok(())
    }
}

fn validate_bucket_config(bucket_config: &BucketConfig) -> Result<()> {
    if bucket_config.buckets.is_empty() {
        return Err(anyhow!(
            "bucket_config.buckets must be non-empty when configured"
        ));
    }
    let mut ids = std::collections::HashSet::new();
    let mut ranks = std::collections::HashSet::new();
    let mut stage_workers = std::collections::HashSet::new();
    let mut has_prefill_bucket = false;
    for bucket in &bucket_config.buckets {
        has_prefill_bucket |= bucket.stage == BucketStage::Prefill;
        if bucket.id.is_empty() || !ids.insert(bucket.id.as_str()) {
            return Err(anyhow!(
                "bucket_config bucket id must be non-empty and unique: {:?}",
                bucket.id
            ));
        }
        if !ranks.insert((bucket.stage, bucket.rank)) {
            return Err(anyhow!(
                "bucket_config rank must be unique within each stage: {}",
                bucket.rank
            ));
        }
        if bucket.worker_ids.is_empty() {
            return Err(anyhow!(
                "bucket_config bucket {:?} has no worker_ids",
                bucket.id
            ));
        }
        let mut worker_ids = std::collections::HashSet::new();
        for worker_id in &bucket.worker_ids {
            if worker_id.is_empty() || !worker_ids.insert(worker_id.as_str()) {
                return Err(anyhow!(
                    "bucket_config bucket {:?} has an empty or duplicate worker id",
                    bucket.id
                ));
            }
            if !stage_workers.insert((bucket.stage, worker_id.as_str())) {
                return Err(anyhow!(
                    "bucket_config worker {:?} belongs to more than one {:?} bucket",
                    worker_id,
                    bucket.stage
                ));
            }
        }
        validate_range(
            bucket.min_extend_tokens,
            bucket.max_extend_tokens,
            &bucket.id,
            "extend",
        )?;
        validate_range(
            bucket.min_sequence_tokens,
            bucket.max_sequence_tokens,
            &bucket.id,
            "sequence",
        )?;
        if bucket.max_context_tokens == Some(0) {
            return Err(anyhow!(
                "bucket_config bucket {:?} max_context_tokens must be > 0",
                bucket.id
            ));
        }
        if bucket.ttft_p95_at_capacity_ms == Some(0) {
            return Err(anyhow!(
                "bucket_config bucket {:?} TTFT p95 must be > 0",
                bucket.id
            ));
        }
        if bucket
            .tps_p05_at_capacity
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(anyhow!(
                "bucket_config bucket {:?} TPS p05 must be finite and > 0",
                bucket.id
            ));
        }
        if bucket.max_pending_prefill_tokens == Some(0) {
            return Err(anyhow!(
                "bucket_config bucket {:?} max_pending_prefill_tokens must be > 0",
                bucket.id
            ));
        }
        match bucket.stage {
            BucketStage::Prefill
                if bucket.min_sequence_tokens.is_some()
                    || bucket.max_sequence_tokens.is_some()
                    || bucket.tps_p05_at_capacity.is_some() =>
            {
                return Err(anyhow!(
                    "bucket_config Prefill bucket {:?} contains Decode-only sequence/TPS fields",
                    bucket.id
                ));
            }
            BucketStage::Decode
                if bucket.min_extend_tokens.is_some()
                    || bucket.max_extend_tokens.is_some()
                    || bucket.ttft_p95_at_capacity_ms.is_some()
                    || bucket.max_pending_prefill_tokens.is_some() =>
            {
                return Err(anyhow!(
                    "bucket_config Decode bucket {:?} contains Prefill-only extend/TTFT/pending fields",
                    bucket.id
                ));
            }
            _ => {}
        }
    }
    if !has_prefill_bucket {
        return Err(anyhow!(
            "bucket_config must contain at least one Prefill bucket; enabling Bucket routing otherwise leaves every request without a Prefill domain"
        ));
    }
    Ok(())
}

fn validate_range(min: Option<u64>, max: Option<u64>, id: &str, name: &str) -> Result<()> {
    if min.zip(max).is_some_and(|(min, max)| min > max) {
        return Err(anyhow!(
            "bucket_config bucket {id:?} has invalid {name} range: min > max"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid-shape `Config` with the given static worker
    /// URLs and model id, so the `validate()` branches can be exercised
    /// directly. CLI parsing and the static-vs-k8s mapping are covered in
    /// the `cli` module tests; the k8s selector grammar in `types`.
    fn cfg(model_id: &str, urls: &[&str]) -> Config {
        Config {
            server: ServerConfig {
                host: "127.0.0.1".into(),
                port: 30000,
            },
            observability: ObservabilityConfig::default(),
            model: ModelConfig {
                id: model_id.into(),
                tokenizer_path: "/tmp/tok.json".into(),
                policy: PolicyKind::RoundRobin,
                decode_policy: DecodePolicyKind::PowerOfTwo,
                bucket_config: None,
                circuit_breaker: None,
                cache_aware: None,
                sticky: None,
                affinity: None,
                fused: None,
                eligibility: None,
            },
            discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
                urls: urls.iter().map(|s| s.to_string()).collect(),
            }),
            proxy: ProxyConfig::default(),
            active_load: ActiveLoadConfig::default(),
        }
    }

    #[test]
    fn accepts_minimal_static_config() {
        cfg("qwen3", &["http://10.0.0.1:30000"]).validate().unwrap();
    }

    #[test]
    fn rejects_empty_model_id() {
        let err = cfg("", &["http://10.0.0.1:30000"])
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("model id"), "got: {err}");
    }

    #[test]
    fn rejects_empty_static_urls_list() {
        let err = cfg("qwen3", &[]).validate().unwrap_err().to_string();
        assert!(err.contains("non-empty"), "got: {err}");
    }

    #[test]
    fn rejects_static_urls_empty_entry() {
        let err = cfg("qwen3", &["http://x:30000", ""])
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn rejects_static_urls_whitespace_only_entry() {
        let err = cfg("qwen3", &["http://x:30000", "   "])
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty or whitespace"), "got: {err}");
    }

    #[test]
    fn rejects_static_urls_trailing_slash_near_duplicate() {
        let err = cfg("qwen3", &["http://x:30000", "http://x:30000/"])
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("duplicate"), "got: {err}");
    }

    #[test]
    fn rejects_static_urls_non_http_scheme() {
        let err = cfg("qwen3", &["ws://x:30000"])
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported scheme"), "got: {err}");
    }

    #[test]
    fn rejects_a_worker_reused_by_two_buckets_of_the_same_stage() {
        let mut config = cfg("qwen3", &["http://x:30000"]);
        config.model.bucket_config = Some(BucketConfig {
            buckets: vec![
                BucketSpec {
                    id: "p-short".into(),
                    stage: BucketStage::Prefill,
                    rank: 10,
                    worker_ids: vec!["p1".into()],
                    min_extend_tokens: None,
                    max_extend_tokens: Some(1_024),
                    min_sequence_tokens: None,
                    max_sequence_tokens: None,
                    max_context_tokens: Some(4_096),
                    ttft_p95_at_capacity_ms: Some(100),
                    tps_p05_at_capacity: None,
                    max_pending_prefill_tokens: None,
                },
                BucketSpec {
                    id: "p-long".into(),
                    stage: BucketStage::Prefill,
                    rank: 20,
                    worker_ids: vec!["p1".into()],
                    min_extend_tokens: Some(1_025),
                    max_extend_tokens: None,
                    min_sequence_tokens: None,
                    max_sequence_tokens: None,
                    max_context_tokens: Some(8_192),
                    ttft_p95_at_capacity_ms: Some(200),
                    tps_p05_at_capacity: None,
                    max_pending_prefill_tokens: None,
                },
            ],
            ttft_slo_policy: SloBucketPolicy::SloFirst,
            tps_slo_policy: SloBucketPolicy::Disabled,
        });

        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("more than one"), "got: {error}");
    }

    #[test]
    fn accepts_the_same_rank_in_independent_prefill_and_decode_stages() {
        let mut config = cfg("qwen3", &["http://x:30000"]);
        config.model.bucket_config = Some(BucketConfig {
            buckets: vec![
                BucketSpec {
                    id: "p-fast".into(),
                    stage: BucketStage::Prefill,
                    rank: 10,
                    worker_ids: vec!["p1".into()],
                    min_extend_tokens: None,
                    max_extend_tokens: None,
                    min_sequence_tokens: None,
                    max_sequence_tokens: None,
                    max_context_tokens: Some(4_096),
                    ttft_p95_at_capacity_ms: Some(100),
                    tps_p05_at_capacity: None,
                    max_pending_prefill_tokens: None,
                },
                BucketSpec {
                    id: "d-fast".into(),
                    stage: BucketStage::Decode,
                    rank: 10,
                    worker_ids: vec!["d1".into()],
                    min_extend_tokens: None,
                    max_extend_tokens: None,
                    min_sequence_tokens: None,
                    max_sequence_tokens: None,
                    max_context_tokens: Some(4_096),
                    ttft_p95_at_capacity_ms: None,
                    tps_p05_at_capacity: Some(20.0),
                    max_pending_prefill_tokens: None,
                },
            ],
            ttft_slo_policy: SloBucketPolicy::SloFirst,
            tps_slo_policy: SloBucketPolicy::SloFirst,
        });

        config
            .validate()
            .expect("Prefill and Decode ranks only need to be unique within their stage");
    }

    #[test]
    fn rejects_bucket_config_without_a_prefill_domain() {
        let mut config = cfg("qwen3", &["http://x:30000"]);
        config.model.bucket_config = Some(BucketConfig {
            buckets: vec![BucketSpec {
                id: "d-only".into(),
                stage: BucketStage::Decode,
                rank: 10,
                worker_ids: vec!["d1".into()],
                min_extend_tokens: None,
                max_extend_tokens: None,
                min_sequence_tokens: None,
                max_sequence_tokens: None,
                max_context_tokens: Some(4_096),
                ttft_p95_at_capacity_ms: None,
                tps_p05_at_capacity: Some(20.0),
                max_pending_prefill_tokens: None,
            }],
            ttft_slo_policy: SloBucketPolicy::Disabled,
            tps_slo_policy: SloBucketPolicy::SloFirst,
        });

        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("Prefill bucket"), "got: {error}");
    }

    #[test]
    fn rejects_stage_inapplicable_bucket_fields_instead_of_ignoring_them() {
        let mut prefill = cfg("qwen3", &["http://x:30000"]);
        prefill.model.bucket_config = Some(BucketConfig {
            buckets: vec![BucketSpec {
                id: "p".into(),
                stage: BucketStage::Prefill,
                rank: 10,
                worker_ids: vec!["p1".into()],
                min_extend_tokens: None,
                max_extend_tokens: None,
                min_sequence_tokens: None,
                max_sequence_tokens: Some(4_096),
                max_context_tokens: Some(4_096),
                ttft_p95_at_capacity_ms: Some(100),
                tps_p05_at_capacity: None,
                max_pending_prefill_tokens: None,
            }],
            ttft_slo_policy: SloBucketPolicy::SloFirst,
            tps_slo_policy: SloBucketPolicy::Disabled,
        });
        let error = prefill.validate().unwrap_err().to_string();
        assert!(error.contains("Decode-only"), "got: {error}");

        let mut decode = cfg("qwen3", &["http://x:30000"]);
        decode.model.bucket_config = Some(BucketConfig {
            buckets: vec![
                BucketSpec {
                    id: "p".into(),
                    stage: BucketStage::Prefill,
                    rank: 10,
                    worker_ids: vec!["p1".into()],
                    min_extend_tokens: None,
                    max_extend_tokens: None,
                    min_sequence_tokens: None,
                    max_sequence_tokens: None,
                    max_context_tokens: Some(4_096),
                    ttft_p95_at_capacity_ms: Some(100),
                    tps_p05_at_capacity: None,
                    max_pending_prefill_tokens: None,
                },
                BucketSpec {
                    id: "d".into(),
                    stage: BucketStage::Decode,
                    rank: 20,
                    worker_ids: vec!["d1".into()],
                    min_extend_tokens: Some(1),
                    max_extend_tokens: None,
                    min_sequence_tokens: None,
                    max_sequence_tokens: Some(4_096),
                    max_context_tokens: Some(4_096),
                    ttft_p95_at_capacity_ms: None,
                    tps_p05_at_capacity: Some(20.0),
                    max_pending_prefill_tokens: None,
                },
            ],
            ttft_slo_policy: SloBucketPolicy::SloFirst,
            tps_slo_policy: SloBucketPolicy::SloFirst,
        });
        let error = decode.validate().unwrap_err().to_string();
        assert!(error.contains("Prefill-only"), "got: {error}");
    }

    #[test]
    fn bucket_json_rejects_unknown_profile_fields() {
        let raw = r#"{
          "buckets": [{
            "id": "p-fast",
            "stage": "prefill",
            "rank": 10,
            "worker_ids": ["p1"],
            "ttft_p95_at_capcity_ms": 100
          }]
        }"#;

        let error = serde_json::from_str::<BucketConfig>(raw)
            .expect_err("a misspelled capacity profile must fail startup")
            .to_string();
        assert!(error.contains("ttft_p95_at_capcity_ms"), "got: {error}");
    }
}
