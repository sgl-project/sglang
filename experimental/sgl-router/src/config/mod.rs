pub mod cli;
pub mod types;
pub use cli::Cli;
pub use types::*;

use anyhow::{anyhow, Result};

/// The k8s default `terminationGracePeriodSeconds`, assumed when the operator
/// has not declared the pod's real one. A `shutdown_drain_secs` at or above the
/// grace period leaves no time for the in-flight drain, so the pod is SIGKILLed
/// before it finishes — the opposite of what the drain is for.
pub const K8S_DEFAULT_GRACE_SECS: u64 = 30;

/// Ceiling on `shutdown_drain_secs`, enforced by [`Config::validate`]. Sized
/// for the workload rather than for the k8s default grace period: a single
/// streaming completion can hold the router for many minutes, so a deployment
/// that does not want terminations cutting one off runs a
/// `terminationGracePeriodSeconds` in the tens of minutes and a drain to match.
/// Deciding whether a particular drain fits a particular grace period is
/// [`shutdown_drain_advisory`]'s job — advice, because the operator can raise
/// the budget. This constant is the separate, harder gate: it rejects a value
/// that is not a drain at all, an extra digit or seconds confused with
/// milliseconds, which no grace period could ever service.
pub const MAX_SHUTDOWN_DRAIN_SECS: u64 = 1800;

/// A `shutdown_drain_secs` that leaves no room under the grace period for the
/// in-flight drain that follows the pause. Carries the compared values as
/// fields so the caller logs a static message with structured data rather than
/// interpolating the numbers into the message text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShutdownDrainAdvisory {
    pub shutdown_drain_secs: u64,
    /// The budget the drain was compared against.
    pub termination_grace_secs: u64,
    /// Whether that budget came from the operator or from
    /// [`K8S_DEFAULT_GRACE_SECS`]. An assumed budget makes the advisory a
    /// guess; a declared one makes it a fact.
    pub grace_declared: bool,
}

/// Advisory (not a hard error: the drain may well be right and the grace period
/// raised to match) for a drain that leaves no room for the in-flight drain.
/// The bound is `>=`, not `>`: a drain of exactly the grace period already
/// consumes all of it.
///
/// `termination_grace_secs` is the pod's real `terminationGracePeriodSeconds`
/// when the operator declared it. The router cannot read its own pod spec, so
/// `None` falls back to the k8s default — which is why declaring the real value
/// is the way to silence this on a deployment that raised the grace period
/// deliberately, rather than lowering a drain that was correct.
pub fn shutdown_drain_advisory(
    shutdown_drain_secs: u64,
    termination_grace_secs: Option<u64>,
) -> Option<ShutdownDrainAdvisory> {
    let grace = termination_grace_secs.unwrap_or(K8S_DEFAULT_GRACE_SECS);
    (shutdown_drain_secs >= grace).then_some(ShutdownDrainAdvisory {
        shutdown_drain_secs,
        termination_grace_secs: grace,
        grace_declared: termination_grace_secs.is_some(),
    })
}

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
        if self.server.shutdown_drain_secs > MAX_SHUTDOWN_DRAIN_SECS {
            return Err(anyhow!(
                "shutdown_drain_secs must be at most {MAX_SHUTDOWN_DRAIN_SECS} (got {}); \
                 past the ceiling a value is a typo rather than a drain, and the pod \
                 would be SIGKILLed long before the pause elapsed. A long but deliberate \
                 drain is fine — declare --termination-grace-secs so startup can check \
                 it against the pod's real budget",
                self.server.shutdown_drain_secs,
            ));
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
            server: ServerConfig::default(),
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

    #[test]
    fn shutdown_drain_advisory_is_silent_below_the_k8s_default_grace() {
        // Anything strictly under the k8s default terminationGracePeriodSeconds
        // (30 s) still leaves room for the in-flight drain, so it is safe
        // without operator action.
        assert!(shutdown_drain_advisory(29, None).is_none());
        assert!(shutdown_drain_advisory(0, None).is_none());
    }

    /// The default drain is exactly the assumed grace period, so out of the box
    /// it leaves nothing for the in-flight drain and says so on every startup.
    /// Asserted rather than left implicit because it is the one case an
    /// operator meets without choosing it: a later edit to either constant that
    /// silenced the warning would be changing the default deployment's
    /// behaviour, and should have to say so here.
    #[test]
    fn the_default_drain_warns_until_the_grace_period_is_raised() {
        let advisory = shutdown_drain_advisory(default_shutdown_drain_secs(), None)
            .expect("the default drain must warn against the assumed k8s grace period");
        assert_eq!(advisory.termination_grace_secs, K8S_DEFAULT_GRACE_SECS);
        assert!(
            !advisory.grace_declared,
            "an assumed budget must not be reported as declared",
        );
        // Raising the pod's grace period past the drain is what silences it —
        // the action the warning asks for has to actually work.
        assert!(
            shutdown_drain_advisory(
                default_shutdown_drain_secs(),
                Some(K8S_DEFAULT_GRACE_SECS * 2),
            )
            .is_none(),
            "a grace period declared with room for the in-flight drain must silence it",
        );
    }

    #[test]
    fn shutdown_drain_advisory_warns_once_the_drain_consumes_the_whole_grace() {
        // A drain of exactly the 30 s k8s default leaves zero seconds for the
        // in-flight drain, so the pod is SIGKILLed mid-drain — the boundary
        // itself must warn, not just values past it. The ceiling is in the list
        // because it is startable: `validate` accepts it, so the advisory is
        // the only thing left to say it does not fit the default grace period.
        for drain in [K8S_DEFAULT_GRACE_SECS, 120, MAX_SHUTDOWN_DRAIN_SECS] {
            let advisory = shutdown_drain_advisory(drain, None)
                .unwrap_or_else(|| panic!("{drain}s must warn"));
            assert_eq!(advisory.shutdown_drain_secs, drain);
            assert_eq!(advisory.termination_grace_secs, K8S_DEFAULT_GRACE_SECS);
            assert!(
                !advisory.grace_declared,
                "an undeclared grace period must be reported as assumed, not as fact",
            );
        }
    }

    /// The advisory's whole purpose is to be silenceable by declaring the real
    /// budget: a 60 s drain under a 120 s grace period is a correct
    /// configuration, and warning about it trains operators to ignore the line.
    #[test]
    fn shutdown_drain_advisory_respects_a_declared_grace_period() {
        assert!(
            shutdown_drain_advisory(60, Some(120)).is_none(),
            "a drain with room under the declared grace period must not warn",
        );
        let advisory = shutdown_drain_advisory(60, Some(60))
            .expect("a drain consuming the whole declared grace period must warn");
        assert_eq!(advisory.termination_grace_secs, 60);
        assert!(
            advisory.grace_declared,
            "a declared grace period must be reported as declared",
        );
        // ...and declaring a *shorter* budget than the k8s default must be able
        // to warn about a drain the default would have waved through.
        assert!(
            shutdown_drain_advisory(10, Some(10)).is_some(),
            "a short declared grace period must still be compared against",
        );
        // The configuration the ceiling was raised for: a completion streaming
        // for minutes wants a drain of minutes, under a grace period declared
        // to match. That is correct, not merely tolerated, so it must be silent.
        assert!(
            shutdown_drain_advisory(MAX_SHUTDOWN_DRAIN_SECS, Some(3600)).is_none(),
            "a long drain under a grace period declared to cover it must not warn",
        );
    }

    /// `validate` is the hard gate the advisory deliberately is not: past the
    /// ceiling the value can only be a typo, and starting on it would make
    /// every later termination a SIGKILL.
    #[test]
    fn validate_rejects_a_shutdown_drain_past_the_ceiling() {
        let mut config = cfg("qwen3-0.6b", &["http://10.0.0.1:30000"]);
        config.server.shutdown_drain_secs = MAX_SHUTDOWN_DRAIN_SECS;
        config
            .validate()
            .expect("the ceiling itself must remain startable");

        config.server.shutdown_drain_secs = MAX_SHUTDOWN_DRAIN_SECS + 1;
        let error = config
            .validate()
            .expect_err("a drain past the ceiling must fail startup")
            .to_string();
        assert!(
            error.contains("shutdown_drain_secs"),
            "the error must name the flag to fix: {error}"
        );
    }
}
