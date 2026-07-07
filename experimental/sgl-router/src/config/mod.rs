pub mod cli;
pub mod types;
pub use cli::Cli;
pub use types::*;

use anyhow::{anyhow, Result};

/// The k8s default `terminationGracePeriodSeconds`. A `shutdown_drain_secs`
/// above this will be SIGKILLed mid-drain unless the operator has raised the
/// grace period — the opposite of what the drain is for.
const K8S_DEFAULT_GRACE_SECS: u64 = 30;

/// Advisory (not a hard error: the router can't read the pod's actual
/// `terminationGracePeriodSeconds`) for a `shutdown_drain_secs` that likely
/// exceeds the grace period. Returns `Some(message)` to warn, `None` if safe.
pub fn shutdown_drain_advisory(shutdown_drain_secs: u64) -> Option<String> {
    (shutdown_drain_secs > K8S_DEFAULT_GRACE_SECS).then(|| {
        format!(
            "shutdown_drain_secs={shutdown_drain_secs} exceeds the k8s default \
             terminationGracePeriodSeconds ({K8S_DEFAULT_GRACE_SECS}s); the pod will be \
             SIGKILLed mid-drain unless terminationGracePeriodSeconds is raised to at \
             least the drain plus in-flight request time"
        )
    })
}

/// Advisory (not a hard error: an ungated retry is functional, just not what
/// the `--enable-retry` help text promises) for retry enabled without an
/// admission cap: the retry's load gate has no capacity to check, so the
/// single re-dispatch proceeds even onto a busy worker. Returns
/// `Some(message)` to warn, `None` if safe.
pub fn retry_without_cap_advisory(
    retry: &RetryConfig,
    admission: &AdmissionConfig,
) -> Option<String> {
    (retry.enabled && matches!(admission, AdmissionConfig::Disabled)).then(|| {
        "retry is enabled without --max-concurrent-requests-per-worker; the retry's \
         load gate is a no-op, so a re-dispatch can land on a worker regardless of \
         its in-flight load. Set an admission cap to make retries load-gated."
            .to_string()
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
        if let Some(msg) = shutdown_drain_advisory(self.server.shutdown_drain_secs) {
            tracing::warn!("{msg}");
        }
        if let Some(msg) = retry_without_cap_advisory(&self.retry, &self.admission) {
            tracing::warn!("{msg}");
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
                ..Default::default()
            },
            observability: ObservabilityConfig::default(),
            model: ModelConfig {
                id: model_id.into(),
                tokenizer_path: "/tmp/tok.json".into(),
                tokenizer_shards: 1,
                policy: PolicyKind::RoundRobin,
                circuit_breaker: None,
                cache_aware: None,
                sticky: None,
            },
            discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
                urls: urls.iter().map(|s| s.to_string()).collect(),
            }),
            proxy: ProxyConfig::default(),
            active_load: ActiveLoadConfig::default(),
            admission: AdmissionConfig::default(),
            retry: RetryConfig::default(),
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
    fn shutdown_drain_advisory_is_silent_at_or_below_the_k8s_default_grace() {
        // The default drain (5 s) and any value at or under the k8s default
        // terminationGracePeriodSeconds (30 s) is safe without operator action.
        assert!(shutdown_drain_advisory(default_shutdown_drain_secs()).is_none());
        assert!(shutdown_drain_advisory(30).is_none());
        assert!(shutdown_drain_advisory(0).is_none());
    }

    #[test]
    fn retry_disabled_by_default() {
        assert!(!RetryConfig::default().enabled);
    }

    #[test]
    fn retry_without_cap_advisory_warns_only_for_enabled_retry_without_admission() {
        let enabled = RetryConfig { enabled: true };
        let disabled = RetryConfig { enabled: false };
        let cap = AdmissionConfig::Enabled {
            max_concurrent_per_worker: std::num::NonZeroUsize::new(4).unwrap(),
            max_queued_requests: None,
        };
        assert!(
            retry_without_cap_advisory(&enabled, &AdmissionConfig::Disabled).is_some(),
            "retry without a cap must produce the ungated-retry advisory"
        );
        assert!(retry_without_cap_advisory(&enabled, &cap).is_none());
        assert!(retry_without_cap_advisory(&disabled, &AdmissionConfig::Disabled).is_none());
        assert!(retry_without_cap_advisory(&disabled, &cap).is_none());
    }

    #[test]
    fn shutdown_drain_advisory_warns_above_the_k8s_default_grace() {
        // A drain past the 30 s k8s default will be SIGKILLed mid-drain unless
        // the operator has raised terminationGracePeriodSeconds — surface it.
        let msg = shutdown_drain_advisory(120).expect("120 s must produce an advisory");
        assert!(
            msg.contains("terminationGracePeriodSeconds"),
            "advisory must name the k8s knob to raise: {msg}"
        );
    }
}
