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
            },
            observability: ObservabilityConfig::default(),
            model: ModelConfig {
                id: model_id.into(),
                tokenizer_path: "/tmp/tok.json".into(),
                policy: PolicyKind::RoundRobin,
                circuit_breaker: None,
                cache_aware: None,
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
}
