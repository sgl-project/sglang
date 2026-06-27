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
                    // Strip the optional `@min_priority=N` capability suffix
                    // BEFORE URL-parsing: the suffix is not part of the URL,
                    // and leaving it on would make `url::Url` misparse
                    // `host:port@min_priority=N` as userinfo (hiding a bad
                    // base URL) and would let `http://x` and
                    // `http://x@min_priority=100` dedupe as distinct. This
                    // also surfaces a malformed suffix (`@min_priority=abc`)
                    // at validate time, matching the discovery task's own
                    // parse. Same parser → single source of truth.
                    let (base, _min_priority) =
                        crate::discovery::static_urls::parse_worker_entry(trimmed).map_err(
                            |e| anyhow!("discovery.static_urls.urls entry {raw:?} is invalid: {e}"),
                        )?;
                    let parsed = url::Url::parse(&base).map_err(|e| {
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
                sticky: None,
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
    fn accepts_static_url_with_min_priority_suffix() {
        // The `@min_priority=N` capability suffix is stripped before URL
        // validation; a well-formed base URL + integer suffix is valid.
        cfg("qwen3", &["http://rtx-01:30000@min_priority=100"])
            .validate()
            .expect("valid base URL + integer min_priority suffix should pass");
    }

    #[test]
    fn rejects_static_url_malformed_min_priority_suffix() {
        // A non-integer suffix is a config typo that must fail at startup,
        // not silently drop the gate. Surfaced via the shared parser.
        let err = cfg("qwen3", &["http://rtx-01:30000@min_priority=abc"])
            .validate()
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("min_priority") || err.contains("invalid"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_static_url_bad_base_hidden_by_suffix() {
        // Regression: `http://host:notaport@min_priority=100`. With the
        // suffix attached, `url::Url` would misparse it as userinfo
        // (`host:notaport`) + host (`min_priority=100`) and PASS. Stripping
        // the suffix first exposes the real base `http://host:notaport`,
        // whose non-numeric port `url::Url` correctly rejects.
        let err = cfg("qwen3", &["http://host:notaport@min_priority=100"])
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("not a valid URL"), "got: {err}");
    }

    #[test]
    fn dedupes_static_url_base_across_min_priority_suffix() {
        // `http://x:30000` and `http://x:30000@min_priority=100` resolve to
        // the same base worker URL — registering both would point two
        // registry entries at one SGLang. Dedup must catch it after the
        // suffix is stripped (it would NOT if validation ran on the raw
        // suffixed string).
        let err = cfg(
            "qwen3",
            &["http://x:30000", "http://x:30000@min_priority=100"],
        )
        .validate()
        .unwrap_err()
        .to_string();
        assert!(err.contains("duplicate"), "got: {err}");
    }
}
