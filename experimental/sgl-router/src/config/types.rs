use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    #[serde(default)]
    pub observability: ObservabilityConfig,
    pub models: Vec<ModelConfig>,
    pub discovery: DiscoveryConfig,
    #[serde(default)]
    pub proxy: ProxyConfig,
    #[serde(default)]
    pub active_load: ActiveLoadConfig,
}

/// Outbound proxy tuning — controls how long the router waits on each
/// per-worker HTTP request. The default mirrors SGLang's typical
/// prefill / decode latency budget (long context windows take time);
/// e2e tests typically lower it so per-request failures surface fast
/// enough to trip the circuit breaker within the test's wall-time.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ProxyConfig {
    /// Maximum time to wait for a single upstream HTTP request to
    /// return headers + body. Default 300 s. The circuit breaker
    /// records a failure when this fires; the chat handler's caller
    /// observes `ApiError::UpstreamTimeout`.
    #[serde(default = "default_proxy_request_timeout_secs")]
    pub request_timeout_secs: u64,
}

fn default_proxy_request_timeout_secs() -> u64 {
    300
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            request_timeout_secs: default_proxy_request_timeout_secs(),
        }
    }
}

/// Active-load (per-request) tracking tuning — controls how long a
/// leaked or stalled request entry stays in the registry before the
/// janitor reclaims it. Setting it short in tests lets
/// `test_stale_request_expired_returns_504` fire the janitor within
/// the test's wall-time budget; production defaults to 10 minutes,
/// kept above `proxy.request_timeout_secs` so the proxy timeout is
/// the one users hit first for normal slow upstreams.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ActiveLoadConfig {
    /// How long a request entry can live in the registry before the
    /// janitor fires its `cancel_token` and the chat handler returns
    /// 504 `stale_request_expired`. Default 600 s.
    #[serde(default = "default_stale_request_timeout_secs")]
    pub stale_request_timeout_secs: u64,
}

fn default_stale_request_timeout_secs() -> u64 {
    600
}

impl Default for ActiveLoadConfig {
    fn default() -> Self {
        Self {
            stale_request_timeout_secs: default_stale_request_timeout_secs(),
        }
    }
}

/// Routing policy selector — the enum form lets serde reject unknown
/// values at deserialization time and removes the runtime string match in
/// the policy factory.
///
/// Serialised as `"round_robin"` / `"random"` / `"power_of_two"` /
/// `"cache_aware_zmq"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyKind {
    #[default]
    RoundRobin,
    Random,
    PowerOfTwo,
    /// Cache-aware routing fed by SGLang's ZMQ KV-cache event publisher.
    /// Requires the model to have a tokenizer loaded; cache_aware tuning
    /// lives on `ModelConfig::cache_aware`.
    CacheAwareZmq,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    #[serde(default = "default_log_level")]
    pub log_level: String,
    /// Selects the tracing-subscriber output format. Serde rejects
    /// unrecognized values at config-load (`"jsonl"` and similar
    /// plausible typos surface as an error instead of silently
    /// degrading to text), matching the discoverability pattern used
    /// by `policy` and `discovery.backend`.
    #[serde(default)]
    pub log_format: LogFormat,
}

/// `text` for human-readable dev output, `json` for one-line-per-record
/// JSON suitable for k8s log aggregators (fluent-bit / vector / Loki).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    #[default]
    Text,
    Json,
}

fn default_log_level() -> String {
    "info".to_string()
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            log_level: default_log_level(),
            log_format: LogFormat::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub id: String,
    pub tokenizer_path: String,
    #[serde(default)]
    pub policy: PolicyKind,
    #[serde(default)]
    pub circuit_breaker: Option<CircuitBreakerConfig>,
    /// Tuning for the cache-aware ZMQ policy. Ignored unless
    /// `policy = "cache_aware_zmq"`. `None` falls back to defaults at
    /// policy construction time.
    #[serde(default)]
    pub cache_aware: Option<CacheAwareConfig>,
}

/// Per-model cache-aware-ZMQ tuning.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CacheAwareConfig {
    /// Lower bound on `matched_blocks / total_blocks` for the tree match
    /// to win the selection. Below this, the policy falls back to
    /// min-load. Default 0.5 — a half-cached prompt is still a strong
    /// signal but not so weak that random hash collisions could trigger
    /// affinity to an arbitrary worker.
    #[serde(default = "default_cache_threshold")]
    pub cache_threshold: f32,
    /// Absolute load spread (`max - min`) above which the cache check is
    /// skipped in favour of min-load. Default 32 — picked to dominate
    /// over typical batch-of-8 effect.
    #[serde(default = "default_balance_abs")]
    pub balance_abs_threshold: usize,
    /// Multiplicative load spread (`max > min * balance_rel_threshold`)
    /// that the absolute check is gated on. Default 1.1 — 10 % relative
    /// difference triggers re-balancing.
    #[serde(default = "default_balance_rel")]
    pub balance_rel_threshold: f32,
}

impl Default for CacheAwareConfig {
    fn default() -> Self {
        Self {
            cache_threshold: default_cache_threshold(),
            balance_abs_threshold: default_balance_abs(),
            balance_rel_threshold: default_balance_rel(),
        }
    }
}

fn default_cache_threshold() -> f32 {
    0.5
}
fn default_balance_abs() -> usize {
    32
}
fn default_balance_rel() -> f32 {
    1.1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Consecutive failures required before the breaker opens.  Encoded
    /// as `NonZeroU32` so a config setting `threshold = 0` (which would
    /// open the breaker before any failure) is rejected at deserialization
    /// rather than silently behaving as "always open".
    #[serde(default = "default_cb_threshold")]
    pub threshold: NonZeroU32,
    #[serde(default = "default_cb_cool_down")]
    pub cool_down_secs: u64,
}

fn default_cb_threshold() -> NonZeroU32 {
    NonZeroU32::new(3).expect("3 is non-zero")
}
fn default_cb_cool_down() -> u64 {
    30
}

/// Config-level discovery section. Deserialized from:
///
/// TOML:
/// ```toml
/// [discovery]
/// backend = "static_file"
/// [discovery.static_file]
/// path = "/etc/experimental/sgl-router/workers.toml"
/// ```
///
/// YAML:
/// ```yaml
/// discovery:
///   backend: static_file
///   static_file:
///     path: /etc/experimental/sgl-router/workers.toml
/// ```
///
/// After deserialization `validate()` converts the raw fields into
/// `DiscoveryConfig::backend` for convenient pattern matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfigRaw {
    pub backend: String,
    pub static_file: Option<StaticFileDiscoveryConfig>,
    pub k8s: Option<K8sDiscoveryConfig>,
}

/// Post-validation discovery config with a resolved `DiscoveryBackend` enum.
/// Constructed by `Config::from_path` after `validate()`.
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    pub backend: DiscoveryBackend,
}

impl<'de> Deserialize<'de> for DiscoveryConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = DiscoveryConfigRaw::deserialize(deserializer)?;
        raw.try_into().map_err(serde::de::Error::custom)
    }
}

impl Serialize for DiscoveryConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let raw: DiscoveryConfigRaw = self.clone().into();
        raw.serialize(serializer)
    }
}

impl TryFrom<DiscoveryConfigRaw> for DiscoveryConfig {
    type Error = String;

    fn try_from(raw: DiscoveryConfigRaw) -> Result<Self, Self::Error> {
        let backend = match raw.backend.as_str() {
            "static_file" => {
                let s = raw.static_file.ok_or(
                    "discovery.backend = \"static_file\" requires [discovery.static_file] section",
                )?;
                DiscoveryBackend::StaticFile(s)
            }
            "k8s" => {
                let k = raw
                    .k8s
                    .ok_or("discovery.backend = \"k8s\" requires [discovery.k8s] section")?;
                DiscoveryBackend::K8s(k)
            }
            other => {
                return Err(format!(
                    "unknown discovery.backend = {other:?}; valid: \"static_file\", \"k8s\""
                ))
            }
        };
        Ok(DiscoveryConfig { backend })
    }
}

impl From<DiscoveryConfig> for DiscoveryConfigRaw {
    fn from(cfg: DiscoveryConfig) -> Self {
        match cfg.backend {
            DiscoveryBackend::StaticFile(s) => DiscoveryConfigRaw {
                backend: "static_file".to_string(),
                static_file: Some(s),
                k8s: None,
            },
            DiscoveryBackend::K8s(k) => DiscoveryConfigRaw {
                backend: "k8s".to_string(),
                static_file: None,
                k8s: Some(k),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum DiscoveryBackend {
    StaticFile(StaticFileDiscoveryConfig),
    K8s(K8sDiscoveryConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticFileDiscoveryConfig {
    pub path: String,
    #[serde(default = "default_poll_ms")]
    pub poll_interval_ms: u64,
}

fn default_poll_ms() -> u64 {
    200
}

/// Configuration for the Kubernetes `EndpointSlice` discovery backend.
///
/// Two operating modes, distinguished by which selector fields are set:
///
/// 1. **Plain** — all matched workers share the same role:
///    ```toml
///    [discovery.k8s]
///    namespace = "default"
///    label_selector = "app=sglang"
///    ```
///
/// 2. **PD disaggregation** — prefill and decode workers are separated by
///    different selectors:
///    ```toml
///    [discovery.k8s]
///    namespace = "default"
///    prefill_selector = "app=sglang,role=prefill"
///    decode_selector  = "app=sglang,role=decode"
///    ```
///
/// In PD mode, the selectors drive **slice-classification** (which
/// EndpointSlices feed the prefill pool vs the decode pool). The actual
/// `WorkerMode` and `bootstrap_port` for each worker are filled in by
/// the worker manager from each worker's `/server_info` introspection,
/// so PD works without any pod-level annotations — see
/// [`crate::workers::introspect`] for the `disaggregation_mode` and
/// `disaggregation_bootstrap_port` extraction.
///
/// `mode()` validates the combination and returns the resolved
/// [`K8sDiscoveryMode`]; any other selector combination is rejected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct K8sDiscoveryConfig {
    pub namespace: String,
    #[serde(default)]
    pub label_selector: Option<String>,
    #[serde(default)]
    pub prefill_selector: Option<String>,
    #[serde(default)]
    pub decode_selector: Option<String>,
}

/// Resolved discovery mode derived from a [`K8sDiscoveryConfig`].
///
/// The discovery backend uses this to:
/// * pick the server-side `LIST` label selector (Plain: the single selector;
///   PD: empty, with classification done client-side per slice), and
/// * assign each `EndpointSlice` a [`crate::discovery::WorkerMode`] in
///   `extract_workers`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum K8sDiscoveryMode {
    /// One global label selector; every matched EndpointSlice becomes a
    /// `WorkerMode::Plain` worker.
    Plain { label_selector: String },
    /// Two label selectors; an EndpointSlice's labels are matched against
    /// each to classify it as `WorkerMode::Prefill` or `WorkerMode::Decode`.
    PdDisaggregation {
        prefill_selector: String,
        decode_selector: String,
    },
}

/// Error returned by [`K8sDiscoveryConfig::mode`] when the selector
/// combination is invalid.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("discovery.k8s requires either `label_selector` (plain) or both `prefill_selector` and `decode_selector` (PD); none were set")]
    NoSelector,
    #[error("discovery.k8s: `label_selector` (plain) and `prefill_selector`/`decode_selector` (PD) are mutually exclusive — set one or the other, not both")]
    MixedModes,
    #[error("discovery.k8s: PD mode requires BOTH `prefill_selector` and `decode_selector`")]
    PartialPdSelectors,
    #[error(
        "discovery.k8s: {selector}_selector `{value}` uses unsupported syntax — \
         only equality terms (`key=value` or `key==value`) joined by `,` are accepted. \
         Set-based operators (`in`, `notin`), presence tests, and `!=` silently match \
         zero endpoints at runtime and are rejected at config-load time."
    )]
    UnsupportedSelectorGrammar {
        selector: &'static str,
        value: String,
    },
}

/// Returns `true` when `selector` parses as a comma-separated equality
/// selector — every term has the shape `key=value` or `key==value`.
/// See [`ConfigError::UnsupportedSelectorGrammar`] for rationale.
fn is_equality_selector(selector: &str) -> bool {
    for term in selector.split(',') {
        let term = term.trim();
        if term.is_empty() {
            // Treat lone trailing commas / whitespace as fine; the runtime
            // splitter ignores empty terms.
            continue;
        }
        if let Some((k, _)) = term.split_once("==") {
            if k.trim().is_empty() {
                return false;
            }
            continue;
        }
        if let Some((k, v)) = term.split_once('=') {
            // Reject `!=` (rendered as `key!` + `=value` by split_once).
            if k.trim().is_empty() || k.trim().ends_with('!') {
                return false;
            }
            // Reject leading whitespace-only values? An empty value is
            // technically legal in K8s — `label_selector = "tier="` matches
            // pods that explicitly set `tier=""`. Allow it.
            let _ = v;
            continue;
        }
        // No `=` at all → set-based operator, presence test, or garbage.
        return false;
    }
    true
}

impl K8sDiscoveryConfig {
    /// Validate the selector combination and return the resolved mode.
    pub fn mode(&self) -> Result<K8sDiscoveryMode, ConfigError> {
        let plain = self.label_selector.as_deref();
        let prefill = self.prefill_selector.as_deref();
        let decode = self.decode_selector.as_deref();

        match (plain, prefill, decode) {
            (Some(label), None, None) => {
                if !is_equality_selector(label) {
                    return Err(ConfigError::UnsupportedSelectorGrammar {
                        selector: "label",
                        value: label.to_string(),
                    });
                }
                Ok(K8sDiscoveryMode::Plain {
                    label_selector: label.to_string(),
                })
            }
            (None, Some(prefill), Some(decode)) => {
                // Both selectors validated individually so the operator
                // sees which one is malformed. WorkerMode + bootstrap_port
                // for each prefill pod are filled in by the worker
                // manager from each worker's `/server_info` — these
                // selectors only drive client-side classification per
                // EndpointSlice (see `classify_mode` in discovery/k8s.rs).
                if !is_equality_selector(prefill) {
                    return Err(ConfigError::UnsupportedSelectorGrammar {
                        selector: "prefill",
                        value: prefill.to_string(),
                    });
                }
                if !is_equality_selector(decode) {
                    return Err(ConfigError::UnsupportedSelectorGrammar {
                        selector: "decode",
                        value: decode.to_string(),
                    });
                }
                Ok(K8sDiscoveryMode::PdDisaggregation {
                    prefill_selector: prefill.to_string(),
                    decode_selector: decode.to_string(),
                })
            }
            (None, None, None) => Err(ConfigError::NoSelector),
            (None, Some(_), None) | (None, None, Some(_)) => Err(ConfigError::PartialPdSelectors),
            (Some(_), _, _) => Err(ConfigError::MixedModes),
        }
    }
}

#[cfg(test)]
mod k8s_discovery_config_tests {
    use super::*;

    fn cfg(plain: Option<&str>, prefill: Option<&str>, decode: Option<&str>) -> K8sDiscoveryConfig {
        K8sDiscoveryConfig {
            namespace: "ns".to_string(),
            label_selector: plain.map(str::to_string),
            prefill_selector: prefill.map(str::to_string),
            decode_selector: decode.map(str::to_string),
        }
    }

    #[test]
    fn mode_constructs_pd_disaggregation_from_prefill_and_decode_selectors() {
        // K8s PD now works without per-pod annotations: each worker's
        // `/server_info` carries `disaggregation_bootstrap_port`, and the
        // worker manager applies it post-discovery. The K8s config layer's
        // job is just to validate the selector combination.
        let m = cfg(None, Some("app=sglang,role=p"), Some("app=sglang,role=d"))
            .mode()
            .expect("PD mode is now valid");
        assert_eq!(
            m,
            K8sDiscoveryMode::PdDisaggregation {
                prefill_selector: "app=sglang,role=p".to_string(),
                decode_selector: "app=sglang,role=d".to_string(),
            }
        );
    }

    #[test]
    fn mode_pd_rejects_set_based_prefill_selector() {
        // Both PD selectors get the same equality-only grammar check as
        // the plain label_selector. A set-based prefill selector would
        // silently match zero pods at runtime → fail-fast at load.
        let err = cfg(None, Some("app in (sglang, vllm)"), Some("app=sglang"))
            .mode()
            .unwrap_err();
        assert!(
            matches!(
                err,
                ConfigError::UnsupportedSelectorGrammar {
                    selector: "prefill",
                    ..
                },
            ),
            "expected UnsupportedSelectorGrammar(prefill), got {err:?}",
        );
    }

    #[test]
    fn mode_pd_rejects_set_based_decode_selector() {
        let err = cfg(None, Some("app=sglang"), Some("app in (sglang, vllm)"))
            .mode()
            .unwrap_err();
        assert!(
            matches!(
                err,
                ConfigError::UnsupportedSelectorGrammar {
                    selector: "decode",
                    ..
                },
            ),
            "expected UnsupportedSelectorGrammar(decode), got {err:?}",
        );
    }

    #[test]
    fn mode_accepts_plain_with_equality_selector() {
        let m = cfg(Some("app=sglang"), None, None).mode().unwrap();
        assert_eq!(
            m,
            K8sDiscoveryMode::Plain {
                label_selector: "app=sglang".to_string()
            }
        );
    }

    #[test]
    fn mode_rejects_set_based_selector_in_plain_mode() {
        // labels_match_selector only handles `key=value` / `key==value`;
        // anything else (presence tests, set-based ops, inequality) silently
        // returns false at runtime, so zero pods match and discovery emits
        // an empty worker set with no diagnostic. Validate up front.
        let err = cfg(Some("app in (sglang, vllm)"), None, None)
            .mode()
            .unwrap_err();
        assert!(
            matches!(err, ConfigError::UnsupportedSelectorGrammar { .. }),
            "expected UnsupportedSelectorGrammar, got {err:?}"
        );
    }

    #[test]
    fn mode_rejects_presence_only_term() {
        let err = cfg(Some("app"), None, None).mode().unwrap_err();
        assert!(
            matches!(err, ConfigError::UnsupportedSelectorGrammar { .. }),
            "expected UnsupportedSelectorGrammar, got {err:?}"
        );
    }

    #[test]
    fn mode_rejects_negation_term() {
        let err = cfg(Some("app!=sglang"), None, None).mode().unwrap_err();
        assert!(
            matches!(err, ConfigError::UnsupportedSelectorGrammar { .. }),
            "expected UnsupportedSelectorGrammar, got {err:?}"
        );
    }

    #[test]
    fn mode_accepts_comma_separated_equality_terms() {
        // The canonical Plain-mode selector form: `key1=v1,key2=v2`.
        let m = cfg(Some("app=sglang,zone=us-east"), None, None)
            .mode()
            .unwrap();
        assert_eq!(
            m,
            K8sDiscoveryMode::Plain {
                label_selector: "app=sglang,zone=us-east".to_string()
            }
        );
    }
}
