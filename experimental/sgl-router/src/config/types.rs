use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    #[serde(default)]
    pub observability: ObservabilityConfig,
    pub models: Vec<ModelConfig>,
    pub discovery: DiscoveryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

fn default_log_level() -> String {
    "info".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub id: String,
    pub tokenizer_path: String,
    #[serde(default = "default_policy")]
    pub policy: String,
    #[serde(default)]
    pub circuit_breaker: Option<CircuitBreakerConfig>,
}

fn default_policy() -> String {
    "round_robin".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    #[serde(default = "default_cb_threshold")]
    pub threshold: u32,
    #[serde(default = "default_cb_cool_down")]
    pub cool_down_secs: u64,
}

fn default_cb_threshold() -> u32 {
    3
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
        "discovery.k8s: PD `{which}_selector` must be a non-empty label selector — \
         an empty selector matches every endpoint and would silently classify every \
         worker as the same role"
    )]
    EmptyPdSelector { which: &'static str },
}

impl K8sDiscoveryConfig {
    /// Validate the selector combination and return the resolved mode.
    pub fn mode(&self) -> Result<K8sDiscoveryMode, ConfigError> {
        let plain = self.label_selector.as_deref();
        let prefill = self.prefill_selector.as_deref();
        let decode = self.decode_selector.as_deref();

        match (plain, prefill, decode) {
            (Some(label), None, None) => Ok(K8sDiscoveryMode::Plain {
                label_selector: label.to_string(),
            }),
            (None, Some(p), Some(d)) => {
                if p.is_empty() {
                    return Err(ConfigError::EmptyPdSelector { which: "prefill" });
                }
                if d.is_empty() {
                    return Err(ConfigError::EmptyPdSelector { which: "decode" });
                }
                Ok(K8sDiscoveryMode::PdDisaggregation {
                    prefill_selector: p.to_string(),
                    decode_selector: d.to_string(),
                })
            }
            (None, None, None) => Err(ConfigError::NoSelector),
            (None, Some(_), None) | (None, None, Some(_)) => Err(ConfigError::PartialPdSelectors),
            (Some(_), _, _) => Err(ConfigError::MixedModes),
        }
    }
}
