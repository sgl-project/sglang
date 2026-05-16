use reqwest::Url;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    #[serde(default)]
    pub observability: ObservabilityConfig,
    pub models: Vec<ModelConfig>,
    pub workers: Vec<WorkerConfig>,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    /// Parsed at config-load time via [`reqwest::Url`] (re-export of
    /// `url::Url`). The deserialize hook below enforces:
    ///   * scheme is `http` or `https`
    ///   * the input is not the empty string
    ///
    /// We pick bare [`reqwest::Url`] (not a newtype) for now — there's no
    /// behaviour yet that diverges from a raw URL. If we later need to
    /// enforce extra invariants (e.g. no userinfo, no fragment), wrap in a
    /// `WorkerUrl(Url)` newtype at that point.
    ///
    /// Note on normalization: `url::Url::parse("http://x:30000")` yields a
    /// URL whose path is `/`, so its `as_str()` is `"http://x:30000/"`. The
    /// trailing slash is added by the URL parser; this is fine because we
    /// build downstream paths via [`reqwest::Url::join`], which replaces
    /// the path for absolute inputs (`/v1/...`) and never double-slashes.
    #[serde(deserialize_with = "deserialize_http_url")]
    pub url: Url,
    /// Total wall-clock timeout per non-streaming upstream request. Streaming
    /// endpoints do NOT honour this (long generations are valid). Connect
    /// timeout is a separate, hard-coded 5 s.
    /// Defaults to 60 s when missing.
    #[serde(default)]
    pub request_timeout_ms: Option<u64>,
}

/// Deserialize a worker URL with the validation we want at config-load
/// time: must be a syntactically valid URL AND its scheme must be `http`
/// or `https`. Anything else (empty string, missing scheme like
/// `"x:30000"` which would parse as scheme `"x"`, opaque schemes like
/// `file://`) is rejected with a useful message.
fn deserialize_http_url<'de, D>(d: D) -> Result<Url, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;
    let s = String::deserialize(d)?;
    if s.is_empty() {
        return Err(D::Error::custom("worker.url is empty"));
    }
    let url = Url::parse(&s).map_err(|e| D::Error::custom(format!("worker.url {s:?}: {e}")))?;
    match url.scheme() {
        "http" | "https" => Ok(url),
        other => Err(D::Error::custom(format!(
            "worker.url {s:?}: unsupported scheme {other:?}; want http or https"
        ))),
    }
}
