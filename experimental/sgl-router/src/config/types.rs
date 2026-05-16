use reqwest::Url;
use serde::{Deserialize, Deserializer, Serialize};
use std::time::Duration;

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
    ///
    /// Accepts any `humantime`-formatted duration: `"60s"`, `"500ms"`,
    /// `"2m"`, `"1m30s"`. Reject-on-load for malformed input (`"infinity"`,
    /// negative values) — typos in production configs should fail loudly at
    /// startup, not silently flip to a default.
    ///
    /// `None` (field absent) means the caller picks a default (currently
    /// 60 s in `main.rs`); tests can construct a `WorkerConfig` with
    /// `request_timeout: None` without forcing a magic-number choice here.
    #[serde(default, with = "humantime_serde")]
    pub request_timeout: Option<Duration>,
}

/// Deserialize a worker URL with the validation we want at config-load
/// time:
///   * syntactically valid URL
///   * scheme is `http` or `https`
///   * no non-trivial path (i.e. path is `/` or empty after normalization)
///   * no query string
///   * no fragment
///
/// The path/query/fragment restrictions exist because downstream code builds
/// upstream URLs via [`Url::join`] with an absolute path (`"/v1/..."`), which
/// silently replaces any base path and drops the query. Allowing
/// `http://x:30000/api/?key=foo` at config-load would let an operator put a
/// load-balancer prefix here and have every request silently routed away
/// from it.
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
        "http" | "https" => {}
        other => {
            return Err(D::Error::custom(format!(
                "worker.url {s:?}: unsupported scheme {other:?}; want http or https"
            )))
        }
    }
    // `url::Url::parse("http://x:30000")` normalizes to a path of `/`, so
    // anything beyond that is an operator-supplied path prefix that would
    // be lost on `Url::join("/v1/...")`.
    if url.path() != "/" && !url.path().is_empty() {
        return Err(D::Error::custom(format!(
            "worker.url {s:?}: path {:?} is not allowed; \
             provide only scheme://host[:port] (path prefix would be dropped on join)",
            url.path()
        )));
    }
    if url.query().is_some() {
        return Err(D::Error::custom(format!(
            "worker.url {s:?}: query string is not allowed; \
             provide only scheme://host[:port] (query would be dropped on join)"
        )));
    }
    if url.fragment().is_some() {
        return Err(D::Error::custom(format!(
            "worker.url {s:?}: fragment is not allowed; \
             provide only scheme://host[:port]"
        )));
    }
    Ok(url)
}
