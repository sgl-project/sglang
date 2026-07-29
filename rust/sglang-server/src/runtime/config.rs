//! Runtime configuration: the rust-server boot knobs
//! ([`RustServerServerArgs`]), the typed view of the scheduler's `server_args`
//! dump ([`ServerArgs`] / [`ModelConfig`]), and the [`RuntimeConfig`] pairing
//! them for `runtime::start`.

use std::fmt;
use std::net::SocketAddr;
use std::sync::Arc;

/// Boot knobs specific to the embedded rust server — none of these exist in
/// the Python `server_args` dump (see [`ServerArgs`]); they arrive as explicit
/// `Server::start` parameters.
#[derive(Clone, Debug)]
pub struct RustServerServerArgs {
    pub http_addr: SocketAddr,
    pub api_worker_num: usize,
    pub ingress_ring_cap: usize,
    pub egress_ring_cap: usize,
    pub channel_cap: usize,
    /// CPU core ids the pools pin to (e.g. this rank's NUMA-local cores minus
    /// the scheduler's reserved launch cores). `None` → run unpinned.
    pub cores: Option<Vec<usize>>,
}

impl Default for RustServerServerArgs {
    fn default() -> Self {
        Self {
            http_addr: "127.0.0.1:30000".parse().unwrap(),
            api_worker_num: 2,
            ingress_ring_cap: 8192,
            egress_ring_cap: 8192,
            channel_cap: 8192,
            cores: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    /// Rust-server-only boot knobs (listen address, pool/ring sizes, pinning).
    pub rust_server_args: RustServerServerArgs,
    /// The scheduler's `server_args` dump (worker counts, tokenizer source,
    /// config-endpoint metadata). `Arc` so cloning the config (and, downstream,
    /// each `AppState`) is cheap; immutable after construction.
    pub server_args: Arc<ServerArgs>,
    /// Optional read-only PD snapshot; the frontend cannot stop its owner.
    pub pd_readiness: Option<crate::pd::transport::PdReadinessHandle>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            rust_server_args: RustServerServerArgs::default(),
            server_args: Arc::new(
                ServerArgs::from_json("{}").expect("empty server_args blob parses"),
            ),
            pd_readiness: None,
        }
    }
}

/// The scheduler's startup blob (`RustServer._build_server_args`) parsed once into
/// typed fields: values are post-`__post_init__`; security- and PD-relevant
/// fields are retained explicitly and never surfaced by raw-dump endpoints.
#[derive(serde::Deserialize)]
pub struct ServerArgs {
    /// HF repo id / local dir of the model, reported by `/get_model_info`.
    #[serde(default)]
    pub model_path: String,
    /// Model name reported by `/v1/models` and `/server_info`.
    #[serde(default)]
    pub served_model_name: String,
    /// Tokenizer source (model dir / `tokenizer.json` / HF repo id). Empty only
    /// in minimal standalone blobs — then boot requires `skip_tokenizer_init`.
    #[serde(default)]
    pub tokenizer_path: String,
    /// HF revision, used only when `tokenizer_path` is a repo id. `None` → main.
    #[serde(default)]
    pub revision: Option<String>,
    /// HTTP bind address (see [`Self::bind`]).
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    /// Log levels driving the access log — uvicorn runs at
    /// `log_level_http or log_level` (see [`Self::http_access_log_enabled`]).
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default)]
    pub log_level_http: Option<String>,
    /// Pinned tokenizer threads / detok shards (Python asserts both ≥ 1).
    #[serde(default = "default_worker_num")]
    pub tokenizer_worker_num: usize,
    #[serde(default = "default_worker_num")]
    pub detokenizer_worker_num: usize,
    /// Token-ids-in / token-ids-out mode: no tokenizer load, raw `output_ids`
    /// frames (drives the `Skip` detok backend and the ingress branch).
    #[serde(default)]
    pub skip_tokenizer_init: bool,
    /// Streamed `/generate` frames carry per-step deltas instead of cumulative
    /// text. Matches the Python `TokenizerManager`.
    #[serde(default)]
    pub incremental_streaming_output: bool,
    /// The resolved Python `ModelConfig`, attached to the blob at dump time.
    #[serde(default)]
    pub model_config: ModelConfig,
    /// Default sampling params advertised by `/get_model_info`, verbatim from
    /// `server_args.preferred_sampling_params` (a JSON object or null).
    #[serde(default)]
    pub preferred_sampling_params: Option<serde_json::Value>,
    /// Over-long inputs are truncated to fit the context instead of 400ing, and
    /// `max_new_tokens` is clamped rather than rejected (Python
    /// `TokenizerManager._validate_one_request`).
    #[serde(default)]
    pub allow_auto_truncate: bool,
    /// `return_hidden_states` is refused unless the server was launched with it:
    /// the scheduler simply won't produce them, so the request would 200 with the
    /// field silently missing.
    #[serde(default)]
    pub enable_return_hidden_states: bool,
    /// Output slots reserved per request on top of its input (eagle stores draft
    /// tokens there). Not a `server_args` field — `TokenizerManager` derives it and
    /// `RustServer._build_server_args` stamps it in, so both sides count alike.
    #[serde(default)]
    pub num_reserved_tokens: u64,
    /// Launch-time stamps (not `server_args` fields): sglang package version
    /// and the scheduler-derived KV token capacity, reported by `/server_info`.
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub max_total_num_tokens: Option<u64>,
    /// Bearer token protecting business endpoints. Never rendered by Debug
    /// logging or server-info responses.
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_disaggregation_mode")]
    pub disaggregation_mode: String,
    #[serde(default)]
    pub disaggregation_bootstrap_port: Option<u16>,
    #[serde(default)]
    pub pd_control_psk_file: Option<String>,
}

impl fmt::Debug for ServerArgs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ServerArgs")
            .field("model_path", &self.model_path)
            .field("served_model_name", &self.served_model_name)
            .field("tokenizer_path", &self.tokenizer_path)
            .field("revision", &self.revision)
            .field("host", &self.host)
            .field("port", &self.port)
            .field("log_level", &self.log_level)
            .field("log_level_http", &self.log_level_http)
            .field("tokenizer_worker_num", &self.tokenizer_worker_num)
            .field("detokenizer_worker_num", &self.detokenizer_worker_num)
            .field("skip_tokenizer_init", &self.skip_tokenizer_init)
            .field(
                "incremental_streaming_output",
                &self.incremental_streaming_output,
            )
            .field("model_config", &self.model_config)
            .field("preferred_sampling_params", &self.preferred_sampling_params)
            .field("allow_auto_truncate", &self.allow_auto_truncate)
            .field(
                "enable_return_hidden_states",
                &self.enable_return_hidden_states,
            )
            .field("num_reserved_tokens", &self.num_reserved_tokens)
            .field("version", &self.version)
            .field("max_total_num_tokens", &self.max_total_num_tokens)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("disaggregation_mode", &self.disaggregation_mode)
            .field(
                "disaggregation_bootstrap_port",
                &self.disaggregation_bootstrap_port,
            )
            .field(
                "pd_control_psk_file",
                &self.pd_control_psk_file.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

/// The slice of the resolved Python `ModelConfig` the rust server reads.
#[derive(Debug, Default, serde::Deserialize)]
pub struct ModelConfig {
    /// Resolved context length (`max_model_len` in `/v1/models`); mandatory at
    /// boot ([`ServerArgs::validate_mandatory`]).
    #[serde(default)]
    pub context_len: Option<u64>,
    /// Bounds client-supplied token ids — ingress 400s out-of-vocab ids before
    /// they crash the scheduler's embedding lookup;  mandatory at
    /// boot ([`ServerArgs::validate_mandatory`]).
    #[serde(default)]
    pub vocab_size: Option<u64>,
}

fn default_host() -> String {
    "127.0.0.1".into()
}
fn default_port() -> u16 {
    30000
}
fn default_log_level() -> String {
    "info".into()
}
fn default_worker_num() -> usize {
    1
}
fn default_disaggregation_mode() -> String {
    "null".into()
}

impl ServerArgs {
    /// Parse the blob; errors on malformed JSON or a wrongly-typed field.
    pub fn from_json(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| e.to_string())
    }

    /// Fail fast at startup if a field an endpoint depends on is missing.
    pub fn validate_mandatory(&self) -> Result<(), String> {
        if self.served_model_name.is_empty() {
            return Err("no 'served_model_name' in server_args".into());
        }
        if self.model_config.context_len.is_none() {
            return Err("no resolvable context length (model_config.context_len)".into());
        }
        if self.model_config.vocab_size.is_none() {
            return Err("no resolvable vocab size (model_config.vocab_size)".into());
        }
        if self.api_key.as_deref() == Some("") {
            return Err("api_key must not be empty".into());
        }
        if !matches!(
            self.disaggregation_mode.as_str(),
            "null" | "prefill" | "decode"
        ) {
            return Err("invalid disaggregation_mode".into());
        }
        if self.disaggregation_bootstrap_port == Some(0) {
            return Err("disaggregation_bootstrap_port must be non-zero".into());
        }
        if self.disaggregation_mode == "prefill" && self.disaggregation_bootstrap_port.is_none() {
            return Err("prefill mode requires disaggregation_bootstrap_port".into());
        }
        if self.disaggregation_mode != "null"
            && self
                .pd_control_psk_file
                .as_deref()
                .is_none_or(str::is_empty)
        {
            return Err("PD mode requires non-empty pd_control_psk_file".into());
        }
        Ok(())
    }

    /// Bind address `host:port`. `host` is expected to be an IP — the result is
    /// parsed as a `SocketAddr`.
    pub fn bind(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// Whether the HTTP access log is emitted, mirroring the Python server:
    /// uvicorn runs at `log_level_http or log_level` and prints access lines
    /// only at info/debug. `--log-level-http warning` turns them off.
    pub fn http_access_log_enabled(&self) -> bool {
        let level = self
            .log_level_http
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or(&self.log_level);
        matches!(
            level.to_ascii_lowercase().as_str(),
            "trace" | "debug" | "info"
        )
    }

    /// Pinned API threads for the embedded HTTP api-server. Python `server_args`
    /// has no such field — this is derived: enough to cover the widest pool.
    pub fn api_worker_num(&self) -> usize {
        4.max(self.tokenizer_worker_num)
            .max(self.detokenizer_worker_num)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_pd_json(extra: &str) -> String {
        format!(
            r#"{{
                "served_model_name": "model",
                "model_config": {{"context_len": 1024, "vocab_size": 32000}},
                "disaggregation_mode": "prefill",
                "disaggregation_bootstrap_port": 8998,
                "pd_control_psk_file": "/run/secrets/pd-control-psk",
                "api_key": "client-secret"
                {extra}
            }}"#
        )
    }

    #[test]
    fn debug_redacts_api_key_and_psk_path() {
        let args = ServerArgs::from_json(&valid_pd_json("")).unwrap();
        let debug = format!("{args:?}");
        assert!(!debug.contains("client-secret"));
        assert!(!debug.contains("/run/secrets/pd-control-psk"));
        assert!(debug.contains("<redacted>"));
    }

    #[test]
    fn pd_config_requires_nonzero_prefill_port_and_nonempty_psk_path() {
        let valid = ServerArgs::from_json(&valid_pd_json("")).unwrap();
        assert!(valid.validate_mandatory().is_ok());

        for invalid in [
            r#"{
                "served_model_name": "model",
                "model_config": {"context_len": 1024, "vocab_size": 32000},
                "disaggregation_mode": "prefill",
                "pd_control_psk_file": "/run/secrets/pd-control-psk"
            }"#,
            r#"{
                "served_model_name": "model",
                "model_config": {"context_len": 1024, "vocab_size": 32000},
                "disaggregation_mode": "prefill",
                "disaggregation_bootstrap_port": 0,
                "pd_control_psk_file": "/run/secrets/pd-control-psk"
            }"#,
            r#"{
                "served_model_name": "model",
                "model_config": {"context_len": 1024, "vocab_size": 32000},
                "disaggregation_mode": "prefill",
                "disaggregation_bootstrap_port": 8998,
                "pd_control_psk_file": ""
            }"#,
        ] {
            assert!(
                ServerArgs::from_json(invalid)
                    .unwrap()
                    .validate_mandatory()
                    .is_err()
            );
        }
    }
}
