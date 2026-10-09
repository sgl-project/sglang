//! The command line, mirroring `python/sglang/benchmark/serving.py`'s flags.
//!
//! Every flag that this port implements keeps its Python spelling and default,
//! so an existing command line runs here unchanged. Flags the Python script
//! has for surfaces this port does not implement (the TRT / gserver / truss
//! backends, the image, mooncake, agentic-trace and generated-shared-prefix
//! datasets, LoRA, profiling by stage) are absent rather than accepted and
//! ignored: a silently dropped flag would report a benchmark nobody ran.

use clap::{Parser, ValueEnum};
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Backend {
    /// `/generate`, the native SGLang route.
    Sglang,
    /// Alias of `sglang`, as in the Python script.
    SglangNative,
    /// `/v1/completions` on an SGLang server.
    SglangOai,
    /// `/v1/chat/completions` on an SGLang server.
    SglangOaiChat,
    Vllm,
    VllmChat,
    Lmdeploy,
    LmdeployChat,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum DatasetName {
    /// Prompts sampled from ShareGPT and repeated or truncated to the
    /// requested input length.
    Random,
    /// Prompts built from arithmetic token-id sequences, no corpus needed.
    RandomIds,
    /// The first turn of each ShareGPT conversation, with its reply's length
    /// as the output length.
    Sharegpt,
}

#[derive(Debug, Deserialize, Parser)]
// Every field defaults, so a JSON config sets only what it means to change.
// `deny_unknown_fields` makes a misspelled key an error rather than a setting
// that silently did not apply.
#[serde(default, deny_unknown_fields)]
#[command(
    name = "sglang-bench",
    about = "Serving benchmark client (the Rust port of sglang.benchmark.serving)",
    after_help = "Examples:\n  \
      sglang-bench --dataset-name random --num-prompts 20000 --random-input-len 1024 --random-output-len 512\n  \
      sglang-bench --backend sglang-oai-chat --dataset-name sharegpt --num-prompts 5000 --max-concurrency 512"
)]
pub struct Args {
    #[arg(long, value_enum, default_value = "sglang")]
    pub backend: Backend,

    /// Full base URL. Overrides `--host` / `--port`.
    #[arg(long)]
    pub base_url: Option<String>,

    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Defaults per backend, as in the Python script (30000 for SGLang,
    /// 8000 for vLLM, 23333 for LMDeploy).
    #[arg(long)]
    pub port: Option<u16>,

    #[arg(long, default_value_t = 60)]
    pub ready_check_timeout_sec: u64,

    #[arg(long, value_enum, default_value = "sharegpt")]
    pub dataset_name: DatasetName,

    /// Path to a local ShareGPT JSON. Empty downloads and caches it.
    #[arg(long, default_value = "")]
    pub dataset_path: String,

    /// Name or path of the model. Defaults to the server's `/v1/models`.
    #[arg(long)]
    pub model: Option<String>,

    /// The model name to send, when it differs from `--model`.
    #[arg(long)]
    pub served_model_name: Option<String>,

    /// Tokenizer path or HF repo id. Defaults to `--model`.
    #[arg(long)]
    pub tokenizer: Option<String>,

    #[arg(long, default_value_t = 1000)]
    pub num_prompts: usize,

    #[arg(long)]
    pub sharegpt_output_len: Option<usize>,

    #[arg(long)]
    pub sharegpt_context_len: Option<usize>,

    #[arg(long, default_value_t = 1024)]
    pub random_input_len: usize,

    #[arg(long, default_value_t = 1024)]
    pub random_output_len: usize,

    #[arg(long, default_value_t = 0.0)]
    pub random_range_ratio: f64,

    /// Requests per second, Poisson-paced. `inf` sends everything at once.
    #[arg(long, default_value = "inf")]
    pub request_rate: String,

    /// Cap on requests in flight. Unset means unbounded.
    #[arg(long)]
    pub max_concurrency: Option<usize>,

    #[arg(long, default_value_t = 1)]
    pub warmup_requests: usize,

    /// Tokio worker threads that parse the response streams. Defaults to the
    /// available parallelism; this is the knob the Python script lacks, where
    /// one asyncio thread parses every stream.
    #[arg(long)]
    pub worker_threads: Option<usize>,

    #[arg(long)]
    pub output_file: Option<String>,

    /// Also write the per-request columns (input/output lens, TTFTs, ITLs,
    /// generated texts, errors) into the result line.
    #[arg(long)]
    pub output_details: bool,

    #[arg(long)]
    pub disable_tqdm: bool,

    #[arg(long)]
    pub disable_stream: bool,

    #[arg(long)]
    pub disable_ignore_eos: bool,

    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    #[arg(long, default_value_t = 0.0)]
    pub temperature: f64,

    #[arg(long, default_value_t = 1.0)]
    pub top_p: f64,

    #[arg(long)]
    pub return_logprob: bool,

    #[arg(long, default_value_t = 0)]
    pub top_logprobs_num: u32,

    #[arg(long, default_value_t = -1)]
    pub logprob_start_len: i64,

    /// JSON object merged into every request body, overriding the defaults.
    #[arg(long)]
    pub extra_request_body: Option<String>,

    /// `key: value` header, repeatable.
    #[arg(long = "header", value_name = "KEY: VALUE")]
    pub headers: Vec<String>,

    /// Send `input_ids` instead of `text`. Only for the random datasets.
    #[arg(long)]
    pub tokenize_prompt: bool,

    /// Skip re-tokenizing the generated text. The retokenized totals are then
    /// reported as 0; use it when only throughput and latency matter.
    #[arg(long)]
    pub disable_retokenize: bool,

    /// POST `/flush_cache` after the warmup.
    #[arg(long)]
    pub flush_cache: bool,

    #[arg(long, default_value_t = 60.0)]
    pub flush_cache_timeout: f64,

    /// Start and stop the server profiler around the measured run.
    #[arg(long)]
    pub profile: bool,

    /// Free-form label recorded in the result line.
    #[arg(long)]
    pub tag: Option<String>,
}

/// The defaults are clap's, read back from an empty command line, so the two
/// entry points cannot drift: there is one place a default is written.
impl Default for Args {
    fn default() -> Self {
        Self::parse_from(["sglang-bench"])
    }
}

impl Backend {
    /// The Python `_BACKEND_API_PATHS` entry.
    pub fn api_path(self) -> &'static str {
        match self {
            Self::Sglang | Self::SglangNative => "/generate",
            Self::SglangOai | Self::Vllm | Self::Lmdeploy => "/v1/completions",
            Self::SglangOaiChat | Self::VllmChat | Self::LmdeployChat => "/v1/chat/completions",
        }
    }

    /// The Python per-backend default port.
    pub fn default_port(self) -> u16 {
        match self {
            Self::Sglang | Self::SglangNative | Self::SglangOai | Self::SglangOaiChat => 30000,
            Self::Vllm | Self::VllmChat => 8000,
            Self::Lmdeploy | Self::LmdeployChat => 23333,
        }
    }

    /// The wire protocol this backend speaks, which decides how a streamed
    /// chunk is read.
    pub fn protocol(self) -> Protocol {
        match self {
            Self::Sglang | Self::SglangNative => Protocol::SglangGenerate,
            Self::SglangOai | Self::Vllm | Self::Lmdeploy => Protocol::OpenaiCompletions,
            Self::SglangOaiChat | Self::VllmChat | Self::LmdeployChat => Protocol::OpenaiChat,
        }
    }

    /// The name this backend reports as, matching the Python `--backend` value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sglang => "sglang",
            Self::SglangNative => "sglang-native",
            Self::SglangOai => "sglang-oai",
            Self::SglangOaiChat => "sglang-oai-chat",
            Self::Vllm => "vllm",
            Self::VllmChat => "vllm-chat",
            Self::Lmdeploy => "lmdeploy",
            Self::LmdeployChat => "lmdeploy-chat",
        }
    }

    pub fn is_sglang(self) -> bool {
        self.as_str().starts_with("sglang")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protocol {
    SglangGenerate,
    OpenaiCompletions,
    OpenaiChat,
}

impl DatasetName {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::RandomIds => "random-ids",
            Self::Sharegpt => "sharegpt",
        }
    }

    pub fn is_random(self) -> bool {
        matches!(self, Self::Random | Self::RandomIds)
    }
}

impl Args {
    /// `--request-rate`, with `inf` as the unpaced default.
    pub fn parse_request_rate(&self) -> anyhow::Result<f64> {
        let raw = self.request_rate.trim();
        if raw.eq_ignore_ascii_case("inf") || raw.eq_ignore_ascii_case("infinity") {
            return Ok(f64::INFINITY);
        }
        let rate: f64 = raw
            .parse()
            .map_err(|_| anyhow::anyhow!("--request-rate must be a number or `inf`, got {raw}"))?;
        anyhow::ensure!(rate > 0.0, "--request-rate must be positive, got {rate}");
        Ok(rate)
    }

    /// The base URL the client posts to, `--base-url` or `http://host:port`.
    /// A bare IPv6 host is bracketed, as `resolve_base_url` does in Python.
    pub fn base_url(&self) -> String {
        if let Some(url) = &self.base_url {
            return url.trim_end_matches('/').to_owned();
        }
        let port = self.port.unwrap_or_else(|| self.backend.default_port());
        let host = if self.host.contains(':') && !self.host.starts_with('[') {
            format!("[{}]", self.host)
        } else {
            self.host.clone()
        };
        format!("http://{host}:{port}")
    }

    pub fn api_url(&self) -> String {
        format!("{}{}", self.base_url(), self.backend.api_path())
    }

    /// `--extra-request-body`, which must be a JSON object.
    pub fn parse_extra_body(&self) -> anyhow::Result<serde_json::Map<String, serde_json::Value>> {
        let Some(raw) = &self.extra_request_body else {
            return Ok(serde_json::Map::new());
        };
        match serde_json::from_str(raw) {
            Ok(serde_json::Value::Object(map)) => Ok(map),
            Ok(_) => anyhow::bail!("--extra-request-body must be a JSON object"),
            Err(error) => anyhow::bail!("--extra-request-body is not valid JSON: {error}"),
        }
    }

    /// Validate the flag combinations the Python script asserts on.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.num_prompts > 0, "--num-prompts must be at least 1");
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.random_range_ratio),
            "--random-range-ratio must be in [0, 1], got {}",
            self.random_range_ratio
        );
        anyhow::ensure!(
            !self.tokenize_prompt || self.dataset_name.is_random(),
            "--tokenize-prompt is only supported by the random datasets"
        );
        anyhow::ensure!(
            !self.tokenize_prompt || self.backend.protocol() != Protocol::OpenaiChat,
            "--tokenize-prompt cannot be used with a chat backend, which sends messages"
        );
        if let Some(len) = self.sharegpt_output_len {
            anyhow::ensure!(len >= 4, "--sharegpt-output-len is too small, got {len}");
        }
        if let Some(limit) = self.max_concurrency {
            anyhow::ensure!(limit > 0, "--max-concurrency must be at least 1");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(extra: &[&str]) -> Args {
        let mut argv = vec!["sglang-bench"];
        argv.extend_from_slice(extra);
        Args::parse_from(argv)
    }

    /// The URL is assembled as the Python script's `resolve_base_url` does,
    /// including the per-backend default port and IPv6 bracketing.
    #[test]
    fn urls_follow_the_backend_defaults() {
        assert_eq!(args(&[]).api_url(), "http://0.0.0.0:30000/generate");
        assert_eq!(
            args(&["--backend", "vllm"]).api_url(),
            "http://0.0.0.0:8000/v1/completions"
        );
        assert_eq!(
            args(&["--backend", "lmdeploy-chat", "--port", "1"]).api_url(),
            "http://0.0.0.0:1/v1/chat/completions"
        );
        assert_eq!(
            args(&["--host", "::1"]).api_url(),
            "http://[::1]:30000/generate"
        );
        assert_eq!(
            args(&["--base-url", "https://example.com/"]).api_url(),
            "https://example.com/generate"
        );
    }

    #[test]
    fn request_rate_accepts_inf_and_rejects_nonsense() {
        assert!(args(&[]).parse_request_rate().unwrap().is_infinite());
        assert_eq!(
            args(&["--request-rate", "7.5"])
                .parse_request_rate()
                .unwrap(),
            7.5
        );
        assert!(args(&["--request-rate", "0"]).parse_request_rate().is_err());
        assert!(
            args(&["--request-rate", "fast"])
                .parse_request_rate()
                .is_err()
        );
    }

    #[test]
    fn extra_request_body_must_be_an_object() {
        let parsed = args(&["--extra-request-body", r#"{"top_k": 5}"#])
            .parse_extra_body()
            .unwrap();
        assert_eq!(parsed["top_k"], 5);
        assert!(
            args(&["--extra-request-body", "[1]"])
                .parse_extra_body()
                .is_err()
        );
        assert!(
            args(&["--extra-request-body", "{"])
                .parse_extra_body()
                .is_err()
        );
        assert!(args(&[]).parse_extra_body().unwrap().is_empty());
    }

    /// Flag combinations the Python script asserts on must fail here too,
    /// rather than producing a benchmark whose shape nobody asked for.
    #[test]
    fn validation_rejects_unsupported_combinations() {
        assert!(args(&[]).validate().is_ok());
        assert!(args(&["--num-prompts", "0"]).validate().is_err());
        assert!(args(&["--random-range-ratio", "1.5"]).validate().is_err());
        assert!(args(&["--tokenize-prompt"]).validate().is_err());
        assert!(
            args(&["--tokenize-prompt", "--dataset-name", "random"])
                .validate()
                .is_ok()
        );
        assert!(
            args(&[
                "--tokenize-prompt",
                "--dataset-name",
                "random",
                "--backend",
                "sglang-oai-chat"
            ])
            .validate()
            .is_err()
        );
        assert!(args(&["--sharegpt-output-len", "2"]).validate().is_err());
        assert!(args(&["--max-concurrency", "0"]).validate().is_err());
    }

    /// A JSON config sets only the keys it names; the rest take the same
    /// defaults the command line would apply.
    #[test]
    fn a_config_fills_missing_keys_from_the_cli_defaults() {
        let args: Args = serde_json::from_str(
            r#"{"backend": "sglang-oai-chat", "dataset_name": "random-ids", "num_prompts": 7}"#,
        )
        .unwrap();
        assert_eq!(args.backend, Backend::SglangOaiChat);
        assert_eq!(args.dataset_name, DatasetName::RandomIds);
        assert_eq!(args.num_prompts, 7);
        // Untouched keys match the flag defaults.
        assert_eq!(args.random_input_len, 1024);
        assert_eq!(args.seed, 42);
        assert_eq!(args.host, "0.0.0.0");
        assert!(args.parse_request_rate().unwrap().is_infinite());
        assert!(!args.disable_stream);
    }

    /// An empty config is the full default set, and a misspelled key is an
    /// error: silently ignoring it would report a benchmark that did not run
    /// the way the caller asked.
    #[test]
    fn a_config_rejects_unknown_keys() {
        let defaults: Args = serde_json::from_str("{}").unwrap();
        assert_eq!(defaults.num_prompts, Args::default().num_prompts);
        let typo = serde_json::from_str::<Args>(r#"{"num_promts": 3}"#);
        assert!(typo.is_err());
    }
}
