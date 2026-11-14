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

/// The wire the request travels on. Orthogonal to `--backend`, which picks
/// the API shape: `grpc` carries the same `/generate` contract as HTTP, on
/// the Rust server's `sglang.api.v1` service.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Protocol {
    /// JSON over HTTP, streamed as SSE. What the Python client speaks.
    #[default]
    Http,
    /// The Rust server's gRPC service; native-API backends only.
    Grpc,
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

    /// Wire protocol. `grpc` needs a native-API backend (`sglang`) and talks
    /// to the Rust server's gRPC listener, whose default port is the HTTP
    /// port plus 10000.
    #[arg(long, value_enum, default_value = "http")]
    pub protocol: Protocol,

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
    /// The response side follows automatically: a server launched with
    /// `--skip-tokenizer-init` returns `output_ids` and no text, and tokens
    /// are then counted from the ids.
    #[arg(long)]
    pub tokenize_prompt: bool,

    /// Vocabulary size for `random-ids` with `--tokenize-prompt`. Given, no
    /// tokenizer is loaded at all, so the client needs no model files; the
    /// retokenized totals are then reported as 0.
    #[arg(long)]
    pub vocab_size: Option<u32>,

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

    /// The Python per-backend default port. The Rust server's gRPC listener
    /// defaults to its HTTP port plus 10000 (`--grpc-port`).
    pub fn default_port(self, protocol: Protocol) -> u16 {
        let http = match self {
            Self::Sglang | Self::SglangNative | Self::SglangOai | Self::SglangOaiChat => 30000,
            Self::Vllm | Self::VllmChat => 8000,
            Self::Lmdeploy | Self::LmdeployChat => 23333,
        };
        match protocol {
            Protocol::Http => http,
            Protocol::Grpc => http + 10000,
        }
    }

    /// The API shape this backend speaks, which decides how a request is
    /// built and how a streamed chunk is read.
    pub fn api(self) -> Api {
        match self {
            Self::Sglang | Self::SglangNative => Api::SglangGenerate,
            Self::SglangOai | Self::Vllm | Self::Lmdeploy => Api::OpenaiCompletions,
            Self::SglangOaiChat | Self::VllmChat | Self::LmdeployChat => Api::OpenaiChat,
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

/// The request and frame shape, independent of the wire it travels on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    /// SGLang's native `/generate`, over HTTP or the gRPC service.
    SglangGenerate,
    OpenaiCompletions,
    OpenaiChat,
}

impl Protocol {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::Grpc => "grpc",
        }
    }
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
        let port = self
            .port
            .unwrap_or_else(|| self.backend.default_port(self.protocol));
        let host = if self.host.contains(':') && !self.host.starts_with('[') {
            format!("[{}]", self.host)
        } else {
            self.host.clone()
        };
        format!("http://{host}:{port}")
    }

    /// Where requests go: the route under the base URL over HTTP, the bare
    /// endpoint over gRPC, whose service lives at the connection's root.
    pub fn api_url(&self) -> String {
        match self.protocol {
            Protocol::Http => format!("{}{}", self.base_url(), self.backend.api_path()),
            Protocol::Grpc => self.base_url(),
        }
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
            !self.tokenize_prompt || self.backend.api() != Api::OpenaiChat,
            "--tokenize-prompt cannot be used with a chat backend, which sends messages"
        );
        anyhow::ensure!(
            self.protocol == Protocol::Http || self.backend.api() == Api::SglangGenerate,
            "--protocol grpc carries the native /generate contract only; use --backend sglang \
             (the OpenAI routes have no gRPC service)"
        );
        if let Some(len) = self.sharegpt_output_len {
            anyhow::ensure!(len >= 4, "--sharegpt-output-len is too small, got {len}");
        }
        if let Some(limit) = self.max_concurrency {
            anyhow::ensure!(limit > 0, "--max-concurrency must be at least 1");
        }
        if let Some(size) = self.vocab_size {
            anyhow::ensure!(size > 0, "--vocab-size must be at least 1");
            anyhow::ensure!(
                self.tokenize_prompt && self.dataset_name == DatasetName::RandomIds,
                "--vocab-size is only meaningful with --tokenize-prompt --dataset-name random-ids; \
                 every other dataset needs the tokenizer"
            );
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
        // gRPC has no route path, and its default port is the server's
        // HTTP port plus 10000.
        assert_eq!(
            args(&["--protocol", "grpc"]).api_url(),
            "http://0.0.0.0:40000"
        );
        assert_eq!(
            args(&["--protocol", "grpc", "--port", "7"]).api_url(),
            "http://0.0.0.0:7"
        );
    }

    /// Only the native API has a gRPC service, so the protocol is rejected
    /// for the OpenAI-shaped backends rather than quietly sent over HTTP.
    #[test]
    fn grpc_requires_a_native_backend() {
        assert!(args(&["--protocol", "grpc"]).validate().is_ok());
        assert!(
            args(&["--protocol", "grpc", "--backend", "sglang-native"])
                .validate()
                .is_ok()
        );
        for backend in ["sglang-oai", "sglang-oai-chat", "vllm", "lmdeploy-chat"] {
            assert!(
                args(&["--protocol", "grpc", "--backend", backend])
                    .validate()
                    .is_err(),
                "{backend}"
            );
        }
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

    /// `--vocab-size` replaces the tokenizer, so it is only coherent where the
    /// tokenizer's sole job was to supply that number.
    #[test]
    fn vocab_size_requires_the_tokenizer_free_dataset() {
        let ok = [
            "--vocab-size",
            "1000",
            "--tokenize-prompt",
            "--dataset-name",
            "random-ids",
        ];
        assert!(args(&ok).validate().is_ok());
        assert!(args(&["--vocab-size", "1000"]).validate().is_err());
        assert!(
            args(&[
                "--vocab-size",
                "1000",
                "--tokenize-prompt",
                "--dataset-name",
                "random"
            ])
            .validate()
            .is_err()
        );
        assert!(
            args(&[
                "--vocab-size",
                "0",
                "--tokenize-prompt",
                "--dataset-name",
                "random-ids"
            ])
            .validate()
            .is_err()
        );
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
