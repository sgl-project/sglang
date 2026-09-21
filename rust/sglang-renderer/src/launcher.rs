//! Process launch configuration for the standalone renderer.

use std::collections::{BTreeSet, HashMap};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use hf_hub::api::tokio::{ApiBuilder, ApiRepo};
use hf_hub::{Cache, Repo, RepoType};
use serde_json::Value;

use crate::preprocessing::{resolve_model_file, resolve_tokenizer_file};
use crate::{RendererConfig, RendererLimits, RendererRuntimeConfig, SamplingDefaults, serve};

const DEFAULT_CONTEXT_LEN: u64 = 2048;

#[derive(Debug, Parser)]
#[command(
    name = "sglang-renderer",
    about = "Run the SGLang Rust renderer with an optional SGLang engine"
)]
struct Cli {
    /// Model directory, config file, or Hugging Face repository id.
    #[arg(value_name = "MODEL")]
    model: String,

    /// Optional SGLang engine origin exposing /generate.
    ///
    /// When omitted, only rendering and tokenization routes are served.
    #[arg(long, value_name = "URL")]
    engine_url: Option<String>,

    /// Proxy routes not owned by the renderer to the SGLang engine origin.
    #[arg(long, requires = "engine_url")]
    proxy_unhandled_routes: bool,

    #[arg(long)]
    tokenizer_path: Option<String>,
    #[arg(long)]
    revision: Option<String>,
    #[arg(long)]
    served_model_name: Option<String>,
    #[arg(long, default_value_t = IpAddr::V4(Ipv4Addr::LOCALHOST))]
    host: IpAddr,
    #[arg(long, default_value_t = 30000)]
    port: u16,
    #[arg(long, default_value_t = 2)]
    http_workers: usize,
    #[arg(long, default_value_t = 1)]
    tokenizer_workers: usize,
    #[arg(long, default_value_t = 128)]
    queue_capacity: usize,
    #[arg(long)]
    chat_template: Option<String>,
    #[arg(long)]
    tool_call_parser: Option<String>,
    #[arg(long)]
    reasoning_parser: Option<String>,
    #[arg(long, value_parser = parse_json_object)]
    default_chat_template_kwargs: Option<HashMap<String, Value>>,
    #[arg(long, value_enum, default_value_t)]
    sampling_defaults: SamplingDefaultsSource,
    /// Already-resolved sampling defaults. When set with context length and
    /// vocabulary size, model metadata is not reopened by this process.
    #[arg(long, value_parser = parse_sampling_defaults)]
    resolved_sampling_params: Option<SamplingDefaults>,
    #[arg(long)]
    context_length: Option<u64>,
    #[arg(long)]
    vocab_size: Option<u64>,
    #[arg(long, default_value_t = 0)]
    num_reserved_tokens: u64,
    #[arg(long)]
    allow_auto_truncate: bool,
    #[arg(long)]
    enable_return_hidden_states: bool,
    #[arg(long)]
    stream_response_default_include_usage: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum SamplingDefaultsSource {
    #[default]
    Model,
    Openai,
}

#[derive(Debug)]
struct DirectArgs {
    model: String,
    engine_url: Option<String>,
    proxy_unhandled_routes: bool,
    tokenizer_path: String,
    revision: Option<String>,
    served_model_name: String,
    http_addr: SocketAddr,
    http_workers: usize,
    tokenizer_workers: usize,
    queue_capacity: usize,
    chat_template: Option<String>,
    tool_call_parser: Option<String>,
    reasoning_parser: Option<String>,
    default_chat_template_kwargs: HashMap<String, Value>,
    sampling_defaults: SamplingDefaultsSource,
    resolved_sampling_params: Option<SamplingDefaults>,
    context_length: Option<u64>,
    vocab_size: Option<u64>,
    num_reserved_tokens: u64,
    allow_auto_truncate: bool,
    enable_return_hidden_states: bool,
    stream_response_default_include_usage: bool,
}

pub fn run_cli() -> Result<(), String> {
    let args = Cli::parse().into_direct_args();
    let http_workers = args.http_workers;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(http_workers.max(1))
        .enable_all()
        .build()
        .map_err(|error| format!("building renderer runtime failed: {error}"))?;
    runtime.block_on(async { serve(args.resolve().await?).await })
}

impl Cli {
    fn into_direct_args(self) -> DirectArgs {
        let model = self.model;
        let tokenizer_path = self.tokenizer_path.unwrap_or_else(|| model.clone());
        let served_model_name = self.served_model_name.unwrap_or_else(|| model.clone());
        let http_addr = SocketAddr::new(self.host, self.port);
        DirectArgs {
            model,
            engine_url: self.engine_url,
            proxy_unhandled_routes: self.proxy_unhandled_routes,
            tokenizer_path,
            revision: self.revision,
            served_model_name,
            http_addr,
            http_workers: self.http_workers,
            tokenizer_workers: self.tokenizer_workers,
            queue_capacity: self.queue_capacity,
            chat_template: self.chat_template,
            tool_call_parser: self.tool_call_parser,
            reasoning_parser: self.reasoning_parser,
            default_chat_template_kwargs: self.default_chat_template_kwargs.unwrap_or_default(),
            sampling_defaults: self.sampling_defaults,
            resolved_sampling_params: self.resolved_sampling_params,
            context_length: self.context_length,
            vocab_size: self.vocab_size,
            num_reserved_tokens: self.num_reserved_tokens,
            allow_auto_truncate: self.allow_auto_truncate,
            enable_return_hidden_states: self.enable_return_hidden_states,
            stream_response_default_include_usage: self.stream_response_default_include_usage,
        }
    }
}

impl DirectArgs {
    async fn resolve(self) -> Result<RendererRuntimeConfig, String> {
        let (context_len, vocab_size, default_sampling_params) = match self.resolved_sampling_params
        {
            Some(default_sampling_params) => {
                let context_len = self.context_length.ok_or_else(|| {
                    "--resolved-sampling-params requires --context-length".to_string()
                })?;
                let vocab_size = self.vocab_size.ok_or_else(|| {
                    "--resolved-sampling-params requires --vocab-size".to_string()
                })?;
                (context_len, vocab_size, default_sampling_params)
            }
            None => {
                let files = resolve_required_files(
                    &self.model,
                    &self.tokenizer_path,
                    self.revision.as_deref(),
                )
                .await?;
                let model_config = read_json(&files.config_path)?;
                let derived_context_len = derive_context_len(&model_config)?;
                let context_len = match self.context_length {
                    Some(context_len)
                        if context_len > derived_context_len && !allow_longer_context() =>
                    {
                        return Err(format!(
                            "user-specified context length {context_len} exceeds the model-derived context length {derived_context_len}; set SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 to allow it"
                        ));
                    }
                    Some(context_len) => context_len,
                    None => derived_context_len,
                };
                let vocab_size = self
                    .vocab_size
                    .or_else(|| derive_vocab_size(&model_config))
                    .ok_or_else(|| {
                        "model config does not define vocab_size; pass --vocab-size explicitly"
                            .to_string()
                    })?;
                let default_sampling_params = match self.sampling_defaults {
                    SamplingDefaultsSource::Openai => SamplingDefaults::default(),
                    SamplingDefaultsSource::Model => files
                        .generation_config_path
                        .as_deref()
                        .map(read_sampling_defaults)
                        .transpose()?
                        .unwrap_or_default(),
                };
                (context_len, vocab_size, default_sampling_params)
            }
        };

        Ok(RendererRuntimeConfig {
            http_addr: self.http_addr,
            http_workers: self.http_workers,
            tokenizer_workers: self.tokenizer_workers,
            queue_capacity: self.queue_capacity,
            engine_url: self.engine_url,
            proxy_unhandled_routes: self.proxy_unhandled_routes,
            renderer: RendererConfig {
                served_model_name: self.served_model_name,
                tokenizer_path: self.tokenizer_path,
                revision: self.revision,
                model_path: self.model,
                chat_template: self.chat_template,
                tool_call_parser: self.tool_call_parser,
                reasoning_parser: self.reasoning_parser,
                default_chat_template_kwargs: self.default_chat_template_kwargs,
                stream_response_default_include_usage: self.stream_response_default_include_usage,
                default_sampling_params,
                limits: RendererLimits {
                    vocab_size,
                    context_len,
                    num_reserved_tokens: self.num_reserved_tokens,
                    allow_auto_truncate: self.allow_auto_truncate,
                    enable_return_hidden_states: self.enable_return_hidden_states,
                },
            },
        })
    }
}

#[derive(Debug)]
struct ResolvedFiles {
    config_path: PathBuf,
    generation_config_path: Option<PathBuf>,
}

async fn resolve_required_files(
    model: &str,
    tokenizer: &str,
    revision: Option<&str>,
) -> Result<ResolvedFiles, String> {
    let model_is_local = Path::new(model).exists();
    let tokenizer_is_local = Path::new(tokenizer).exists();
    let mut config_path = resolve_model_file(model, revision, "config.json").map(PathBuf::from);
    let mut tokenizer_ready = resolve_tokenizer_file(tokenizer, revision).is_some();

    if model_is_local && config_path.is_none() {
        return Err(format!(
            "local model source {model:?} does not contain config.json"
        ));
    }
    if tokenizer_is_local && !tokenizer_ready {
        return Err(format!(
            "local tokenizer source {tokenizer:?} does not contain tokenizer.json, tiktoken.model, or *.tiktoken"
        ));
    }

    let need_model = config_path.is_none();
    let need_tokenizer = !tokenizer_ready;
    if need_model || need_tokenizer {
        if offline_mode() {
            return Err(format!(
                "required renderer metadata is not cached for model {model:?} and tokenizer {tokenizer:?}, and HF_HUB_OFFLINE is enabled"
            ));
        }
        if model == tokenizer {
            download_repository(model, revision, need_model, need_tokenizer).await?;
        } else {
            if need_model {
                download_repository(model, revision, true, false).await?;
            }
            if need_tokenizer {
                download_repository(tokenizer, revision, false, true).await?;
            }
        }
        config_path = resolve_model_file(model, revision, "config.json").map(PathBuf::from);
        tokenizer_ready = resolve_tokenizer_file(tokenizer, revision).is_some();
    }

    let config_path = config_path.ok_or_else(|| {
        format!(
            "model {model:?} does not expose config.json at revision {:?}",
            revision.unwrap_or("main")
        )
    })?;
    if !tokenizer_ready {
        return Err(format!(
            "tokenizer {tokenizer:?} does not expose tokenizer.json, tiktoken.model, or *.tiktoken at revision {:?}",
            revision.unwrap_or("main")
        ));
    }
    let generation_config_path =
        resolve_model_file(model, revision, "generation_config.json").map(PathBuf::from);
    Ok(ResolvedFiles {
        config_path,
        generation_config_path,
    })
}

async fn download_repository(
    repo_id: &str,
    revision: Option<&str>,
    include_model_metadata: bool,
    include_tokenizer: bool,
) -> Result<(), String> {
    let mut builder = ApiBuilder::from_env()
        .with_cache_dir(hf_cache().path().clone())
        .with_progress(false);
    if let Ok(token) = std::env::var("HF_TOKEN")
        && !token.is_empty()
    {
        builder = builder.with_token(Some(token));
    }
    let api = builder
        .build()
        .map_err(|error| format!("building Hugging Face client failed: {error}"))?;
    let repo = api.repo(Repo::with_revision(
        repo_id.to_string(),
        RepoType::Model,
        revision.unwrap_or("main").to_string(),
    ));
    let info = repo.info().await.map_err(|error| {
        format!(
            "fetching Hugging Face metadata for {repo_id:?} at revision {:?} failed: {error}",
            revision.unwrap_or("main")
        )
    })?;
    let siblings = info
        .siblings
        .into_iter()
        .map(|sibling| sibling.rfilename)
        .collect::<BTreeSet<_>>();

    if include_model_metadata {
        if !siblings.contains("config.json") {
            return Err(format!(
                "Hugging Face model {repo_id:?} does not contain config.json"
            ));
        }
        download_file(&repo, repo_id, "config.json").await?;
        if siblings.contains("generation_config.json") {
            download_file(&repo, repo_id, "generation_config.json").await?;
        }
    }
    if include_tokenizer {
        for filename in ["tokenizer_config.json", "config.json"] {
            if siblings.contains(filename) {
                download_file(&repo, repo_id, filename).await?;
            }
        }
        let mut tokenizer_names = Vec::new();
        if siblings.contains("tokenizer.json") {
            tokenizer_names.push("tokenizer.json");
        }
        if siblings.contains("tiktoken.model") {
            tokenizer_names.push("tiktoken.model");
        } else if let Some(name) = siblings.iter().find(|name| name.ends_with(".tiktoken")) {
            tokenizer_names.push(name);
        }
        if tokenizer_names.is_empty() {
            return Err(format!(
                "Hugging Face model {repo_id:?} does not contain tokenizer.json, tiktoken.model, or *.tiktoken"
            ));
        }
        for tokenizer_name in tokenizer_names {
            download_file(&repo, repo_id, tokenizer_name).await?;
        }
        let template_name = ["chat_template.json", "chat_template.jinja"]
            .into_iter()
            .find(|name| siblings.contains(*name))
            .or_else(|| {
                siblings
                    .iter()
                    .find(|name| name.ends_with(".jinja"))
                    .map(String::as_str)
            });
        if let Some(template_name) = template_name {
            download_file(&repo, repo_id, template_name).await?;
        }
    }
    Ok(())
}

async fn download_file(repo: &ApiRepo, repo_id: &str, filename: &str) -> Result<PathBuf, String> {
    repo.get(filename).await.map_err(|error| {
        format!("downloading {filename:?} for Hugging Face model {repo_id:?} failed: {error}")
    })
}

fn hf_cache() -> Cache {
    ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"]
        .iter()
        .find_map(|name| std::env::var(name).ok())
        .map(PathBuf::from)
        .map(Cache::new)
        .unwrap_or_else(Cache::from_env)
}

fn read_json(path: &Path) -> Result<Value, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|error| format!("reading {} failed: {error}", path.display()))?;
    serde_json::from_str(&contents)
        .map_err(|error| format!("parsing {} failed: {error}", path.display()))
}

fn read_sampling_defaults(path: &Path) -> Result<SamplingDefaults, String> {
    let value = read_json(path)?;
    serde_json::from_value(value).map_err(|error| {
        format!(
            "parsing sampling defaults from {} failed: {error}",
            path.display()
        )
    })
}

fn derive_context_len(config: &Value) -> Result<u64, String> {
    let text = effective_text_config(config);
    let factor = inherited_value(text, config, "rope_scaling")
        .and_then(Value::as_object)
        .map(|rope| {
            if rope.contains_key("original_max_position_embeddings")
                || rope.get("rope_type").and_then(Value::as_str) == Some("llama3")
            {
                1.0
            } else {
                rope.get("factor").and_then(Value::as_f64).unwrap_or(1.0)
            }
        })
        .unwrap_or(1.0);
    for key in [
        "max_sequence_length",
        "seq_length",
        "max_seq_len",
        "model_max_length",
        "max_position_embeddings",
    ] {
        if let Some(value) = inherited_value(text, config, key).and_then(Value::as_u64) {
            let scaled = factor * value as f64;
            if !scaled.is_finite() || scaled <= 0.0 || scaled > u64::MAX as f64 {
                return Err(format!(
                    "invalid context length {value} with rope scaling factor {factor}"
                ));
            }
            return Ok(scaled as u64);
        }
    }
    Ok(DEFAULT_CONTEXT_LEN)
}

fn derive_vocab_size(config: &Value) -> Option<u64> {
    let text = effective_text_config(config);
    let architecture = config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|architectures| architectures.first())
        .and_then(Value::as_str);
    let key = if architecture == Some("GlmImageForConditionalGeneration") {
        "vision_vocab_size"
    } else {
        "vocab_size"
    };
    inherited_value(text, config, key).and_then(Value::as_u64)
}

fn effective_text_config(config: &Value) -> &Value {
    let is_non_hf_llava = config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|architectures| architectures.first())
        .and_then(Value::as_str)
        .is_some_and(|architecture| {
            architecture.starts_with("Llava") && architecture.ends_with("ForCausalLM")
        });
    if is_non_hf_llava {
        return config;
    }
    if let Some(thinker) = config.get("thinker_config") {
        return thinker.get("text_config").unwrap_or(thinker);
    }
    for key in ["llm_config", "language_config", "text_config"] {
        if let Some(text) = config.get(key) {
            return text;
        }
    }
    config
}

fn inherited_value<'a>(text: &'a Value, root: &'a Value, key: &str) -> Option<&'a Value> {
    text.get(key).or_else(|| root.get(key))
}

fn parse_json_object(value: &str) -> Result<HashMap<String, Value>, String> {
    serde_json::from_str(value).map_err(|error| format!("expected a JSON object: {error}"))
}

fn parse_sampling_defaults(value: &str) -> Result<SamplingDefaults, String> {
    serde_json::from_str(value)
        .map_err(|error| format!("expected resolved sampling parameters as JSON: {error}"))
}

fn offline_mode() -> bool {
    std::env::var("HF_HUB_OFFLINE").ok().is_some_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn allow_longer_context() -> bool {
    std::env::var("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN")
        .ok()
        .is_some_and(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::json;

    use super::*;

    fn direct_cli(model: &Path) -> Cli {
        Cli::try_parse_from(["sglang-renderer", model.to_str().unwrap()]).unwrap()
    }

    fn fixture_model(config: Value, generation_config: Option<Value>) -> PathBuf {
        let directory =
            std::env::temp_dir().join(format!("sglang-renderer-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&directory).unwrap();
        fs::write(directory.join("config.json"), config.to_string()).unwrap();
        fs::write(directory.join("tokenizer.json"), "{}").unwrap();
        if let Some(generation_config) = generation_config {
            fs::write(
                directory.join("generation_config.json"),
                generation_config.to_string(),
            )
            .unwrap();
        }
        directory
    }

    #[test]
    fn cli_uses_sglang_renderer_defaults() {
        let directory = fixture_model(
            json!({"vocab_size": 128, "max_position_embeddings": 4096}),
            None,
        );
        let args = direct_cli(&directory).into_direct_args();

        assert_eq!(args.served_model_name, directory.to_string_lossy());
        assert_eq!(args.tokenizer_path, directory.to_string_lossy());
        assert_eq!(args.http_addr, "127.0.0.1:30000".parse().unwrap());
        assert_eq!(args.http_workers, 2);
        assert_eq!(args.tokenizer_workers, 1);
        assert_eq!(args.queue_capacity, 128);
        assert_eq!(args.engine_url, None);
        assert_eq!(args.sampling_defaults, SamplingDefaultsSource::Model);
        assert_eq!(args.resolved_sampling_params, None);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn proxying_unhandled_routes_requires_an_engine_url() {
        let error = Cli::try_parse_from(["sglang-renderer", "model", "--proxy-unhandled-routes"])
            .unwrap_err();

        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }

    #[tokio::test]
    async fn direct_resolution_matches_model_metadata_and_cli_overrides() {
        let directory = fixture_model(
            json!({
                "vocab_size": 10,
                "max_position_embeddings": 8192,
                "thinker_config": {
                    "text_config": {
                        "vocab_size": 128,
                        "max_position_embeddings": 4096,
                        "rope_scaling": {"factor": 2.0}
                    }
                }
            }),
            Some(json!({
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 20,
                "min_p": 0.1,
                "repetition_penalty": 1.05,
                "max_new_tokens": 32
            })),
        );
        let cli = Cli::try_parse_from([
            "sglang-renderer",
            directory.to_str().unwrap(),
            "--engine-url",
            "http://127.0.0.1:30001",
            "--proxy-unhandled-routes",
            "--served-model-name",
            "fixture",
            "--context-length",
            "2048",
            "--vocab-size",
            "256",
            "--num-reserved-tokens",
            "8",
            "--default-chat-template-kwargs",
            r#"{"enable_thinking":false}"#,
        ])
        .unwrap();
        let config = cli.into_direct_args().resolve().await.unwrap();

        assert_eq!(config.engine_url.as_deref(), Some("http://127.0.0.1:30001"));
        assert!(config.proxy_unhandled_routes);
        assert_eq!(config.renderer.served_model_name, "fixture");
        assert_eq!(config.renderer.limits.context_len, 2048);
        assert_eq!(config.renderer.limits.vocab_size, 256);
        assert_eq!(config.renderer.limits.num_reserved_tokens, 8);
        assert_eq!(config.renderer.default_sampling_params.top_k, Some(20));
        assert_eq!(config.renderer.default_sampling_params.min_p, Some(0.1));
        assert_eq!(
            config.renderer.default_chat_template_kwargs,
            HashMap::from([("enable_thinking".to_string(), json!(false))])
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn resolved_metadata_does_not_reopen_a_gguf_model_source() {
        let directory =
            std::env::temp_dir().join(format!("sglang-renderer-{}", uuid::Uuid::new_v4()));
        let tokenizer = directory.join("tokenizer");
        let model = directory.join("model.gguf");
        fs::create_dir_all(&tokenizer).unwrap();
        fs::write(tokenizer.join("tokenizer.json"), "{}").unwrap();
        fs::write(&model, "not needed by the renderer").unwrap();

        let cli = Cli::try_parse_from([
            "sglang-renderer",
            model.to_str().unwrap(),
            "--engine-url",
            "http://127.0.0.1:30001",
            "--tokenizer-path",
            tokenizer.to_str().unwrap(),
            "--context-length",
            "4096",
            "--vocab-size",
            "128",
            "--resolved-sampling-params",
            r#"{"temperature":0.7,"top_k":20}"#,
        ])
        .unwrap();
        let config = cli.into_direct_args().resolve().await.unwrap();

        assert_eq!(config.renderer.model_path, model.to_string_lossy());
        assert_eq!(config.renderer.limits.context_len, 4096);
        assert_eq!(config.renderer.limits.vocab_size, 128);
        assert_eq!(
            config.renderer.default_sampling_params,
            SamplingDefaults {
                temperature: Some(0.7),
                top_k: Some(20),
                ..SamplingDefaults::default()
            }
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn context_derivation_matches_python_key_and_rope_precedence() {
        assert_eq!(
            derive_context_len(&json!({
                "seq_length": 1000,
                "max_position_embeddings": 2000,
                "rope_scaling": {"factor": 4.0}
            }))
            .unwrap(),
            4000
        );
        assert_eq!(
            derive_context_len(&json!({
                "max_position_embeddings": 2000,
                "rope_scaling": {
                    "factor": 4.0,
                    "original_max_position_embeddings": 2000
                }
            }))
            .unwrap(),
            2000
        );
        assert_eq!(derive_context_len(&json!({})).unwrap(), 2048);
    }
}
