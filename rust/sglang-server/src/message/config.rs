//! Runtime configuration: the rust-server boot knobs
//! ([`RustServerServerArgs`]), the scheduler's typed `server_args` handoff
//! ([`ServerArgs`] / [`ModelConfig`]), the [`RuntimeConfig`] pairing them for
//! `runtime::start`, and the native MM pipeline handoff ([`MmSpec`]).
//!
//! [`ServerArgs`] / [`ModelConfig`] / [`DefaultSamplingParams`] /
//! [`DisaggregationMode`] / [`MmSpec`] / [`MmFamily`] / [`MmResample`] are
//! also `#[pyclass]`es: the Python scheduler (`RustServer._build_server_args`
//! / `_build_mm_spec`) constructs them directly by keyword and hands them to
//! `Server`. There is one schema — this file — and
//! pyo3 enforces it at construction: every field is a required, typed
//! constructor argument, so a drifted caller fails at boot rather than running
//! on a silently-defaulted knob. The `#[pyo3::pymethods]` constructors below
//! each struct — plus the one hand-written extraction,
//! [`PreferredSamplingParams`] — are the only Python-facing code in this file;
//! the rest is pure Rust.

use std::net::SocketAddr;
use std::sync::Arc;

use serde::Serialize;

/// Boot knobs specific to the embedded rust server — none of these exist in
/// the Python-built [`ServerArgs`]; they arrive as explicit
/// `Server::start` parameters.
#[derive(Clone, Debug)]
pub struct RustServerServerArgs {
    pub http_addr: SocketAddr,
    pub http_api_worker_num: usize,
    pub to_scheduler_cap: usize,
    pub from_scheduler_cap: usize,
    pub channel_cap: usize,
    /// CPU core ids the pools pin to (e.g. this rank's NUMA-local cores minus
    /// the scheduler's reserved launch cores). `None` → run unpinned.
    pub cores: Option<Vec<usize>>,
}

impl Default for RustServerServerArgs {
    fn default() -> Self {
        Self {
            http_addr: "127.0.0.1:30000".parse().unwrap(),
            http_api_worker_num: 2,
            to_scheduler_cap: 8192,
            from_scheduler_cap: 8192,
            channel_cap: 8192,
            cores: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    /// Rust-server-only boot knobs (listen address, pool/ring sizes, pinning).
    pub rust_server_args: RustServerServerArgs,
    /// The scheduler's [`ServerArgs`] (worker counts, tokenizer source,
    /// config-endpoint metadata). `Arc` so cloning the config (and, downstream,
    /// each `AppState`) is cheap; immutable after construction.
    pub server_args: Arc<ServerArgs>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            rust_server_args: RustServerServerArgs::default(),
            server_args: Arc::new(ServerArgs::default()),
        }
    }
}

/// The scheduler's launch-time handoff (`RustServer._build_server_args`):
/// the `server_args` fields the rust server reads, the resolved
/// [`ModelConfig`], and launch-time stamps. Values are post-`__post_init__`
/// (all paths and names resolved). Constructed from Python via the `#[new]` in
/// `lib.rs`, whose keyword parameters are exactly these fields.
#[pyo3::pyclass(frozen, from_py_object, module = "sglang.srt.rust_extensions._server")]
#[derive(Clone, Debug)]
pub struct ServerArgs {
    /// HF repo id / local dir of the model, reported by `/get_model_info`.
    pub model_path: String,
    /// Model name reported by `/v1/models` and `/server_info`.
    pub served_model_name: String,
    /// Tokenizer source (model dir / `tokenizer.json` / HF repo id). Empty only
    /// in standalone (test) configs — then boot requires `skip_tokenizer_init`.
    pub tokenizer_path: String,
    /// HF revision, used only when `tokenizer_path` is a repo id. `None` → main.
    pub revision: Option<String>,
    /// Weight format selected by `--load-format`, reported by `/get_model_info`.
    /// The blob carries the post-`__post_init__` value (`auto` is already
    /// narrowed to `gguf` / `mistral` / `runai_streamer` / `remote` where the
    /// checkpoint demands it). Not consumed for loading -- the scheduler owns
    /// that; `None` only when the blob omits the key.
    pub load_format: Option<String>,
    /// Operator-supplied weight version, reported by `/model_info`. Defaults to
    /// `"default"` on the Python side, so it is present in every blob; `None`
    /// only when the blob omits the key.
    pub weight_version: Option<String>,
    /// HTTP bind address (see [`Self::bind`]).
    pub host: String,
    pub port: u16,
    /// Log levels driving the access log — uvicorn runs at
    /// `log_level_http or log_level` (see [`Self::http_access_log_enabled`]).
    pub log_level: String,
    pub log_level_http: Option<String>,
    /// Optional built-in chat-template name or path to a Jinja/legacy JSON
    /// template file. Without an override, uses the tokenizer config template.
    pub chat_template: Option<String>,
    /// Parser selected by `--tool-call-parser`.
    pub tool_call_parser: Option<String>,
    /// Reasoning splitter selected by `--reasoning-parser` (e.g. deepseek-r1).
    /// When set, chat completions strip the model's reasoning markers out of
    /// `content` into `reasoning_content` — both unary and streaming.
    pub reasoning_parser: Option<String>,
    /// Python's global default for whether an SSE stream ends with a usage chunk.
    pub stream_response_default_include_usage: bool,
    /// Pinned tokenizer threads / detok shards (Python asserts both ≥ 1).
    pub tokenizer_worker_num: usize,
    pub detokenizer_worker_num: usize,
    /// Token-ids-in / token-ids-out mode: no tokenizer load, raw `output_ids`
    /// frames.
    pub skip_tokenizer_init: bool,
    /// Streamed `/generate` frames carry per-step deltas instead of cumulative
    /// text. Matches the Python `TokenizerManager`.
    pub incremental_streaming_output: bool,
    /// PD-disaggregation role. (On prefill, the KV bootstrap registry is mounted
    /// on the api router — see [`Self::enable_pd_bootstrap`].)
    pub disaggregation_mode: DisaggregationMode,
    /// The resolved Python `ModelConfig`, attached at handoff time.
    pub model_config: ModelConfig,
    /// Default sampling params advertised by `/get_model_info`, verbatim from
    /// `server_args.preferred_sampling_params` (a JSON object or null).
    pub preferred_sampling_params: Option<PreferredSamplingParams>,
    /// Over-long inputs are truncated to fit the context instead of 400ing, and
    /// `max_new_tokens` is clamped rather than rejected (Python
    /// `TokenizerManager._validate_one_request`).
    pub allow_auto_truncate: bool,
    /// `return_hidden_states` is refused unless the server was launched with it:
    /// the scheduler simply won't produce them, so the request would 200 with the
    /// field silently missing.
    pub enable_return_hidden_states: bool,
    /// Output slots reserved per request on top of its input (eagle stores draft
    /// tokens there). Not a `server_args` field — `TokenizerManager` derives it and
    /// `RustServer._build_server_args` stamps it in, so both sides count alike.
    pub num_reserved_tokens: u64,
    /// Launch-time stamps (not `server_args` fields): sglang package version
    /// and the scheduler-derived KV token capacity, reported by `/server_info`.
    pub version: String,
    pub max_total_num_tokens: u64,
}

#[pyo3::pymethods]
impl ServerArgs {
    #[new]
    #[pyo3(signature = (*,
        model_path,
        served_model_name,
        tokenizer_path,
        revision,
        load_format,
        weight_version,
        host,
        port,
        log_level,
        log_level_http,
        chat_template,
        tool_call_parser,
        reasoning_parser,
        stream_response_default_include_usage,
        tokenizer_worker_num,
        detokenizer_worker_num,
        skip_tokenizer_init,
        incremental_streaming_output,
        disaggregation_mode,
        model_config,
        preferred_sampling_params,
        allow_auto_truncate,
        enable_return_hidden_states,
        num_reserved_tokens,
        version,
        max_total_num_tokens,
    ))]
    // The parameter list IS the schema; one keyword per field, all required.
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        model_path: String,
        served_model_name: String,
        tokenizer_path: String,
        revision: Option<String>,
        load_format: Option<String>,
        weight_version: Option<String>,
        host: String,
        port: u16,
        log_level: String,
        log_level_http: Option<String>,
        chat_template: Option<String>,
        tool_call_parser: Option<String>,
        reasoning_parser: Option<String>,
        stream_response_default_include_usage: bool,
        tokenizer_worker_num: usize,
        detokenizer_worker_num: usize,
        skip_tokenizer_init: bool,
        incremental_streaming_output: bool,
        disaggregation_mode: DisaggregationMode,
        model_config: ModelConfig,
        preferred_sampling_params: Option<PreferredSamplingParams>,
        allow_auto_truncate: bool,
        enable_return_hidden_states: bool,
        num_reserved_tokens: u64,
        version: String,
        max_total_num_tokens: u64,
    ) -> Self {
        Self {
            model_path,
            served_model_name,
            tokenizer_path,
            revision,
            load_format,
            weight_version,
            host,
            port,
            log_level,
            log_level_http,
            chat_template,
            tool_call_parser,
            reasoning_parser,
            stream_response_default_include_usage,
            tokenizer_worker_num,
            detokenizer_worker_num,
            skip_tokenizer_init,
            incremental_streaming_output,
            disaggregation_mode,
            model_config,
            preferred_sampling_params,
            allow_auto_truncate,
            enable_return_hidden_states,
            num_reserved_tokens,
            version,
            max_total_num_tokens,
        }
    }
}

impl Default for ServerArgs {
    /// A standalone (test) config: no model, no tokenizer, unified role, but a
    /// complete `model_config` so the runtime boots. Real launches never use
    /// this — Python supplies every field.
    fn default() -> Self {
        Self {
            model_path: String::new(),
            served_model_name: String::new(),
            tokenizer_path: String::new(),
            revision: None,
            load_format: None,
            weight_version: None,
            host: "127.0.0.1".into(),
            port: 30000,
            log_level: "info".into(),
            log_level_http: None,
            chat_template: None,
            tool_call_parser: None,
            reasoning_parser: None,
            stream_response_default_include_usage: false,
            tokenizer_worker_num: 1,
            detokenizer_worker_num: 1,
            skip_tokenizer_init: false,
            incremental_streaming_output: false,
            disaggregation_mode: DisaggregationMode::Null,
            model_config: ModelConfig::default(),
            preferred_sampling_params: None,
            allow_auto_truncate: false,
            enable_return_hidden_states: false,
            num_reserved_tokens: 0,
            version: String::new(),
            max_total_num_tokens: 0,
        }
    }
}

/// `--preferred-sampling-params`, carried verbatim: `/get_model_info` echoes
/// whatever Python advertises, and the keys are whatever `SamplingParams`
/// accepts, so there is no fixed field list to model as a `#[pyclass]`.
#[derive(Clone, Debug, Serialize)]
#[serde(transparent)]
pub struct PreferredSamplingParams(pub serde_json::Value);

impl<'py> pyo3::FromPyObject<'_, 'py> for PreferredSamplingParams {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'_, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let text = obj.extract::<String>()?;
        serde_json::from_str(&text).map(Self).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "preferred_sampling_params is not valid JSON: {e}"
            ))
        })
    }
}

/// PD-disaggregation role, the values of `--disaggregation-mode`. Exposed to
/// Python as an enum (`DisaggregationMode.Null` / `.Prefill` / `.Decode`).
#[pyo3::pyclass(
    eq,
    frozen,
    from_py_object,
    module = "sglang.srt.rust_extensions._server"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisaggregationMode {
    /// Unified prefill + decode.
    Null,
    Prefill,
    Decode,
}

/// The slice of the resolved Python `ModelConfig` the rust server reads.
#[pyo3::pyclass(frozen, from_py_object, module = "sglang.srt.rust_extensions._server")]
#[derive(Clone, Debug)]
pub struct ModelConfig {
    /// Resolved context length (`max_model_len` in `/v1/models`); the ceiling
    /// for input + `max_new_tokens`.
    pub context_len: u64,
    /// Bounds client-supplied token ids — return 400s out-of-vocab ids before
    /// they crash the scheduler's embedding lookup.
    pub vocab_size: u64,
    /// Whether the model accepts multimodal inputs. Gates the MM Encoding branch
    /// in to-scheduler; `false` silently ignores mm fields, as the Python
    /// `TokenizerManager` does with `mm_processor is None`.
    pub is_multimodal: bool,
    /// Resolved default sampling parameters, from Python's
    /// `ModelConfig.get_default_sampling_params()`. Already gated on
    /// `--sampling-defaults`: holds the model's generation_config.json values
    /// in "model" mode, and is all-`None` in "openai" mode. Consumed when a chat
    /// request omits `temperature`/`top_p` — the conversion must not skip
    /// straight to the OpenAI terminal defaults.
    pub default_sampling_params: DefaultSamplingParams,
}

#[pyo3::pymethods]
impl ModelConfig {
    #[new]
    #[pyo3(signature = (*, context_len, vocab_size, is_multimodal, default_sampling_params))]
    fn py_new(
        context_len: u64,
        vocab_size: u64,
        is_multimodal: bool,
        default_sampling_params: DefaultSamplingParams,
    ) -> Self {
        Self {
            context_len,
            vocab_size,
            is_multimodal,
            default_sampling_params,
        }
    }
}

impl Default for ModelConfig {
    /// Test-only: a small but complete model so the runtime boots.
    fn default() -> Self {
        Self {
            context_len: 2048,
            vocab_size: 1000,
            is_multimodal: false,
            default_sampling_params: DefaultSamplingParams::default(),
        }
    }
}

/// One `SamplingParams` field per key `get_default_sampling_params()` may emit
/// (`repetition_penalty`, `temperature`, `top_k`, `top_p`, `min_p`); `None`
/// where the generation config does not set it.
///
/// `top_k` / `min_p` / `repetition_penalty` are carried for parity with the
/// Python dict but not yet consumed: the Dynamo chat request type only carries
/// `temperature` and `top_p`, so the conversion resolves just those two.
#[pyo3::pyclass(frozen, from_py_object, module = "sglang.srt.rust_extensions._server")]
#[derive(Clone, Debug, Default)]
#[allow(dead_code)]
pub struct DefaultSamplingParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
}

#[pyo3::pymethods]
impl DefaultSamplingParams {
    #[new]
    #[pyo3(signature = (*, temperature = None, top_p = None, top_k = None, min_p = None, repetition_penalty = None))]
    fn py_new(
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<i64>,
        min_p: Option<f64>,
        repetition_penalty: Option<f64>,
    ) -> Self {
        Self {
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
        }
    }
}

/// The native MM pipeline handoff, built by `RustServer._build_mm_spec` from
/// the resolved `NativeMmSpec` and passed to `Server.start_mm_workers`. Same
/// contract as [`ServerArgs`]: every field is a required, typed constructor
/// keyword, so a drifted Python caller fails at boot.
#[pyo3::pyclass(frozen, from_py_object, module = "sglang.srt.rust_extensions._server")]
#[derive(Clone, Debug)]
pub struct MmSpec {
    /// Park feature buffers in POSIX shm rather than inline. Set by the Python
    /// launcher (`NativeMmHost._use_feature_shm`) exactly when the scheduler
    /// broadcasts across TP ranks and will unwrap `ShmPointerMMData`.
    pub feature_shm: bool,
    /// The family pipeline and its resolved processor parameters.
    pub pipeline: sglang_mm::registry::PipelineSpec,
}

#[pyo3::pymethods]
impl MmSpec {
    /// The parameter list is flat because every family so far shares the
    /// Qwen-VL processor geometry; a family with different knobs adds its own
    /// keywords and match arm here.
    #[new]
    #[pyo3(signature = (*,
        family,
        feature_shm,
        image_token_id,
        patch_size,
        merge_size,
        temporal_patch_size,
        min_pixels,
        max_pixels,
        image_mean,
        image_std,
        resample,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        family: MmFamily,
        feature_shm: bool,
        image_token_id: i32,
        patch_size: usize,
        merge_size: usize,
        temporal_patch_size: usize,
        min_pixels: usize,
        max_pixels: usize,
        image_mean: [f32; 3],
        image_std: [f32; 3],
        resample: MmResample,
    ) -> Self {
        use sglang_mm::registry::PipelineSpec;
        let pipeline = match family {
            MmFamily::QwenVl => PipelineSpec::QwenVl(sglang_mm::qwen_vl::QwenVlSpec {
                image_token_id,
                patch_size,
                merge_size,
                temporal_patch_size,
                min_pixels,
                max_pixels,
                image_mean,
                image_std,
                resample: resample.into(),
            }),
        };
        Self {
            feature_shm,
            pipeline,
        }
    }
}

/// Which `sglang_mm` family pipeline serves the model — one variant per
/// [`sglang_mm::registry::PipelineSpec`] arm. Exposed to Python as an enum
/// (`MmFamily.QwenVl`); `NativeMmFamily.name` maps onto it at handoff.
#[pyo3::pyclass(
    eq,
    frozen,
    from_py_object,
    module = "sglang.srt.rust_extensions._server"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmFamily {
    QwenVl,
}

/// The HF image processor the native resize must reproduce bit-exactly (see
/// [`sglang_mm::qwen_vl::Resampler`]). Exposed to Python as an enum
/// (`MmResample.AtenU8` / `.Pil`); `NativeMmFamily.image_processors` maps each
/// processor class onto it.
#[pyo3::pyclass(
    eq,
    frozen,
    from_py_object,
    module = "sglang.srt.rust_extensions._server"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmResample {
    /// `Qwen2VLImageProcessor` / `…Fast` — torchvision on a uint8 tensor.
    AtenU8,
    /// `Qwen2VLImageProcessorPil`, behind `--disable-fast-image-processor`.
    Pil,
}

impl From<MmResample> for sglang_mm::qwen_vl::Resampler {
    fn from(r: MmResample) -> Self {
        match r {
            MmResample::AtenU8 => Self::AtenU8,
            MmResample::Pil => Self::Pil,
        }
    }
}

fn join_host_port(host: &str, port: u16) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]:{port}") // bare IPv6 (`::`) needs brackets to bind
    } else {
        format!("{host}:{port}")
    }
}

impl ServerArgs {
    /// Fail fast at startup on values the types cannot express.
    pub fn validate(&self) -> Result<(), String> {
        if self.served_model_name.is_empty() {
            return Err("empty 'served_model_name' in server_args".into());
        }
        Ok(())
    }

    /// True on a prefill or decode node — requests need bootstrap routing.
    pub fn is_disaggregation(&self) -> bool {
        self.disaggregation_mode != DisaggregationMode::Null
    }

    /// Serve the PD KV bootstrap registry on the api listener: every prefill
    /// rust server hosts it, unconditionally — no extra topology gating. KV
    /// managers and decode nodes reach the registry at the resolved
    /// `disaggregation_bootstrap_port`, which rust-server mode aliases to the
    /// api port, so whichever prefill server that port names is the one that
    /// receives the registrations.
    pub fn enable_pd_bootstrap(&self) -> bool {
        self.disaggregation_mode == DisaggregationMode::Prefill
    }

    /// Whether the served model is multimodal, from the scheduler's config. See
    /// [`ModelConfig::is_multimodal`].
    pub fn model_is_multimodal(&self) -> bool {
        self.model_config.is_multimodal
    }

    /// Bind address `host:port`. `host` is expected to be an IP — the result is
    /// parsed as a `SocketAddr`, so a bare IPv6 host gets bracketed.
    pub fn bind(&self) -> String {
        join_host_port(&self.host, self.port)
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
    pub fn http_api_worker_num(&self) -> usize {
        4.max(self.tokenizer_worker_num)
            .max(self.detokenizer_worker_num)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bind_brackets_bare_ipv6() {
        let sa = ServerArgs {
            host: "::".into(),
            port: 30001,
            ..Default::default()
        };
        assert_eq!(sa.bind(), "[::]:30001");
        assert_eq!(ServerArgs::default().bind(), "127.0.0.1:30000");
    }

    #[test]
    fn pd_role_derivations() {
        let prefill = ServerArgs {
            disaggregation_mode: DisaggregationMode::Prefill,
            ..Default::default()
        };
        assert!(prefill.is_disaggregation());
        assert!(prefill.enable_pd_bootstrap());
        let decode = ServerArgs {
            disaggregation_mode: DisaggregationMode::Decode,
            ..Default::default()
        };
        assert!(decode.is_disaggregation());
        assert!(!decode.enable_pd_bootstrap());
        assert!(!ServerArgs::default().is_disaggregation());
    }

    #[test]
    fn validate_requires_served_model_name() {
        assert!(ServerArgs::default().validate().is_err());
        let sa = ServerArgs {
            served_model_name: "m".into(),
            ..Default::default()
        };
        assert!(sa.validate().is_ok());
    }

    /// `--log-level-http` overrides `--log-level` for the access log; unset or
    /// empty falls through.
    #[test]
    fn access_log_follows_http_level_then_global() {
        let mut sa = ServerArgs::default();
        assert!(sa.http_access_log_enabled());
        sa.log_level_http = Some("warning".into());
        assert!(!sa.http_access_log_enabled());
        sa.log_level_http = Some(String::new());
        sa.log_level = "error".into();
        assert!(!sa.http_access_log_enabled());
    }
}
