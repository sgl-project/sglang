//! Reusable SGLang request preprocessing.

use std::sync::Arc;

use dynamo_renderer::deepseek::v4::DeepSeekV4Formatter;
use dynamo_renderer::deepseek::v32::DeepSeekV32Formatter;
use dynamo_renderer::{PromptFormatter, kimi_k3_formatter_for, native_formatter_for};
use futures::future::try_join_all;

use super::template::{DeepSeekV4Profile, ThinkingTemplates, load_chat_formatter};
use super::tokenizer::{
    PooledTokenizer, TextTokenizer, check_total_tokens, resolve_chat_template_file,
    resolve_model_file, validate_request_id, validate_text_request, validate_token_ids_request,
};
use crate::{
    ChatFormatter, ChatPreprocessor, ChatRequest, ChatResponseProcessor, GenerateRequest,
    LoweredChat, RendererConfig, RendererError, TextRequest, TokenIdsRequest,
};

use super::TextRequestGroup;

/// Shared preprocessing used by inference and render-only frontends.
pub struct RendererService {
    config: RendererConfig,
    chat_preprocessor: ChatPreprocessor,
    tokenizer: PooledTokenizer,
}

/// Prepared token-only chat requests plus the state needed to interpret their
/// generated output.
pub struct PreparedChat {
    pub requests: Vec<GenerateRequest>,
    pub response_processor: ChatResponseProcessor,
}

impl RendererService {
    pub fn with_tokenizer(
        config: RendererConfig,
        tokenizer: Arc<dyn TextTokenizer>,
        worker_count: usize,
        queue_capacity: usize,
    ) -> Self {
        let (formatter, formatter_error) = load_chat_support(&config);
        let chat_preprocessor =
            ChatPreprocessor::new(&config, formatter).with_formatter_error(formatter_error);
        Self {
            config,
            chat_preprocessor,
            tokenizer: PooledTokenizer::new(tokenizer, worker_count, queue_capacity),
        }
    }

    pub fn config(&self) -> &RendererConfig {
        &self.config
    }

    pub(crate) fn preprocess_chat(
        &self,
        request: ChatRequest,
    ) -> Result<LoweredChat, RendererError> {
        self.chat_preprocessor.preprocess(request)
    }

    pub async fn prepare_chat(&self, request: ChatRequest) -> Result<PreparedChat, RendererError> {
        let lowered = self.preprocess_chat(request)?;
        Ok(PreparedChat {
            requests: self
                .prepare_text_request_groups(lowered.text_requests)
                .await?,
            response_processor: lowered.response_processor,
        })
    }

    pub async fn prepare_text_requests(
        &self,
        requests: Vec<TextRequest>,
    ) -> Result<Vec<GenerateRequest>, RendererError> {
        self.prepare_text_request_groups(requests.into_iter().map(Into::into).collect())
            .await
    }

    pub(crate) async fn prepare_text_request_groups(
        &self,
        groups: Vec<TextRequestGroup>,
    ) -> Result<Vec<GenerateRequest>, RendererError> {
        let groups = try_join_all(
            groups
                .into_iter()
                .map(|group| async move { self.prepare_text_request_group(group).await }),
        )
        .await?;
        Ok(groups.into_iter().flatten().collect())
    }

    pub async fn tokenize_prompt(
        &self,
        text: String,
        add_special_tokens: bool,
    ) -> Result<crate::TokenIds, RendererError> {
        let request = TextRequest::text("tokenize", text, add_special_tokens, Default::default());
        Ok(self.tokenizer.tokenize(request).await?.input_ids)
    }

    pub async fn tokenize_chat(
        &self,
        request: ChatRequest,
    ) -> Result<crate::TokenIds, RendererError> {
        let request = self.chat_preprocessor.lower_to_text(request)?;
        Ok(self.tokenizer.tokenize(request).await?.input_ids)
    }

    pub fn prepare_token_ids_requests(
        &self,
        requests: Vec<TokenIdsRequest>,
    ) -> Result<Vec<GenerateRequest>, RendererError> {
        requests
            .into_iter()
            .map(|request| self.prepare_token_ids(request).map(GenerateRequest::from))
            .collect()
    }

    async fn prepare_text(
        &self,
        mut request: TextRequest,
    ) -> Result<TokenIdsRequest, RendererError> {
        validate_text_request(&request, &self.config.limits)?;
        request
            .options
            .sampling_params
            .normalize(self.config.limits.vocab_size)?;
        let mut request = self.tokenizer.tokenize(request).await?;
        check_total_tokens(&mut request, &self.config.limits)?;
        Ok(request)
    }

    async fn prepare_text_request_group(
        &self,
        group: TextRequestGroup,
    ) -> Result<Vec<GenerateRequest>, RendererError> {
        let TextRequestGroup {
            prompt,
            add_special_tokens,
            options,
            requests,
        } = group;
        for request in &requests {
            validate_request_id(&request.rid)?;
        }
        let mut requests = requests.into_iter();
        let first = requests
            .next()
            .ok_or_else(|| RendererError::from("text request group must contain a request"))?;
        let tokenized = self
            .prepare_text(TextRequest {
                rid: first.rid,
                prompt,
                add_special_tokens,
                options,
                metadata: first.metadata,
            })
            .await?;
        let additional = requests
            .map(|request| {
                GenerateRequest::from(TokenIdsRequest {
                    rid: request.rid,
                    input_ids: tokenized.input_ids.clone(),
                    options: tokenized.options.clone(),
                    metadata: request.metadata,
                })
            })
            .collect::<Vec<_>>();
        let mut prepared = Vec::with_capacity(1 + additional.len());
        prepared.push(tokenized.into());
        prepared.extend(additional);
        Ok(prepared)
    }

    fn prepare_token_ids(
        &self,
        mut request: TokenIdsRequest,
    ) -> Result<TokenIdsRequest, RendererError> {
        validate_token_ids_request(&request, &self.config.limits)?;
        request
            .options
            .sampling_params
            .normalize(self.config.limits.vocab_size)?;
        check_total_tokens(&mut request, &self.config.limits)?;
        Ok(request)
    }
}

fn load_chat_support(config: &RendererConfig) -> (Option<ChatFormatter>, Option<String>) {
    if config.tokenizer_path.is_empty() {
        return (None, None);
    }
    let tokenizer_config_file = resolve_model_file(
        &config.tokenizer_path,
        config.revision.as_deref(),
        "tokenizer_config.json",
    );
    let model_source = if config.model_path.is_empty() {
        config.tokenizer_path.as_str()
    } else {
        config.model_path.as_str()
    };
    let model_config_file =
        resolve_model_file(model_source, config.revision.as_deref(), "config.json");
    let identity = match model_config_file.as_deref().map(load_model_identity) {
        Some(Err(error)) => return (None, Some(error)),
        Some(Ok(identity)) => identity,
        None => ModelIdentity::default(),
    };
    let model_type_lower = identity.model_type.as_deref().map(str::to_ascii_lowercase);
    let display_name_lower = model_source.to_ascii_lowercase();
    if config.chat_template.is_none() {
        if identity.is_deepseek_v4() {
            let profile =
                match resolve_dsv4_profile(&identity, model_source, config.revision.as_deref()) {
                    Ok(profile) => profile,
                    Err(error) => return (None, Some(error)),
                };
            return (
                Some(ChatFormatter::DeepSeekV4 {
                    formatter: PromptFormatter::OAI(Arc::new(DeepSeekV4Formatter::new_chat())),
                    profile,
                    environment_effort: std::env::var("SGLANG_DSV4_REASONING_EFFORT").ok(),
                }),
                None,
            );
        }
        if identity.is_deepseek_v32() {
            return (
                Some(ChatFormatter::HuggingFace {
                    formatter: PromptFormatter::OAI(Arc::new(DeepSeekV32Formatter::new_chat())),
                    thinking: ThinkingTemplates::native(false, false),
                }),
                None,
            );
        }
        if let Some(formatter) = kimi_k3_formatter_for(&model_type_lower, &display_name_lower, true)
        {
            return (
                Some(ChatFormatter::HuggingFace {
                    formatter,
                    thinking: ThinkingTemplates::native(true, true),
                }),
                None,
            );
        }
        if model_type_lower.as_deref() == Some("inkling_mm_model")
            && let Some(formatter) = native_formatter_for(&model_type_lower, &display_name_lower)
        {
            return (
                Some(ChatFormatter::HuggingFace {
                    formatter,
                    thinking: ThinkingTemplates::always(),
                }),
                None,
            );
        }
        if let Some(formatter) = native_formatter_for(&model_type_lower, &display_name_lower) {
            return (
                Some(ChatFormatter::HuggingFace {
                    formatter,
                    // The remaining display-name fallback formatters are
                    // constructed with their thinking mode enabled.
                    thinking: ThinkingTemplates::native(true, false),
                }),
                None,
            );
        }
    }
    let discovered_template = config
        .chat_template
        .is_none()
        .then(|| resolve_chat_template_file(&config.tokenizer_path, config.revision.as_deref()))
        .flatten();
    let template_source = config
        .chat_template
        .as_deref()
        .or(discovered_template.as_deref());
    match load_chat_formatter(
        tokenizer_config_file.as_deref(),
        (!config.model_path.is_empty()).then_some(config.model_path.as_str()),
        template_source,
    ) {
        Ok(mut formatter) => {
            if identity.is_kimi_k25()
                && let ChatFormatter::HuggingFace {
                    formatter: inner,
                    thinking,
                } = formatter
            {
                formatter = ChatFormatter::KimiK25 {
                    formatter: inner,
                    thinking,
                };
            }
            tracing::info!(
                config = ?tokenizer_config_file.as_deref().unwrap_or("<built-in / inferred>"),
                template = ?template_source,
                "loaded OpenAI chat template"
            );
            (Some(formatter), None)
        }
        Err(error) => {
            tracing::warn!(%error, "OpenAI chat completions disabled");
            (
                None,
                Some(format!("this model has no usable chat template: {error}")),
            )
        }
    }
}

#[derive(Debug, Default)]
struct ModelIdentity {
    model_type: Option<String>,
    architectures: Vec<String>,
    dsv4_reasoning_effort_profile: Option<String>,
}

impl ModelIdentity {
    fn is_deepseek_v4(&self) -> bool {
        self.model_type.as_deref() == Some("deepseek_v4")
            || self
                .architectures
                .iter()
                .any(|architecture| architecture.starts_with("DeepseekV4"))
    }

    fn is_deepseek_v32(&self) -> bool {
        matches!(
            self.model_type.as_deref(),
            Some("deepseek_v32" | "deepseek_v3_2")
        ) || self
            .architectures
            .iter()
            .any(|architecture| architecture == "DeepseekV32ForCausalLM")
    }

    fn is_kimi_k25(&self) -> bool {
        self.model_type.as_deref() == Some("kimi_k25")
            || self
                .architectures
                .iter()
                .any(|architecture| architecture == "KimiK25ForConditionalGeneration")
    }
}

fn load_model_identity(config_file: &str) -> Result<ModelIdentity, String> {
    let Ok(config) = std::fs::read_to_string(config_file) else {
        return Ok(ModelIdentity::default());
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&config) else {
        return Ok(ModelIdentity::default());
    };
    let model_type = config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let architectures = config
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .map(|architectures| {
            architectures
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let dsv4_reasoning_effort_profile = match config.get("dsv4_reasoning_effort_profile") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(profile)) => Some(profile.clone()),
        Some(profile) => {
            return Err(format!(
                "invalid dsv4_reasoning_effort_profile: {profile}; expected \"preview\" or \"official\""
            ));
        }
    };
    Ok(ModelIdentity {
        model_type,
        architectures,
        dsv4_reasoning_effort_profile,
    })
}

fn resolve_dsv4_profile(
    identity: &ModelIdentity,
    model_source: &str,
    revision: Option<&str>,
) -> Result<DeepSeekV4Profile, String> {
    if let Some(profile) = identity.dsv4_reasoning_effort_profile.as_deref() {
        return match profile {
            "preview" => Ok(DeepSeekV4Profile::Preview),
            "official" => Ok(DeepSeekV4Profile::Official),
            _ => Err(format!(
                "invalid dsv4_reasoning_effort_profile: {profile:?}; expected \"preview\" or \"official\""
            )),
        };
    }
    let Some(encoder) = resolve_model_file(model_source, revision, "encoding/encoding_dsv4.py")
    else {
        return Ok(DeepSeekV4Profile::Preview);
    };
    let Ok(metadata) = std::fs::metadata(&encoder) else {
        return Ok(DeepSeekV4Profile::Preview);
    };
    if metadata.len() > 1 << 20 {
        return Ok(DeepSeekV4Profile::Preview);
    }
    let Ok(source) = std::fs::read_to_string(encoder) else {
        return Ok(DeepSeekV4Profile::Preview);
    };
    let default = top_level_python_assignment(&source, "DEFAULT_REASONING_EFFORT")
        .and_then(python_string_literal);
    let prompt_keys = top_level_python_assignment(&source, "REASONING_EFFORT_PROMPTS")
        .and_then(python_dict_keys)
        .unwrap_or_default();
    if default.as_deref() == Some("low")
        && ["low", "high", "max"]
            .iter()
            .all(|key| prompt_keys.iter().any(|candidate| candidate == key))
    {
        Ok(DeepSeekV4Profile::Official)
    } else {
        Ok(DeepSeekV4Profile::Preview)
    }
}

fn top_level_python_assignment<'a>(source: &'a str, name: &str) -> Option<&'a str> {
    let mut offset = 0;
    for line in source.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if !trimmed.starts_with(char::is_whitespace)
            && let Some((target, _)) = trimmed.split_once('=')
            && target
                .split(':')
                .next()
                .is_some_and(|target| target.trim() == name)
        {
            let equals = line.find('=')?;
            return Some(&source[offset + equals + 1..]);
        }
        offset += line.len();
    }
    None
}

fn python_string_literal(source: &str) -> Option<String> {
    let source = source.trim_start();
    let quote = source.chars().next()?;
    if !matches!(quote, '\'' | '"') {
        return None;
    }
    let mut escaped = false;
    let mut value = String::new();
    for character in source[quote.len_utf8()..].chars() {
        if escaped {
            value.push(character);
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else if character == quote {
            return Some(value);
        } else {
            value.push(character);
        }
    }
    None
}

fn python_dict_keys(source: &str) -> Option<Vec<String>> {
    let source = source.trim_start();
    if !source.starts_with('{') {
        return None;
    }
    let mut keys = Vec::new();
    let mut depth = 0usize;
    let mut index = 0usize;
    let bytes = source.as_bytes();
    while index < bytes.len() {
        match bytes[index] {
            b'{' | b'[' | b'(' => {
                depth += 1;
                index += 1;
            }
            b'}' | b']' | b')' => {
                depth = depth.checked_sub(1)?;
                index += 1;
                if depth == 0 {
                    return Some(keys);
                }
            }
            quote @ (b'\'' | b'"') => {
                let start = index + 1;
                index = start;
                let mut escaped = false;
                while index < bytes.len() {
                    if escaped {
                        escaped = false;
                    } else if bytes[index] == b'\\' {
                        escaped = true;
                    } else if bytes[index] == quote {
                        break;
                    }
                    index += 1;
                }
                if index == bytes.len() {
                    return None;
                }
                let value = std::str::from_utf8(&bytes[start..index]).ok()?;
                index += 1;
                if depth == 1 {
                    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
                        index += 1;
                    }
                    if bytes.get(index) == Some(&b':') {
                        keys.push(value.to_owned());
                    }
                }
            }
            b'#' => {
                while index < bytes.len() && bytes[index] != b'\n' {
                    index += 1;
                }
            }
            _ => index += 1,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::GenerateRequestIdentity;
    use crate::{
        GenerateRequestMetadata, GenerationOptions, OneOrMany, RendererLimits, SamplingDefaults,
        SamplingParams,
    };
    use dynamo_protocols::types::{ChatCompletionRequestMessage, CreateChatCompletionRequest};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn model_config(model_path: String) -> RendererConfig {
        RendererConfig {
            served_model_name: "model".into(),
            tokenizer_path: model_path.clone(),
            revision: None,
            model_path,
            chat_template: None,
            tool_call_parser: None,
            reasoning_parser: None,
            default_chat_template_kwargs: Default::default(),
            stream_response_default_include_usage: false,
            default_sampling_params: SamplingDefaults::default(),
            limits: RendererLimits {
                vocab_size: 128,
                context_len: 128,
                num_reserved_tokens: 0,
                allow_auto_truncate: false,
                enable_return_hidden_states: false,
            },
        }
    }

    fn chat_request() -> ChatRequest {
        ChatRequest {
            rid: "chatcmpl-test".into(),
            model: "model".into(),
            messages: serde_json::from_value(serde_json::json!([
                {"role": "user", "content": "hello"}
            ]))
            .unwrap(),
            tools: None,
            tool_choice: None,
            response_format: None,
            reasoning_effort: None,
            continue_final_message: false,
            chat_template_args: None,
            sampling_params: SamplingParams::default(),
            choice_count: 1,
            stream: false,
            return_logprob: false,
            top_logprobs_num: 0,
            parallel_tool_calls: true,
            metadata: GenerateRequestMetadata::default(),
        }
    }

    struct UnexpectedTokenizer;

    impl TextTokenizer for UnexpectedTokenizer {
        fn encode(
            &self,
            _text: &str,
            _add_special_tokens: bool,
        ) -> Result<crate::TokenIds, RendererError> {
            panic!("token-ID input must not enter the tokenizer")
        }
    }

    struct CountingTokenizer {
        calls: Arc<AtomicUsize>,
    }

    impl TextTokenizer for CountingTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> Result<crate::TokenIds, RendererError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(text.split_whitespace().map(|_| 7).collect())
        }
    }

    #[test]
    fn text_choices_tokenize_once_per_prompt() {
        futures::executor::block_on(async {
            let calls = Arc::new(AtomicUsize::new(0));
            let service = RendererService::with_tokenizer(
                model_config(String::new()),
                Arc::new(CountingTokenizer {
                    calls: calls.clone(),
                }),
                2,
                4,
            );
            let group = |prompt: &str, ids: &[&str]| TextRequestGroup {
                prompt: dynamo_renderer::RenderedPrompt::text(prompt.to_owned()),
                add_special_tokens: true,
                options: GenerationOptions {
                    sampling_params: SamplingParams {
                        max_new_tokens: Some(4),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                requests: ids
                    .iter()
                    .map(|rid| GenerateRequestIdentity {
                        rid: (*rid).to_owned(),
                        metadata: GenerateRequestMetadata::default(),
                    })
                    .collect(),
            };

            let prepared = service
                .prepare_text_request_groups(vec![
                    group("one two", &["a-0", "a-1", "a-2"]),
                    group("three", &["b-0", "b-1"]),
                ])
                .await
                .unwrap();

            assert_eq!(calls.load(Ordering::Relaxed), 2);
            assert_eq!(
                prepared
                    .iter()
                    .map(|request| request.rid.as_str())
                    .collect::<Vec<_>>(),
                ["a-0", "a-1", "a-2", "b-0", "b-1"]
            );
            assert_eq!(prepared[0].input_ids, [7, 7]);
            assert_eq!(prepared[2].input_ids, [7, 7]);
            assert_eq!(prepared[3].input_ids, [7]);
        });
    }

    #[test]
    fn chat_lowering_carries_rendered_prompt_and_template_stops() {
        let config = RendererConfig {
            served_model_name: "model".into(),
            tokenizer_path: ".".into(),
            revision: None,
            model_path: String::new(),
            chat_template: Some("chatml".into()),
            tool_call_parser: None,
            reasoning_parser: None,
            default_chat_template_kwargs: Default::default(),
            stream_response_default_include_usage: false,
            default_sampling_params: SamplingDefaults::default(),
            limits: RendererLimits {
                vocab_size: 128,
                context_len: 128,
                num_reserved_tokens: 0,
                allow_auto_truncate: false,
                enable_return_hidden_states: false,
            },
        };
        let messages: Vec<ChatCompletionRequestMessage> =
            serde_json::from_value(serde_json::json!([
                {"role": "user", "content": "hello"}
            ]))
            .unwrap();
        let request = ChatRequest {
            rid: "chatcmpl-test".into(),
            model: "model".into(),
            messages,
            tools: None,
            tool_choice: None,
            response_format: None,
            reasoning_effort: None,
            continue_final_message: false,
            chat_template_args: Some(std::collections::HashMap::from([(
                "enable_thinking".to_owned(),
                serde_json::Value::Bool(false),
            )])),
            sampling_params: SamplingParams {
                stop: Some(OneOrMany::One("client-stop".into())),
                ..Default::default()
            },
            choice_count: 1,
            stream: false,
            return_logprob: false,
            top_logprobs_num: 0,
            parallel_tool_calls: true,
            metadata: GenerateRequestMetadata::default(),
        };

        let service = RendererService::with_tokenizer(config, Arc::new(UnexpectedTokenizer), 1, 1);
        let chat = service.preprocess_chat(request).unwrap();
        let text_request = &chat.text_requests[0];

        assert!(text_request.prompt.as_str().contains("<|im_start|>user"));
        assert!(matches!(
            text_request.options.sampling_params.stop.as_ref(),
            Some(OneOrMany::Many(stops))
                if stops.iter().map(String::as_str).collect::<Vec<_>>()
                    == ["<|endoftext|>", "<|im_end|>", "client-stop"]
        ));
    }

    #[test]
    fn dedicated_jinja_template_is_discovered_from_model_directory() {
        let directory = std::env::temp_dir().join(format!(
            "sglang-renderer-dedicated-template-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("tokenizer_config.json"), "{}").unwrap();
        std::fs::write(
            directory.join("chat_template.jinja"),
            "{% for message in messages %}{{ message.content }}{% endfor %}",
        )
        .unwrap();

        let (formatter, error) =
            load_chat_support(&model_config(directory.to_string_lossy().into_owned()));

        assert!(formatter.is_some(), "{error:?}");
        assert!(error.is_none());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn kimi_k3_native_formatter_preserves_segments() {
        let directory =
            std::env::temp_dir().join(format!("sglang-renderer-kimi-k3-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("config.json"), r#"{"model_type":"kimi_k3"}"#).unwrap();
        let (formatter, error) =
            load_chat_support(&model_config(directory.to_string_lossy().into_owned()));
        let formatter = formatter.unwrap_or_else(|| panic!("{error:?}"));
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

        let prompt = formatter.render_prompt(&request).unwrap();

        assert!(prompt.segments().is_some());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn native_formatters_forward_their_effective_thinking_mode() {
        let root = std::env::temp_dir().join(format!(
            "sglang-renderer-native-thinking-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();

        let kimi = root.join("kimi");
        std::fs::create_dir_all(&kimi).unwrap();
        std::fs::write(kimi.join("config.json"), r#"{"model_type":"kimi_k3"}"#).unwrap();
        let mut config = model_config(kimi.to_string_lossy().into_owned());
        config.reasoning_parser = Some("kimi_k3".into());
        config.tool_call_parser = Some("kimi_k3".into());
        let service = RendererService::with_tokenizer(config, Arc::new(UnexpectedTokenizer), 1, 1);
        let enabled = service.preprocess_chat(chat_request()).unwrap();
        assert!(enabled.text_requests[0].options.require_reasoning);

        let mut disabled_request = chat_request();
        disabled_request.chat_template_args = Some(std::collections::HashMap::from([(
            "thinking".into(),
            serde_json::Value::Bool(false),
        )]));
        let disabled = service.preprocess_chat(disabled_request).unwrap();
        assert!(!disabled.text_requests[0].options.require_reasoning);

        let mut named_request = chat_request();
        named_request.tools = serde_json::from_value(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}}
            }
        }]))
        .unwrap();
        named_request.tool_choice = serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": {"name": "get_weather"}
        }))
        .unwrap();
        let named = service.preprocess_chat(named_request).unwrap();
        assert!(!named.text_requests[0].options.require_reasoning);

        let deepseek = root.join("deepseek");
        std::fs::create_dir_all(&deepseek).unwrap();
        std::fs::write(
            deepseek.join("config.json"),
            r#"{"model_type":"deepseek_v32"}"#,
        )
        .unwrap();
        let mut config = model_config(deepseek.to_string_lossy().into_owned());
        config.reasoning_parser = Some("deepseek-v3".into());
        let service = RendererService::with_tokenizer(config, Arc::new(UnexpectedTokenizer), 1, 1);
        let default = service.preprocess_chat(chat_request()).unwrap();
        assert!(!default.text_requests[0].options.require_reasoning);

        let mut explicit = chat_request();
        explicit.chat_template_args = Some(std::collections::HashMap::from([(
            "enable_thinking".into(),
            serde_json::Value::Bool(true),
        )]));
        let explicit = service.preprocess_chat(explicit).unwrap();
        assert!(explicit.text_requests[0].options.require_reasoning);

        let mut effort = chat_request();
        effort.reasoning_effort = Some(serde_json::from_value(serde_json::json!("high")).unwrap());
        let effort = service.preprocess_chat(effort).unwrap();
        assert!(effort.text_requests[0].options.require_reasoning);

        let inkling = root.join("inkling");
        std::fs::create_dir_all(&inkling).unwrap();
        std::fs::write(
            inkling.join("config.json"),
            r#"{"model_type":"inkling_mm_model"}"#,
        )
        .unwrap();
        let mut config = model_config(inkling.to_string_lossy().into_owned());
        config.reasoning_parser = Some("inkling".into());
        let service = RendererService::with_tokenizer(config, Arc::new(UnexpectedTokenizer), 1, 1);
        let inkling = service.preprocess_chat(chat_request()).unwrap();
        assert!(inkling.text_requests[0].options.require_reasoning);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn explicit_chat_template_overrides_native_model_detection() {
        for model_type in ["kimi_k3", "deepseek_v4", "deepseek_v32", "inkling_mm_model"] {
            let directory = std::env::temp_dir().join(format!(
                "sglang-renderer-template-override-{model_type}-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(
                directory.join("config.json"),
                serde_json::json!({"model_type": model_type}).to_string(),
            )
            .unwrap();
            std::fs::write(directory.join("tokenizer_config.json"), "{}").unwrap();
            let mut config = model_config(directory.to_string_lossy().into_owned());
            let template = directory.join("override.jinja");
            std::fs::write(&template, "OVERRIDE {{ messages[0].content }}").unwrap();
            config.chat_template = Some(template.to_string_lossy().into_owned());
            let (formatter, error) = load_chat_support(&config);
            let formatter = formatter.unwrap_or_else(|| panic!("{model_type}: {error:?}"));
            let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model": "model",
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .unwrap();

            let prompt = formatter.render_prompt(&request).unwrap();

            assert_eq!(prompt.as_str(), "OVERRIDE hello", "{model_type}");
            std::fs::remove_dir_all(directory).unwrap();
        }
    }

    #[test]
    fn top_level_reasoning_effort_reaches_deepseek_v4_formatter() {
        let directory = std::env::temp_dir().join(format!(
            "sglang-renderer-deepseek-v4-effort-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("config.json"),
            r#"{"model_type":"deepseek_v4"}"#,
        )
        .unwrap();
        let mut config = model_config(directory.to_string_lossy().into_owned());
        config.reasoning_parser = Some("deepseek-v4".into());
        let messages = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": "hello"}
        ]))
        .unwrap();
        let request = ChatRequest {
            rid: "chatcmpl-test".into(),
            model: "model".into(),
            messages,
            tools: None,
            tool_choice: None,
            response_format: None,
            reasoning_effort: Some(serde_json::from_value(serde_json::json!("max")).unwrap()),
            continue_final_message: false,
            chat_template_args: None,
            sampling_params: SamplingParams::default(),
            choice_count: 1,
            stream: false,
            return_logprob: false,
            top_logprobs_num: 0,
            parallel_tool_calls: true,
            metadata: GenerateRequestMetadata::default(),
        };
        let service = RendererService::with_tokenizer(config, Arc::new(UnexpectedTokenizer), 1, 1);

        let mut default_request = request.clone();
        default_request.reasoning_effort = None;
        let default_chat = service.preprocess_chat(default_request.clone()).unwrap();
        assert!(!default_chat.text_requests[0].options.require_reasoning);

        default_request.chat_template_args = Some(std::collections::HashMap::from([(
            "thinking".into(),
            serde_json::Value::Bool(false),
        )]));
        let explicit_chat = service.preprocess_chat(default_request).unwrap();
        assert_eq!(
            default_chat.text_requests[0].prompt,
            explicit_chat.text_requests[0].prompt
        );

        let chat = service.preprocess_chat(request).unwrap();

        assert!(chat.text_requests[0].options.require_reasoning);
        assert!(
            chat.text_requests[0]
                .prompt
                .as_str()
                .contains("Reasoning Effort: Absolute maximum")
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn deepseek_v32_metadata_overrides_bundled_jinja_for_exp_checkpoints() {
        let directory = std::env::temp_dir().join(format!(
            "sglang-renderer-deepseek-v32-exp-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("config.json"),
            r#"{"model_type":"deepseek_v32","architectures":["DeepseekV32ForCausalLM"]}"#,
        )
        .unwrap();
        std::fs::write(
            directory.join("tokenizer_config.json"),
            r#"{"chat_template":"BUNDLED TEMPLATE WITHOUT TOOLS"}"#,
        )
        .unwrap();
        let (formatter, error) =
            load_chat_support(&model_config(directory.to_string_lossy().into_owned()));
        let formatter = formatter.unwrap_or_else(|| panic!("{error:?}"));
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }]
        }))
        .unwrap();

        let prompt = formatter.render_prompt(&request).unwrap();

        assert!(prompt.as_str().contains("get_weather"));
        assert!(prompt.as_str().contains("｜DSML｜"));
        assert!(!prompt.as_str().contains("BUNDLED TEMPLATE"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn kimi_k25_preprocesses_tools_for_checkpoint_jinja() {
        let directory =
            std::env::temp_dir().join(format!("sglang-renderer-kimi-k25-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("config.json"),
            r#"{"model_type":"kimi_k25","architectures":["KimiK25ForConditionalGeneration"]}"#,
        )
        .unwrap();
        std::fs::write(
            directory.join("tokenizer_config.json"),
            serde_json::json!({
                "chat_template": "{% if tools_ts_str is defined %}{{ tools_ts_str }}{% else %}JSON {{ tools|tojson }}{% endif %}"
            })
            .to_string(),
        )
        .unwrap();
        let (formatter, error) =
            load_chat_support(&model_config(directory.to_string_lossy().into_owned()));
        let formatter = formatter.unwrap_or_else(|| panic!("{error:?}"));
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }
            }]
        }))
        .unwrap();

        let prompt = formatter.render_prompt(&request).unwrap();

        assert!(prompt.as_str().contains("namespace functions"));
        assert!(prompt.as_str().contains("type get_weather"));
        assert!(!prompt.as_str().starts_with("JSON "));

        let unsupported: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "unsupported",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {"oneOf": [{"type": "string"}]}
                        }
                    }
                }
            }]
        }))
        .unwrap();
        let fallback = formatter.render_prompt(&unsupported).unwrap();
        assert!(fallback.as_str().starts_with("JSON "));
        assert!(fallback.as_str().contains("unsupported"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn deepseek_v4_profile_resolution_uses_override_then_checkpoint_source() {
        let directory = std::env::temp_dir().join(format!(
            "sglang-renderer-deepseek-v4-profile-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(directory.join("encoding")).unwrap();
        std::fs::write(
            directory.join("encoding/encoding_dsv4.py"),
            "DEFAULT_REASONING_EFFORT: str = 'low'\n\
REASONING_EFFORT_PROMPTS = {'low': '', 'high': 'absolute', 'max': 'beyond'}",
        )
        .unwrap();
        let source = directory.to_string_lossy();
        assert_eq!(
            resolve_dsv4_profile(&ModelIdentity::default(), &source, None).unwrap(),
            DeepSeekV4Profile::Official
        );
        let preview = ModelIdentity {
            dsv4_reasoning_effort_profile: Some("preview".into()),
            ..Default::default()
        };
        assert_eq!(
            resolve_dsv4_profile(&preview, &source, None).unwrap(),
            DeepSeekV4Profile::Preview
        );
        let invalid = ModelIdentity {
            dsv4_reasoning_effort_profile: Some("future".into()),
            ..Default::default()
        };
        assert!(resolve_dsv4_profile(&invalid, &source, None).is_err());

        std::fs::write(
            directory.join("encoding/encoding_dsv4.py"),
            r#"DEFAULT_REASONING_EFFORT = "high"
REASONING_EFFORT_PROMPTS = {"low": "", "high": "absolute", "max": "Beyond maximum"}"#,
        )
        .unwrap();
        assert_eq!(
            resolve_dsv4_profile(&ModelIdentity::default(), &source, None).unwrap(),
            DeepSeekV4Profile::Preview
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn chat_template_argument_precedence_is_request_then_top_level_then_defaults() {
        let directory = std::env::temp_dir().join(format!(
            "sglang-renderer-template-defaults-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("tokenizer_config.json"), "{}").unwrap();
        let mut config = model_config(directory.to_string_lossy().into_owned());
        let template = directory.join("arguments.jinja");
        std::fs::write(&template, "{{ marker }}|{{ reasoning_effort }}").unwrap();
        config.chat_template = Some(template.to_string_lossy().into_owned());
        config.default_chat_template_kwargs = std::collections::HashMap::from([
            ("marker".into(), serde_json::json!("default")),
            ("reasoning_effort".into(), serde_json::json!("low")),
        ]);
        let messages = serde_json::from_value(serde_json::json!([
            {"role": "user", "content": "hello"}
        ]))
        .unwrap();
        let mut request = ChatRequest {
            rid: "chatcmpl-test".into(),
            model: "model".into(),
            messages,
            tools: None,
            tool_choice: None,
            response_format: None,
            reasoning_effort: Some(serde_json::from_value(serde_json::json!("max")).unwrap()),
            continue_final_message: false,
            chat_template_args: Some(std::collections::HashMap::from([(
                "marker".into(),
                serde_json::json!("request"),
            )])),
            sampling_params: SamplingParams::default(),
            choice_count: 1,
            stream: false,
            return_logprob: false,
            top_logprobs_num: 0,
            parallel_tool_calls: true,
            metadata: GenerateRequestMetadata::default(),
        };
        let service = RendererService::with_tokenizer(config, Arc::new(UnexpectedTokenizer), 1, 1);

        let chat = service.preprocess_chat(request.clone()).unwrap();
        assert_eq!(chat.text_requests[0].prompt.as_str(), "request|max");

        request
            .chat_template_args
            .as_mut()
            .unwrap()
            .insert("reasoning_effort".into(), serde_json::json!("medium"));
        let chat = service.preprocess_chat(request).unwrap();
        assert_eq!(chat.text_requests[0].prompt.as_str(), "request|medium");
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn token_id_completion_bypasses_tokenization_and_builds_generate_request() {
        let config = RendererConfig {
            served_model_name: "model".into(),
            tokenizer_path: String::new(),
            revision: None,
            model_path: String::new(),
            chat_template: None,
            tool_call_parser: None,
            reasoning_parser: None,
            default_chat_template_kwargs: Default::default(),
            stream_response_default_include_usage: false,
            default_sampling_params: SamplingDefaults::default(),
            limits: RendererLimits {
                vocab_size: 128,
                context_len: 5,
                num_reserved_tokens: 0,
                allow_auto_truncate: true,
                enable_return_hidden_states: false,
            },
        };
        let service = RendererService::with_tokenizer(config, Arc::new(UnexpectedTokenizer), 1, 1);
        let request = TokenIdsRequest::new(
            "cmpl-test-0",
            vec![11, 12, 13],
            GenerationOptions {
                sampling_params: SamplingParams {
                    max_new_tokens: Some(4),
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        let requests = service.prepare_token_ids_requests(vec![request]).unwrap();

        assert_eq!(requests[0].input_ids, vec![11, 12, 13]);
        assert_eq!(requests[0].sampling_params.max_new_tokens, Some(2));
    }
}
