//! Reusable SGLang request preprocessing.

use std::sync::Arc;

use futures::future::try_join_all;
use sglang_processor::ChatFormatterOptions;

use super::tokenizer::{
    PooledTokenizer, TextTokenizer, check_total_tokens, validate_request_id, validate_text_request,
    validate_token_ids_request,
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

    pub async fn detokenize(
        &self,
        token_ids: crate::TokenIds,
        skip_special_tokens: bool,
    ) -> Result<String, RendererError> {
        let vocab_size = self.config.limits.vocab_size;
        let token_ids = token_ids
            .into_iter()
            .map(|id| {
                u32::try_from(id)
                    .ok()
                    .filter(|&id| u64::from(id) < vocab_size)
                    .ok_or_else(|| {
                        RendererError::Validation(format!(
                            "tokens contains out-of-vocabulary token id {id}; valid range is [0, {vocab_size})"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.tokenizer
            .detokenize(token_ids, skip_special_tokens)
            .await
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
    sglang_processor::select_chat_formatter(&ChatFormatterOptions {
        tokenizer_path: config.tokenizer_path.clone(),
        model_path: config.model_path.clone(),
        revision: config.revision.clone(),
        chat_template: config.chat_template.clone(),
    })
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
        ) -> Result<crate::TokenIds, sglang_processor::ProcessorError> {
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
        ) -> Result<crate::TokenIds, sglang_processor::ProcessorError> {
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
        };
        for (profile, max_prefix, high_prefix) in [
            ("preview", "Absolute maximum", None),
            ("official", "Beyond maximum", Some("Absolute maximum")),
        ] {
            std::fs::write(
                directory.join("config.json"),
                serde_json::json!({
                    "model_type": "deepseek_v4",
                    "dsv4_reasoning_effort_profile": profile,
                })
                .to_string(),
            )
            .unwrap();
            let service = RendererService::with_tokenizer(
                config.clone(),
                Arc::new(UnexpectedTokenizer),
                1,
                1,
            );
            for (effort, args, thinking, prefix) in [
                (None, serde_json::json!({}), false, None),
                (Some("max"), serde_json::json!({}), true, Some(max_prefix)),
                (Some("high"), serde_json::json!({}), true, high_prefix),
                (Some("none"), serde_json::json!({}), false, None),
                (
                    Some("max"),
                    serde_json::json!({"thinking": false}),
                    false,
                    None,
                ),
                (
                    Some("none"),
                    serde_json::json!({"thinking": true}),
                    true,
                    None,
                ),
                (
                    Some("max"),
                    serde_json::json!({"reasoning_effort": "low"}),
                    true,
                    None,
                ),
            ] {
                let mut request = request.clone();
                request.reasoning_effort =
                    effort.map(|effort| serde_json::from_value(serde_json::json!(effort)).unwrap());
                request.chat_template_args = Some(serde_json::from_value(args).unwrap());
                let chat = service.preprocess_chat(request).unwrap();
                let text_request = &chat.text_requests[0];
                assert_eq!(text_request.options.require_reasoning, thinking);
                let prompt = text_request.prompt.as_str();
                assert!(prompt.ends_with(if thinking { "<think>" } else { "</think>" }));
                assert_eq!(
                    prompt.matches("Reasoning Effort:").count(),
                    usize::from(prefix.is_some()),
                    "{profile}, {effort:?}: {prompt}"
                );
                if let Some(prefix) = prefix {
                    assert!(prompt.starts_with(&format!(
                        "<｜begin▁of▁sentence｜>Reasoning Effort: {prefix}"
                    )));
                }
            }
        }
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
