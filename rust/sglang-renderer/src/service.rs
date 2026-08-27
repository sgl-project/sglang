//! Reusable SGLang request preprocessing.

use std::sync::Arc;

use dynamo_protocols::types::{CreateChatCompletionRequest, CreateCompletionRequest, Prompt};
use dynamo_renderer::{kimi_k3_formatter_for, native_formatter_for};
use futures::future::{BoxFuture, try_join_all};

#[cfg(any(feature = "http", test))]
use crate::GenerateRequestMetadata;
#[cfg(any(feature = "http", test))]
use crate::protocol::openai::lower_chat_request_with_template_args;
use crate::protocol::openai::{
    lower_chat_request, lower_text_completion_request, lower_token_ids_completion_request,
};
use crate::template::load_chat_formatter;
use crate::tokenizer::{
    check_total_tokens, resolve_chat_template_file, resolve_model_file, validate_text_request,
    validate_token_ids_request,
};
use crate::{
    ChatFormatter, ChatPreprocessor, ChatRequest, GenerateRequest, LoweredChat, RendererConfig,
    RendererError, TextRequest, TokenIdsRequest,
};

/// Host-provided tokenizer-dependent CPU execution for one model-facing text
/// request. Validation, sampling normalization, and context checks remain
/// owned by `RendererService`.
pub trait TokenizationBackend: Send + Sync {
    fn tokenize(
        &self,
        request: TextRequest,
    ) -> BoxFuture<'static, Result<TokenIdsRequest, RendererError>>;
}

/// Internal OpenAI wire lowering before model-specific preprocessing.
struct OpenAIRequestLowerer {
    config: RendererConfig,
}

/// Shared preprocessing used by inference and render-only frontends.
pub struct RendererService {
    lowerer: OpenAIRequestLowerer,
    chat_preprocessor: ChatPreprocessor,
    backend: Arc<dyn TokenizationBackend>,
}

#[cfg(any(feature = "http", test))]
pub(crate) struct ChatLoweringOptions {
    pub chat_template_args: Option<std::collections::HashMap<String, serde_json::Value>>,
    pub continue_final_message: bool,
    pub sampling_overrides: crate::SamplingParamsOverrides,
    pub metadata: GenerateRequestMetadata,
}

#[cfg(any(feature = "http", test))]
pub(crate) struct ChatGenerateRequests {
    pub requests: Vec<GenerateRequest>,
    pub response_processor: crate::ChatResponseProcessor,
}

impl OpenAIRequestLowerer {
    fn new(config: RendererConfig) -> Self {
        Self { config }
    }

    fn config(&self) -> &RendererConfig {
        &self.config
    }

    fn lower_chat(
        &self,
        request: CreateChatCompletionRequest,
        response_id: &str,
    ) -> Result<ChatRequest, RendererError> {
        lower_chat_request(&self.config, request, response_id)
    }

    #[cfg(any(feature = "http", test))]
    pub(crate) fn lower_chat_with_template_args(
        &self,
        request: CreateChatCompletionRequest,
        response_id: &str,
        chat_template_args: Option<std::collections::HashMap<String, serde_json::Value>>,
        continue_final_message: bool,
    ) -> Result<ChatRequest, RendererError> {
        lower_chat_request_with_template_args(
            &self.config,
            request,
            response_id,
            chat_template_args,
            continue_final_message,
        )
    }

    fn lower_text_completions(
        &self,
        request: CreateCompletionRequest,
        response_id: &str,
    ) -> Result<Vec<TextRequest>, RendererError> {
        lower_text_completion_request(&self.config, &request, response_id)
    }

    fn lower_token_ids_completions(
        &self,
        request: CreateCompletionRequest,
        response_id: &str,
    ) -> Result<Vec<TokenIdsRequest>, RendererError> {
        lower_token_ids_completion_request(&self.config, &request, response_id)
    }
}

impl RendererService {
    pub fn new(config: RendererConfig, backend: Arc<dyn TokenizationBackend>) -> Self {
        let lowerer = OpenAIRequestLowerer::new(config);
        let (formatter, formatter_error) = load_chat_support(lowerer.config());
        let chat_preprocessor = ChatPreprocessor::new(lowerer.config(), formatter)
            .with_formatter_error(formatter_error);
        Self {
            lowerer,
            chat_preprocessor,
            backend,
        }
    }

    pub async fn prepare_chat(
        &self,
        request: CreateChatCompletionRequest,
        response_id: &str,
    ) -> Result<Vec<GenerateRequest>, RendererError> {
        let chat = self.lowerer.lower_chat(request, response_id)?;
        let lowered = self.chat_preprocessor.preprocess(chat)?;
        self.prepare_text_many(lowered.text_requests).await
    }

    #[cfg(any(feature = "http", test))]
    pub(crate) fn lower_chat_with_options(
        &self,
        request: CreateChatCompletionRequest,
        response_id: &str,
        options: ChatLoweringOptions,
    ) -> Result<LoweredChat, RendererError> {
        let mut chat = self.lowerer.lower_chat_with_template_args(
            request,
            response_id,
            options.chat_template_args,
            options.continue_final_message,
        )?;
        options.sampling_overrides.apply(&mut chat.sampling_params);
        chat.metadata = options.metadata;
        self.chat_preprocessor.preprocess(chat)
    }

    #[cfg(any(feature = "http", test))]
    pub(crate) async fn prepare_chat_with_template_args(
        &self,
        request: CreateChatCompletionRequest,
        response_id: &str,
        options: ChatLoweringOptions,
    ) -> Result<ChatGenerateRequests, RendererError> {
        let lowered = self.lower_chat_with_options(request, response_id, options)?;
        Ok(ChatGenerateRequests {
            requests: self.prepare_text_many(lowered.text_requests).await?,
            response_processor: lowered.response_processor,
        })
    }

    pub fn config(&self) -> &RendererConfig {
        self.lowerer.config()
    }

    #[cfg(any(feature = "http", test))]
    pub(crate) fn lower_text_completions_with_metadata(
        &self,
        request: CreateCompletionRequest,
        response_id: &str,
        sampling_overrides: crate::SamplingParamsOverrides,
        metadata: GenerateRequestMetadata,
    ) -> Result<Vec<TextRequest>, RendererError> {
        let mut requests = self.lowerer.lower_text_completions(request, response_id)?;
        for request in &mut requests {
            sampling_overrides
                .clone()
                .apply(&mut request.options.sampling_params);
            request.metadata = metadata.clone();
        }
        Ok(requests)
    }

    #[cfg(any(feature = "http", test))]
    pub(crate) fn lower_token_ids_completions_with_metadata(
        &self,
        request: CreateCompletionRequest,
        response_id: &str,
        sampling_overrides: crate::SamplingParamsOverrides,
        metadata: GenerateRequestMetadata,
    ) -> Result<Vec<TokenIdsRequest>, RendererError> {
        let mut requests = self
            .lowerer
            .lower_token_ids_completions(request, response_id)?;
        for request in &mut requests {
            sampling_overrides
                .clone()
                .apply(&mut request.options.sampling_params);
            request.metadata = metadata.clone();
        }
        Ok(requests)
    }

    pub async fn prepare_text_request(
        &self,
        request: TextRequest,
    ) -> Result<GenerateRequest, RendererError> {
        self.prepare_text(request).await.map(Into::into)
    }

    pub async fn prepare_token_ids_request(
        &self,
        request: TokenIdsRequest,
    ) -> Result<GenerateRequest, RendererError> {
        self.prepare_token_ids(request).map(Into::into)
    }

    pub async fn tokenize_prompt(
        &self,
        text: String,
        add_special_tokens: bool,
    ) -> Result<crate::TokenIds, RendererError> {
        let request = TextRequest::text("tokenize", text, add_special_tokens, Default::default());
        Ok(self.backend.tokenize(request).await?.input_ids)
    }

    pub(crate) async fn tokenize_chat(
        &self,
        request: ChatRequest,
    ) -> Result<crate::TokenIds, RendererError> {
        let request = self.chat_preprocessor.lower_to_text(request)?;
        Ok(self.backend.tokenize(request).await?.input_ids)
    }

    pub async fn prepare_completions(
        &self,
        request: CreateCompletionRequest,
        response_id: &str,
    ) -> Result<Vec<GenerateRequest>, RendererError> {
        match &request.prompt {
            Prompt::String(_) | Prompt::StringArray(_) => {
                let requests = self.lowerer.lower_text_completions(request, response_id)?;
                self.prepare_text_many(requests).await
            }
            Prompt::IntegerArray(_) | Prompt::ArrayOfIntegerArray(_) => {
                let requests = self
                    .lowerer
                    .lower_token_ids_completions(request, response_id)?;
                self.prepare_token_ids_many(requests)
            }
        }
    }

    #[cfg(any(feature = "http", test))]
    pub(crate) async fn prepare_text_completions_with_metadata(
        &self,
        request: CreateCompletionRequest,
        response_id: &str,
        sampling_overrides: crate::SamplingParamsOverrides,
        metadata: GenerateRequestMetadata,
    ) -> Result<Vec<GenerateRequest>, RendererError> {
        let requests = self.lower_text_completions_with_metadata(
            request,
            response_id,
            sampling_overrides,
            metadata,
        )?;
        self.prepare_text_many(requests).await
    }

    #[cfg(any(feature = "http", test))]
    pub(crate) async fn prepare_token_ids_completions_with_metadata(
        &self,
        request: CreateCompletionRequest,
        response_id: &str,
        sampling_overrides: crate::SamplingParamsOverrides,
        metadata: GenerateRequestMetadata,
    ) -> Result<Vec<GenerateRequest>, RendererError> {
        let requests = self.lower_token_ids_completions_with_metadata(
            request,
            response_id,
            sampling_overrides,
            metadata,
        )?;
        self.prepare_token_ids_many(requests)
    }

    async fn prepare_text_many(
        &self,
        requests: Vec<TextRequest>,
    ) -> Result<Vec<GenerateRequest>, RendererError> {
        Ok(self
            .prepare_text_inputs(requests)
            .await?
            .into_iter()
            .map(GenerateRequest::from)
            .collect())
    }

    async fn prepare_text_inputs(
        &self,
        requests: Vec<TextRequest>,
    ) -> Result<Vec<TokenIdsRequest>, RendererError> {
        try_join_all(
            requests
                .into_iter()
                .map(|request| self.prepare_text(request)),
        )
        .await
    }

    fn prepare_token_ids_many(
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
        validate_text_request(&request, &self.lowerer.config.limits)?;
        request.options.sampling_params.normalize(
            self.lowerer.config.skip_tokenizer_init,
            self.lowerer.config.vocab_size,
        )?;
        let mut request = self.backend.tokenize(request).await?;
        check_total_tokens(&mut request, &self.lowerer.config.limits)?;
        Ok(request)
    }

    fn prepare_token_ids(
        &self,
        mut request: TokenIdsRequest,
    ) -> Result<TokenIdsRequest, RendererError> {
        validate_token_ids_request(&request, &self.lowerer.config.limits)?;
        request.options.sampling_params.normalize(
            self.lowerer.config.skip_tokenizer_init,
            self.lowerer.config.vocab_size,
        )?;
        check_total_tokens(&mut request, &self.lowerer.config.limits)?;
        Ok(request)
    }
}

fn load_chat_support(config: &RendererConfig) -> (Option<ChatFormatter>, Option<String>) {
    if config.skip_tokenizer_init || config.tokenizer_path.is_empty() {
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
    let model_type_lower = model_config_file
        .as_deref()
        .and_then(load_model_type)
        .map(|model_type| model_type.to_ascii_lowercase());
    let display_name_lower = model_source.to_ascii_lowercase();
    if let Some(formatter) = kimi_k3_formatter_for(&model_type_lower, &display_name_lower, true)
        .or_else(|| native_formatter_for(&model_type_lower, &display_name_lower))
    {
        return (Some(ChatFormatter::HuggingFace(formatter)), None);
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
        Ok(formatter) => {
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

fn load_model_type(config_file: &str) -> Option<String> {
    let config = std::fs::read_to_string(config_file).ok()?;
    serde_json::from_str::<serde_json::Value>(&config)
        .ok()?
        .get("model_type")?
        .as_str()
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OneOrMany, RendererLimits, SamplingDefaults};

    fn model_config(model_path: String) -> RendererConfig {
        RendererConfig {
            served_model_name: "model".into(),
            tokenizer_path: model_path.clone(),
            revision: None,
            model_path,
            chat_template: None,
            tool_call_parser: None,
            reasoning_parser: None,
            stream_response_default_include_usage: false,
            skip_tokenizer_init: false,
            vocab_size: 128,
            default_sampling_params: SamplingDefaults::default(),
            limits: RendererLimits {
                skip_tokenizer_init: false,
                vocab_size: 128,
                context_len: 128,
                num_reserved_tokens: 0,
                allow_auto_truncate: false,
                enable_return_hidden_states: false,
            },
        }
    }

    struct UnexpectedTokenizer;

    impl TokenizationBackend for UnexpectedTokenizer {
        fn tokenize(
            &self,
            _request: TextRequest,
        ) -> BoxFuture<'static, Result<TokenIdsRequest, RendererError>> {
            Box::pin(async { panic!("token-ID input must not enter the tokenizer backend") })
        }
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
            stream_response_default_include_usage: false,
            skip_tokenizer_init: false,
            vocab_size: 128,
            default_sampling_params: SamplingDefaults::default(),
            limits: RendererLimits {
                skip_tokenizer_init: false,
                vocab_size: 128,
                context_len: 128,
                num_reserved_tokens: 0,
                allow_auto_truncate: false,
                enable_return_hidden_states: false,
            },
        };
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "stop": "client-stop"
        }))
        .unwrap();

        let service = RendererService::new(config, Arc::new(UnexpectedTokenizer));
        let chat = service
            .lower_chat_with_options(
                request,
                "chatcmpl-test",
                ChatLoweringOptions {
                    chat_template_args: Some(std::collections::HashMap::from([(
                        "enable_thinking".to_owned(),
                        serde_json::Value::Bool(false),
                    )])),
                    continue_final_message: false,
                    sampling_overrides: crate::SamplingParamsOverrides::default(),
                    metadata: GenerateRequestMetadata::default(),
                },
            )
            .unwrap();
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
    fn token_id_completion_bypasses_tokenization_and_builds_generate_request() {
        let config = RendererConfig {
            served_model_name: "model".into(),
            tokenizer_path: String::new(),
            revision: None,
            model_path: String::new(),
            chat_template: None,
            tool_call_parser: None,
            reasoning_parser: None,
            stream_response_default_include_usage: false,
            skip_tokenizer_init: false,
            vocab_size: 128,
            default_sampling_params: SamplingDefaults::default(),
            limits: RendererLimits {
                skip_tokenizer_init: false,
                vocab_size: 128,
                context_len: 5,
                num_reserved_tokens: 0,
                allow_auto_truncate: true,
                enable_return_hidden_states: false,
            },
        };
        let service = RendererService::new(config, Arc::new(UnexpectedTokenizer));
        let request: CreateCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "prompt": [11, 12, 13],
            "max_tokens": 4
        }))
        .unwrap();

        let requests =
            futures::executor::block_on(service.prepare_completions(request, "cmpl-test")).unwrap();

        assert_eq!(requests[0].input_ids, vec![11, 12, 13]);
        assert_eq!(requests[0].sampling_params.max_new_tokens, Some(2));
    }
}
