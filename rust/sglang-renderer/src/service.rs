//! Transport-independent renderer orchestration.

use std::sync::Arc;

use dynamo_protocols::types::{CreateChatCompletionRequest, CreateCompletionRequest};
use futures::future::{BoxFuture, try_join_all};

use crate::openai::{
    LoweredChat, lower_chat_request, lower_completion_request, render_chat_prompt,
};
use crate::template::load_chat_formatter;
use crate::tokenizer::{check_total_tokens, resolve_model_file, validate_generation_input};
use crate::{
    ChatFormatter, GenerationInput, PreparedGenerateRequest, RendererConfig, RendererError,
    TextRequest, TokenIdsRequest,
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

/// Engine-free OpenAI protocol lowering shared by inference and rendering.
///
/// This type deliberately has no preprocessing backend. Full inference
/// lowers protocol semantics here, then submits text or token-ID inputs to its
/// existing request FSM.
pub struct OpenAIRequestLowerer {
    config: RendererConfig,
    chat_formatter: Option<ChatFormatter>,
}

/// Standalone request preparation: shared OpenAI processing plus host-provided
/// CPU work.
///
/// This service returns prepared generation requests. Chat response processors
/// created during lowering are discarded by this preparation-only path.
pub struct RendererService {
    lowerer: OpenAIRequestLowerer,
    backend: Arc<dyn TokenizationBackend>,
}

impl OpenAIRequestLowerer {
    pub fn new(config: RendererConfig) -> Self {
        let chat_formatter = load_chat_support(&config);
        Self {
            config,
            chat_formatter,
        }
    }

    pub fn config(&self) -> &RendererConfig {
        &self.config
    }

    pub async fn lower_chat(
        &self,
        mut request: CreateChatCompletionRequest,
        response_id: &str,
    ) -> Result<LoweredChat, RendererError> {
        lower_chat_request(
            &self.config,
            self.chat_formatter.clone(),
            &mut request,
            response_id,
        )
        .await
    }

    pub fn lower_completions(
        &self,
        request: CreateCompletionRequest,
        response_id: &str,
    ) -> Result<Vec<GenerationInput>, RendererError> {
        lower_completion_request(&self.config, &request, response_id)
    }

    async fn render_chat_prompt(
        &self,
        mut request: CreateChatCompletionRequest,
    ) -> Result<String, RendererError> {
        render_chat_prompt(&self.config, self.chat_formatter.clone(), &mut request).await
    }
}

impl RendererService {
    pub fn new(lowerer: OpenAIRequestLowerer, backend: Arc<dyn TokenizationBackend>) -> Self {
        Self { lowerer, backend }
    }

    pub async fn prepare_chat(
        &self,
        request: CreateChatCompletionRequest,
        response_id: &str,
    ) -> Result<Vec<PreparedGenerateRequest>, RendererError> {
        let lowered = self.lowerer.lower_chat(request, response_id).await?;
        self.prepare_many(lowered.generation_inputs).await
    }

    pub fn config(&self) -> &RendererConfig {
        self.lowerer.config()
    }

    pub async fn lower_chat(
        &self,
        request: CreateChatCompletionRequest,
        response_id: &str,
    ) -> Result<LoweredChat, RendererError> {
        self.lowerer.lower_chat(request, response_id).await
    }

    pub fn lower_completions(
        &self,
        request: CreateCompletionRequest,
        response_id: &str,
    ) -> Result<Vec<GenerationInput>, RendererError> {
        self.lowerer.lower_completions(request, response_id)
    }

    pub async fn prepare_generation_input(
        &self,
        request: GenerationInput,
    ) -> Result<TokenIdsRequest, RendererError> {
        self.prepare_one(request).await
    }

    pub async fn tokenize_prompt(
        &self,
        text: String,
        add_special_tokens: bool,
    ) -> Result<crate::TokenIds, RendererError> {
        let request = TextRequest {
            rid: "tokenize".into(),
            text,
            skip_special_tokens: !add_special_tokens,
            options: Default::default(),
        };
        Ok(self.backend.tokenize(request).await?.input_ids)
    }

    pub async fn tokenize_chat(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<crate::TokenIds, RendererError> {
        let request = TextRequest {
            rid: "tokenize".into(),
            text: self.lowerer.render_chat_prompt(request).await?,
            skip_special_tokens: true,
            options: Default::default(),
        };
        Ok(self.backend.tokenize(request).await?.input_ids)
    }

    pub async fn prepare_completions(
        &self,
        request: CreateCompletionRequest,
        response_id: &str,
    ) -> Result<Vec<PreparedGenerateRequest>, RendererError> {
        let requests = self.lowerer.lower_completions(request, response_id)?;
        self.prepare_many(requests).await
    }

    async fn prepare_many(
        &self,
        requests: Vec<GenerationInput>,
    ) -> Result<Vec<PreparedGenerateRequest>, RendererError> {
        let requests = try_join_all(
            requests
                .into_iter()
                .map(|request| self.prepare_one(request)),
        )
        .await?;
        Ok(requests
            .into_iter()
            .map(PreparedGenerateRequest::from)
            .collect())
    }

    async fn prepare_one(
        &self,
        mut request: GenerationInput,
    ) -> Result<TokenIdsRequest, RendererError> {
        validate_generation_input(&request, &self.lowerer.config.limits)?;
        request.options_mut().sampling_params.normalize(
            self.lowerer.config.skip_tokenizer_init,
            self.lowerer.config.vocab_size,
        )?;
        let mut request = match request {
            GenerationInput::Text(request) => self.backend.tokenize(request).await?,
            GenerationInput::TokenIds(request) => request,
        };
        check_total_tokens(&mut request, &self.lowerer.config.limits)?;
        Ok(request)
    }
}

fn load_chat_support(config: &RendererConfig) -> Option<ChatFormatter> {
    if config.skip_tokenizer_init || config.tokenizer_path.is_empty() {
        return None;
    }
    let config_file = resolve_model_file(
        &config.tokenizer_path,
        config.revision.as_deref(),
        "tokenizer_config.json",
    );
    match load_chat_formatter(
        config_file.as_deref(),
        (!config.model_path.is_empty()).then_some(config.model_path.as_str()),
        config.chat_template.as_deref(),
    ) {
        Ok(formatter) => {
            tracing::info!(
                config = ?config_file.as_deref().unwrap_or("<built-in / inferred>"),
                "loaded OpenAI chat template"
            );
            Some(formatter)
        }
        Err(error) => {
            tracing::warn!(%error, "OpenAI chat completions disabled");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OneOrMany, RendererLimits, SamplingDefaults};

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
        let lowerer = OpenAIRequestLowerer::new(RendererConfig {
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
        });
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "stop": "client-stop"
        }))
        .unwrap();

        let lowered =
            futures::executor::block_on(lowerer.lower_chat(request, "chatcmpl-test")).unwrap();
        let GenerationInput::Text(text_request) = &lowered.generation_inputs[0] else {
            panic!("Chat lowering must produce text input");
        };

        assert!(text_request.text.contains("<|im_start|>user"));
        assert!(matches!(
            text_request.options.sampling_params.stop.as_ref(),
            Some(OneOrMany::Many(stops))
                if stops.iter().map(String::as_str).collect::<Vec<_>>()
                    == ["<|endoftext|>", "<|im_end|>", "client-stop"]
        ));
    }

    #[test]
    fn token_id_completion_bypasses_tokenization_and_is_still_prepared() {
        let lowerer = OpenAIRequestLowerer::new(RendererConfig {
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
        });
        let service = RendererService::new(lowerer, Arc::new(UnexpectedTokenizer));
        let request: CreateCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "prompt": [11, 12, 13],
            "max_tokens": 4
        }))
        .unwrap();

        let prepared =
            futures::executor::block_on(service.prepare_completions(request, "cmpl-test")).unwrap();

        assert_eq!(prepared[0].input_ids, vec![11, 12, 13]);
        assert_eq!(prepared[0].sampling_params.max_new_tokens, Some(2));
    }
}
