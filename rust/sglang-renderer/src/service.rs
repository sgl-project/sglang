//! Transport-independent renderer orchestration.

use std::sync::Arc;

use dynamo_protocols::types::{CreateChatCompletionRequest, CreateCompletionRequest};
use futures::future::{BoxFuture, try_join_all};

use crate::openai::{lower_chat_requests, lower_completion_requests};
use crate::template::load_chat_formatter;
use crate::tokenizer::{check_total_tokens, resolve_model_file, validate_request};
use crate::{
    ChatFormatter, ChatResponseProcessor, RendererConfig, RendererError, TextCompletionRequest,
};

/// Host-provided tokenizer-dependent CPU execution for one lowered text
/// completion. Validation, sampling normalization, and context checks remain
/// owned by `RendererService`.
pub trait TokenizationBackend: Send + Sync {
    fn tokenize(
        &self,
        request: TextCompletionRequest,
    ) -> BoxFuture<'static, Result<TextCompletionRequest, RendererError>>;
}

/// Engine-free OpenAI request lowering shared by inference and rendering.
///
/// This type deliberately has no preprocessing backend. Full inference lowers
/// requests here, then submits them to its request FSM for normalization and
/// tokenization.
pub struct RequestLowerer {
    config: RendererConfig,
    chat_formatter: Option<ChatFormatter>,
}

/// Standalone request preparation: shared lowering plus host-provided CPU work.
pub struct RendererService {
    lowerer: RequestLowerer,
    backend: Arc<dyn TokenizationBackend>,
}

/// Lowered text-completion requests plus the processor retained to interpret
/// their eventual engine responses.
///
/// `completion_requests` are tokenized only when returned by `prepare_chat`.
pub struct LoweredChat {
    pub completion_requests: Vec<TextCompletionRequest>,
    pub response_processor: ChatResponseProcessor,
}

impl RequestLowerer {
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
        request: &mut CreateChatCompletionRequest,
        response_id: &str,
    ) -> Result<LoweredChat, RendererError> {
        let lowered = lower_chat_requests(
            &self.config,
            self.chat_formatter.clone(),
            request,
            response_id,
        )
        .await?;
        let uses_tool_call_structural_tag = lowered
            .completion_requests
            .first()
            .is_some_and(|request| request.sampling_params.structural_tag.is_some());
        let response_processor = ChatResponseProcessor::new(
            lowered.parser,
            self.config.reasoning_parser.clone(),
            lowered.tools,
            request.tool_choice.clone(),
            uses_tool_call_structural_tag,
            request.parallel_tool_calls.unwrap_or(true),
            lowered.completion_requests.len(),
        );
        Ok(LoweredChat {
            completion_requests: lowered.completion_requests,
            response_processor,
        })
    }

    pub fn lower_completions(
        &self,
        request: &CreateCompletionRequest,
        response_id: &str,
    ) -> Result<Vec<TextCompletionRequest>, RendererError> {
        lower_completion_requests(&self.config, request, response_id)
    }
}

impl RendererService {
    pub fn new(lowerer: RequestLowerer, backend: Arc<dyn TokenizationBackend>) -> Self {
        Self { lowerer, backend }
    }

    pub async fn prepare_chat(
        &self,
        request: &mut CreateChatCompletionRequest,
        response_id: &str,
    ) -> Result<LoweredChat, RendererError> {
        let mut lowered = self.lowerer.lower_chat(request, response_id).await?;
        lowered.completion_requests = self.prepare_many(lowered.completion_requests).await?;
        Ok(lowered)
    }

    pub async fn prepare_completions(
        &self,
        request: &CreateCompletionRequest,
        response_id: &str,
    ) -> Result<Vec<TextCompletionRequest>, RendererError> {
        let requests = self.lowerer.lower_completions(request, response_id)?;
        self.prepare_many(requests).await
    }

    async fn prepare_many(
        &self,
        requests: Vec<TextCompletionRequest>,
    ) -> Result<Vec<TextCompletionRequest>, RendererError> {
        try_join_all(
            requests
                .into_iter()
                .map(|request| self.prepare_one(request)),
        )
        .await
    }

    async fn prepare_one(
        &self,
        mut request: TextCompletionRequest,
    ) -> Result<TextCompletionRequest, RendererError> {
        validate_request(&request, &self.lowerer.config.limits)?;
        request.sampling_params.normalize(
            self.lowerer.config.skip_tokenizer_init,
            self.lowerer.config.vocab_size,
        )?;
        if !request.already_tokenized() {
            request = self.backend.tokenize(request).await?;
        }
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
