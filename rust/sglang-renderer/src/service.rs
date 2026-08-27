//! Transport-independent renderer orchestration.

use std::sync::Arc;

use dynamo_parsers::ToolDefinition;
use dynamo_protocols::types::{
    ChatCompletionToolChoiceOption, CreateChatCompletionRequest, CreateCompletionRequest,
    ServiceTier as ChatServiceTier,
};
use futures::future::{BoxFuture, try_join_all};

use crate::openai::{lower_chat_requests, lower_completion_requests};
use crate::template::load_chat_formatter;
use crate::tokenizer::{check_total_tokens, resolve_model_file, validate_request};
use crate::{ChatFormatter, RendererConfig, RendererError, RendererRequest};

/// Host-provided CPU execution for one lowered renderer request.
pub trait PreprocessBackend: Send + Sync {
    fn prepare(
        &self,
        request: RendererRequest,
    ) -> BoxFuture<'static, Result<RendererRequest, RendererError>>;
}

pub struct RendererService {
    config: RendererConfig,
    chat_formatter: Option<ChatFormatter>,
    backend: Arc<dyn PreprocessBackend>,
}

pub struct ChatResponsePlan {
    pub response_id: String,
    pub model: String,
    pub stream: bool,
    pub choice_count: usize,
    pub want_logprobs: bool,
    pub include_usage: bool,
    pub parser: Option<String>,
    pub reasoning_parser: Option<String>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub stream_tool_choice: Option<ChatCompletionToolChoiceOption>,
    pub uses_tool_call_structural_tag: bool,
    pub parallel_tool_calls: bool,
    pub service_tier: Option<ChatServiceTier>,
}

/// One lowered OpenAI chat request and the response context retained by the
/// frontend. `requests` are tokenized only when returned by `prepare_chat`.
pub struct ChatRequestBatch {
    pub requests: Vec<RendererRequest>,
    pub response: ChatResponsePlan,
}

impl RendererService {
    pub fn new(config: RendererConfig, backend: Arc<dyn PreprocessBackend>) -> Self {
        let chat_formatter = load_chat_support(&config);
        Self {
            config,
            chat_formatter,
            backend,
        }
    }

    pub fn config(&self) -> &RendererConfig {
        &self.config
    }

    pub async fn lower_chat(
        &self,
        request: &mut CreateChatCompletionRequest,
        response_id: &str,
    ) -> Result<ChatRequestBatch, RendererError> {
        let lowered = lower_chat_requests(
            &self.config,
            self.chat_formatter.clone(),
            request,
            response_id,
        )
        .await?;
        let response = ChatResponsePlan {
            response_id: response_id.to_owned(),
            model: request.model.clone(),
            stream: request.stream.unwrap_or(false),
            choice_count: request.n.unwrap_or(1) as usize,
            want_logprobs: request.logprobs.unwrap_or(false),
            include_usage: request
                .stream_options
                .as_ref()
                .is_some_and(|options| options.include_usage)
                || self.config.stream_response_default_include_usage,
            parser: lowered.parser,
            reasoning_parser: self.config.reasoning_parser.clone(),
            tools: lowered.tools,
            stream_tool_choice: request.tool_choice.clone(),
            uses_tool_call_structural_tag: lowered
                .requests
                .first()
                .is_some_and(|request| request.sampling_params.structural_tag.is_some()),
            parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
            service_tier: request.service_tier.clone(),
        };
        Ok(ChatRequestBatch {
            requests: lowered.requests,
            response,
        })
    }

    pub async fn prepare_chat(
        &self,
        request: &mut CreateChatCompletionRequest,
        response_id: &str,
    ) -> Result<ChatRequestBatch, RendererError> {
        let mut batch = self.lower_chat(request, response_id).await?;
        batch.requests = self.prepare_many(batch.requests).await?;
        Ok(batch)
    }

    pub fn lower_completions(
        &self,
        request: &CreateCompletionRequest,
        response_id: &str,
    ) -> Result<Vec<RendererRequest>, RendererError> {
        lower_completion_requests(&self.config, request, response_id)
    }

    pub async fn prepare_completions(
        &self,
        request: &CreateCompletionRequest,
        response_id: &str,
    ) -> Result<Vec<RendererRequest>, RendererError> {
        let requests = self.lower_completions(request, response_id)?;
        self.prepare_many(requests).await
    }

    async fn prepare_many(
        &self,
        requests: Vec<RendererRequest>,
    ) -> Result<Vec<RendererRequest>, RendererError> {
        try_join_all(
            requests
                .into_iter()
                .map(|request| self.prepare_one(request)),
        )
        .await
    }

    async fn prepare_one(
        &self,
        mut request: RendererRequest,
    ) -> Result<RendererRequest, RendererError> {
        if self.config.skip_tokenizer_init {
            validate_request(&request, &self.config.limits)?;
            request
                .sampling_params
                .normalize(true, self.config.vocab_size)?;
            check_total_tokens(&mut request, &self.config.limits)?;
            return Ok(request);
        }
        self.backend.prepare(request).await
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
