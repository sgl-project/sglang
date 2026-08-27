//! Engine-free request preparation shared by inference and standalone render.

use std::{collections::BTreeMap, sync::Arc};

use dynamo_parsers::ToolDefinition;
use dynamo_protocols::types::{
    ChatCompletionToolChoiceOption, CreateChatCompletionRequest, CreateCompletionRequest,
    ServiceTier as ChatServiceTier,
};
use futures::future::try_join_all;
use serde::Serialize;
use tokio::sync::oneshot;

use crate::api_server::openai::{
    ChatFormatter, OpenAIRequestError, load_chat_support, lower_chat_requests,
    lower_completion_requests,
};
use crate::message::config::{DefaultSamplingParams, ServerArgs};
use crate::message::request::{GenerateRequest, Request};
use crate::message::sampling::SamplingParams;
use crate::tokenizer_manager::to_scheduler::{
    Limits, check_total_tokens, validate_generate_request,
};
use crate::tokenizer_manager::tokenizer::{TextTokenizer, tokenize_generate_request};
use crate::utils::error::Error;

/// One item on the tokenizer/preprocessing worker pool.
///
/// Normal inference keeps its request FSM and returns through `TmEvent`; direct
/// render calls return the prepared native request through a one-shot channel.
pub(crate) enum PreprocessJob {
    Inference(Request),
    Render(Box<RenderJob>),
}

pub(crate) struct RenderJob {
    pub(crate) request: GenerateRequest,
    pub(crate) reply: oneshot::Sender<Result<GenerateRequest, Error>>,
}

/// Transport-independent preprocessing failures. HTTP and a future gRPC
/// adapter map these at their own boundary.
#[derive(Debug, thiserror::Error)]
pub(crate) enum RenderServiceError {
    #[error("{0}")]
    Request(#[from] OpenAIRequestError),
    #[error("{0}")]
    Prepare(#[from] Error),
    #[error("renderer is shutting down")]
    Unavailable,
    #[error("render preprocessing worker failed")]
    WorkerDropped,
}

impl RenderServiceError {
    pub(crate) fn http_status(&self) -> u16 {
        match self {
            Self::Request(_) => 400,
            Self::Prepare(error) => error.http_status(),
            Self::Unavailable => 503,
            Self::WorkerDropped => 500,
        }
    }
}

/// Immutable subset of server configuration read during request preparation.
///
/// Both CLI entrypoints derive this from the same typed `ServerArgs`, while
/// render state stays independent of scheduler, transport and GPU settings.
#[derive(Clone)]
pub(crate) struct RendererConfig {
    pub(crate) served_model_name: String,
    pub(crate) tokenizer_path: String,
    pub(crate) revision: Option<String>,
    pub(crate) model_path: String,
    pub(crate) chat_template: Option<String>,
    pub(crate) tool_call_parser: Option<String>,
    pub(crate) reasoning_parser: Option<String>,
    pub(crate) stream_response_default_include_usage: bool,
    pub(crate) skip_tokenizer_init: bool,
    pub(crate) vocab_size: u64,
    pub(crate) default_sampling_params: DefaultSamplingParams,
    limits: Limits,
}

impl From<&ServerArgs> for RendererConfig {
    fn from(args: &ServerArgs) -> Self {
        Self {
            served_model_name: args.served_model_name.clone(),
            tokenizer_path: args.tokenizer_path.clone(),
            revision: args.revision.clone(),
            model_path: args.model_path.clone(),
            chat_template: args.chat_template.clone(),
            tool_call_parser: args.tool_call_parser.clone(),
            reasoning_parser: args.reasoning_parser.clone(),
            stream_response_default_include_usage: args.stream_response_default_include_usage,
            skip_tokenizer_init: args.skip_tokenizer_init,
            vocab_size: args.model_config.vocab_size,
            default_sampling_params: args.model_config.default_sampling_params.clone(),
            limits: Limits::from(args),
        }
    }
}

/// The engine-free source of truth for OpenAI request preparation.
///
/// Full inference and `sglang render` own the same service type. The service
/// knows request conversion, chat formatting and the CPU preprocessing queue;
/// it has no scheduler, detokenizer, DP/PD transport or response-stream state.
pub(crate) struct RendererService {
    config: RendererConfig,
    chat_formatter: Option<ChatFormatter>,
    jobs: flume::Sender<PreprocessJob>,
}

/// Response-side context retained by the full OpenAI frontend. It is not part
/// of the token-in wire returned by standalone render endpoints.
pub(crate) struct ChatResponsePlan {
    pub(crate) response_id: String,
    pub(crate) model: String,
    pub(crate) stream: bool,
    pub(crate) choice_count: usize,
    pub(crate) want_logprobs: bool,
    pub(crate) include_usage: bool,
    pub(crate) parser: Option<String>,
    pub(crate) reasoning_parser: Option<String>,
    pub(crate) tools: Option<Vec<ToolDefinition>>,
    pub(crate) stream_tool_choice: Option<ChatCompletionToolChoiceOption>,
    pub(crate) uses_tool_call_structural_tag: bool,
    pub(crate) parallel_tool_calls: bool,
    pub(crate) service_tier: Option<ChatServiceTier>,
}

pub(crate) struct PreparedChat {
    pub(crate) requests: Vec<GenerateRequest>,
    pub(crate) response: ChatResponsePlan,
}

impl RendererService {
    pub(crate) fn new(server_args: Arc<ServerArgs>, jobs: flume::Sender<PreprocessJob>) -> Self {
        let config = RendererConfig::from(&*server_args);
        let chat_formatter = load_chat_support(&config);
        Self {
            config,
            chat_formatter,
            jobs,
        }
    }

    pub(crate) fn config(&self) -> &RendererConfig {
        &self.config
    }

    pub(crate) async fn prepare_chat(
        &self,
        request: &mut CreateChatCompletionRequest,
        response_id: &str,
    ) -> Result<PreparedChat, RenderServiceError> {
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
        Ok(PreparedChat {
            requests: self.prepare_many(lowered.requests).await?,
            response,
        })
    }

    pub(crate) async fn prepare_completions(
        &self,
        request: &CreateCompletionRequest,
        response_id: &str,
    ) -> Result<Vec<GenerateRequest>, RenderServiceError> {
        let requests = lower_completion_requests(&self.config, request, response_id)?;
        self.prepare_many(requests).await
    }

    async fn prepare_many(
        &self,
        requests: Vec<GenerateRequest>,
    ) -> Result<Vec<GenerateRequest>, RenderServiceError> {
        try_join_all(
            requests
                .into_iter()
                .map(|request| self.prepare_one(request)),
        )
        .await
    }

    async fn prepare_one(
        &self,
        mut request: GenerateRequest,
    ) -> Result<GenerateRequest, RenderServiceError> {
        // Token-in servers intentionally have no tokenizer worker. They still
        // share request lowering and all checks that do not require encoding.
        if self.config.skip_tokenizer_init {
            validate_generate_request(&request.rid, &request, &self.config.limits)?;
            if request.has_multimodal() {
                return Err(Error::Validation(
                    "multimodal inputs are not supported by the standalone renderer".into(),
                )
                .into());
            }
            request
                .sampling_params
                .normalize(true, self.config.vocab_size)
                .map_err(Error::from)?;
            check_total_tokens(&mut request, &self.config.limits)?;
            return Ok(request);
        }
        let (reply, result) = oneshot::channel();
        self.jobs
            .send_async(PreprocessJob::Render(Box::new(RenderJob {
                request,
                reply,
            })))
            .await
            .map_err(|_| RenderServiceError::Unavailable)?;
        result
            .await
            .map_err(|error| {
                tracing::error!(%error, "renderer worker dropped reply");
                RenderServiceError::WorkerDropped
            })?
            .map_err(RenderServiceError::Prepare)
    }
}

/// Apply the exact validation, normalization, tokenization and context checks
/// used by a direct render call. The shared worker pool invokes this off the
/// async HTTP runtime.
pub(crate) fn prepare_direct_request(
    mut request: GenerateRequest,
    tokenizer: &dyn TextTokenizer,
    auto_specials: &[i32],
    limits: &Limits,
) -> Result<GenerateRequest, Error> {
    validate_generate_request(&request.rid, &request, limits)?;
    if request.has_multimodal() {
        return Err(Error::Validation(
            "multimodal inputs are not supported by the standalone renderer".into(),
        ));
    }
    request
        .sampling_params
        .normalize(limits.skip_tokenizer_init, limits.vocab_size)?;
    if !request.already_tokenized() {
        tokenize_generate_request(&mut request, tokenizer, auto_specials)?;
    }
    check_total_tokens(&mut request, limits)?;
    Ok(request)
}

/// Public token-in JSON accepted by the existing `/generate` endpoint.
///
/// This is deliberately separate from both the mutable in-process
/// `GenerateRequest` and scheduler-facing `TokenizedGenerateReqInput`.
#[derive(Debug, Serialize)]
pub(crate) struct PreparedGenerateRequest {
    rid: String,
    input_ids: Vec<i32>,
    sampling_params: RenderedSamplingParams,
    stream: bool,
    return_logprob: bool,
    logprob_start_len: i64,
    top_logprobs_num: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    token_ids_logprob: Option<Vec<i32>>,
    return_hidden_states: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_text_in_logprobs: Option<bool>,
}

impl From<GenerateRequest> for PreparedGenerateRequest {
    fn from(mut request: GenerateRequest) -> Self {
        Self {
            rid: request.rid.client_facing().to_owned(),
            input_ids: request
                .input_ids
                .take()
                .expect("text preparation always produces input_ids"),
            sampling_params: request.sampling_params.into(),
            stream: request.stream,
            return_logprob: request.return_logprob,
            logprob_start_len: request.logprob_start_len,
            top_logprobs_num: request.top_logprobs_num,
            token_ids_logprob: request.token_ids_logprob,
            return_hidden_states: request.return_hidden_states,
            return_text_in_logprobs: request.return_text_in_logprobs,
        }
    }
}

/// Public `/generate` sampling shape. Internal scheduler-only normalization
/// fields are folded back into the aliases accepted by the public endpoint.
#[derive(Debug, Serialize)]
struct RenderedSamplingParams {
    max_new_tokens: Option<i64>,
    stop: Vec<String>,
    stop_token_ids: Option<Vec<i64>>,
    stop_regex: Vec<String>,
    temperature: f64,
    top_p: f64,
    top_k: i64,
    min_p: f64,
    frequency_penalty: f64,
    presence_penalty: f64,
    repetition_penalty: f64,
    min_new_tokens: i64,
    n: i64,
    json_schema: Option<String>,
    regex: Option<String>,
    ebnf: Option<String>,
    structural_tag: Option<String>,
    ignore_eos: bool,
    skip_special_tokens: bool,
    spaces_between_special_tokens: bool,
    no_stop_trim: bool,
    stream_interval: Option<i64>,
    logit_bias: Option<BTreeMap<String, f64>>,
    sampling_seed: Option<i64>,
    custom_params: Option<serde_json::Value>,
}

impl From<SamplingParams> for RenderedSamplingParams {
    fn from(params: SamplingParams) -> Self {
        Self {
            max_new_tokens: params.max_new_tokens,
            stop: params.stop_strs,
            stop_token_ids: params.stop_token_ids,
            stop_regex: params.stop_regex_strs,
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            min_p: params.min_p,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            repetition_penalty: params.repetition_penalty,
            min_new_tokens: params.min_new_tokens,
            n: params.n,
            json_schema: params.json_schema,
            regex: params.regex,
            ebnf: params.ebnf,
            structural_tag: params.structural_tag,
            ignore_eos: params.ignore_eos,
            skip_special_tokens: params.skip_special_tokens,
            spaces_between_special_tokens: params.spaces_between_special_tokens,
            no_stop_trim: params.no_stop_trim,
            stream_interval: params.stream_interval,
            logit_bias: params.logit_bias,
            sampling_seed: params.sampling_seed,
            custom_params: params.custom_params,
        }
    }
}
