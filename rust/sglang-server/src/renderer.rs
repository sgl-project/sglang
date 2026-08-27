//! Adapters between the engine-free renderer crate and the server pipeline.

use std::sync::Arc;

use futures::future::BoxFuture;
use tokio::sync::oneshot;

use crate::message::config::ServerArgs;
use crate::message::ids::Rid;
use crate::message::request::{GenerateRequest, Request};

pub(crate) use sglang_renderer::{
    GenerationInput, OpenAIRequestProcessor, RendererConfig, RendererError as RenderServiceError,
    RendererService, TextRequest, TokenIdsRequest,
};
use sglang_renderer::{
    GenerationOptions, RendererLimits, SamplingDefaults, TokenIds, TokenizationBackend,
};

/// Work accepted by the shared tokenization pool. Inference requests retain
/// their FSM, while standalone requests use the renderer crate's contracts.
pub(crate) enum TokenizationJob {
    Inference(Request),
    Standalone(Box<StandaloneTokenizationJob>),
}

pub(crate) struct StandaloneTokenizationJob {
    pub(crate) request: TextRequest,
    pub(crate) reply: oneshot::Sender<Result<TokenIdsRequest, RenderServiceError>>,
}

struct ServerTokenizationBackend {
    jobs: flume::Sender<TokenizationJob>,
}

impl TokenizationBackend for ServerTokenizationBackend {
    fn tokenize(
        &self,
        request: TextRequest,
    ) -> BoxFuture<'static, Result<TokenIdsRequest, RenderServiceError>> {
        let jobs = self.jobs.clone();
        Box::pin(async move {
            let (reply, result) = oneshot::channel();
            jobs.send_async(TokenizationJob::Standalone(Box::new(
                StandaloneTokenizationJob { request, reply },
            )))
            .await
            .map_err(|_| RenderServiceError::Unavailable)?;
            result.await.map_err(|error| {
                tracing::error!(%error, "renderer worker dropped reply");
                RenderServiceError::WorkerDropped
            })?
        })
    }
}

pub(crate) fn new_renderer_service(
    server_args: Arc<ServerArgs>,
    jobs: flume::Sender<TokenizationJob>,
) -> RendererService {
    RendererService::new(
        new_request_processor(&server_args),
        Arc::new(ServerTokenizationBackend { jobs }),
    )
}

pub(crate) fn new_request_processor(server_args: &ServerArgs) -> OpenAIRequestProcessor {
    OpenAIRequestProcessor::new(renderer_config(server_args))
}

fn renderer_config(args: &ServerArgs) -> RendererConfig {
    RendererConfig {
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
        default_sampling_params: SamplingDefaults {
            temperature: args.model_config.default_sampling_params.temperature,
            top_p: args.model_config.default_sampling_params.top_p,
        },
        limits: RendererLimits {
            skip_tokenizer_init: args.skip_tokenizer_init,
            vocab_size: args.model_config.vocab_size,
            context_len: args.model_config.context_len,
            num_reserved_tokens: args.num_reserved_tokens,
            allow_auto_truncate: args.allow_auto_truncate,
            enable_return_hidden_states: args.enable_return_hidden_states,
        },
    }
}

pub(crate) fn render_http_status(error: &RenderServiceError) -> u16 {
    match error.kind() {
        sglang_renderer::RendererErrorKind::InvalidRequest => 400,
        sglang_renderer::RendererErrorKind::Unavailable => 503,
        sglang_renderer::RendererErrorKind::Tokenize
        | sglang_renderer::RendererErrorKind::Internal => 500,
    }
}

impl From<TextRequest> for GenerateRequest {
    fn from(request: TextRequest) -> Self {
        native_generate_request(
            request.rid,
            Some(request.text),
            None,
            request.skip_special_tokens,
            request.options,
        )
    }
}

impl From<TokenIdsRequest> for GenerateRequest {
    fn from(request: TokenIdsRequest) -> Self {
        native_generate_request(
            request.rid,
            None,
            Some(request.input_ids),
            false,
            request.options,
        )
    }
}

fn native_generate_request(
    rid: String,
    text: Option<String>,
    input_ids: Option<TokenIds>,
    skip_special_tokens: bool,
    options: GenerationOptions,
) -> GenerateRequest {
    GenerateRequest {
        rid: Rid::from_client(&rid),
        text,
        input_ids,
        skip_special_tokens,
        sampling_params: options.sampling_params,
        stream: options.stream,
        return_logprob: options.return_logprob,
        logprob_start_len: options.logprob_start_len,
        top_logprobs_num: options.top_logprobs_num,
        token_ids_logprob: options.token_ids_logprob,
        return_hidden_states: options.return_hidden_states,
        return_text_in_logprobs: options.return_text_in_logprobs,
        ..Default::default()
    }
}

impl From<GenerationInput> for GenerateRequest {
    fn from(request: GenerationInput) -> Self {
        match request {
            GenerationInput::Text(request) => request.into(),
            GenerationInput::TokenIds(request) => request.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sglang_renderer::{
        SamplingParams, TextTokenizer, prepare_direct_request, tokenize_text_prompt,
    };

    use crate::tokenizer_manager::to_scheduler::{
        Limits, check_total_tokens, validate_generate_request,
    };

    #[test]
    fn engine_conversion_preserves_renderer_fields() {
        let rendered = TextRequest {
            rid: "client-rid".into(),
            text: "prompt".into(),
            skip_special_tokens: true,
            options: GenerationOptions {
                sampling_params: SamplingParams {
                    max_new_tokens: Some(17),
                    temperature: 0.25,
                    ..Default::default()
                },
                stream: true,
                return_logprob: true,
                logprob_start_len: 2,
                top_logprobs_num: 4,
                token_ids_logprob: Some(vec![5, 6]),
                return_hidden_states: true,
                return_text_in_logprobs: Some(true),
            },
        };

        let engine = GenerateRequest::from(rendered);
        assert_eq!(engine.rid.client_facing(), "client-rid");
        assert_eq!(engine.text.as_deref(), Some("prompt"));
        assert_eq!(engine.input_ids, None);
        assert!(engine.skip_special_tokens);
        assert_eq!(engine.sampling_params.max_new_tokens, Some(17));
        assert_eq!(engine.sampling_params.temperature, 0.25);
        assert!(engine.stream);
        assert!(engine.return_logprob);
        assert_eq!(engine.logprob_start_len, 2);
        assert_eq!(engine.top_logprobs_num, 4);
        assert_eq!(engine.token_ids_logprob, Some(vec![5, 6]));
        assert!(engine.return_hidden_states);
        assert_eq!(engine.return_text_in_logprobs, Some(true));
    }

    struct WordTokenizer;

    impl TextTokenizer for WordTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<i32>, sglang_renderer::RendererError> {
            Ok(text.split_whitespace().map(|_| 7).collect())
        }
    }

    #[test]
    fn standalone_and_inference_stages_prepare_identical_completions() {
        let limits = Limits {
            skip_tokenizer_init: false,
            vocab_size: 128,
            context_len: 5,
            num_reserved_tokens: 0,
            allow_auto_truncate: true,
            enable_return_hidden_states: false,
        };
        let request = TextRequest {
            rid: "completion-1".into(),
            text: "one two three".into(),
            skip_special_tokens: false,
            options: GenerationOptions {
                sampling_params: SamplingParams {
                    max_new_tokens: Some(4),
                    stop: Some(crate::message::types::OneOrMany::One("two words".into())),
                    ..Default::default()
                },
                ..Default::default()
            },
        };

        let standalone = prepare_direct_request(
            GenerationInput::Text(request.clone()),
            &WordTokenizer,
            &[],
            &RendererLimits::from(&limits),
        )
        .unwrap();

        let mut inference = GenerateRequest::from(request);
        validate_generate_request(&inference.rid, &inference, &limits).unwrap();
        inference
            .sampling_params
            .normalize(limits.skip_tokenizer_init, limits.vocab_size)
            .unwrap();
        inference.input_ids = Some(
            tokenize_text_prompt(
                inference.text.as_deref().unwrap_or_default(),
                inference.skip_special_tokens,
                &mut inference.sampling_params,
                &WordTokenizer,
                &[],
            )
            .unwrap(),
        );
        check_total_tokens(&mut inference, &limits).unwrap();

        assert_eq!(inference.input_ids, Some(standalone.input_ids));
        assert_eq!(
            inference.sampling_params,
            standalone.options.sampling_params
        );
        assert_eq!(inference.sampling_params.stop_str_max_len, 2);
        assert_eq!(inference.sampling_params.max_new_tokens, Some(2));
    }
}
