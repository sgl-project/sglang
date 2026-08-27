//! Adapters between the engine-free renderer crate and the server pipeline.

use std::sync::Arc;

use futures::future::BoxFuture;
use futures::stream::StreamExt;
use tokio::sync::oneshot;

use crate::frontend::{AbortGuard, FrontendHandle};
use crate::message::config::ServerArgs;
use crate::message::ids::Rid;
use crate::message::request::RequestKind;
use crate::message::request::{GenerateRequest, Request};
use crate::message::response::{ChunkEvent, ResponseItem};

use sglang_renderer::{
    FrontendError, GenerationEvent, GenerationFinishReason, GenerationOptions, GenerationOutput,
    GenerationOutputExtras, GenerationSubmission, InferenceBackend, InferenceSession, MatchedStop,
    RendererLimits, SamplingDefaults, TokenIds, TokenizationBackend,
};
pub(crate) use sglang_renderer::{
    GenerationInput, OpenAIRequestLowerer, RendererConfig, RendererError as RenderServiceError,
    RendererService, TextRequest, TokenIdsRequest,
};

#[derive(Clone)]
pub(crate) struct ServerInferenceBackend {
    frontend: Arc<FrontendHandle>,
}

impl ServerInferenceBackend {
    pub(crate) fn new(frontend: Arc<FrontendHandle>) -> Self {
        Self { frontend }
    }
}

pub(crate) struct ServerInferenceSession {
    frontend: Arc<FrontendHandle>,
    guard: AbortGuard,
}

impl InferenceBackend for ServerInferenceBackend {
    type Session = ServerInferenceSession;

    fn begin_session(&self) -> Self::Session {
        ServerInferenceSession {
            frontend: self.frontend.clone(),
            guard: self.frontend.empty_abort_guard(),
        }
    }
}

impl InferenceSession for ServerInferenceSession {
    fn submit(
        &mut self,
        request: GenerationInput,
        _stream: bool,
    ) -> BoxFuture<'_, Result<GenerationSubmission, FrontendError>> {
        Box::pin(async move {
            let request = GenerateRequest::from(request);
            let (rid, mut receiver) = self
                .frontend
                .submit(RequestKind::Generate(Box::new(request)))
                .await
                .map_err(|_| FrontendError {
                    status_code: 503,
                    message: "service unavailable".into(),
                })?;
            self.guard.arm(rid.clone());
            let id = rid.as_str().to_owned();
            let events = async_stream::stream! {
                while let Some(item) = receiver.recv().await {
                    match item {
                        ResponseItem::Frame(output) => match generation_output(output) {
                            Ok(output) => yield Ok(GenerationEvent::Frame(output)),
                            Err(error) => { yield Err(error); break; }
                        },
                        ResponseItem::Done(output) => {
                            yield generation_output(output).map(GenerationEvent::Done);
                            break;
                        }
                        ResponseItem::Error(error) => {
                            yield Err(FrontendError {
                                status_code: error.http_status(),
                                message: error.to_string(),
                            });
                            break;
                        }
                        ResponseItem::Control(_) | ResponseItem::Data(_) => {}
                    }
                }
            }
            .boxed();
            Ok(GenerationSubmission { id, events })
        })
    }

    fn detokenize(&mut self, token_ids: TokenIds) -> BoxFuture<'_, Result<String, FrontendError>> {
        Box::pin(async move {
            let (_rid, mut receiver) = self
                .frontend
                .submit(RequestKind::Detokenize { token_ids })
                .await
                .map_err(|_| FrontendError {
                    status_code: 503,
                    message: "service unavailable".into(),
                })?;
            match receiver.recv().await {
                Some(ResponseItem::Data(bytes)) => {
                    String::from_utf8(bytes.to_vec()).map_err(|_| FrontendError {
                        status_code: 500,
                        message: "detokenized prompt is not valid UTF-8".into(),
                    })
                }
                Some(ResponseItem::Error(error)) => Err(FrontendError {
                    status_code: error.http_status(),
                    message: error.to_string(),
                }),
                _ => Err(FrontendError {
                    status_code: 500,
                    message: "failed to decode prompt: reply channel closed".into(),
                }),
            }
        })
    }

    fn complete(&mut self, submission_id: &str) {
        self.guard.disarm(&submission_id.into());
    }
}

fn generation_output(output: ChunkEvent) -> Result<GenerationOutput, FrontendError> {
    if let Some((status_code, message)) = output
        .finish_reason
        .as_ref()
        .and_then(|reason| reason.abort_status())
    {
        return Err(FrontendError {
            status_code,
            message: message.to_owned(),
        });
    }
    let finish_reason = output.finish_reason.as_ref().and_then(|reason| {
        Some(match reason.kind_name()? {
            "stop" => GenerationFinishReason::Stop(reason.matched().map(|matched| match matched {
                crate::message::finish_reason::Matched::Token(id) => MatchedStop::Token(*id),
                crate::message::finish_reason::Matched::Str(text) => {
                    MatchedStop::Text(text.clone())
                }
                crate::message::finish_reason::Matched::Tokens(ids) => {
                    MatchedStop::Tokens(ids.clone())
                }
            })),
            "length" => GenerationFinishReason::Length,
            "abort" => GenerationFinishReason::Abort,
            "content_filter" => GenerationFinishReason::ContentFilter,
            other => GenerationFinishReason::Other(other.to_owned()),
        })
    });
    let extras = output.extras.map(|extras| {
        Box::new(GenerationOutputExtras {
            output_logprobs: extras.out_lp_val,
            output_logprob_token_ids: extras.out_lp_idx,
            output_logprob_text: extras.out_lp_txt,
            input_logprobs: extras.in_lp_val,
            input_logprob_token_ids: extras.in_lp_idx,
            input_logprob_text: extras.in_lp_txt,
            output_top_logprobs: extras.out_top_val,
            output_top_logprob_token_ids: extras.out_top_idx,
            output_top_logprob_lengths: extras.out_top_lens,
            output_top_logprob_text: extras.out_top_txt,
            input_top_logprobs: extras.in_top_val,
            input_top_logprob_token_ids: extras.in_top_idx,
            input_top_logprob_lengths: extras.in_top_lens,
            input_top_logprob_text: extras.in_top_txt,
        })
    });
    Ok(GenerationOutput {
        text: output.text,
        token_ids: output.token_ids,
        finish_reason,
        prompt_tokens: output.prompt_tokens,
        completion_tokens: output.completion_tokens,
        extras,
    })
}

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
        new_request_lowerer(&server_args),
        Arc::new(ServerTokenizationBackend { jobs }),
    )
}

pub(crate) fn new_request_lowerer(server_args: &ServerArgs) -> OpenAIRequestLowerer {
    OpenAIRequestLowerer::new(renderer_config(server_args))
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
