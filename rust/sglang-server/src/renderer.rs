//! Adapters between the engine-free renderer crate and the server pipeline.

use std::sync::Arc;

use futures::future::BoxFuture;
use tokio::sync::oneshot;

use crate::message::config::ServerArgs;
use crate::message::ids::Rid;
use crate::message::request::{GenerateRequest, Request};

pub(crate) use sglang_renderer::{
    PreparedGenerateRequest, RendererConfig, RendererError as RenderServiceError, RendererService,
    RequestLowerer, TextCompletionRequest,
};
use sglang_renderer::{PreprocessBackend, RendererLimits, SamplingDefaults};

/// Shared tokenizer-pool work. Engine requests retain their FSM, while renderer
/// requests use the crate-owned request and error contracts.
pub(crate) enum PreprocessJob {
    Inference(Request),
    Render(Box<RenderJob>),
}

pub(crate) struct RenderJob {
    pub(crate) request: TextCompletionRequest,
    pub(crate) reply: oneshot::Sender<Result<TextCompletionRequest, RenderServiceError>>,
}

struct ServerPreprocessBackend {
    jobs: flume::Sender<PreprocessJob>,
}

impl PreprocessBackend for ServerPreprocessBackend {
    fn prepare(
        &self,
        request: TextCompletionRequest,
    ) -> BoxFuture<'static, Result<TextCompletionRequest, RenderServiceError>> {
        let jobs = self.jobs.clone();
        Box::pin(async move {
            let (reply, result) = oneshot::channel();
            jobs.send_async(PreprocessJob::Render(Box::new(RenderJob {
                request,
                reply,
            })))
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
    jobs: flume::Sender<PreprocessJob>,
) -> RendererService {
    RendererService::new(
        new_request_lowerer(&server_args),
        Arc::new(ServerPreprocessBackend { jobs }),
    )
}

pub(crate) fn new_request_lowerer(server_args: &ServerArgs) -> RequestLowerer {
    RequestLowerer::new(renderer_config(server_args))
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

impl From<TextCompletionRequest> for GenerateRequest {
    fn from(request: TextCompletionRequest) -> Self {
        Self {
            rid: Rid::from_client(&request.rid),
            text: request.text,
            input_ids: request.input_ids,
            skip_special_tokens: request.skip_special_tokens,
            sampling_params: request.sampling_params,
            stream: request.stream,
            return_logprob: request.return_logprob,
            logprob_start_len: request.logprob_start_len,
            top_logprobs_num: request.top_logprobs_num,
            token_ids_logprob: request.token_ids_logprob,
            return_hidden_states: request.return_hidden_states,
            return_text_in_logprobs: request.return_text_in_logprobs,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sglang_renderer::SamplingParams;

    #[test]
    fn engine_conversion_preserves_renderer_fields() {
        let rendered = TextCompletionRequest {
            rid: "client-rid".into(),
            text: Some("prompt".into()),
            input_ids: Some(vec![1, 2, 3]),
            skip_special_tokens: true,
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
        };

        let engine = GenerateRequest::from(rendered);
        assert_eq!(engine.rid.client_facing(), "client-rid");
        assert_eq!(engine.text.as_deref(), Some("prompt"));
        assert_eq!(engine.input_ids, Some(vec![1, 2, 3]));
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
}
