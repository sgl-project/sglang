//! OpenAI render-only operations, without model execution or HTTP framing.

use super::protocol::{
    ChatCompletionRequest, CompletionRequest, lower_chat_request, lower_text_completion_request,
    lower_token_ids_completion_request,
};
use crate::{GenerateRequest, RendererService, ResponseError};
use dynamo_protocols::types::Prompt;

pub(crate) async fn render_chat(
    renderer: &RendererService,
    request: ChatCompletionRequest,
) -> Result<GenerateRequest, ResponseError> {
    if request.n.is_some_and(|n| n > 1) {
        return Err(ResponseError {
            kind: crate::ResponseErrorKind::InvalidRequest,
            message: "the standalone chat renderer currently requires n=1".into(),
        });
    }
    let (_, request) = lower_chat_request(renderer.config(), request)?;
    let mut chat = renderer.prepare_chat(request).await?;
    Ok(chat
        .requests
        .pop()
        .expect("chat generation contains one request"))
}

pub(crate) async fn render_completions(
    renderer: &RendererService,
    request: CompletionRequest,
) -> Result<Vec<GenerateRequest>, ResponseError> {
    let text_prompt = matches!(&request.prompt, Prompt::String(_) | Prompt::StringArray(_));
    if text_prompt {
        let (_, requests) = lower_text_completion_request(renderer.config(), &request)?;
        renderer
            .prepare_text_request_groups(requests)
            .await
            .map_err(ResponseError::from)
    } else {
        let (_, requests) = lower_token_ids_completion_request(renderer.config(), &request)?;
        renderer
            .prepare_token_ids_requests(requests)
            .map_err(ResponseError::from)
    }
}
