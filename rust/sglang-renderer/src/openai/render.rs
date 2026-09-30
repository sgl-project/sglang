//! OpenAI render-only operations, without model execution or HTTP framing.

use super::protocol::{ChatCompletionRequest, CompletionRequest};
use crate::{GenerateRequest, RendererService, ResponseError};

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
    let (_, mut chat) = super::chat::prepare_request(renderer, request).await?;
    Ok(chat
        .requests
        .pop()
        .expect("chat generation contains one request"))
}

pub(crate) async fn render_completions(
    renderer: &RendererService,
    request: CompletionRequest,
) -> Result<Vec<GenerateRequest>, ResponseError> {
    let (_, requests) = super::completions::prepare_request(renderer, &request).await?;
    Ok(requests)
}
