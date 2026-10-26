//! Chat prompt rendering: adapts the server's wire request to the
//! `sglang-processor` formatters.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use sglang_processor::dynamo_protocols::types::CreateChatCompletionRequest;
use sglang_processor::dynamo_renderer::RenderedPrompt;
use tokio::sync::Semaphore;

pub use sglang_processor::ChatFormatter;
use sglang_processor::{TemplateArgs, TemplateError};

pub(super) use crate::message::request::RenderedChat;

/// Extra variables for the chat template (`chat_template_kwargs`).
pub type ChatTemplateKwargs = HashMap<String, serde_json::Value>;

/// Render a chat request for the tokenizer pool: the raw `body` through the
/// SGLang request path, or `request` through the template when that declines it;
/// either way the `continue_final_message` text stays apart from the prompt.
/// A formatter that encodes its own prompt (DeepSeek through deepseek-recipe)
/// also returns the request's image sources, in its placeholder order.
pub(super) fn render_chat(
    formatter: &ChatFormatter,
    body: Value,
    request: &CreateChatCompletionRequest,
    defaults: &ChatTemplateKwargs,
) -> Result<(RenderedChat, Option<Vec<String>>), TemplateError> {
    if let Some(encoded) = formatter.encode_multimodal_request(body.clone(), defaults) {
        let encoded = encoded?;
        let chat = RenderedChat {
            prompt: RenderedPrompt::text(encoded.text),
            continuation: String::new(),
            ids: Some(encoded.ids),
        };
        return Ok((chat, Some(encoded.images)));
    }
    let args = TemplateArgs::new(&body, defaults);
    let ids = formatter
        .encode_request(body.clone(), defaults)
        .transpose()?;
    let (rendered, ids) = match formatter.render_request(body, defaults) {
        Err(TemplateError::NeedsRenderFallback { .. }) => {
            (formatter.render_template(request, &args), None)
        }
        rendered => (rendered, ids),
    };
    let chat = rendered.map(|(prompt, continuation)| RenderedChat {
        prompt,
        continuation,
        ids,
    })?;
    Ok((chat, None))
}

/// Chat prompts a formatter encodes itself (DeepSeek through deepseek-recipe),
/// which takes the better part of a second for a long prompt. They render off
/// the HTTP executor, as many at once as the tokenizer pool has workers.
#[derive(Clone)]
pub struct ChatEncodes(Arc<Semaphore>);

impl ChatEncodes {
    pub(in crate::api_server::http) fn new(workers: usize) -> Self {
        Self(Arc::new(Semaphore::new(workers.max(1))))
    }
}

impl Default for ChatEncodes {
    fn default() -> Self {
        Self::new(1)
    }
}

/// [`render_chat`], as a blocking task when the formatter encodes its own prompt.
pub(super) async fn render_chat_off_executor(
    encodes: &ChatEncodes,
    formatter: &ChatFormatter,
    body: Value,
    request: &CreateChatCompletionRequest,
    defaults: &ChatTemplateKwargs,
) -> Result<(RenderedChat, Option<Vec<String>>), TemplateError> {
    if !matches!(
        formatter,
        ChatFormatter::DeepSeekV4(_) | ChatFormatter::DeepSeekV41(_)
    ) {
        return render_chat(formatter, body, request, defaults);
    }
    let permit = Arc::clone(&encodes.0)
        .acquire_owned()
        .await
        .expect("the chat encode semaphore is never closed");
    let (formatter, request, defaults) = (formatter.clone(), request.clone(), defaults.clone());
    tokio::task::spawn_blocking(move || {
        // Held by the blocking task: a dropped request cannot free its permit
        // while its encode still runs.
        let _permit = permit;
        render_chat(&formatter, body, &request, &defaults)
    })
    .await
    .map_err(|error| TemplateError::Renderer {
        message: error.to_string(),
    })?
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::checkpoint_formatter;
    use super::*;

    /// A DeepSeek encode runs off the HTTP executor: on a current-thread
    /// runtime, the handler awaiting it yields, so the encode cannot finish
    /// before the executor next polls it.
    #[tokio::test(flavor = "current_thread")]
    async fn deepseek_encode_leaves_the_executor_free() {
        let formatter = super::super::test_utils::deepseek_formatter();
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "Explain and"}],
        });
        let request: CreateChatCompletionRequest = serde_json::from_value(body.clone()).unwrap();
        let encode = tokio::spawn(async move {
            let encodes = ChatEncodes::default();
            render_chat_off_executor(&encodes, &formatter, body, &request, &HashMap::new()).await
        });
        tokio::task::yield_now().await;
        assert!(!encode.is_finished(), "the encode ran on the executor");
        let (chat, images) = encode.await.unwrap().unwrap();
        assert!(chat.ids.is_some_and(|ids| !ids.is_empty()));
        assert_eq!(images, Some(vec![]));
    }

    /// Floats Qwen3.8's renderer declines (serde_json prints them unlike Python)
    /// render through the template instead of failing the request.
    #[test]
    fn declined_requests_render_through_the_template() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sglang-processor/tests/fixtures/parity/qwen3.8-27b.json"
        );
        let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let template = fixture["chat_template"].as_str().unwrap();
        let formatter = checkpoint_formatter(&fixture["config"], template);
        assert!(matches!(formatter, ChatFormatter::Qwen38 { .. }));

        for name in ["history_float_arguments", "tool_float_values"] {
            let case = fixture["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|case| case["name"] == name)
                .unwrap();
            let body = case["request"].clone();
            assert!(matches!(
                formatter.render_request(body.clone(), &HashMap::new()),
                Err(TemplateError::NeedsRenderFallback { .. })
            ));
            let request: CreateChatCompletionRequest =
                serde_json::from_value(body.clone()).unwrap();
            let (rendered, _) = render_chat(&formatter, body, &request, &HashMap::new())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let prompt = rendered.prompt.as_str();
            assert!(
                prompt.contains("foo") && prompt.contains("<|im_start|>assistant\n<think>"),
                "{name}: {prompt}"
            );
        }
    }
}
