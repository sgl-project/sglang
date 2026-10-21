//! Chat prompt rendering through `sglang-processor`.
//!
//! The processor owns template loading and the SGLang-parity formatters; this
//! module adapts the server's wire request to it.

use std::collections::HashMap;

use dynamo_protocols::types::{ChatCompletionRequestMessage, CreateChatCompletionRequest};
use serde_json::Value;
use sglang_processor::dynamo_renderer::deepseek::common::tokens;
use sglang_processor::dynamo_renderer::{OAIChatLikeRequest, TextInput};

pub use sglang_processor::ChatFormatter;
use sglang_processor::TemplateError;

pub(super) use crate::message::request::RenderedChat;

/// Extra variables for the chat template (`chat_template_kwargs`).
pub type ChatTemplateKwargs = HashMap<String, serde_json::Value>;

/// Render a chat request for the tokenizer pool.
/// Formatters with an SGLang request path render the raw `body` and return the
/// `continue_final_message` prefix separately; the rest render `request`, and
/// their templates fold any continuation into the prompt.
pub(super) fn render_chat(
    formatter: &ChatFormatter,
    body: Value,
    request: &CreateChatCompletionRequest,
    kwargs: Option<&ChatTemplateKwargs>,
) -> Result<RenderedChat, TemplateError> {
    match formatter {
        ChatFormatter::DeepSeekV4(_)
        | ChatFormatter::DeepSeekV41
        | ChatFormatter::Glm53(_)
        | ChatFormatter::KimiK3
        | ChatFormatter::Qwen38 { .. } => {
            // `--default-chat-template-kwargs` is refused at launch.
            let (prompt, mut continuation) = formatter.render_request(body, &HashMap::new())?;
            // SGLang tokenizes the DeepSeek-V4 prefix on its own and drops its leading BOS.
            if matches!(
                formatter,
                ChatFormatter::DeepSeekV4(_) | ChatFormatter::DeepSeekV41
            ) && let Some(rest) = continuation.strip_prefix(tokens::BOS)
            {
                continuation = rest.to_owned();
            }
            Ok(RenderedChat {
                prompt,
                continuation,
            })
        }
        _ => formatter
            .render_prompt(&TemplateRequest { request, kwargs })
            .map(|prompt| RenderedChat {
                prompt,
                continuation: String::new(),
            }),
    }
}

/// The wire request plus its `chat_template_kwargs`, which the protocol type
/// does not carry.
struct TemplateRequest<'a> {
    request: &'a CreateChatCompletionRequest,
    kwargs: Option<&'a ChatTemplateKwargs>,
}

impl OAIChatLikeRequest for TemplateRequest<'_> {
    fn model(&self) -> String {
        self.request.model()
    }
    fn messages(&self) -> minijinja::Value {
        self.request.messages()
    }
    fn typed_messages(&self) -> Option<&[ChatCompletionRequestMessage]> {
        self.request.typed_messages()
    }
    fn tools(&self) -> Option<minijinja::Value> {
        self.request.tools()
    }
    fn tool_choice(&self) -> Option<minijinja::Value> {
        self.request.tool_choice()
    }
    fn response_format(&self) -> Option<minijinja::Value> {
        self.request.response_format()
    }
    fn reasoning_effort(&self) -> Option<minijinja::Value> {
        self.request.reasoning_effort()
    }
    fn should_add_generation_prompt(&self) -> bool {
        self.request.should_add_generation_prompt()
    }
    fn chat_template_args(&self) -> Option<&ChatTemplateKwargs> {
        self.kwargs
    }
    fn extract_text(&self) -> Option<TextInput> {
        self.request.extract_text()
    }
    fn mm_processor_kwargs(&self) -> Option<&serde_json::Value> {
        self.request.mm_processor_kwargs()
    }
}
