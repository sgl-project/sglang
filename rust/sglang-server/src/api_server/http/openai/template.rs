//! Chat prompt rendering through `sglang-processor`.
//!
//! The processor owns template loading and the SGLang-parity formatters; this
//! module adapts the server's wire request to it.

use std::collections::HashMap;

use serde_json::Value;
use sglang_processor::dynamo_protocols::types::{
    ChatCompletionRequestMessage, CreateChatCompletionRequest,
};
use sglang_processor::dynamo_renderer::{OAIChatLikeRequest, RenderedPrompt, TextInput};
use sglang_processor::minijinja;

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
    defaults: &ChatTemplateKwargs,
) -> Result<RenderedChat, TemplateError> {
    let args = TemplateArgs::new(&body, defaults);
    let through_template = || {
        render_template(formatter, request, &args).map(|prompt| RenderedChat {
            prompt,
            continuation: String::new(),
        })
    };
    if !formatter.renders_requests() {
        return through_template();
    }
    match formatter.render_request(body, defaults) {
        Ok((prompt, continuation)) => Ok(RenderedChat {
            prompt,
            continuation,
        }),
        // The SGLang path declines some valid requests only the engine renders;
        // with no engine here, the formatter's template renders them (one
        // with no other path declines them again).
        Err(TemplateError::NeedsRenderFallback { .. }) => through_template(),
        Err(error) => Err(error),
    }
}

/// What SGLang's chat layer reads off a raw chat body for its template. The
/// body must be one SGLang's `ChatCompletionRequest` accepts; it is not
/// re-validated.
#[derive(Debug, Clone, Default)]
pub struct TemplateArgs {
    kwargs: HashMap<String, Value>,
}

impl TemplateArgs {
    /// Python's template kwargs (`_process_messages`): the server `defaults`
    /// under the request's own `chat_template_kwargs`.
    pub fn new(body: &Value, defaults: &HashMap<String, Value>) -> Self {
        let mut kwargs = defaults.clone();
        let own = body["chat_template_kwargs"]
            .as_object()
            .cloned()
            .unwrap_or_default();
        kwargs.extend(own);
        Self { kwargs }
    }
}

/// Render `request` through the formatter's own template (Jinja, Dynamo's
/// encoder, or the legacy conversation) as SGLang's chat layer does.
fn render_template(
    formatter: &ChatFormatter,
    request: &CreateChatCompletionRequest,
    args: &TemplateArgs,
) -> Result<RenderedPrompt, TemplateError> {
    let request = TemplateRequest {
        request,
        kwargs: (!args.kwargs.is_empty()).then_some(&args.kwargs),
    };
    formatter.render_prompt(&request)
}

/// The wire request plus its `chat_template_kwargs`, which the protocol type
/// does not carry.
struct TemplateRequest<'a> {
    request: &'a CreateChatCompletionRequest,
    kwargs: Option<&'a HashMap<String, Value>>,
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
    fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
        self.kwargs
    }
    fn extract_text(&self) -> Option<TextInput> {
        self.request.extract_text()
    }
    fn mm_processor_kwargs(&self) -> Option<&Value> {
        self.request.mm_processor_kwargs()
    }
}

#[cfg(test)]
mod tests {
    use sglang_processor::ChatFormatterOptions;

    use super::*;

    /// Qwen3.8's renderer declines floats serde_json prints unlike Python;
    /// with no engine to hand them to, the template renders them instead of
    /// the request failing.
    #[test]
    fn declined_requests_render_through_the_template() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sglang-processor/tests/fixtures/parity/qwen3.8-27b.json"
        );
        let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let dir = std::env::temp_dir().join(format!("qwen38-declined-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), fixture["config"].to_string()).unwrap();
        let template = fixture["chat_template"].as_str().unwrap();
        std::fs::write(dir.join("chat_template.jinja"), template).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
        let (formatter, error) = sglang_processor::select_chat_formatter(&ChatFormatterOptions {
            tokenizer_path: dir.to_string_lossy().into_owned(),
            ..Default::default()
        });
        std::fs::remove_dir_all(&dir).unwrap();
        let formatter = formatter.unwrap_or_else(|| panic!("{error:?}"));
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
            let rendered = render_chat(&formatter, body, &request, &HashMap::new())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let prompt = rendered.prompt.as_str();
            assert!(
                prompt.contains("foo") && prompt.contains("<|im_start|>assistant\n<think>"),
                "{name}: {prompt}"
            );
        }
    }
}
