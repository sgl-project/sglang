//! Chat prompt rendering through `sglang-processor`.
//!
//! The processor owns template loading and the SGLang-parity formatters; this
//! module adapts the server's wire request to it.

use std::collections::HashMap;

use dynamo_protocols::types::{ChatCompletionRequestMessage, CreateChatCompletionRequest};
use serde_json::Value;
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
    defaults: &ChatTemplateKwargs,
) -> Result<RenderedChat, TemplateError> {
    match formatter {
        ChatFormatter::DeepSeekV4(_)
        | ChatFormatter::DeepSeekV41
        | ChatFormatter::Glm53(_)
        | ChatFormatter::KimiK3
        | ChatFormatter::Qwen38 { .. } => {
            // The continuation's leading BOS is dropped once its ids are encoded.
            let (prompt, continuation) = match formatter.render_request(body, defaults) {
                Ok(rendered) => rendered,
                // There is no engine to hand a request the SGLang path declines
                // to, so the formatter's own template renders it, as it did
                // before that path existed. DeepSeek-V4 has no other path.
                Err(TemplateError::EngineOnly { .. })
                    if !matches!(
                        formatter,
                        ChatFormatter::DeepSeekV4(_) | ChatFormatter::DeepSeekV41
                    ) =>
                {
                    return render_template(formatter, request, kwargs, defaults);
                }
                Err(error) => return Err(error),
            };
            Ok(RenderedChat {
                prompt,
                continuation,
            })
        }
        _ => render_template(formatter, request, kwargs, defaults),
    }
}

/// Render `request` through the formatter's template (Jinja, Dynamo's
/// encoder, or the legacy conversation).
fn render_template(
    formatter: &ChatFormatter,
    request: &CreateChatCompletionRequest,
    kwargs: Option<&ChatTemplateKwargs>,
    defaults: &ChatTemplateKwargs,
) -> Result<RenderedChat, TemplateError> {
    // Python's `_process_messages`: the defaults under the request's own.
    let mut merged = defaults.clone();
    merged.extend(kwargs.cloned().unwrap_or_default());
    let kwargs = (!merged.is_empty()).then_some(&merged);
    formatter
        .render_prompt(&TemplateRequest { request, kwargs })
        .map(|prompt| RenderedChat {
            prompt,
            continuation: String::new(),
        })
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
                Err(TemplateError::EngineOnly { .. })
            ));
            let request: CreateChatCompletionRequest =
                serde_json::from_value(body.clone()).unwrap();
            let rendered = render_chat(&formatter, body, &request, None, &HashMap::new())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let prompt = rendered.prompt.as_str();
            assert!(
                prompt.contains("foo") && prompt.contains("<|im_start|>assistant\n<think>"),
                "{name}: {prompt}"
            );
        }
    }
}
