//! Chat prompt rendering through `sglang-processor`.
//!
//! The processor owns template loading and the SGLang-parity formatters; this
//! module adapts the server's wire request to it.

use std::collections::HashMap;

use serde_json::Value;
use sglang_processor::dynamo_protocols::types::CreateChatCompletionRequest;

pub use sglang_processor::ChatFormatter;
use sglang_processor::{TemplateArgs, TemplateError};

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
    match render_body(formatter, body, defaults) {
        // Formatters without an SGLang request path render through their
        // template, and so do the valid requests that path declines, which
        // only the engine renders; one with no other path declines them again.
        Some(Err(TemplateError::NeedsRenderFallback { .. })) | None => formatter
            .render_template(request, &args)
            .map(|prompt| RenderedChat {
                prompt,
                continuation: String::new(),
            }),
        Some(rendered) => rendered,
    }
}

/// Render the raw body as SGLang does, retaining the continuation boundary.
pub(super) fn render_body(
    formatter: &ChatFormatter,
    body: Value,
    defaults: &ChatTemplateKwargs,
) -> Option<Result<RenderedChat, TemplateError>> {
    if !formatter.renders_requests() {
        return None;
    }
    Some(
        formatter
            .render_request(body, defaults)
            .map(|(prompt, continuation)| RenderedChat {
                prompt,
                continuation,
            }),
    )
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
