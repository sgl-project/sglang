//! Chat prompt rendering: adapts the server's wire request to the
//! `sglang-processor` formatters.

use std::collections::HashMap;

use serde_json::Value;
use sglang_processor::dynamo_protocols::types::CreateChatCompletionRequest;

pub use sglang_processor::ChatFormatter;
use sglang_processor::{TemplateArgs, TemplateError};

pub(super) use crate::message::request::RenderedChat;

/// Extra variables for the chat template (`chat_template_kwargs`).
pub type ChatTemplateKwargs = HashMap<String, serde_json::Value>;

/// Render a chat request for the tokenizer pool: the raw `body` through the
/// SGLang request path, or `request` through the template when that declines it.
pub(super) fn render_chat(
    formatter: &ChatFormatter,
    body: Value,
    request: &CreateChatCompletionRequest,
    defaults: &ChatTemplateKwargs,
) -> Result<RenderedChat, TemplateError> {
    let args = TemplateArgs::new(&body, defaults);
    let ids = formatter
        .encode_request(body.clone(), defaults)
        .transpose()?;
    match formatter.render_request(body, defaults) {
        Err(TemplateError::NeedsRenderFallback { .. }) => formatter
            .render_template(request, &args)
            .map(|prompt| RenderedChat {
                prompt,
                continuation: String::new(),
                ids: None,
            }),
        rendered => rendered.map(|(prompt, continuation)| RenderedChat {
            prompt,
            continuation,
            ids,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::checkpoint_formatter;
    use super::*;

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
