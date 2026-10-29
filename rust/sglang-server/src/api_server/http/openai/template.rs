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
/// SGLang request path, or `request` through the template when that declines it;
/// either way the `continue_final_message` text stays apart from the prompt.
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
    let (rendered, ids) = match formatter.render_request(body, defaults) {
        Err(TemplateError::NeedsRenderFallback { .. }) => {
            (formatter.render_template(request, &args), None)
        }
        rendered => (rendered, ids),
    };
    rendered.map(|(prompt, continuation)| RenderedChat {
        prompt,
        continuation,
        ids,
    })
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

    /// MiniMax-M3's request validation is a hard error, not a decline: Python
    /// rejects these with 400 (`_validate_minimax_m3_request`), and sglang-server
    /// has no engine to fall back to — so they must not reach the generic
    /// template, which would serve a late `root` or an unanswered tool call.
    #[test]
    fn minimax_m3_validation_errors_do_not_fall_back() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sglang-processor/tests/fixtures/parity/minimax-m3.1-flash.json"
        );
        let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let template = fixture["chat_template"].as_str().unwrap();
        let formatter = checkpoint_formatter(&fixture["config"], template);
        assert!(matches!(formatter, ChatFormatter::MiniMaxM3(_)));

        // An assistant tool call answered by a *second*, unpaired tool message:
        // `_validate_minimax_m3_request` rejects it (tool calls and results must
        // pair up), so rendering must fail rather than fall back to the template.
        let body = serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "foo", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "a", "content": "r"},
                {"role": "tool", "tool_call_id": "b", "content": "extra"},
            ],
        });
        let request: CreateChatCompletionRequest = serde_json::from_value(body.clone()).unwrap();
        match render_chat(&formatter, body, &request, &HashMap::new()) {
            Err(TemplateError::Renderer { .. }) => {}
            Err(other) => panic!("expected a Renderer error, got {other:?}"),
            Ok(_) => panic!("an unpaired tool message must not render"),
        }
    }
}
