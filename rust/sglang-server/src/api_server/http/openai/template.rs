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

/// Render a chat request for the tokenizer pool, returning the
/// `continue_final_message` prefix apart from the prompt. Formatters with an
/// SGLang request path render the raw `body`; the rest render `request`
/// through their template, as SGLang's chat layer does.
pub(super) fn render_chat(
    formatter: &ChatFormatter,
    body: Value,
    request: &CreateChatCompletionRequest,
    defaults: &ChatTemplateKwargs,
) -> Result<RenderedChat, TemplateError> {
    let args = TemplateArgs::new(&body, defaults);
    match render_body(formatter, body, defaults) {
        // There is no engine to hand a request the SGLang path declines to, so
        // the formatter's own template renders it, as it did before that path
        // existed; a formatter with no other path declines it again.
        Some(Err(TemplateError::EngineOnly { .. })) | None => formatter
            .render_template(request, &args)
            .map(|(prompt, continuation)| RenderedChat {
                prompt,
                continuation,
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
    // The continuation's leading BOS is dropped once its ids are encoded.
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
    use serde_json::json;
    use sglang_processor::ChatFormatterOptions;

    use super::*;

    /// The formatter a checkpoint with `config` and Jinja `template` selects.
    fn formatter(config: &Value, template: &str) -> ChatFormatter {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("jinja-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        std::fs::write(dir.join("chat_template.jinja"), template).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
        let (formatter, error) = sglang_processor::select_chat_formatter(&ChatFormatterOptions {
            tokenizer_path: dir.to_string_lossy().into_owned(),
            ..Default::default()
        });
        std::fs::remove_dir_all(&dir).unwrap();
        formatter.unwrap_or_else(|| panic!("{error:?}"))
    }

    fn render(formatter: &ChatFormatter, body: Value) -> RenderedChat {
        let request = serde_json::from_value(body.clone()).unwrap();
        render_chat(formatter, body, &request, &HashMap::new()).unwrap()
    }

    /// A DeepSeek-V4.1 developer turn, which SGLang's encoder declines, renders
    /// through Dynamo's, as before that encoder existed.
    #[test]
    fn declined_deepseek_v41_requests_render_through_dynamo() {
        let messages = json!([
            {"role": "developer", "content": "Be brief."},
            {"role": "user", "content": "Hi"}
        ]);
        let body = json!({"model": "m", "messages": messages});
        let formatter = ChatFormatter::DeepSeekV41;
        assert!(matches!(
            formatter.render_request(body.clone(), &HashMap::new()),
            Err(TemplateError::EngineOnly { .. })
        ));
        let prompt = render(&formatter, body).prompt.as_str().to_owned();
        assert!(
            prompt.contains("Be brief.") && prompt.contains("Hi"),
            "{prompt}"
        );
    }

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
        let template = fixture["chat_template"].as_str().unwrap();
        let formatter = formatter(&fixture["config"], template);
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
