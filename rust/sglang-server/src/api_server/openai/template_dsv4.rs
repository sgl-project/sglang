//! Dynamo's V4 encoder with SGLang's message normalization and effort prompts.

use std::collections::BTreeMap;

use dynamo_protocols::types::CreateChatCompletionRequest;
use dynamo_renderer::deepseek::v4::{ThinkingMode, encode_messages_with_options, tokens};
use serde_json::{Value, json};

use super::template::TemplateError;
use crate::utils::environ::env_bool;

pub(super) fn render(
    request: &CreateChatCompletionRequest,
    effort_prompts: &BTreeMap<String, String>,
    thinking: Option<bool>,
) -> Result<String, TemplateError> {
    let mut messages: Vec<Value> = request
        .messages
        .iter()
        .map(serde_json::to_value)
        .collect::<Result<_, _>>()
        .map_err(render_error)?;
    for message in &mut messages {
        // Python flattens text parts with spaces before invoking encoding_dsv4.
        message["content"] = match &message["content"] {
            Value::Array(parts) => Value::String(
                parts
                    .iter()
                    .filter_map(|part| part["text"].as_str())
                    .collect::<Vec<_>>()
                    .join(" "),
            ),
            Value::Null => json!(""),
            content => content.clone(),
        };
    }
    if messages
        .first()
        .is_none_or(|message| message["role"] != "system")
    {
        messages.insert(0, json!({"role": "system", "content": ""}));
    }
    if let Some(tools) = &request.tools {
        // Match Python Tool.model_dump(): field order and defaults affect prompt bytes.
        messages[0]["tools"] = Value::Array(
            tools
                .iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "function": {
                            "description": tool.function.description,
                            "name": tool.function.name,
                            "parameters": tool.function.parameters,
                            "strict": tool.function.strict.unwrap_or(false),
                        }
                    })
                })
                .collect(),
        );
    }

    let thinking = thinking.unwrap_or_else(|| env_bool("SGLANG_DEFAULT_THINKING", false));
    let effort = serde_json::to_value(&request.reasoning_effort).map_err(render_error)?;
    let env_effort = std::env::var("SGLANG_DSV4_REASONING_EFFORT").ok();
    let preamble = effort
        .as_str()
        .or(env_effort.as_deref())
        .filter(|_| thinking)
        .and_then(|effort| effort_prompts.get(effort))
        .map_or("", String::as_str);
    let body = encode_messages_with_options(
        &messages,
        if thinking {
            ThinkingMode::Thinking
        } else {
            ThinkingMode::Chat
        },
        false,
        true,
        // The Python-resolved profile owns effort handling; Dynamo only models preview.
        None,
    )
    .map_err(render_error)?;
    Ok(format!("{}{preamble}{body}", tokens::BOS))
}

fn render_error(error: impl std::fmt::Display) -> TemplateError {
    TemplateError::Renderer {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::openai::load_chat_support;
    use crate::message::config::ServerArgs;

    #[test]
    fn native_encoder_needs_no_template_and_honors_thinking_override() {
        let mut args = ServerArgs {
            tokenizer_path: "/nonexistent/deepseek-v4-tokenizer".into(),
            chat_template: Some("/nonexistent/template.jinja".into()),
            ..Default::default()
        };
        args.model_config.dsv4_effort_prompts = Some(BTreeMap::from([
            ("low".into(), String::new()),
            ("high".into(), "effort\n".into()),
        ]));
        let formatter = load_chat_support(&args).unwrap();
        let request = serde_json::from_value(json!({
            "model": "test", "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "high",
        }))
        .unwrap();
        assert_eq!(
            formatter.render(&request, Some(false)).unwrap(),
            "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"
        );
        assert_eq!(
            formatter.render(&request, Some(true)).unwrap(),
            "<｜begin▁of▁sentence｜>effort\n<｜User｜>Hello<｜Assistant｜><think>"
        );
        args.model_config.dsv4_effort_prompts = None;
        assert!(load_chat_support(&args).is_none());
    }
}
