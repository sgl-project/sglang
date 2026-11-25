//! The text-only dsv32 branch of OpenAIServingChat._apply_jinja_template.
use std::collections::HashMap;

use dynamo_renderer::deepseek::v32::{ThinkingMode, encode_messages};
use serde_json::{Value, json};

use super::{continues_final_message, deepseek_v4_thinking, split_continuation};
use crate::render::{RenderError, template::text_messages};

pub(crate) fn render(
    body: Value,
    defaults: &HashMap<String, Value>,
) -> Result<(String, String), RenderError> {
    let mut messages = text_messages(&body, Some(false))?;
    let prefix = split_continuation(&mut messages, continues_final_message(&body)?);
    let first = messages.first().ok_or("no messages to render".to_owned())?;
    if first["role"] != "system" {
        messages.insert(0, json!({"role":"system","content":""}));
    }
    let mode = if deepseek_v4_thinking(&body, defaults) {
        // Python's encoder requires reasoning on assistant turns after the last user.
        let after_user = messages
            .iter()
            .rposition(|m| m["role"] == "user")
            .map_or(0, |i| i + 1);
        if messages[after_user..].iter().any(|m| {
            m["role"] == "assistant" && m["reasoning_content"].as_str().unwrap_or("").is_empty()
        }) {
            return Err("missing DeepSeek assistant reasoning".into());
        }
        ThinkingMode::Thinking
    } else {
        ThinkingMode::Chat
    };
    let prompt = encode_messages(&messages, mode, true).map_err(|e| e.to_string())?;
    Ok((prompt, prefix))
}
