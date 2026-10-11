//! DeepSeek-V4 prompts as SGLang's `serving_chat.py` builds them, on Dynamo's V4 encoder.

use std::collections::HashMap;

use dynamo_renderer::deepseek::v4::{self, ReasoningEffort, ThinkingMode};
use serde::de::IgnoredAny;
use serde_json::{Map, Value, json};

use crate::model_files::resolve_model_file;
use crate::render::reasoning::{requested_effort, requested_thinking};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4Profile {
    Preview,
    Official,
}

/// Render an SGLang chat request body. Returns the prompt and the
/// `continue_final_message` prefix, which SGLang tokenizes separately.
pub(crate) fn render(
    profile: DeepSeekV4Profile,
    mut request: Value,
    defaults: &HashMap<String, Value>,
) -> Result<(String, String), String> {
    let Some(Value::Array(messages)) = request.get_mut("messages").map(Value::take) else {
        return Err("messages must be an array".into());
    };
    let mut messages = messages.into_iter().map(engine_message).collect::<Vec<_>>();
    for message in &mut messages {
        flatten_content(message);
        check_tool_arguments(message)?;
    }
    let request = &request;
    let prefix = split_continuation(&mut messages, request)?;
    // SGLang fails on `messages[0]` here.
    if messages.is_empty() {
        return Err("no messages to render".into());
    }
    if let Some(task) = request.get("task").filter(|task| !task.is_null()) {
        let message = messages
            .iter_mut()
            .rev()
            .find(|m| matches!(m["role"].as_str(), Some("user" | "developer")))
            .ok_or("task requires a user or developer message")?;
        message["task"] = task.clone();
    }
    drop_merged_tasks(&mut messages);
    normalize_messages(&mut messages)?;
    if messages.first().is_none_or(|m| m["role"] != "system") {
        messages.insert(0, json!({"role": "system", "content": ""}));
    }
    // Unlike the Jinja path, SGLang passes every request tool, whatever the tool_choice.
    if let Some(tools) = request["tools"]
        .as_array()
        .filter(|tools| !tools.is_empty())
    {
        messages[0]["tools"] = normalize_tools(tools)?.into();
    }
    let mode = if thinking(request, defaults) {
        ThinkingMode::Thinking
    } else {
        ThinkingMode::Chat
    };
    let fallback = std::env::var("SGLANG_DSV4_REASONING_EFFORT").ok();
    let effort = effort(profile, request, defaults, fallback);
    let prompt = v4::encode_messages_with_options(&messages, mode, true, true, effort)
        .map_err(|error| error.to_string())?;
    Ok((prompt, prefix))
}

/// The `continue_final_message` prefix, taken off a final assistant turn with
/// text content; without the flag that turn renders as a user turn.
pub(super) fn split_continuation(
    messages: &mut Vec<Value>,
    request: &Value,
) -> Result<String, String> {
    let Some(last) = messages.last_mut().filter(|m| m["role"] == "assistant") else {
        return Ok(String::new());
    };
    let Some(content) = last["content"].as_str().map(str::to_owned) else {
        return Ok(String::new());
    };
    if request
        .get("continue_final_message")
        .map_or(Ok(false), pydantic_bool)?
    {
        messages.pop();
        return Ok(content);
    }
    *last = json!({"role": "user", "content": content});
    Ok(String::new())
}

/// `merge_tool_messages` folds a user turn into a preceding user or
/// tool-result turn, dropping the later turn's task.
/// Workaround for ai-dynamo/frontend-crates#366; drop once dynamo-renderer includes it.
pub(super) fn drop_merged_tasks(messages: &mut [Value]) {
    for index in 1..messages.len() {
        if messages[index]["role"] == "user"
            && matches!(messages[index - 1]["role"].as_str(), Some("user" | "tool"))
            && let Some(message) = messages[index].as_object_mut()
        {
            message.remove("task");
        }
    }
}

/// `serving_chat.py`: kwargs `thinking` wins, then what `reasoning` and the
/// effort imply, then the server's default kwargs and `SGLANG_DEFAULT_THINKING`.
pub(crate) fn thinking(request: &Value, defaults: &HashMap<String, Value>) -> bool {
    if let Some(thinking) = request["chat_template_kwargs"].get("thinking") {
        return minijinja::Value::from_serialize(thinking).is_true();
    }
    if let Some(thinking) = requested_thinking(request) {
        return thinking;
    }
    if let Some(thinking) = defaults.get("thinking") {
        return minijinja::Value::from_serialize(thinking).is_true();
    }
    std::env::var("SGLANG_DEFAULT_THINKING")
        .is_ok_and(|v| ["true", "1", "yes", "y"].contains(&v.to_lowercase().as_str()))
}

/// `encoding_dsv4.REASONING_EFFORT_PROFILES`: kwargs effort replaces the request
/// effort, then the server's default kwargs and `fallback`
/// (`SGLANG_DSV4_REASONING_EFFORT`) fill in; other tiers add no prefix.
fn effort(
    profile: DeepSeekV4Profile,
    request: &Value,
    defaults: &HashMap<String, Value>,
    fallback: Option<String>,
) -> Option<ReasoningEffort> {
    let requested = Some(&request["chat_template_kwargs"]["reasoning_effort"])
        .filter(|effort| !effort.is_null())
        .or_else(|| requested_effort(request))
        .or_else(|| {
            defaults
                .get("reasoning_effort")
                .filter(|effort| !effort.is_null())
        })
        .map(|effort| effort.as_str().map(str::to_owned))
        .unwrap_or(fallback);
    match (profile, requested.as_deref()) {
        (DeepSeekV4Profile::Official, Some("high")) | (DeepSeekV4Profile::Preview, Some("max")) => {
            Some(ReasoningEffort::High)
        }
        (DeepSeekV4Profile::Official, Some("max")) => Some(ReasoningEffort::Max),
        _ => None,
    }
}

/// The pydantic dump SGLang renders: roles lowercased, unknown and null fields
/// dropped, `user` reduced to role and content, null content blanked.
pub(super) fn engine_message(message: Value) -> Value {
    let mut message = match message {
        Value::Object(message) => message,
        _ => Map::new(),
    };
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_lowercase();
    let kept: &[&str] = if role == "user" {
        &["content"]
    } else {
        &[
            "content",
            "tool_call_id",
            "name",
            "reasoning_content",
            "tool_calls",
            "tools",
        ]
    };
    message.retain(|key, value| kept.contains(&key.as_str()) && !value.is_null());
    message.insert("role".into(), role.into());
    message.entry("content").or_insert_with(|| "".into());
    message.into()
}

/// `process_content_for_template_format`: text parts joined with spaces.
fn flatten_content(message: &mut Value) {
    if let Some(parts) = message["content"].as_array() {
        message["content"] = parts
            .iter()
            .filter(|part| matches!(part["type"].as_str(), Some("text" | "input_text")))
            .filter_map(|part| part["text"].as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .into();
    }
}

/// `serving_chat.normalize_assistant_tool_call_arguments`: string arguments must
/// be a JSON object; other values wait for the final-turn handling.
pub(super) fn check_tool_arguments(message: &mut Value) -> Result<(), String> {
    for arguments in tool_arguments(message) {
        if let Some(text) = arguments.as_str()
            && (serde_json::from_str::<IgnoredAny>(text).is_err()
                || !text.trim_start().starts_with('{'))
        {
            return Err("assistant tool arguments must be a JSON object".into());
        }
    }
    Ok(())
}

/// The messages the encoder sees: empty tool lists dropped, message tools dumped,
/// and tool arguments as JSON object text, which Dynamo parses itself.
pub(super) fn normalize_messages(messages: &mut [Value]) -> Result<(), String> {
    for message in messages {
        for arguments in tool_arguments(message) {
            match arguments {
                Value::String(_) => {}
                Value::Object(_) => *arguments = arguments.to_string().into(),
                _ => return Err("assistant tool arguments must be a JSON object".into()),
            }
        }
        let message = message.as_object_mut().ok_or("message must be an object")?;
        // Empty lists: workaround for ai-dynamo/frontend-crates#366; drop once
        // dynamo-renderer includes it.
        match message.get("tools").and_then(Value::as_array) {
            Some(tools) if tools.is_empty() => _ = message.remove("tools"),
            Some(tools) => _ = message.insert("tools".into(), normalize_tools(tools)?.into()),
            None => {}
        }
        if message
            .get("tool_calls")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
        {
            message.remove("tool_calls");
        }
    }
    Ok(())
}

/// Each assistant tool call's `function.arguments`, absent ones as `null`.
fn tool_arguments(message: &mut Value) -> impl Iterator<Item = &mut Value> {
    let is_assistant = message["role"] == "assistant";
    message
        .get_mut("tool_calls")
        .filter(|_| is_assistant)
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
        .filter_map(|call| call.get_mut("function")?.as_object_mut())
        .map(|function| function.entry("arguments").or_insert(Value::Null))
}

/// `protocol.py::Function.model_dump()`, in declared field order; the encoder
/// serializes it verbatim.
fn normalize_tools(tools: &[Value]) -> Result<Vec<Value>, String> {
    tools
        .iter()
        .map(|tool| {
            let f = &tool["function"];
            let name = f["name"].as_str().ok_or("tool function requires name")?;
            let mut function = Map::new();
            function.insert("description".into(), f["description"].clone());
            function.insert("name".into(), name.into());
            function.insert("parameters".into(), f["parameters"].clone());
            let strict = f.get("strict").map_or(Ok(false), pydantic_bool)?;
            function.insert("strict".into(), strict.into());
            if let Some(defer) = [f.get("defer_loading"), tool.get("defer_loading")]
                .into_iter()
                .flatten()
                .find(|v| !v.is_null())
            {
                function.insert("defer_loading".into(), pydantic_bool(defer)?.into());
            }
            Ok(json!({"type": "function", "function": function}))
        })
        .collect()
}

/// pydantic's lax `bool`: booleans, 0 and 1, and the yes/no words in any case.
pub(super) fn pydantic_bool(value: &Value) -> Result<bool, String> {
    let parsed = match value {
        Value::Bool(value) => Some(*value),
        Value::Number(number) => match number.as_f64() {
            Some(0.0) => Some(false),
            Some(1.0) => Some(true),
            _ => None,
        },
        Value::String(text) => match text.to_lowercase().as_str() {
            "0" | "off" | "f" | "false" | "n" | "no" => Some(false),
            "1" | "on" | "t" | "true" | "y" | "yes" => Some(true),
            _ => None,
        },
        _ => None,
    };
    parsed.ok_or_else(|| format!("expected a boolean, got {value}"))
}

impl DeepSeekV4Profile {
    /// `chat_encoding.py`: the config override, else the checkpoint encoder's
    /// declarations (`encoding/encoding_dsv4.py` source); preview when absent.
    pub fn from_checkpoint(profile: Option<&str>, encoder: Option<&str>) -> Result<Self, String> {
        if let Some(profile) = profile {
            return match profile {
                "preview" => Ok(Self::Preview),
                "official" => Ok(Self::Official),
                _ => Err(format!(
                    "invalid dsv4_reasoning_effort_profile: {profile:?}; expected \"preview\" or \"official\""
                )),
            };
        }
        let Some(source) = encoder.filter(|source| source.len() <= 1 << 20) else {
            return Ok(Self::Preview);
        };
        let default = top_level_python_assignment(source, "DEFAULT_REASONING_EFFORT")
            .and_then(python_string_literal);
        let prompt_keys = top_level_python_assignment(source, "REASONING_EFFORT_PROMPTS")
            .and_then(python_dict_keys)
            .unwrap_or_default();
        if default.as_deref() == Some("low")
            && ["low", "high", "max"]
                .iter()
                .all(|key| prompt_keys.iter().any(|candidate| candidate == key))
        {
            Ok(Self::Official)
        } else {
            Ok(Self::Preview)
        }
    }
}

pub(crate) fn resolve_dsv4_profile(
    profile: Option<&str>,
    model_source: &str,
    revision: Option<&str>,
) -> Result<DeepSeekV4Profile, String> {
    let encoder = profile
        .is_none()
        .then(|| resolve_model_file(model_source, revision, "encoding/encoding_dsv4.py"))
        .flatten()
        .and_then(|encoder| std::fs::read_to_string(encoder).ok());
    DeepSeekV4Profile::from_checkpoint(profile, encoder.as_deref())
}

fn top_level_python_assignment<'a>(source: &'a str, name: &str) -> Option<&'a str> {
    let mut offset = 0;
    for line in source.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if !trimmed.starts_with(char::is_whitespace)
            && let Some((target, _)) = trimmed.split_once('=')
            && target
                .split(':')
                .next()
                .is_some_and(|target| target.trim() == name)
        {
            let equals = line.find('=')?;
            return Some(&source[offset + equals + 1..]);
        }
        offset += line.len();
    }
    None
}

fn python_string_literal(source: &str) -> Option<String> {
    let source = source.trim_start();
    let quote = source.chars().next()?;
    if !matches!(quote, '\'' | '"') {
        return None;
    }
    let mut escaped = false;
    let mut value = String::new();
    for character in source[quote.len_utf8()..].chars() {
        if escaped {
            value.push(character);
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else if character == quote {
            return Some(value);
        } else {
            value.push(character);
        }
    }
    None
}

fn python_dict_keys(source: &str) -> Option<Vec<String>> {
    let source = source.trim_start();
    if !source.starts_with('{') {
        return None;
    }
    let mut keys = Vec::new();
    let mut depth = 0usize;
    let mut index = 0usize;
    let bytes = source.as_bytes();
    while index < bytes.len() {
        match bytes[index] {
            b'{' | b'[' | b'(' => {
                depth += 1;
                index += 1;
            }
            b'}' | b']' | b')' => {
                depth = depth.checked_sub(1)?;
                index += 1;
                if depth == 0 {
                    return Some(keys);
                }
            }
            quote @ (b'\'' | b'"') => {
                let start = index + 1;
                index = start;
                let mut escaped = false;
                while index < bytes.len() {
                    if escaped {
                        escaped = false;
                    } else if bytes[index] == b'\\' {
                        escaped = true;
                    } else if bytes[index] == quote {
                        break;
                    }
                    index += 1;
                }
                if index == bytes.len() {
                    return None;
                }
                let value = std::str::from_utf8(&bytes[start..index]).ok()?;
                index += 1;
                if depth == 1 {
                    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
                        index += 1;
                    }
                    if bytes.get(index) == Some(&b':') {
                        keys.push(value.to_owned());
                    }
                }
            }
            b'#' => {
                while index < bytes.len() && bytes[index] != b'\n' {
                    index += 1;
                }
            }
            _ => index += 1,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use dynamo_renderer::deepseek::v4::ReasoningEffort::{High, Max};
    use serde_json::json;

    use super::DeepSeekV4Profile::{Official, Preview};
    use super::{DeepSeekV4Profile, effort, resolve_dsv4_profile};

    #[test]
    fn effort_maps_profile_tiers_and_env_fills_only_a_missing_effort() {
        for (profile, requested, fallback, expected) in [
            (Preview, Some("max"), None, Some(High)),
            (Preview, Some("high"), None, None),
            (Official, Some("high"), None, Some(High)),
            (Official, Some("max"), None, Some(Max)),
            (Official, Some("xhigh"), None, None),
            (Official, None, Some("max"), Some(Max)),
            (Official, Some("low"), Some("max"), None),
        ] {
            let request = json!({ "reasoning_effort": requested });
            let fallback = fallback.map(str::to_owned);
            assert_eq!(
                effort(profile, &request, &Default::default(), fallback),
                expected,
                "{profile:?} {requested:?}"
            );
        }
    }

    #[test]
    fn deepseek_v4_profile_resolution_uses_override_then_checkpoint_source() {
        let directory = std::env::temp_dir().join(format!(
            "sglang-processor-deepseek-v4-profile-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(directory.join("encoding")).unwrap();
        std::fs::write(
            directory.join("encoding/encoding_dsv4.py"),
            "DEFAULT_REASONING_EFFORT: str = 'low'\n\
REASONING_EFFORT_PROMPTS = {'low': '', 'high': 'absolute', 'max': 'beyond'}",
        )
        .unwrap();
        let source = directory.to_string_lossy();
        assert_eq!(
            resolve_dsv4_profile(None, &source, None).unwrap(),
            DeepSeekV4Profile::Official
        );
        assert_eq!(
            resolve_dsv4_profile(Some("preview"), &source, None).unwrap(),
            DeepSeekV4Profile::Preview
        );
        assert!(resolve_dsv4_profile(Some("future"), &source, None).is_err());

        std::fs::write(
            directory.join("encoding/encoding_dsv4.py"),
            r#"DEFAULT_REASONING_EFFORT = "high"
REASONING_EFFORT_PROMPTS = {"low": "", "high": "absolute", "max": "Beyond maximum"}"#,
        )
        .unwrap();
        assert_eq!(
            resolve_dsv4_profile(None, &source, None).unwrap(),
            DeepSeekV4Profile::Preview
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
