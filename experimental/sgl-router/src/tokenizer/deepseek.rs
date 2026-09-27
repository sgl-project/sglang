// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SGLang's native DeepSeek serving semantics around Dynamo's V4 encoder.

use anyhow::{bail, ensure, Context, Result};
use dynamo_renderer::deepseek::{
    v4::{self, ReasoningEffort, ThinkingMode},
    v41,
};
use serde_json::{json, Map, Value};

use super::{adapter::ModelFiles, chat_formatter::ChatTemplateKwargs};

#[derive(Clone, Copy)]
pub(super) enum Encoder {
    V4(V4Profile),
    V41,
}

impl Encoder {
    pub fn normalize(self, messages: &mut [Value]) -> Result<()> {
        normalize_messages(messages, matches!(self, Self::V4(_)))
    }

    pub fn render(
        self,
        request: &Value,
        mut messages: Vec<Value>,
        kwargs: &ChatTemplateKwargs,
    ) -> Result<String> {
        // SGLang drops a later user's task when merging it into an existing
        // user/tool-result turn. Dynamo otherwise preserves that field.
        for index in 1..messages.len() {
            if messages[index]["role"] == "user"
                && matches!(messages[index - 1]["role"].as_str(), Some("user" | "tool"))
            {
                messages[index].as_object_mut().unwrap().remove("task");
            }
        }
        match self {
            Self::V4(profile) => profile.render(request, messages, kwargs),
            Self::V41 => render_v41(request, messages, kwargs),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum V4Profile {
    #[default]
    Preview,
    Official,
}

impl V4Profile {
    pub fn load(files: &ModelFiles, config: &Value) -> Result<Self> {
        if let Some(profile) = config
            .get("dsv4_reasoning_effort_profile")
            .filter(|v| !v.is_null())
        {
            return match profile.as_str() {
                Some("preview") => Ok(Self::Preview),
                Some("official") => Ok(Self::Official),
                _ => bail!("invalid dsv4_reasoning_effort_profile: {profile}"),
            };
        }
        // Inspect checkpoint source as data, never execute remote Python. Like
        // SGLang, an absent/unrecognized encoder falls back to the preview profile.
        let source = files.text("encoding/encoding_dsv4.py")?.unwrap_or_default();
        Ok(Self::detect(&source))
    }

    /// `chat_encoding._detect_dsv4_reasoning_effort_profile`: official when the
    /// encoder declares a low default and low/high/max prompts, else preview.
    fn detect(source: &str) -> Self {
        if source.len() > 1 << 20 {
            return Self::Preview;
        }
        let low_default = source.lines().any(|line| {
            line.strip_prefix("DEFAULT_REASONING_EFFORT")
                .and_then(|rest| rest.split_once('='))
                .is_some_and(|(_, value)| matches!(value.trim().trim_matches(['"', '\'']), "low"))
        });
        let prompts = source
            .split_once("\nREASONING_EFFORT_PROMPTS")
            .and_then(|(_, rest)| rest.split_once('}'))
            .map_or("", |(body, _)| body);
        let has_keys = ["low", "high", "max"].iter().all(|key| {
            prompts.contains(&format!("\"{key}\"")) || prompts.contains(&format!("'{key}'"))
        });
        if low_default && has_keys {
            Self::Official
        } else {
            Self::Preview
        }
    }

    pub fn render(
        self,
        request: &Value,
        mut messages: Vec<Value>,
        kwargs: &ChatTemplateKwargs,
    ) -> Result<String> {
        ensure!(
            !messages.is_empty(),
            "DeepSeek requires messages after continuation extraction"
        );
        if messages[0]["role"] != "system" {
            messages.insert(0, json!({"role":"system", "content":""}));
        }
        // Unlike the HF template path, SGLang's native encoder takes *all*
        // request.tools, even for tool_choice=none or a named function.
        if let Some(tools) = request["tools"].as_array().filter(|t| !t.is_empty()) {
            messages[0]["tools"] = normalize_tools(tools)?.into();
        }
        let effort = match (
            self,
            request_effort(request).as_ref().and_then(Value::as_str),
        ) {
            (Self::Official, Some("high")) | (Self::Preview, Some("max")) => {
                Some(ReasoningEffort::High)
            }
            (Self::Official, Some("max")) => Some(ReasoningEffort::Max),
            _ => None,
        };
        // Only thinking is a native-encoder kwarg. In particular, the engine
        // does not read kwargs.reasoning_effort or kwargs.drop_thinking.
        let thinking = kwargs
            .get("thinking")
            .is_some_and(|v| minijinja::Value::from_serialize(v).is_true());
        v4::encode_messages_with_options(
            &messages,
            if thinking {
                ThinkingMode::Thinking
            } else {
                ThinkingMode::Chat
            },
            true,
            true,
            effort,
        )
    }
}

/// `serving_chat._convert_to_internal_request`: `chat_template_kwargs.reasoning_effort`
/// replaces the validated request effort verbatim (thinking was already derived
/// from the request fields); request-level values are coerced as pydantic does.
pub(super) fn request_effort(request: &Value) -> Option<Value> {
    if let Some(effort) = request["chat_template_kwargs"]
        .get("reasoning_effort")
        .filter(|v| !v.is_null())
    {
        return Some(effort.clone());
    }
    let effort = [
        request["reasoning"].get("effort"),
        request["reasoning"].get("reasoning_effort"),
        request.get("reasoning_effort"),
    ]
    .into_iter()
    .flatten()
    .find(|v| !v.is_null())?;
    Some(match effort {
        Value::String(s) => s
            .parse::<f64>()
            .map_or_else(|_| effort.clone(), Value::from),
        Value::Number(n) => n.as_f64().map_or_else(|| effort.clone(), Value::from),
        _ => effort.clone(),
    })
}

/// Before continuation extraction, SGLang flattens V4 parts with spaces and
/// parses assistant arguments as JSON objects. Dynamo expects those arguments
/// serialized, but must not apply its permissive malformed-JSON fallback here.
fn normalize_messages(messages: &mut [Value], flatten_content: bool) -> Result<()> {
    for message in messages {
        // V4.1 keeps parts lists (its encoder joins them), so a parts-list
        // final assistant turn is not a continuation prefix there, as in SGLang.
        if let Some(parts) = message["content"].as_array().filter(|_| flatten_content) {
            message["content"] = parts
                .iter()
                .filter(|p| matches!(p["type"].as_str(), Some("text" | "input_text")))
                .filter_map(|p| p["text"].as_str())
                .collect::<Vec<_>>()
                .join(" ")
                .into();
        }
        if let Some(tools) = message["tools"].as_array() {
            if tools.is_empty() {
                message.as_object_mut().unwrap().remove("tools");
            } else {
                message["tools"] = normalize_tools(tools)?.into();
            }
        }
        if message["role"] == "assistant" {
            if message["tool_calls"].as_array().is_some_and(Vec::is_empty) {
                message.as_object_mut().unwrap().remove("tool_calls");
            }
            for call in message["tool_calls"].as_array_mut().into_iter().flatten() {
                let arguments = &mut call["function"]["arguments"];
                let parsed = match arguments.as_str() {
                    Some(text) => serde_json::from_str::<Value>(text)
                        .context("assistant tool arguments must be valid JSON")?,
                    None => arguments.clone(),
                };
                ensure!(
                    parsed.is_object(),
                    "assistant tool arguments must be a JSON object"
                );
                *arguments = serde_json::to_string(&parsed)?.into();
            }
        }
    }
    Ok(())
}

/// protocol.py::Function.model_dump(), including declared field order and
/// defaults. The native encoder serializes this dictionary verbatim.
fn normalize_tools(tools: &[Value]) -> Result<Vec<Value>> {
    tools
        .iter()
        .map(|tool| {
            let f = &tool["function"];
            let name = f["name"].as_str().context("tool function requires name")?;
            let mut function = Map::new();
            function.insert("description".into(), f["description"].clone());
            function.insert("name".into(), name.into());
            function.insert("parameters".into(), f["parameters"].clone());
            function.insert(
                "strict".into(),
                f.get("strict").cloned().unwrap_or(false.into()),
            );
            if let Some(defer) = f
                .get("defer_loading")
                .filter(|v| !v.is_null())
                .or_else(|| tool.get("defer_loading").filter(|v| !v.is_null()))
            {
                function.insert("defer_loading".into(), defer.clone());
            }
            Ok(json!({"type":"function", "function":function}))
        })
        .collect()
}

/// Use Dynamo's low-level encoder so SGLang, rather than Dynamo's OpenAI
/// defaults, controls tool selection and the numeric reasoning budget.
fn render_v41(
    request: &Value,
    mut messages: Vec<Value>,
    kwargs: &ChatTemplateKwargs,
) -> Result<String> {
    ensure!(
        !messages.is_empty(),
        "DeepSeek requires messages after continuation extraction"
    );
    let thinking = kwargs
        .get("thinking")
        .is_some_and(|v| minijinja::Value::from_serialize(v).is_true());
    // Dynamo maps developer to system, unlike this SGLang encoder. Leave
    // that unsupported shape to the worker instead of forwarding wrong IDs.
    ensure!(
        !messages.iter().any(|m| m["role"] == "developer"),
        "V4.1 developer messages require engine-side rendering"
    );
    if let Some(tools) = request["tools"].as_array().filter(|t| !t.is_empty()) {
        if messages[0]["role"] != "system" {
            messages.insert(0, json!({"role":"system", "content":""}));
        }
        // dsv41_tool_payload: only supplied fields, in the OpenAI field order.
        messages[0]["tools"] = tools
            .iter()
            .map(|tool| {
                let f = &tool["function"];
                let mut function: Map<String, Value> = [
                    "name",
                    "description",
                    "parameters",
                    "strict",
                    "defer_loading",
                ]
                .into_iter()
                .filter_map(|key| {
                    f.get(key)
                        .filter(|v| !v.is_null())
                        .map(|v| (key.into(), v.clone()))
                })
                .collect();
                if !function.contains_key("defer_loading") {
                    if let Some(v) = tool.get("defer_loading").filter(|v| !v.is_null()) {
                        function.insert("defer_loading".into(), v.clone());
                    }
                }
                json!({"type":"function", "function":function})
            })
            .collect();
    }
    // chat_encoding.parse_dsv41_reasoning_effort; unsupported values use the
    // engine default (`high`), matching SGLANG_DSV41_REASONING_EFFORT unset.
    let budget = match request_effort(request) {
        Some(Value::String(s)) => match s.as_str() {
            "low" => Some(25),
            "high" => Some(50),
            "xhigh" => Some(75),
            "max" => Some(100),
            _ => None,
        },
        Some(Value::Number(n)) => match (n.as_i64(), n.as_f64()) {
            (Some(i), _) => u8::try_from(i).ok().filter(|i| (1..=100).contains(i)),
            (None, Some(f)) => (0.0..=0.99)
                .contains(&f)
                .then(|| (f * 100.0).round_ties_even().max(1.0) as u8),
            _ => None,
        },
        _ => None,
    }
    .unwrap_or(50);
    v41::encode_messages(
        &messages,
        if thinking {
            ThinkingMode::Thinking
        } else {
            ThinkingMode::Chat
        },
        true,
        budget,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_profile_and_override() {
        assert_eq!(
            V4Profile::detect("REASONING_EFFORT_MAX = 'text'"),
            V4Profile::Preview
        );
        assert_eq!(V4Profile::detect("DEFAULT_REASONING_EFFORT: str = 'low'\nREASONING_EFFORT_PROMPTS = {\n'low': '', 'high': 'H', 'max': 'M'\n}"), V4Profile::Official);
        assert_eq!(
            V4Profile::detect(
                "# DEFAULT_REASONING_EFFORT = 'low'\n# REASONING_EFFORT_PROMPTS = {}"
            ),
            V4Profile::Preview
        );
        assert_eq!(
            V4Profile::detect("DEFAULT_REASONING_EFFORT = 'low'\nREASONING_EFFORT_PROMPTS = {}"),
            V4Profile::Preview
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, "{}").unwrap();
        let files = ModelFiles::open(path.to_str().unwrap());
        assert_eq!(
            V4Profile::load(&files, &json!({})).unwrap(),
            V4Profile::Preview
        );
        assert_eq!(
            V4Profile::load(&files, &json!({"dsv4_reasoning_effort_profile":"official"})).unwrap(),
            V4Profile::Official
        );
        assert!(V4Profile::load(&files, &json!({"dsv4_reasoning_effort_profile":"typo"})).is_err());
    }
}
