// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Chat rendering via dynamo-render for cache-aware routing and input ID forwarding.
//!
//! Mirrors SGLang reasoning controls, assistant continuations, and DeepSeek-V4
//! task selection before rendering. Forwarding remains guarded until parity
//! has been verified for each request shape.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use dynamo_renderer::{
    deepseek_formatter_for, may_be_fix_tool_schema, ChatTemplate, ContextMixins,
    OAIChatLikeRequest, OAIPromptFormatter, PromptFormatter,
};
use minijinja::Value;
use serde_json::Value as JsonValue;

pub type ChatTemplateKwargs = HashMap<String, JsonValue>;

// HF special_tokens_map names. dynamo-render supplies bos/eos/unk; the rest
// are passed as template context defaults.
const SPECIAL_TOKEN_KEYS: [&str; 7] = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
];

/// Renders and tokenizes chat requests through dynamo-render.
pub struct ChatFormatter {
    formatter: Arc<dyn OAIPromptFormatter>,
    /// Template context defaults; request `chat_template_kwargs` override them.
    defaults: ChatTemplateKwargs,
    /// Stripped from a separately tokenized continuation prefix, as SGLang does.
    bos_token: Option<String>,
    is_deepseek_v4: bool,
}

impl ChatFormatter {
    /// Load model files and select a template or native formatter from dynamo-render.
    pub fn load(model_id: &str, tokenizer_path: &str) -> Result<Option<Self>> {
        let files = super::adapter::ModelFiles::open(tokenizer_path);
        let model_type = files
            .json("config.json")?
            .and_then(|cfg| cfg["model_type"].as_str().map(str::to_owned));
        match model_type.as_deref() {
            // These require tokenization paths not yet supported by this adapter.
            Some("inkling_mm_model" | "kimi_k3") => return Ok(None),
            Some(t) if t.starts_with("deepseek_v4") => {
                return Ok(Self::deepseek_native(model_type.as_deref(), model_id));
            }
            _ => {}
        }
        let cfg = files
            .json("tokenizer_config.json")?
            .unwrap_or_else(|| serde_json::json!({}));
        let jinja = files.text("chat_template.jinja")?;
        Ok(Self::from_tokenizer_config(cfg, jinja.as_deref())?
            .or_else(|| Self::deepseek_native(model_type.as_deref(), model_id)))
    }

    /// HF Jinja template from `tokenizer_config.json`, overridden by a sibling
    /// `chat_template.jinja` when present (transformers' precedence); `Ok(None)`
    /// when the model ships neither.
    pub fn from_tokenizer_config(
        mut cfg: JsonValue,
        chat_template_jinja: Option<&str>,
    ) -> Result<Option<Self>> {
        let config = cfg
            .as_object_mut()
            .context("tokenizer_config.json must be an object")?;
        // Tokenizer-only settings have types the renderer's config does not support.
        config.retain(|key, _| {
            key == "chat_template"
                || SPECIAL_TOKEN_KEYS.contains(&key.as_str())
                || key == "additional_special_tokens"
        });
        if let Some(template) = chat_template_jinja {
            cfg["chat_template"] = template.into();
        }
        // HuggingFace supplies None when no retrieval documents are present.
        let mut defaults = HashMap::from([("documents".into(), JsonValue::Null)]);
        for key in SPECIAL_TOKEN_KEYS {
            // Convert HF AddedToken objects to strings for dynamo-render.
            if let Some(content) = added_token_content(&cfg[key]) {
                cfg[key] = content.into();
            }
            if matches!(key, "bos_token" | "eos_token" | "unk_token") {
                // Missing tokens render as `None` in dynamo-render; HF uses "".
                // These config values take precedence over kwargs.
                if cfg[key].is_null() {
                    cfg[key] = "".into();
                }
            } else {
                // Never reach the template except through kwargs.
                defaults.insert(key.to_owned(), cfg[key].as_str().unwrap_or_default().into());
            }
        }
        if let Some(extra) = cfg["additional_special_tokens"].as_array() {
            let extra: Vec<JsonValue> = extra
                .iter()
                .map(|t| added_token_content(t).map_or_else(|| t.clone(), Into::into))
                .collect();
            cfg["additional_special_tokens"] = extra.clone().into();
            defaults.insert("additional_special_tokens".into(), extra.into());
        }
        // HF's `[{name, template}]` list form -> dynamo-render's `[{name: template}]`.
        for entry in cfg["chat_template"].as_array_mut().into_iter().flatten() {
            if let (Some(name), Some(template)) =
                (entry["name"].as_str(), entry["template"].as_str())
            {
                *entry = serde_json::json!({ name: template });
            }
        }
        let template: ChatTemplate =
            serde_json::from_value(cfg).context("parse tokenizer_config.json")?;
        if template.chat_template.is_none() {
            return Ok(None);
        }
        let bos_token = template.bos_tok();
        let PromptFormatter::OAI(formatter) =
            PromptFormatter::from_parts(template, ContextMixins::default(), true)
                .context("compile chat template")?;
        Ok(Some(Self {
            formatter,
            defaults,
            bos_token,
            is_deepseek_v4: false,
        }))
    }

    /// dynamo-render's code-based DeepSeek encoders, for V4 (including variants
    /// such as V4.1) and V3.2 non-Exp: the only built-in formatters verified
    /// against the engine. `model_type` (from `config.json`) is authoritative;
    /// the model id's last path segment is the fallback.
    pub fn deepseek_native(model_type: Option<&str>, model_id: &str) -> Option<Self> {
        let name = model_id
            .rsplit('/')
            .next()
            .unwrap_or(model_id)
            .to_lowercase();
        // The engine treats every `deepseek_v4*` variant (e.g. V4.1) as V4.
        let model_type = model_type.map(str::to_lowercase).map(|t| {
            if t.starts_with("deepseek_v4") {
                "deepseek_v4".into()
            } else {
                t
            }
        });
        let PromptFormatter::OAI(formatter) = deepseek_formatter_for(&model_type, &name)?;
        // Same rule dynamo-render applies for the name fallback: `deepseek` + one
        // separator + a `v4` segment.
        let version = name.strip_prefix("deepseek").unwrap_or("");
        let version = version.strip_prefix(['-', '_', '.']).unwrap_or(version);
        let is_deepseek_v4 = model_type
            .as_deref()
            .map_or(version.split(['-', '_', '.']).next() == Some("v4"), |t| {
                t == "deepseek_v4"
            });
        // Engine defaults: chat mode (`SGLANG_DEFAULT_THINKING=false`) and no
        // reasoning-effort preamble; dynamo-render defaults to thinking at high effort.
        let defaults = HashMap::from([
            ("thinking".into(), false.into()),
            ("reasoning_effort".into(), "low".into()),
        ]);
        Some(Self {
            formatter,
            defaults,
            bos_token: Some("<｜begin▁of▁sentence｜>".into()),
            is_deepseek_v4,
        })
    }

    /// Request kwargs plus the thinking/effort defaults SGLang derives from
    /// `reasoning` / `reasoning_effort` (`protocol.py::normalize_reasoning_inputs`).
    fn template_kwargs(&self, request: &JsonValue) -> Result<ChatTemplateKwargs> {
        let mut kwargs: ChatTemplateKwargs = match request.get("chat_template_kwargs") {
            None | Some(JsonValue::Null) => ChatTemplateKwargs::new(),
            Some(v) => serde_json::from_value(v.clone()).context("chat_template_kwargs")?,
        };
        // `reasoning.effort` overrides top-level `reasoning_effort`; `enabled`
        // alone turns thinking on; any effort decides thinking by `!= "none"`.
        // Explicit kwargs keep their values (the engine uses setdefault).
        let reasoning = &request["reasoning"];
        let mut thinking = None;
        if reasoning.is_object() {
            let enabled = match reasoning
                .get("enabled")
                .filter(|v| !v.is_null())
                .or_else(|| reasoning.get("enable"))
            {
                Some(JsonValue::Bool(enabled)) => *enabled,
                Some(JsonValue::String(s)) => {
                    ["1", "true", "yes", "y", "on"].contains(&s.trim().to_lowercase().as_str())
                }
                _ => false,
            };
            if enabled {
                thinking = Some(true);
            }
        }
        let effort = [
            reasoning.get("effort"),
            reasoning.get("reasoning_effort"),
            request.get("reasoning_effort"),
        ]
        .into_iter()
        .flatten()
        .find(|v| !v.is_null())
        .cloned();
        if let Some(effort) = &effort {
            thinking = Some(effort != "none");
        }
        if let Some(thinking) = thinking {
            kwargs.entry("thinking".into()).or_insert(thinking.into());
            kwargs
                .entry("enable_thinking".into())
                .or_insert(thinking.into());
        }
        if let Some(mut effort) = effort {
            // The engine's official V4 profile accepts only these; others map to
            // no preamble.
            if self.is_deepseek_v4 && !matches!(effort.as_str(), Some("low" | "high" | "max")) {
                effort = "low".into();
            }
            kwargs.entry("reasoning_effort".into()).or_insert(effort);
        }
        for (key, value) in &self.defaults {
            kwargs.entry(key.clone()).or_insert_with(|| value.clone());
        }
        Ok(kwargs)
    }

    /// Rendered prompt plus the assistant continuation prefix SGLang tokenizes
    /// separately (`_handle_last_assistant_message`).
    fn render_parts(&self, request: &JsonValue) -> Result<(String, String)> {
        let kwargs = self.template_kwargs(request)?;
        let continuing = request["continue_final_message"] == true;
        let mut messages: Vec<JsonValue> = request["messages"]
            .as_array()
            .context("messages must be an array")?
            .iter()
            .map(engine_message)
            .collect();
        let mut prefix = String::new();
        if let Some(last) = messages.last_mut().filter(|m| m["role"] == "assistant") {
            if let Some(content) = last["content"].as_str() {
                if continuing {
                    prefix = content.to_owned();
                    messages.pop();
                } else {
                    *last = serde_json::json!({"role": "user", "content": content});
                }
            }
        }
        if self.is_deepseek_v4 {
            if let Some(task) = request.get("task").filter(|v| !v.is_null()).cloned() {
                let message = messages
                    .iter_mut()
                    .rev()
                    .find(|m| matches!(m["role"].as_str(), Some("user" | "developer")))
                    .context("task requires a user or developer message")?;
                message["task"] = task;
            }
        }
        let prompt = self
            .formatter
            .render(&ChatRequest {
                request,
                messages,
                kwargs,
            })
            .context("render chat template")?;
        Ok((prompt, prefix))
    }

    pub fn encode(
        &self,
        tokenizer: &dynamo_tokenizers::Tokenizer,
        request: &JsonValue,
    ) -> Result<Vec<u32>> {
        let (prompt, prefix) = self.render_parts(request)?;
        let mut ids = super::adapter::encode(tokenizer, &prompt)?;
        if !prefix.is_empty() {
            // SGLang encodes the assistant prefix separately and removes its leading BOS.
            let mut suffix = super::adapter::encode(tokenizer, &prefix)?;
            if let Some(bos) = self.bos_token.as_deref().filter(|s| !s.is_empty()) {
                let bos = super::adapter::encode(tokenizer, bos)?;
                if bos.len() == 1 && suffix.first() == bos.first() {
                    suffix.remove(0);
                }
            }
            ids.extend(suffix);
        }
        Ok(ids)
    }

    /// Use `encode` for token ids to preserve continuation boundaries.
    pub fn render(&self, request: &JsonValue) -> Result<String> {
        let (prompt, prefix) = self.render_parts(request)?;
        Ok(prompt + &prefix)
    }
}

/// Message fields the engine's request schema keeps for non-user roles.
const GENERIC_MESSAGE_KEYS: [&str; 7] = [
    "role",
    "content",
    "tool_call_id",
    "name",
    "reasoning_content",
    "tool_calls",
    "tools",
];

/// Normalize a message as SGLang's pydantic dump does before rendering: roles
/// lowercased, unknown and null fields dropped, `user` reduced to role and
/// content, null content blanked.
fn engine_message(message: &JsonValue) -> JsonValue {
    let role = message["role"].as_str().unwrap_or_default().to_lowercase();
    let mut out = serde_json::Map::new();
    if role != "user" {
        for key in GENERIC_MESSAGE_KEYS {
            if let Some(v) = message.get(key).filter(|v| !v.is_null()) {
                out.insert(key.into(), v.clone());
            }
        }
    }
    let content = match &message["content"] {
        JsonValue::Null => "".into(),
        content => content.clone(),
    };
    out.insert("role".into(), role.into());
    out.insert("content".into(), content);
    out.into()
}

/// `content` of an HF `AddedToken` object (`{"content": "<s>", "lstrip": ...}`).
fn added_token_content(token: &JsonValue) -> Option<String> {
    token
        .as_object()?
        .get("content")?
        .as_str()
        .map(str::to_owned)
}

struct ChatRequest<'a> {
    request: &'a JsonValue,
    /// Normalized copy of `request["messages"]`.
    messages: Vec<JsonValue>,
    kwargs: ChatTemplateKwargs,
}

impl OAIChatLikeRequest for ChatRequest<'_> {
    fn model(&self) -> String {
        self.request["model"]
            .as_str()
            .unwrap_or_default()
            .to_owned()
    }
    fn messages(&self) -> Value {
        Value::from_serialize(&self.messages)
    }
    fn tools(&self) -> Option<Value> {
        let tools = self.request.get("tools")?;
        // HF and the engine treat an empty list as "no tools"; dynamo-render's schema
        // fixer would hand the template `[]`, which tools-branching templates
        // render as a tool preamble.
        if tools.as_array().is_none_or(|t| t.is_empty()) {
            return None;
        }
        let mut tools = tools.clone();
        // SGLang renders only the named tool for a function `tool_choice`.
        if let Some(name) = self.request["tool_choice"]["function"]["name"].as_str() {
            tools
                .as_array_mut()?
                .retain(|tool| tool["function"]["name"] == name);
        }
        may_be_fix_tool_schema(tools)
    }
    fn tool_choice(&self) -> Option<Value> {
        self.request.get("tool_choice").map(Value::from_serialize)
    }
    fn reasoning_effort(&self) -> Option<Value> {
        self.kwargs
            .get("reasoning_effort")
            .map(Value::from_serialize)
    }
    /// Withheld: the engine enforces `response_format` by constrained decoding
    /// and never renders it, while dynamo-render's DeepSeek formatters would
    /// append a "## Response Format" schema preamble to the system turn.
    fn response_format(&self) -> Option<Value> {
        None
    }
    fn should_add_generation_prompt(&self) -> bool {
        true
    }
    fn chat_template_args(&self) -> Option<&ChatTemplateKwargs> {
        Some(&self.kwargs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SIMPLE_TEMPLATE: &str = "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}";

    fn jinja(cfg: JsonValue) -> ChatFormatter {
        ChatFormatter::from_tokenizer_config(cfg, None)
            .unwrap()
            .expect("config has a chat_template")
    }

    fn request(messages: JsonValue) -> JsonValue {
        json!({"model": "m", "messages": messages})
    }

    fn deepseek_v4() -> ChatFormatter {
        ChatFormatter::deepseek_native(Some("deepseek_v4"), "any").unwrap()
    }

    #[test]
    fn no_chat_template_returns_none() {
        let cfg = json!({"bos_token": "<s>", "eos_token": "</s>"});
        assert!(ChatFormatter::from_tokenizer_config(cfg, None)
            .unwrap()
            .is_none());
    }

    /// A sibling `chat_template.jinja` wins over `tokenizer_config.json`, as in
    /// transformers, and suffices on its own.
    #[test]
    fn chat_template_jinja_file_takes_precedence() {
        let cfg = json!({"chat_template": "CONFIG"});
        let enc = ChatFormatter::from_tokenizer_config(cfg, Some("FILE"))
            .unwrap()
            .unwrap();
        assert_eq!(enc.render(&request(json!([]))).unwrap(), "FILE");
        let enc = ChatFormatter::from_tokenizer_config(json!({}), Some("FILE"))
            .unwrap()
            .unwrap();
        assert_eq!(enc.render(&request(json!([]))).unwrap(), "FILE");
    }

    /// Both the string and the `AddedToken` object forms are accepted.
    #[test]
    fn additional_special_tokens_are_supplied() {
        let enc = jinja(json!({
            "chat_template": "{{ additional_special_tokens | join(',') }}",
            "additional_special_tokens": ["<a>", {"content": "<b>", "special": true}]
        }));
        assert_eq!(enc.render(&request(json!([]))).unwrap(), "<a>,<b>");
    }

    #[test]
    fn renders_roles_bos_and_generation_prompt() {
        let enc = jinja(json!({"chat_template": SIMPLE_TEMPLATE, "bos_token": "<s>"}));
        let out = enc
            .render(&request(json!([
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"}
            ])))
            .unwrap();
        assert_eq!(
            out,
            "<s><|system|>\nbe brief<|end|>\n<|user|>\nhi<|end|>\n<|assistant|>\n"
        );
    }

    /// The list form `[{name, template}, ...]` and the `AddedToken` object form
    /// of `bos_token` are both accepted.
    #[test]
    fn list_form_and_bos_object_form() {
        let enc = jinja(json!({
            "chat_template": [
                {"name": "tool_use", "template": "TOOLS"},
                {"name": "default", "template": "{{ bos_token }}X"},
            ],
            "bos_token": {"content": "<|begin|>", "lstrip": false, "normalized": false,
                          "rstrip": false, "single_word": false, "special": true},
        }));
        assert_eq!(enc.render(&request(json!([]))).unwrap(), "<|begin|>X");
    }

    #[test]
    fn absent_documents_match_huggingface_default() {
        let enc = jinja(json!({
            "chat_template": "{% if documents is not none %}DOCS{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}"
        }));
        let mut req = request(json!([{"role":"user","content":"hi"}]));
        assert_eq!(enc.render(&req).unwrap(), "hi");
        req["chat_template_kwargs"] = json!({"documents": [{"text": "reference"}]});
        assert_eq!(enc.render(&req).unwrap(), "DOCShi");
    }

    #[test]
    fn absent_special_tokens_render_empty() {
        let enc = jinja(json!({
            "chat_template": "A{{ bos_token }}{{ eos_token }}{{ unk_token }}{{ sep_token }}{{ pad_token }}{{ cls_token }}{{ mask_token }}B"
        }));
        assert_eq!(enc.render(&request(json!([]))).unwrap(), "AB");
    }

    #[test]
    fn all_special_tokens_from_config_are_supplied() {
        let enc = jinja(json!({
            "chat_template": "{{ bos_token }}{{ eos_token }}{{ unk_token }}{{ sep_token }}{{ pad_token }}{{ cls_token }}{{ mask_token }}",
            "bos_token": {"content": "<s>"},
            "eos_token": "</s>",
            "unk_token": {"content": "<unk>"},
            "sep_token": "<sep>",
            "pad_token": {"content": "<pad>"},
            "cls_token": "<cls>",
            "mask_token": "<mask>"
        }));
        assert_eq!(
            enc.render(&request(json!([]))).unwrap(),
            "<s></s><unk><sep><pad><cls><mask>"
        );
    }

    /// An unrecognized token shape is a load error, not a silent "".
    #[test]
    fn malformed_special_token_fails_to_load() {
        let cfg = json!({"chat_template": "X", "eos_token": ["</s>"]});
        assert!(ChatFormatter::from_tokenizer_config(cfg, None).is_err());
    }

    #[test]
    fn tokenizer_settings_do_not_affect_chat_rendering() {
        let enc = jinja(json!({
            "chat_template": "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}",
            "bos_token": "<s>",
            "sp_model_kwargs": {"enable_sampling": false, "nbest_size": -1, "alpha": 0.1},
            "added_tokens_decoder": {"0": {"content": "<s>"}},
            "truncation_size": 4096
        }));
        assert_eq!(
            enc.render(&request(json!([{"role": "user", "content": "hi"}])))
                .unwrap(),
            "<s>hi"
        );
    }

    #[test]
    fn chat_template_kwargs_reach_the_template() {
        let enc = jinja(json!({"chat_template": "t={{ enable_thinking }}"}));
        let mut req = request(json!([]));
        assert_eq!(enc.render(&req).unwrap(), "t=");
        req["chat_template_kwargs"] = json!({"enable_thinking": true});
        assert_eq!(enc.render(&req).unwrap(), "t=True");
        req["chat_template_kwargs"] = json!("not an object");
        assert!(enc.render(&req).is_err());
    }

    /// Tools pass through to the template; an empty list counts as no tools.
    #[test]
    fn tools_reach_the_template_and_empty_means_none() {
        let enc = jinja(json!({
            "chat_template": "{% if tools is not none %}T:{{ tools | length }}{% endif %}X"
        }));
        let mut req = request(json!([]));
        assert_eq!(enc.render(&req).unwrap(), "X");
        req["tools"] = json!([]);
        assert_eq!(enc.render(&req).unwrap(), "X");
        req["tools"] = json!([{"type": "function", "function": {"name": "f"}}]);
        assert_eq!(enc.render(&req).unwrap(), "T:1X");
    }

    #[test]
    fn raise_exception_surfaces_as_error() {
        let enc = jinja(json!({"chat_template": "{{ raise_exception('bad messages') }}"}));
        let err = enc.render(&request(json!([]))).unwrap_err();
        assert!(format!("{err:#}").contains("bad messages"), "got: {err:#}");
    }

    #[test]
    fn missing_messages_is_an_error() {
        assert!(deepseek_v4().render(&json!({"model": "m"})).is_err());
    }

    #[test]
    fn deepseek_native_detection() {
        assert!(ChatFormatter::deepseek_native(None, "deepseek-ai/DeepSeek-V4-Flash").is_some());
        assert!(ChatFormatter::deepseek_native(None, "deepseek-v4-tiny").is_some());
        assert!(ChatFormatter::deepseek_native(Some("deepseek_v4"), "alias").is_some());
        assert!(ChatFormatter::deepseek_native(Some("deepseek_v41"), "alias").is_some());
        assert!(ChatFormatter::deepseek_native(Some("deepseek_v32"), "DeepSeek-V3.2").is_some());
        assert!(ChatFormatter::deepseek_native(Some("inkling_mm_model"), "inkling").is_none());
        assert!(ChatFormatter::deepseek_native(Some("llama"), "deepseek-v4").is_none());
        assert!(ChatFormatter::deepseek_native(None, "deepseek-ai/DeepSeek-V3.2-Exp").is_none());
        assert!(ChatFormatter::deepseek_native(None, "Qwen/Qwen3-0.6B").is_none());
    }

    /// Byte-exact against the engine's `/tokenize` in its default chat mode:
    /// `[{user:"ABCD"}]` -> `[0, 128803, 51453, 128804, 128822]`.
    #[test]
    fn v4_single_user_turn() {
        let out = deepseek_v4()
            .render(&request(json!([{"role":"user","content":"ABCD"}])))
            .unwrap();
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>ABCD<｜Assistant｜></think>"
        );
    }

    #[test]
    fn v4_system_then_multi_turn() {
        let out = deepseek_v4()
            .render(&request(json!([
                {"role":"system","content":"SYS"},
                {"role":"user","content":"U1"},
                {"role":"assistant","content":"A1"},
                {"role":"user","content":"U2"}
            ])))
            .unwrap();
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜>SYS<｜User｜>U1<｜Assistant｜></think>A1<｜end▁of▁sentence｜><｜User｜>U2<｜Assistant｜></think>"
        );
    }

    /// The engine's V4 encoder reads only `chat_template_kwargs.thinking`
    /// (`serving_chat.py`); `enable_thinking` alone leaves chat mode, while
    /// `reasoning_effort` sets both keys through the normalization above.
    #[test]
    fn v4_thinking_kwarg_overrides_chat_default() {
        let mut req = request(json!([{"role":"user","content":"ABCD"}]));
        req["chat_template_kwargs"] = json!({"thinking": true});
        let out = deepseek_v4().render(&req).unwrap();
        assert!(out.ends_with("<｜Assistant｜><think>"), "got: {out}");
        req["chat_template_kwargs"] = json!({"enable_thinking": true});
        let out = deepseek_v4().render(&req).unwrap();
        assert!(out.ends_with("<｜Assistant｜></think>"), "got: {out}");
        req["chat_template_kwargs"] = JsonValue::Null;
        req["reasoning_effort"] = json!("high");
        let out = deepseek_v4().render(&req).unwrap();
        assert!(out.contains("<｜Assistant｜><think>"), "got: {out}");
    }

    #[test]
    fn request_controls_reach_dynamo() {
        let jinja = jinja(
            json!({"chat_template": "{{ tools | tojson }} {{ thinking }} {{ reasoning_effort }}"}),
        );
        for formatter in [jinja, deepseek_v4()] {
            let mut req = request(json!([{"role":"user","content":"hi"}]));
            req["tools"] = json!([
                {"type":"function","function":{"name":"first"}},
                {"type":"function","function":{"name":"second"}}
            ]);
            req["tool_choice"] = json!({"type":"function","function":{"name":"second"}});
            req["reasoning"] = json!({"effort":"high"});
            let out = formatter.render(&req).unwrap();
            assert!(out.contains("second") && !out.contains("first"));
            assert!(out.contains("high") || out.contains("Reasoning Effort:"));
            req["tool_choice"] = json!("none");
            req["reasoning_effort"] = json!("none");
            req["reasoning"] = JsonValue::Null;
            let out = formatter.render(&req).unwrap();
            assert!(!out.contains("first") && !out.contains("second"));
            assert!(out.contains("False none") || out.ends_with("</think>"));
        }
    }

    /// Mirrors `protocol.py::normalize_reasoning_inputs`: effort decides both
    /// thinking keys via setdefault, so explicit kwargs win.
    #[test]
    fn reasoning_effort_sets_thinking_defaults_with_explicit_kwargs_winning() {
        let enc = jinja(json!({
            "chat_template": "{{ reasoning_effort }}:{{ thinking }}:{{ enable_thinking }}"
        }));
        let mut req = request(json!([{"role": "user", "content": "hi"}]));
        req["reasoning_effort"] = json!("none");
        assert_eq!(enc.render(&req).unwrap(), "none:False:False");
        req["reasoning_effort"] = json!("high");
        assert_eq!(enc.render(&req).unwrap(), "high:True:True");
        req["chat_template_kwargs"] = json!({"enable_thinking": false});
        assert_eq!(enc.render(&req).unwrap(), "high:True:False");
        req["reasoning"] = json!({"effort": "low"});
        assert_eq!(enc.render(&req).unwrap(), "low:True:False");
        req["reasoning"] = json!({"enabled": "yes"});
        req["reasoning_effort"] = JsonValue::Null;
        req["chat_template_kwargs"] = JsonValue::Null;
        assert_eq!(enc.render(&req).unwrap(), ":True:True");
    }

    /// Messages reach the template shaped like the engine's request schema.
    #[test]
    fn messages_match_engine_schema() {
        let enc = jinja(json!({"chat_template": "{{ messages | tojson }}"}));
        let out = enc
            .render(&request(json!([
                {"role": "User", "content": "hi", "name": "bob", "extra": 1},
                {"role": "assistant", "name": "a", "tool_calls": null, "tool_call_id": "c1"},
                {"role": "user", "content": "again"}
            ])))
            .unwrap();
        let rendered: JsonValue = serde_json::from_str(&out).unwrap();
        assert_eq!(
            rendered,
            json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "name": "a", "tool_call_id": "c1"},
                {"role": "user", "content": "again"}
            ])
        );
    }

    #[test]
    fn v4_task_uses_dynamo_task_tokens() {
        let mut req = request(json!([{"role":"user","content":"example.com"}]));
        for (task, suffix) in [
            ("domain", "<｜domain｜>"),
            ("action", "<｜Assistant｜></think><｜action｜>"),
        ] {
            req["task"] = json!(task);
            assert_eq!(
                deepseek_v4().render(&req).unwrap(),
                format!("<｜begin▁of▁sentence｜><｜User｜>example.com{suffix}")
            );
        }
        req["messages"] = json!([{"role":"system","content":"hi"}]);
        assert!(deepseek_v4().render(&req).is_err());
    }

    #[test]
    fn continuation_preserves_token_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg: JsonValue =
            serde_json::from_str(include_str!("../../tests/fixtures/tiny_tokenizer.json")).unwrap();
        cfg["model"]["vocab"]["ab"] = json!(257);
        cfg["model"]["merges"] = json!(["a b"]);
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, cfg.to_string()).unwrap();
        let tokenizer = super::super::adapter::load(path.to_str().unwrap()).unwrap();
        let formatter = jinja(json!({"chat_template":"a", "bos_token":"<|endoftext|>"}));
        let mut req = request(json!([
            {"role":"user","content":"hi"}, {"role":"assistant","content":"b"}
        ]));
        req["continue_final_message"] = json!(true);
        assert_eq!(formatter.encode(&tokenizer, &req).unwrap(), vec![97, 98]);
        assert_eq!(
            super::super::adapter::encode(&tokenizer, "ab").unwrap(),
            vec![257]
        );
        req["messages"][1]["content"] = json!("<|endoftext|>b");
        assert_eq!(formatter.encode(&tokenizer, &req).unwrap(), vec![97, 98]);
        req["continue_final_message"] = json!(false);
        req["messages"][1]["content"] = json!("b");
        let out = deepseek_v4().render(&req).unwrap();
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>hi\n\nb<｜Assistant｜></think>"
        );
    }

    /// The engine never renders `response_format` into the prompt.
    #[test]
    fn v4_ignores_response_format() {
        let mut req = request(json!([{"role":"user","content":"ABCD"}]));
        let plain = deepseek_v4().render(&req).unwrap();
        req["response_format"] = json!({"type": "json_object"});
        assert_eq!(deepseek_v4().render(&req).unwrap(), plain);
    }
}
