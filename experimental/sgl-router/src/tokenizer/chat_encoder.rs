// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Chat-prompt rendering for cache-aware routing, via `dynamo-renderer`.
//!
//! The engine caches KV blocks keyed on the tokens of the *rendered* chat
//! prompt, so the router renders the same prompt before hashing. Tokenization
//! adds no special tokens, so the rendered text carries BOS and role markers.

use std::collections::HashMap;

use anyhow::{Context, Result};
use dynamo_renderer::{
    may_be_fix_tool_schema, native_formatter_for, ChatTemplate, ContextMixins, OAIChatLikeRequest,
    PromptContextMixin, PromptFormatter,
};
use minijinja::Value;

pub type ChatTemplateKwargs = HashMap<String, serde_json::Value>;

// HuggingFace supplies these names through special_tokens_map. Dynamo supplies
// only bos/eos/unk itself; the remaining tokens need to be template defaults.
const SPECIAL_TOKEN_KEYS: [&str; 7] = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
];

pub struct ChatEncoder {
    formatter: PromptFormatter,
    /// Engine-side defaults a request's `chat_template_kwargs` override.
    defaults: ChatTemplateKwargs,
}

impl ChatEncoder {
    /// HF Jinja template from `tokenizer_config.json`, overridden by a sibling
    /// `chat_template.jinja` when present (transformers' precedence); `Ok(None)`
    /// when the model ships neither.
    pub fn from_tokenizer_config(
        cfg: &serde_json::Value,
        chat_template_jinja: Option<&str>,
    ) -> Result<Option<Self>> {
        let mut cfg = cfg.clone();
        if let Some(template) = chat_template_jinja {
            cfg["chat_template"] = template.into();
        }
        let mut defaults: ChatTemplateKwargs = SPECIAL_TOKEN_KEYS
            .into_iter()
            .map(|key| {
                let token = cfg[key]
                    .as_str()
                    .or_else(|| cfg[key]["content"].as_str())
                    .unwrap_or_default()
                    .to_owned();
                // An absent bos/eos/unk otherwise becomes a Jinja None, which
                // prints "None" instead of HF's empty undefined value. Strings
                // also accept AddedToken objects without requiring unused flags.
                cfg[key] = token.clone().into();
                (key.to_owned(), token.into())
            })
            .collect();
        if let Some(extra) = cfg
            .get("additional_special_tokens")
            .filter(|v| v.is_array())
        {
            defaults.insert("additional_special_tokens".into(), extra.clone());
        }
        // HF's `[{name, template}]` list form -> Dynamo's `[{name: template}]`.
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
        let formatter = PromptFormatter::from_parts(
            template,
            ContextMixins::new(&[PromptContextMixin::OaiChat]),
            true,
        )
        .context("compile chat template")?;
        Ok(Some(Self {
            formatter,
            defaults,
        }))
    }

    /// Dynamo's built-in encoder for models that ship no template (DeepSeek-V4).
    /// `model_type` (from `config.json`) is authoritative; the model id's last
    /// path segment is the fallback.
    pub fn native(model_type: Option<&str>, model_id: &str) -> Option<Self> {
        let name = model_id.rsplit('/').next().unwrap_or(model_id);
        let formatter =
            native_formatter_for(&model_type.map(str::to_lowercase), &name.to_lowercase())?;
        // Engine defaults: chat mode (`SGLANG_DEFAULT_THINKING=false`) and no
        // reasoning-effort preamble; Dynamo defaults to thinking at high effort.
        let defaults = HashMap::from([
            ("thinking".into(), false.into()),
            ("reasoning_effort".into(), "low".into()),
        ]);
        Some(Self {
            formatter,
            defaults,
        })
    }

    /// Render a chat request (`messages`, `tools`, `chat_template_kwargs`) into
    /// the prompt text the engine tokenizes, with `add_generation_prompt = true`.
    pub fn render(&self, request: &serde_json::Value) -> Result<String> {
        let mut kwargs = self.defaults.clone();
        // The engine folds the top-level `reasoning_effort` into the template
        // kwargs, with explicit `chat_template_kwargs` winning.
        if let Some(effort) = request.get("reasoning_effort").filter(|v| !v.is_null()) {
            kwargs.insert("reasoning_effort".into(), effort.clone());
        }
        if let Some(extra) = request
            .get("chat_template_kwargs")
            .and_then(|v| v.as_object())
        {
            kwargs.extend(extra.iter().map(|(k, v)| (k.clone(), v.clone())));
        }
        let PromptFormatter::OAI(formatter) = &self.formatter;
        formatter
            .render(&ChatRequest { request, kwargs })
            .context("render chat template")
    }
}

struct ChatRequest<'a> {
    request: &'a serde_json::Value,
    kwargs: ChatTemplateKwargs,
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

/// Reshape a message the way the engine's pydantic schema does before it
/// reaches the template: lowercase the role, drop unknown and null fields
/// (a user message keeps only `role` and `content`), and default a missing
/// `content` to `""`.
fn engine_message(message: &serde_json::Value) -> serde_json::Value {
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
        serde_json::Value::Null => "".into(),
        content => content.clone(),
    };
    out.insert("role".into(), role.into());
    out.insert("content".into(), content);
    out.into()
}

impl OAIChatLikeRequest for ChatRequest<'_> {
    fn model(&self) -> String {
        self.request["model"]
            .as_str()
            .unwrap_or_default()
            .to_owned()
    }
    fn messages(&self) -> Value {
        let messages: Vec<_> = self.request["messages"]
            .as_array()
            .map(|m| m.iter().map(engine_message).collect())
            .unwrap_or_default();
        Value::from_serialize(&messages)
    }
    fn tools(&self) -> Option<Value> {
        may_be_fix_tool_schema(self.request.get("tools")?.clone())
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

    fn jinja(cfg: serde_json::Value) -> ChatEncoder {
        ChatEncoder::from_tokenizer_config(&cfg, None)
            .unwrap()
            .expect("config has a chat_template")
    }

    fn request(messages: serde_json::Value) -> serde_json::Value {
        json!({"model": "m", "messages": messages})
    }

    fn v4() -> ChatEncoder {
        ChatEncoder::native(Some("deepseek_v4"), "any").unwrap()
    }

    #[test]
    fn no_chat_template_returns_none() {
        let cfg = json!({"bos_token": "<s>", "eos_token": "</s>"});
        assert!(ChatEncoder::from_tokenizer_config(&cfg, None)
            .unwrap()
            .is_none());
    }

    /// A sibling `chat_template.jinja` wins over `tokenizer_config.json`, as in
    /// transformers, and suffices on its own.
    #[test]
    fn chat_template_jinja_file_takes_precedence() {
        let cfg = json!({"chat_template": "CONFIG"});
        let enc = ChatEncoder::from_tokenizer_config(&cfg, Some("FILE"))
            .unwrap()
            .unwrap();
        assert_eq!(enc.render(&request(json!([]))).unwrap(), "FILE");
        let enc = ChatEncoder::from_tokenizer_config(&json!({}), Some("FILE"))
            .unwrap()
            .unwrap();
        assert_eq!(enc.render(&request(json!([]))).unwrap(), "FILE");
    }

    /// Messages reach the template shaped like the engine's pydantic dump:
    /// lowercase role, no unknown or null fields, `content` defaulting to "".
    #[test]
    fn messages_match_engine_schema() {
        let enc = jinja(json!({"chat_template": "{{ messages | tojson }}"}));
        let out = enc
            .render(&request(json!([
                {"role": "User", "content": "hi", "name": "bob", "extra": 1},
                {"role": "assistant", "name": "a", "tool_calls": null, "tool_call_id": "c1"}
            ])))
            .unwrap();
        let rendered: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(
            rendered,
            json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "name": "a", "tool_call_id": "c1"}
            ])
        );
    }

    #[test]
    fn additional_special_tokens_are_supplied() {
        let enc = jinja(json!({
            "chat_template": "{{ additional_special_tokens | join(',') }}",
            "additional_special_tokens": ["<a>", "<b>"]
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

    #[test]
    fn chat_template_kwargs_reach_the_template() {
        let enc = jinja(json!({"chat_template": "t={{ enable_thinking }}"}));
        let mut req = request(json!([]));
        assert_eq!(enc.render(&req).unwrap(), "t=");
        req["chat_template_kwargs"] = json!({"enable_thinking": true});
        assert_eq!(enc.render(&req).unwrap(), "t=True");
    }

    #[test]
    fn raise_exception_surfaces_as_error() {
        let enc = jinja(json!({"chat_template": "{{ raise_exception('bad messages') }}"}));
        let err = enc.render(&request(json!([]))).unwrap_err();
        assert!(format!("{err:#}").contains("bad messages"), "got: {err:#}");
    }

    #[test]
    fn native_detection() {
        assert!(ChatEncoder::native(None, "deepseek-ai/DeepSeek-V4-Flash").is_some());
        assert!(ChatEncoder::native(None, "deepseek-v4-tiny").is_some());
        assert!(ChatEncoder::native(Some("deepseek_v4"), "alias").is_some());
        assert!(ChatEncoder::native(Some("llama"), "deepseek-v4").is_none());
        assert!(ChatEncoder::native(None, "deepseek-ai/DeepSeek-V3.2-Exp").is_none());
        assert!(ChatEncoder::native(None, "Qwen/Qwen3-0.6B").is_none());
    }

    /// Byte-exact against the engine's `/tokenize` in its default chat mode:
    /// `[{user:"ABCD"}]` -> `[0, 128803, 51453, 128804, 128822]`.
    #[test]
    fn v4_single_user_turn() {
        let out = v4()
            .render(&request(json!([{"role":"user","content":"ABCD"}])))
            .unwrap();
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>ABCD<｜Assistant｜></think>"
        );
    }

    #[test]
    fn v4_system_then_multi_turn() {
        let out = v4()
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

    #[test]
    fn v4_thinking_kwarg_overrides_chat_default() {
        let mut req = request(json!([{"role":"user","content":"ABCD"}]));
        req["chat_template_kwargs"] = json!({"thinking": true});
        let out = v4().render(&req).unwrap();
        assert!(out.ends_with("<｜Assistant｜><think>"), "got: {out}");
    }
}
