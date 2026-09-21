// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Chat rendering via dynamo-render for cache-aware routing and input ID forwarding.
//!
//! Dynamo owns prompt semantics. This adapter loads model assets, exposes request
//! fields through its rendering trait, and preserves segments during tokenization.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use dynamo_renderer::{
    deepseek_formatter_for, kimi_k3_formatter_for, ChatTemplate, ContextMixins, OAIChatLikeRequest,
    OAIPromptFormatter, PromptFormatter, RenderedPrompt,
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
}

impl ChatFormatter {
    /// Load model files and select a template or native formatter from dynamo-render.
    pub fn load(model_id: &str, tokenizer_path: &str) -> Result<Option<Self>> {
        let files = super::adapter::ModelFiles::open(tokenizer_path);
        let model_type = files
            .json("config.json")?
            .and_then(|cfg| cfg["model_type"].as_str().map(str::to_lowercase));
        // As for DeepSeek, `model_type` is authoritative and the model id's last
        // path segment is the fallback.
        let name = model_id
            .rsplit('/')
            .next()
            .unwrap_or(model_id)
            .to_lowercase();
        if let Some(PromptFormatter::OAI(formatter)) =
            kimi_k3_formatter_for(&model_type, &name, true)
        {
            return Ok(Some(Self {
                formatter,
                defaults: HashMap::new(),
            }));
        }
        match model_type.as_deref() {
            // These require tokenization paths not yet supported by this adapter.
            Some("inkling_mm_model") => return Ok(None),
            Some("deepseek_v4") => {
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
        let mut defaults = HashMap::new();
        for key in SPECIAL_TOKEN_KEYS {
            // Convert HF AddedToken objects to strings for dynamo-render.
            if let Some(content) = added_token_content(&cfg[key]) {
                cfg[key] = content.into();
            }
            // Expose additional token metadata that ChatTemplate does not put in
            // the Jinja context. Missing values keep Dynamo's own defaults.
            if !matches!(key, "bos_token" | "eos_token" | "unk_token") {
                if let Some(value) = cfg.get(key) {
                    defaults.insert(key.to_owned(), value.clone());
                }
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
        let PromptFormatter::OAI(formatter) =
            PromptFormatter::from_parts(template, ContextMixins::default(), true)
                .context("compile chat template")?;
        Ok(Some(Self {
            formatter,
            defaults,
        }))
    }

    /// Select Dynamo's native DeepSeek formatter without changing its model
    /// detection, thinking defaults, or reasoning-effort semantics.
    pub fn deepseek_native(model_type: Option<&str>, model_id: &str) -> Option<Self> {
        let name = model_id
            .rsplit('/')
            .next()
            .unwrap_or(model_id)
            .to_lowercase();
        let model_type = model_type.map(str::to_lowercase);
        let PromptFormatter::OAI(formatter) = deepseek_formatter_for(&model_type, &name)?;
        Some(Self {
            formatter,
            defaults: HashMap::new(),
        })
    }

    fn render_prompt(&self, request: &JsonValue) -> Result<RenderedPrompt> {
        request["messages"]
            .as_array()
            .context("messages must be an array")?;
        let mut kwargs: ChatTemplateKwargs = match request.get("chat_template_kwargs") {
            None | Some(JsonValue::Null) => ChatTemplateKwargs::new(),
            Some(value) => serde_json::from_value(value.clone()).context("chat_template_kwargs")?,
        };
        for (key, value) in &self.defaults {
            kwargs.entry(key.clone()).or_insert_with(|| value.clone());
        }
        self.formatter
            .render_prompt(&ChatRequest { request, kwargs })
            .context("render chat template")
    }

    pub fn encode(
        &self,
        tokenizer: &dynamo_tokenizers::Tokenizer,
        request: &JsonValue,
    ) -> Result<Vec<u32>> {
        let prompt = self.render_prompt(request)?;
        match prompt.encode_segments() {
            Some(segments) => Ok(tokenizer.encode_segments(&segments)?.token_ids().to_vec()),
            None => super::adapter::encode(tokenizer, prompt.as_str()),
        }
    }

    /// Display text only; use `encode` to retain Dynamo's segment boundaries.
    pub fn render(&self, request: &JsonValue) -> Result<String> {
        Ok(self.render_prompt(request)?.into_text())
    }
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
    kwargs: ChatTemplateKwargs,
}

impl ChatRequest<'_> {
    fn field(&self, key: &str) -> Option<Value> {
        self.request
            .get(key)
            .filter(|value| !value.is_null())
            .map(Value::from_serialize)
    }
}

impl OAIChatLikeRequest for ChatRequest<'_> {
    fn model(&self) -> String {
        self.request["model"]
            .as_str()
            .unwrap_or_default()
            .to_owned()
    }
    fn messages(&self) -> Value {
        Value::from_serialize(&self.request["messages"])
    }
    fn tools(&self) -> Option<Value> {
        self.field("tools")
    }
    fn tool_choice(&self) -> Option<Value> {
        self.field("tool_choice")
    }
    fn reasoning_effort(&self) -> Option<Value> {
        self.field("reasoning_effort")
    }
    fn response_format(&self) -> Option<Value> {
        self.field("response_format")
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
    fn absent_documents_keep_dynamo_default() {
        let enc = jinja(json!({
            "chat_template": "{% if documents is not none %}DOCS{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}"
        }));
        let mut req = request(json!([{"role":"user","content":"hi"}]));
        assert_eq!(enc.render(&req).unwrap(), "DOCShi");
        req["chat_template_kwargs"] = json!({"documents": [{"text": "reference"}]});
        assert_eq!(enc.render(&req).unwrap(), "DOCShi");
    }

    #[test]
    fn absent_special_tokens_keep_dynamo_defaults() {
        let enc = jinja(json!({
            "chat_template": "A{{ bos_token }}{{ eos_token }}{{ unk_token }}{{ sep_token }}{{ pad_token }}{{ cls_token }}{{ mask_token }}B"
        }));
        assert_eq!(enc.render(&request(json!([]))).unwrap(), "ANoneNoneNoneB");
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

    /// Explicit empty tool arrays reach Dynamo unchanged.
    #[test]
    fn tools_reach_the_template_unchanged() {
        let enc = jinja(json!({
            "chat_template": "{% if tools is not none %}T:{{ tools | length }}{% endif %}X"
        }));
        let mut req = request(json!([]));
        assert_eq!(enc.render(&req).unwrap(), "X");
        req["tools"] = json!([]);
        assert_eq!(enc.render(&req).unwrap(), "T:0X");
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
        assert!(ChatFormatter::deepseek_native(Some("deepseek_v41"), "alias").is_none());
        assert!(ChatFormatter::deepseek_native(Some("deepseek_v32"), "DeepSeek-V3.2").is_some());
        assert!(ChatFormatter::deepseek_native(Some("inkling_mm_model"), "inkling").is_none());
        assert!(ChatFormatter::deepseek_native(Some("llama"), "deepseek-v4").is_none());
        assert!(ChatFormatter::deepseek_native(None, "deepseek-ai/DeepSeek-V3.2-Exp").is_none());
        assert!(ChatFormatter::deepseek_native(None, "Qwen/Qwen3-0.6B").is_none());
    }
}
