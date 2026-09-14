// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Chat rendering via Dynamo for cache-aware routing and input ID forwarding.
//!
//! The router renders what Dynamo renders. Engine-specific normalization of
//! request fields is deliberately not replicated here; the `input_ids` forward
//! guard in the chat route omits ids for every request shape whose engine-side
//! rendering has not been verified against this one.

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

// HuggingFace supplies these names through special_tokens_map. Dynamo supplies
// only bos/eos/unk itself; the rest are passed as template context defaults.
const SPECIAL_TOKEN_KEYS: [&str; 7] = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
];

/// Renders and tokenizes chat requests through Dynamo.
pub struct ChatFormatter {
    formatter: Arc<dyn OAIPromptFormatter>,
    /// Template context defaults; request `chat_template_kwargs` override them.
    defaults: ChatTemplateKwargs,
}

impl ChatFormatter {
    /// Load model files and select a Dynamo formatter. The renderer itself
    /// accepts parsed config; it does not fetch files or choose HF vs native.
    pub fn load(model_id: &str, tokenizer_path: &str) -> Result<Option<Self>> {
        let files = super::adapter::ModelFiles::open(tokenizer_path);
        let model_type = files
            .json("config.json")?
            .and_then(|cfg| cfg["model_type"].as_str().map(str::to_owned));
        match model_type.as_deref() {
            // These require tokenization paths not yet supported by this adapter.
            Some("inkling_mm_model" | "kimi_k3") => return Ok(None),
            Some(t) if t.starts_with("deepseek_v4") => {
                return Ok(Self::native(model_type.as_deref(), model_id));
            }
            _ => {}
        }
        let cfg = files
            .json("tokenizer_config.json")?
            .unwrap_or_else(|| serde_json::json!({}));
        let jinja = files.text("chat_template.jinja")?;
        Ok(Self::from_tokenizer_config(cfg, jinja.as_deref())?
            .or_else(|| Self::native(model_type.as_deref(), model_id)))
    }

    /// HF Jinja template from `tokenizer_config.json`, overridden by a sibling
    /// `chat_template.jinja` when present (transformers' precedence); `Ok(None)`
    /// when the model ships neither.
    pub fn from_tokenizer_config(
        mut cfg: JsonValue,
        chat_template_jinja: Option<&str>,
    ) -> Result<Option<Self>> {
        if let Some(template) = chat_template_jinja {
            cfg["chat_template"] = template.into();
        }
        // HuggingFace supplies None when no retrieval documents are present.
        let mut defaults = HashMap::from([("documents".into(), JsonValue::Null)]);
        for key in SPECIAL_TOKEN_KEYS {
            // Dynamo parses the plain-string form; reduce HF's `AddedToken`
            // object form to its content. Any other shape fails loudly below.
            if let Some(content) = added_token_content(&cfg[key]) {
                cfg[key] = content.into();
            }
            if matches!(key, "bos_token" | "eos_token" | "unk_token") {
                // Dynamo supplies these from the parsed config but renders an
                // absent one as `None`; HF renders "". Kwargs cannot override it.
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
        let PromptFormatter::OAI(formatter) =
            PromptFormatter::from_parts(template, ContextMixins::default(), true)
                .context("compile chat template")?;
        Ok(Some(Self {
            formatter,
            defaults,
        }))
    }

    /// Dynamo's DeepSeek encoders (V4 family, V3.2), the only built-in ones
    /// verified against the engine. `model_type` (from `config.json`) is
    /// authoritative; the model id's last path segment is the fallback.
    pub fn native(model_type: Option<&str>, model_id: &str) -> Option<Self> {
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

    fn template_kwargs(&self, request: &JsonValue) -> Result<ChatTemplateKwargs> {
        let mut kwargs: ChatTemplateKwargs = match request.get("chat_template_kwargs") {
            None | Some(JsonValue::Null) => ChatTemplateKwargs::new(),
            Some(v) => serde_json::from_value(v.clone()).context("chat_template_kwargs")?,
        };
        for (key, value) in &self.defaults {
            kwargs.entry(key.clone()).or_insert_with(|| value.clone());
        }
        Ok(kwargs)
    }

    /// Render the prompt text Dynamo produces for `request`.
    pub fn render(&self, request: &JsonValue) -> Result<String> {
        anyhow::ensure!(request["messages"].is_array(), "messages must be an array");
        let kwargs = self.template_kwargs(request)?;
        self.formatter
            .render(&ChatRequest { request, kwargs })
            .context("render chat template")
    }

    pub fn encode(
        &self,
        tokenizer: &dynamo_tokenizers::Tokenizer,
        request: &JsonValue,
    ) -> Result<Vec<u32>> {
        super::adapter::encode(tokenizer, &self.render(request)?)
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

/// Mirrors Dynamo's own impl for its wire type: request fields pass through.
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
        let tools = self.request.get("tools")?;
        // HF and the engine treat an empty list as "no tools"; Dynamo's schema
        // fixer would hand the template `[]`, which tools-branching templates
        // render as a tool preamble.
        if tools.as_array().is_none_or(|t| t.is_empty()) {
            return None;
        }
        may_be_fix_tool_schema(tools.clone())
    }
    fn tool_choice(&self) -> Option<Value> {
        self.request.get("tool_choice").map(Value::from_serialize)
    }
    fn reasoning_effort(&self) -> Option<Value> {
        self.request
            .get("reasoning_effort")
            .map(Value::from_serialize)
    }
    fn response_format(&self) -> Option<Value> {
        self.request
            .get("response_format")
            .map(Value::from_serialize)
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

    fn v4() -> ChatFormatter {
        ChatFormatter::native(Some("deepseek_v4"), "any").unwrap()
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
        assert!(v4().render(&json!({"model": "m"})).is_err());
    }

    #[test]
    fn native_detection() {
        assert!(ChatFormatter::native(None, "deepseek-ai/DeepSeek-V4-Flash").is_some());
        assert!(ChatFormatter::native(None, "deepseek-v4-tiny").is_some());
        assert!(ChatFormatter::native(Some("deepseek_v4"), "alias").is_some());
        assert!(ChatFormatter::native(Some("deepseek_v41"), "alias").is_some());
        assert!(ChatFormatter::native(Some("deepseek_v32"), "DeepSeek-V3.2").is_some());
        assert!(ChatFormatter::native(Some("inkling_mm_model"), "inkling").is_none());
        assert!(ChatFormatter::native(Some("llama"), "deepseek-v4").is_none());
        assert!(ChatFormatter::native(None, "deepseek-ai/DeepSeek-V3.2-Exp").is_none());
        assert!(ChatFormatter::native(None, "Qwen/Qwen3-0.6B").is_none());
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

    /// Request kwargs override the chat-mode default.
    #[test]
    fn v4_thinking_kwarg_overrides_chat_default() {
        let mut req = request(json!([{"role":"user","content":"ABCD"}]));
        req["chat_template_kwargs"] = json!({"thinking": true});
        let out = v4().render(&req).unwrap();
        assert!(out.ends_with("<｜Assistant｜><think>"), "got: {out}");
    }
}
