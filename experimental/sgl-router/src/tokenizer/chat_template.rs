// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Chat-template rendering for cache-aware routing.
//!
//! The engine caches KV blocks keyed on tokens it produces *after* applying the
//! model's chat template (BOS + role/special markers + content). The router's
//! cache-aware selection must hash the same token sequence, so it renders the
//! same template before tokenizing — otherwise its query hashes never match the
//! engine's stored blocks and cache-aware routing silently degrades to min-load
//! (`sgl_router_overlap_blocks_sum` stuck at 0).
//!
//! The template and its special-token strings come from the model's
//! `tokenizer_config.json` — the HuggingFace built-in template, which is what
//! the engine uses unless launched with an explicit chat-template override.
//!
//! Tokenization runs with `add_special_tokens = false` (see [`super::adapter::encode`]),
//! so the rendered text must already contain `bos_token` and the role markers as
//! literal text. That matches HuggingFace `apply_chat_template(tokenize=True)`
//! semantics, where the template — not the tokenizer's special-token insertion —
//! is the single source of the leading specials.

use anyhow::{Context, Result};
use minijinja::{context, Environment, Error as JinjaError, ErrorKind as JinjaErrorKind};

/// Template registered under a fixed name in the per-model environment.
const TEMPLATE_NAME: &str = "chat";

/// A compiled chat template plus the special-token strings it references.
///
/// One per model, built once at startup from `tokenizer_config.json` and held
/// in the [`super::TokenizerRegistry`]. Rendering is read-only and thread-safe.
pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("bos_token", &self.bos_token)
            .field("eos_token", &self.eos_token)
            .finish()
    }
}

impl ChatTemplate {
    /// Build from a parsed `tokenizer_config.json`. Returns `Ok(None)` when the
    /// config carries no `chat_template` (the model is then routed via the raw
    /// prompt-text path, unchanged).
    pub fn from_tokenizer_config(cfg: &serde_json::Value) -> Result<Option<Self>> {
        let Some(template_src) = extract_chat_template(cfg) else {
            return Ok(None);
        };
        let bos_token = extract_token_str(cfg, "bos_token");
        let eos_token = extract_token_str(cfg, "eos_token");

        let mut env = Environment::new();
        // HuggingFace compiles chat templates with trim_blocks + lstrip_blocks;
        // mirror that or rendered whitespace (and thus tokens) diverge.
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        // Python str/dict methods used by real templates (.startswith, .items,
        // .strip, ...) that minijinja doesn't implement natively.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.add_template_owned(TEMPLATE_NAME, template_src)
            .context("compile chat template from tokenizer_config.json")?;

        Ok(Some(Self {
            env,
            bos_token,
            eos_token,
        }))
    }

    /// Render `messages` (the request's `messages` array) into the prompt text
    /// the engine would tokenize, with `add_generation_prompt = true`.
    ///
    /// `messages` is passed through as-is; templates expect string `content`.
    /// Multimodal content arrays are out of scope (text-only routing) and will
    /// surface as a render error, which the caller treats as "fall back to the
    /// raw prompt-text path".
    ///
    /// Request-level template inputs beyond `messages` (`tools`, `documents`,
    /// ...) are not threaded through: a request carrying them renders without
    /// them, so its hashes won't match the engine and it routes by min-load —
    /// no worse than before this path existed. Plain chat routes by templated
    /// prefix.
    pub fn render(&self, messages: &serde_json::Value) -> Result<String> {
        let tmpl = self
            .env
            .get_template(TEMPLATE_NAME)
            .context("chat template not registered")?;
        tmpl.render(context! {
            messages => messages,
            add_generation_prompt => true,
            bos_token => self.bos_token,
            eos_token => self.eos_token,
        })
        .context("render chat template")
    }
}

/// Pull the chat-template source out of `tokenizer_config.json`.
///
/// Accepts both shapes HuggingFace ships:
///   - `"chat_template": "<jinja>"` — the common single-template case.
///   - `"chat_template": [{"name": "default", "template": "<jinja>"}, ...]` —
///     multi-template models; we take the entry named `default`, else the first.
fn extract_chat_template(cfg: &serde_json::Value) -> Option<String> {
    match cfg.get("chat_template")? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(arr) => arr
            .iter()
            .find(|e| e.get("name").and_then(|n| n.as_str()) == Some("default"))
            .or_else(|| arr.first())
            .and_then(|e| e.get("template").and_then(|t| t.as_str()))
            .map(str::to_owned),
        _ => None,
    }
}

/// Read a special-token string, accepting both the plain-string form and the
/// `AddedToken` object form (`{"content": "<tok>", ...}`) HuggingFace uses.
fn extract_token_str(cfg: &serde_json::Value, key: &str) -> Option<String> {
    match cfg.get(key)? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => {
            o.get("content").and_then(|c| c.as_str()).map(str::to_owned)
        }
        _ => None,
    }
}

/// `raise_exception(msg)` — templates call this to reject malformed message
/// sequences (e.g. a non-alternating role order). Surfaces as a render error.
fn raise_exception(msg: String) -> std::result::Result<String, JinjaError> {
    Err(JinjaError::new(JinjaErrorKind::InvalidOperation, msg))
}

/// `strftime_now(format)` — current local time, matching the helper HuggingFace
/// injects so templates can stamp the date. Both engine and router render
/// within the same day, so the date prefix is stable enough to share a cache
/// block.
fn strftime_now(format: String) -> String {
    chrono::Local::now().format(&format).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A small but representative instruct template: emits `bos_token`, wraps
    /// each turn in role markers, and appends a generation prompt. Exercises the
    /// variables the renderer must supply (`messages`, `bos_token`,
    /// `add_generation_prompt`).
    const SIMPLE_TEMPLATE: &str = "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}";

    fn messages() -> serde_json::Value {
        json!([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"}
        ])
    }

    #[test]
    fn no_chat_template_returns_none() {
        let cfg = json!({"bos_token": "<s>", "eos_token": "</s>"});
        assert!(ChatTemplate::from_tokenizer_config(&cfg).unwrap().is_none());
    }

    #[test]
    fn renders_roles_bos_and_generation_prompt() {
        let cfg = json!({
            "chat_template": SIMPLE_TEMPLATE,
            "bos_token": "<s>",
            "eos_token": "</s>",
        });
        let tmpl = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        let out = tmpl.render(&messages()).unwrap();
        assert_eq!(
            out,
            "<s><|system|>\nbe brief<|end|>\n<|user|>\nhi<|end|>\n<|assistant|>\n"
        );
    }

    /// `add_generation_prompt` is always true on the routing side (we hash the
    /// prompt the engine will prefill, which includes the assistant header).
    #[test]
    fn generation_prompt_is_always_appended() {
        let cfg = json!({ "chat_template": SIMPLE_TEMPLATE, "bos_token": "<s>" });
        let tmpl = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        assert!(tmpl
            .render(&messages())
            .unwrap()
            .ends_with("<|assistant|>\n"));
    }

    /// The list form `[{name, template}, ...]` selects the `default` entry.
    #[test]
    fn list_form_selects_default_template() {
        let cfg = json!({
            "chat_template": [
                {"name": "tool_use", "template": "TOOLS"},
                {"name": "default", "template": SIMPLE_TEMPLATE},
            ],
            "bos_token": "<s>",
        });
        let tmpl = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        assert!(tmpl
            .render(&messages())
            .unwrap()
            .starts_with("<s><|system|>"));
    }

    /// `bos_token` in the `AddedToken` object form is read from `.content`.
    #[test]
    fn bos_token_object_form_is_extracted() {
        let cfg = json!({
            "chat_template": "{{ bos_token }}X",
            "bos_token": {"content": "<|begin|>", "lstrip": false},
        });
        let tmpl = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        assert_eq!(tmpl.render(&json!([])).unwrap(), "<|begin|>X");
    }

    /// `raise_exception` surfaces as a render error (caller then falls back to
    /// the raw prompt-text path rather than failing the request).
    #[test]
    fn raise_exception_surfaces_as_error() {
        let cfg = json!({
            "chat_template": "{{ raise_exception('bad messages') }}",
            "bos_token": "<s>",
        });
        let tmpl = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        let err = tmpl.render(&messages()).unwrap_err();
        // The minijinja message is the cause; check the full anyhow chain.
        assert!(format!("{err:#}").contains("bad messages"), "got: {err:#}");
    }

    /// pycompat exposes Python str methods (`.startswith`, `.upper`, ...) that
    /// real HuggingFace templates lean on; without the callback these error.
    #[test]
    fn pycompat_string_methods_available() {
        let cfg = json!({
            "chat_template": "{% for m in messages %}{% if m['role'].startswith('sys') %}{{ m['content'].upper() }}{% endif %}{% endfor %}",
            "bos_token": "<s>",
        });
        let tmpl = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        assert_eq!(tmpl.render(&messages()).unwrap(), "BE BRIEF");
    }

    /// trim_blocks + lstrip_blocks match HuggingFace's compilation: the newline
    /// after a block tag and leading whitespace before one are stripped, so a
    /// block-per-line template renders without spurious blank lines.
    #[test]
    fn trim_and_lstrip_blocks_match_huggingface() {
        let cfg = json!({
            "chat_template": "{% for m in messages %}\n  {% if true %}\n{{ m['role'] }}\n  {% endif %}\n{% endfor %}",
            "bos_token": "<s>",
        });
        let tmpl = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        // Each iteration emits just "<role>\n"; lstrip removes the two leading
        // spaces before the `{% if %}`/`{% endif %}`, trim removes the newline
        // immediately after each block tag.
        assert_eq!(tmpl.render(&messages()).unwrap(), "system\nuser\n");
    }
}
