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
//! Tokenization does not auto-prepend special tokens (the `dynamo_tokenizers`
//! HF wrapper defaults `add_special_tokens` to false and the router never
//! overrides it; [`super::adapter::encode`]
//! adds none of its own), so the rendered text must already contain `bos_token`
//! and the role markers as literal text. That matches HuggingFace
//! `apply_chat_template(tokenize=True)` semantics, where the template — not the
//! tokenizer's special-token insertion — is the single source of the leading
//! specials.

use anyhow::{bail, Context, Result};
use minijinja::{
    value::{Kwargs, Value as JinjaValue},
    Environment, Error as JinjaError, ErrorKind as JinjaErrorKind, UndefinedBehavior,
};
use serde_json::{Map as JsonMap, Value as JsonValue};
use std::collections::BTreeMap;

use super::pyjson;

/// Template registered under a fixed name in the per-model environment.
const TEMPLATE_NAME: &str = "chat";

/// The named special tokens HuggingFace injects into the template context via
/// `special_tokens_map`. Each is supplied from `tokenizer_config.json`, or as
/// the empty string when absent — jinja2 renders an undefined name as `""`, so
/// an absent token must not surface as anything else (minijinja would otherwise
/// print a `none` value as the literal string "none", silently diverging every
/// block hash from the engine's).
const SPECIAL_TOKEN_KEYS: [&str; 7] = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
];

/// Context names the renderer owns. A `chat_template_kwargs` entry by one of
/// these names is dropped rather than allowed to shadow it (HuggingFace would
/// raise a duplicate-keyword `TypeError` for most of them).
const RESERVED_CONTEXT_KEYS: [&str; 4] =
    ["messages", "tools", "documents", "add_generation_prompt"];

/// Request-level inputs the engine's generic Jinja path threads into
/// `apply_chat_template` (`serving_chat.py::_apply_jinja_template`).
///
/// Resolved once per request by [`JinjaRenderOpts::resolve`]; the Jinja
/// encoder reads it together with the request's top-level `tools`.
#[derive(Clone, Debug, Default)]
pub struct JinjaRenderOpts {
    /// The request's raw `tool_choice`. `"none"` suppresses tools entirely and
    /// a named-function choice narrows them to that function, as the engine's
    /// `_process_messages` does.
    pub tool_choice: Option<JsonValue>,
    /// Extra template variables: `reasoning_effort` (when set) followed by the
    /// request's `chat_template_kwargs`.
    pub template_kwargs: JsonMap<String, JsonValue>,
}

impl JinjaRenderOpts {
    /// Mirror the engine's kwargs resolution: `chat_template_kwargs.reasoning_effort`
    /// is popped and, when non-null, overrides the top-level `reasoning_effort`
    /// (`_convert_to_internal_request`); the effective effort is passed first and
    /// the remaining `chat_template_kwargs` are layered on top.
    ///
    /// RESIDUE: an engine launched with `--default-chat-template-kwargs` merges
    /// those defaults server-side; the router cannot see them.
    pub fn resolve(request: &JsonValue) -> Self {
        let mut ctk = request
            .get("chat_template_kwargs")
            .and_then(JsonValue::as_object)
            .cloned()
            .unwrap_or_default();
        let effort = ctk
            .remove("reasoning_effort")
            .filter(|v| !v.is_null())
            .or_else(|| {
                request
                    .get("reasoning_effort")
                    .cloned()
                    .filter(|v| !v.is_null())
            });

        let mut template_kwargs = JsonMap::new();
        if let Some(effort) = effort {
            template_kwargs.insert("reasoning_effort".into(), effort);
        }
        for (k, v) in ctk {
            if !RESERVED_CONTEXT_KEYS.contains(&k.as_str()) {
                template_kwargs.insert(k, v);
            }
        }
        JinjaRenderOpts {
            tool_choice: request.get("tool_choice").cloned(),
            template_kwargs,
        }
    }
}

/// A compiled chat template plus the special-token strings it references.
///
/// One per model, built once at startup from `tokenizer_config.json` and held
/// in the [`super::TokenizerRegistry`]. Rendering is read-only and thread-safe.
pub struct ChatTemplate {
    env: Environment<'static>,
    /// `(name, token)` pairs for [`SPECIAL_TOKEN_KEYS`]; absent tokens are `""`.
    special_tokens: Vec<(&'static str, String)>,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("special_tokens", &self.special_tokens)
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
        let special_tokens = SPECIAL_TOKEN_KEYS
            .iter()
            .map(|&key| (key, extract_token_str(cfg, key).unwrap_or_default()))
            .collect();

        let mut env = Environment::new();
        // HuggingFace compiles chat templates with trim_blocks + lstrip_blocks;
        // mirror that or rendered whitespace (and thus tokens) diverge.
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        // Printing a variable the router didn't supply (a custom
        // `chat_template_kwargs` entry, a date var, ...) must be a render
        // error so the caller falls back to raw-text hashing — under the
        // default lenient behavior it would render as `""` and produce a
        // plausible-but-divergent prompt whose hashes silently never match
        // the engine's. If-tests and iteration over undefined stay permitted
        // (`{% if enable_thinking is defined %}`-style guards are common).
        env.set_undefined_behavior(UndefinedBehavior::SemiStrict);
        // Python str/dict methods used by real templates (.startswith, .items,
        // .strip, ...) that minijinja doesn't implement natively.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        // HuggingFace replaces jinja2's `tojson` with `json.dumps` (default
        // `", "`/`": "` separators, `ensure_ascii=False`); minijinja's built-in
        // emits compact, HTML-escaped JSON. Tool schemas are printed through it
        // at the FRONT of tool-carrying prompts, so the two must agree byte for
        // byte or block 0 never matches.
        env.add_filter("tojson", py_tojson);
        env.add_template_owned(TEMPLATE_NAME, template_src)
            .context("compile chat template from tokenizer_config.json")?;

        Ok(Some(Self {
            env,
            special_tokens,
        }))
    }

    /// Render `messages` (the request's `messages` array) into the prompt text
    /// the engine would tokenize, with `add_generation_prompt = true`.
    ///
    /// Mirrors the engine's generic Jinja path (`_apply_jinja_template`):
    /// messages are normalized as the engine does before rendering (see
    /// [`engine_messages`]), `tools` is the request's top-level `tools` after
    /// the engine's `tool_choice` filtering and pydantic dump (see
    /// [`engine_tools`]), and `opts.template_kwargs` (`reasoning_effort` +
    /// `chat_template_kwargs`) are supplied as template variables. As in the
    /// engine, a render that fails with OpenAI-wrapped tools is retried once
    /// with the bare `function` objects.
    ///
    /// Multimodal content arrays are out of scope (text-only routing): a
    /// template may stringify the array (divergent hashes → min-load) or error
    /// (raw prompt-text fallback); neither fails the request. `documents` is
    /// supplied as `none`. Any other variable the template prints is a render
    /// error (semi-strict undefined), falling back to raw rather than hashing a
    /// silently divergent prompt.
    pub fn render(
        &self,
        messages: &JsonValue,
        tools: Option<&JsonValue>,
        opts: &JinjaRenderOpts,
    ) -> Result<String> {
        let messages = engine_messages(messages)?;
        let tools = engine_tools(tools, opts.tool_choice.as_ref())?;
        match self.render_with(&messages, tools.as_ref(), opts) {
            Ok(s) => Ok(s),
            Err(first) => {
                let Some(flat) = tools.as_ref().map(flatten_tools) else {
                    return Err(first);
                };
                self.render_with(&messages, Some(&flat), opts)
                    .map_err(|_| first)
            }
        }
    }

    fn render_with(
        &self,
        messages: &JsonValue,
        tools: Option<&JsonValue>,
        opts: &JinjaRenderOpts,
    ) -> Result<String> {
        let tmpl = self
            .env
            .get_template(TEMPLATE_NAME)
            .context("chat template not registered")?;
        let mut ctx: BTreeMap<&str, JinjaValue> = BTreeMap::new();
        // Template kwargs first so the renderer-owned names below win.
        for (k, v) in &opts.template_kwargs {
            ctx.insert(k.as_str(), JinjaValue::from_serialize(v));
        }
        ctx.insert("messages", JinjaValue::from_serialize(messages));
        ctx.insert("add_generation_prompt", JinjaValue::from(true));
        ctx.insert(
            "tools",
            tools.map_or(JinjaValue::from(()), JinjaValue::from_serialize),
        );
        ctx.insert("documents", JinjaValue::from(()));
        for (name, token) in &self.special_tokens {
            ctx.insert(name, JinjaValue::from(token.clone()));
        }
        tmpl.render(ctx).context("render chat template")
    }
}

/// Normalize `messages` the way the engine's generic Jinja path does before
/// rendering:
///
///   * `content` that is absent or `null` becomes `""`;
///   * an assistant turn's `tool_calls[].function.arguments` given as a JSON
///     string is parsed into its object (`normalize_assistant_tool_call_arguments`,
///     strict) — templates iterate it with `.items()`;
///   * a `tool` turn whose content is a list of pure text parts is flattened
///     to their `" "`-joined text (`normalize_tool_content`).
///
/// Arguments that are not a JSON object are an error: the engine rejects the
/// request with 400, so there is no prompt to match.
///
/// RESIDUE: the engine renders pydantic `model_dump()`s, which also carry
/// `null` for every optional field the client omitted, and (for templates it
/// detects as reordering tool results) canonicalizes tool-result order to the
/// `tool_calls` order. Templates that only test those fields render the same.
fn engine_messages(messages: &JsonValue) -> Result<JsonValue> {
    let Some(list) = messages.as_array() else {
        return Ok(messages.clone());
    };
    let mut out = Vec::with_capacity(list.len());
    for message in list {
        let mut message = message.clone();
        if let Some(obj) = message.as_object_mut() {
            if obj.get("content").is_none_or(JsonValue::is_null) {
                obj.insert("content".into(), JsonValue::String(String::new()));
            }
            let role = obj
                .get("role")
                .and_then(JsonValue::as_str)
                .unwrap_or_default();
            if role == "assistant" {
                normalize_tool_call_arguments(obj)?;
            } else if role == "tool" {
                if let Some(text) = flatten_text_parts(&obj["content"]) {
                    obj.insert("content".into(), JsonValue::String(text));
                }
            }
        }
        out.push(message);
    }
    Ok(JsonValue::Array(out))
}

fn normalize_tool_call_arguments(message: &mut JsonMap<String, JsonValue>) -> Result<()> {
    let Some(calls) = message.get_mut("tool_calls").and_then(JsonValue::as_array_mut) else {
        return Ok(());
    };
    for call in calls {
        let Some(function) = call.get_mut("function").and_then(JsonValue::as_object_mut) else {
            continue;
        };
        let Some(JsonValue::String(raw)) = function.get("arguments") else {
            continue;
        };
        let parsed: JsonValue = serde_json::from_str(raw)
            .context("assistant tool call function.arguments must be valid JSON")?;
        if !parsed.is_object() {
            bail!("assistant tool call function.arguments must be a JSON object");
        }
        function.insert("arguments".into(), parsed);
    }
    Ok(())
}

/// `" "`-joined text of a content list made only of OpenAI text parts (or bare
/// strings); `None` for anything else, which the engine leaves untouched.
fn flatten_text_parts(content: &JsonValue) -> Option<String> {
    let parts = content.as_array()?;
    let mut texts = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            JsonValue::String(s) => texts.push(s.as_str()),
            JsonValue::Object(o) if o.get("type").and_then(JsonValue::as_str) == Some("text") => {
                texts.push(
                    o.get("text")
                        .and_then(JsonValue::as_str)
                        .unwrap_or_default(),
                )
            }
            _ => return None,
        }
    }
    Some(texts.join(" "))
}

/// The `tools` the engine hands the template (`_process_messages`):
///
///   * none unless the request carries a non-empty top-level `tools` and
///     `tool_choice` is not `"none"`;
///   * a named-function `tool_choice` keeps only that function (none if it
///     names no listed tool);
///   * each tool is emitted as its pydantic `Tool.model_dump()`, see
///     [`dump_tool`].
fn engine_tools(
    tools: Option<&JsonValue>,
    tool_choice: Option<&JsonValue>,
) -> Result<Option<JsonValue>> {
    let Some(tools) = tools.and_then(JsonValue::as_array).filter(|t| !t.is_empty()) else {
        return Ok(None);
    };
    let wanted = match tool_choice {
        Some(JsonValue::String(s)) if s == "none" => return Ok(None),
        Some(JsonValue::Object(choice)) => Some(
            choice
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(JsonValue::as_str)
                .unwrap_or_default()
                .to_owned(),
        ),
        _ => None,
    };
    let mut dumped = Vec::with_capacity(tools.len());
    for tool in tools {
        let tool = dump_tool(tool)?;
        if let Some(name) = &wanted {
            if tool["function"]["name"].as_str() != Some(name.as_str()) {
                continue;
            }
        }
        dumped.push(tool);
    }
    Ok((!dumped.is_empty()).then_some(JsonValue::Array(dumped)))
}

/// Reproduce `Tool.model_dump()` from `protocol.py`, field order included:
/// `{"type", "function": {"description", "name", "parameters", "strict"[,
/// "defer_loading"]}, "defer_loading"}`. Omitted optionals dump as `null`
/// (`strict` as `false`), unknown keys are dropped, and a tool-level
/// `defer_loading` propagates into a function that has none. Templates that
/// print whole tool objects (`tool.items()` → `tojson`) see these bytes.
fn dump_tool(tool: &JsonValue) -> Result<JsonValue> {
    let function = tool
        .get("function")
        .and_then(JsonValue::as_object)
        .context("tool is missing its function object")?;
    let name = function
        .get("name")
        .and_then(JsonValue::as_str)
        .context("tool function is missing its name")?;
    let tool_defer = tool
        .get("defer_loading")
        .cloned()
        .unwrap_or(JsonValue::Null);
    let fn_defer = function
        .get("defer_loading")
        .cloned()
        .filter(|v| !v.is_null())
        .unwrap_or_else(|| tool_defer.clone());

    let mut f = JsonMap::new();
    f.insert(
        "description".into(),
        function
            .get("description")
            .cloned()
            .unwrap_or(JsonValue::Null),
    );
    f.insert("name".into(), JsonValue::String(name.to_owned()));
    f.insert(
        "parameters".into(),
        function
            .get("parameters")
            .cloned()
            .unwrap_or(JsonValue::Null),
    );
    f.insert(
        "strict".into(),
        JsonValue::Bool(
            function
                .get("strict")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false),
        ),
    );
    if !fn_defer.is_null() {
        f.insert("defer_loading".into(), fn_defer);
    }

    let mut t = JsonMap::new();
    t.insert(
        "type".into(),
        tool.get("type")
            .cloned()
            .unwrap_or_else(|| JsonValue::String("function".into())),
    );
    t.insert("function".into(), JsonValue::Object(f));
    t.insert("defer_loading".into(), tool_defer);
    Ok(JsonValue::Object(t))
}

/// The engine's retry shape for templates that expect bare function objects.
fn flatten_tools(tools: &JsonValue) -> JsonValue {
    match tools.as_array() {
        Some(list) => JsonValue::Array(
            list.iter()
                .map(|t| t.get("function").cloned().unwrap_or_else(|| t.clone()))
                .collect(),
        ),
        None => tools.clone(),
    }
}

/// `tojson` as HuggingFace defines it for chat templates:
/// `json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
/// separators=separators, sort_keys=sort_keys)` with `ensure_ascii=False` by
/// default. `indent` / `separators` are not reproduced; a template that passes
/// them fails to render and routes on the raw-text fallback instead of hashing
/// a divergent prompt.
fn py_tojson(value: JinjaValue, kwargs: Kwargs) -> std::result::Result<String, JinjaError> {
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    for unsupported in ["indent", "separators"] {
        let v: Option<JinjaValue> = kwargs.get(unsupported)?;
        if v.is_some_and(|v| !v.is_none()) {
            return Err(JinjaError::new(
                JinjaErrorKind::InvalidOperation,
                format!("tojson({unsupported}=...) is not reproduced by the router"),
            ));
        }
    }
    kwargs.assert_all_used()?;
    if value.is_undefined() {
        return Err(JinjaError::new(
            JinjaErrorKind::UndefinedError,
            "tojson of an undefined value",
        ));
    }
    let json = serde_json::to_value(&value).map_err(|e| {
        JinjaError::new(
            JinjaErrorKind::BadSerialization,
            "tojson: value is not JSON",
        )
        .with_source(e)
    })?;
    let json = if sort_keys.unwrap_or(false) {
        pyjson::deep_sort(&json)
    } else {
        json
    };
    let out = pyjson::py_json(&json);
    Ok(if ensure_ascii.unwrap_or(false) {
        escape_non_ascii(&out)
    } else {
        out
    })
}

/// Python's `ensure_ascii=True` escaping: every non-ASCII code point as
/// lowercase `\uXXXX`, astral ones as a UTF-16 surrogate pair. Non-ASCII can
/// only occur inside JSON strings, so escaping the whole document is exact.
fn escape_non_ascii(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut buf = [0u16; 2];
    for c in s.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            for unit in c.encode_utf16(&mut buf) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out
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

    fn render(tmpl: &ChatTemplate, messages: &JsonValue) -> Result<String> {
        tmpl.render(messages, None, &JinjaRenderOpts::default())
    }

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
        let out = render(&tmpl, &messages()).unwrap();
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
        assert!(render(&tmpl, &messages())
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
        assert!(render(&tmpl, &messages())
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
        assert_eq!(render(&tmpl, &json!([])).unwrap(), "<|begin|>X");
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
        let err = render(&tmpl, &messages()).unwrap_err();
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
        assert_eq!(render(&tmpl, &messages()).unwrap(), "BE BRIEF");
    }

    /// An absent special token renders as `""` exactly like an undefined name
    /// under HuggingFace's jinja2 — never as minijinja's literal `"none"`,
    /// which would corrupt block 0 (and thus every chained block hash).
    #[test]
    fn absent_special_tokens_render_empty() {
        let cfg = json!({"chat_template": "A{{ bos_token }}{{ pad_token }}B"});
        let tmpl = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        assert_eq!(render(&tmpl, &json!([])).unwrap(), "AB");
    }

    /// Every name in HuggingFace's `special_tokens_map` is threaded from
    /// `tokenizer_config.json`, not just `bos_token`/`eos_token`.
    #[test]
    fn named_special_tokens_from_config_are_supplied() {
        let cfg = json!({
            "chat_template": "{{ pad_token }}|{{ unk_token }}",
            "pad_token": "<pad>",
            "unk_token": {"content": "<unk>"},
        });
        let tmpl = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        assert_eq!(render(&tmpl, &json!([])).unwrap(), "<pad>|<unk>");
    }

    /// Printing a variable the router doesn't supply is a render error
    /// (semi-strict undefined) so the caller falls back to raw-text hashing,
    /// instead of rendering a plausible-but-divergent prompt.
    #[test]
    fn printing_unsupplied_variable_fails_render() {
        let cfg = json!({
            "chat_template": "{{ custom_kwarg }}",
            "bos_token": "<s>",
        });
        let tmpl = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        render(&tmpl, &messages()).unwrap_err();
    }

    /// Undefined names stay usable in if-tests (semi-strict only rejects
    /// printing them); common `{% if enable_thinking is defined %}`-style
    /// guards must keep rendering.
    #[test]
    fn undefined_in_if_test_is_permitted() {
        let cfg = json!({
            "chat_template": "{% if enable_thinking is defined and enable_thinking %}T{% endif %}X",
        });
        let tmpl = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        assert_eq!(render(&tmpl, &messages()).unwrap(), "X");
    }

    /// `tools` is `none` in the render context — the same context HuggingFace
    /// renders with for a request that carries no tools — so tools-branching
    /// templates take the no-tools path instead of erroring or mis-branching.
    #[test]
    fn tools_supplied_as_none_takes_no_tools_branch() {
        let cfg = json!({
            "chat_template": "{% if tools is not none %}TOOLS{% endif %}X",
        });
        let tmpl = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        assert_eq!(render(&tmpl, &messages()).unwrap(), "X");
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
        assert_eq!(render(&tmpl, &messages()).unwrap(), "system\nuser\n");
    }

    fn tmpl(src: &str) -> ChatTemplate {
        ChatTemplate::from_tokenizer_config(&json!({ "chat_template": src }))
            .unwrap()
            .unwrap()
    }

    /// `tojson` is Python's `json.dumps`: `", "`/`": "` separators, key order
    /// kept, no HTML escaping, `ensure_ascii` opt-in, `sort_keys` honored.
    #[test]
    fn tojson_matches_python_json_dumps() {
        let t = tmpl(
            "{{ messages[0].v | tojson }}|{{ messages[0].v | tojson(ensure_ascii=True) }}|{{ messages[0].v | tojson(sort_keys=True) }}",
        );
        let msgs = json!([{"v": {"z": "<é>", "a": [1, 1e-6]}}]);
        // Built with `char::from(92)` (a backslash) so no editor or tool can
        // collapse the escape sequence the assertion is about.
        let escaped = format!("<{}u00e9>", char::from(92));
        assert_eq!(
            render(&t, &msgs).unwrap(),
            format!(
                r#"{{"z": "<é>", "a": [1, 1e-06]}}|{{"z": "{escaped}", "a": [1, 1e-06]}}|{{"a": [1, 1e-06], "z": "<é>"}}"#
            )
        );
        let t = tmpl("{{ messages | tojson(indent=2) }}");
        render(&t, &msgs).unwrap_err();
    }

    /// String `arguments` are parsed into their object (templates iterate them
    /// with `.items()`), null content becomes `""`, and text-part tool results
    /// are flattened — the engine's pre-render normalization.
    #[test]
    fn messages_are_normalized_like_the_engine() {
        let t = tmpl(
            "{% for m in messages %}[{{ m.role }}:{{ m.content }}]{% for tc in m.tool_calls or [] %}{% for k, v in tc.function.arguments.items() %}{{ k }}={{ v }};{% endfor %}{% endfor %}{% endfor %}",
        );
        let msgs = json!([
            {"role": "assistant", "content": null, "tool_calls": [
                {"type": "function", "function": {"name": "f", "arguments": "{\"b\": 1, \"a\": \"x\"}"}}]},
            {"role": "tool", "content": [{"type": "text", "text": "p1"}, {"type": "text", "text": "p2"}]}
        ]);
        assert_eq!(
            render(&t, &msgs).unwrap(),
            "[assistant:]b=1;a=x;[tool:p1 p2]"
        );

        let bad = json!([{"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "f", "arguments": "[1]"}}]}]);
        render(&t, &bad).unwrap_err();
    }

    /// Tools reach the template as pydantic `Tool.model_dump()`s, filtered by
    /// `tool_choice` exactly as the engine does.
    #[test]
    fn tools_are_dumped_and_filtered_like_the_engine() {
        let t = tmpl("{% if tools %}{{ tools | tojson }}{% else %}NONE{% endif %}");
        let tools = json!([
            {"type": "function", "defer_loading": true,
             "function": {"parameters": {"type": "object"}, "name": "a", "extra": 1}},
            {"type": "function", "function": {"name": "b", "description": "d", "strict": true}}
        ]);
        let with = |choice: Option<JsonValue>| {
            let opts = JinjaRenderOpts {
                tool_choice: choice,
                ..Default::default()
            };
            t.render(&json!([]), Some(&tools), &opts).unwrap()
        };
        assert_eq!(
            with(None),
            r#"[{"type": "function", "function": {"description": null, "name": "a", "parameters": {"type": "object"}, "strict": false, "defer_loading": true}, "defer_loading": true}, {"type": "function", "function": {"description": "d", "name": "b", "parameters": null, "strict": true}, "defer_loading": null}]"#
        );
        assert_eq!(with(Some(json!("none"))), "NONE");
        assert!(
            with(Some(json!({"type": "function", "function": {"name": "b"}})))
                .starts_with(r#"[{"type": "function", "function": {"description": "d""#)
        );
        assert_eq!(
            with(Some(
                json!({"type": "function", "function": {"name": "zzz"}})
            )),
            "NONE"
        );
        assert_eq!(
            t.render(&json!([]), Some(&json!([])), &JinjaRenderOpts::default())
                .unwrap(),
            "NONE"
        );
    }

    /// `chat_template_kwargs.reasoning_effort` wins over the top-level field,
    /// other kwargs pass through, and renderer-owned names cannot be shadowed.
    #[test]
    fn template_kwargs_resolve_like_the_engine() {
        let opts = JinjaRenderOpts::resolve(&json!({
            "reasoning_effort": "low",
            "chat_template_kwargs": {"reasoning_effort": "high", "clear_thinking": true, "messages": 1}
        }));
        assert_eq!(
            JsonValue::Object(opts.template_kwargs.clone()),
            json!({"reasoning_effort": "high", "clear_thinking": true})
        );
        let top_only = JinjaRenderOpts::resolve(&json!({
            "reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": null}
        }));
        assert_eq!(top_only.template_kwargs["reasoning_effort"], json!("low"));

        let t = tmpl("{{ reasoning_effort }}/{{ clear_thinking }}/{{ messages | length }}");
        // Booleans print as Python's `True`, as jinja2 renders them.
        assert_eq!(t.render(&json!([]), None, &opts).unwrap(), "high/True/0");
    }
}
