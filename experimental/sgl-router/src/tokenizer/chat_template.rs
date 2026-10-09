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

use anyhow::{Context, Result};
use minijinja::{
    value::{Kwargs, Value as JinjaValue},
    Environment, Error as JinjaError, ErrorKind as JinjaErrorKind, UndefinedBehavior,
};
use std::collections::BTreeMap;

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

/// A compiled chat template plus the special-token strings it references.
///
/// One per model, built once at startup from `tokenizer_config.json` and held
/// in the [`super::TokenizerRegistry`]. Rendering is read-only and thread-safe.
pub struct ChatTemplate {
    env: Environment<'static>,
    /// `(name, token)` pairs for [`SPECIAL_TOKEN_KEYS`]; absent tokens are `""`.
    special_tokens: Vec<(&'static str, String)>,
    /// Rendering with [`JinjaRenderOpts`] is verified token-identical to the
    /// engine for this model (see [`super::ForwardParity::JinjaFull`]).
    full_forwarding: bool,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("special_tokens", &self.special_tokens)
            .field("full_forwarding", &self.full_forwarding)
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
        // Undefined behaves as in jinja2 (`msg.name == 'x'` is false, if-tests
        // pass), except that printing it is an error: a variable the router
        // didn't supply would render as `""` and silently diverge from the
        // engine's prompt, so the caller falls back to raw-text hashing.
        env.set_undefined_behavior(UndefinedBehavior::Lenient);
        env.set_formatter(|out, state, value| {
            if value.is_undefined() {
                return Err(JinjaError::new(
                    JinjaErrorKind::UndefinedError,
                    "printed a variable the router does not supply",
                ));
            }
            minijinja::escape_formatter(out, state, value)
        });
        // Python str/dict methods used by real templates (.startswith, .items,
        // .strip, ...) that minijinja doesn't implement natively.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        // transformers' template filters. minijinja's own `tojson` takes no
        // `ensure_ascii` (a compile-time-valid, render-time error) and
        // HTML-escapes `<>&'`; `fromjson` does not exist.
        env.add_filter("tojson", hf_tojson);
        env.add_filter("fromjson", hf_fromjson);
        env.add_template_owned(TEMPLATE_NAME, strip_generation_tags(&template_src))
            .context("compile chat template from tokenizer_config.json")?;

        Ok(Some(Self {
            env,
            special_tokens,
            full_forwarding: false,
        }))
    }

    /// Mark this model's rendering as verified token-identical to the engine
    /// for every request shape [`JinjaRenderOpts`] mirrors, so its ids may be
    /// forwarded beyond the conservative subset.
    pub(crate) fn with_full_forwarding(mut self) -> Self {
        self.full_forwarding = true;
        self
    }

    pub(crate) fn full_forwarding(&self) -> bool {
        self.full_forwarding
    }

    /// Render `messages` with no tools and no template kwargs (see
    /// [`ChatTemplate::render_with`]).
    pub fn render(&self, messages: &serde_json::Value) -> Result<String> {
        self.render_with(messages, &JinjaRenderOpts::default())
    }

    /// Render `messages` (the request's `messages` array) into the prompt text
    /// the engine would tokenize, with `add_generation_prompt = true`.
    ///
    /// `messages` is passed through as-is; templates expect string `content`.
    /// Multimodal content arrays are out of scope (text-only routing): a
    /// template may stringify the array (divergent hashes → min-load) or error
    /// (raw prompt-text fallback); neither fails the request.
    ///
    /// `opts` carries what SGLang's generic Jinja path threads into
    /// `apply_chat_template`: `tools` (`none` when absent — the no-tools path)
    /// and the merged template kwargs. `documents` is always `none`. Any other
    /// variable the template prints is a render error,
    /// falling back to raw rather than hashing a silently divergent prompt.
    pub fn render_with(
        &self,
        messages: &serde_json::Value,
        opts: &JinjaRenderOpts,
    ) -> Result<String> {
        let tmpl = self
            .env
            .get_template(TEMPLATE_NAME)
            .context("chat template not registered")?;
        let mut ctx: BTreeMap<&str, JinjaValue> = BTreeMap::new();
        for (name, token) in &self.special_tokens {
            ctx.insert(name, JinjaValue::from(token.clone()));
        }
        // transformers renders with `**special_tokens_map, **kwargs`, so a
        // kwarg may override a special token but not the names it passes
        // explicitly (those are a TypeError engine-side; the forwarding
        // predicate withholds such requests).
        for (key, value) in &opts.kwargs {
            if !RESERVED_RENDER_KEYS.contains(&key.as_str()) {
                ctx.insert(key, JinjaValue::from_serialize(value));
            }
        }
        ctx.insert("messages", JinjaValue::from_serialize(messages));
        ctx.insert("add_generation_prompt", JinjaValue::from(true));
        ctx.insert(
            "tools",
            opts.tools
                .as_ref()
                .map_or_else(|| JinjaValue::from(()), JinjaValue::from_serialize),
        );
        ctx.insert("documents", JinjaValue::from(()));
        tmpl.render(ctx).context("render chat template")
    }
}

/// Names `apply_chat_template` passes to the template explicitly; a template
/// kwarg cannot replace them.
pub(crate) const RESERVED_RENDER_KEYS: [&str; 4] =
    ["messages", "tools", "documents", "add_generation_prompt"];

/// The request-level inputs SGLang's generic Jinja path (`serving_chat.py`
/// `_apply_jinja_template`) threads into `apply_chat_template`, resolved from
/// the raw request body the way the engine resolves them from its pydantic
/// model. Without these a tools / `reasoning_effort` / `chat_template_kwargs`
/// request renders the default prompt and its hashes never match the engine's.
#[derive(Clone, Debug, Default)]
pub struct JinjaRenderOpts {
    /// `request.tools` as `Tool.model_dump()` emits them, after `tool_choice`
    /// selection; `None` renders the no-tools path.
    pub tools: Option<serde_json::Value>,
    /// The merged `extra_template_kwargs`.
    pub kwargs: serde_json::Map<String, serde_json::Value>,
}

impl JinjaRenderOpts {
    /// `engine_defaults` is the engine's `--default-chat-template-kwargs`
    /// (see [`crate::workers::introspect::EngineChatTemplate`]).
    pub fn resolve(
        request: &serde_json::Value,
        engine_defaults: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Self {
        JinjaRenderOpts {
            tools: sglang_template_tools(request),
            kwargs: sglang_template_kwargs(request, engine_defaults),
        }
    }
}

/// Mirror SGLang's tool selection for the template: nothing for
/// `tool_choice: "none"`, only the named tool for a function `tool_choice`,
/// else every request-level tool, each dumped like pydantic's `Tool`.
///
/// Engine: `serving_chat.py` `_process_messages` (`tools = [item.model_dump()
/// ...]`). Message-level tools gate that branch but are never rendered from it;
/// the forwarding predicate withholds requests that carry them.
fn sglang_template_tools(request: &serde_json::Value) -> Option<serde_json::Value> {
    let tools = request.get("tools")?.as_array().filter(|t| !t.is_empty())?;
    let selected: Vec<serde_json::Value> = match request.get("tool_choice") {
        Some(serde_json::Value::String(s)) if s == "none" => return None,
        Some(serde_json::Value::Object(choice)) => {
            let name = choice.get("function").and_then(|f| f.get("name"))?;
            tools
                .iter()
                .filter(|t| t.get("function").and_then(|f| f.get("name")) == Some(name))
                .map(dump_tool)
                .collect()
        }
        _ => tools.iter().map(dump_tool).collect(),
    };
    (!selected.is_empty()).then_some(serde_json::Value::Array(selected))
}

/// `Tool.model_dump()`: declared fields only, in declaration order, defaults
/// filled — `{type, function: {description, name, parameters, strict
/// [, defer_loading]}, defer_loading}`. `Function` drops a `None`
/// `defer_loading`; `Tool` keeps its own, and propagates it into `function`
/// when the function has none (`_propagate_defer_loading`).
fn dump_tool(tool: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    let field = |obj: &Value, key: &str| obj.get(key).filter(|v| !v.is_null()).cloned();
    let func = tool.get("function").cloned().unwrap_or(Value::Null);
    let tool_defer = field(tool, "defer_loading");
    let mut f = serde_json::Map::new();
    f.insert(
        "description".into(),
        field(&func, "description").unwrap_or(Value::Null),
    );
    f.insert(
        "name".into(),
        func.get("name").cloned().unwrap_or(Value::Null),
    );
    f.insert(
        "parameters".into(),
        field(&func, "parameters").unwrap_or(Value::Null),
    );
    f.insert(
        "strict".into(),
        field(&func, "strict").unwrap_or(Value::Bool(false)),
    );
    if let Some(d) = field(&func, "defer_loading").or_else(|| tool_defer.clone()) {
        f.insert("defer_loading".into(), d);
    }
    let mut t = serde_json::Map::new();
    t.insert(
        "type".into(),
        field(tool, "type").unwrap_or_else(|| Value::String("function".into())),
    );
    t.insert("function".into(), Value::Object(f));
    t.insert("defer_loading".into(), tool_defer.unwrap_or(Value::Null));
    Value::Object(t)
}

/// The engine's `extra_template_kwargs`: a top-level `reasoning_effort` sets
/// `thinking` / `enable_thinking` (`effort != "none"`) unless the request does
/// (`ChatCompletionRequest` validator); engine defaults fill the keys the
/// request's `chat_template_kwargs` omits; and the request's effort
/// (`chat_template_kwargs.reasoning_effort`, else top-level `reasoning_effort`)
/// wins over a default one.
///
/// SGLang's `serving_chat.py` intends this (`_process_messages` only adopts a
/// default effort when the request has none) but its final
/// `extra_template_kwargs.update(chat_template_kwargs)` lets a default
/// `reasoning_effort` override the request's; the router follows the intended
/// behavior, so forwarded ids honor the client's effort.
fn sglang_template_kwargs(
    request: &serde_json::Value,
    engine_defaults: Option<&serde_json::Map<String, serde_json::Value>>,
) -> serde_json::Map<String, serde_json::Value> {
    let set = |v: &&serde_json::Value| !v.is_null();
    let mut ctk = request
        .get("chat_template_kwargs")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    if let Some(top) = request.get("reasoning_effort").filter(set) {
        let thinking = serde_json::Value::Bool(top.as_str() != Some("none"));
        for key in ["thinking", "enable_thinking"] {
            ctk.entry(key).or_insert_with(|| thinking.clone());
        }
    }
    let effort = ctk
        .remove("reasoning_effort")
        .filter(|v| !v.is_null())
        .or_else(|| request.get("reasoning_effort").filter(set).cloned());
    for (k, v) in engine_defaults.into_iter().flatten() {
        ctk.entry(k.clone()).or_insert_with(|| v.clone());
    }
    if let Some(effort) = effort {
        ctk.insert("reasoning_effort".into(), effort);
    }
    ctk
}

/// transformers' `tojson` filter: `json.dumps(x, ensure_ascii=False,
/// indent=None, separators=None, sort_keys=False)`. Unlike minijinja's builtin
/// it accepts these kwargs, uses Python's default separators (`", "`, `": "`,
/// or `","` with an indent) and does not HTML-escape — a byte difference in a
/// rendered tool schema shifts every block hash after it.
fn hf_tojson(value: JinjaValue, kwargs: Kwargs) -> std::result::Result<JinjaValue, JinjaError> {
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let indent: Option<usize> = kwargs.get("indent")?;
    let separators: Option<Vec<String>> = kwargs.get("separators")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    kwargs.assert_all_used()?;
    let (item_sep, key_sep) = match separators.as_deref() {
        Some([item, key]) => (item.clone(), key.clone()),
        Some(_) => {
            return Err(JinjaError::new(
                JinjaErrorKind::InvalidOperation,
                "tojson separators must be a pair",
            ))
        }
        None if indent.is_some() => (",".to_owned(), ": ".to_owned()),
        None => (", ".to_owned(), ": ".to_owned()),
    };
    let json = serde_json::to_value(&value)
        .map_err(|e| JinjaError::new(JinjaErrorKind::InvalidOperation, e.to_string()))?;
    let style = PyJsonStyle {
        ensure_ascii: ensure_ascii.unwrap_or(false),
        indent,
        item_sep,
        key_sep,
        sort_keys: sort_keys.unwrap_or(false),
    };
    let mut out = String::new();
    style.write(&json, 0, &mut out);
    Ok(JinjaValue::from_safe_string(out))
}

/// transformers' `fromjson` filter: `json.loads`.
fn hf_fromjson(text: String) -> std::result::Result<JinjaValue, JinjaError> {
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| JinjaError::new(JinjaErrorKind::InvalidOperation, format!("fromjson: {e}")))?;
    Ok(JinjaValue::from_serialize(&json))
}

/// Python `json.dumps` output formatting.
struct PyJsonStyle {
    ensure_ascii: bool,
    indent: Option<usize>,
    item_sep: String,
    key_sep: String,
    sort_keys: bool,
}

impl PyJsonStyle {
    fn write(&self, v: &serde_json::Value, depth: usize, out: &mut String) {
        use serde_json::Value;
        match v {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => match (n.as_i64(), n.as_u64(), n.as_f64()) {
                (Some(i), _, _) => out.push_str(&i.to_string()),
                (_, Some(u), _) => out.push_str(&u.to_string()),
                (_, _, Some(f)) => out.push_str(&py_float_repr(f)),
                _ => out.push_str(&n.to_string()),
            },
            Value::String(s) => self.write_str(s, out),
            Value::Array(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(&self.item_sep);
                    }
                    self.newline(depth + 1, out);
                    self.write(item, depth + 1, out);
                }
                self.newline(depth, out);
                out.push(']');
            }
            Value::Object(map) => {
                if map.is_empty() {
                    out.push_str("{}");
                    return;
                }
                let mut entries: Vec<_> = map.iter().collect();
                if self.sort_keys {
                    entries.sort_by(|a, b| a.0.cmp(b.0));
                }
                out.push('{');
                for (i, (k, item)) in entries.into_iter().enumerate() {
                    if i > 0 {
                        out.push_str(&self.item_sep);
                    }
                    self.newline(depth + 1, out);
                    self.write_str(k, out);
                    out.push_str(&self.key_sep);
                    self.write(item, depth + 1, out);
                }
                self.newline(depth, out);
                out.push('}');
            }
        }
    }

    fn newline(&self, depth: usize, out: &mut String) {
        if let Some(width) = self.indent {
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', width * depth));
        }
    }

    fn write_str(&self, s: &str, out: &mut String) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{08}' => out.push_str("\\b"),
                '\u{0c}' => out.push_str("\\f"),
                c if (c as u32) < 0x20 || (self.ensure_ascii && (c as u32) > 0x7e) => {
                    let mut units = [0u16; 2];
                    for unit in c.encode_utf16(&mut units) {
                        out.push_str(&format!("\\u{unit:04x}"));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
}

/// Python `repr(float)`: shortest round-trip digits, positional for exponents
/// in `[-4, 16)`, else `d.ddde±XX`.
fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    let sci = format!("{f:e}");
    let (mantissa, exp) = sci.split_once('e').expect("{:e} always has an exponent");
    let exp: i32 = exp.parse().expect("{:e} exponent is an integer");
    if (-4..16).contains(&exp) {
        let s = format!("{f}");
        if s.contains('.') {
            s
        } else {
            format!("{s}.0")
        }
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.abs())
    }
}

/// Remove `{% generation %}` / `{% endgeneration %}` (with optional `-`
/// whitespace control) and keep the body.
///
/// transformers registers these as an extension that only marks assistant spans
/// for `return_assistant_tokens_mask`; they render nothing. minijinja has no
/// such statement, so the whole template would fail to compile (Step-5 ships
/// one) and routing would fall back to raw text.
fn strip_generation_tags(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(start) = rest.find("{%") {
        let Some(len) = rest[start..].find("%}").map(|e| e + 2) else {
            break;
        };
        let inner = rest[start + 2..start + len - 2]
            .trim_start_matches('-')
            .trim_end_matches('-')
            .trim();
        out.push_str(&rest[..start]);
        if inner != "generation" && inner != "endgeneration" {
            out.push_str(&rest[start..start + len]);
        }
        rest = &rest[start + len..];
    }
    out.push_str(rest);
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
    fn strip_generation_tags_keeps_body_and_other_tags() {
        assert_eq!(
            strip_generation_tags("A{% generation %}B{% endgeneration %}C"),
            "ABC"
        );
        assert_eq!(
            strip_generation_tags("A{%- generation -%}B{%-endgeneration%}C"),
            "ABC"
        );
        assert_eq!(
            strip_generation_tags("{% if x %}{% generation %}y{% endgeneration %}{% endif %}{% set generation_x = 1 %}"),
            "{% if x %}y{% endif %}{% set generation_x = 1 %}"
        );
        assert_eq!(
            strip_generation_tags("no tags {{ a }} %}"),
            "no tags {{ a }} %}"
        );
        assert_eq!(
            strip_generation_tags("dangling {% generation"),
            "dangling {% generation"
        );
    }

    /// Step-5 wraps assistant content in `{% generation %}`; minijinja has no
    /// such statement, so without stripping the template fails to compile.
    fn render_tojson(expr: &str, value: serde_json::Value) -> String {
        let cfg = json!({ "chat_template": format!("{{{{ messages | {expr} }}}}") });
        ChatTemplate::from_tokenizer_config(&cfg)
            .unwrap()
            .unwrap()
            .render(&value)
            .unwrap()
    }

    /// Expected strings are Python's `json.dumps` output for the same value
    /// (transformers' `tojson` is a thin wrapper over it).
    #[test]
    fn tojson_matches_python_json_dumps() {
        let v = json!({"b":1,"a":[1.5,1e20,1.5e-5,0.0001,null,true],
                       "s":"<x>&'中😀\n\"\\","e":{},"l":[]});
        assert_eq!(
            render_tojson("tojson", v.clone()),
            r#"{"b": 1, "a": [1.5, 1e+20, 1.5e-05, 0.0001, null, true], "s": "<x>&'中😀\n\"\\", "e": {}, "l": []}"#
        );
        assert_eq!(
            render_tojson("tojson(ensure_ascii=False)", v.clone()),
            render_tojson("tojson", v.clone())
        );
        assert_eq!(
            render_tojson("tojson(ensure_ascii=True)", v),
            r#"{"b": 1, "a": [1.5, 1e+20, 1.5e-05, 0.0001, null, true], "s": "<x>&'\u4e2d\ud83d\ude00\n\"\\", "e": {}, "l": []}"#
        );
        assert_eq!(
            render_tojson("tojson(indent=2)", json!({"k":[1,{"z":2}]})),
            "{\n  \"k\": [\n    1,\n    {\n      \"z\": 2\n    }\n  ]\n}"
        );
        assert_eq!(
            render_tojson(
                "tojson(sort_keys=True, separators=[',', ':'])",
                json!({"b":1,"a":2})
            ),
            r#"{"a":2,"b":1}"#
        );
    }

    #[test]
    fn fromjson_parses_and_rejects_bad_json() {
        let cfg =
            json!({"chat_template": "{% set a = messages | fromjson %}{{ a.x }}-{{ a.y[1] }}"});
        let t = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        assert_eq!(t.render(&json!(r#"{"x":"v","y":[0,7]}"#)).unwrap(), "v-7");
        assert!(t.render(&json!("not json")).is_err());
    }

    #[test]
    fn render_with_threads_tools_and_kwargs_but_not_reserved_names() {
        let cfg = json!({
            "chat_template": "{{ bos_token }}{% if tools %}T={{ tools | length }};{% endif %}{% if reasoning_effort is defined %}R={{ reasoning_effort }};{% endif %}{{ messages | length }}",
            "bos_token": "<s>"
        });
        let t = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        let msgs = json!([{"role": "user", "content": "hi"}]);
        assert_eq!(t.render(&msgs).unwrap(), "<s>1");
        let mut opts = JinjaRenderOpts {
            tools: Some(json!([{"type": "function"}, {"type": "function"}])),
            ..Default::default()
        };
        opts.kwargs.insert("reasoning_effort".into(), json!("low"));
        opts.kwargs.insert("bos_token".into(), json!("[B]"));
        opts.kwargs.insert("messages".into(), json!([1, 2, 3]));
        assert_eq!(t.render_with(&msgs, &opts).unwrap(), "[B]T=2;R=low;1");
    }

    #[test]
    fn template_tools_follow_tool_choice_and_pydantic_dump() {
        let tools = json!([
            {"type": "function", "function": {"name": "a", "parameters": {"type": "object"}}, "extra": 1},
            {"function": {"name": "b", "description": "B", "strict": true}, "defer_loading": true}
        ]);
        let dumped_a = json!({"type": "function", "function": {"description": null, "name": "a",
            "parameters": {"type": "object"}, "strict": false}, "defer_loading": null});
        let dumped_b = json!({"type": "function", "function": {"description": "B", "name": "b",
            "parameters": null, "strict": true, "defer_loading": true}, "defer_loading": true});
        let resolve = |extra: serde_json::Value| {
            let mut req = json!({ "tools": tools });
            req.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            JinjaRenderOpts::resolve(&req, None).tools
        };
        assert_eq!(resolve(json!({})), Some(json!([dumped_a, dumped_b])));
        assert_eq!(
            resolve(json!({"tool_choice": "required"})),
            Some(json!([dumped_a, dumped_b]))
        );
        assert_eq!(resolve(json!({"tool_choice": "none"})), None);
        assert_eq!(
            resolve(json!({"tool_choice": {"type": "function", "function": {"name": "b"}}})),
            Some(json!([dumped_b]))
        );
        assert_eq!(
            resolve(json!({"tool_choice": {"type": "function", "function": {"name": "zz"}}})),
            None
        );
        // Key order is part of the contract: `tojson` emits it verbatim.
        let a = serde_json::to_string(&resolve(json!({})).unwrap()[0]).unwrap();
        assert!(a.starts_with(r#"{"type":"function","function":{"description":null,"name":"a""#));
        assert_eq!(
            JinjaRenderOpts::resolve(&json!({"tools": []}), None).tools,
            None
        );
    }

    /// `testdata/jinja_template_kwargs_cases.json`: expectations produced by
    /// SGLang's own request validation and kwargs merge (see its `_doc`).
    #[test]
    fn template_kwargs_follow_engine_precedence() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("testdata/jinja_template_kwargs_cases.json"))
                .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let got =
                JinjaRenderOpts::resolve(&case["request"], case["defaults"].as_object()).kwargs;
            assert_eq!(
                serde_json::Value::Object(got),
                case["expected"],
                "request {} defaults {}",
                case["request"],
                case["defaults"]
            );
        }
    }

    #[test]
    fn template_with_generation_tags_compiles_and_renders_like_without() {
        let with = "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{% if m['role'] == 'assistant' %}{% generation %}{{ m['content'] }}{% endgeneration %}{% else %}{{ m['content'] }}{% endif %}{% endfor %}";
        let without = "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{% if m['role'] == 'assistant' %}{{ m['content'] }}{% else %}{{ m['content'] }}{% endif %}{% endfor %}";
        let msgs =
            json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]);
        let a = ChatTemplate::from_tokenizer_config(
            &json!({"chat_template": with, "bos_token": "<s>"}),
        )
        .unwrap()
        .unwrap()
        .render(&msgs)
        .unwrap();
        let b = ChatTemplate::from_tokenizer_config(
            &json!({"chat_template": without, "bos_token": "<s>"}),
        )
        .unwrap()
        .unwrap()
        .render(&msgs)
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(a, "<s><|user|>hi<|assistant|>yo");
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

    /// An absent special token renders as `""` exactly like an undefined name
    /// under HuggingFace's jinja2 — never as minijinja's literal `"none"`,
    /// which would corrupt block 0 (and thus every chained block hash).
    #[test]
    fn absent_special_tokens_render_empty() {
        let cfg = json!({"chat_template": "A{{ bos_token }}{{ pad_token }}B"});
        let tmpl = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        assert_eq!(tmpl.render(&json!([])).unwrap(), "AB");
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
        assert_eq!(tmpl.render(&json!([])).unwrap(), "<pad>|<unk>");
    }

    /// Printing a variable the router doesn't supply is a render error so the
    /// caller falls back to raw-text hashing, instead of rendering a
    /// plausible-but-divergent prompt.
    #[test]
    fn printing_unsupplied_variable_fails_render() {
        let cfg = json!({
            "chat_template": "{{ custom_kwarg }}",
            "bos_token": "<s>",
        });
        let tmpl = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        tmpl.render(&messages()).unwrap_err();
    }

    /// Undefined names stay usable in if-tests; common
    /// `{% if enable_thinking is defined %}`-style guards must keep rendering.
    #[test]
    fn undefined_in_if_test_is_permitted() {
        let cfg = json!({
            "chat_template": "{% if enable_thinking is defined and enable_thinking %}T{% endif %}X",
        });
        let tmpl = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        assert_eq!(tmpl.render(&messages()).unwrap(), "X");
    }

    /// Comparing a missing field is false, as in jinja2. Step-5's template
    /// reads `message.name` on every non-first system message.
    #[test]
    fn comparing_a_missing_field_is_false() {
        let cfg = json!({
            "chat_template": "{% for m in messages %}{{ 'obs' if (m.role == 'system' \
                and m.name == 'observation') else m.role }};{% endfor %}",
        });
        let tmpl = ChatTemplate::from_tokenizer_config(&cfg).unwrap().unwrap();
        let msgs = json!([{"role": "user", "content": "a"}, {"role": "system", "content": "b"},
                          {"role": "system", "content": "c", "name": "observation"}]);
        assert_eq!(tmpl.render(&msgs).unwrap(), "user;system;obs;");
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
        assert_eq!(tmpl.render(&messages()).unwrap(), "X");
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
