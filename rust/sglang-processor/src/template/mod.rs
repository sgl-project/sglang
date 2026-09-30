//! Resolve chat-template names and files to chat prompt formatters.
//
//! Hugging Face tokenizer configs contain Jinja templates. SGLang also accepts
//! legacy conversation JSON files and the names in Python's template registry.
//! Legacy definitions are rendered by a native port of Python's
//! `Conversation.get_prompt()` so there is exactly one implementation of the
//! per-style formatting logic (no Jinja translation to drift).

use std::collections::HashMap;
use std::path::PathBuf;

use dynamo_protocols::types::ChatCompletionRequestMessage;
use dynamo_renderer::{OAIChatLikeRequest, PromptFormatter, RenderedPrompt};
use minijinja::machinery::{Token, tokenize};
use serde_json::Value;
use thiserror::Error;

/// Legacy stop strings: a single separator-style stop or a list of stops.
#[derive(Debug, Clone, PartialEq)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

pub use self::deepseek_v4::DeepSeekV4Profile;
use self::{
    deepseek_v4::dynamo_reasoning_effort,
    kimi_k25::{deep_sort, encode_tools_to_typescript},
};

mod builtins;
mod deepseek_v4;
mod kimi_k25;
mod legacy;
mod loader;
mod selection;

pub use self::selection::{ChatFormatterOptions, select_chat_formatter};

/// A chat prompt formatter: either the model's HuggingFace Jinja template or a
/// legacy SGLang conversation template.
#[derive(Clone)]
pub enum ChatFormatter {
    HuggingFace {
        formatter: PromptFormatter,
        thinking: ThinkingTemplates,
    },
    KimiK25 {
        formatter: PromptFormatter,
        thinking: ThinkingTemplates,
    },
    DeepSeekV4 {
        formatter: PromptFormatter,
        profile: DeepSeekV4Profile,
        environment_effort: Option<String>,
    },
    Legacy(Box<LegacyFormatter>),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum ThinkingPolicy {
    #[default]
    Unknown,
    Always,
    TemplateToggle {
        key: &'static str,
        default_enabled: bool,
    },
    NativeToggle {
        default_enabled: bool,
        named_tool_disables: bool,
    },
    ReasoningEffort,
}

impl ThinkingPolicy {
    fn apply(
        self,
        args: &mut Option<HashMap<String, Value>>,
        named_tool_choice: bool,
    ) -> Option<bool> {
        match self {
            Self::Unknown => None,
            Self::Always => Some(true),
            Self::TemplateToggle {
                key,
                default_enabled,
            } => {
                let enabled = args
                    .as_ref()
                    .and_then(|args| args.get(key))
                    .and_then(Value::as_bool)
                    .unwrap_or(default_enabled);
                args.get_or_insert_default()
                    .insert(key.to_owned(), Value::Bool(enabled));
                Some(enabled)
            }
            Self::NativeToggle {
                default_enabled,
                named_tool_disables,
            } => {
                let enabled = if named_tool_disables && named_tool_choice {
                    false
                } else {
                    dynamo_renderer::thinking_bool_from_args(args.as_ref())
                        .unwrap_or(default_enabled)
                };
                args.get_or_insert_default()
                    .insert("thinking".to_owned(), Value::Bool(enabled));
                Some(enabled)
            }
            Self::ReasoningEffort => Some(
                args.as_ref()
                    .and_then(|args| args.get("reasoning_effort"))
                    .is_some_and(|effort| effort.as_str() != Some("none")),
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ThinkingTemplates {
    default: ThinkingPolicy,
    tool_use: Option<ThinkingPolicy>,
}

impl ThinkingTemplates {
    pub(crate) fn native(default_enabled: bool, named_tool_disables: bool) -> Self {
        let policy = ThinkingPolicy::NativeToggle {
            default_enabled,
            named_tool_disables,
        };
        Self {
            default: policy,
            tool_use: Some(policy),
        }
    }

    pub(crate) fn always() -> Self {
        Self {
            default: ThinkingPolicy::Always,
            tool_use: Some(ThinkingPolicy::Always),
        }
    }

    fn for_request(self, tools_enabled: bool) -> ThinkingPolicy {
        if tools_enabled {
            self.tool_use.unwrap_or(self.default)
        } else {
            self.default
        }
    }

    fn from_config(config: &Value) -> Self {
        let Some(template) = config.get("chat_template") else {
            return Self::default();
        };
        if let Some(template) = template.as_str() {
            let policy = detect_thinking_policy(template);
            return Self {
                default: policy,
                tool_use: Some(policy),
            };
        }
        let mut policies = Self::default();
        for templates in template.as_array().into_iter().flatten() {
            let Some(templates) = templates.as_object() else {
                continue;
            };
            for (name, template) in templates {
                let Some(template) = template.as_str() else {
                    continue;
                };
                match name.as_str() {
                    "default" => policies.default = detect_thinking_policy(template),
                    "tool_use" => {
                        policies.tool_use = Some(detect_thinking_policy(template));
                    }
                    _ => {}
                }
            }
        }
        policies
    }
}

fn detect_thinking_policy(template: &str) -> ThinkingPolicy {
    if template.contains("<|channel|>")
        || ((!template.contains("enable_thinking") && !template.contains("thinking"))
            && (template.contains(r"<|im_start|>assistant\n<think>\n")
                || template.contains("<|im_start|>assistant\n<think>\n")))
    {
        return ThinkingPolicy::Always;
    }

    if template.contains("reasoning_effort") && template.contains("[THINK]") {
        return ThinkingPolicy::ReasoningEffort;
    }

    let Some(tokens) = jinja_code_tokens(template) else {
        return ThinkingPolicy::Unknown;
    };
    for key in ["enable_thinking", "thinking"] {
        if let Some(default_enabled) = detect_toggle_default(&tokens, key) {
            return ThinkingPolicy::TemplateToggle {
                key,
                default_enabled,
            };
        }
    }
    ThinkingPolicy::Unknown
}

fn jinja_code_tokens(template: &str) -> Option<Vec<String>> {
    let mut tokens = Vec::new();
    for token in tokenize(template, false, Default::default(), Default::default()) {
        let (token, _) = token.ok()?;
        let value = match token {
            Token::Ident(value) => value.to_owned(),
            Token::Pipe => "|".to_owned(),
            Token::Assign => "=".to_owned(),
            Token::Comma => ",".to_owned(),
            Token::ParenOpen => "(".to_owned(),
            Token::ParenClose => ")".to_owned(),
            Token::Dot => ".".to_owned(),
            Token::BlockStart | Token::VariableStart => ";".to_owned(),
            Token::BlockEnd | Token::VariableEnd => ";".to_owned(),
            Token::TemplateData(_) | Token::Str(_) | Token::String(_) => continue,
            _ => continue,
        };
        tokens.push(value);
    }
    Some(tokens)
}

fn detect_toggle_default(tokens: &[String], key: &str) -> Option<bool> {
    if has_default_filter(tokens, key, false) || has_guarded_default(tokens, key, false) {
        return Some(false);
    }
    if has_default_filter(tokens, key, true)
        || has_guarded_default(tokens, key, true)
        || contains_tokens(
            tokens,
            &[
                "set", key, "=", key, "if", key, "is", "defined", "else", "true",
            ],
        )
        || contains_tokens(tokens, &[key, "is", "defined", "and", key, "is", "false"])
        || contains_tokens(tokens, &[key, "is", "defined", "and", "not", key])
        || contains_tokens(tokens, &[key, "is", "not", "defined", "or", key])
        || contains_after(tokens, &["namespace", "("], &[key, "=", "true"], None)
    {
        return Some(true);
    }
    None
}

fn has_default_filter(tokens: &[String], key: &str, expected: bool) -> bool {
    for (index, token) in tokens.iter().enumerate() {
        if token != key || index.checked_sub(1).is_some_and(|i| tokens[i] == ".") {
            continue;
        }
        let Some(filter) = tokens.get(index + 1..index + 5) else {
            continue;
        };
        if filter[0] != "|" || !matches!(filter[1].as_str(), "default" | "d") || filter[2] != "(" {
            continue;
        }
        let Some(default_enabled) = jinja_bool(&filter[3]) else {
            continue;
        };
        let mut cursor = index + 5;
        let boolean_mode = match tokens.get(cursor).map(String::as_str) {
            Some(")") => false,
            Some(",") => {
                cursor += 1;
                match tokens.get(cursor).map(String::as_str) {
                    Some("true") => true,
                    Some("false") => false,
                    Some("boolean") if tokens.get(cursor + 1).is_some_and(|token| token == "=") => {
                        let Some(value) =
                            tokens.get(cursor + 2).and_then(|value| jinja_bool(value))
                        else {
                            continue;
                        };
                        value
                    }
                    _ => continue,
                }
            }
            _ => continue,
        };
        if default_enabled == expected && !(default_enabled && boolean_mode) {
            return true;
        }
    }
    false
}

fn has_guarded_default(tokens: &[String], key: &str, enabled: bool) -> bool {
    let value = if enabled { "true" } else { "false" };
    for guard in [
        ["if", "not", key, "is", "defined"],
        ["if", key, "is", "not", "defined"],
    ] {
        if contains_after(tokens, &guard, &["set", key, "=", value], Some("endif")) {
            return true;
        }
    }
    false
}

fn contains_tokens(tokens: &[String], expected: &[&str]) -> bool {
    tokens.windows(expected.len()).any(|window| {
        window
            .iter()
            .map(String::as_str)
            .eq(expected.iter().copied())
    })
}

fn contains_after(tokens: &[String], prefix: &[&str], suffix: &[&str], stop: Option<&str>) -> bool {
    for start in 0..tokens.len().saturating_sub(prefix.len()).saturating_add(1) {
        if !tokens[start..]
            .iter()
            .take(prefix.len())
            .map(String::as_str)
            .eq(prefix.iter().copied())
        {
            continue;
        }
        let remainder = &tokens[start + prefix.len()..];
        let end = stop
            .and_then(|stop| remainder.iter().position(|token| token == stop))
            .unwrap_or(remainder.len());
        if contains_tokens(&remainder[..end], suffix) {
            return true;
        }
    }
    false
}

fn jinja_bool(value: &str) -> Option<bool> {
    match value {
        "true" | "True" => Some(true),
        "false" | "False" => Some(false),
        _ => None,
    }
}

impl ChatFormatter {
    /// Render the request's messages to a single prompt string.
    #[cfg(test)]
    pub(super) fn render(&self, request: &dyn OAIChatLikeRequest) -> Result<String, TemplateError> {
        self.render_prompt(request).map(RenderedPrompt::into_text)
    }

    /// Render the request while preserving tokenizer trust boundaries required
    /// by native formatters such as Kimi K3.
    pub fn render_prompt(
        &self,
        request: &dyn OAIChatLikeRequest,
    ) -> Result<RenderedPrompt, TemplateError> {
        match self {
            ChatFormatter::HuggingFace { formatter, .. } => render_oai(formatter, request),
            ChatFormatter::KimiK25 { formatter, .. } => {
                let mut args = request.chat_template_args().cloned().unwrap_or_default();
                if let Some(tools) = request.tools() {
                    let mut tools =
                        serde_json::to_value(tools).map_err(|error| TemplateError::Renderer {
                            message: format!("failed to serialize Kimi K2.5 tools: {error}"),
                        })?;
                    deep_sort(&mut tools);
                    if let Some(tool_array) = tools.as_array()
                        && let Some(typescript) = encode_tools_to_typescript(tool_array)
                    {
                        args.insert("tools_ts_str".into(), Value::String(typescript));
                    }
                    args.insert("tools".into(), tools);
                }
                render_oai(formatter, &TemplateArgsRequest { request, args })
            }
            ChatFormatter::DeepSeekV4 {
                formatter,
                profile,
                environment_effort,
            } => {
                let mut args = request.chat_template_args().cloned().unwrap_or_default();
                let requested = args
                    .get("reasoning_effort")
                    .and_then(Value::as_str)
                    .or(environment_effort.as_deref());
                let mapped = dynamo_reasoning_effort(*profile, requested);
                let thinking =
                    dynamo_renderer::thinking_bool_from_args(Some(&args)).unwrap_or(false);
                args.insert("thinking".into(), Value::Bool(thinking));
                args.insert("reasoning_effort".into(), Value::String(mapped.into()));
                render_oai(formatter, &TemplateArgsRequest { request, args })
            }
            ChatFormatter::Legacy(formatter) => formatter.render(request).map(RenderedPrompt::text),
        }
    }

    /// The template's stop strings — Python `Conversation.stop_str`
    /// (`str | list[str] | None`). Legacy/builtin templates define them (e.g.
    /// chatml's `<|im_end|>`); the HuggingFace renderer carries none, matching
    /// Python's jinja path, which keeps only the request's own stops.
    pub fn stop_strs(&self) -> Option<OneOrMany<String>> {
        match self {
            ChatFormatter::HuggingFace { .. }
            | ChatFormatter::KimiK25 { .. }
            | ChatFormatter::DeepSeekV4 { .. } => None,
            ChatFormatter::Legacy(formatter) => formatter.spec.stop_str.clone(),
        }
    }

    /// Resolve the template's effective thinking mode and materialize its
    /// default under the exact kwarg the template consumes.
    pub fn resolve_thinking(
        &self,
        args: &mut Option<HashMap<String, Value>>,
        tools_enabled: bool,
        named_tool_choice: bool,
    ) -> Option<bool> {
        match self {
            ChatFormatter::HuggingFace { thinking, .. }
            | ChatFormatter::KimiK25 { thinking, .. } => thinking
                .for_request(tools_enabled)
                .apply(args, named_tool_choice),
            ChatFormatter::DeepSeekV4 { .. } => {
                let enabled =
                    dynamo_renderer::thinking_bool_from_args(args.as_ref()).unwrap_or(false);
                args.get_or_insert_default()
                    .insert("thinking".into(), Value::Bool(enabled));
                Some(enabled)
            }
            ChatFormatter::Legacy(_) => None,
        }
    }
}

fn render_oai(
    formatter: &PromptFormatter,
    request: &dyn OAIChatLikeRequest,
) -> Result<RenderedPrompt, TemplateError> {
    let PromptFormatter::OAI(formatter) = formatter;
    formatter
        .render_prompt(request)
        .map_err(|error| TemplateError::Renderer {
            message: error.to_string(),
        })
}

/// Formatter-facing request view over adapted template arguments.
/// Native effort access and template arguments share the adapted value without
/// changing the original request.
struct TemplateArgsRequest<'a> {
    request: &'a dyn OAIChatLikeRequest,
    args: HashMap<String, Value>,
}

impl OAIChatLikeRequest for TemplateArgsRequest<'_> {
    fn model(&self) -> String {
        self.request.model()
    }

    fn messages(&self) -> minijinja::Value {
        self.request.messages()
    }

    fn typed_messages(&self) -> Option<&[ChatCompletionRequestMessage]> {
        self.request.typed_messages()
    }

    fn tools(&self) -> Option<minijinja::Value> {
        self.request.tools()
    }

    fn tool_choice(&self) -> Option<minijinja::Value> {
        self.request.tool_choice()
    }

    fn response_format(&self) -> Option<minijinja::Value> {
        self.request.response_format()
    }

    fn reasoning_effort(&self) -> Option<minijinja::Value> {
        // Native formatters read this accessor before consulting template arguments.
        self.args
            .get("reasoning_effort")
            .map(minijinja::Value::from_serialize)
            .or_else(|| self.request.reasoning_effort())
    }

    fn should_add_generation_prompt(&self) -> bool {
        self.request.should_add_generation_prompt()
    }

    fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
        Some(&self.args)
    }
}

#[derive(Debug, Error)]
pub enum TemplateError {
    #[error("failed to read {kind} `{path}`: {source}")]
    Read {
        kind: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse {kind} `{path}`: {source}")]
    Parse {
        kind: &'static str,
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("chat template `{path}` is not a built-in name or a valid file path")]
    NotFound { path: PathBuf },

    #[error("chat template `{path}` is not a file")]
    NotFile { path: PathBuf },

    #[error("tokenizer config must be a JSON object")]
    ConfigNotObject,

    #[error("invalid chat template config: {source}")]
    Config {
        #[source]
        source: serde_json::Error,
    },

    #[error("tokenizer has no chat template")]
    Missing,

    #[error("tokenizer_config.json is required for this chat template source but was not found")]
    MissingConfig,

    #[error("invalid chat template: {message}")]
    Renderer { message: String },

    #[error("legacy chat template `{path}` must be a JSON object")]
    LegacyNotObject { path: PathBuf },

    #[error("legacy chat template `{path}` requires string field `{field}`")]
    LegacyMissingField { path: PathBuf, field: String },

    #[error("unknown separator style `{style}` in `{path}`")]
    UnknownStyle { path: PathBuf, style: String },

    #[error("unknown separator style `{style}`")]
    InvalidStyle { style: String },

    #[error("sep2 is required for separator style `{style}` but is not set")]
    MissingSep2 { style: String },

    #[error("stop_str must be a single string for separator style `{style}`")]
    InvalidStopString { style: String },

    #[error("the {role} message should be a single text")]
    NonTextContent { role: &'static str },

    #[error("multimodal {role} message content is not supported by legacy templates")]
    MediaContent { role: &'static str },

    #[error("unsupported message role `{role}` in legacy chat template")]
    UnsupportedRole { role: &'static str },
}

use self::legacy::LegacyFormatter;
pub use self::loader::load_chat_formatter;

#[cfg(test)]
use self::{
    builtins::builtin_template, legacy::LegacySpec, loader::infer_legacy_template_from_model_path,
};

#[cfg(test)]
mod tests;
