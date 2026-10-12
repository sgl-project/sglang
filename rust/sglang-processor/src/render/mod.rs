//! Chat prompt rendering over `dynamo-renderer`, plus the formatters SGLang adds.

use std::collections::HashMap;
use std::path::PathBuf;

use dynamo_protocols::types::ChatCompletionRequestMessage;
use dynamo_renderer::{OAIChatLikeRequest, PromptFormatter, RenderedPrompt};
use serde_json::Value;
use thiserror::Error;

mod legacy;
mod loader;
mod models;
mod selection;
mod thinking;

use self::legacy::LegacyFormatter;
pub use self::loader::load_chat_formatter;
pub use self::models::DeepSeekV4Profile;
use self::models::{deep_sort, dynamo_reasoning_effort, encode_tools_to_typescript};
pub use self::selection::{ChatFormatterOptions, select_chat_formatter};
pub use self::thinking::ThinkingTemplates;

/// Legacy stop strings: a single separator-style stop or a list of stops.
#[derive(Debug, Clone, PartialEq)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

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
