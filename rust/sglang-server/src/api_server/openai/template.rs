//! Resolve chat-template names and files to chat prompt formatters.
//
//! Hugging Face tokenizer configs contain Jinja templates. SGLang also accepts
//! legacy conversation JSON files and the names in Python's template registry.
//! Legacy definitions are rendered by a native port of Python's
//! `Conversation.get_prompt()` so there is exactly one implementation of the
//! per-style formatting logic (no Jinja translation to drift).

use std::path::{Path, PathBuf};

use dynamo_protocols::types::{
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestAssistantMessageContentPart,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestSystemMessageContentPart, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, CreateChatCompletionRequest,
};
use dynamo_renderer::{ChatTemplate, ContextMixins, PromptContextMixin, PromptFormatter};
use serde_json::Value;
use thiserror::Error;

use crate::message::OneOrMany;

const SUPPORTED_STYLES: &[&str] = &[
    "ADD_COLON_SINGLE",
    "ADD_COLON_TWO",
    "ADD_COLON_SPACE_SINGLE",
    "NO_COLON_SINGLE",
    "NO_COLON_TWO",
    "ADD_NEW_LINE_SINGLE",
    "LLAMA2",
    "LLAMA3",
    "LLAMA4",
    "CHATGLM",
    "CHATML",
    "CHATINTERN",
    "DOLLY",
    "RWKV",
    "PHOENIX",
    "ROBIN",
    "FALCON_CHAT",
    "CHATGLM3",
    "DEEPSEEK_CHAT",
    "METAMATH",
    "DeepSeekVL2",
    "QWEN2_VL_EMBED",
    "QWEN2_AUDIO",
    "GEMMA3",
    "MPT",
    "PADDLE_OCR",
    "UNLIMITED_OCR",
];

/// A chat prompt formatter: either the model's HuggingFace Jinja template or a
/// legacy SGLang conversation template.
#[derive(Clone)]
pub enum ChatFormatter {
    HuggingFace(PromptFormatter),
    Legacy(Box<LegacyFormatter>),
}

impl ChatFormatter {
    /// Render the request's messages to a single prompt string.
    pub(super) fn render(
        &self,
        request: &CreateChatCompletionRequest,
    ) -> Result<String, TemplateError> {
        match self {
            ChatFormatter::HuggingFace(formatter) => {
                let PromptFormatter::OAI(formatter) = formatter;
                formatter
                    .render(request)
                    .map_err(|error| TemplateError::Renderer {
                        message: error.to_string(),
                    })
            }
            ChatFormatter::Legacy(formatter) => formatter.render(request),
        }
    }
}

/// A legacy conversation template, mirroring Python's `Conversation` fields.
#[derive(Debug, Clone)]
pub(super) struct LegacySpec {
    /// Python `Conversation.name` — drives the CHATGLM round-offset quirk.
    pub(super) name: String,
    pub(super) system_template: String,
    pub(super) system_message: String,
    /// `(user_role, assistant_role)` — Python `Conversation.roles`.
    pub(super) roles: (String, String),
    pub(super) style: String,
    pub(super) sep: String,
    /// `None` = Python's `Conversation.sep2` default. Styles that alternate
    /// seps (`seps[i % 2]`) need it set; Python crashes on `None` there and we
    /// error deliberately.
    pub(super) sep2: Option<String>,
    /// Python `Conversation.stop_str` (`str | list[str] | None`).
    pub(super) stop_str: Option<OneOrMany<String>>,
    pub(super) image_token: String,
    pub(super) audio_token: String,
}

impl Default for LegacySpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            system_template: String::new(),
            system_message: String::new(),
            roles: (String::new(), String::new()),
            style: String::new(),
            sep: String::new(),
            sep2: None,
            stop_str: None,
            image_token: "<image>".into(),
            audio_token: "<audio>".into(),
        }
    }
}

/// Native port of Python `generate_chat_conv` + `Conversation.get_prompt()`:
/// fold system messages into the system prompt, keep user/assistant messages in
/// order, always append the assistant opening, then render per `sep_style`.
#[derive(Clone)]
pub struct LegacyFormatter {
    spec: LegacySpec,
}

impl LegacyFormatter {
    pub(super) fn render(
        &self,
        request: &CreateChatCompletionRequest,
    ) -> Result<String, TemplateError> {
        let mut system_message = self.spec.system_message.clone();
        let mut messages: Vec<(String, String)> = Vec::new();
        for message in &request.messages {
            match message {
                ChatCompletionRequestMessage::System(message) => {
                    system_message = extract_system_text(&message.content)?;
                }
                ChatCompletionRequestMessage::User(message) => {
                    let content = match &message.content {
                        ChatCompletionRequestUserMessageContent::Text(text) => text.clone(),
                        ChatCompletionRequestUserMessageContent::Array(parts) => {
                            let mut text = String::new();
                            for part in parts {
                                match part {
                                    ChatCompletionRequestUserMessageContentPart::Text(part) => {
                                        text.push_str(&part.text);
                                    }
                                    // Python would splice media tokens in here;
                                    // the OpenAI adapter rejects media content
                                    // upstream, so this is unreachable — error
                                    // rather than silently drop.
                                    _ => {
                                        return Err(TemplateError::MediaContent { role: "user" });
                                    }
                                }
                            }
                            text
                        }
                    };
                    messages.push((self.spec.roles.0.clone(), content));
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    let content = message
                        .content
                        .as_ref()
                        .map(extract_assistant_text)
                        .transpose()?
                        .unwrap_or_default();
                    messages.push((self.spec.roles.1.clone(), content));
                }
                other => {
                    return Err(TemplateError::UnsupportedRole {
                        role: match other {
                            ChatCompletionRequestMessage::Developer(_) => "developer",
                            ChatCompletionRequestMessage::Tool(_) => "tool",
                            ChatCompletionRequestMessage::Function(_) => "function",
                            _ => unreachable!(),
                        },
                    });
                }
            }
        }
        // Python's `generate_chat_conv` appends the assistant opening.
        messages.push((self.spec.roles.1.clone(), String::new()));
        self.render_prompt(&system_message, &messages)
    }

    fn render_prompt(
        &self,
        system_message: &str,
        messages: &[(String, String)],
    ) -> Result<String, TemplateError> {
        let spec = &self.spec;
        // Python: `self.system_template.format(system_message=self.system_message)`.
        let system_prompt = spec
            .system_template
            .replace("{system_message}", system_message);
        let user_role = &spec.roles.0;
        let assistant_role = &spec.roles.1;
        let mut ret = String::new();

        match spec.style.as_str() {
            "ADD_COLON_SINGLE" => {
                ret.push_str(&system_prompt);
                ret.push_str(&spec.sep);
                for (role, content) in messages {
                    if content.is_empty() {
                        ret.push_str(&format!("{role}:"));
                    } else {
                        ret.push_str(&format!("{role}: {content}{}", spec.sep));
                    }
                }
            }
            "ADD_COLON_TWO" => {
                let sep2 = sep2(spec, "ADD_COLON_TWO")?;
                ret.push_str(&system_prompt);
                ret.push_str(&spec.sep);
                for (i, (role, content)) in messages.iter().enumerate() {
                    if content.is_empty() {
                        ret.push_str(&format!("{role}:"));
                    } else {
                        ret.push_str(&format!("{role}: {content}{}", sep_even_odd(spec, sep2, i)));
                    }
                }
            }
            "ADD_COLON_SPACE_SINGLE" => {
                ret.push_str(&system_prompt);
                ret.push_str(&spec.sep);
                for (role, content) in messages {
                    if content.is_empty() {
                        ret.push_str(&format!("{role}: "));
                    } else {
                        ret.push_str(&format!("{role}: {content}{}", spec.sep));
                    }
                }
            }
            "ADD_NEW_LINE_SINGLE" => {
                if !system_message.is_empty() && !system_prompt.is_empty() {
                    ret.push_str(&system_prompt);
                    ret.push_str(&spec.sep);
                }
                for (role, content) in messages {
                    if content.is_empty() {
                        ret.push_str(&format!("{role}\n"));
                    } else {
                        ret.push_str(&format!("{role}\n{content}{}", spec.sep));
                    }
                }
            }
            "QWEN2_VL_EMBED" => {
                if !system_prompt.is_empty() {
                    ret.push_str(&system_prompt);
                    ret.push_str(&spec.sep);
                }
                for (role, content) in messages {
                    if content.is_empty() {
                        ret.push_str(&format!("{role}\n"));
                    } else {
                        ret.push_str(&format!("{role}\n{content}{}", spec.sep));
                    }
                }
                match &spec.stop_str {
                    Some(OneOrMany::One(stop)) => ret.push_str(stop),
                    // Python `ret += self.stop_str` raises TypeError for
                    // `None` / list; error deliberately instead.
                    _ => {
                        return Err(TemplateError::InvalidStopString {
                            style: "QWEN2_VL_EMBED".into(),
                        });
                    }
                }
            }
            "NO_COLON_SINGLE" => {
                ret.push_str(&system_prompt);
                for (role, content) in messages {
                    if content.is_empty() {
                        ret.push_str(role);
                    } else {
                        ret.push_str(&format!("{role}{content}{}", spec.sep));
                    }
                }
            }
            "NO_COLON_TWO" => {
                let sep2 = sep2(spec, "NO_COLON_TWO")?;
                ret.push_str(&system_prompt);
                for (i, (role, content)) in messages.iter().enumerate() {
                    if content.is_empty() {
                        ret.push_str(role);
                    } else {
                        ret.push_str(&format!("{role}{content}{}", sep_even_odd(spec, sep2, i)));
                    }
                }
            }
            "RWKV" => {
                ret.push_str(&system_prompt);
                for (role, content) in messages {
                    if content.is_empty() {
                        ret.push_str(&format!("{role}:"));
                    } else {
                        ret.push_str(&format!(
                            "{role}: {}",
                            content.replace("\r\n", "\n").replace("\n\n", "\n")
                        ));
                        ret.push_str("\n\n");
                    }
                }
            }
            "LLAMA4" => {
                if !system_message.is_empty() {
                    ret.push_str(&system_prompt);
                }
                for (role, content) in messages {
                    if content.is_empty() {
                        ret.push_str(&format!("<|header_start|>{role}<|header_end|>\n\n"));
                    } else {
                        ret.push_str(&format!(
                            "<|header_start|>{role}<|header_end|>\n\n{}<|eot|>",
                            content.trim()
                        ));
                    }
                }
            }
            "LLAMA3" => {
                if !system_message.is_empty() {
                    ret.push_str(&system_prompt);
                }
                for (role, content) in messages {
                    if content.is_empty() {
                        ret.push_str(&format!("<|start_header_id|>{role}<|end_header_id|>\n\n"));
                    } else {
                        ret.push_str(&format!(
                            "<|start_header_id|>{role}<|end_header_id|>\n\n{}<|eot_id|>",
                            content.trim()
                        ));
                    }
                }
            }
            "LLAMA2" => {
                let sep2 = sep2(spec, "LLAMA2")?;
                if system_message.is_empty() {
                    ret.push_str("[INST] ");
                } else {
                    ret.push_str(&system_prompt);
                }
                for (i, (_, content)) in messages.iter().enumerate() {
                    // Python: `tag = self.roles[i % 2]` — parity, not the
                    // stored role, and the first message has no tag.
                    let tag = if i % 2 == 0 {
                        user_role
                    } else {
                        assistant_role
                    };
                    if content.is_empty() {
                        ret.push_str(tag);
                    } else if i == 0 {
                        ret.push_str(&format!("{content} "));
                    } else {
                        ret.push_str(&format!("{tag} {content}{}", sep_even_odd(spec, sep2, i)));
                    }
                }
            }
            "CHATGLM" => {
                // Python: `round_add_n = 1 if self.name == "chatglm2" else 0`.
                let round_add_n = if spec.name == "chatglm2" { 1 } else { 0 };
                if !system_prompt.is_empty() {
                    ret.push_str(&system_prompt);
                    ret.push_str(&spec.sep);
                }
                for (i, (role, content)) in messages.iter().enumerate() {
                    if i % 2 == 0 {
                        ret.push_str(&format!("[Round {}]{}", i / 2 + round_add_n, spec.sep));
                    }
                    if content.is_empty() {
                        ret.push_str(&format!("{role}："));
                    } else {
                        ret.push_str(&format!("{role}：{content}{}", spec.sep));
                    }
                }
            }
            "CHATML" => {
                if !system_prompt.is_empty() {
                    ret.push_str(&system_prompt);
                    ret.push_str(&spec.sep);
                    ret.push('\n');
                }
                for (role, content) in messages {
                    if content.is_empty() {
                        ret.push_str(&format!("{role}\n"));
                    } else {
                        ret.push_str(&format!("{role}\n{content}{}\n", spec.sep));
                    }
                }
            }
            "CHATGLM3" => {
                if !system_message.is_empty() {
                    ret.push_str(&system_prompt);
                }
                for (role, content) in messages {
                    if content.is_empty() {
                        ret.push_str(role);
                    } else {
                        ret.push_str(&format!("{role}\n{content}"));
                    }
                }
            }
            "CHATINTERN" => {
                let sep2 = sep2(spec, "CHATINTERN")?;
                ret.push_str(&system_prompt);
                for (i, (role, content)) in messages.iter().enumerate() {
                    if i % 2 == 0 {
                        ret.push_str("<s>");
                    }
                    if content.is_empty() {
                        ret.push_str(&format!("{role}:"));
                    } else {
                        ret.push_str(&format!(
                            "{role}:{content}{}\n",
                            sep_even_odd(spec, sep2, i)
                        ));
                    }
                }
            }
            "DOLLY" => {
                let sep2 = sep2(spec, "DOLLY")?;
                ret.push_str(&system_prompt);
                for (i, (role, content)) in messages.iter().enumerate() {
                    if content.is_empty() {
                        ret.push_str(&format!("{role}:\n"));
                    } else {
                        ret.push_str(&format!(
                            "{role}:\n{content}{}",
                            sep_even_odd(spec, sep2, i)
                        ));
                        if i % 2 == 1 {
                            ret.push_str("\n\n");
                        }
                    }
                }
            }
            "PHOENIX" => {
                ret.push_str(&system_prompt);
                for (role, content) in messages {
                    if content.is_empty() {
                        ret.push_str(&format!("{role}: <s>"));
                    } else {
                        ret.push_str(&format!("{role}: <s>{content}</s>"));
                    }
                }
            }
            "ROBIN" => {
                ret.push_str(&system_prompt);
                ret.push_str(&spec.sep);
                for (role, content) in messages {
                    if content.is_empty() {
                        ret.push_str(&format!("{role}:\n"));
                    } else {
                        ret.push_str(&format!("{role}:\n{content}{}", spec.sep));
                    }
                }
            }
            "FALCON_CHAT" => {
                if !system_message.is_empty() {
                    ret.push_str(&system_prompt);
                    ret.push_str(&spec.sep);
                }
                for (role, content) in messages {
                    if content.is_empty() {
                        ret.push_str(&format!("{role}:"));
                    } else {
                        ret.push_str(&format!("{role}: {content}{}", spec.sep));
                    }
                }
            }
            "METAMATH" => {
                let sep2 = sep2(spec, "METAMATH")?;
                if !system_prompt.is_empty() {
                    ret.push_str(&system_prompt);
                    ret.push_str(&spec.sep);
                }
                for (i, (role, content)) in messages.iter().enumerate() {
                    // Python: sep2 prefixes odd messages; sep ends even ones.
                    if content.is_empty() {
                        if i % 2 == 0 {
                            ret.push_str(&format!("{role}:\n"));
                        } else {
                            ret.push_str(&format!("{role}: {sep2}"));
                        }
                    } else if i % 2 == 0 {
                        ret.push_str(&format!("{role}:\n{content}{}", spec.sep));
                    } else {
                        ret.push_str(&format!("{role}: {sep2}{content}"));
                    }
                }
            }
            "DEEPSEEK_CHAT" => {
                let sep2 = sep2(spec, "DEEPSEEK_CHAT")?;
                ret.push_str(&system_prompt);
                for (i, (role, content)) in messages.iter().enumerate() {
                    if content.is_empty() {
                        ret.push_str(&format!("{role}:"));
                    } else {
                        ret.push_str(&format!("{role}: {content}{}", sep_even_odd(spec, sep2, i)));
                    }
                }
            }
            "DeepSeekVL2" => {
                let sep2 = sep2(spec, "DeepSeekVL2")?;
                if !system_prompt.is_empty() {
                    ret.push_str(&system_prompt);
                    ret.push_str(&spec.sep);
                }
                for (i, (role, content)) in messages.iter().enumerate() {
                    if content.is_empty() {
                        ret.push_str(&format!("{role}:"));
                    } else {
                        ret.push_str(&format!("{role}: {content}{}", sep_even_odd(spec, sep2, i)));
                    }
                }
            }
            "GEMMA3" => {
                ret.push_str(&system_prompt);
                for (i, (role, content)) in messages.iter().enumerate() {
                    if content.is_empty() {
                        ret.push_str(role);
                    } else if i == 0 {
                        ret.push_str(&format!("{content}{}", spec.sep));
                    } else {
                        ret.push_str(&format!("{role}{content}{}", spec.sep));
                    }
                }
            }
            "MPT" => {
                ret.push_str(&system_prompt);
                ret.push_str(&spec.sep);
                for (role, content) in messages {
                    if content.is_empty() {
                        ret.push_str(role);
                    } else {
                        ret.push_str(&format!("{role}{content}{}", spec.sep));
                    }
                }
            }
            "QWEN2_AUDIO" => {
                if !system_prompt.is_empty() {
                    ret.push_str(&system_prompt);
                    ret.push_str(&spec.sep);
                }
                let mut counter = 1usize;
                for (role, content) in messages {
                    if content.is_empty() {
                        ret.push_str(&format!("{role}\n"));
                    } else {
                        let mut message = content.clone();
                        while message.contains(&spec.audio_token) {
                            // Python: `audio_token.format(idx=counter)`. A
                            // token without `{idx}` makes the replace a no-op
                            // and Python's loop infinite; bail out instead of
                            // hanging the server.
                            let indexed = spec.audio_token.replace("{idx}", &counter.to_string());
                            if indexed == spec.audio_token {
                                break;
                            }
                            message = message.replacen(&spec.audio_token, &indexed, 1);
                            counter += 1;
                        }
                        ret.push_str(&format!("{role}\n{message}{}", spec.sep));
                    }
                }
            }
            "PADDLE_OCR" => {
                ret.push_str(&system_prompt);
                for (role, content) in messages {
                    if content.is_empty() {
                        ret.push_str(&format!("{role}: "));
                    } else if role == user_role {
                        ret.push_str(&format!("{role}: "));
                        if content.contains(&spec.image_token) {
                            ret.push_str(
                                &content
                                    .replace(&format!("{}\n", spec.image_token), &spec.image_token),
                            );
                        } else {
                            ret.push_str(content);
                        }
                        ret.push('\n');
                    } else {
                        ret.push_str(&format!("{role}: {content}{}", spec.sep));
                    }
                }
            }
            "UNLIMITED_OCR" => {
                let sep2 = sep2(spec, "UNLIMITED_OCR")?;
                if !system_prompt.is_empty() {
                    ret.push_str(&system_prompt);
                    ret.push_str(&spec.sep);
                }
                for (i, (role, content)) in messages.iter().enumerate() {
                    if content.is_empty() {
                        ret.push_str(role);
                    } else {
                        ret.push_str(&format!("{role}{content}{}", sep_even_odd(spec, sep2, i)));
                    }
                }
            }
            other => {
                return Err(TemplateError::InvalidStyle {
                    style: other.to_owned(),
                });
            }
        }
        Ok(ret)
    }
}

/// Python `seps = [self.sep, self.sep2]` indexed by message parity.
fn sep_even_odd<'a>(spec: &'a LegacySpec, sep2: &'a str, index: usize) -> &'a str {
    if index.is_multiple_of(2) {
        &spec.sep
    } else {
        sep2
    }
}

fn sep2<'a>(spec: &'a LegacySpec, style: &str) -> Result<&'a str, TemplateError> {
    spec.sep2
        .as_deref()
        .ok_or_else(|| TemplateError::MissingSep2 {
            style: style.to_owned(),
        })
}

/// Python `generate_chat_conv` system extraction: a plain string, or an array
/// with exactly one `text` part.
fn extract_system_text(
    content: &ChatCompletionRequestSystemMessageContent,
) -> Result<String, TemplateError> {
    match content {
        ChatCompletionRequestSystemMessageContent::Text(text) => Ok(text.clone()),
        ChatCompletionRequestSystemMessageContent::Array(parts) => {
            let mut texts = parts.iter().map(|part| match part {
                ChatCompletionRequestSystemMessageContentPart::Text(part) => part.text.as_str(),
            });
            match (texts.next(), texts.next()) {
                (Some(text), None) => Ok(text.to_owned()),
                _ => Err(TemplateError::NonTextContent { role: "system" }),
            }
        }
    }
}

/// Python `generate_chat_conv` assistant extraction: a plain string, or an
/// array with exactly one `text` part (`refusal` parts are rejected).
fn extract_assistant_text(
    content: &ChatCompletionRequestAssistantMessageContent,
) -> Result<String, TemplateError> {
    match content {
        ChatCompletionRequestAssistantMessageContent::Text(text) => Ok(text.clone()),
        ChatCompletionRequestAssistantMessageContent::Array(parts) => {
            let mut texts = parts.iter().filter_map(|part| match part {
                ChatCompletionRequestAssistantMessageContentPart::Text(part) => {
                    Some(part.text.as_str())
                }
                ChatCompletionRequestAssistantMessageContentPart::Refusal(_) => None,
            });
            match (texts.next(), texts.next()) {
                (Some(text), None) => Ok(text.to_owned()),
                _ => Err(TemplateError::NonTextContent { role: "assistant" }),
            }
        }
    }
}

#[derive(Debug, Error)]
pub(super) enum TemplateError {
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

pub(super) fn load_chat_formatter(
    config_file: &str,
    chat_template_arg: Option<&str>,
) -> Result<ChatFormatter, TemplateError> {
    // Python resolves registry names before looking at the filesystem — and
    // before touching the tokenizer config, so a built-in name works even when
    // `tokenizer_config.json` is absent.
    if let Some(argument) = chat_template_arg
        && let Some(spec) = builtin_template(argument)
    {
        return Ok(ChatFormatter::Legacy(Box::new(LegacyFormatter { spec })));
    }

    let config_path = Path::new(config_file);
    let config_text = read_to_string(config_path, "tokenizer config")?;
    let mut config = parse_json(&config_text, config_path, "tokenizer config")?;

    let Some(argument) = chat_template_arg else {
        return formatter_from_config(&config);
    };

    let path = Path::new(argument);
    if !path.exists() {
        return Err(TemplateError::NotFound {
            path: path.to_path_buf(),
        });
    }
    if !path.is_file() {
        return Err(TemplateError::NotFile {
            path: path.to_path_buf(),
        });
    }

    if path.extension().and_then(|extension| extension.to_str()) == Some("jinja") {
        let template = read_to_string(path, "chat template")?;
        set_chat_template(
            &mut config,
            Value::String(template.trim_matches('\n').replace("\\n", "\n")),
        )?;
        return formatter_from_config(&config);
    }

    let template_text = read_to_string(path, "chat template")?;
    let template = parse_json(&template_text, path, "chat template")?;

    // HF-style JSON files may carry chat_template directly. Legacy SGLang
    // files carry Conversation fields and are translated below.
    if let Some(chat_template) = template.get("chat_template") {
        set_chat_template(&mut config, chat_template.clone())?;
        formatter_from_config(&config)
    } else {
        Ok(ChatFormatter::Legacy(Box::new(LegacyFormatter {
            spec: parse_legacy_template(&template, path)?,
        })))
    }
}

fn read_to_string(path: &Path, kind: &'static str) -> Result<String, TemplateError> {
    std::fs::read_to_string(path).map_err(|source| TemplateError::Read {
        kind,
        path: path.to_path_buf(),
        source,
    })
}

fn parse_json(text: &str, path: &Path, kind: &'static str) -> Result<Value, TemplateError> {
    serde_json::from_str(text).map_err(|source| TemplateError::Parse {
        kind,
        path: path.to_path_buf(),
        source,
    })
}

fn set_chat_template(config: &mut Value, chat_template: Value) -> Result<(), TemplateError> {
    let Some(config) = config.as_object_mut() else {
        return Err(TemplateError::ConfigNotObject);
    };
    config.insert("chat_template".to_string(), chat_template);
    Ok(())
}

fn formatter_from_config(config: &Value) -> Result<ChatFormatter, TemplateError> {
    let template: ChatTemplate = serde_json::from_value(config.clone())
        .map_err(|source| TemplateError::Config { source })?;
    if template.chat_template.is_none() {
        return Err(TemplateError::Missing);
    }
    let formatter = PromptFormatter::from_parts(
        template,
        ContextMixins::new(&[PromptContextMixin::OaiChat]),
        true,
    )
    .map_err(|error| TemplateError::Renderer {
        message: error.to_string(),
    })?;
    Ok(ChatFormatter::HuggingFace(formatter))
}

/// Port of Python `_load_json_chat_template`: fields mirror `Conversation`
/// exactly (missing `sep2`/`image_token`/`audio_token` stay at Python defaults).
fn parse_legacy_template(value: &Value, path: &Path) -> Result<LegacySpec, TemplateError> {
    let object = value
        .as_object()
        .ok_or_else(|| TemplateError::LegacyNotObject {
            path: path.to_path_buf(),
        })?;

    let required_string = |name: &str| -> Result<String, TemplateError> {
        object
            .get(name)
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .ok_or_else(|| TemplateError::LegacyMissingField {
                path: path.to_path_buf(),
                field: name.to_string(),
            })
    };

    let style = required_string("sep_style")?;
    if !SUPPORTED_STYLES.contains(&style.as_str()) {
        return Err(TemplateError::UnknownStyle {
            path: path.to_path_buf(),
            style,
        });
    }

    // Python `Conversation.stop_str: str | list[str] | None` — the key is
    // required (`template["stop_str"]` raises KeyError when missing), but an
    // explicit `null` value means `None`.
    let stop_str = match object.get("stop_str") {
        Some(Value::String(value)) => Some(OneOrMany::One(value.clone())),
        Some(Value::Array(values)) => {
            let strings = values
                .iter()
                .map(Value::as_str)
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| TemplateError::LegacyMissingField {
                    path: path.to_path_buf(),
                    field: "stop_str".to_string(),
                })?;
            Some(OneOrMany::Many(
                strings.into_iter().map(str::to_owned).collect(),
            ))
        }
        Some(Value::Null) => None,
        Some(_) => {
            return Err(TemplateError::LegacyMissingField {
                path: path.to_path_buf(),
                field: "stop_str".to_string(),
            });
        }
        None => {
            return Err(TemplateError::LegacyMissingField {
                path: path.to_path_buf(),
                field: "stop_str".to_string(),
            });
        }
    };

    Ok(LegacySpec {
        name: required_string("name")?,
        // Python: `system_template=template["system"] + "\n{system_message}"`.
        system_template: format!("{}\n{{system_message}}", required_string("system")?),
        system_message: object
            .get("system_message")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        roles: (required_string("user")?, required_string("assistant")?),
        style,
        sep: object
            .get("sep")
            .and_then(Value::as_str)
            .unwrap_or("\n")
            .to_string(),
        sep2: None,
        stop_str,
        ..Default::default()
    })
}

fn builtin_template(name: &str) -> Option<LegacySpec> {
    let spec = match name {
        "llama-2" => LegacySpec {
            name: name.into(),
            system_template: "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n".into(),
            roles: ("[INST]".into(), "[/INST]".into()),
            style: "LLAMA2".into(),
            sep: " ".into(),
            sep2: Some(" </s><s>".into()),
            stop_str: Some(OneOrMany::Many(vec![
                "[INST]".into(),
                "[/INST]".into(),
                "<<SYS>>".into(),
                "<</SYS>>".into(),
            ])),
            ..Default::default()
        },
        "mistral" | "devstral" => LegacySpec {
            name: name.into(),
            system_template: "[SYSTEM_PROMPT]\n{system_message}\n[/SYSTEM_PROMPT]\n\n".into(),
            roles: ("[INST]".into(), "[/INST]".into()),
            style: "LLAMA2".into(),
            sep: " ".into(),
            sep2: Some(" </s><s>".into()),
            stop_str: Some(OneOrMany::Many(vec![
                "[INST]".into(),
                "[/INST]".into(),
                "[SYSTEM_PROMPT]".into(),
                "[/SYSTEM_PROMPT]".into(),
            ])),
            ..Default::default()
        },
        "llama-4" => LegacySpec {
            name: name.into(),
            system_template: "<|header_start|>system<|header_end|>\n\n{system_message}<|eot|>"
                .into(),
            roles: ("user".into(), "assistant".into()),
            style: "LLAMA4".into(),
            stop_str: Some(OneOrMany::Many(vec![
                "<|end_of_text|>".into(),
                "<|eot|>".into(),
                "<|eom|>".into(),
            ])),
            ..Default::default()
        },
        "phi-4-mm" => LegacySpec {
            name: name.into(),
            system_template: "{system_message}".into(),
            roles: ("<|user|>".into(), "<|assistant|>".into()),
            style: "NO_COLON_SINGLE".into(),
            sep: "<|end|>".into(),
            stop_str: Some(OneOrMany::One("<|end|>".into())),
            image_token: "<|endoftext10|>".into(),
            audio_token: "<|endoftext11|>".into(),
            ..Default::default()
        },
        "chatml" | "chatml-llava" => LegacySpec {
            name: name.into(),
            system_template: "<|im_start|>system\n{system_message}".into(),
            system_message: "You are a helpful assistant.".into(),
            roles: ("<|im_start|>user".into(), "<|im_start|>assistant".into()),
            style: "CHATML".into(),
            sep: "<|im_end|>".into(),
            stop_str: Some(OneOrMany::Many(vec![
                "<|endoftext|>".into(),
                "<|im_end|>".into(),
            ])),
            ..Default::default()
        },
        "vicuna_v1.1" => LegacySpec {
            name: name.into(),
            system_template: "{system_message}".into(),
            system_message: "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.".into(),
            roles: ("USER".into(), "ASSISTANT".into()),
            style: "ADD_COLON_TWO".into(),
            sep: " ".into(),
            sep2: Some("</s>".into()),
            ..Default::default()
        },
        "llama_3_vision" | "llava_llama_3" => LegacySpec {
            name: name.into(),
            system_template: "<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"
                .into(),
            system_message: "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.".into(),
            roles: ("user".into(), "assistant".into()),
            style: "LLAMA3".into(),
            stop_str: Some(OneOrMany::Many(vec![
                "<|end_of_text|>".into(),
                "<|eot_id|>".into(),
            ])),
            ..Default::default()
        },
        "internlm2-chat" => LegacySpec {
            name: name.into(),
            system_template: "<|im_start|>system\n{system_message}".into(),
            roles: ("<|im_start|>user".into(), "<|im_start|>assistant".into()),
            style: "ADD_COLON_SINGLE".into(),
            sep: "\n".into(),
            stop_str: Some(OneOrMany::Many(vec![
                "<|im_end|>".into(),
                "<|action_end|>".into(),
            ])),
            ..Default::default()
        },
        "internvl-2-5" => LegacySpec {
            name: name.into(),
            system_template: "<|im_start|>system\n{system_message}".into(),
            system_message: "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。".into(),
            roles: ("<|im_start|>user\n".into(), "<|im_start|>assistant\n".into()),
            style: "MPT".into(),
            sep: "<|im_end|>\n".into(),
            stop_str: Some(OneOrMany::Many(vec![
                "<|im_end|>".into(),
                "<|action_end|>".into(),
            ])),
            ..Default::default()
        },
        "qwen2-vl" => LegacySpec {
            name: name.into(),
            system_template: "<|im_start|>system\n{system_message}".into(),
            system_message: "You are a helpful assistant.".into(),
            roles: ("<|im_start|>user".into(), "<|im_start|>assistant".into()),
            style: "ADD_NEW_LINE_SINGLE".into(),
            sep: "<|im_end|>\n".into(),
            stop_str: Some(OneOrMany::Many(vec!["<|im_end|>".into()])),
            ..Default::default()
        },
        "deepseek-ocr" => LegacySpec {
            name: name.into(),
            style: "NO_COLON_SINGLE".into(),
            stop_str: Some(OneOrMany::Many(vec!["<｜end▁of▁sentence｜>".into()])),
            ..Default::default()
        },
        "unlimited-ocr" => LegacySpec {
            name: name.into(),
            system_template: "{system_message}".into(),
            style: "UNLIMITED_OCR".into(),
            sep2: Some(String::new()),
            ..Default::default()
        },
        "paddle-ocr" => LegacySpec {
            name: name.into(),
            system_template: "<|begin_of_sentence|>{system_message}".into(),
            roles: ("User".into(), "Assistant".into()),
            style: "PADDLE_OCR".into(),
            sep: "<|end_of_sentence|>".into(),
            stop_str: Some(OneOrMany::Many(vec!["<|end_of_sentence|>".into()])),
            image_token: "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>".into(),
            ..Default::default()
        },
        "deepseek-vl2" => LegacySpec {
            name: name.into(),
            system_template: "{system_message}".into(),
            roles: ("<|User|>".into(), "<|Assistant|>".into()),
            style: "DeepSeekVL2".into(),
            sep: "\n\n".into(),
            sep2: Some("<｜end▁of▁sentence｜>".into()),
            stop_str: Some(OneOrMany::Many(vec![
                "User:".into(),
                "<｜end▁of▁sentence｜>".into(),
            ])),
            ..Default::default()
        },
        "gemma-it" => LegacySpec {
            name: name.into(),
            system_template: "<start_of_turn>user\n{system_message}\n\n".into(),
            system_message: "You are a helpful assistant.".into(),
            roles: ("<start_of_turn>user\n".into(), "<start_of_turn>model\n".into()),
            style: "GEMMA3".into(),
            sep: "<end_of_turn>\n".into(),
            stop_str: Some(OneOrMany::Many(vec!["<end_of_turn>".into()])),
            image_token: "<start_of_image>".into(),
            audio_token: "<start_of_audio>".into(),
            ..Default::default()
        },
        "gme-qwen2-vl" => LegacySpec {
            name: name.into(),
            system_template: "<|im_start|>system\n{system_message}".into(),
            system_message: "You are a helpful assistant.".into(),
            roles: ("<|im_start|>user".into(), "<|im_start|>assistant".into()),
            style: "QWEN2_VL_EMBED".into(),
            sep: "<|im_end|>\n".into(),
            stop_str: Some(OneOrMany::One("<|endoftext|>".into())),
            ..Default::default()
        },
        "minicpmv" => LegacySpec {
            name: name.into(),
            system_template: "<|im_start|>system\n{system_message}.".into(),
            system_message: "You are a helpful assistant".into(),
            roles: ("<|im_start|>user".into(), "<|im_start|>assistant".into()),
            style: "ADD_NEW_LINE_SINGLE".into(),
            sep: "<|im_end|>\n".into(),
            stop_str: Some(OneOrMany::Many(vec![
                "<|im_end|>".into(),
                "<|endoftext|>".into(),
            ])),
            ..Default::default()
        },
        "janus-pro" => LegacySpec {
            name: name.into(),
            system_template: "{system_message}.".into(),
            system_message: "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language".into(),
            roles: ("User".into(), "Assistant".into()),
            style: "ADD_COLON_TWO".into(),
            sep: "\n\n".into(),
            sep2: Some("<｜end▁of▁sentence｜>".into()),
            stop_str: Some(OneOrMany::Many(vec![
                "<|User|>".into(),
                "<｜end▁of▁sentence｜>".into(),
            ])),
            ..Default::default()
        },
        "minicpmo" => LegacySpec {
            name: name.into(),
            system_template: "<|im_start|>system\n{system_message}".into(),
            system_message: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                .into(),
            roles: ("<|im_start|>user".into(), "<|im_start|>assistant".into()),
            style: "ADD_NEW_LINE_SINGLE".into(),
            sep: "<|im_end|>\n".into(),
            stop_str: Some(OneOrMany::Many(vec![
                "<|im_end|>".into(),
                "<|endoftext|>".into(),
            ])),
            ..Default::default()
        },
        "kimi-vl" => LegacySpec {
            name: name.into(),
            system_template: "<|im_system|>system<|im_middle|>{system_message}".into(),
            system_message: "You are a helpful assistant".into(),
            roles: (
                "<|im_user|>user<|im_middle|>".into(),
                "<|im_assistant|>assistant<|im_middle|>".into(),
            ),
            style: "NO_COLON_SINGLE".into(),
            sep: "<|im_end|>".into(),
            stop_str: Some(OneOrMany::One("<|im_end|>".into())),
            ..Default::default()
        },
        "qwen2-audio" => LegacySpec {
            name: name.into(),
            system_template: "<|im_start|>system\n{system_message}".into(),
            system_message: "You are a helpful assistant.".into(),
            roles: ("<|im_start|>user".into(), "<|im_start|>assistant".into()),
            style: "QWEN2_AUDIO".into(),
            sep: "<|im_end|>\n".into(),
            stop_str: Some(OneOrMany::Many(vec!["<|im_end|>".into()])),
            audio_token: "Audio {idx}: <|audio_bos|><|AUDIO|><|audio_eos|>\n".into(),
            ..Default::default()
        },
        "moss-vl" => LegacySpec {
            name: name.into(),
            system_template: "<|im_start|>system\n{system_message}".into(),
            roles: ("<|im_start|>user".into(), "<|im_start|>assistant".into()),
            style: "ADD_NEW_LINE_SINGLE".into(),
            sep: "<|im_end|>\n".into(),
            stop_str: Some(OneOrMany::Many(vec!["<|im_end|>".into()])),
            ..Default::default()
        },
        "points-v15-chat" => LegacySpec {
            name: name.into(),
            roles: ("<|im_start|>user".into(), "<|im_start|>assistant".into()),
            style: "ADD_NEW_LINE_SINGLE".into(),
            sep: "<|im_end|>\n".into(),
            stop_str: Some(OneOrMany::Many(vec!["<|im_end|>".into()])),
            ..Default::default()
        },
        "whisper" => LegacySpec {
            name: name.into(),
            style: "NO_COLON_SINGLE".into(),
            stop_str: Some(OneOrMany::Many(vec!["<|endoftext|>".into()])),
            audio_token: String::new(),
            ..Default::default()
        },
        _ => return None,
    };
    Some(spec)
}

#[cfg(test)]
mod tests {
    use dynamo_protocols::types::{
        ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartText,
        ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
        ChatCompletionRequestUserMessageContentPart, CreateChatCompletionRequest,
    };

    use super::{
        ChatFormatter, LegacyFormatter, LegacySpec, OneOrMany, builtin_template,
        load_chat_formatter,
    };

    fn request() -> CreateChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"}
            ]
        }))
        .unwrap()
    }

    fn spec(style: &str) -> LegacySpec {
        LegacySpec {
            name: "test".into(),
            system_template: "{system_message}".into(),
            system_message: "sys".into(),
            roles: ("USER".into(), "ASSISTANT".into()),
            style: style.into(),
            sep: "|sep|".into(),
            sep2: Some("|sep2|".into()),
            stop_str: Some(OneOrMany::One("<stop>".into())),
            ..Default::default()
        }
    }

    fn render(style: &str) -> String {
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "World"}
            ]
        }))
        .unwrap();
        LegacyFormatter { spec: spec(style) }
            .render(&request)
            .unwrap()
    }

    #[test]
    fn built_in_chatml_does_not_require_a_file() {
        let formatter = ChatFormatter::Legacy(Box::new(LegacyFormatter {
            spec: builtin_template("chatml").unwrap(),
        }));
        let rendered = formatter.render(&request()).unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// Every separator style renders exactly like Python's `Conversation.get_prompt()`.
    /// Messages: system "sys", user "Hello", assistant "World", then the
    /// unconditional assistant opening.
    #[test]
    fn all_separator_styles_render_like_python() {
        let cases = [
            (
                "ADD_COLON_SINGLE",
                "sys|sep|USER: Hello|sep|ASSISTANT: World|sep|ASSISTANT:",
            ),
            (
                "ADD_COLON_TWO",
                "sys|sep|USER: Hello|sep|ASSISTANT: World|sep2|ASSISTANT:",
            ),
            (
                "ADD_COLON_SPACE_SINGLE",
                "sys|sep|USER: Hello|sep|ASSISTANT: World|sep|ASSISTANT: ",
            ),
            (
                "ADD_NEW_LINE_SINGLE",
                "sys|sep|USER\nHello|sep|ASSISTANT\nWorld|sep|ASSISTANT\n",
            ),
            (
                "QWEN2_VL_EMBED",
                "sys|sep|USER\nHello|sep|ASSISTANT\nWorld|sep|ASSISTANT\n<stop>",
            ),
            (
                "NO_COLON_SINGLE",
                "sysUSERHello|sep|ASSISTANTWorld|sep|ASSISTANT",
            ),
            (
                "NO_COLON_TWO",
                "sysUSERHello|sep|ASSISTANTWorld|sep2|ASSISTANT",
            ),
            ("RWKV", "sysUSER: Hello\n\nASSISTANT: World\n\nASSISTANT:"),
            (
                "LLAMA4",
                "sys<|header_start|>USER<|header_end|>\n\nHello<|eot|><|header_start|>ASSISTANT<|header_end|>\n\nWorld<|eot|><|header_start|>ASSISTANT<|header_end|>\n\n",
            ),
            (
                "LLAMA3",
                "sys<|start_header_id|>USER<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>ASSISTANT<|end_header_id|>\n\nWorld<|eot_id|><|start_header_id|>ASSISTANT<|end_header_id|>\n\n",
            ),
            ("LLAMA2", "sysHello ASSISTANT World|sep2|USER"),
            (
                "CHATGLM",
                "sys|sep|[Round 0]|sep|USER：Hello|sep|ASSISTANT：World|sep|[Round 1]|sep|ASSISTANT：",
            ),
            (
                "CHATML",
                "sys|sep|\nUSER\nHello|sep|\nASSISTANT\nWorld|sep|\nASSISTANT\n",
            ),
            ("CHATGLM3", "sysUSER\nHelloASSISTANT\nWorldASSISTANT"),
            (
                "CHATINTERN",
                "sys<s>USER:Hello|sep|\nASSISTANT:World|sep2|\n<s>ASSISTANT:",
            ),
            (
                "DOLLY",
                "sysUSER:\nHello|sep|ASSISTANT:\nWorld|sep2|\n\nASSISTANT:\n",
            ),
            (
                "PHOENIX",
                "sysUSER: <s>Hello</s>ASSISTANT: <s>World</s>ASSISTANT: <s>",
            ),
            (
                "ROBIN",
                "sys|sep|USER:\nHello|sep|ASSISTANT:\nWorld|sep|ASSISTANT:\n",
            ),
            (
                "FALCON_CHAT",
                "sys|sep|USER: Hello|sep|ASSISTANT: World|sep|ASSISTANT:",
            ),
            (
                "METAMATH",
                "sys|sep|USER:\nHello|sep|ASSISTANT: |sep2|WorldASSISTANT:\n",
            ),
            (
                "DEEPSEEK_CHAT",
                "sysUSER: Hello|sep|ASSISTANT: World|sep2|ASSISTANT:",
            ),
            (
                "DeepSeekVL2",
                "sys|sep|USER: Hello|sep|ASSISTANT: World|sep2|ASSISTANT:",
            ),
            ("GEMMA3", "sysHello|sep|ASSISTANTWorld|sep|ASSISTANT"),
            ("MPT", "sys|sep|USERHello|sep|ASSISTANTWorld|sep|ASSISTANT"),
            (
                "QWEN2_AUDIO",
                "sys|sep|USER\nHello|sep|ASSISTANT\nWorld|sep|ASSISTANT\n",
            ),
            (
                "PADDLE_OCR",
                "sysUSER: Hello\nASSISTANT: World|sep|ASSISTANT: ",
            ),
            (
                "UNLIMITED_OCR",
                "sys|sep|USERHello|sep|ASSISTANTWorld|sep2|ASSISTANT",
            ),
        ];
        for (style, expected) in cases {
            assert_eq!(render(style), expected, "style {style}");
        }
    }

    /// LLAMA2's no-system path starts with `[INST] ` and tags messages by
    /// index parity — the opening at an even index takes the user tag.
    #[test]
    fn llama2_without_system_starts_with_inst() {
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "World"}
            ]
        }))
        .unwrap();
        let mut spec = spec("LLAMA2");
        spec.system_message = String::new();
        let rendered = LegacyFormatter { spec }.render(&request).unwrap();
        assert_eq!(rendered, "[INST] Hello ASSISTANT World|sep2|USER");
    }

    /// FALCON_CHAT without a system message starts with the first user message.
    #[test]
    fn falcon_chat_without_system_starts_with_user() {
        let mut spec = spec("FALCON_CHAT");
        spec.system_message = String::new();
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "World"}
            ]
        }))
        .unwrap();
        let rendered = LegacyFormatter { spec }.render(&request).unwrap();
        assert_eq!(rendered, "USER: Hello|sep|ASSISTANT: World|sep|ASSISTANT:");
    }

    /// QWEN2_AUDIO indexes each audio-token occurrence (Python
    /// `audio_token.format(idx=counter)`).
    #[test]
    fn qwen2_audio_indexes_audio_tokens() {
        let mut spec = spec("QWEN2_AUDIO");
        spec.audio_token = "Audio {idx}: <|audio_bos|><|AUDIO|><|audio_eos|>\n".into();
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{
                "role": "user",
                "content": "x Audio {idx}: <|audio_bos|><|AUDIO|><|audio_eos|>\n y Audio {idx}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
            }]
        }))
        .unwrap();
        let rendered = LegacyFormatter { spec }.render(&request).unwrap();
        assert_eq!(
            rendered,
            "sys|sep|USER\nx Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n y Audio 2: <|audio_bos|><|AUDIO|><|audio_eos|>\n|sep|ASSISTANT\n"
        );
    }

    /// PADDLE_OCR drops the newline after an image token in user messages.
    #[test]
    fn paddle_ocr_normalizes_image_token_newline() {
        let mut spec = spec("PADDLE_OCR");
        spec.image_token = "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>".into();
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{
                "role": "user",
                "content": "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>\nWhat is this?"
            }]
        }))
        .unwrap();
        let rendered = LegacyFormatter { spec }.render(&request).unwrap();
        assert_eq!(
            rendered,
            "sysUSER: <|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>What is this?\nASSISTANT: "
        );
    }

    /// stop_str preserves Python's `str | list[str] | None` typing.
    #[test]
    fn stop_str_keeps_python_type_semantics() {
        let request = serde_json::from_value::<CreateChatCompletionRequest>(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        // QWEN2_VL_EMBED requires a single string stop (Python raises TypeError
        // on a list / None) — error deliberately.
        let mut spec = spec("QWEN2_VL_EMBED");
        spec.stop_str = Some(OneOrMany::Many(vec!["a".into(), "b".into()]));
        let error = LegacyFormatter { spec }.render(&request).unwrap_err();
        assert!(error.to_string().contains("stop_str"));
        // The built-in gme-qwen2-vl registers a single-string stop.
        let spec = builtin_template("gme-qwen2-vl").unwrap();
        assert!(matches!(spec.stop_str, Some(OneOrMany::One(_))));
        // gemma-it registers a list.
        let spec = builtin_template("gemma-it").unwrap();
        assert!(matches!(spec.stop_str, Some(OneOrMany::Many(_))));
        // An explicit `"stop_str": null` in a legacy JSON file maps to `None`
        // (Python accepts a present null), while a missing key stays an error.
        let base = std::env::temp_dir().join(format!(
            "sglang-openai-template-stopnull-{}-test.json",
            std::process::id()
        ));
        std::fs::write(
            &base,
            r#"{
                "name": "test",
                "system": "System",
                "user": "USER",
                "assistant": "ASSISTANT",
                "sep_style": "ADD_COLON_SINGLE",
                "stop_str": null
            }"#,
        )
        .unwrap();
        let formatter =
            load_chat_formatter(base.to_str().unwrap(), Some(base.to_str().unwrap())).unwrap();
        let ChatFormatter::Legacy(formatter) = &formatter else {
            panic!("expected a legacy formatter");
        };
        assert!(formatter.spec.stop_str.is_none());
        let _ = std::fs::remove_file(base);
    }

    #[test]
    fn json_legacy_template_is_rendered_natively() {
        let base = std::env::temp_dir().join(format!(
            "sglang-openai-template-base-{}-test.json",
            std::process::id()
        ));
        let legacy = base.with_file_name("sglang-openai-template-legacy.json");
        std::fs::write(&base, r#"{"chat_template":"unused"}"#).unwrap();
        std::fs::write(
            &legacy,
            r#"{
                "name": "test-legacy",
                "system": "System",
                "system_message": "default",
                "user": "USER",
                "assistant": "ASSISTANT",
                "sep_style": "ADD_COLON_SINGLE",
                "sep": "\n",
                "stop_str": "<stop>"
            }"#,
        )
        .unwrap();

        let formatter =
            load_chat_formatter(base.to_str().unwrap(), Some(legacy.to_str().unwrap())).unwrap();
        let rendered = formatter.render(&request()).unwrap();
        assert_eq!(rendered, "System\nBe concise.\nUSER: Hello\nASSISTANT:");

        let _ = std::fs::remove_file(base);
        let _ = std::fs::remove_file(legacy);
    }

    /// Content extraction matches `generate_chat_conv`: system/assistant arrays
    /// must be a single text part; user arrays concatenate text parts; tool
    /// roles are rejected.
    #[test]
    fn content_extraction_matches_python() {
        let formatter = LegacyFormatter {
            spec: spec("CHATML"),
        };
        // User array content concatenates text parts.
        let request = serde_json::from_value::<CreateChatCompletionRequest>(serde_json::json!({
            "model": "test",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]
            }]
        }))
        .unwrap();
        let rendered = formatter.render(&request).unwrap();
        assert!(rendered.contains("USER\nHello world|sep|"));

        // System array with exactly one text part is fine.
        let request = serde_json::from_value::<CreateChatCompletionRequest>(serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
                {"role": "user", "content": "hi"}
            ]
        }))
        .unwrap();
        let rendered = formatter.render(&request).unwrap();
        assert!(!rendered.starts_with("sys|sep|")); // overridden by "Be brief."
        assert!(rendered.contains("Be brief."));

        // System array with two parts is rejected.
        let request = serde_json::from_value::<CreateChatCompletionRequest>(serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                {"role": "user", "content": "hi"}
            ]
        }))
        .unwrap();
        let error = formatter.render(&request).unwrap_err();
        assert!(error.to_string().contains("system message"));

        // Tool messages are rejected like Python's "Unknown role".
        let request = serde_json::from_value::<CreateChatCompletionRequest>(serde_json::json!({
            "model": "test",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "tool", "tool_call_id": "c1", "content": "42"}
            ]
        }))
        .unwrap();
        let error = formatter.render(&request).unwrap_err();
        assert!(error.to_string().contains("tool"));
    }

    /// The typed message structs are usable directly (no serde path needed).
    #[test]
    fn typed_messages_render_identically() {
        let request = CreateChatCompletionRequest {
            messages: vec![
                ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: ChatCompletionRequestSystemMessageContent::Text("sys".into()),
                    name: None,
                }),
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Array(vec![
                        ChatCompletionRequestUserMessageContentPart::Text(
                            ChatCompletionRequestMessageContentPartText {
                                text: "Hello".into(),
                            },
                        ),
                    ]),
                    name: None,
                }),
            ],
            ..Default::default()
        };
        let rendered = LegacyFormatter {
            spec: spec("ADD_COLON_SINGLE"),
        }
        .render(&request)
        .unwrap();
        assert_eq!(rendered, "sys|sep|USER: Hello|sep|ASSISTANT:");
    }
}
