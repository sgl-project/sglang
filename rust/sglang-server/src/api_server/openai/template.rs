//! Resolve chat-template names and files to chat prompt formatters.
//
//! Hugging Face tokenizer configs contain Jinja templates. SGLang also accepts
//! legacy conversation JSON files and the names in Python's template registry.
//! Legacy definitions are rendered by a native port of Python's
//! `Conversation.get_prompt()` so there is exactly one implementation of the
//! per-style formatting logic (no Jinja translation to drift).

use std::path::Path;

use dynamo_protocols::types::{
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestAssistantMessageContentPart,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestSystemMessageContentPart, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, CreateChatCompletionRequest,
};
use dynamo_renderer::{ChatTemplate, ContextMixins, PromptContextMixin, PromptFormatter};
use serde_json::Value;

use crate::error::TemplateError;
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

    /// The template's stop strings — Python `Conversation.stop_str`
    /// (`str | list[str] | None`). Legacy/builtin templates define them (e.g.
    /// chatml's `<|im_end|>`); the HuggingFace renderer carries none, matching
    /// Python's jinja path, which keeps only the request's own stops.
    pub(super) fn stop_strs(&self) -> Option<OneOrMany<String>> {
        match self {
            ChatFormatter::HuggingFace(_) => None,
            ChatFormatter::Legacy(formatter) => formatter.spec.stop_str.clone(),
        }
    }
}

/// A legacy conversation template, mirroring Python's `Conversation` fields.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(default)]
struct LegacySpec {
    /// Python `Conversation.name` — drives the CHATGLM round-offset quirk.
    name: String,
    system_template: String,
    system_message: String,
    /// `(user_role, assistant_role)` — Python `Conversation.roles`.
    roles: (String, String),
    style: String,
    sep: String,
    /// `None` = Python's `Conversation.sep2` default. Styles that alternate
    /// seps (`seps[i % 2]`) need it set; Python crashes on `None` there and we
    /// error deliberately.
    sep2: Option<String>,
    /// Python `Conversation.stop_str` (`str | list[str] | None`).
    stop_str: Option<OneOrMany<String>>,
    image_token: String,
    audio_token: String,
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
    fn render(&self, request: &CreateChatCompletionRequest) -> Result<String, TemplateError> {
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
                                    // TODO: splice media tokens like Python
                                    // `generate_chat_conv` (image/audio/video +
                                    // prefix/newline/supplement quirks; spec needs
                                    // `video_token` + `image_token_at_prefix`).
                                    // Unreachable until the adapter's media 400 lifts.
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
        let mut ret = String::new();

        match spec.style.as_str() {
            "ADD_COLON_SINGLE" => style_add_colon_single(spec, &system_prompt, messages, &mut ret)?,
            "ADD_COLON_TWO" => style_add_colon_two(spec, &system_prompt, messages, &mut ret)?,
            "ADD_COLON_SPACE_SINGLE" => {
                style_add_colon_space_single(spec, &system_prompt, messages, &mut ret)?
            }
            "ADD_NEW_LINE_SINGLE" => {
                style_add_new_line_single(spec, system_message, &system_prompt, messages, &mut ret)?
            }
            "QWEN2_VL_EMBED" => style_qwen2_vl_embed(spec, &system_prompt, messages, &mut ret)?,
            "NO_COLON_SINGLE" => style_no_colon_single(spec, &system_prompt, messages, &mut ret)?,
            "NO_COLON_TWO" => style_no_colon_two(spec, &system_prompt, messages, &mut ret)?,
            "RWKV" => style_rwkv(spec, &system_prompt, messages, &mut ret)?,
            "LLAMA4" => style_llama4(spec, system_message, &system_prompt, messages, &mut ret)?,
            "LLAMA3" => style_llama3(spec, system_message, &system_prompt, messages, &mut ret)?,
            "LLAMA2" => style_llama2(spec, system_message, &system_prompt, messages, &mut ret)?,
            "CHATGLM" => style_chatglm(spec, &system_prompt, messages, &mut ret)?,
            "CHATML" => style_chatml(spec, &system_prompt, messages, &mut ret)?,
            "CHATGLM3" => style_chatglm3(spec, system_message, &system_prompt, messages, &mut ret)?,
            "CHATINTERN" => style_chatintern(spec, &system_prompt, messages, &mut ret)?,
            "DOLLY" => style_dolly(spec, &system_prompt, messages, &mut ret)?,
            "PHOENIX" => style_phoenix(spec, &system_prompt, messages, &mut ret)?,
            "ROBIN" => style_robin(spec, &system_prompt, messages, &mut ret)?,
            "FALCON_CHAT" => {
                style_falcon_chat(spec, system_message, &system_prompt, messages, &mut ret)?
            }
            "METAMATH" => style_metamath(spec, &system_prompt, messages, &mut ret)?,
            "DEEPSEEK_CHAT" => style_deepseek_chat(spec, &system_prompt, messages, &mut ret)?,
            "DeepSeekVL2" => style_deepseekvl2(spec, &system_prompt, messages, &mut ret)?,
            "GEMMA3" => style_gemma3(spec, &system_prompt, messages, &mut ret)?,
            "MPT" => style_mpt(spec, &system_prompt, messages, &mut ret)?,
            "QWEN2_AUDIO" => style_qwen2_audio(spec, &system_prompt, messages, &mut ret)?,
            "PADDLE_OCR" => style_paddle_ocr(spec, &system_prompt, messages, &mut ret)?,
            "UNLIMITED_OCR" => style_unlimited_ocr(spec, &system_prompt, messages, &mut ret)?,
            other => {
                return Err(TemplateError::InvalidStyle {
                    style: other.to_owned(),
                });
            }
        }
        Ok(ret)
    }
}

/// Python `SeparatorStyle.ADD_COLON_SINGLE` arm of `Conversation.get_prompt()`.
fn style_add_colon_single(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    ret.push_str(system_prompt);
    ret.push_str(&spec.sep);
    for (role, content) in messages {
        if content.is_empty() {
            ret.push_str(&format!("{role}:"));
        } else {
            ret.push_str(&format!("{role}: {content}{}", spec.sep));
        }
    }
    Ok(())
}

/// Python `SeparatorStyle.ADD_COLON_TWO` arm of `Conversation.get_prompt()`.
fn style_add_colon_two(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    let sep2 = sep2(spec, "ADD_COLON_TWO")?;
    ret.push_str(system_prompt);
    ret.push_str(&spec.sep);
    for (i, (role, content)) in messages.iter().enumerate() {
        if content.is_empty() {
            ret.push_str(&format!("{role}:"));
        } else {
            ret.push_str(&format!("{role}: {content}{}", sep_even_odd(spec, sep2, i)));
        }
    }
    Ok(())
}

/// Python `SeparatorStyle.ADD_COLON_SPACE_SINGLE` arm of `Conversation.get_prompt()`.
fn style_add_colon_space_single(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    ret.push_str(system_prompt);
    ret.push_str(&spec.sep);
    for (role, content) in messages {
        if content.is_empty() {
            ret.push_str(&format!("{role}: "));
        } else {
            ret.push_str(&format!("{role}: {content}{}", spec.sep));
        }
    }
    Ok(())
}

/// Python `SeparatorStyle.ADD_NEW_LINE_SINGLE` arm of `Conversation.get_prompt()`.
fn style_add_new_line_single(
    spec: &LegacySpec,
    system_message: &str,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    if !system_message.is_empty() && !system_prompt.is_empty() {
        ret.push_str(system_prompt);
        ret.push_str(&spec.sep);
    }
    for (role, content) in messages {
        if content.is_empty() {
            ret.push_str(&format!("{role}\n"));
        } else {
            ret.push_str(&format!("{role}\n{content}{}", spec.sep));
        }
    }
    Ok(())
}

/// Python `SeparatorStyle.QWEN2_VL_EMBED` arm of `Conversation.get_prompt()`.
fn style_qwen2_vl_embed(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    if !system_prompt.is_empty() {
        ret.push_str(system_prompt);
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
    Ok(())
}

/// Python `SeparatorStyle.NO_COLON_SINGLE` arm of `Conversation.get_prompt()`.
fn style_no_colon_single(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    ret.push_str(system_prompt);
    for (role, content) in messages {
        if content.is_empty() {
            ret.push_str(role);
        } else {
            ret.push_str(&format!("{role}{content}{}", spec.sep));
        }
    }
    Ok(())
}

/// Python `SeparatorStyle.NO_COLON_TWO` arm of `Conversation.get_prompt()`.
fn style_no_colon_two(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    let sep2 = sep2(spec, "NO_COLON_TWO")?;
    ret.push_str(system_prompt);
    for (i, (role, content)) in messages.iter().enumerate() {
        if content.is_empty() {
            ret.push_str(role);
        } else {
            ret.push_str(&format!("{role}{content}{}", sep_even_odd(spec, sep2, i)));
        }
    }
    Ok(())
}

/// Python `SeparatorStyle.RWKV` arm of `Conversation.get_prompt()`.
fn style_rwkv(
    _spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    ret.push_str(system_prompt);
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
    Ok(())
}

/// Python `SeparatorStyle.LLAMA4` arm of `Conversation.get_prompt()`.
fn style_llama4(
    _spec: &LegacySpec,
    system_message: &str,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    if !system_message.is_empty() {
        ret.push_str(system_prompt);
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
    Ok(())
}

/// Python `SeparatorStyle.LLAMA3` arm of `Conversation.get_prompt()`.
fn style_llama3(
    _spec: &LegacySpec,
    system_message: &str,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    if !system_message.is_empty() {
        ret.push_str(system_prompt);
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
    Ok(())
}

/// Python `SeparatorStyle.LLAMA2` arm of `Conversation.get_prompt()`.
fn style_llama2(
    spec: &LegacySpec,
    system_message: &str,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    let user_role = &spec.roles.0;
    let assistant_role = &spec.roles.1;
    let sep2 = sep2(spec, "LLAMA2")?;
    if system_message.is_empty() {
        ret.push_str("[INST] ");
    } else {
        ret.push_str(system_prompt);
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
    Ok(())
}

/// Python `SeparatorStyle.CHATGLM` arm of `Conversation.get_prompt()`.
fn style_chatglm(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    // Python: `round_add_n = 1 if self.name == "chatglm2" else 0`.
    let round_add_n = if spec.name == "chatglm2" { 1 } else { 0 };
    if !system_prompt.is_empty() {
        ret.push_str(system_prompt);
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
    Ok(())
}

/// Python `SeparatorStyle.CHATML` arm of `Conversation.get_prompt()`.
fn style_chatml(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    if !system_prompt.is_empty() {
        ret.push_str(system_prompt);
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
    Ok(())
}

/// Python `SeparatorStyle.CHATGLM3` arm of `Conversation.get_prompt()`.
fn style_chatglm3(
    _spec: &LegacySpec,
    system_message: &str,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    if !system_message.is_empty() {
        ret.push_str(system_prompt);
    }
    for (role, content) in messages {
        if content.is_empty() {
            ret.push_str(role);
        } else {
            ret.push_str(&format!("{role}\n{content}"));
        }
    }
    Ok(())
}

/// Python `SeparatorStyle.CHATINTERN` arm of `Conversation.get_prompt()`.
fn style_chatintern(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    let sep2 = sep2(spec, "CHATINTERN")?;
    ret.push_str(system_prompt);
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
    Ok(())
}

/// Python `SeparatorStyle.DOLLY` arm of `Conversation.get_prompt()`.
fn style_dolly(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    let sep2 = sep2(spec, "DOLLY")?;
    ret.push_str(system_prompt);
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
    Ok(())
}

/// Python `SeparatorStyle.PHOENIX` arm of `Conversation.get_prompt()`.
fn style_phoenix(
    _spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    ret.push_str(system_prompt);
    for (role, content) in messages {
        if content.is_empty() {
            ret.push_str(&format!("{role}: <s>"));
        } else {
            ret.push_str(&format!("{role}: <s>{content}</s>"));
        }
    }
    Ok(())
}

/// Python `SeparatorStyle.ROBIN` arm of `Conversation.get_prompt()`.
fn style_robin(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    ret.push_str(system_prompt);
    ret.push_str(&spec.sep);
    for (role, content) in messages {
        if content.is_empty() {
            ret.push_str(&format!("{role}:\n"));
        } else {
            ret.push_str(&format!("{role}:\n{content}{}", spec.sep));
        }
    }
    Ok(())
}

/// Python `SeparatorStyle.FALCON_CHAT` arm of `Conversation.get_prompt()`.
fn style_falcon_chat(
    spec: &LegacySpec,
    system_message: &str,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    if !system_message.is_empty() {
        ret.push_str(system_prompt);
        ret.push_str(&spec.sep);
    }
    for (role, content) in messages {
        if content.is_empty() {
            ret.push_str(&format!("{role}:"));
        } else {
            ret.push_str(&format!("{role}: {content}{}", spec.sep));
        }
    }
    Ok(())
}

/// Python `SeparatorStyle.METAMATH` arm of `Conversation.get_prompt()`.
fn style_metamath(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    let sep2 = sep2(spec, "METAMATH")?;
    if !system_prompt.is_empty() {
        ret.push_str(system_prompt);
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
    Ok(())
}

/// Python `SeparatorStyle.DEEPSEEK_CHAT` arm of `Conversation.get_prompt()`.
fn style_deepseek_chat(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    let sep2 = sep2(spec, "DEEPSEEK_CHAT")?;
    ret.push_str(system_prompt);
    for (i, (role, content)) in messages.iter().enumerate() {
        if content.is_empty() {
            ret.push_str(&format!("{role}:"));
        } else {
            ret.push_str(&format!("{role}: {content}{}", sep_even_odd(spec, sep2, i)));
        }
    }
    Ok(())
}

/// Python `SeparatorStyle.DeepSeekVL2` arm of `Conversation.get_prompt()`.
fn style_deepseekvl2(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    let sep2 = sep2(spec, "DeepSeekVL2")?;
    if !system_prompt.is_empty() {
        ret.push_str(system_prompt);
        ret.push_str(&spec.sep);
    }
    for (i, (role, content)) in messages.iter().enumerate() {
        if content.is_empty() {
            ret.push_str(&format!("{role}:"));
        } else {
            ret.push_str(&format!("{role}: {content}{}", sep_even_odd(spec, sep2, i)));
        }
    }
    Ok(())
}

/// Python `SeparatorStyle.GEMMA3` arm of `Conversation.get_prompt()`.
fn style_gemma3(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    ret.push_str(system_prompt);
    for (i, (role, content)) in messages.iter().enumerate() {
        if content.is_empty() {
            ret.push_str(role);
        } else if i == 0 {
            ret.push_str(&format!("{content}{}", spec.sep));
        } else {
            ret.push_str(&format!("{role}{content}{}", spec.sep));
        }
    }
    Ok(())
}

/// Python `SeparatorStyle.MPT` arm of `Conversation.get_prompt()`.
fn style_mpt(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    ret.push_str(system_prompt);
    ret.push_str(&spec.sep);
    for (role, content) in messages {
        if content.is_empty() {
            ret.push_str(role);
        } else {
            ret.push_str(&format!("{role}{content}{}", spec.sep));
        }
    }
    Ok(())
}

/// Python `SeparatorStyle.QWEN2_AUDIO` arm of `Conversation.get_prompt()`.
fn style_qwen2_audio(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    if !system_prompt.is_empty() {
        ret.push_str(system_prompt);
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
    Ok(())
}

/// Python `SeparatorStyle.PADDLE_OCR` arm of `Conversation.get_prompt()`.
fn style_paddle_ocr(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    let user_role = &spec.roles.0;
    let _assistant_role = &spec.roles.1;
    ret.push_str(system_prompt);
    for (role, content) in messages {
        if content.is_empty() {
            ret.push_str(&format!("{role}: "));
        } else if role == user_role {
            ret.push_str(&format!("{role}: "));
            if content.contains(&spec.image_token) {
                ret.push_str(
                    &content.replace(&format!("{}\n", spec.image_token), &spec.image_token),
                );
            } else {
                ret.push_str(content);
            }
            ret.push('\n');
        } else {
            ret.push_str(&format!("{role}: {content}{}", spec.sep));
        }
    }
    Ok(())
}

/// Python `SeparatorStyle.UNLIMITED_OCR` arm of `Conversation.get_prompt()`.
fn style_unlimited_ocr(
    spec: &LegacySpec,
    system_prompt: &str,
    messages: &[(String, String)],
    ret: &mut String,
) -> Result<(), TemplateError> {
    let sep2 = sep2(spec, "UNLIMITED_OCR")?;
    if !system_prompt.is_empty() {
        ret.push_str(system_prompt);
        ret.push_str(&spec.sep);
    }
    for (i, (role, content)) in messages.iter().enumerate() {
        if content.is_empty() {
            ret.push_str(role);
        } else {
            ret.push_str(&format!("{role}{content}{}", sep_even_odd(spec, sep2, i)));
        }
    }
    Ok(())
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

pub(super) fn load_chat_formatter(
    config_file: Option<&str>,
    model_path: Option<&str>,
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

    // Python `load_chat_template` (no `--chat-template`): infer a legacy
    // template from the model path before falling back to the HF template, so
    // a legacy model whose config has no `chat_template` still gets one.
    if chat_template_arg.is_none()
        && let Some(model_path) = model_path
        && let Some(spec) = infer_legacy_template_from_model_path(model_path)
    {
        tracing::info!(%model_path, "inferred legacy chat template from model path");
        return Ok(ChatFormatter::Legacy(Box::new(LegacyFormatter { spec })));
    }

    // Every remaining source builds the HF renderer around the tokenizer
    // config (the template itself, or the argument injected into it).
    let Some(config_file) = config_file else {
        return Err(TemplateError::MissingConfig);
    };

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

/// Port of Python `get_conv_template_by_model_path` (conversation.py
/// `matching_function_registry`, run in registration order): infer a legacy
/// built-in template from the model path, optionally consulting the model's
/// `config.json` `model_type`. `None` when nothing matches — the HF template
/// is the fallback then, as in Python.
fn infer_legacy_template_from_model_path(model_path: &str) -> Option<LegacySpec> {
    let lower = model_path.to_lowercase();
    // Regexes without regex: every Python pattern here is a plain substring or
    // a `prefix.*suffix` pair, both on a lowercased path.
    let contains = |needle: &str| lower.contains(needle);
    let precedes = |prefix: &str, suffix: &str| {
        lower
            .find(prefix)
            .is_some_and(|start| lower[start + prefix.len()..].contains(suffix))
    };

    if lower
        .split(|c: char| !c.is_alphanumeric())
        .any(|word| word == "points")
    {
        return builtin_template("points-v15-chat");
    }
    if precedes("moss", "vl") {
        return builtin_template("moss-vl");
    }
    if contains("internvl") {
        return builtin_template("internvl-2-5");
    }
    if contains("janus") {
        return builtin_template("janus-pro");
    }
    if contains("vicuna") || contains("llava-v1.5") || contains("llava-next-video-7b") {
        return builtin_template("vicuna_v1.1");
    }
    if precedes("deepseek", "vl2") {
        return builtin_template("deepseek-vl2");
    }
    if contains("llava-v1.6-34b")
        || contains("llava-v1.6-yi-34b")
        || contains("llava-next-video-34b")
        || contains("llava-onevision-qwen2")
    {
        return builtin_template("chatml-llava");
    }
    // MiniCPM: 4.6+ uses its own template and must not fall back to the
    // legacy conv template.
    if contains("minicpm-v-4.6")
        || contains("minicpm-v-4_6")
        || contains("minicpm-o-4.6")
        || contains("minicpm-o-4_6")
    {
        return None;
    }
    if contains("minicpm-v") {
        return builtin_template("minicpmv");
    }
    if contains("minicpm-o") {
        return builtin_template("minicpmo");
    }
    if contains("phi-4-multimodal") {
        return builtin_template("phi-4-mm");
    }
    if contains("deepseek-ocr") {
        return builtin_template("deepseek-ocr");
    }
    if contains("unlimited") {
        return builtin_template("unlimited-ocr");
    }
    if contains("paddleocr") {
        return builtin_template("paddle-ocr");
    }
    if contains("whisper") {
        return builtin_template("whisper");
    }

    // Model-type matchers read `<model_path>/config.json` (local dirs only —
    // Python's `get_model_type` cannot resolve HF repo ids either).
    let model_type = read_model_type(model_path)?;
    // Python `MODEL_TYPE_TO_TEMPLATE`; minicpmv4_6 is deliberately absent.
    let name = match model_type.as_str() {
        "moss_vl" => "moss-vl",
        "internvl_chat" => "internvl-2-5",
        "multi_modality" => "janus-pro",
        "deepseek_vl_v2" => "deepseek-vl2",
        "minicpmv" => "minicpmv",
        "minicpmo" => "minicpmo",
        "phi4mm" => "phi-4-mm",
        "deepseek-ocr" => "deepseek-ocr",
        "unlimited-ocr" => "unlimited-ocr",
        "paddleocr_vl" => "paddle-ocr",
        _ => return None,
    };
    builtin_template(name)
}

/// Python `get_model_type`: the `model_type` field of the model's `config.json`.
fn read_model_type(model_path: &str) -> Option<String> {
    let config_path = Path::new(model_path).join("config.json");
    if !config_path.is_file() {
        return None;
    }
    let config: Value = parse_json(
        &read_to_string(&config_path, "model config").ok()?,
        &config_path,
        "model config",
    )
    .ok()?;
    config.get("model_type")?.as_str().map(str::to_owned)
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

/// One embedded-registry entry: the alias list and the (name-less) spec.
#[derive(serde::Deserialize)]
struct BuiltinEntry {
    names: Vec<String>,
    spec: LegacySpec,
}

/// The builtin legacy templates, ported from Python `conversation.py`'s
/// registry.
fn builtin_template(name: &str) -> Option<LegacySpec> {
    static REGISTRY: std::sync::OnceLock<Vec<BuiltinEntry>> = std::sync::OnceLock::new();
    let registry = REGISTRY.get_or_init(|| {
        serde_json::from_str(include_str!("builtin_templates.json"))
            .expect("embedded builtin template registry parses")
    });
    registry
        .iter()
        .find(|entry| entry.names.iter().any(|alias| alias == name))
        .map(|entry| LegacySpec {
            name: name.to_string(),
            ..entry.spec.clone()
        })
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
        ChatFormatter, LegacyFormatter, LegacySpec, OneOrMany, SUPPORTED_STYLES, TemplateError,
        builtin_template, infer_legacy_template_from_model_path, load_chat_formatter,
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
        let formatter = load_chat_formatter(
            Some(base.to_str().unwrap()),
            None,
            Some(base.to_str().unwrap()),
        )
        .unwrap();
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

        let formatter = load_chat_formatter(
            Some(base.to_str().unwrap()),
            None,
            Some(legacy.to_str().unwrap()),
        )
        .unwrap();
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

    /// A built-in `--chat-template` name resolves without any tokenizer config.
    #[test]
    fn builtin_argument_works_without_tokenizer_config() {
        let formatter = load_chat_formatter(None, None, Some("chatml")).unwrap();
        let ChatFormatter::Legacy(formatter) = &formatter else {
            panic!("expected a legacy formatter");
        };
        assert_eq!(formatter.spec.name, "chatml");
    }

    /// Python `load_chat_template`: without `--chat-template`, the model path
    /// infers a legacy template before the HF fallback — so a legacy model
    /// with no `chat_template` in its config still gets one, and even a config
    /// that HAS one loses to the inference.
    #[test]
    fn model_path_inference_precedes_tokenizer_config() {
        let base = std::env::temp_dir().join(format!(
            "sglang-openai-template-infer-{}-test.json",
            std::process::id()
        ));
        std::fs::write(
            &base,
            r#"{"tokenizer_class":"LlamaTokenizer","chat_template":"{{messages}}"}"#,
        )
        .unwrap();

        // Path matcher: vicuna/llava-v1.5-style paths.
        let formatter = load_chat_formatter(
            Some(base.to_str().unwrap()),
            Some("models/vicuna-7b-v1.5"),
            None,
        )
        .unwrap();
        let ChatFormatter::Legacy(formatter) = &formatter else {
            panic!("expected a legacy formatter");
        };
        assert_eq!(formatter.spec.name, "vicuna_v1.1");
        // No config at all + path matcher.
        let formatter = load_chat_formatter(None, Some("deepseek-vl2-7b"), None).unwrap();
        let ChatFormatter::Legacy(formatter) = &formatter else {
            panic!("expected a legacy formatter");
        };
        assert_eq!(formatter.spec.name, "deepseek-vl2");

        // Model-type matcher: reads `<model_path>/config.json`.
        let model_dir = std::env::temp_dir().join(format!(
            "sglang-openai-template-infer-model-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            r#"{"model_type":"phi4mm","architectures":["Phi4MMForCausalLM"]}"#,
        )
        .unwrap();
        let formatter = load_chat_formatter(None, Some(model_dir.to_str().unwrap()), None).unwrap();
        let ChatFormatter::Legacy(formatter) = &formatter else {
            panic!("expected a legacy formatter");
        };
        assert_eq!(formatter.spec.name, "phi-4-mm");

        let _ = std::fs::remove_file(&base);
        let _ = std::fs::remove_dir_all(model_dir);
    }

    /// Every name the model-path matchers can produce must exist in the
    /// built-in table (parity guard for `MODEL_TYPE_TO_TEMPLATE`).
    #[test]
    fn inferred_template_names_resolve_to_builtins() {
        for model_path in [
            "points-7b-chat",
            "moss-vl",
            "moss2-vl",
            "internvl-2.5",
            "janus-pro",
            "vicuna-7b",
            "llava-v1.5-7b",
            "deepseek-vl2-small",
            "llava-v1.6-34b",
            "minicpm-v-2.6",
            "minicpm-o-4.5",
            "phi-4-multimodal",
            "deepseek-ocr",
            "unlimited-ocr",
            "paddleocr-vl",
            "whisper",
        ] {
            let spec = infer_legacy_template_from_model_path(model_path)
                .unwrap_or_else(|| panic!("no inference for {model_path}"));
            let _ = spec;
        }
    }

    /// MiniCPM 4.6+ must NOT fall back to the legacy template; with no config
    /// and nothing else to try, that surfaces as the missing-config error.
    #[test]
    fn minicpm_4_6_skips_legacy_inference() {
        assert!(infer_legacy_template_from_model_path("minicpm-v-4.6").is_none());
        assert!(matches!(
            load_chat_formatter(None, Some("minicpm-v-4.6"), None),
            Err(TemplateError::MissingConfig)
        ));
    }

    /// Registry completeness + integrity pin for the embedded
    /// `builtin_templates.json`.
    #[test]
    fn builtin_registry_is_complete_and_renderable() {
        let names = [
            "llama-2",
            "mistral",
            "devstral",
            "llama-4",
            "phi-4-mm",
            "chatml",
            "chatml-llava",
            "vicuna_v1.1",
            "llama_3_vision",
            "llava_llama_3",
            "internlm2-chat",
            "internvl-2-5",
            "qwen2-vl",
            "deepseek-ocr",
            "unlimited-ocr",
            "paddle-ocr",
            "deepseek-vl2",
            "gemma-it",
            "gme-qwen2-vl",
            "minicpmv",
            "janus-pro",
            "minicpmo",
            "kimi-vl",
            "qwen2-audio",
            "moss-vl",
            "points-v15-chat",
            "whisper",
        ];
        for name in names {
            let spec = builtin_template(name)
                .unwrap_or_else(|| panic!("builtin template {name} missing from registry"));
            assert_eq!(spec.name, name, "lookup stamps the requested alias");
            assert!(
                SUPPORTED_STYLES.contains(&spec.style.as_str()),
                "{name}: unknown style {:?}",
                spec.style
            );
        }
        assert!(builtin_template("no-such-template").is_none());
    }
}
