//! Legacy SGLang conversation-template rendering.

use dynamo_protocols::types::{
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestAssistantMessageContentPart,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestSystemMessageContentPart, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, CreateChatCompletionRequest,
};

use crate::message::types::OneOrMany;

use super::template::TemplateError;

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
    pub(super) spec: LegacySpec,
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
