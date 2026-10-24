//! The kwargs SGLang's `_process_messages` hands a template without an SGLang
//! request path (Jinja, Dynamo's encoders, legacy conversations), and its
//! `_handle_last_assistant_message`.

use std::collections::HashMap;

use dynamo_protocols::types::{ChatCompletionRequestMessage, CreateChatCompletionRequest};
use dynamo_renderer::RenderedPrompt;
use serde_json::Value;

use super::{
    ChatFormatter, TemplateArgsRequest, TemplateError, requested_effort, requested_thinking,
};

/// What SGLang's chat layer reads off a raw chat body (not re-validated) for its template.
#[derive(Debug, Clone, Default)]
pub struct TemplateArgs {
    kwargs: HashMap<String, Value>,
    continuing: bool,
}

impl TemplateArgs {
    /// The server `defaults`, under the thinking toggle the request's effort
    /// implies, under the request's own `chat_template_kwargs`. Its effort moves
    /// onto the request's (`_convert_to_internal_request`), unless a default names one.
    pub fn new(body: &Value, defaults: &HashMap<String, Value>) -> Self {
        let mut kwargs = defaults.clone();
        if let Some(thinking) = requested_thinking(body) {
            kwargs.insert("thinking".into(), thinking.into());
            kwargs.insert("enable_thinking".into(), thinking.into());
        }
        let mut own = body["chat_template_kwargs"]
            .as_object()
            .cloned()
            .unwrap_or_default();
        // `pop("reasoning_effort", None)`: a null keeps the request's effort.
        let effort = own
            .remove("reasoning_effort")
            .filter(|effort| !effort.is_null())
            .or_else(|| requested_effort(body).cloned());
        kwargs.extend(own);
        if let Some(effort) = effort {
            kwargs.entry("reasoning_effort".into()).or_insert(effort);
        }
        Self {
            kwargs,
            continuing: body["continue_final_message"] == true,
        }
    }
}

impl ChatFormatter {
    /// Render `request` through the formatter's own template as SGLang's chat
    /// layer does, returning the prompt and the `continue_final_message` text
    /// apart from it.
    pub fn render_template(
        &self,
        request: &CreateChatCompletionRequest,
        args: &TemplateArgs,
    ) -> Result<(RenderedPrompt, String), TemplateError> {
        let (messages, continuation) = match self {
            ChatFormatter::Legacy(_) => None,
            // SGLang's K3 encoder splits a trailing assistant turn off only to continue it.
            ChatFormatter::KimiK3 if !args.continuing => None,
            _ => last_assistant_message(request, args.continuing),
        }
        .map_or((None, String::new()), |(messages, prefix)| {
            (Some(messages), prefix)
        });
        let args = args.kwargs.clone();
        let request = TemplateArgsRequest {
            request,
            args,
            messages,
        };
        Ok((self.render_prompt(&request)?, continuation))
    }
}

/// Python's `_handle_last_assistant_message`: a trailing assistant message's
/// text is the continuation under `continue_final_message`, else a user turn.
/// The messages to render and the continuation, if the last one is such.
fn last_assistant_message(
    request: &CreateChatCompletionRequest,
    continuing: bool,
) -> Option<(minijinja::Value, String)> {
    if !matches!(
        request.messages.last(),
        Some(ChatCompletionRequestMessage::Assistant(_))
    ) {
        return None;
    }
    let mut messages = serde_json::to_value(&request.messages).ok()?;
    let list = messages.as_array_mut()?;
    // `_process_messages` first sets a null content to "".
    let content = match &list.last()?["content"] {
        Value::Null => String::new(),
        content => content.as_str()?.to_owned(),
    };
    let prefix = if continuing {
        list.pop();
        content
    } else {
        *list.last_mut()? = serde_json::json!({"role": "user", "content": content});
        String::new()
    };
    Some((minijinja::Value::from_serialize(&messages), prefix))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::render::{ChatFormatterOptions, select_chat_formatter};

    /// The formatter a checkpoint with this Jinja `template` selects.
    fn jinja(template: &str) -> ChatFormatter {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("template-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        std::fs::write(dir.join("chat_template.jinja"), template).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
        let (formatter, error) = select_chat_formatter(&ChatFormatterOptions {
            tokenizer_path: dir.to_string_lossy().into_owned(),
            ..Default::default()
        });
        std::fs::remove_dir_all(&dir).unwrap();
        formatter.unwrap_or_else(|| panic!("{error:?}"))
    }

    fn render(
        formatter: &ChatFormatter,
        body: Value,
        defaults: &HashMap<String, Value>,
    ) -> (String, String) {
        let request = serde_json::from_value(body.clone()).unwrap();
        let (prompt, continuation) = formatter
            .render_template(&request, &TemplateArgs::new(&body, defaults))
            .unwrap();
        (prompt.into_text(), continuation)
    }

    /// Python's `_handle_last_assistant_message`: the trailing assistant text
    /// is continued apart from the prompt, or asked as a user turn without
    /// `continue_final_message`.
    #[test]
    fn trailing_assistant_message_is_continued_or_asked() {
        let formatter = jinja(
            "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}\
             {% if add_generation_prompt %}<assistant>{% endif %}",
        );
        let none = HashMap::new();
        let messages = json!([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hel"}
        ]);
        let body = json!({"model": "m", "messages": messages, "continue_final_message": true});
        assert_eq!(
            render(&formatter, body, &none),
            ("<user>Hi<assistant>".into(), "Hel".into())
        );
        let body = json!({"model": "m", "messages": messages});
        assert_eq!(
            render(&formatter, body, &none),
            ("<user>Hi<user>Hel<assistant>".into(), String::new())
        );

        // A null content is "" first, so the turn is still asked as a user's.
        let call =
            json!({"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}});
        let messages = json!([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": null, "tool_calls": [call]}
        ]);
        let body = json!({"model": "m", "messages": messages});
        assert_eq!(
            render(&formatter, body, &none).0,
            "<user>Hi<user><assistant>"
        );
    }

    /// Kimi K3's template fallback keeps a trailing assistant turn closed
    /// unless it is continued, as SGLang's K3 encoder does.
    #[test]
    fn kimi_k3_trailing_assistant_message_is_only_continued() {
        let formatter = ChatFormatter::KimiK3;
        let none = HashMap::new();
        let messages = json!([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hel"}
        ]);
        let body = json!({"model": "m", "messages": messages});
        let request: CreateChatCompletionRequest = serde_json::from_value(body.clone()).unwrap();
        let closed = formatter.render_prompt(&request).unwrap().into_text();
        assert_eq!(render(&formatter, body, &none), (closed, String::new()));

        let body = json!({"model": "m", "messages": messages, "continue_final_message": true});
        let (prompt, continuation) = render(&formatter, body, &none);
        assert_eq!(continuation, "Hel");
        assert!(!prompt.contains("Hel"), "{prompt}");
    }

    /// The request's effort reaches the template with the thinking toggle it
    /// implies, beneath the request's own `chat_template_kwargs`; a server
    /// default effort wins over the request's, as on Python.
    #[test]
    fn template_sees_the_requested_effort() {
        let formatter = jinja("{{ reasoning_effort }}|{{ thinking }}|{{ enable_thinking }}");
        let none = HashMap::new();
        let messages = json!([{"role": "user", "content": "Hi"}]);
        let body = json!({"model": "m", "messages": messages, "reasoning_effort": "none"});
        assert_eq!(render(&formatter, body, &none).0, "none|False|False");
        let body = json!({
            "model": "m",
            "messages": messages,
            "reasoning_effort": "high",
            "chat_template_kwargs": {"enable_thinking": false}
        });
        assert_eq!(render(&formatter, body, &none).0, "high|True|False");

        let body = json!({"model": "m", "messages": messages, "reasoning_effort": "high"});
        let defaults = HashMap::from([("reasoning_effort".into(), json!("low"))]);
        assert_eq!(render(&formatter, body, &defaults).0, "low|True|True");

        // A null kwargs effort is no effort, so the request's still applies.
        let body = json!({
            "model": "m",
            "messages": messages,
            "reasoning_effort": "high",
            "chat_template_kwargs": {"reasoning_effort": null}
        });
        assert_eq!(render(&formatter, body, &none).0, "high|True|True");
    }

    /// `reasoning_effort` sets both thinking toggles beneath the request's own
    /// kwargs, and the server defaults fill in only what is left.
    #[test]
    fn effort_sets_thinking_beneath_the_requests_kwargs() {
        let defaults = HashMap::from([
            ("enable_thinking".to_owned(), json!(true)),
            ("other".to_owned(), json!(1)),
        ]);
        let body = json!({
            "reasoning_effort": "none",
            "chat_template_kwargs": {"thinking": true},
        });
        let args = TemplateArgs::new(&body, &defaults);
        assert_eq!(
            args.kwargs,
            HashMap::from([
                ("reasoning_effort".to_owned(), json!("none")),
                ("thinking".to_owned(), json!(true)),
                ("enable_thinking".to_owned(), json!(false)),
                ("other".to_owned(), json!(1)),
            ])
        );
    }
}
