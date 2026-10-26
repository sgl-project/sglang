//! The kwargs SGLang's `_process_messages` hands a template without an SGLang
//! request path (Jinja, Dynamo's encoders, legacy conversations), and its
//! `_handle_last_assistant_message`.

use std::collections::HashMap;

use dynamo_protocols::types::{ChatCompletionRequestMessage, CreateChatCompletionRequest};
use dynamo_renderer::RenderedPrompt;
use serde_json::{Value, json};

use super::models::{continues_final_message, function_dump, pydantic_bool, split_continuation};
use super::{
    ChatFormatter, TemplateArgsRequest, TemplateError, requested_effort, requested_thinking,
};

/// What SGLang's chat layer reads off a raw chat body (not re-validated) for its template.
#[derive(Debug, Clone)]
pub struct TemplateArgs {
    kwargs: HashMap<String, Value>,
    continuing: Result<bool, String>,
    tools: Option<Value>,
}

impl TemplateArgs {
    /// The server `defaults`, under the request effort's thinking toggle, under its own
    /// `chat_template_kwargs`; a default effort wins (`_convert_to_internal_request`).
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
            continuing: continues_final_message(body),
            tools: jinja_tools(body),
        }
    }
}

impl ChatFormatter {
    /// Render `request` through the formatter's own template as SGLang's chat
    /// layer does, returning the prompt and the `continue_final_message` text.
    pub fn render_template(
        &self,
        request: &CreateChatCompletionRequest,
        args: &TemplateArgs,
    ) -> Result<(RenderedPrompt, String), TemplateError> {
        let continuing = args
            .continuing
            .clone()
            .map_err(|message| TemplateError::Renderer { message })?;
        let (messages, continuation) = match self {
            ChatFormatter::Legacy(_) => None,
            // SGLang's K3 encoder splits a trailing assistant turn off only to continue it.
            ChatFormatter::KimiK3 if !continuing => None,
            _ => last_assistant_message(request, continuing)?,
        }
        .unzip();
        let jinja = matches!(
            self,
            ChatFormatter::HuggingFace { .. }
                | ChatFormatter::Glm53(_)
                | ChatFormatter::Qwen38 { .. }
        );
        let request = TemplateArgsRequest {
            request,
            args: args.kwargs.clone(),
            messages,
            tools: jinja.then(|| args.tools.clone()),
        };
        let prompt = self.render_prompt(&request)?;
        Ok((prompt, continuation.unwrap_or_default()))
    }
}

/// The tools Python's Jinja path hands the template: the request's (only the
/// named one under a named `tool_choice`, none under `"none"`), as `Tool.model_dump()`.
fn jinja_tools(body: &Value) -> Option<Value> {
    let tools = body["tools"].as_array()?;
    let named = match &body["tool_choice"] {
        Value::String(choice) if choice == "none" => return None,
        choice => choice["function"]["name"].as_str(),
    };
    let tools: Vec<Value> = tools
        .iter()
        .filter(|tool| named.is_none_or(|name| tool["function"]["name"] == name))
        // A tool Python's validation rejects never reaches its template.
        .filter_map(tool_model_dump)
        .collect();
    (!tools.is_empty()).then_some(Value::Array(tools))
}

/// `Tool.model_dump()`, after `_validate_request` normalizes the schema types.
fn tool_model_dump(tool: &Value) -> Option<Value> {
    let defer = match &tool["defer_loading"] {
        Value::Null => Value::Null,
        defer => pydantic_bool(defer).ok()?.into(),
    };
    let kind = tool
        .get("type")
        .cloned()
        .unwrap_or_else(|| "function".into());
    Some(json!({"type": kind, "function": function_dump(tool).ok()?, "defer_loading": defer}))
}

/// [`split_continuation`] over the request's messages when the last is an
/// assistant turn, its null content first set to "" as `_process_messages` does.
fn last_assistant_message(
    request: &CreateChatCompletionRequest,
    continuing: bool,
) -> Result<Option<(minijinja::Value, String)>, TemplateError> {
    if !matches!(
        request.messages.last(),
        Some(ChatCompletionRequestMessage::Assistant(_))
    ) {
        return Ok(None);
    }
    let Ok(Value::Array(mut messages)) = serde_json::to_value(&request.messages) else {
        return Ok(None);
    };
    if let Some(content) = messages.last_mut().map(|last| &mut last["content"])
        && content.is_null()
    {
        *content = "".into();
    }
    let prefix = split_continuation(&mut messages, continuing);
    // `apply_chat_template` fails on an empty conversation.
    if messages.is_empty() {
        return Err(TemplateError::Renderer {
            message: "no messages to render".into(),
        });
    }
    Ok(Some((minijinja::Value::from_serialize(&messages), prefix)))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn jinja(template: &str) -> ChatFormatter {
        ChatFormatter::from_tokenizer_config(&json!({ "chat_template": template })).unwrap()
    }

    fn render(
        formatter: &ChatFormatter,
        mut body: Value,
        defaults: &HashMap<String, Value>,
    ) -> (String, String) {
        body["model"] = "m".into();
        let request = serde_json::from_value(body.clone()).unwrap();
        let (prompt, continuation) = formatter
            .render_template(&request, &TemplateArgs::new(&body, defaults))
            .unwrap();
        (prompt.into_text(), continuation)
    }

    fn trailing(content: Value) -> Value {
        json!([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": content}])
    }

    /// Python's `_handle_last_assistant_message`: a trailing assistant turn is
    /// continued apart from the prompt, or asked as a user turn.
    #[test]
    fn trailing_assistant_message_is_continued_or_asked() {
        let formatter = jinja(
            "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}\
             {% if add_generation_prompt %}<assistant>{% endif %}",
        );
        let none = HashMap::new();
        // pydantic's lax `bool`, as Python's request model reads the flag.
        for flag in [json!(true), json!(1), json!("true")] {
            let body = json!({"messages": trailing("Hel".into()), "continue_final_message": flag});
            let expected = ("<user>Hi<assistant>".into(), "Hel".into());
            assert_eq!(render(&formatter, body, &none), expected);
        }
        let body = json!({"messages": trailing("Hel".into())});
        let expected = ("<user>Hi<user>Hel<assistant>".into(), String::new());
        assert_eq!(render(&formatter, body, &none), expected);

        // A null content is "" first, so the turn is still asked as a user's.
        let mut messages = trailing(Value::Null);
        messages[1]["tool_calls"] =
            json!([{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]);
        let (prompt, _) = render(&formatter, json!({"messages": messages}), &none);
        assert_eq!(prompt, "<user>Hi<user><assistant>");

        // Continuing the only turn leaves nothing for `apply_chat_template`.
        let body = json!({
            "model": "m",
            "messages": [{"role": "assistant", "content": "Hel"}],
            "continue_final_message": true,
        });
        let request = serde_json::from_value(body.clone()).unwrap();
        let args = TemplateArgs::new(&body, &none);
        assert!(formatter.render_template(&request, &args).is_err());
    }

    /// A Jinja template sees the tools as Python's `Tool.model_dump()`, not as sent.
    #[test]
    fn jinja_template_sees_dumped_tools() {
        let formatter = jinja("{{ tools | tojson }}");
        let tool = json!({"type": "function", "function": {"name": "f", "parameters": {}}});
        let body = json!({"messages": [{"role": "user", "content": "Hi"}], "tools": [tool]});
        let dumped = r#"[{"type": "function", "function": {"name": "f", "parameters": {}, "description": null}, "defer_loading": null}]"#;
        assert_eq!(render(&formatter, body, &HashMap::new()).0, dumped);
        // A set `strict` stays where the client put it; a tool-level flag propagates.
        let tool = json!({"type": "function", "defer_loading": true,
            "function": {"strict": true, "name": "f", "description": "d"}});
        let body = json!({"messages": [{"role": "user", "content": "Hi"}], "tools": [tool]});
        let dumped = r#"[{"type": "function", "function": {"strict": true, "name": "f", "description": "d", "parameters": null, "defer_loading": true}, "defer_loading": true}]"#;
        assert_eq!(render(&formatter, body, &HashMap::new()).0, dumped);
    }

    /// Kimi K3's template fallback, like SGLang's K3 encoder, splits a trailing
    /// assistant turn off only to continue it.
    #[test]
    fn kimi_k3_trailing_assistant_message_is_only_continued() {
        let formatter = ChatFormatter::KimiK3;
        let none = HashMap::new();
        let body = json!({"model": "m", "messages": trailing("Hel".into())});
        let request: CreateChatCompletionRequest = serde_json::from_value(body.clone()).unwrap();
        let closed = formatter.render_prompt(&request).unwrap().into_text();
        assert_eq!(render(&formatter, body, &none), (closed, String::new()));

        let body = json!({"messages": trailing("Hel".into()), "continue_final_message": true});
        let (prompt, continuation) = render(&formatter, body, &none);
        assert_eq!(continuation, "Hel");
        assert!(!prompt.contains("Hel"), "{prompt}");
    }

    /// The request's effort and the thinking toggle it implies sit beneath its
    /// own `chat_template_kwargs`; a server default effort wins, as on Python.
    #[test]
    fn template_sees_the_requested_effort() {
        let formatter = jinja("{{ reasoning_effort }}|{{ thinking }}|{{ enable_thinking }}");
        let render = |effort: &str, kwargs: Value, defaults: &HashMap<String, Value>| {
            let messages = json!([{"role": "user", "content": "Hi"}]);
            let body = json!({
                "messages": messages,
                "reasoning_effort": effort,
                "chat_template_kwargs": kwargs
            });
            render(&formatter, body, defaults).0
        };
        let none = HashMap::new();
        assert_eq!(render("none", json!({}), &none), "none|False|False");
        let kwargs = json!({"enable_thinking": false});
        assert_eq!(render("high", kwargs, &none), "high|True|False");
        // A null kwargs effort is no effort, so the request's still applies.
        let kwargs = json!({"reasoning_effort": null});
        assert_eq!(render("high", kwargs, &none), "high|True|True");
        let defaults = HashMap::from([("reasoning_effort".into(), json!("low"))]);
        assert_eq!(render("high", json!({}), &defaults), "low|True|True");
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
