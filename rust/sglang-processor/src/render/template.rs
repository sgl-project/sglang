//! The kwargs SGLang's `_process_messages` hands a template without an SGLang
//! request path (Jinja, Dynamo's encoders, legacy conversations), and its
//! `_handle_last_assistant_message`.

use std::collections::HashMap;

use dynamo_protocols::types::{ChatCompletionRequestMessage, CreateChatCompletionRequest};
use dynamo_renderer::RenderedPrompt;
use serde_json::{Map, Value, json};

use super::models::{normalize_json_schema_types, pydantic_bool};
use super::{
    ChatFormatter, TemplateArgsRequest, TemplateError, requested_effort, requested_thinking,
};

/// What SGLang's chat layer reads off a raw chat body (not re-validated) for its template.
#[derive(Debug, Clone, Default)]
pub struct TemplateArgs {
    kwargs: HashMap<String, Value>,
    continuing: bool,
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
            continuing: body["continue_final_message"] == true,
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
        let (messages, continuation) = match self {
            ChatFormatter::Legacy(_) => None,
            // SGLang's K3 encoder splits a trailing assistant turn off only to continue it.
            ChatFormatter::KimiK3 if !args.continuing => None,
            _ => last_assistant_message(request, args.continuing),
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
        .map(tool_model_dump)
        .collect();
    (!tools.is_empty()).then_some(Value::Array(tools))
}

/// `Tool.model_dump()`, after `_validate_request` normalizes the schema types.
fn tool_model_dump(tool: &Value) -> Value {
    let flag = |value: &Value| pydantic_bool(value).ok();
    let f = &tool["function"];
    let mut parameters = f["parameters"].clone();
    normalize_json_schema_types(&mut parameters);
    // `Function`'s fields as declared; `Tool._propagate_defer_loading` fills its
    // flag, and `Function._serialize` drops an unset `strict` or `defer_loading`.
    let declared = [
        ("description", Some(f["description"].clone())),
        ("name", Some(f["name"].clone())),
        ("parameters", Some(parameters)),
        ("strict", flag(&f["strict"]).map(Value::from)),
        (
            "defer_loading",
            flag(&f["defer_loading"])
                .or_else(|| flag(&tool["defer_loading"]))
                .map(Value::from),
        ),
    ];
    // `Function._serialize`: the keys in the order the client sent them, then
    // the rest as declared.
    let sent = f
        .as_object()
        .into_iter()
        .flat_map(|f| f.keys().map(String::as_str));
    let mut function = Map::new();
    for key in sent.chain(declared.iter().map(|(key, _)| *key)) {
        if let Some((_, Some(value))) = declared.iter().find(|(name, _)| *name == key) {
            function.entry(key).or_insert_with(|| value.clone());
        }
    }
    let kind = tool
        .get("type")
        .cloned()
        .unwrap_or_else(|| "function".into());
    json!({"type": kind, "function": function, "defer_loading": flag(&tool["defer_loading"])})
}

/// Python's `_handle_last_assistant_message`: a trailing assistant message's text
/// is the continuation under `continue_final_message`, else a user turn.
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
        let body = json!({"messages": trailing("Hel".into()), "continue_final_message": true});
        let expected = ("<user>Hi<assistant>".into(), "Hel".into());
        assert_eq!(render(&formatter, body, &none), expected);
        let body = json!({"messages": trailing("Hel".into())});
        let expected = ("<user>Hi<user>Hel<assistant>".into(), String::new());
        assert_eq!(render(&formatter, body, &none), expected);

        // A null content is "" first, so the turn is still asked as a user's.
        let mut messages = trailing(Value::Null);
        messages[1]["tool_calls"] =
            json!([{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]);
        let (prompt, _) = render(&formatter, json!({"messages": messages}), &none);
        assert_eq!(prompt, "<user>Hi<user><assistant>");
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
