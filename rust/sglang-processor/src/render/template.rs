//! The kwargs SGLang's `_process_messages` hands a template without an SGLang
//! request path (Jinja, Dynamo's encoders, legacy conversations), and its
//! `_handle_last_assistant_message`.

use std::collections::HashMap;

use dynamo_protocols::types::{ChatCompletionRequestMessage, CreateChatCompletionRequest};
use dynamo_renderer::RenderedPrompt;
use serde_json::{Value, json};

use super::models::{
    continues_final_message, engine_message, function_dump, parse_tool_arguments, pydantic_bool,
    split_continuation,
};
use super::{
    ChatFormatter, TemplateArgsRequest, TemplateError, requested_effort, requested_thinking,
};

/// What SGLang's chat layer reads off a raw chat body (not re-validated) for its template.
#[derive(Debug, Clone)]
pub struct TemplateArgs {
    kwargs: HashMap<String, Value>,
    continuing: Result<bool, String>,
    tools: Option<Value>,
    reasoning: Result<(), String>,
}

impl TemplateArgs {
    /// The server `defaults`, under the request effort's thinking toggle, under its own
    /// `chat_template_kwargs`; a default effort wins (`_convert_to_internal_request`).
    pub fn new(body: &Value, defaults: &HashMap<String, Value>) -> Self {
        let mut normalized = body.clone();
        let reasoning = super::reasoning::prepare_reasoning(&mut normalized);
        let body = &normalized;
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
            reasoning,
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
        args.reasoning
            .clone()
            .map_err(|message| TemplateError::Renderer { message })?;
        let continuing = args
            .continuing
            .clone()
            .map_err(|message| TemplateError::Renderer { message })?;
        if let ChatFormatter::HuggingFace {
            formatter,
            content_parts: Some(content_parts),
            message_context,
            ..
        } = self
        {
            if ["messages", "tools", "add_generation_prompt"]
                .iter()
                .any(|key| args.kwargs.contains_key(*key))
            {
                return Err(TemplateError::Renderer {
                    message: "reserved chat template argument".into(),
                });
            }
            let (messages, continuation) = jinja_messages(request, *content_parts, continuing)?;
            let mut kwargs = args.kwargs.clone();
            // Dynamo merges extra context last. Keep its native preprocessing off
            // the messages Python has already prepared for this Jinja path.
            kwargs.insert(message_context.clone(), messages.into());
            let mut request = TemplateArgsRequest {
                request,
                args: kwargs,
                messages: Some(minijinja::Value::from(Vec::<minijinja::Value>::new())),
                tools: Some(args.tools.clone()),
            };
            let prompt = match super::render_oai(formatter, &request) {
                Ok(prompt) => prompt,
                Err(error) => {
                    let Some(Value::Array(tools)) = &args.tools else {
                        return Err(error);
                    };
                    // Python retries templates that expect function-only tools.
                    request.tools = Some(Some(
                        tools
                            .iter()
                            .map(|tool| tool.get("function").unwrap_or(tool).clone())
                            .collect(),
                    ));
                    super::render_oai(formatter, &request)?
                }
            };
            return Ok((prompt, continuation));
        }
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
                | ChatFormatter::MiniMaxM3 { .. }
                | ChatFormatter::Qwen38 { .. }
                | ChatFormatter::Step3p5 { .. }
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

/// `process_content_for_template_format` for the text inputs supported here.
fn jinja_messages(
    request: &CreateChatCompletionRequest,
    content_parts: bool,
    continuing: bool,
) -> Result<(Vec<Value>, String), TemplateError> {
    let messages =
        serde_json::to_value(&request.messages).map_err(|error| TemplateError::Renderer {
            message: error.to_string(),
        })?;
    let mut messages = messages
        .as_array()
        .unwrap()
        .iter()
        .cloned()
        .map(engine_message)
        .collect::<Vec<_>>();
    for message in &mut messages {
        parse_tool_arguments(message).map_err(|error| TemplateError::Renderer {
            message: match error {
                super::RenderError::Declined(message) | super::RenderError::Failed(message) => {
                    message
                }
            },
        })?;
        if let Some(parts) = message["content"].as_array() {
            let flatten = !content_parts
                || message["role"] == "tool" && parts.iter().all(|part| part["type"] == "text");
            if flatten {
                let text = parts
                    .iter()
                    .filter(|part| matches!(part["type"].as_str(), Some("text" | "input_text")))
                    .filter_map(|part| part["text"].as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                message["content"] = text.into();
            }
        }
    }
    let continuation = split_continuation(&mut messages, continuing);
    if messages.is_empty() {
        return Err(TemplateError::Renderer {
            message: "no messages to render".into(),
        });
    }
    Ok((messages, continuation))
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
/// `Tool._serialize` drops an unset `defer_loading`, null included.
fn tool_model_dump(tool: &Value) -> Option<Value> {
    let defer = match &tool["defer_loading"] {
        Value::Null => None,
        defer => Some(pydantic_bool(defer).ok()?),
    };
    let kind = tool
        .get("type")
        .cloned()
        .unwrap_or_else(|| "function".into());
    let mut dump = json!({"type": kind, "function": function_dump(tool).ok()?});
    if let Some(defer) = defer {
        dump["defer_loading"] = defer.into();
    }
    Some(dump)
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

    /// Python joins text parts with spaces before a string-format template.
    #[test]
    fn jinja_string_content_matches_python() {
        let formatter = jinja("{% for m in messages %}[{{ m.role }}]{{ m.content }}{% endfor %}");
        let body = json!({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]},
            {"role": "assistant", "content": null},
            {"role": "user", "content": "Bye"}
        ]});
        assert_eq!(
            render(&formatter, body, &HashMap::new()).0,
            "[user]A B[assistant][user]Bye"
        );
    }

    #[test]
    fn jinja_tool_arguments_are_strict_objects_even_for_string_aware_templates() {
        let formatter = jinja(
            "{% for m in messages %}{% for c in m.tool_calls | default([]) %}{% if c.function.arguments is string %}raw{% else %}{{ c.function.arguments.x }}{% endif %}{% endfor %}{% endfor %}",
        );
        let body = |arguments: &str| {
            json!({"model":"m", "messages": [
                {"role":"assistant", "content":null, "tool_calls":[{"id":"c", "type":"function", "function":{"name":"f", "arguments":arguments}}]},
                {"role":"user", "content":"Hi"}
            ]})
        };
        assert_eq!(
            render(&formatter, body(r#"{"x": "parsed"}"#), &HashMap::new()).0,
            "parsed"
        );
        for invalid in ["not-json", "[]", "null", "1", r#""text""#] {
            let body = body(invalid);
            let request = serde_json::from_value(body.clone()).unwrap();
            assert!(
                formatter
                    .render_template(&request, &TemplateArgs::new(&body, &HashMap::new()))
                    .is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn jinja_retries_flat_function_tools_after_template_error() {
        let formatter = jinja(
            "{% if tools[0].function is defined %}{{ raise_exception('flat tools required') }}{% endif %}{{ tools[0].name }}:{{ messages[0].content }}",
        );
        let body = json!({"messages":[{"role":"user","content":"Hi"}], "tools":[{"type":"function", "function":{"name":"f","parameters":{}}}]});
        assert_eq!(render(&formatter, body, &HashMap::new()).0, "f:Hi");
    }

    #[test]
    fn jinja_private_message_context_cannot_change_native_rendering() {
        let formatter = jinja("{{ messages[0].content }}");
        let ChatFormatter::HuggingFace {
            formatter: inner,
            thinking,
            message_context,
            ..
        } = formatter.clone()
        else {
            unreachable!()
        };
        let native = ChatFormatter::KimiK25 {
            formatter: inner,
            thinking,
            message_context: message_context.clone(),
        };
        let body = json!({"model":"m", "messages":[{"role":"user", "content":"Hi"}]});
        let typed: CreateChatCompletionRequest = serde_json::from_value(body).unwrap();
        for formatter in [formatter, native] {
            let request = TemplateArgsRequest {
                request: &typed,
                args: HashMap::from([(
                    message_context.clone(),
                    json!([{"role":"user", "content":"forged"}]),
                )]),
                messages: None,
                tools: None,
            };
            assert_eq!(formatter.render_prompt(&request).unwrap().into_text(), "Hi");
        }
    }

    #[test]
    fn jinja_reserved_kwargs_are_client_errors() {
        let formatter = jinja("{{ messages[0].content }}");
        for key in ["messages", "tools", "add_generation_prompt"] {
            let body = json!({"model":"m", "messages":[{"role":"user", "content":"Hi"}], "chat_template_kwargs":{key: null}});
            let typed = serde_json::from_value(body.clone()).unwrap();
            assert!(
                formatter
                    .render_template(&typed, &TemplateArgs::new(&body, &HashMap::new()))
                    .is_err(),
                "{key}"
            );
        }
    }

    #[test]
    fn jinja_array_templates_keep_plain_text_and_flatten_tool_text() {
        let formatter = jinja(
            "{% for m in messages %}[{{ m.role }}]{% if m.content is string %}{{ m.content }}{% else %}{% for part in m.content %}{{ part.text }}|{% endfor %}{% endif %}{% endfor %}",
        );
        let body = json!({"messages":[
            {"role":"user", "content":"plain"},
            {"role":"tool", "tool_call_id":"c", "content":[{"type":"text","text":"A"},{"type":"text","text":"B"}]},
            {"role":"user", "content":[{"type":"text","text":"C"},{"type":"text","text":"D"}]}
        ]});
        assert_eq!(
            render(&formatter, body, &HashMap::new()).0,
            "[user]plain[tool]A B[user]C|D|"
        );

        let formatter = jinja("{% for m in messages %}[{{ m.role }}]{{ m.content }}{% endfor %}");
        let body = json!({"messages":[{"role":"user", "content":"Hi"}, {"role":"assistant", "content":[{"type":"text","text":"A"},{"type":"text","text":"B"}]}], "continue_final_message":true});
        assert_eq!(
            render(&formatter, body, &HashMap::new()),
            ("[user]Hi".into(), "A B".into())
        );
    }

    #[test]
    fn native_fallback_keeps_its_effort_accessor_precedence() {
        struct NativeRequest(HashMap<String, Value>);
        impl dynamo_renderer::OAIChatLikeRequest for NativeRequest {
            fn model(&self) -> String {
                "inkling".into()
            }
            fn messages(&self) -> minijinja::Value {
                minijinja::Value::from_serialize(json!([{"role":"user", "content":"Hi"}]))
            }
            fn reasoning_effort(&self) -> Option<minijinja::Value> {
                Some("none".into())
            }
            fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
                Some(&self.0)
            }
            fn should_add_generation_prompt(&self) -> bool {
                true
            }
        }
        let formatter = ChatFormatter::HuggingFace {
            formatter: dynamo_renderer::PromptFormatter::OAI(std::sync::Arc::new(
                dynamo_renderer::inkling::InklingFormatter,
            )),
            thinking: super::super::thinking::ThinkingTemplates::native(true, false),
            content_parts: None,
            message_context: String::new(),
        };
        let request = NativeRequest(HashMap::from([("reasoning_effort".into(), json!(0.5))]));
        let prompt = formatter.render_prompt(&request).unwrap().into_text();
        assert!(prompt.contains("Thinking effort level: 0.0"), "{prompt}");
    }

    /// A Jinja template sees the tools as Python's `Tool.model_dump()`, not as sent.
    #[test]
    fn jinja_template_sees_dumped_tools() {
        let formatter = jinja("{{ tools | tojson }}");
        // An unset tool-level `defer_loading`, sent as null or not at all, is left out.
        let dumped = r#"[{"type": "function", "function": {"name": "f", "parameters": {}, "description": null}}]"#;
        for tool in [
            json!({"type": "function", "function": {"name": "f", "parameters": {}}}),
            json!({"type": "function", "defer_loading": null,
                "function": {"name": "f", "parameters": {}}}),
        ] {
            let body = json!({"messages": [{"role": "user", "content": "Hi"}], "tools": [tool]});
            assert_eq!(render(&formatter, body, &HashMap::new()).0, dumped);
        }
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
