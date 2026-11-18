//! StepFun's Step-3.5-Flash, Step-3.7-Flash and Step-5-Preview prompts as
//! SGLang's `serving_chat.py` renders them with the checkpoint's Jinja template,
//! on Dynamo's template formatter.

use std::collections::HashMap;

use dynamo_renderer::PromptFormatter;
use serde_json::{Value, json};

use super::glm53::has_negative_zero;
use super::pydantic_dump::{
    continues_final_message, engine_message, parse_tool_arguments, split_continuation,
};
use super::qwen38::{template_args, template_tools};
use super::template_context::PreservedRequest;
use crate::py::py_truthy;
use crate::render::RenderError;
use crate::render::reasoning::{EFFORT_TIERS, requested_effort};

/// `apply_chat_template` arguments SGLang passes itself, which fail its call as
/// kwargs. A falsy `continue_final_message` or `return_assistant_tokens_mask` is
/// that argument's default and a server's default `chat_template` renders
/// another template; the engine renders those.
const FAILING_KWARGS: &[&str] = &[
    "add_generation_prompt",
    "conversation",
    "messages",
    "return_dict",
    "tokenize",
    "tools",
];

/// `apply_chat_template` flags that fail its call only when truthy.
const DEFAULT_FALSE_KWARGS: &[&str] = &["continue_final_message", "return_assistant_tokens_mask"];

/// Dynamo's own template context, which `chat_template_kwargs` cannot override
/// as they override `apply_chat_template`'s special tokens in Python.
const CONTEXT_TOKENS: &[&str] = &["bos_token", "eos_token", "unk_token"];

const MEDIA_PARTS: &[&str] = &[
    "image_url",
    "input_image",
    "video_url",
    "audio_url",
    "input_audio",
];

/// StepFun's templates (Step-3.5-Flash, Step-3.7-Flash, Step-5-Preview):
/// Qwen3-Coder's tool-call format, tool results in a `tool_response` turn, and
/// `<im_patch>` for images.
pub(crate) fn is_step3p5(template: &str) -> bool {
    [
        "<|im_start|>tool_response",
        "<im_patch>",
        "<function=",
        "<parameter=",
    ]
    .iter()
    .all(|text| template.contains(text))
}

/// Render an SGLang chat request body. Returns the prompt and the
/// `continue_final_message` prefix, which SGLang tokenizes separately.
pub(crate) fn render(
    formatter: &PromptFormatter,
    content_parts: Option<bool>,
    message_context: &str,
    mut request: Value,
    defaults: &HashMap<String, Value>,
) -> Result<(String, String), RenderError> {
    let Some(Value::Array(messages)) = request.get_mut("messages").map(Value::take) else {
        return Err("messages must be an array".into());
    };
    // `_validate_request`.
    if messages.is_empty() {
        return Err("Messages cannot be empty.".into());
    }
    validate_tool_choice(&request, &messages)?;
    coerce_effort(&mut request)?;
    failing_kwargs(&request, defaults)?;
    let mut args = template_args(&request, defaults)?;
    if let Some(key) = CONTEXT_TOKENS.iter().find(|key| args.contains_key(**key)) {
        return Err(RenderError::Declined(format!("chat_template_kwargs.{key}")));
    }
    let tools = template_tools(&request)?;
    let mut messages: Vec<Value> = messages.into_iter().map(engine_message).collect();
    for message in &mut messages {
        parse_tool_arguments(message)?;
    }
    let mut messages = messages
        .into_iter()
        .map(|message| template_message(message, content_parts))
        .collect::<Result<Vec<_>, _>>()?;
    let prefix = split_continuation(&mut messages, continues_final_message(&request)?);
    // `apply_chat_template` fails on an empty conversation.
    if messages.is_empty() {
        return Err("no messages to render".into());
    }
    if messages
        .iter()
        .chain(tools.iter().flatten())
        .any(has_negative_zero)
    {
        return Err(RenderError::Declined("-0".into()));
    }
    let PromptFormatter::OAI(formatter) = formatter;
    args.insert(message_context.into(), Value::Array(messages));
    let prompt = formatter
        .render(&PreservedRequest { tools, args })
        .map_err(|error| format!("{error:#}"))?;
    Ok((prompt, prefix))
}

/// `_validate_request`'s `tool_choice` checks, over the request's tools and the
/// tools `root`, `system` and `developer` messages carry (`_effective_tools`).
fn validate_tool_choice(request: &Value, messages: &[Value]) -> Result<(), String> {
    let message_tools: Vec<&Value> = messages
        .iter()
        .filter(|message| {
            let role = message["role"].as_str().unwrap_or_default().to_lowercase();
            ["root", "system", "developer"].contains(&role.as_str())
        })
        .filter_map(|message| message["tools"].as_array())
        .flatten()
        .collect();
    let tools: Vec<&Value> = request["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .chain(message_tools.iter().copied())
        .collect();
    let name = |tool: &Value| tool["function"]["name"].clone();
    match &request["tool_choice"] {
        Value::String(choice) if choice.eq_ignore_ascii_case("required") && tools.is_empty() => {
            return Err("Tools cannot be empty if tool choice is set to required.".into());
        }
        Value::Null | Value::String(_) => {}
        choice => {
            if tools.is_empty() {
                return Err(
                    "Tools cannot be empty if tool choice is set to a specific tool.".into(),
                );
            }
            let wanted = &choice["function"]["name"];
            if !tools.iter().any(|tool| name(tool) == *wanted) {
                let wanted = wanted.as_str().unwrap_or_default();
                return Err(format!("Tool '{wanted}' not found in tools list."));
            }
        }
    }
    if message_tools.is_empty() {
        return Ok(());
    }
    let names: Vec<Value> = tools.iter().map(|tool| name(tool)).collect();
    if (1..names.len()).any(|i| names[..i].contains(&names[i])) {
        return Err("Tool names must be unique across request and message tools.".into());
    }
    Ok(())
}

/// The request's kwargs over the server's defaults that `apply_chat_template`
/// rejects, a 400 in Python rather than a different render; `_validate_request`
/// rejects the request's own `chat_template` unless the server trusts it.
fn failing_kwargs(request: &Value, defaults: &HashMap<String, Value>) -> Result<(), String> {
    let own = request["chat_template_kwargs"].as_object();
    if own.is_some_and(|kwargs| kwargs.contains_key("chat_template")) {
        return Err("Request-supplied chat_template is not allowed.".into());
    }
    let kwarg = |key: &&str| {
        own.and_then(|kwargs| kwargs.get(*key))
            .or_else(|| defaults.get(*key))
    };
    let failing = FAILING_KWARGS.iter().find(|key| kwarg(key).is_some());
    let truthy = DEFAULT_FALSE_KWARGS
        .iter()
        .find(|key| kwarg(key).is_some_and(py_truthy));
    match failing.or(truthy) {
        Some(key) => Err(format!("chat_template_kwargs.{key}")),
        None => Ok(()),
    }
}

/// pydantic's `ReasoningEffortType` on the request's own effort (`reasoning`'s
/// first, as `normalize_reasoning_inputs` reads it): a tier, or a float in
/// [0, 0.99], numeric strings included. The float reaches the template, which
/// cannot print it unless a kwargs effort replaces it.
fn coerce_effort(request: &mut Value) -> Result<(), String> {
    let Some(effort) = requested_effort(request).cloned() else {
        return Ok(());
    };
    let float = match &effort {
        Value::String(tier) if EFFORT_TIERS.contains(&tier.as_str()) => return Ok(()),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        Value::Number(number) => number.as_f64(),
        _ => None,
    }
    .filter(|f| (0.0..=0.99).contains(f))
    .ok_or_else(|| format!("invalid reasoning effort: {effort}"))?;
    if let Some(reasoning) = request.get_mut("reasoning").and_then(Value::as_object_mut) {
        reasoning.remove("effort");
        reasoning.remove("reasoning_effort");
    }
    request["reasoning_effort"] = float.into();
    Ok(())
}

/// The message SGLang's Jinja path hands the template: content as
/// `process_content_for_template_format` (the template's content format) and
/// `normalize_tool_content` leave it.
fn template_message(mut message: Value, content_parts: Option<bool>) -> Result<Value, RenderError> {
    if let Some(fields) = message.as_object_mut() {
        // StepFun's templates read only the request's tools.
        fields.remove("tools");
    }
    let Some(parts) = message["content"].as_array() else {
        return Ok(message);
    };
    let content_parts = content_parts.ok_or_else(|| {
        RenderError::Declined("cannot determine the template's content format".into())
    })?;
    let mut kept = Vec::new();
    for part in parts {
        match part["type"].as_str() {
            Some("text" | "input_text") => kept.push(json!({"type": "text", "text": part["text"]})),
            Some(kind) if MEDIA_PARTS.contains(&kind) => {
                return Err(RenderError::Declined("StepFun media".into()));
            }
            // The template skips every part but text and images.
            Some("tool_reference") if content_parts => {
                kept.push(json!({"type": "tool_reference", "name": part["name"]}));
            }
            _ => {}
        }
    }
    let all_text = kept.iter().all(|part| part["type"] == "text");
    message["content"] = if !content_parts || (message["role"] == "tool" && all_text) {
        let texts: Vec<_> = kept
            .iter()
            .filter_map(|part| part["text"].as_str())
            .collect();
        texts.join(" ").into()
    } else {
        kept.into()
    };
    Ok(message)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::{Value, json};

    use crate::render::{ChatFormatter, TemplateError};

    /// StepFun's markers, and the keyword that selects content parts lists.
    const MARKERS: &str = "{# <|im_start|>tool_response <im_patch> <function= <parameter= image #}";

    fn formatter(template: &str) -> ChatFormatter {
        ChatFormatter::from_tokenizer_config(&json!({
            "chat_template": template,
            "bos_token": "<s>",
        }))
        .unwrap()
    }

    fn render(formatter: &ChatFormatter, request: Value) -> Result<String, TemplateError> {
        formatter
            .render_request(request, &HashMap::new())
            .map(|(prompt, prefix)| prompt.into_text() + &prefix)
    }

    /// StepFun is known by its tool-response turn, image patch and Qwen3-Coder
    /// tool-call tags; the checkpoint templates take content parts lists.
    #[test]
    fn template_selects_step3p5() {
        assert!(matches!(
            formatter(MARKERS),
            ChatFormatter::Step3p5 {
                content_parts: Some(true),
                ..
            }
        ));
        assert!(matches!(
            formatter(&MARKERS.replace("<im_patch>", "")),
            ChatFormatter::HuggingFace { .. }
        ));
        for name in ["step-5-preview", "step-3.7-flash", "step-3.5-flash"] {
            let path = format!(
                "{}/tests/fixtures/parity/{name}.json",
                env!("CARGO_MANIFEST_DIR")
            );
            let fixture: Value =
                serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            assert!(
                matches!(
                    formatter(fixture["chat_template"].as_str().unwrap()),
                    ChatFormatter::Step3p5 {
                        content_parts: Some(true),
                        ..
                    }
                ),
                "{name}"
            );
        }
    }

    /// The template sees SGLang's messages, not Dynamo's rewrite of them: parts
    /// lists stay lists, and a later system turn keeps its role and name.
    #[test]
    fn template_sees_sglang_messages() {
        let formatter = formatter(&format!(
            "{MARKERS}{{{{ bos_token }}}}{{% for m in messages %}}[{{{{ m.role }}}}\
             {{% if m.name is defined %}}:{{{{ m.name }}}}{{% endif %}}|\
             {{% if m.content is string %}}{{{{ m.content }}}}\
             {{% else %}}{{% for p in m.content %}}<{{{{ p.text }}}}>{{% endfor %}}{{% endif %}}]\
             {{% endfor %}}"
        ));
        let prompt = render(
            &formatter,
            json!({"messages": [
                {"role": "user", "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]},
                {"role": "system", "name": "observation", "content": "O"},
            ]}),
        )
        .unwrap();
        assert_eq!(prompt, "<s>[user|<A><B>][system:observation|O]");
    }

    /// A template whose content format cannot be told leaves parts lists to the
    /// engine; Dynamo's special tokens would win over kwargs, so those decline too.
    #[test]
    fn declines_what_it_cannot_render_as_sglang() {
        let declined = |result: Result<String, TemplateError>| {
            matches!(result, Err(TemplateError::NeedsRenderFallback { .. }))
        };
        // Without the keyword, the format comes from the template's syntax tree,
        // which HF's `generation` block keeps minijinja from parsing.
        let unparsable = formatter(&format!(
            "{}{{% generation %}}{{% endgeneration %}}",
            MARKERS.replace("image", "")
        ));
        assert!(matches!(
            unparsable,
            ChatFormatter::Step3p5 {
                content_parts: None,
                ..
            }
        ));
        let parts =
            json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "A"}]}]});
        assert!(declined(render(&unparsable, parts)));
        let checkpoint = formatter(&format!("{MARKERS}{{{{ bos_token }}}}"));
        let kwargs = json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "chat_template_kwargs": {"bos_token": "X"},
        });
        assert!(declined(render(&checkpoint, kwargs)));
        let media = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}]});
        assert!(declined(render(&checkpoint, media)));
        // orjson parses nesting serde_json's recursion limit rejects.
        let nested = format!("{{\"a\": {}{}}}", "[".repeat(200), "]".repeat(200));
        let deep = json!({"messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "f", "arguments": nested}},
            ]},
            {"role": "tool", "tool_call_id": "a", "content": "r"},
        ]});
        assert!(declined(render(&checkpoint, deep)));
    }

    /// Kwargs that fail Python's `apply_chat_template` fail here too, rather than
    /// falling back to a template render a host could serve.
    #[test]
    fn apply_chat_template_kwargs_fail() {
        let checkpoint = formatter(&format!("{MARKERS}{{{{ messages[0].content }}}}"));
        let request = |kwargs: Value| {
            json!({
                "messages": [{"role": "user", "content": "Hi"}],
                "chat_template_kwargs": kwargs,
            })
        };
        for kwargs in [
            json!({"tokenize": false}),
            json!({"tools": []}),
            json!({"continue_final_message": true}),
        ] {
            assert!(
                matches!(
                    render(&checkpoint, request(kwargs.clone())),
                    Err(TemplateError::Renderer { .. })
                ),
                "{kwargs}"
            );
        }
        let defaults = HashMap::from([("return_dict".to_owned(), json!(false))]);
        assert!(matches!(
            checkpoint.render_request(request(json!({})), &defaults),
            Err(TemplateError::Renderer { .. })
        ));
        // The request's own template is a 400 (`_validate_request`); a server
        // default renders another template, on the engine.
        assert!(matches!(
            render(&checkpoint, request(json!({"chat_template": "X"}))),
            Err(TemplateError::Renderer { .. })
        ));
        let defaults = HashMap::from([("chat_template".to_owned(), json!("X"))]);
        assert!(matches!(
            checkpoint.render_request(request(json!({})), &defaults),
            Err(TemplateError::NeedsRenderFallback { .. })
        ));
        // A falsy flag is `apply_chat_template`'s default: left to the engine.
        assert!(matches!(
            render(
                &checkpoint,
                request(json!({"continue_final_message": false}))
            ),
            Err(TemplateError::NeedsRenderFallback { .. })
        ));
    }

    /// pydantic reads a numeric-string effort as a float, which a kwargs effort can
    /// still replace; an effort outside the tiers and [0, 0.99] is rejected.
    #[test]
    fn request_effort_follows_pydantic() {
        let formatter = formatter(&format!("{MARKERS}{{{{ reasoning_effort }}}}"));
        let request = |effort: Value, kwargs: Value| {
            json!({
                "messages": [{"role": "user", "content": "Hi"}],
                "reasoning_effort": effort,
                "chat_template_kwargs": kwargs,
            })
        };
        assert_eq!(
            render(&formatter, request(json!("0.5"), json!({}))).unwrap(),
            "0.5"
        );
        assert_eq!(
            render(
                &formatter,
                request(json!("0.5"), json!({"reasoning_effort": "low"}))
            )
            .unwrap(),
            "low"
        );
        for invalid in [json!("bogus"), json!(5), json!(true), json!("nan")] {
            assert!(
                matches!(
                    render(&formatter, request(invalid.clone(), json!({}))),
                    Err(TemplateError::Renderer { .. })
                ),
                "{invalid}"
            );
        }
    }
}
