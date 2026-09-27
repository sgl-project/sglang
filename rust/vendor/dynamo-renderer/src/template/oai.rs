// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Modified by SGLang: exposes the renderer's existing content-array classification through
// OAIPromptFormatter and PromptFormatter; other changes are comment-only provenance or spelling metadata.

use super::*;

use crate::{OAIChatLikeRequest, TextInput};
use minijinja::{context, value::Value};
use serde_json::json;
use std::result::Result::Ok;

/// Fix a tool schema that is missing `type`/`properties`. `pub` so consumers
/// can normalize their own `tools` value when implementing
/// [`crate::OAIChatLikeRequest::tools`].
pub fn may_be_fix_tool_schema(tools: serde_json::Value) -> Option<Value> {
    // No need to validate or enforce other schema checks as the basic Named function schema is already validated while creating the request.
    // Empty parameters is allowed by OpenAI at request level. Need to enforce it at template level.
    // Whenever parameters is empty, insert "type": "object" and "properties": {}
    let mut updated_tools = Vec::new();
    if let Some(arr) = tools.as_array() {
        for tool in arr {
            let mut tool = tool.clone();
            if let Some(function) = tool.get_mut("function") {
                // Backfill a missing/null `description`. It's optional in the
                // OpenAI tool schema, but some chat templates (e.g. gpt-oss
                // harmony) concatenate it unconditionally and fail on an
                // `undefined`/null value.
                if let Some(obj) = function.as_object_mut()
                    && !matches!(obj.get("description"), Some(serde_json::Value::String(_)))
                {
                    obj.insert(
                        "description".to_string(),
                        serde_json::Value::String(String::new()),
                    );
                }
            }
            if let Some(function) = tool.get_mut("function")
                && let Some(parameters) = function.get_mut("parameters")
            {
                // Only operate if parameters is an object
                if parameters.is_object() {
                    let mut needs_type = false;
                    let mut needs_properties = false;
                    let is_empty = parameters
                        .as_object()
                        .map(|o| o.is_empty())
                        .unwrap_or(false);

                    // If empty, we need to insert both
                    if is_empty {
                        needs_type = true;
                        needs_properties = true;
                    } else {
                        // If not empty, check if type/properties are missing
                        if let Some(obj) = parameters.as_object() {
                            if !obj.contains_key("type") {
                                needs_type = true;
                            }
                            if !obj.contains_key("properties") {
                                needs_properties = true;
                            }
                        }
                    }

                    if (needs_type || needs_properties)
                        && let Some(obj) = parameters.as_object_mut()
                    {
                        if needs_type {
                            obj.insert(
                                "type".to_string(),
                                serde_json::Value::String("object".to_string()),
                            );
                        }
                        if needs_properties {
                            obj.insert(
                                "properties".to_string(),
                                serde_json::Value::Object(Default::default()),
                            );
                        }
                    }
                }
            }
            updated_tools.push(tool);
        }
    }
    Some(Value::from_serialize(&updated_tools))
}

/// Default media type conversions for multimodal content.
/// Maps source types (e.g., "image_url") to target placeholder types (e.g., "image").
const DEFAULT_MEDIA_TYPE_CONVERSIONS: &[(&str, &str)] = &[
    ("image_url", "image"),
    ("video_url", "video"),
    ("audio_url", "audio"),
];

/// Convert media URL content parts to empty placeholder types.
fn convert_media_url_to_placeholder(
    content_array: &[serde_json::Value],
    conversions: &[(&str, &str)],
) -> Vec<serde_json::Value> {
    content_array
        .iter()
        .map(|part| {
            let part_type = part.get("type").and_then(|t| t.as_str()).unwrap_or("");

            if let Some((_, target_type)) = conversions.iter().find(|(src, _)| *src == part_type) {
                serde_json::json!({"type": target_type})
            } else {
                part.clone()
            }
        })
        .collect()
}

fn may_be_fix_msg_content(
    messages: serde_json::Value,
    preserve_arrays: bool,
    image_placeholder_template: Option<&str>,
) -> Value {
    // preserve_arrays=true: strings → arrays (multimodal)
    // preserve_arrays=false: text-only arrays → strings (standard)
    // image_placeholder_template: when `preserve_arrays=false` and the array
    // mixes text + image parts, this template (e.g. `<|image_{n}|>`) lets us
    // flatten by substituting image parts with model-family placeholders
    // instead of leaving the raw array for the template, which would crash
    // string-content templates like Phi-3-vision's `'+' message.content`.

    let Some(arr) = messages.as_array() else {
        return Value::from_serialize(&messages);
    };

    let updated_messages: Vec<_> = arr
        .iter()
        .map(|msg| {
            match msg.get("content") {
                // Case 1: String to Array (for multimodal templates)
                Some(serde_json::Value::String(text)) if preserve_arrays => {
                    let mut modified_msg = msg.clone();
                    if let Some(msg_object) = modified_msg.as_object_mut() {
                        let content_array = serde_json::json!([{
                            "type": "text",
                            "text": text
                        }]);
                        msg_object.insert("content".to_string(), content_array);
                    }
                    modified_msg
                }
                // Case 2: Array processing
                Some(serde_json::Value::Array(content_array)) => {
                    // First, convert any media URL parts to placeholders (e.g., image_url → image)
                    let content_array = convert_media_url_to_placeholder(
                        content_array,
                        DEFAULT_MEDIA_TYPE_CONVERSIONS,
                    );

                    // Check if it's text-only (after media URL conversion)
                    let is_text_only_array = !content_array.is_empty()
                        && content_array.iter().all(|part| {
                            part.get("type")
                                .and_then(|type_field| type_field.as_str())
                                .map(|type_str| type_str == "text")
                                .unwrap_or(false)
                        });

                    let mut modified_msg = msg.clone();
                    if let Some(msg_object) = modified_msg.as_object_mut() {
                        if is_text_only_array && !preserve_arrays {
                            // Flatten text-only arrays to string for standard templates
                            let text_parts: Vec<&str> = content_array
                                .iter()
                                .filter_map(|part| part.get("text")?.as_str())
                                .collect();
                            let concatenated_text = text_parts.join("\n");
                            msg_object.insert(
                                "content".to_string(),
                                serde_json::Value::String(concatenated_text),
                            );
                        } else if !preserve_arrays
                            && !content_array.is_empty()
                            && let Some(placeholder_tpl) = image_placeholder_template
                        {
                            // Mixed text+image array for a string-content
                            // template — flatten with model-family image
                            // placeholders inlined where the image parts were.
                            // An empty `placeholder_tpl` ("") drops the image
                            // parts entirely while keeping the text — used by
                            // pure pass-through / encoder-decoder templates
                            // (Nemotron-Parse) whose vision encoder consumes the
                            // image out-of-band, so no text token represents it.
                            // The `is_empty` guard preserves a literal `[]`
                            // content (matches pre-PR behavior); flattening
                            // an empty array to `""` would silently change
                            // what the template renders.
                            let flattened = flatten_mixed_content(&content_array, placeholder_tpl);
                            msg_object.insert(
                                "content".to_string(),
                                serde_json::Value::String(flattened),
                            );
                        } else {
                            // Keep as array (with media_url → media placeholder conversion applied)
                            msg_object.insert(
                                "content".to_string(),
                                serde_json::Value::Array(content_array),
                            );
                        }
                    }
                    modified_msg
                }
                _ => msg.clone(), // No conversion needed
            }
        })
        .collect();

    Value::from_serialize(&updated_messages)
}

/// Concatenate a mixed-content array (text parts + image placeholders) into a
/// single string. Text parts contribute their `text` field as-is; non-text
/// parts (image, video, audio after the URL→placeholder conversion in
/// `convert_media_url_to_placeholder`) emit the per-family placeholder with
/// `{n}` substituted by the 1-based index of the image in the message.
///
/// Used in `may_be_fix_msg_content` when `preserve_arrays=false` and the
/// template knows a placeholder convention — currently Phi-3-vision
/// (`<|image_{n}|>`), LLaVA-1.5 (`<image>`), and pure pass-through /
/// encoder-decoder templates (`""`, image emits nothing — Nemotron-Parse).
/// With an empty `placeholder_tpl` the non-text parts contribute no characters,
/// so the result is the concatenated text parts only.
///
/// **Caveat — non-text index slot:** `img_idx` increments for every non-text
/// part, not just images. The current supported families (Phi-3, LLaVA-1.5)
/// are image-only so there's no collision today, but a future image+video
/// family would silently consume an image-index slot for each video/audio
/// part and emit the image placeholder there. When adding a family that
/// mixes modalities in one message, either:
///   1. expand this function with per-modality placeholder strings, or
///   2. assert in `convert_media_url_to_placeholder` that only "image"
///      placeholders reach this path.
fn flatten_mixed_content(parts: &[serde_json::Value], placeholder_tpl: &str) -> String {
    let mut out = String::new();
    let mut img_idx: u32 = 1;
    for part in parts {
        let type_str = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if type_str == "text" {
            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                out.push_str(text);
            }
        } else if !type_str.is_empty() {
            let placeholder = placeholder_tpl.replace("{n}", &img_idx.to_string());
            out.push_str(&placeholder);
            img_idx += 1;
        }
    }
    out
}

fn normalize_tool_calls_arguments_in_messages(messages: &mut serde_json::Value) {
    // Deserialize `tool_calls[].function.arguments` from JSON strings to
    // objects/arrays before template rendering — avoids double encoding
    // and enables iteration. Skipped for templates whose own `is string`
    // branch wants the raw string verbatim (see render()).
    let Some(msgs) = messages.as_array_mut() else {
        return;
    };

    for msg in msgs.iter_mut() {
        if let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|v| v.as_array_mut()) {
            for tc in tool_calls {
                if let Some(function) = tc.get_mut("function").and_then(|v| v.as_object_mut())
                    && let Some(args) = function.get_mut("arguments")
                    && let Some(s) = args.as_str()
                    && let Ok(parsed) = serde_json::from_str(s)
                {
                    *args = parsed;
                }
            }
        }
    }
}

fn normalize_function_call_arguments_in_messages(messages: &mut serde_json::Value) {
    // Legacy (deprecated) OpenAI `function_call.arguments` path. Kept separate
    // from `tool_calls` normalization so the per-template `arguments is string`
    // opt-out — which only refers to `tool_call.arguments` inside the
    // tool_calls loop — does not accidentally suppress this path.
    let Some(msgs) = messages.as_array_mut() else {
        return;
    };

    for msg in msgs.iter_mut() {
        if let Some(function_call) = msg.get_mut("function_call").and_then(|v| v.as_object_mut())
            && let Some(args) = function_call.get_mut("arguments")
            && let Some(s) = args.as_str()
            && let Ok(parsed) = serde_json::from_str(s)
        {
            *args = parsed;
        }
    }
}

/// Inject `reasoning_content` back into the `content` field as `<think>` blocks.
///
/// Chat templates only reference `{{ message.content }}` — they don't know about
/// `reasoning_content`. Without this injection, the model's prior chain-of-thought
/// is silently dropped across turns.
///
/// Uses `<think>`/`</think>` delimiters — the same tags that reasoning models emit
/// and that the reasoning parser strips on output. Reasoning is prepended to content
/// to match the original generation order (`<think>...</think> response`).
///
/// Segments are concatenated rather than interleaved with tool_calls because Jinja
/// templates render `tool_calls` separately from `content`. The model still sees
/// all reasoning text before the template-rendered tool call block.
fn inject_reasoning_content_into_messages(messages: &mut serde_json::Value) {
    let Some(msgs) = messages.as_array_mut() else {
        return;
    };

    for msg in msgs.iter_mut() {
        if msg.get("role").and_then(|r| r.as_str()) != Some("assistant") {
            continue;
        }

        let reasoning = match msg.get("reasoning_content") {
            Some(serde_json::Value::String(s)) if !s.is_empty() => {
                format!("<think>{}</think>", s)
            }
            Some(serde_json::Value::Array(segments)) => {
                let mut result = String::new();
                for seg in segments {
                    if let Some(s) = seg.as_str()
                        && !s.is_empty()
                    {
                        result.push_str("<think>");
                        result.push_str(s);
                        result.push_str("</think>");
                    }
                }
                if result.is_empty() {
                    continue;
                }
                result
            }
            _ => continue,
        };

        match msg.get("content") {
            // Content is a string or null — prepend reasoning as text
            Some(serde_json::Value::String(s)) if !s.is_empty() => {
                msg["content"] = serde_json::Value::String(format!("{}{}", reasoning, s));
            }
            None | Some(serde_json::Value::Null) | Some(serde_json::Value::String(_)) => {
                msg["content"] = serde_json::Value::String(reasoning);
            }
            // Content is an array (multimodal) — prepend as a text part
            Some(serde_json::Value::Array(_)) => {
                let think_part = serde_json::json!({
                    "type": "text",
                    "text": reasoning
                });
                if let Some(arr) = msg.get_mut("content").and_then(|v| v.as_array_mut()) {
                    arr.insert(0, think_part);
                }
            }
            // Other types (number, bool, object) — skip, don't corrupt
            _ => continue,
        }

        // Remove so the template doesn't see both the injected <think> in content
        // and the original reasoning_content field.
        if let Some(obj) = msg.as_object_mut() {
            obj.remove("reasoning_content");
        }
    }
}

/// Default [`OAIChatLikeRequest`] impl for the bare `dynamo-protocols` chat
/// request. Lets any consumer (e.g. a standalone OpenAI frontend over an
/// engine) render HF chat templates directly from the wire type, without
/// defining their own wrapper. Consumers with extra fields (Dynamo's
/// `NvCreateChatCompletionRequest`) provide their own impl.
impl OAIChatLikeRequest for dynamo_protocols::types::CreateChatCompletionRequest {
    fn model(&self) -> String {
        self.model.clone()
    }

    fn messages(&self) -> Value {
        let messages_json = serde_json::to_value(&self.messages).unwrap();
        Value::from_serialize(&messages_json)
    }

    fn typed_messages(&self) -> Option<&[dynamo_protocols::types::ChatCompletionRequestMessage]> {
        Some(self.messages.as_slice())
    }

    fn tools(&self) -> Option<Value> {
        if self.tools.is_none() {
            None
        } else {
            Some(may_be_fix_tool_schema(
                serde_json::to_value(&self.tools).unwrap(),
            )?)
        }
    }

    fn tool_choice(&self) -> Option<Value> {
        if self.tool_choice.is_none() {
            None
        } else {
            Some(Value::from_serialize(&self.tool_choice))
        }
    }

    fn response_format(&self) -> Option<Value> {
        self.response_format.as_ref().map(Value::from_serialize)
    }

    fn reasoning_effort(&self) -> Option<Value> {
        self.reasoning_effort.as_ref().map(Value::from_serialize)
    }

    fn should_add_generation_prompt(&self) -> bool {
        // Using vLLM default behavior
        true
    }

    fn extract_text(&self) -> Option<TextInput> {
        Some(TextInput::Single(String::new()))
    }

    fn mm_processor_kwargs(&self) -> Option<&serde_json::Value> {
        self.mm_processor_kwargs.as_ref()
    }
}

/// Joins two message contents without flattening to text: a part array is
/// spliced part for part, so image parts survive a merge. A string target is
/// promoted to an array when the source carries parts.
fn merge_message_content(
    target: serde_json::Value,
    source: serde_json::Value,
) -> serde_json::Value {
    use serde_json::Value;
    let text_part = |text: String| json!({"type": "text", "text": text});
    match (target, source) {
        (Value::String(mut target), Value::String(source)) => {
            if !target.is_empty() && !source.is_empty() {
                target.push_str("\n\n");
            }
            target.push_str(&source);
            Value::String(target)
        }
        (Value::Array(mut target), Value::Array(source)) => {
            target.extend(source);
            Value::Array(target)
        }
        (Value::Array(mut target), Value::String(source)) => {
            if !source.is_empty() {
                target.push(text_part(source));
            }
            Value::Array(target)
        }
        (Value::String(target), Value::Array(source)) => {
            let mut parts = Vec::with_capacity(source.len() + 1);
            if !target.is_empty() {
                parts.push(text_part(target));
            }
            parts.extend(source);
            Value::Array(parts)
        }
        (Value::Null, source) => source,
        // Content is a string or a part array in every shape the schema allows.
        (target, _) => target,
    }
}

/// Merges `source` into `target`, leaving every other field on `target` intact.
fn append_message_content(target: &mut serde_json::Value, source: serde_json::Value) {
    let Some(target) = target.as_object_mut() else {
        return;
    };
    let merged = merge_message_content(
        target.remove("content").unwrap_or(serde_json::Value::Null),
        source,
    );
    target.insert("content".to_string(), merged);
}

fn take_message_content(message: &mut serde_json::Value) -> serde_json::Value {
    message
        .get_mut("content")
        .map(serde_json::Value::take)
        .unwrap_or(serde_json::Value::Null)
}

/// Rewrites an agent-client message stream into a shape strict chat templates
/// accept. Each rewrite is gated on the restriction the load-time probe actually
/// found, so a template is never reshaped for a rule it does not enforce.
fn normalize_system_messages(messages: &mut serde_json::Value, rules: SystemNormalization) {
    let serde_json::Value::Array(list) = messages else {
        return;
    };
    let role_is =
        |m: &serde_json::Value, r: &str| m.get("role").and_then(|v| v.as_str()) == Some(r);

    if rules.demote_nonleading_system {
        // A template that rejects a system turn away from index 0 rejects a
        // leading run of them too, so the run collapses into the first.
        let leading = list.iter().take_while(|m| role_is(m, "system")).count();
        if leading > 1 {
            for mut trailing in list.drain(1..leading).collect::<Vec<_>>() {
                let content = take_message_content(&mut trailing);
                append_message_content(&mut list[0], content);
            }
        }

        // Demoted in place, not folded to the front: a mid-conversation reminder
        // that toggles would otherwise invalidate the whole KV prefix each turn.
        let leading = list.iter().take_while(|m| role_is(m, "system")).count();
        for m in list.iter_mut().skip(leading) {
            if role_is(m, "system")
                && let Some(m) = m.as_object_mut()
            {
                m.insert("role".to_string(), json!("user"));
            }
        }
    }

    if rules.coalesce_consecutive_users {
        let mut coalesced: Vec<serde_json::Value> = Vec::with_capacity(list.len());
        for mut m in list.drain(..) {
            if role_is(&m, "user") && coalesced.last().is_some_and(|p| role_is(p, "user")) {
                let content = take_message_content(&mut m);
                append_message_content(coalesced.last_mut().unwrap(), content);
            } else {
                coalesced.push(m);
            }
        }
        *list = coalesced;
    }
}

impl OAIPromptFormatter for HfTokenizerConfigJsonFormatter {
    fn supports_add_generation_prompt(&self) -> bool {
        self.supports_add_generation_prompt
    }

    fn requires_content_arrays(&self) -> bool {
        self.requires_content_arrays
    }

    fn render(&self, req: &dyn OAIChatLikeRequest) -> Result<String> {
        let mixins = Value::from_dyn_object(self.mixins.clone());

        let tools = req.tools();
        // Strip tools when tool_choice is "none" and the flag is enabled, so the model
        // doesn't see tool definitions and generate raw XML tool calls in its response.
        let tools = if self.exclude_tools_when_tool_choice_none {
            match req.tool_choice() {
                Some(ref tc) if tc.as_str() == Some("none") => None,
                _ => tools,
            }
        } else {
            tools
        };
        // has_tools should be true if tools is a non-empty array
        let has_tools = tools.as_ref().and_then(|v| v.len()).is_some_and(|l| l > 0);
        let add_generation_prompt = req.should_add_generation_prompt();

        tracing::trace!(
            "Rendering prompt with tools: {:?}, add_generation_prompt: {}",
            has_tools,
            add_generation_prompt
        );

        // Pick the concrete template before applying any template-specific
        // message rewrites or field normalization.
        let (
            template_name,
            template_handles_tool_calls_args_string,
            template_handles_reasoning,
            system_normalization,
        ) = if has_tools {
            (
                "tool_use",
                self.tool_use_template_handles_tool_calls_arguments_string,
                self.tool_use_template_handles_reasoning,
                self.tool_use_system_normalization,
            )
        } else {
            (
                "default",
                self.default_template_handles_tool_calls_arguments_string,
                self.default_template_handles_reasoning,
                self.default_system_normalization,
            )
        };

        let messages_canonical = req.messages();
        let mut messages_for_template: serde_json::Value =
            serde_json::to_value(&messages_canonical).unwrap();

        if system_normalization.is_required() {
            normalize_system_messages(&mut messages_for_template, system_normalization);
        }

        messages_for_template = serde_json::to_value(may_be_fix_msg_content(
            messages_for_template,
            self.requires_content_arrays,
            self.image_placeholder_template,
        ))
        .unwrap();

        // Pre-parse JSON-string `arguments` into objects — but only for templates
        // that unconditionally `| tojson` them. Templates that branch on
        // `tool_call.arguments is string` (Qwen3, Hermes) want the raw string
        // verbatim so the rendered bytes match what the model emitted on the
        // prior turn. Re-serializing through minijinja's compact `tojson` here
        // breaks append-only prefix matching across multi-step tool use.
        if !template_handles_tool_calls_args_string {
            normalize_tool_calls_arguments_in_messages(&mut messages_for_template);
        }
        // Legacy `function_call.arguments` is always normalized — the
        // `arguments is string` opt-out only covers the modern `tool_calls`
        // branch.
        normalize_function_call_arguments_in_messages(&mut messages_for_template);

        // Inject reasoning_content as <think> blocks into content — but only if
        // the template doesn't handle it natively. Templates like Nemotron and
        // Qwen3 reference reasoning_content directly in their Jinja logic; injecting
        // would produce duplicate <think> blocks.
        if !template_handles_reasoning {
            inject_reasoning_content_into_messages(&mut messages_for_template);
        }

        let ctx = context! {
            messages => messages_for_template,
            tools => tools,
            bos_token => self.config.bos_tok(),
            eos_token => self.config.eos_tok(),
            unk_token => self.config.unk_tok(),
            add_generation_prompt => add_generation_prompt,
            ..mixins
        };

        // Merge any additional args into the context last so they take precedence
        let ctx = if let Some(args) = req.chat_template_args() {
            let extra = Value::from_serialize(args);
            context! { ..ctx, ..extra }
        } else {
            ctx
        };

        let tmpl: minijinja::Template<'_, '_> = self.env.get_template(template_name)?;
        Ok(tmpl.render(&ctx)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::ChatCompletionRequestMessage as Msg;
    // The crate's renderer tests exercise the bare-protocol request type via the
    // default `OAIChatLikeRequest` impl above; Dynamo's `Nv*` wrapper lives in lib/llm.
    use dynamo_protocols::types::CreateChatCompletionRequest as NvCreateChatCompletionRequest;
    use minijinja::{Environment, context};

    // --- adaptive system-message normalization (#11762) --------------------

    use super::super::tokcfg::ChatTemplate as SysChatTemplate;
    use super::super::{
        ContextMixins as SysMixins, HfTokenizerConfigJsonFormatter as SysFormatter,
    };

    fn formatter_for(template: &str) -> SysFormatter {
        // Dummy tokens: real templates that append eos/bos won't render without them.
        let ct: SysChatTemplate = serde_json::from_value(json!({
            "chat_template": template,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        }))
        .unwrap();
        SysFormatter::new(ct, SysMixins::new(&[])).unwrap()
    }

    fn formatter_for_templates(default: &str, tool_use: &str) -> SysFormatter {
        let ct: SysChatTemplate = serde_json::from_value(json!({
            "chat_template": [
                {"default": default},
                {"tool_use": tool_use},
            ],
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        }))
        .unwrap();
        SysFormatter::new(ct, SysMixins::new(&[])).unwrap()
    }

    fn try_formatter_for(template: &str) -> Option<SysFormatter> {
        let ct: SysChatTemplate = serde_json::from_value(json!({
            "chat_template": template,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        }))
        .ok()?;
        SysFormatter::new(ct, SysMixins::new(&[])).ok()
    }

    fn render_shape(f: &SysFormatter, messages: serde_json::Value) -> Result<String> {
        let req: NvCreateChatCompletionRequest =
            serde_json::from_value(json!({ "model": "test", "messages": messages })).unwrap();
        f.render(&req)
    }

    fn render_shape_with_tools(f: &SysFormatter, messages: serde_json::Value) -> Result<String> {
        let req: NvCreateChatCompletionRequest = serde_json::from_value(json!({
            "model": "test",
            "messages": messages,
            "tools": [{
                "type": "function",
                "function": {"name": "noop", "parameters": {}}
            }]
        }))
        .unwrap();
        f.render(&req)
    }

    const PERMISSIVE_TMPL: &str = concat!(
        "{%- for m in messages -%}",
        "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n",
        "{%- endfor -%}"
    );
    // Rejects a non-leading system (Qwen3.5 shape); accepts consecutive users.
    const STRICT_LEADING_TMPL: &str = concat!(
        "{%- for m in messages -%}",
        "{%- if m.role == 'system' and not loop.first -%}",
        "{{ raise_exception('System message must be at the beginning.') }}",
        "{%- endif -%}",
        "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n",
        "{%- endfor -%}"
    );
    // Rejects consecutive user turns only; accepts a non-leading system.
    const ALTERNATION_TMPL: &str = concat!(
        "{%- set ns = namespace(prev='') -%}",
        "{%- for m in messages -%}",
        "{%- if m.role == 'user' and ns.prev == 'user' -%}",
        "{{ raise_exception('Conversation roles must alternate.') }}",
        "{%- endif -%}",
        "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n",
        "{%- set ns.prev = m.role -%}",
        "{%- endfor -%}"
    );
    // Rejects both restrictions (Gemma-3 / Mistral shape).
    const STRICT_BOTH_TMPL: &str = concat!(
        "{%- set ns = namespace(prev='') -%}",
        "{%- for m in messages -%}",
        "{%- if m.role == 'system' and not loop.first -%}",
        "{{ raise_exception('System message must be at the beginning.') }}",
        "{%- endif -%}",
        "{%- if m.role == 'user' and ns.prev == 'user' -%}",
        "{{ raise_exception('Conversation roles must alternate.') }}",
        "{%- endif -%}",
        "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n",
        "{%- set ns.prev = m.role -%}",
        "{%- endfor -%}"
    );
    // Mirrors templates that only enforce role constraints when a particular
    // tools shape is present.
    const DEFAULT_NONE_GATED_TMPL: &str = concat!(
        "{%- set strict = tools is not none -%}",
        "{%- for m in messages -%}",
        "{%- if strict and m.role == 'system' and not loop.first -%}",
        "{{ raise_exception('System message must be at the beginning.') }}",
        "{%- endif -%}",
        "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n",
        "{%- endfor -%}"
    );
    const TOOL_NONEMPTY_GATED_TMPL: &str = concat!(
        "{%- set strict = tools|length > 0 -%}",
        "{%- for m in messages -%}",
        "{%- if strict and m.role == 'system' and not loop.first -%}",
        "{{ raise_exception('System message must be at the beginning.') }}",
        "{%- endif -%}",
        "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n",
        "{%- endfor -%}"
    );
    // Ignores string content and only renders text parts from content arrays.
    const STRICT_ARRAY_TMPL: &str = concat!(
        "{%- for m in messages -%}",
        "{%- if m.role == 'system' and not loop.first -%}",
        "{{ raise_exception('System message must be at the beginning.') }}",
        "{%- endif -%}",
        "<|im_start|>{{ m.role }}\n",
        "{%- if m.content is not string -%}",
        "{%- for part in m.content -%}{{ part.text }}{%- endfor -%}",
        "{%- endif -%}",
        "<|im_end|>\n",
        "{%- endfor -%}"
    );

    // Claude Code first-turn shape: top-level system + a mid-array system.
    fn claude_shape() -> serde_json::Value {
        json!([
            {"role": "system", "content": "You are Claude Code."},
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "mid-conversation reminder"},
        ])
    }

    fn all_restrictions() -> SystemNormalization {
        SystemNormalization {
            demote_nonleading_system: true,
            coalesce_consecutive_users: true,
        }
    }

    #[test]
    fn permissive_template_is_not_flagged_and_renders_untouched() {
        let f = formatter_for(PERMISSIVE_TMPL);
        assert!(!f.default_system_normalization.is_required());
        assert!(!f.tool_use_system_normalization.is_required());
        let out = render_shape(&f, claude_shape()).unwrap();
        assert!(out.contains("<|im_start|>system\nmid-conversation reminder<|im_end|>"));
    }

    #[test]
    fn strict_leading_template_demotes_mid_system_but_keeps_user_turns_apart() {
        let f = formatter_for(STRICT_LEADING_TMPL);
        assert!(f.default_system_normalization.demote_nonleading_system);
        // The template accepts consecutive users, so demotion must not merge them.
        assert!(!f.default_system_normalization.coalesce_consecutive_users);

        // This shape returned a 500 before the probe existed.
        let out = render_shape(&f, claude_shape()).unwrap();
        assert_eq!(out.matches("<|im_start|>system").count(), 1);
        assert!(out.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(out.contains("<|im_start|>user\nmid-conversation reminder<|im_end|>"));
    }

    #[test]
    fn alternation_template_coalesces_users_but_keeps_mid_system() {
        let f = formatter_for(ALTERNATION_TMPL);
        assert!(f.default_system_normalization.coalesce_consecutive_users);
        // The template accepts a non-leading system, so it stays a system turn.
        assert!(!f.default_system_normalization.demote_nonleading_system);

        let out = render_shape(&f, claude_shape()).unwrap();
        assert!(out.contains("<|im_start|>system\nmid-conversation reminder<|im_end|>"));

        let out = render_shape(
            &f,
            json!([
                {"role": "system", "content": "s"},
                {"role": "user", "content": "hello"},
                {"role": "user", "content": "again"},
            ]),
        )
        .unwrap();
        assert_eq!(out.matches("<|im_start|>user").count(), 1);
        assert!(out.contains("<|im_start|>user\nhello\n\nagain<|im_end|>"));
    }

    #[test]
    fn strict_both_template_demotes_then_coalesces() {
        let f = formatter_for(STRICT_BOTH_TMPL);
        assert!(f.default_system_normalization.demote_nonleading_system);
        assert!(f.default_system_normalization.coalesce_consecutive_users);

        let out = render_shape(&f, claude_shape()).unwrap();
        assert_eq!(out.matches("<|im_start|>system").count(), 1);
        assert!(out.contains("<|im_start|>user\nhello\n\nmid-conversation reminder<|im_end|>"));
    }

    #[test]
    fn system_normalization_flag_is_selected_per_template() {
        let f = formatter_for_templates(PERMISSIVE_TMPL, STRICT_LEADING_TMPL);
        assert!(!f.default_system_normalization.is_required());
        assert!(f.tool_use_system_normalization.is_required());

        let no_tools = render_shape(&f, claude_shape()).unwrap();
        assert!(no_tools.contains("<|im_start|>system\nmid-conversation reminder<|im_end|>"));
        let with_tools = render_shape_with_tools(&f, claude_shape()).unwrap();
        assert_eq!(with_tools.matches("<|im_start|>system").count(), 1);
        assert!(with_tools.contains("<|im_start|>user\nmid-conversation reminder<|im_end|>"));

        let f = formatter_for_templates(STRICT_LEADING_TMPL, PERMISSIVE_TMPL);
        assert!(f.default_system_normalization.is_required());
        assert!(!f.tool_use_system_normalization.is_required());
        let with_tools = render_shape_with_tools(&f, claude_shape()).unwrap();
        assert!(with_tools.contains("<|im_start|>system\nmid-conversation reminder<|im_end|>"));
    }

    #[test]
    fn system_normalization_probe_uses_runtime_tools_shape() {
        let f = formatter_for_templates(DEFAULT_NONE_GATED_TMPL, TOOL_NONEMPTY_GATED_TMPL);
        assert!(!f.default_system_normalization.is_required());
        assert!(f.tool_use_system_normalization.is_required());

        let no_tools = render_shape(&f, claude_shape()).unwrap();
        assert!(no_tools.contains("<|im_start|>system\nmid-conversation reminder<|im_end|>"));

        let with_tools = render_shape_with_tools(&f, claude_shape()).unwrap();
        assert_eq!(with_tools.matches("<|im_start|>system").count(), 1);
        assert!(with_tools.contains("<|im_start|>user\nmid-conversation reminder<|im_end|>"));
    }

    #[test]
    fn system_normalization_precedes_required_content_array_conversion() {
        let f = formatter_for(STRICT_ARRAY_TMPL);
        assert!(f.requires_content_arrays);
        assert!(f.default_system_normalization.demote_nonleading_system);

        let out = render_shape(
            &f,
            json!([
                {"role": "system", "content": "A"},
                {"role": "system", "content": "B"},
                {"role": "user", "content": "hello"},
            ]),
        )
        .unwrap();
        assert!(out.contains("A\n\nB"));
    }

    #[test]
    fn normalize_preserves_multimodal_user_content_and_fields() {
        let mut m = json!([
            {
                "role": "user",
                "name": "kept",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image"},
                ],
            },
            {"role": "system", "content": "remember"},
        ]);
        normalize_system_messages(&mut m, all_restrictions());
        assert_eq!(
            m,
            json!([{
                "role": "user",
                "name": "kept",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image"},
                    {"type": "text", "text": "remember"},
                ],
            }])
        );
    }

    /// The mirror of the case above: the parts belong to the turn being merged
    /// away, not the one being merged into.
    #[test]
    fn coalesce_preserves_multimodal_content_of_the_merged_turn() {
        let mut m = json!([
            {"role": "user", "content": "look"},
            {"role": "user", "content": [
                {"type": "text", "text": "at this"},
                {"type": "image_url", "image_url": {"url": "http://img"}},
            ]},
        ]);
        normalize_system_messages(&mut m, all_restrictions());
        assert_eq!(
            m,
            json!([{
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "text", "text": "at this"},
                    {"type": "image_url", "image_url": {"url": "http://img"}},
                ],
            }])
        );
    }

    #[test]
    fn normalize_merges_leading_run_and_coalesces() {
        let mut m = json!([
            {"role": "system", "content": "A"},
            {"role": "system", "content": "B"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "reminder"},
        ]);
        normalize_system_messages(&mut m, all_restrictions());
        assert_eq!(
            m,
            json!([
                {"role": "system", "content": "A\n\nB"},
                {"role": "user", "content": "hi\n\nreminder"},
            ])
        );
    }

    /// Each restriction drives only its own rewrite, so a template is never
    /// reshaped for a rule it does not enforce.
    #[test]
    fn each_restriction_applies_only_its_own_rewrite() {
        let shape = json!([
            {"role": "system", "content": "A"},
            {"role": "system", "content": "B"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "reminder"},
        ]);

        let mut demote_only = shape.clone();
        normalize_system_messages(
            &mut demote_only,
            SystemNormalization {
                demote_nonleading_system: true,
                coalesce_consecutive_users: false,
            },
        );
        assert_eq!(
            demote_only,
            json!([
                {"role": "system", "content": "A\n\nB"},
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "reminder"},
            ])
        );

        let mut coalesce_only = shape.clone();
        normalize_system_messages(
            &mut coalesce_only,
            SystemNormalization {
                demote_nonleading_system: false,
                coalesce_consecutive_users: true,
            },
        );
        assert_eq!(coalesce_only, shape);
    }

    #[test]
    fn normalize_preserves_array_system_content() {
        let mut m = json!([
            {"role": "user", "content": "hi"},
            {"role": "system", "content": [{"type": "text", "text": "one"},
                                           {"type": "text", "text": "two"}]},
        ]);
        normalize_system_messages(&mut m, all_restrictions());
        assert_eq!(
            m,
            json!([{"role": "user", "content": [
                {"type": "text", "text": "hi"},
                {"type": "text", "text": "one"},
                {"type": "text", "text": "two"},
            ]}])
        );
    }

    /// Renders every agent-client shape through every template in a corpus of
    /// real models. A failure means the probe missed a strict template, or the
    /// normalization it triggered was not enough to satisfy one.
    ///
    /// TEMPLATE_CORPUS points at a dir of `<name>.jinja` files + `manifest.json`
    /// mapping each to `{model}` (see experiments/system-probe):
    ///   TEMPLATE_CORPUS=... cargo test -p dynamo-renderer \
    ///     adaptive_system_corpus_audit -- --ignored --nocapture
    #[test]
    #[ignore]
    fn adaptive_system_corpus_audit() {
        let dir =
            std::env::var("TEMPLATE_CORPUS").expect("set TEMPLATE_CORPUS to the templates dir");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json")).unwrap())
                .unwrap();

        // Tool-call shapes are omitted: they need a per-family `tools` payload,
        // and the hermetic tests already cover them.
        let sys = |c: &str| json!({"role": "system", "content": c});
        let usr = |c: &str| json!({"role": "user", "content": c});
        let asst = |c: &str| json!({"role": "assistant", "content": c});
        let shapes: Vec<(&str, serde_json::Value)> = vec![
            ("turn1", json!([sys("s"), usr("u"), sys("mid")])),
            (
                "multiturn",
                json!([sys("s"), usr("u"), sys("mid"), asst("a"), usr("u2")]),
            ),
            (
                "mid_after_asst",
                json!([sys("s"), usr("u"), asst("a"), sys("mid"), usr("u2")]),
            ),
            ("double_leading", json!([sys("s0"), sys("s1"), usr("u")])),
            ("consec_user", json!([sys("s"), usr("u0"), usr("u1")])),
            (
                "tail_reminder",
                json!([
                    sys("s"),
                    usr("u"),
                    asst("a"),
                    usr("u2"),
                    sys("mid"),
                    usr("u3")
                ]),
            ),
            ("leading_only_baseline", json!([sys("s"), usr("u")])),
        ];

        let mut total = 0usize;
        let mut flagged = 0usize;
        let mut demote_only = 0usize;
        let mut coalesce = 0usize;
        let mut failures: Vec<String> = Vec::new();
        for (file, meta) in manifest.as_object().unwrap() {
            let tmpl = std::fs::read_to_string(format!("{dir}/{file}.jinja")).unwrap();
            let model = meta["model"].as_str().unwrap_or(file);
            // Some real templates use custom tags minijinja can't compile,
            // which says nothing about system normalization.
            let f = match try_formatter_for(&tmpl) {
                Some(f) => f,
                None => {
                    eprintln!("[skip-compile] {model}");
                    continue;
                }
            };
            // Vision templates need image inputs, so they can't render text-only
            // here. The probe's baseline guard leaves them unflagged anyway.
            if render_shape(&f, json!([sys("s"), usr("u")])).is_err() {
                eprintln!("[skip-baseline] {model}");
                continue;
            }
            total += 1;
            let rules = f.default_system_normalization;
            let flag = rules.is_required();
            if flag {
                flagged += 1;
            }
            if rules.demote_nonleading_system {
                demote_only += usize::from(!rules.coalesce_consecutive_users);
            }
            if rules.coalesce_consecutive_users {
                coalesce += 1;
            }
            for (name, shape) in &shapes {
                if render_shape(&f, shape.clone()).is_err() {
                    failures.push(format!("{model} | shape={name} | flag={flag}"));
                }
            }
            if flag {
                eprintln!(
                    "[ok] demote={} coalesce={} {model}",
                    rules.demote_nonleading_system, rules.coalesce_consecutive_users
                );
            }
        }
        eprintln!(
            "\naudited {total} templates ({flagged} flagged: {demote_only} demote-only, \
             {coalesce} coalescing); {} shape failures",
            failures.len()
        );
        for f in &failures {
            eprintln!("  FAIL {f}");
        }
        assert!(
            failures.is_empty(),
            "{} template/shape combinations did not render (probe insufficient or normalization insufficient)",
            failures.len()
        );
    }

    /// End-to-end guard for the minijinja stack-overflow fix, exercised through
    /// Dynamo's real chat-template render path. A template that accumulates
    /// messages via `ns.items = ns.items + [m]` and then takes `|length`
    /// previously overflowed the native stack for long conversations (~1500+
    /// turns on a worker thread), core-dumping the frontend. Runs on a 2 MiB
    /// stack — the size of a Dynamo tokio worker thread — so a regression aborts
    /// deterministically instead of depending on the platform default.
    #[test]
    fn test_render_long_conversation_does_not_overflow_stack() {
        let handle = std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(|| {
                let template_string = concat!(
                    "{%- set ns = namespace(items=[]) -%}",
                    "{%- for m in messages -%}",
                    "{%- set ns.items = ns.items + [m] -%}",
                    "{%- endfor -%}",
                    "COUNT={{ ns.items | length }}"
                );
                let chat_template: ChatTemplate =
                    serde_json::from_value(serde_json::json!({ "chat_template": template_string }))
                        .unwrap();
                let formatter =
                    HfTokenizerConfigJsonFormatter::new(chat_template, ContextMixins::new(&[]))
                        .unwrap();

                let n = 3000;
                let messages: Vec<serde_json::Value> = (0..n)
                    .map(|i| serde_json::json!({"role": "user", "content": format!("turn {i}")}))
                    .collect();
                let request: NvCreateChatCompletionRequest =
                    serde_json::from_value(serde_json::json!({
                        "model": "test",
                        "messages": messages,
                    }))
                    .unwrap();

                // The crash path: `|length` -> minijinja `Value::len()`.
                let rendered = formatter.render(&request).unwrap();
                assert_eq!(rendered.trim(), format!("COUNT={n}"));
            })
            .unwrap();
        handle.join().unwrap();
    }

    /// Dev utility (ignored by default): dump the prompt Dynamo's renderer
    /// produces for a tool-calling chat request, so it can be diffed against
    /// vLLM's `openai_harmony` rendering — to see whether the gpt-oss Jinja
    /// `chat_template` actually emits the harmony "tool calls go to the
    /// commentary channel" guidance + `functions` namespace.
    ///
    /// Point GPTOSS_CHAT_TEMPLATE at the model's tokenizer_config.json (its
    /// `chat_template` field is extracted) OR a raw chat_template.jinja file:
    ///   GPTOSS_CHAT_TEMPLATE=/path/openai-gpt-oss-120b/tokenizer_config.json \
    ///     cargo test -p dynamo-renderer dump_gptoss_tool_prompt -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dump_gptoss_tool_prompt() {
        use super::tokcfg::ChatTemplate;
        use super::{ContextMixins, HfTokenizerConfigJsonFormatter};

        let path = std::env::var("GPTOSS_CHAT_TEMPLATE").expect(
            "set GPTOSS_CHAT_TEMPLATE to the tokenizer_config.json, chat_template.jinja, or model dir path",
        );
        let input_path = std::path::Path::new(&path);
        let file_path = if input_path.is_dir() {
            // Prefer tokenizer_config.json from model dir if provided
            input_path.join("tokenizer_config.json")
        } else {
            input_path.to_path_buf()
        };
        let raw = std::fs::read_to_string(&file_path).expect("read chat template file");
        // Resolve the actual Jinja template. gpt-oss ships its chat template in a
        // separate `chat_template.jinja` file, NOT inside tokenizer_config.json,
        // so:
        //   * if the file is JSON with a `chat_template` field, use it;
        //   * otherwise, if a sibling `chat_template.jinja` exists, read that;
        //   * otherwise treat the file itself as the template.
        let template_string: String = match serde_json::from_str::<serde_json::Value>(&raw) {
            Ok(v) if v.get("chat_template").is_some() => v["chat_template"]
                .as_str()
                .expect("chat_template field must be a string")
                .to_string(),
            _ => {
                let sibling = std::path::Path::new(&path)
                    .parent()
                    .map(|d| d.join("chat_template.jinja"));
                match sibling {
                    Some(p) if p.exists() => {
                        eprintln!(
                            "[info] {path} had no chat_template field; using {}",
                            p.display()
                        );
                        std::fs::read_to_string(&p).expect("read sibling chat_template.jinja")
                    }
                    _ => raw,
                }
            }
        };

        // Guard against silently echoing a non-template (e.g. a tokenizer_config.json
        // with no chat_template and no sibling .jinja).
        assert!(
            template_string.contains("{%") || template_string.contains("{{"),
            "resolved template has no Jinja tags — GPTOSS_CHAT_TEMPLATE ({path}) is probably \
             tokenizer_config.json with no chat_template field and no sibling chat_template.jinja. \
             Point it at the chat_template.jinja file."
        );

        let chat_template: ChatTemplate =
            serde_json::from_value(serde_json::json!({ "chat_template": template_string }))
                .unwrap();

        let formatter =
            HfTokenizerConfigJsonFormatter::new(chat_template, ContextMixins::new(&[])).unwrap();

        // Declare tools — the tool-channel guidance only renders when tools are present.
        let request: NvCreateChatCompletionRequest = serde_json::from_str(
            r#"{
              "model": "openai/gpt-oss-120b",
              "messages": [{"role":"user","content":"Search the repo for the string \"countHook\"."}],
              "tools": [
                {"type":"function","function":{"name":"grep","description":"search files","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"}},"required":["pattern"]}}},
                {"type":"function","function":{"name":"read","description":"read a file","parameters":{"type":"object","properties":{"filePath":{"type":"string"}},"required":["filePath"]}}}
              ]
            }"#,
        )
        .unwrap();

        let rendered = formatter.render(&request).unwrap();
        eprintln!("================ RENDERED gpt-oss PROMPT (tools declared) ================");
        eprintln!("{rendered}");
        eprintln!("================ END RENDERED PROMPT ================");
        eprintln!("[diagnostics] does the rendered prompt contain…");
        for needle in [
            "commentary",
            "Calls to these tools",
            "functions",
            "# Tools",
            "<|channel|>",
            "constrain",
            "analysis",
        ] {
            eprintln!(
                "  {:>22}: {}",
                format!("{needle:?}"),
                rendered.contains(needle)
            );
        }
    }

    /// Tests that media URL content parts are converted to empty placeholders.
    #[test]
    fn test_convert_media_url_to_placeholder_single_type() {
        let content_array = vec![
            serde_json::json!({"type": "text", "text": "Check this image:"}),
            serde_json::json!({"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}),
            serde_json::json!({"type": "text", "text": "What do you see?"}),
        ];

        let conversions = &[("image_url", "image")];
        let result = convert_media_url_to_placeholder(&content_array, conversions);

        assert_eq!(result.len(), 3);
        // Text parts should be unchanged
        assert_eq!(result[0]["type"], "text");
        assert_eq!(result[0]["text"], "Check this image:");
        // image_url should be converted to image placeholder
        assert_eq!(result[1]["type"], "image");
        assert!(result[1].get("image_url").is_none());
        // Text parts should be unchanged
        assert_eq!(result[2]["type"], "text");
        assert_eq!(result[2]["text"], "What do you see?");
    }

    /// Tests that multiple media URL parts of the same type are all converted.
    #[test]
    fn test_convert_media_url_to_placeholder_multiple_same_type() {
        let content_array = vec![
            serde_json::json!({"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}}),
            serde_json::json!({"type": "text", "text": "vs"}),
            serde_json::json!({"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}),
        ];

        let conversions = &[("image_url", "image")];
        let result = convert_media_url_to_placeholder(&content_array, conversions);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0]["type"], "image");
        assert_eq!(result[1]["type"], "text");
        assert_eq!(result[2]["type"], "image");
    }

    /// Tests that only specified media types are converted, others preserved.
    #[test]
    fn test_convert_media_url_to_placeholder_selective_conversion() {
        let content_array = vec![
            serde_json::json!({"type": "audio_url", "audio_url": {"url": "https://example.com/audio.mp3"}}),
            serde_json::json!({"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}),
            serde_json::json!({"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}),
        ];

        // Only convert image_url
        let conversions = &[("image_url", "image")];
        let result = convert_media_url_to_placeholder(&content_array, conversions);

        assert_eq!(result.len(), 3);
        // audio_url and video_url should be preserved as-is
        assert_eq!(result[0]["type"], "audio_url");
        assert!(result[0].get("audio_url").is_some());
        assert_eq!(result[1]["type"], "video_url");
        assert!(result[1].get("video_url").is_some());
        // Only image_url should be converted
        assert_eq!(result[2]["type"], "image");
        assert!(result[2].get("image_url").is_none());
    }

    /// Tests converting multiple different media types at once.
    #[test]
    fn test_convert_media_url_to_placeholder_multiple_types() {
        let content_array = vec![
            serde_json::json!({"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}),
            serde_json::json!({"type": "text", "text": "and listen to"}),
            serde_json::json!({"type": "audio_url", "audio_url": {"url": "https://example.com/audio.mp3"}}),
            serde_json::json!({"type": "text", "text": "and watch"}),
            serde_json::json!({"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}),
        ];

        // Convert all media types
        let conversions = &[
            ("image_url", "image"),
            ("audio_url", "audio"),
            ("video_url", "video"),
        ];
        let result = convert_media_url_to_placeholder(&content_array, conversions);

        assert_eq!(result.len(), 5);
        assert_eq!(result[0]["type"], "image");
        assert!(result[0].get("image_url").is_none());
        assert_eq!(result[1]["type"], "text");
        assert_eq!(result[2]["type"], "audio");
        assert!(result[2].get("audio_url").is_none());
        assert_eq!(result[3]["type"], "text");
        assert_eq!(result[4]["type"], "video");
        assert!(result[4].get("video_url").is_none());
    }

    /// Tests that empty conversions list preserves all content.
    #[test]
    fn test_convert_media_url_to_placeholder_no_conversions() {
        let content_array = vec![
            serde_json::json!({"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}),
            serde_json::json!({"type": "text", "text": "hello"}),
        ];

        let conversions: &[(&str, &str)] = &[];
        let result = convert_media_url_to_placeholder(&content_array, conversions);

        assert_eq!(result.len(), 2);
        // Everything should be preserved as-is
        assert_eq!(result[0]["type"], "image_url");
        assert!(result[0].get("image_url").is_some());
        assert_eq!(result[1]["type"], "text");
    }

    /// Tests that DEFAULT_MEDIA_TYPE_CONVERSIONS only converts image_url,
    /// and preserves other media types like video_url and audio_url.
    #[test]
    fn test_default_media_type_conversions_only_converts_image_url() {
        let content_array = vec![
            serde_json::json!({"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}),
            serde_json::json!({"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}),
            serde_json::json!({"type": "audio_url", "audio_url": {"url": "https://example.com/audio.mp3"}}),
            serde_json::json!({"type": "text", "text": "hello"}),
        ];

        // Use the actual DEFAULT_MEDIA_TYPE_CONVERSIONS
        let result =
            convert_media_url_to_placeholder(&content_array, DEFAULT_MEDIA_TYPE_CONVERSIONS);

        assert_eq!(result.len(), 4);

        // image_url SHOULD be converted to image (it's in the default map)
        assert_eq!(result[0]["type"], "image");
        assert!(result[0].get("image_url").is_none());

        // video_url should NOT be converted (not in the default map)
        assert_eq!(result[1]["type"], "video");
        assert!(result[1].get("video_url").is_none());

        // audio_url should NOT be converted (not in the default map)
        assert_eq!(result[2]["type"], "audio");
        assert!(result[2].get("audio_url").is_none());

        // text should be unchanged
        assert_eq!(result[3]["type"], "text");
        assert_eq!(result[3]["text"], "hello");
    }

    #[test]
    fn test_may_be_fix_tool_schema_missing_type_and_properties() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {},
                        "strict": null
                    }
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let tools = serde_json::to_value(request.tools()).unwrap();

        assert!(tools[0]["function"]["parameters"]["type"] == "object");
        assert!(
            tools[0]["function"]["parameters"]["properties"]
                == serde_json::Value::Object(Default::default())
        );
    }

    #[test]
    fn test_may_be_fix_tool_schema_missing_type() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City and state, e.g., 'San Francisco, CA'"
                                }
                            }
                        },
                        "strict": null
                    }
                }
            ]
        }"#;
        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();

        let tools = serde_json::to_value(request.tools()).unwrap();

        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");

        let mut expected_properties = serde_json::Map::new();
        let mut location = serde_json::Map::new();
        location.insert(
            "type".to_string(),
            serde_json::Value::String("string".to_string()),
        );
        location.insert(
            "description".to_string(),
            serde_json::Value::String("City and state, e.g., 'San Francisco, CA'".to_string()),
        );
        expected_properties.insert("location".to_string(), serde_json::Value::Object(location));

        assert_eq!(
            tools[0]["function"]["parameters"]["properties"],
            serde_json::Value::Object(expected_properties)
        );
    }

    #[test]
    fn test_may_be_fix_tool_schema_missing_properties() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {"type": "object"},
                        "strict": null
                    }
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let tools = serde_json::to_value(request.tools()).unwrap();

        assert_eq!(
            tools[0]["function"]["parameters"]["properties"],
            serde_json::Value::Object(Default::default())
        );
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn test_may_be_fix_tool_schema_missing_description() {
        // `description` is optional in the OpenAI tool schema, but some chat
        // templates (e.g. gpt-oss harmony) concatenate it unconditionally and
        // fail on an `undefined`/null value. It must be backfilled to "".
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "parameters": {
                            "type": "object",
                            "properties": { "x": { "type": "string" } },
                            "required": ["x"],
                            "additionalProperties": false
                        },
                        "strict": null
                    }
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let tools = serde_json::to_value(request.tools()).unwrap();

        assert_eq!(
            tools[0]["function"]["description"],
            serde_json::Value::String(String::new())
        );
    }

    #[test]
    fn test_may_be_fix_tool_schema_null_description() {
        // An explicit null `description` must also be normalized to "".
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": null,
                        "parameters": {"type": "object", "properties": {}},
                        "strict": null
                    }
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let tools = serde_json::to_value(request.tools()).unwrap();

        assert_eq!(
            tools[0]["function"]["description"],
            serde_json::Value::String(String::new())
        );
    }

    #[test]
    fn test_may_be_fix_tool_schema_preserves_description() {
        // A present `description` must be left untouched.
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {"type": "object", "properties": {}},
                        "strict": null
                    }
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let tools = serde_json::to_value(request.tools()).unwrap();

        assert_eq!(
            tools[0]["function"]["description"],
            "Get the current weather in a given location"
        );
    }

    /// Tests that content arrays (containing only text parts) are correctly concatenated.
    #[test]
    fn test_may_be_fix_msg_content_user_multipart() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "part 1"},
                        {"type": "text", "text": "part 2"}
                    ]
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // Test array → string normalization (preserve_arrays=false for standard templates)
        let messages =
            serde_json::to_value(may_be_fix_msg_content(messages_raw, false, None)).unwrap();

        // Verify: text-only array is concatenated into a single string
        assert_eq!(
            messages[0]["content"],
            serde_json::Value::String("part 1\npart 2".to_string())
        );
    }

    /// Tests that the function correctly handles a conversation
    /// with multiple roles and mixed message types:
    #[test]
    fn test_may_be_fix_msg_content_mixed_messages() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "World"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": "Hi there!"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Another"},
                        {"type": "text", "text": "multi-part"},
                        {"type": "text", "text": "message"}
                    ]
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // Test array → string normalization (preserve_arrays=false for standard templates)
        let messages =
            serde_json::to_value(may_be_fix_msg_content(messages_raw, false, None)).unwrap();

        // Verify: System message with string content remains unchanged
        assert_eq!(
            messages[0]["content"],
            serde_json::Value::String("You are a helpful assistant".to_string())
        );

        // Verify: User message with text-only array is concatenated
        assert_eq!(
            messages[1]["content"],
            serde_json::Value::String("Hello\nWorld".to_string())
        );

        // Verify: Assistant message with string content remains unchanged
        assert_eq!(
            messages[2]["content"],
            serde_json::Value::String("Hi there!".to_string())
        );

        // Verify: Second user message with text-only array is concatenated
        assert_eq!(
            messages[3]["content"],
            serde_json::Value::String("Another\nmulti-part\nmessage".to_string())
        );
    }

    /// Tests that empty content arrays remain unchanged.
    #[test]
    fn test_may_be_fix_msg_content_empty_array() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": []
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // Empty arrays should be preserved regardless of preserve_arrays setting
        let messages =
            serde_json::to_value(may_be_fix_msg_content(messages_raw, false, None)).unwrap();

        // Verify: Empty arrays are preserved as-is
        assert!(messages[0]["content"].is_array());
        assert_eq!(messages[0]["content"].as_array().unwrap().len(), 0);
    }

    /// Empty arrays must stay as `[]` even when a flatten-time placeholder
    /// template is provided (Phi-3 / LLaVA-1.5 path). Without the
    /// `!content_array.is_empty()` guard in `may_be_fix_msg_content`,
    /// an empty content array would silently flatten to `""` and the
    /// chat template would render an entirely empty message instead of
    /// failing or being preserved.
    #[test]
    fn test_may_be_fix_msg_content_empty_array_with_placeholder_template() {
        let json_str = r#"{
            "model": "phi-3-vision",
            "messages": [
                {
                    "role": "user",
                    "content": []
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // preserve_arrays=false + image_placeholder_template=Some(...) is
        // the combination that previously flattened `[]` to `""`.
        let messages = serde_json::to_value(may_be_fix_msg_content(
            messages_raw,
            false,
            Some("<|image_{n}|>"),
        ))
        .unwrap();

        assert!(
            messages[0]["content"].is_array(),
            "empty array should be preserved as `[]`, not flattened to `\"\"`"
        );
        assert_eq!(messages[0]["content"].as_array().unwrap().len(), 0);
    }

    /// Tests that messages with simple string content remain unchanged.
    #[test]
    fn test_may_be_fix_msg_content_single_text() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": "Simple text message"
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // Test with preserve_arrays=false (standard templates)
        let messages =
            serde_json::to_value(may_be_fix_msg_content(messages_raw, false, None)).unwrap();

        // Verify: String content is not modified
        assert_eq!(
            messages[0]["content"],
            serde_json::Value::String("Simple text message".to_string())
        );
    }

    /// Tests that content arrays with mixed types (text + non-text) remain as arrays,
    /// and that image_url is converted to image placeholder.
    #[test]
    fn test_may_be_fix_msg_content_mixed_types() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Check this image:"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                        {"type": "text", "text": "What do you see?"}
                    ]
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // Mixed content should be preserved regardless of preserve_arrays setting
        let messages =
            serde_json::to_value(may_be_fix_msg_content(messages_raw, false, None)).unwrap();

        // Verify: Mixed content types are preserved as array for template handling
        // image_url should be converted to image placeholder
        assert!(messages[0]["content"].is_array());
        let content_array = messages[0]["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 3);
        assert_eq!(content_array[0]["type"], "text");
        assert_eq!(content_array[1]["type"], "image");
        assert!(content_array[1].get("image_url").is_none());
        assert_eq!(content_array[2]["type"], "text");
    }

    /// Mixed text+image array with a string-content template and an
    /// `<|image_{n}|>`-style placeholder (Phi-3-vision) — content must be
    /// flattened to a single string with numbered image markers in place of
    /// the image parts. The previous default (leave as array) would crash
    /// the Phi-3 template's `'+' message.content` concatenation.
    #[test]
    fn test_may_be_fix_msg_content_flattens_phi3_style() {
        let json_str = r#"{
            "model": "phi-3-vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First "},
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
                        {"type": "text", "text": " then "},
                        {"type": "image_url", "image_url": {"url": "https://example.com/b.jpg"}},
                        {"type": "text", "text": "?"}
                    ]
                }
            ]
        }"#;
        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        let messages = serde_json::to_value(may_be_fix_msg_content(
            messages_raw,
            false,
            Some("<|image_{n}|>"),
        ))
        .unwrap();

        let content = messages[0]["content"].as_str().expect("content flattened");
        assert_eq!(content, "First <|image_1|> then <|image_2|>?");
    }

    /// Same flattening with a static placeholder (LLaVA-1.5 `<image>`).
    #[test]
    fn test_may_be_fix_msg_content_flattens_llava_style() {
        let json_str = r#"{
            "model": "llava-1.5-7b-hf",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe: "},
                        {"type": "image_url", "image_url": {"url": "https://example.com/x.jpg"}}
                    ]
                }
            ]
        }"#;
        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        let messages =
            serde_json::to_value(may_be_fix_msg_content(messages_raw, false, Some("<image>")))
                .unwrap();

        let content = messages[0]["content"].as_str().expect("content flattened");
        assert_eq!(content, "Describe: <image>");
    }

    /// Nemotron-Parse pass-through path: a mixed text+image array with an
    /// empty placeholder (`""`) flattens to the text parts only — the image
    /// contributes nothing because the vision encoder consumes it out-of-band.
    /// Without this, `{{ message.content }}` would JSON-serialize the array
    /// into the prompt (the gibberish failure mode).
    #[test]
    fn test_may_be_fix_msg_content_flattens_empty_placeholder() {
        let json_str = r#"{
            "model": "nvidia/NVIDIA-Nemotron-Parse-v1.2",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                    ]
                }
            ]
        }"#;
        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        let messages =
            serde_json::to_value(may_be_fix_msg_content(messages_raw, false, Some(""))).unwrap();

        let content = messages[0]["content"].as_str().expect("content flattened");
        assert_eq!(
            content,
            "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
        );
    }

    /// End-to-end render through the Nemotron-Parse pass-through chat template:
    /// a text+image chat request must produce exactly the control-token prompt,
    /// with the image dropped from the rendered text. The renderer is agnostic
    /// to the control tokens themselves (they are just the text part, passed
    /// through verbatim), so both `predict_no_text_in_pic` and
    /// `predict_text_in_pic` prompts round-trip identically.
    #[test]
    fn test_render_nemotron_parse_passthrough() {
        use super::super::tokcfg::ChatTemplate;
        use super::{ContextMixins, HfTokenizerConfigJsonFormatter};

        let chat_template: ChatTemplate = serde_json::from_value(serde_json::json!({
            "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        }))
        .unwrap();
        let formatter =
            HfTokenizerConfigJsonFormatter::new(chat_template, ContextMixins::new(&[])).unwrap();

        for prompt in [
            "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>",
            "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>",
        ] {
            let request: NvCreateChatCompletionRequest =
                serde_json::from_value(serde_json::json!({
                    "model": "nvidia/NVIDIA-Nemotron-Parse-v1.2",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                        ]
                    }]
                }))
                .unwrap();

            let rendered = formatter.render(&request).unwrap();
            assert_eq!(
                rendered, prompt,
                "rendered prompt must be the control tokens only, with no JSON-serialized image array"
            );
        }
    }

    /// Tests that content arrays containing only non-text types remain as arrays,
    /// and image_url types are converted to image placeholders.
    #[test]
    fn test_may_be_fix_msg_content_non_text_only() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
                        {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}}
                    ]
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // Non-text arrays should be preserved regardless of preserve_arrays setting
        let messages =
            serde_json::to_value(may_be_fix_msg_content(messages_raw, false, None)).unwrap();

        // Verify: Non-text content arrays are preserved, with image_url converted to image
        assert!(messages[0]["content"].is_array());
        let content_array = messages[0]["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 2);
        assert_eq!(content_array[0]["type"], "image");
        assert_eq!(content_array[1]["type"], "image");
    }

    #[test]
    fn test_none_tools_safe_for_all_templates() {
        use super::tokcfg::ChatTemplate;
        use super::{ContextMixins, HfTokenizerConfigJsonFormatter};

        // Due to minijinja limitations the expressions in conditional statements may not be short-circuited
        // This checks that our custom length filter works to avoid errors in this scenario
        // length should return 0 if tools is None and 'if tools is iterable and tools | length > 0' should evaluate to false
        let length_template = r#"
{%- if tools is iterable and tools | length > 0 %}
Tools available: {{ tools | length }}
{%- else %}
No tools
{%- endif %}
"#;

        // Because we return None for tools when there are no tools this scenario should also be evaluate to false
        // This is similar to the default jinja template behavior seen with llama models which check if tools is not none to activate tool mode
        let no_tool_template = r#"
{%- if tools is not none %}
TOOL MODE
{%- else %}
NORMAL MODE
{%- endif %}
"#;

        let chat_template: ChatTemplate = serde_json::from_value(serde_json::json!({
            "chat_template": [
                {"safe_length": length_template},
                {"no_tool": no_tool_template}
            ]
        }))
        .unwrap();

        let formatter =
            HfTokenizerConfigJsonFormatter::new(chat_template, ContextMixins::new(&[])).unwrap();

        let ctx = context! { tools => Option::<Value>::None };

        let result1 = formatter
            .env
            .get_template("safe_length")
            .unwrap()
            .render(&ctx);
        println!("Safe length template with no tools => None: {:?}", result1);
        assert!(
            result1.is_ok(),
            "Jinja template with and conditional and length filter should handle None: {:?}",
            result1
        );
        assert!(
            result1.unwrap().contains("No tools"),
            "Should show 'No tools'"
        );

        let result2 = formatter.env.get_template("no_tool").unwrap().render(&ctx);
        println!("Default template with no tools => None: {:?}", result2);
        assert!(
            result2.is_ok(),
            "Jinja template with if tools is not none conditional should handle None: {:?}",
            result2
        );
        assert!(result2.unwrap().contains("NORMAL MODE"));
    }

    /// Tests mixed content type scenarios.
    #[test]
    fn test_may_be_fix_msg_content_multiple_content_types() {
        // Scenario 1: Multiple different content types (text + image + audio)
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Listen to this:"},
                        {"type": "audio_url", "audio_url": {"url": "https://example.com/audio.mp3"}},
                        {"type": "text", "text": "And look at:"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                        {"type": "text", "text": "What do you think?"}
                    ]
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();
        let messages =
            serde_json::to_value(may_be_fix_msg_content(messages_raw, false, None)).unwrap();

        // Mixed types should preserve array structure, with image_url converted to image
        assert!(messages[0]["content"].is_array());
        let content_array = messages[0]["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 5);
        assert_eq!(content_array[0]["type"], "text");
        assert_eq!(content_array[1]["type"], "audio");
        assert_eq!(content_array[2]["type"], "text");
        assert_eq!(content_array[3]["type"], "image");
        assert_eq!(content_array[4]["type"], "text");

        // Scenario 2: Unknown/future content types mixed with text
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Check this:"},
                        {"type": "video_url", "video_url": {"url": "https://example.com/vid.mp4"}},
                        {"type": "text", "text": "Interesting?"}
                    ]
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();
        let messages =
            serde_json::to_value(may_be_fix_msg_content(messages_raw, false, None)).unwrap();

        // Unknown types mixed with text should preserve array
        assert!(messages[0]["content"].is_array());
        assert_eq!(messages[0]["content"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_normalize_tool_arguments_tojson() {
        let tmpl = r#"{{ messages[0].tool_calls[0].function.arguments | tojson }}"#;

        // Message with tool_calls containing JSON string arguments
        let mut messages = serde_json::Value::Array(vec![serde_json::json!({
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "arguments": "{\"format\":\"celsius\",\"location\":\"San Francisco, CA\"}"
                }
            }]
        })]);

        normalize_tool_calls_arguments_in_messages(&mut messages);

        let mut env = Environment::new();
        env.add_filter("tojson", super::super::tokcfg::tojson);
        env.add_template("t", tmpl).unwrap();
        let out = env
            .get_template("t")
            .unwrap()
            .render(context! { messages => messages.as_array().unwrap() })
            .unwrap();

        // Should produce clean JSON without double-encoding, with Python
        // json.dumps separators (what transformers' tojson emits).
        assert_eq!(
            out,
            r#"{"format": "celsius", "location": "San Francisco, CA"}"#
        );
    }

    #[test]
    fn test_normalize_tool_arguments_items_loop() {
        let tmpl = r#"{% for k, v in messages[0].tool_calls[0].function.arguments|items %}{{k}}={{v}};{% endfor %}"#;

        let mut messages = serde_json::Value::Array(vec![serde_json::json!({
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "f",
                    "arguments": "{\"a\":1,\"b\":\"x\"}"
                }
            }]
        })]);

        normalize_tool_calls_arguments_in_messages(&mut messages);

        let mut env = Environment::new();
        env.add_template("t", tmpl).unwrap();
        let out = env
            .get_template("t")
            .unwrap()
            .render(context! { messages => messages.as_array().unwrap() })
            .unwrap();

        assert!(out == "a=1;b=x;" || out == "b=x;a=1;");
    }

    #[test]
    fn test_normalize_tool_arguments_legacy_function_call() {
        // Test deprecated function_call format (OpenAI compat)
        let mut messages = serde_json::Value::Array(vec![serde_json::json!({
            "role": "assistant",
            "function_call": {
                "name": "get_weather",
                "arguments": "{\"location\":\"NYC\"}"
            }
        })]);

        normalize_function_call_arguments_in_messages(&mut messages);

        assert_eq!(
            messages[0]["function_call"]["arguments"],
            serde_json::json!({"location": "NYC"})
        );
    }

    #[test]
    fn test_normalize_tool_arguments_malformed_json_passthrough() {
        // Malformed JSON should be left as a string
        let mut messages = serde_json::Value::Array(vec![serde_json::json!({
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "f",
                    "arguments": "not valid json at all"
                }
            }]
        })]);

        normalize_tool_calls_arguments_in_messages(&mut messages);

        assert_eq!(
            messages[0]["tool_calls"][0]["function"]["arguments"],
            serde_json::Value::String("not valid json at all".to_string())
        );
    }

    #[test]
    fn test_normalize_tool_arguments_with_multimodal_content() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Check this:"},
                        {"type": "video_url", "video_url": {"url": "https://example.com/vid.mp4"}},
                        {"type": "text", "text": "Interesting?"}
                    ]
                },
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "analyze_video",
                            "arguments": "{\"url\":\"https://example.com/vid.mp4\",\"format\":\"mp4\"}"
                        }
                    }]
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // Apply content normalization with preserve_arrays=false (standard templates)
        let mut messages =
            serde_json::to_value(may_be_fix_msg_content(messages_raw, false, None)).unwrap();

        normalize_tool_calls_arguments_in_messages(&mut messages);

        // Multimodal content preserved as array (mixed types not flattened)
        assert!(messages[0]["content"].is_array());
        assert_eq!(messages[0]["content"].as_array().unwrap().len(), 3);

        // Tool arguments deserialized to object
        assert!(messages[1]["tool_calls"][0]["function"]["arguments"].is_object());
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["arguments"]["url"],
            "https://example.com/vid.mp4"
        );
    }

    /// Tests string → array normalization for multimodal templates
    #[test]
    fn test_may_be_fix_msg_content_string_to_array() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?"
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // Test with preserve_arrays=true (multimodal templates)
        let messages =
            serde_json::to_value(may_be_fix_msg_content(messages_raw, true, None)).unwrap();

        // Verify: String is converted to array format
        assert!(messages[0]["content"].is_array());
        let content_array = messages[0]["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 1);
        assert_eq!(content_array[0]["type"], "text");
        assert_eq!(content_array[0]["text"], "Hello, how are you?");
    }

    /// Tests that arrays are preserved when preserve_arrays=true
    #[test]
    fn test_may_be_fix_msg_content_array_preserved_with_multimodal() {
        let json_str = r#"{
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "part 1"},
                        {"type": "text", "text": "part 2"}
                    ]
                }
            ]
        }"#;

        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();
        let messages_raw = serde_json::to_value(request.messages()).unwrap();

        // Test with preserve_arrays=true (multimodal templates)
        let messages =
            serde_json::to_value(may_be_fix_msg_content(messages_raw, true, None)).unwrap();

        // Verify: Array is preserved as-is
        assert!(messages[0]["content"].is_array());
        let content_array = messages[0]["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 2);
        assert_eq!(content_array[0]["text"], "part 1");
        assert_eq!(content_array[1]["text"], "part 2");
    }

    fn user() -> Msg {
        Msg::User(Default::default())
    }
    fn tool() -> Msg {
        Msg::Tool(Default::default())
    }

    fn dummy_state(messages: Vec<Msg>) -> NvCreateChatCompletionRequest {
        let json = serde_json::json!({
            "model": "test-model",
            "messages": messages
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn add_after_user() {
        let s = dummy_state(vec![user()]);
        assert!(s.should_add_generation_prompt());
    }

    #[test]
    fn add_after_tool() {
        let s = dummy_state(vec![tool()]);
        assert!(s.should_add_generation_prompt());
    }

    #[test]
    fn add_when_empty() {
        let s = dummy_state(vec![]);
        assert!(s.should_add_generation_prompt());
    }

    /// Helper to build a formatter with a simple tool-aware template.
    fn tool_aware_formatter(
        exclude_tools_when_tool_choice_none: bool,
    ) -> HfTokenizerConfigJsonFormatter {
        let template = r#"
{%- if tools is iterable and tools | length > 0 %}
TOOL_MODE tools={{ tools | length }}
{%- else %}
NORMAL_MODE
{%- endif %}
{{ messages[0].content }}"#;

        let chat_template: super::tokcfg::ChatTemplate =
            serde_json::from_value(serde_json::json!({ "chat_template": template })).unwrap();

        HfTokenizerConfigJsonFormatter::with_options(
            chat_template,
            ContextMixins::new(&[]),
            exclude_tools_when_tool_choice_none,
        )
        .unwrap()
    }

    fn gemma4_tool_template_for_tests() -> &'static str {
        r#"
{{ bos_token }}
{%- set loop_messages = messages -%}
{%- set ns_turn = namespace(last_user_idx=-1) -%}
{%- for i in range(loop_messages | length) -%}
    {%- if loop_messages[i]['role'] == 'user' -%}
        {%- set ns_turn.last_user_idx = i -%}
    {%- endif -%}
{%- endfor -%}
{%- for message in loop_messages -%}
    {%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}
    {{- '<|turn>' + role + '\n' }}

    {%- if message.get('reasoning') and loop.index0 > ns_turn.last_user_idx and message.get('tool_calls') -%}
        {{- '<|channel>thought\n' + message['reasoning'] + '\n<channel|>'}}
    {%- endif -%}

            {%- if message['tool_calls'] -%}
                {%- for tool_call in message['tool_calls'] -%}
                    {%- set function = tool_call['function'] -%}
                    {{- '<|tool_call>call:' + function['name'] + '{' -}}
                    {%- if function['arguments'] is mapping -%}
                        {%- set ns_args = namespace(found_first=false) -%}
                        {%- for key, value in function['arguments'] | dictsort -%}
                            {%- if ns_args.found_first %},{% endif -%}
                            {%- set ns_args.found_first = true -%}
                            {{- key -}}:{{- value -}}
                        {%- endfor -%}
                    {%- elif function['arguments'] is string -%}
                        {{- function['arguments'] -}}
                    {%- endif -%}
                    {{- '}<tool_call|>' -}}
                {%- endfor -%}
            {%- endif -%}

            {%- if message['content'] is string -%}
                {{- message['content'] -}}
            {%- endif -%}
    {{- '<turn|>\n' -}}
{%- endfor -%}
"#
    }

    fn make_gemma4_tool_formatter_for_tests() -> HfTokenizerConfigJsonFormatter {
        let chat_template: ChatTemplate = serde_json::from_value(serde_json::json!({
            "chat_template": gemma4_tool_template_for_tests()
        }))
        .unwrap();
        HfTokenizerConfigJsonFormatter::new(chat_template, ContextMixins::new(&[])).unwrap()
    }

    /// Helper to build a request with tools and optional tool_choice.
    fn request_with_tool_choice(tool_choice: &str) -> NvCreateChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
                }
            }],
            "tool_choice": tool_choice
        }))
        .unwrap()
    }

    #[test]
    fn test_exclude_tools_strips_when_tool_choice_none() {
        let formatter = tool_aware_formatter(true);
        let request = request_with_tool_choice("none");
        let result = formatter.render(&request).unwrap();
        assert!(
            result.contains("NORMAL_MODE"),
            "With exclude_tools=true and tool_choice=none, tools should be stripped. Got: {}",
            result
        );
    }

    #[test]
    fn test_exclude_tools_keeps_when_tool_choice_auto() {
        let formatter = tool_aware_formatter(true);
        let request = request_with_tool_choice("auto");
        let result = formatter.render(&request).unwrap();
        assert!(
            result.contains("TOOL_MODE"),
            "With tool_choice=auto, tools should be included. Got: {}",
            result
        );
    }

    #[test]
    fn test_no_exclude_tools_keeps_when_tool_choice_none() {
        let formatter = tool_aware_formatter(false);
        let request = request_with_tool_choice("none");
        let result = formatter.render(&request).unwrap();
        assert!(
            result.contains("TOOL_MODE"),
            "With exclude_tools=false and tool_choice=none, tools should NOT be stripped. Got: {}",
            result
        );
    }

    #[test]
    fn test_inject_reasoning_content_segments_with_tool_calls() {
        // Assistant message with reasoning_content segments and tool_calls
        let mut messages = serde_json::json!([
            {
                "role": "user",
                "content": "What is sqrt(144) and sqrt(256)?"
            },
            {
                "role": "assistant",
                "content": "Let me calculate those.",
                "reasoning_content": ["I need to compute sqrt(144)", "Now sqrt(256)", ""],
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": "{\"expr\": \"sqrt(144)\"}"
                        }
                    },
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": "{\"expr\": \"sqrt(256)\"}"
                        }
                    }
                ]
            }
        ]);

        inject_reasoning_content_into_messages(&mut messages);

        let assistant = &messages[1];

        // reasoning_content should be removed
        assert!(
            assistant.get("reasoning_content").is_none(),
            "reasoning_content should be removed after injection"
        );

        // content should have <think> blocks prepended (empty segment skipped)
        let content = assistant["content"].as_str().unwrap();
        assert!(
            content.starts_with("<think>I need to compute sqrt(144)</think>"),
            "content should start with first reasoning segment, got: {}",
            content
        );
        assert!(
            content.contains("<think>Now sqrt(256)</think>"),
            "content should contain second reasoning segment"
        );
        // Empty third segment should NOT produce <think></think>
        assert!(
            !content.contains("<think></think>"),
            "empty segments should be skipped"
        );
        // Original content should be preserved at the end
        assert!(
            content.ends_with("Let me calculate those."),
            "original content should be at the end, got: {}",
            content
        );

        // tool_calls should be untouched
        assert!(assistant.get("tool_calls").is_some());
        assert_eq!(assistant["tool_calls"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_gemma4_template_renders_reasoning_content_segments_around_tool_calls() {
        let formatter = make_gemma4_tool_formatter_for_tests();
        assert!(
            formatter.tool_use_template_handles_reasoning,
            "Gemma4 template adaptation should make reasoning_content native"
        );

        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "gemma4-test",
            "messages": [
                {"role": "user", "content": "inspect two things"},
                {
                    "role": "assistant",
                    "content": null,
                    "reasoning_content": [
                        "Think before the first call.",
                        "Think before the second call.",
                        "Think after both calls."
                    ],
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {
                                "name": "first_tool",
                                "arguments": "{\"path\":\".\"}"
                            }
                        },
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "second_tool",
                                "arguments": "{\"path\":\"/tmp\"}"
                            }
                        }
                    ]
                }
            ]
        }))
        .unwrap();

        let rendered = formatter.render(&request).unwrap();

        let expected = concat!(
            "<|channel>thought\nThink before the first call.\n<channel|>",
            "<|tool_call>call:first_tool{path:.}<tool_call|>",
            "<|channel>thought\nThink before the second call.\n<channel|>",
            "<|tool_call>call:second_tool{path:/tmp}<tool_call|>",
            "<|channel>thought\nThink after both calls.\n<channel|>"
        );
        assert!(
            rendered.contains(expected),
            "Gemma4 reasoning segments should stay adjacent to their tool calls, got: {rendered}"
        );
        assert!(!rendered.contains("<think>"));
        assert!(!rendered.contains("reasoning_content"));
    }

    #[test]
    fn test_gemma4_template_renders_reasoning_content_without_tool_calls() {
        let formatter = make_gemma4_tool_formatter_for_tests();
        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "gemma4-test",
            "messages": [
                {"role": "user", "content": "answer directly"},
                {
                    "role": "assistant",
                    "content": "Direct answer.",
                    "reasoning_content": "Private thought."
                }
            ]
        }))
        .unwrap();

        let rendered = formatter.render(&request).unwrap();

        assert!(
            rendered.contains("<|channel>thought\nPrivate thought.\n<channel|>Direct answer."),
            "Gemma4 reasoning_content should render in the thought channel, got: {rendered}"
        );
        assert!(!rendered.contains("<think>"));
        assert!(!rendered.contains("reasoning_content"));
    }

    /// Regression: when a config ships a separate non-tool `default` template
    /// (dict form), adapting only the `tool_use` template to read
    /// `reasoning_content` must NOT suppress `<think>` injection on the
    /// `default` path. A global `any()` flag would flip true off the adapted
    /// `tool_use` template and silently drop reasoning on no-tool renders.
    #[test]
    fn test_reasoning_flag_is_per_template_not_global() {
        // Plain default template: renders content, never mentions reasoning_content
        // and lacks the Gemma4 fingerprint, so it is left untouched.
        const PLAIN_DEFAULT: &str = "{{ bos_token }}{%- for message in messages -%}\
            {{ message['role'] }}: {{ message['content'] }}\n{%- endfor -%}";

        let chat_template: ChatTemplate = serde_json::from_value(serde_json::json!({
            "chat_template": [
                {"default": PLAIN_DEFAULT},
                {"tool_use": gemma4_tool_template_for_tests()},
            ]
        }))
        .unwrap();
        let formatter =
            HfTokenizerConfigJsonFormatter::new(chat_template, ContextMixins::new(&[])).unwrap();

        // The adapted Gemma4 tool_use template handles reasoning natively; the
        // untouched plain default template does not. The flag must reflect that
        // per-template split, not a global OR across both.
        assert!(
            formatter.tool_use_template_handles_reasoning,
            "adapted gemma4 tool_use template should handle reasoning natively"
        );
        assert!(
            !formatter.default_template_handles_reasoning,
            "plain default template does not reference reasoning_content"
        );

        // A no-tools request routes to `default`. Reasoning must still be injected
        // as a <think> block — not silently dropped by the tool_use template's flag.
        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "gemma4-test",
            "messages": [
                {"role": "user", "content": "answer directly"},
                {
                    "role": "assistant",
                    "content": "Direct answer.",
                    "reasoning_content": "Private thought."
                }
            ]
        }))
        .unwrap();

        let rendered = formatter.render(&request).unwrap();
        assert!(
            rendered.contains("<think>Private thought.</think>Direct answer."),
            "reasoning must be injected on the no-tool default path, got: {rendered}"
        );
    }

    #[test]
    fn test_inject_reasoning_content_text_variant() {
        let mut messages = serde_json::json!([
            {
                "role": "assistant",
                "content": "The answer is 42.",
                "reasoning_content": "Let me think about this carefully."
            }
        ]);

        inject_reasoning_content_into_messages(&mut messages);

        let assistant = &messages[0];
        assert!(assistant.get("reasoning_content").is_none());
        let content = assistant["content"].as_str().unwrap();
        assert_eq!(
            content,
            "<think>Let me think about this carefully.</think>The answer is 42."
        );
    }

    #[test]
    fn test_inject_reasoning_content_null_content() {
        // reasoning_content present but content is null
        let mut messages = serde_json::json!([
            {
                "role": "assistant",
                "content": null,
                "reasoning_content": "Thinking...",
                "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
            }
        ]);

        inject_reasoning_content_into_messages(&mut messages);

        let content = messages[0]["content"].as_str().unwrap();
        assert_eq!(content, "<think>Thinking...</think>");
        assert!(messages[0].get("reasoning_content").is_none());
    }

    #[test]
    fn test_inject_reasoning_content_skips_non_assistant() {
        let mut messages = serde_json::json!([
            {
                "role": "user",
                "content": "hello",
                "reasoning_content": "should not be touched"
            }
        ]);

        inject_reasoning_content_into_messages(&mut messages);

        // User message should be untouched
        assert!(messages[0].get("reasoning_content").is_some());
    }

    // Helper: create a formatter with a minimal chat template for render tests
    fn make_test_formatter() -> HfTokenizerConfigJsonFormatter {
        use super::tokcfg::ChatTemplate;
        use super::{ContextMixins, HfTokenizerConfigJsonFormatter};

        // Minimal template that renders content verbatim — enough to verify
        // that reasoning_content injection works through the full pipeline.
        let template = r#"{%- for message in messages %}{{ message.role }}: {{ message.content }}
{%- endfor %}
{%- if add_generation_prompt %}assistant:{%- endif %}"#;

        let chat_template: ChatTemplate = serde_json::from_value(serde_json::json!({
            "chat_template": template
        }))
        .unwrap();

        HfTokenizerConfigJsonFormatter::new(chat_template, ContextMixins::new(&[])).unwrap()
    }

    // Verify reasoning_content (Text variant) from a prior assistant turn
    // appears as a <think> block in the rendered prompt.
    #[test]
    fn test_reasoning_content_text_roundtrip_render() {
        use super::OAIPromptFormatter;
        let formatter = make_test_formatter();

        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "What is sqrt(144)?"},
                {
                    "role": "assistant",
                    "content": "The answer is 12.",
                    "reasoning_content": "I need to compute the square root of 144."
                },
                {"role": "user", "content": "Are you sure?"}
            ]
        }))
        .unwrap();

        let rendered = formatter.render(&request).unwrap();

        assert!(
            rendered.contains("<think>I need to compute the square root of 144.</think>"),
            "reasoning_content must appear as <think> block, got: {}",
            rendered
        );
        assert!(
            rendered.contains("The answer is 12."),
            "original content must be preserved"
        );
        assert!(
            !rendered.contains("reasoning_content"),
            "raw reasoning_content field should not leak into prompt"
        );
    }

    // Verify a full agentic flow: assistant reasons, calls a tool, gets a
    // result, then reasons again before answering. Both reasoning turns must
    // survive into the rendered prompt.
    #[test]
    fn test_reasoning_content_agentic_tool_call_roundtrip_render() {
        use super::OAIPromptFormatter;
        let formatter = make_test_formatter();

        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "What is sqrt(144) + sqrt(256)?"},
                {
                    "role": "assistant",
                    "content": null,
                    "reasoning_content": "I need to compute both square roots. Let me start with sqrt(144).",
                    "tool_calls": [{
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": "{\"expr\": \"sqrt(144)\"}"
                        }
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_0",
                    "content": "12"
                },
                {
                    "role": "assistant",
                    "content": "sqrt(144) = 12 and sqrt(256) = 16, so the answer is 28.",
                    "reasoning_content": "Got 12 for sqrt(144). Now sqrt(256) = 16. Sum is 28."
                },
                {"role": "user", "content": "Thanks!"}
            ]
        }))
        .unwrap();

        let rendered = formatter.render(&request).unwrap();

        // First assistant turn: reasoning with tool call, null content
        assert!(
            rendered.contains("<think>I need to compute both square roots"),
            "first turn reasoning must be in prompt, got: {}",
            rendered
        );
        // Second assistant turn: reasoning with final answer
        assert!(
            rendered.contains("<think>Got 12 for sqrt(144)"),
            "second turn reasoning must be in prompt"
        );
        assert!(
            rendered.contains("the answer is 28"),
            "final answer content must be preserved"
        );
        // No raw reasoning_content in output
        assert!(
            !rendered.contains("reasoning_content"),
            "raw reasoning_content field should not leak into prompt"
        );
    }

    // Template that does NOT reference reasoning_content — injection should happen.
    #[test]
    fn test_reasoning_injected_when_template_ignores_it() {
        use super::OAIPromptFormatter;
        let formatter = make_test_formatter();

        // Formatter uses a simple template that doesn't reference reasoning_content
        assert!(!formatter.default_template_handles_reasoning);
        assert!(!formatter.tool_use_template_handles_reasoning);

        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "Hi.",
                    "reasoning_content": "The user said hello."
                },
                {"role": "user", "content": "Bye"}
            ]
        }))
        .unwrap();

        let rendered = formatter.render(&request).unwrap();
        assert!(
            rendered.contains("<think>The user said hello.</think>"),
            "injection must happen when template ignores reasoning_content, got: {}",
            rendered
        );
    }

    // Template that DOES reference reasoning_content — injection must be skipped.
    #[test]
    fn test_reasoning_not_injected_when_template_handles_it() {
        use super::tokcfg::ChatTemplate;
        use super::{ContextMixins, HfTokenizerConfigJsonFormatter, OAIPromptFormatter};

        // Template that natively renders reasoning_content (like Nemotron/Qwen3)
        let template = r#"{%- for message in messages %}{%- if message.role == "assistant" and message.reasoning_content is defined and message.reasoning_content %}<think>{{ message.reasoning_content }}</think>
{%- endif %}{{ message.role }}: {{ message.content }}
{%- endfor %}
{%- if add_generation_prompt %}assistant:{%- endif %}"#;

        let chat_template: ChatTemplate = serde_json::from_value(serde_json::json!({
            "chat_template": template
        }))
        .unwrap();

        let formatter =
            HfTokenizerConfigJsonFormatter::new(chat_template, ContextMixins::new(&[])).unwrap();

        // Verify detection worked
        assert!(formatter.default_template_handles_reasoning);
        assert!(formatter.tool_use_template_handles_reasoning);

        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "Hi.",
                    "reasoning_content": "The user said hello."
                },
                {"role": "user", "content": "Bye"}
            ]
        }))
        .unwrap();

        let rendered = formatter.render(&request).unwrap();

        // Template renders reasoning natively — no duplicate injection
        assert!(
            rendered.contains("<think>The user said hello.</think>"),
            "template must render reasoning_content natively, got: {}",
            rendered
        );
        // Must NOT have double <think> blocks
        let think_count = rendered.matches("<think>").count();
        assert_eq!(
            think_count, 1,
            "must have exactly one <think> block (from template), got {} in: {}",
            think_count, rendered
        );
    }

    /// Real Qwen3-4B-Thinking-2507 chat template (verbatim from
    /// `Qwen/Qwen3-4B-Thinking-2507/tokenizer_config.json`). Used to
    /// regression-test append-only rendering across multi-step tool use.
    const QWEN3_THINKING_TEMPLATE: &str = r##"{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n<think>\n' }}
{%- endif %}"##;

    fn qwen3_thinking_formatter() -> HfTokenizerConfigJsonFormatter {
        let chat_template: ChatTemplate = serde_json::from_value(serde_json::json!({
            "chat_template": QWEN3_THINKING_TEMPLATE,
        }))
        .unwrap();
        HfTokenizerConfigJsonFormatter::new(chat_template, ContextMixins::new(&[])).unwrap()
    }

    #[test]
    fn test_qwen3_thinking_template_flags_detected() {
        let formatter = qwen3_thinking_formatter();
        assert!(
            formatter.tool_use_template_handles_reasoning,
            "template references reasoning_content directly"
        );
        // The Qwen3-Thinking template is registered as both `default` and
        // `tool_use` (single-string HF chat template), so both flags must fire.
        assert!(
            formatter.default_template_handles_tool_calls_arguments_string,
            "default template branches on `arguments is string`"
        );
        assert!(
            formatter.tool_use_template_handles_tool_calls_arguments_string,
            "tool_use template branches on `arguments is string`"
        );
    }

    /// Across a multi-step tool-use turn, the rendered prompt for turn N+1
    /// must be a strict prefix-extension of [turn-N prompt + bytes the model
    /// emitted on turn N]. Otherwise KV-cache prefix matching falls off a
    /// cliff every time a tool result comes back.
    ///
    /// The Qwen3-Thinking template's `is string` branch (template lines 63-67)
    /// renders `tool_call.arguments` verbatim from the OpenAI-canonical JSON
    /// string. Pre-parsing that string into an object forces the `else` branch
    /// and re-emits with minijinja's compact `tojson`, breaking append-only.
    #[test]
    fn test_qwen3_thinking_append_only_across_tool_use_turn() {
        let formatter = qwen3_thinking_formatter();

        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }]);

        // Turn 1: server is asked to produce the first assistant turn.
        let turn1_request: NvCreateChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "model": "qwen3-thinking",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What's the weather in San Francisco?"},
                ],
                "tools": tools,
            }))
            .unwrap();
        let p1 = formatter.render(&turn1_request).unwrap();

        // Bytes the model emits next. Spacing matches the Qwen3 training
        // distribution (Python jinja2 / json.dumps defaults: `, ` and `: `).
        // Empty content + reasoning + a tool call.
        let model_emitted = "I'll call get_weather for SF.\n\
            </think>\n\n\
            <tool_call>\n\
            {\"name\": \"get_weather\", \"arguments\": {\"location\": \"San Francisco\", \"unit\": \"celsius\"}}\n\
            </tool_call><|im_end|>\n";
        let wire_after_t1 = format!("{p1}{model_emitted}");

        // Turn 2: client sends the prior assistant turn back in OpenAI canonical
        // form (arguments as a JSON STRING with spaces) plus the tool result.
        let turn2_request: NvCreateChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "model": "qwen3-thinking",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What's the weather in San Francisco?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "I'll call get_weather for SF.",
                        "tool_calls": [{
                            "id": "call_sf",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
                            }
                        }]
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_sf",
                        "content": "{\"temp\": 18, \"conditions\": \"Foggy\"}"
                    }
                ],
                "tools": tools,
            }))
            .unwrap();
        let p2 = formatter.render(&turn2_request).unwrap();

        if !p2.starts_with(&wire_after_t1) {
            // Find first divergence and report it for easy debugging.
            let div = wire_after_t1
                .as_bytes()
                .iter()
                .zip(p2.as_bytes())
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| wire_after_t1.len().min(p2.len()));
            let lo = div.saturating_sub(40);
            panic!(
                "turn-2 prompt is NOT a prefix-extension of [turn-1 + model bytes]\n  \
                 diverges at byte {div}\n  \
                 wire ends: ...{}|{}\n  \
                 t2 has:    ...{}|{}",
                String::from_utf8_lossy(&wire_after_t1.as_bytes()[lo..div]),
                String::from_utf8_lossy(
                    &wire_after_t1.as_bytes()[div..(div + 60).min(wire_after_t1.len())]
                ),
                String::from_utf8_lossy(&p2.as_bytes()[lo..div]),
                String::from_utf8_lossy(&p2.as_bytes()[div..(div + 60).min(p2.len())]),
            );
        }

        // The only new bytes in P2 should be the tool response and the next
        // generation prompt — nothing in the prior conversation should change.
        let suffix = &p2[wire_after_t1.len()..];
        assert!(
            suffix.contains("<tool_response>"),
            "appended bytes must include the tool response, got: {suffix}"
        );
        assert!(
            suffix.ends_with("<|im_start|>assistant\n<think>\n"),
            "appended bytes must end with the next generation prompt, got: {suffix}"
        );
    }
}
