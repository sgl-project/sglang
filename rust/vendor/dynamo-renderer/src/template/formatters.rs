// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::tokcfg::{ChatTemplate, fromjson, raise_exception, strftime_now, tojson};
use super::{ContextMixins, HfTokenizerConfigJsonFormatter, JinjaEnvironment, SystemNormalization};
use either::Either;
use minijinja::{Environment, Value, context};
use serde_json::json;

/// Renders the `default` template with the given `messages` and
/// `add_generation_prompt=false`, returning the output (empty string on any
/// error). Shared probe used by the load-time template-capability detectors
/// (`detect_content_array_usage`, `detect_passthrough_template`).
fn render_default_probe(env: &Environment, messages: serde_json::Value) -> String {
    let ctx = context! {
        messages => messages,
        add_generation_prompt => false,
    };
    env.get_template("default")
        .and_then(|t| t.render(&ctx))
        .unwrap_or_default()
}

/// Passed to probe renders so a template that appends them doesn't raise for a
/// missing token, which would read as a message-shape rejection.
struct ProbeTokens {
    bos: Option<String>,
    eos: Option<String>,
    unk: Option<String>,
}

/// Unlike [`render_default_probe`], surfaces the error: a template's
/// `raise_exception` is the signal that it rejects this message shape.
fn probe_template_raises(
    env: &Environment,
    template_name: &str,
    messages: serde_json::Value,
    tools: &Option<serde_json::Value>,
    tok: &ProbeTokens,
) -> bool {
    let ctx = context! {
        messages => messages,
        add_generation_prompt => false,
        tools => tools,
        bos_token => tok.bos,
        eos_token => tok.eos,
        unk_token => tok.unk,
    };
    match env.get_template(template_name) {
        Ok(t) => t.render(&ctx).is_err(),
        Err(_) => false,
    }
}

/// Detects which message shapes agent clients send that this template rejects,
/// so `render` applies only the rewrites this template needs.
///
/// The non-leading `system` restriction takes two probes because strict families
/// disagree on where they enforce it: Gemma-3 accepts `[system, user, system]`
/// but rejects the same turn after an assistant reply.
fn detect_system_normalization(
    env: &Environment,
    template_name: &str,
    tools: &Option<serde_json::Value>,
    tok: &ProbeTokens,
) -> SystemNormalization {
    let sys = |c: &str| json!({"role": "system", "content": c});
    let usr = |c: &str| json!({"role": "user", "content": c});
    let asst = |c: &str| json!({"role": "assistant", "content": c});

    // A template that can't render even this isn't rejecting shape, it's broken.
    let baseline_ok =
        !probe_template_raises(env, template_name, json!([sys("s"), usr("u")]), tools, tok);
    if !baseline_ok {
        return SystemNormalization::default();
    }
    let nonleading_system = probe_template_raises(
        env,
        template_name,
        json!([sys("s0"), usr("u"), sys("s1")]),
        tools,
        tok,
    );
    let mid_after_assistant = probe_template_raises(
        env,
        template_name,
        json!([sys("s0"), usr("u"), asst("a"), sys("s1"), usr("u2")]),
        tools,
        tok,
    );
    let consecutive_users = probe_template_raises(
        env,
        template_name,
        json!([sys("s"), usr("u0"), usr("u1")]),
        tools,
        tok,
    );
    SystemNormalization {
        demote_nonleading_system: nonleading_system || mid_after_assistant,
        coalesce_consecutive_users: consecutive_users,
    }
}

/// Detects if a template requires content as arrays (multimodal) vs strings (text-only).
/// Returns true if the template only works with array format.
fn detect_content_array_usage(env: &Environment) -> bool {
    let out_array = render_default_probe(
        env,
        json!([{"role": "user", "content": [{"type": "text", "text": "template_test"}]}]),
    );
    let out_string =
        render_default_probe(env, json!([{"role": "user", "content": "template_test"}]));

    // If array works but string doesn't, template requires arrays
    out_array.contains("template_test") && !out_string.contains("template_test")
}

/// Picks an image-placeholder template by sniffing the chat template source
/// for distinctive role/end markers.
///
/// Returned string is a format with `{n}` standing in for the 1-based image
/// index (numbered Phi-3 placeholders) or just a static placeholder (LLaVA).
/// Returns `None` when we don't know a flatten strategy for this template —
/// callers leave the mixed-content array untouched in that case.
///
/// The detection is intentionally narrow: only families whose chat templates
/// concatenate `message.content` with strings (and therefore can't render a
/// content array) need this. Qwen-VL / LLaVA-NeXT iterate `content` natively
/// and don't reach the flatten path at all.
fn detect_image_placeholder_template(env: &Environment) -> Option<&'static str> {
    let src = env
        .get_template("default")
        .ok()
        .map(|t| t.source().to_string())
        .unwrap_or_default();
    // Phi-3-vision template constructs `<|user|>` at runtime via
    // `'<|' + message['role'] + '|>'`, so the literal `<|user|>` never
    // appears in the source. The literals that ARE in the source are
    // `<|end|>` (end-of-turn) and `<|assistant|>` (generation prompt).
    if src.contains("<|end|>") && src.contains("<|assistant|>") {
        return Some("<|image_{n}|>");
    }
    // LLaVA-1.5: USER:/ASSISTANT: convention with `+ message['content']`.
    if src.contains("USER:") && src.contains("ASSISTANT:") {
        return Some("<image>");
    }
    // Pure pass-through templates (e.g. NVIDIA-Nemotron-Parse's
    // `{% for message in messages %}{{ message['content'] }}{% endfor %}`)
    // emit `message.content` verbatim with no role markers or special tokens.
    // These are typically encoder-decoder document models where the image is
    // consumed by the vision encoder out-of-band and contributes NO token to
    // the decoder prompt (the prompt is just the control tokens, e.g.
    // `</s><s><predict_bbox>...`). A mixed text+image content array would
    // otherwise be JSON-serialized into the prompt by `{{ message.content }}`,
    // producing garbage. The empty placeholder makes the flatten path drop the
    // image part from the text while preserving the text parts — exactly what
    // these models expect.
    if detect_passthrough_template(env) {
        return Some("");
    }
    None
}

/// Detects a pure pass-through chat template: one that emits `message.content`
/// verbatim with no role markers, BOS/EOS, or other decoration, AND does not
/// render content arrays natively. Callers use this to pick an empty image
/// placeholder (drop images from the rendered text; the vision encoder consumes
/// them out-of-band — see Nemotron-Parse).
///
/// Two probes against the `default` template (`add_generation_prompt=false`,
/// mirroring `detect_content_array_usage`), both with a control-char sentinel
/// that cannot collide with literal template text:
///  1. string content must round-trip verbatim (the pass-through property), and
///  2. a mixed text+image array must NOT be rendered natively. A pure
///     pass-through stringifies `{{ content }}` on the array (retaining the
///     serialized dict structure); a template with a native array branch emits
///     the text value plus its own image marker. The latter handles images
///     itself, so it must keep the array (return `false` here) rather than get
///     the empty placeholder that would drop the image before its branch runs.
fn detect_passthrough_template(env: &Environment) -> bool {
    const PROBE: &str = "\u{1}dynamo_passthrough_probe\u{1}";
    // (1) String content must pass through verbatim. `.trim()` tolerates a
    // trailing newline some pass-through templates emit.
    let out_string = render_default_probe(env, json!([{"role": "user", "content": PROBE}]));
    if out_string.trim() != PROBE {
        return false;
    }
    // (2) A mixed text+image array must not be rendered natively: a pure
    // pass-through stringifies it (output keeps the `[ ... "type" ...` structure),
    // whereas a native array branch emits the text value + its own image marker.
    let out_mixed = render_default_probe(
        env,
        json!([{"role": "user", "content": [{"type": "text", "text": PROBE}, {"type": "image"}]}]),
    );
    out_mixed.contains('[') && out_mixed.contains("type")
}

/// Remove known non-standard Jinja2 tags from chat templates
///
/// Some models use custom Jinja2 extensions that minijinja doesn't recognize. These tags
/// are typically metadata markers that don't affect the rendered output. For example:
/// - {% generation %} / {% endgeneration %}: Used by vLLM's AssistantTracker to mark
///   assistant-generated content. The tags themselves don't produce output.
///
/// By removing these tags before validation, we allow templates with backend-specific
/// extensions to work with minijinja while maintaining correct output semantics.
///
/// Note: This follows the same approach as Mistral.rs, which also strips these tags
/// for compatibility: https://github.com/EricLBuehler/mistral.rs/blob/2bcf0e9/mistralrs-core/src/pipeline/chat_template.rs#L318-L322
fn remove_known_non_jinja2_tags(template: &str) -> String {
    template
        .replace("{% generation %}", "")
        .replace("{% endgeneration %}", "")
}

/// Normalize common Python/Jinja dict method calls that are ambiguous in minijinja.
///
/// JSON schemas commonly use an `items` key for array item definitions. In
/// minijinja, `foo.items()` can resolve `items` as a map entry before the
/// pycompat method callback sees it, causing "object is not callable" for
/// templates that iterate OpenAI tool schemas. The `items` filter gives the same
/// map iteration behavior without colliding with schema keys.
fn normalize_dict_method_calls(template: &str) -> String {
    let mut out = String::with_capacity(template.len());
    let mut i = 0;

    while i < template.len() {
        if template[i..].starts_with("{#") {
            let Some(end) = find_tag_end(template, i + 2, "#}") else {
                out.push_str(&template[i..]);
                break;
            };
            out.push_str(&template[i..end]);
            i = end;
        } else if template[i..].starts_with("{{") {
            let Some(end) = find_tag_end(template, i + 2, "}}") else {
                out.push_str(&template[i..]);
                break;
            };
            out.push_str("{{");
            out.push_str(&normalize_jinja_code_segment(&template[i + 2..end - 2]));
            out.push_str("}}");
            i = end;
        } else if template[i..].starts_with("{%") {
            let Some(end) = find_tag_end(template, i + 2, "%}") else {
                out.push_str(&template[i..]);
                break;
            };
            let inner = &template[i + 2..end - 2];
            if is_jinja_block_name(inner, "raw") {
                if let Some(raw_end) = find_raw_block_end(template, end) {
                    out.push_str(&template[i..raw_end]);
                    i = raw_end;
                } else {
                    out.push_str(&template[i..]);
                    break;
                }
            } else {
                out.push_str("{%");
                out.push_str(&normalize_jinja_code_segment(inner));
                out.push_str("%}");
                i = end;
            }
        } else {
            let ch = template[i..].chars().next().expect("valid char boundary");
            out.push(ch);
            i += ch.len_utf8();
        }
    }

    out
}

fn find_tag_end(template: &str, start: usize, close: &str) -> Option<usize> {
    template[start..]
        .find(close)
        .map(|relative| start + relative + close.len())
}

fn find_raw_block_end(template: &str, start: usize) -> Option<usize> {
    let mut i = start;
    while let Some(relative_open) = template[i..].find("{%") {
        let open = i + relative_open;
        let end = find_tag_end(template, open + 2, "%}")?;
        if is_jinja_block_name(&template[open + 2..end - 2], "endraw") {
            return Some(end);
        }
        i = end;
    }
    None
}

fn is_jinja_block_name(inner: &str, name: &str) -> bool {
    let trimmed = inner.trim_start();
    let trimmed = trimmed
        .strip_prefix('-')
        .or_else(|| trimmed.strip_prefix('+'))
        .unwrap_or(trimmed)
        .trim_start();
    let Some(rest) = trimmed.strip_prefix(name) else {
        return false;
    };
    rest.chars()
        .next()
        .is_none_or(|ch| ch.is_whitespace() || ch == '-' || ch == '+')
}

fn normalize_jinja_code_segment(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    let mut i = 0;
    let mut quote: Option<char> = None;
    let mut escaped = false;

    while i < segment.len() {
        let ch = segment[i..].chars().next().expect("valid char boundary");

        if let Some(q) = quote {
            out.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == q {
                quote = None;
            }
            i += ch.len_utf8();
        } else if ch == '\'' || ch == '"' {
            quote = Some(ch);
            out.push(ch);
            i += ch.len_utf8();
        } else if segment[i..].starts_with(".items()") {
            out.push_str("|items");
            i += ".items()".len();
        } else {
            out.push(ch);
            i += ch.len_utf8();
        }
    }

    out
}

/// Detects the Gemma4 tool template shape that replays assistant thinking from
/// the upstream/vLLM-style `message.reasoning` field.
fn is_gemma4_reasoning_field_template_source(source: &str) -> bool {
    source.contains("<|channel>thought")
        && source.contains("<|tool_call>call:")
        && (source.contains("message.get('reasoning')")
            || source.contains("message.get(\"reasoning\")"))
        && !source.contains("reasoning_content")
}

/// The stock template has a single `message.reasoning` block before the
/// tool-call loop. Dynamo can carry reasoning as N+1 segments around N tool
/// calls, so the renderer patches that known template shape at load time:
/// segment i is emitted immediately before tool_call i, and the optional
/// trailing segment is emitted after the final tool call.
fn adapt_gemma4_reasoning_template_source(source: &str) -> String {
    const OLD_REASONING_BLOCK: &str = "{%- if message.get('reasoning') and loop.index0 > ns_turn.last_user_idx and message.get('tool_calls') -%}
        {{- '<|channel>thought\\n' + message['reasoning'] + '\\n<channel|>'}}
    {%- endif -%}";
    const NEW_REASONING_BLOCK: &str = "{%- set dyn_gemma4_reasoning_value = message.get('reasoning') or message.get('reasoning_content') -%}
    {%- set dyn_gemma4_reasoning_segments = [] -%}
    {%- if dyn_gemma4_reasoning_value is string -%}
        {%- set dyn_gemma4_reasoning_segments = [dyn_gemma4_reasoning_value] -%}
    {%- elif dyn_gemma4_reasoning_value is sequence -%}
        {%- set dyn_gemma4_reasoning_segments = dyn_gemma4_reasoning_value -%}
    {%- endif -%}
    {%- if dyn_gemma4_reasoning_value and not message.get('tool_calls') -%}
        {%- for dyn_gemma4_reasoning_segment in dyn_gemma4_reasoning_segments -%}
            {%- if dyn_gemma4_reasoning_segment -%}
                {{- '<|channel>thought\\n' + dyn_gemma4_reasoning_segment + '\\n<channel|>'}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}";
    const OLD_TOOL_LOOP_START: &str = "{%- for tool_call in message['tool_calls'] -%}
                    {%- set function = tool_call['function'] -%}";
    const NEW_TOOL_LOOP_START: &str = "{%- for tool_call in message['tool_calls'] -%}
                    {%- set dyn_gemma4_reasoning_segment = dyn_gemma4_reasoning_segments[loop.index0] | default('') -%}
                    {%- if dyn_gemma4_reasoning_segment -%}
                        {{- '<|channel>thought\\n' + dyn_gemma4_reasoning_segment + '\\n<channel|>'}}
                    {%- endif -%}
                    {%- set function = tool_call['function'] -%}";
    const OLD_TOOL_LOOP_END: &str = "{{- '}<tool_call|>' -}}
                {%- endfor -%}";
    const NEW_TOOL_LOOP_END: &str = "{{- '}<tool_call|>' -}}
                {%- endfor -%}
                {%- set dyn_gemma4_trailing_reasoning = dyn_gemma4_reasoning_segments[message['tool_calls'] | length] | default('') -%}
                {%- if dyn_gemma4_trailing_reasoning -%}
                    {{- '<|channel>thought\\n' + dyn_gemma4_trailing_reasoning + '\\n<channel|>'}}
                {%- endif -%}";

    if !source.contains(OLD_REASONING_BLOCK)
        || !source.contains(OLD_TOOL_LOOP_START)
        || !source.contains(OLD_TOOL_LOOP_END)
    {
        return source.to_string();
    }

    source
        .replace(OLD_REASONING_BLOCK, NEW_REASONING_BLOCK)
        .replace(OLD_TOOL_LOOP_START, NEW_TOOL_LOOP_START)
        .replace(OLD_TOOL_LOOP_END, NEW_TOOL_LOOP_END)
}

fn normalize_chat_template_source(source: &str) -> String {
    let source = normalize_dict_method_calls(&remove_known_non_jinja2_tags(source));

    if is_gemma4_reasoning_field_template_source(&source) {
        adapt_gemma4_reasoning_template_source(&source)
    } else {
        source
    }
}

impl JinjaEnvironment {
    fn env(self) -> Environment<'static> {
        self.env
    }
}

impl Default for JinjaEnvironment {
    fn default() -> Self {
        let mut env = Environment::new();

        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);

        JinjaEnvironment { env }
    }
}

impl HfTokenizerConfigJsonFormatter {
    #[cfg(test)]
    pub fn new(config: ChatTemplate, mixins: ContextMixins) -> anyhow::Result<Self> {
        Self::with_options(config, mixins, true)
    }

    pub fn with_options(
        config: ChatTemplate,
        mixins: ContextMixins,
        exclude_tools_when_tool_choice_none: bool,
    ) -> anyhow::Result<Self> {
        let mut env = JinjaEnvironment::default().env();

        let chat_template = config.chat_template.as_ref().ok_or(anyhow::anyhow!(
            "chat_template field is required in the tokenizer_config.json file"
        ))?;

        // Safely handle chat templates that check the length of arguments like `tools` even
        // when `tools=None` when rendered through minijinja. For example:
        // https://github.com/vllm-project/vllm/blob/d95d0f4b985f28ea381e301490f9d479b34d8980/examples/tool_chat_template_hermes.jinja#L36
        env.add_filter("length", |value: Value| -> usize {
            use minijinja::value::ValueKind;
            match value.kind() {
                ValueKind::Undefined | ValueKind::None => 0,
                _ => value.len().unwrap_or(0),
            }
        });

        // add pycompat
        // todo: should we use this: minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

        env.add_filter("tojson", tojson);

        // Templates that round-trip tool call `arguments` (a JSON string) back into an
        // object need this; minijinja has no builtin. Both spellings are in the wild.
        env.add_filter("fromjson", fromjson);
        env.add_filter("from_json", fromjson);

        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);

        let mut supports_add_generation_prompt = None;

        match &chat_template.0 {
            Either::Left(x) => {
                if x.contains("add_generation_prompt") {
                    tracing::debug!(
                        "Chat template contains `add_generation_prompt` key. This model supports add_generation_prompt."
                    );
                    supports_add_generation_prompt = Some(true);
                }
                // Remove known non-standard tags before validation (they don't affect output)
                let template_cleaned = normalize_chat_template_source(x);
                env.add_template_owned("default", template_cleaned.clone())?;
                env.add_template_owned("tool_use", template_cleaned)?;
            }
            Either::Right(map) => {
                for t in map {
                    for (k, v) in t.iter() {
                        if v.contains("add_generation_prompt") {
                            match supports_add_generation_prompt {
                                Some(true) | None => {
                                    tracing::debug!(
                                        "Chat template contains `add_generation_prompt` key. This model supports add_generation_prompt."
                                    );
                                    supports_add_generation_prompt = Some(true);
                                }
                                Some(false) => {
                                    tracing::warn!(
                                        "Not all templates contain `add_generation_prompt` key. This model does not support add_generation_prompt."
                                    );
                                }
                            }
                        } else {
                            supports_add_generation_prompt = Some(false);
                        }
                        // Remove known non-standard tags before validation (they don't affect output)
                        let template_cleaned = normalize_chat_template_source(v);
                        env.add_template_owned(k.to_string(), template_cleaned)?;
                    }
                }
                if env.templates().count() == 0 {
                    anyhow::bail!(
                        "Chat template does not contain a `tool_use` or `default` key. Please ensure it contains at least a `default` key, although `tool_use` should be specified for using tools."
                    );
                }
            }
        }

        // Detect at model load time whether this template requires content arrays
        let requires_content_arrays = detect_content_array_usage(&env);

        // Pick a per-family placeholder for the mixed-content → string flatten
        // path. `None` is the safe default — the existing behavior in
        // `may_be_fix_msg_content` leaves mixed arrays untouched.
        let image_placeholder_template = if requires_content_arrays {
            None
        } else {
            detect_image_placeholder_template(&env)
        };

        // Detect if a given template natively handles reasoning_content (e.g.
        // Nemotron, Qwen3, or a Gemma4 tool template adapted by
        // `normalize_chat_template_source`). If so, we must NOT inject <think>
        // blocks for that template — it renders reasoning itself. Per-template
        // (default vs tool_use) because HF configs can register different sources
        // for each: Gemma4 adapts only its `tool_use` template to read
        // `reasoning_content`, so a global `any()` flag would wrongly suppress
        // injection on the untouched `default` path and silently drop reasoning.
        let template_handles_reasoning = |name: &str| -> bool {
            env.templates()
                .find(|(n, _)| *n == name)
                .map(|(_, tmpl)| tmpl.source().contains("reasoning_content"))
                .unwrap_or(false)
        };
        let default_template_handles_reasoning = template_handles_reasoning("default");
        let tool_use_template_handles_reasoning = template_handles_reasoning("tool_use");

        // Detect if a given template branches on `tool_call.arguments is string` (Qwen3, Hermes).
        // Such templates render a JSON-string `arguments` field verbatim; if we pre-parse
        // it into an object, the `tojson` branch fires instead and emits compact JSON,
        // breaking byte-level append-only across multi-step tool-use turns. The check is
        // per-template (default vs tool_use) because in HF configs they can differ — and
        // because `arguments is string` only appears inside tool-call iteration, the flag
        // is naturally tied to the `tool_use` template in practice. It is also
        // tool_calls-specific: legacy `function_call.arguments` lives outside this branch
        // and must still be normalized.
        let template_handles_args_string = |name: &str| -> bool {
            env.templates()
                .find(|(n, _)| *n == name)
                .map(|(_, tmpl)| tmpl.source().contains("arguments is string"))
                .unwrap_or(false)
        };
        let default_template_handles_tool_calls_arguments_string =
            template_handles_args_string("default");
        let tool_use_template_handles_tool_calls_arguments_string =
            template_handles_args_string("tool_use");

        let probe_tokens = ProbeTokens {
            bos: config.bos_tok(),
            eos: config.eos_tok(),
            unk: config.unk_tok(),
        };
        let default_probe_tools = Option::<serde_json::Value>::None;
        let tool_use_probe_tools = Some(json!([{
            "type": "function",
            "function": {
                "name": "probe",
                "description": "",
                "parameters": {"type": "object", "properties": {}}
            }
        }]));
        let default_system_normalization =
            detect_system_normalization(&env, "default", &default_probe_tools, &probe_tokens);
        let tool_use_system_normalization =
            detect_system_normalization(&env, "tool_use", &tool_use_probe_tools, &probe_tokens);

        Ok(HfTokenizerConfigJsonFormatter {
            env,
            config,
            mixins: Arc::new(mixins),
            supports_add_generation_prompt: supports_add_generation_prompt.unwrap_or(false),
            requires_content_arrays,
            exclude_tools_when_tool_choice_none,
            default_template_handles_reasoning,
            tool_use_template_handles_reasoning,
            image_placeholder_template,
            default_template_handles_tool_calls_arguments_string,
            tool_use_template_handles_tool_calls_arguments_string,
            default_system_normalization,
            tool_use_system_normalization,
        })
    }
}

// impl JinjaEnvironment {
//     /// Renders the template with the provided messages.
//     /// This function reuses the pre-compiled template for efficiency.
//     pub fn render(&self, template_id: &str, ctx: &dyn erased_serde::Serialize) -> Result<String> {
//         let tmpl = self.env.get_template(template_id)?;
//         Ok(tmpl.render(ctx)?)
//     }

//     // fn apply_tool_template()
// }

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds the same `default`-template env the production renderer uses
    /// (`JinjaEnvironment::default()`), with `src` registered as `default`.
    fn env_with_default(src: &str) -> Environment<'static> {
        let mut env = JinjaEnvironment::default().env();
        env.add_template_owned("default", src.to_string()).unwrap();
        env
    }

    #[test]
    fn test_remove_known_non_jinja2_tags() {
        let template =
            "USER: {{ message }} ASSISTANT: {% generation %}Reply here{% endgeneration %}";
        let result = remove_known_non_jinja2_tags(template);
        assert_eq!(result, "USER: {{ message }} ASSISTANT: Reply here");
    }

    #[test]
    fn test_remove_known_non_jinja2_tags_preserves_standard_tags() {
        let template = "{% for item in items %}{{ item }}{% endfor %}";
        let result = remove_known_non_jinja2_tags(template);
        assert_eq!(result, template);
    }

    #[test]
    fn test_remove_known_non_jinja2_tags_multiple() {
        let template = "Start {% generation %}Part 1{% endgeneration %} middle {% generation %}Part 2{% endgeneration %}";
        let result = remove_known_non_jinja2_tags(template);
        assert_eq!(result, "Start Part 1 middle Part 2");
    }

    /// NVIDIA-Nemotron-Parse ships a pure pass-through chat template
    /// (`{% for message in messages %}{{ message['content'] }}{% endfor %}`).
    /// It must be detected as: (a) not requiring content arrays, and
    /// (b) using an empty image placeholder, so a mixed text+image request
    /// flattens to the text-only control-token prompt instead of being
    /// JSON-serialized into the prompt.
    #[test]
    fn test_detect_nemotron_parse_passthrough_template() {
        let env =
            env_with_default("{% for message in messages %}{{ message['content'] }}{% endfor %}");

        assert!(
            detect_passthrough_template(&env),
            "pure pass-through template should be detected"
        );
        assert!(
            !detect_content_array_usage(&env),
            "pass-through template renders string content fine, so does not require arrays"
        );
        assert_eq!(
            detect_image_placeholder_template(&env),
            Some(""),
            "pass-through template should flatten images to an empty placeholder"
        );
    }

    /// A decorated template (role markers / special tokens added around
    /// content) is NOT a pass-through and must not get the empty placeholder.
    #[test]
    fn test_decorated_template_is_not_passthrough() {
        let env = env_with_default(
            "{% for message in messages %}<|{{ message['role'] }}|>{{ message['content'] }}<|end|>{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}",
        );

        assert!(
            !detect_passthrough_template(&env),
            "template that wraps content in role markers is not pass-through"
        );
    }

    /// A template that passes *string* content through verbatim but also has a
    /// native *array* branch emitting an image marker must NOT be treated as
    /// pass-through: its array branch renders images itself, so flattening to an
    /// empty placeholder would drop the image before that branch runs. The
    /// mixed-array probe distinguishes it from a pure pass-through.
    #[test]
    fn test_string_passthrough_with_native_array_branch_is_not_passthrough() {
        let env = env_with_default(
            "{% for message in messages %}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for part in message['content'] %}{% if part['type'] == 'image' %}<image>{% else %}{{ part['text'] }}{% endif %}{% endfor %}{% endif %}{% endfor %}",
        );

        assert!(
            !detect_passthrough_template(&env),
            "template with a native content-array branch is not pass-through"
        );
        assert_eq!(
            detect_image_placeholder_template(&env),
            None,
            "template that natively renders image markers must keep the content array"
        );
    }

    #[test]
    fn test_normalize_dict_method_calls_rewrites_items_method() {
        let template = "{% for k, v in tool.parameters.properties.items() %}{{ k }}{% endfor %}";
        let result = normalize_dict_method_calls(template);
        assert_eq!(
            result,
            "{% for k, v in tool.parameters.properties|items %}{{ k }}{% endfor %}"
        );
    }

    #[test]
    fn test_normalize_dict_method_calls_rewrites_expression_items_method() {
        let template = "{{ tool.parameters.properties.items() }}";
        let result = normalize_dict_method_calls(template);
        assert_eq!(result, "{{ tool.parameters.properties|items }}");
    }

    #[test]
    fn test_normalize_dict_method_calls_preserves_literal_text() {
        let template = "Do not rewrite literal .items() text.";
        let result = normalize_dict_method_calls(template);
        assert_eq!(result, template);
    }

    #[test]
    fn test_normalize_dict_method_calls_preserves_comments_raw_and_strings() {
        let template = concat!(
            "{# comment .items() #}",
            "{% raw %}{{ tool.parameters.properties.items() }}{% endraw %}",
            "{{ '.items()' }}",
            "{{ \".items()\" }}",
        );
        let result = normalize_dict_method_calls(template);
        assert_eq!(result, template);
    }

    #[test]
    fn test_normalize_dict_method_calls_avoids_schema_items_collision() {
        let template = normalize_dict_method_calls(
            "{% for param_name, param_spec in tool.parameters.properties.items() %}{{ param_name }}={{ param_spec.type }};{% endfor %}",
        );

        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template("t", &template).unwrap();

        let tool = json!({
            "parameters": {
                "properties": {
                    "items": {"type": "array", "items": {"type": "object"}},
                    "message": {"type": "string"}
                }
            }
        });

        let out = env
            .get_template("t")
            .unwrap()
            .render(context! { tool => tool })
            .unwrap();

        assert!(out.contains("items=array;"));
        assert!(out.contains("message=string;"));
    }

    #[test]
    fn test_minijinja_parses_midchain_dotted_integer_lookup() {
        let chat_template: ChatTemplate = serde_json::from_value(serde_json::json!({
            "chat_template": r#"{{ m.content.0.type }} {{ "1.5.10" }}"#,
        }))
        .unwrap();

        let formatter =
            HfTokenizerConfigJsonFormatter::new(chat_template, ContextMixins::new(&[])).unwrap();

        let result = formatter
            .env
            .get_template("default")
            .unwrap()
            .render(context! {
                m => json!({
                    "content": [
                        {
                            "type": "tool_reference"
                        }
                    ]
                })
            })
            .unwrap();

        assert_eq!(result, "tool_reference 1.5.10");
    }
}
