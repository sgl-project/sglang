//! Kimi-K3's structural tags for `tool_choice` `"auto"`
//! (`get_kimik3_auto_tool_call_structural_tag`) and for `"required"`, a named
//! tool or `"none"` (`get_kimik3_structural_tag`), from
//! `function_call/kimik3_structural_tag.py`, dumped as xgrammar's
//! `StructuralTag.model_dump(by_alias=True)`.

use serde_json::{Map, Value, json};

use super::Unsupported;

use crate::kimi_k3_format::{
    CALL_CLOSE, CALL_OPEN, MESSAGE_CLOSE, RESPONSE_CLOSE, RESPONSE_OPEN, THINK_CLOSE, THINK_OPEN,
    TOOLS_CLOSE, TOOLS_OPEN,
};

const OPEN_TOKEN: &str = "<|open|>";
const CLOSE_TOKEN: &str = "<|close|>";
const ARGUMENT_CLOSE: &str = "<|close|>argument<|sep|>";

/// `_JSON_TO_XTML_TYPE`: the XTML `type` attribute of each JSON type.
const XTML_TYPES: [(&str, &str); 7] = [
    ("string", "string"),
    ("number", "number"),
    ("integer", "number"),
    ("boolean", "boolean"),
    ("array", "array"),
    ("object", "object"),
    ("null", "null"),
];

const JSON_TYPES: [&str; 7] = [
    "string", "number", "integer", "boolean", "array", "object", "null",
];
/// The schema keywords `_schema_types` reads a type from, in its order.
const TYPE_KEYWORDS: [(&str, &str); 4] = [
    (
        "object",
        "additionalProperties dependentRequired dependentSchemas maxProperties minProperties \
         patternProperties properties propertyNames required",
    ),
    (
        "array",
        "contains items maxContains maxItems minContains minItems prefixItems uniqueItems",
    ),
    ("string", "format maxLength minLength pattern"),
    (
        "number",
        "exclusiveMaximum exclusiveMinimum maximum minimum multipleOf",
    ),
];

/// How deep and how long `_schema_types` may run on one argument. Python
/// recurses until its recursion limit and then drops the tag; a schema past
/// these bounds is left to the engine, which does exactly that.
const MAX_SCHEMA_DEPTH: usize = 64;
const MAX_SCHEMA_STEPS: usize = 10_000;

/// The auto tag over `tools` (request tool definitions), without the `strip`
/// tokens `strip_structural_tag_excludes` removes; `None` without tools. Only
/// the trigger's excludes hold think tokens.
pub(super) fn auto_tool_call_tag(
    tools: &[Value],
    parallel_tool_calls: bool,
    strip: &[&str],
) -> Result<Option<Value>, Unsupported> {
    if tools.is_empty() {
        return Ok(None);
    }
    let call_tags = tools
        .iter()
        .map(|tool| {
            let function = &tool["function"];
            auto_arguments_format(&function["parameters"])
                .map(|arguments| tool_call_tag(function, arguments))
        })
        .collect::<Option<_>>()
        .ok_or(Unsupported("tool_definition"))?;
    let tools_tag = tool_calls_tag(call_tags, parallel_tool_calls);
    // `_auto_suffix`: a retriggered second tools section would evade the single-call limit.
    let excludes: Vec<&str> = [THINK_OPEN, THINK_CLOSE, CALL_OPEN]
        .into_iter()
        .filter(|token| !strip.contains(token))
        .collect();
    let triggered = json!({
        "type": "triggered_tags",
        "triggers": [TOOLS_OPEN],
        "tags": [tools_tag],
        "at_least_one": false,
        "stop_after_first": !parallel_tool_calls,
        "excludes": excludes,
    });
    // A serial grammar completes with the tools section, so it must still
    // admit the message close K3 emits before EOS.
    let format = if parallel_tool_calls {
        triggered
    } else {
        sequence(vec![triggered, optional_message_close()])
    };
    Ok(Some(json!({"type": "structural_tag", "format": format})))
}

/// A `tool_choice` that forces a call (`_select_tools`): `"required"` keeps
/// every tool, a named choice only the tools of that name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ForcedChoice<'a> {
    Required,
    Named(&'a str),
}

/// `get_kimik3_structural_tag` for a forced choice over non-strict tools: an
/// optional response, then a tools section of at least one call to a selected
/// tool. `None` when the choice selects no tool.
///
/// SGLang passes `thinking_mode` only when no reasoning parser owns the think
/// section, and without one `_get_reasoning_from_request` is false, so the
/// lowered path never prefixes `_with_reasoning`'s think tag. No free-text
/// `excludes` here holds a think token, so `strip_structural_tag_excludes`
/// leaves the tag as it is.
pub(super) fn forced_tool_call_tag(
    tools: &[Value],
    choice: ForcedChoice<'_>,
    parallel_tool_calls: bool,
) -> Option<Value> {
    let call_tags: Vec<Value> = tools
        .iter()
        .map(|tool| &tool["function"])
        .filter(|function| match choice {
            ForcedChoice::Required => true,
            ForcedChoice::Named(name) => function["name"] == name,
        })
        // `_tool_arguments_format` of a non-strict tool: any arguments, strings loose.
        .map(|function| {
            tool_call_tag(
                function,
                json!({"type": "star", "content": dynamic_argument_format()}),
            )
        })
        .collect();
    if call_tags.is_empty() {
        return None;
    }
    let format = sequence(vec![
        // Only a response may precede the tools section, so a model that
        // decides no tool is needed cannot close the message or end the turn
        // without making the call.
        json!({"type": "optional", "content": response(any_text(&[OPEN_TOKEN, CLOSE_TOKEN]))}),
        tool_calls_tag(call_tags, parallel_tool_calls),
        optional_message_close(),
    ]);
    Some(json!({"type": "structural_tag", "format": format}))
}

/// `get_kimik3_structural_tag` for `tool_choice: "none"`: the K3 prompt still
/// lists the tools, so the output is held to a response with no tools section,
/// and a nonempty one, since a model that wanted a tool otherwise closes an
/// empty response block and the client gets no content.
pub(super) fn none_tool_call_tag() -> Value {
    // `_NONEMPTY_RESPONSE`: one character that opens neither control token,
    // then free text.
    let text = sequence(vec![
        json!({"type": "regex", "pattern": "[^ \\t\\r\\n<]|<[^|<]"}),
        any_text(&[OPEN_TOKEN, CLOSE_TOKEN]),
    ]);
    let format = sequence(vec![response(text), optional_message_close()]);
    json!({"type": "structural_tag", "format": format})
}

/// `_dynamic_argument_format(True, ..., loose_strings=True)`: an argument of
/// any key and any JSON type, a string one as free text.
fn dynamic_argument_format() -> Value {
    let variants: Vec<Value> = XTML_TYPES
        .iter()
        .map(|(json_type, xtml_type)| {
            // `_value_format`: the restricted schema of `True` is the bare type.
            let value = match *json_type {
                "string" => any_text(&[]),
                _ => json!({
                    "type": "json_schema",
                    "json_schema": {"type": json_type},
                    "style": "json",
                    "any_order": false,
                    "max_whitespace_cnt": null,
                }),
            };
            sequence(vec![
                json!({"type": "regex", "pattern": "[^\"& \\t\\r\\n\\f\\v=<>]+"}),
                json!({"type": "const_string", "value": format!("\" type=\"{xtml_type}\"<|sep|>")}),
                value,
            ])
        })
        .collect();
    json!({
        "type": "tag",
        "begin": "<|open|>argument key=\"",
        "content": {"type": "or", "elements": variants},
        "end": ARGUMENT_CLOSE,
    })
}

/// `_response`: a response of `text`, which admits no control token, and
/// without thinking the prompt already opens the response channel, so the
/// output may start inside it.
fn response(text: Value) -> Value {
    json!({"type": "or", "elements": [
        {"type": "tag", "begin": RESPONSE_OPEN, "content": &text, "end": RESPONSE_CLOSE},
        {"type": "tag", "begin": "", "content": text, "end": RESPONSE_CLOSE},
    ]})
}

/// `_OPTIONAL_MESSAGE_CLOSE`: K3 emits the message close before EOS, and EOS
/// is masked until the grammar completes.
fn optional_message_close() -> Value {
    json!({"type": "optional", "content": {"type": "const_string", "value": MESSAGE_CLOSE}})
}

fn escape_attr(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}

fn any_text(excludes: &[&str]) -> Value {
    json!({"type": "any_text", "excludes": excludes, "max_tokens": null, "max_chars": null})
}

fn sequence(elements: Vec<Value>) -> Value {
    json!({"type": "sequence", "elements": elements})
}

/// `_tool_call_tag`: one call to `function` whose arguments take `arguments`.
fn tool_call_tag(function: &Value, arguments: Value) -> Value {
    let name = escape_attr(function["name"].as_str().unwrap_or_default());
    json!({
        "type": "tag",
        "begin": format!("{CALL_OPEN} tool=\"{name}\" index=\""),
        "content": sequence(vec![
            json!({"type": "regex", "pattern": "[1-9][0-9]*"}),
            json!({"type": "const_string", "value": "\"<|sep|>"}),
            arguments,
        ]),
        "end": CALL_CLOSE,
    })
}

/// `_tool_calls_tag`.
fn tool_calls_tag(call_tags: Vec<Value>, parallel_tool_calls: bool) -> Value {
    let content = if parallel_tool_calls {
        json!({"type": "tags_with_separator", "tags": call_tags, "separator": "", "at_least_one": true, "stop_after_first": false})
    } else if call_tags.len() == 1 {
        call_tags.into_iter().next().expect("one tag")
    } else {
        json!({"type": "or", "elements": call_tags})
    };
    json!({"type": "tag", "begin": TOOLS_OPEN, "content": content, "end": TOOLS_CLOSE})
}

/// `_nonempty_argument_format`: a token-based end keeps the first close
/// token out of both content formats.
fn nonempty_argument_format(key: &str, xtml_type: &str) -> Value {
    let argument = json!({
        "type": "tag",
        "begin": format!("<|open|>argument key=\"{}\" type=\"{xtml_type}\"<|sep|>", escape_attr(key)),
        "content": sequence(vec![
            json!({"type": "exclude_token", "exclude_tokens": []}),
            json!({"type": "any_tokens", "exclude_tokens": [], "max_tokens": null}),
        ]),
        "end": {"type": "token", "token": CLOSE_TOKEN},
    });
    sequence(vec![
        argument,
        json!({"type": "const_string", "value": "argument<|sep|>"}),
    ])
}

/// `_auto_tool_arguments_format`: required arguments present and non-empty,
/// then any text; `None` past the schema bounds.
fn auto_arguments_format(parameters: &Value) -> Option<Value> {
    let loose = || Some(any_text(&[]));
    let Value::Object(schema) = parameters else {
        return loose();
    };
    let empty = Value::Object(Map::new());
    let properties = schema.get("properties").unwrap_or(&empty);
    let no_required = Value::Array(Vec::new());
    let required = schema.get("required").unwrap_or(&no_required);
    let (Value::Object(properties), Value::Array(required)) = (properties, required) else {
        return loose();
    };
    let mut elements = Vec::new();
    for key in required {
        let Some(key) = key.as_str() else {
            return loose();
        };
        let Some(property) = properties
            .get(key)
            .filter(|s| s.is_boolean() || s.is_object())
        else {
            return loose();
        };
        let Some(xtml_type) = single_xtml_type(property, parameters)? else {
            return loose();
        };
        elements.push(nonempty_argument_format(key, xtml_type));
    }
    if elements.is_empty() {
        let mut alternatives = Vec::new();
        for (key, property) in properties {
            if (property.is_boolean() || property.is_object())
                && let Some(xtml_type) = single_xtml_type(property, parameters)?
            {
                alternatives.push(nonempty_argument_format(key, xtml_type));
            }
        }
        match alternatives.len() {
            0 => return loose(),
            1 => elements.push(alternatives.remove(0)),
            _ => elements.push(json!({"type": "or", "elements": alternatives})),
        }
    }
    elements.push(any_text(&[CALL_OPEN, CALL_CLOSE]));
    Some(sequence(elements))
}

/// `_single_xtml_type`; the outer `None` is past the schema bounds.
fn single_xtml_type(schema: &Value, root: &Value) -> Option<Option<&'static str>> {
    let mut steps = 0;
    Some(
        match schema_types(schema, root, &[], &mut steps)?.as_slice() {
            ["integer"] => Some("number"),
            [only] => Some(*only),
            _ => None,
        },
    )
}

/// `_json_type`.
fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) if n.is_f64() => "number",
        Value::Number(_) => "integer",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// `_resolve_local_ref`.
fn resolve_local_ref<'a>(reference: &str, root: &'a Value) -> Option<&'a Value> {
    let path = reference.strip_prefix("#/")?;
    let mut value = root;
    for part in path.split('/') {
        let key = part.replace("~1", "/").replace("~0", "~");
        value = value.as_object()?.get(&key)?;
    }
    (value.is_boolean() || value.is_object()).then_some(value)
}

/// The JSON types in `JSON_TYPES` order that are members of `set`.
fn ordered(set: impl Fn(&str) -> bool) -> Vec<&'static str> {
    JSON_TYPES.into_iter().filter(|t| set(t)).collect()
}

/// `_schema_types`; `seen` holds the `$ref`s already followed on this path.
/// `None` past `MAX_SCHEMA_DEPTH` or `MAX_SCHEMA_STEPS`.
fn schema_types(
    schema: &Value,
    root: &Value,
    seen: &[&str],
    steps: &mut usize,
) -> Option<Vec<&'static str>> {
    *steps += 1;
    if seen.len() > MAX_SCHEMA_DEPTH || *steps > MAX_SCHEMA_STEPS {
        return None;
    }
    let schema = match schema {
        Value::Bool(false) => return Some(Vec::new()),
        Value::Object(schema) => schema,
        _ => return Some(JSON_TYPES.to_vec()),
    };
    if let Some(Value::String(reference)) = schema.get("$ref")
        && !seen.contains(&reference.as_str())
        && let Some(target) = resolve_local_ref(reference, root)
    {
        let seen = [seen, &[reference.as_str()]].concat();
        return schema_types(target, root, &seen, steps);
    }
    // Nesting without `$ref`s also counts toward the depth.
    let nested = [seen, &[""]].concat();
    let mut branches = |options: &Vec<Value>| -> Option<Vec<Vec<&'static str>>> {
        options
            .iter()
            .filter(|o| o.is_boolean() || o.is_object())
            .map(|o| schema_types(o, root, &nested, steps))
            .collect()
    };
    match schema.get("type") {
        Some(Value::String(t)) => {
            return Some(match JSON_TYPES.iter().find(|j| *j == t) {
                Some(t) => vec![*t],
                None => JSON_TYPES.to_vec(),
            });
        }
        Some(Value::Array(types)) => return Some(ordered(|t| types.iter().any(|v| v == t))),
        _ => {}
    }
    for keyword in ["anyOf", "oneOf"] {
        if let Some(Value::Array(options)) = schema.get(keyword) {
            let types = branches(options)?.concat();
            return Some(ordered(|t| types.contains(&t)));
        }
    }
    if let Some(Value::Array(options)) = schema.get("allOf") {
        let widen = |types: Vec<&'static str>| -> Vec<&'static str> {
            let mut types = types;
            if types.contains(&"number") && !types.contains(&"integer") {
                types.push("integer");
            }
            types
        };
        let constrained: Vec<Vec<&'static str>> = branches(options)?
            .into_iter()
            .filter(|types| ordered(|t| types.contains(&t)) != JSON_TYPES.to_vec())
            .collect();
        if let Some((first, rest)) = constrained.split_first() {
            let mut result = widen(first.clone());
            for types in rest {
                let widened = widen(types.clone());
                result.retain(|t| widened.contains(t));
            }
            // number survives only if every branch allows it, making integer redundant.
            if result.contains(&"number") {
                result.retain(|t| *t != "integer");
            }
            return Some(ordered(|t| result.contains(&t)));
        }
    }
    if let Some(value) = schema.get("const") {
        return Some(vec![json_type(value)]);
    }
    if let Some(Value::Array(values)) = schema.get("enum") {
        let types: Vec<&str> = values.iter().map(json_type).collect();
        return Some(ordered(|t| types.contains(&t)));
    }
    Some(
        TYPE_KEYWORDS
            .iter()
            .find(|(_, keywords)| keywords.split_whitespace().any(|k| schema.contains_key(k)))
            .map_or_else(|| JSON_TYPES.to_vec(), |(json_type, _)| vec![*json_type]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tool whose required argument is `hops + 1` `$ref`s from its type, or
    /// with `fanout`, `hops` levels that each refer twice to the next.
    fn ref_tool(hops: usize, fanout: bool) -> Value {
        let mut defs: Map<String, Value> = (0..hops)
            .map(|i| {
                let next = json!({"$ref": format!("#/$defs/D{}", i + 1)});
                let def = match fanout {
                    true => json!({"anyOf": [next, next]}),
                    false => next,
                };
                (format!("D{i}"), def)
            })
            .collect();
        defs.insert(format!("D{hops}"), json!({"type": "integer"}));
        json!({"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"x": {"$ref": "#/$defs/D0"}},
            "required": ["x"], "$defs": defs,
        }}})
    }

    /// `$ref` chains and fan-outs past the bounds go to the engine, on a small stack.
    #[test]
    fn schema_types_are_bounded() {
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let tag = |tool| auto_tool_call_tag(&[tool], true, &[]);
                assert!(matches!(
                    tag(ref_tool(MAX_SCHEMA_DEPTH - 1, false)),
                    Ok(Some(_))
                ));
                assert!(tag(ref_tool(MAX_SCHEMA_DEPTH, false)).is_err());
                assert!(tag(ref_tool(10_000, false)).is_err());
                // Within the depth bound, 2^30 paths exhaust the steps.
                assert!(tag(ref_tool(30, true)).is_err());
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
