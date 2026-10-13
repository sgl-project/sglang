//! Kimi-K3's `tool_choice: "auto"` structural tag
//! (`get_kimik3_auto_tool_call_structural_tag` in
//! `function_call/kimik3_structural_tag.py`), dumped as xgrammar's
//! `StructuralTag.model_dump(by_alias=True)`.

use serde_json::{Map, Value, json};

const THINK_OPEN: &str = "<|open|>think<|sep|>";
const THINK_CLOSE: &str = "<|close|>think<|sep|>";
const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";
const TOOLS_CLOSE: &str = "<|close|>tools<|sep|>";
const CALL_OPEN: &str = "<|open|>call";
const CALL_CLOSE: &str = "<|close|>call<|sep|>";
const CLOSE_TOKEN: &str = "<|close|>";

const JSON_TYPES: [&str; 7] = [
    "string", "number", "integer", "boolean", "array", "object", "null",
];
const STRING_KEYWORDS: [&str; 4] = ["format", "maxLength", "minLength", "pattern"];
const NUMBER_KEYWORDS: [&str; 5] = [
    "exclusiveMaximum",
    "exclusiveMinimum",
    "maximum",
    "minimum",
    "multipleOf",
];
const ARRAY_KEYWORDS: [&str; 8] = [
    "contains",
    "items",
    "maxContains",
    "maxItems",
    "minContains",
    "minItems",
    "prefixItems",
    "uniqueItems",
];
const OBJECT_KEYWORDS: [&str; 9] = [
    "additionalProperties",
    "dependentRequired",
    "dependentSchemas",
    "maxProperties",
    "minProperties",
    "patternProperties",
    "properties",
    "propertyNames",
    "required",
];

/// The auto tag over `tools` (request tool definitions), with `strip` removed
/// from every `excludes` list (`strip_structural_tag_excludes`); `None` without tools.
pub(super) fn auto_tool_call_tag(
    tools: &[Value],
    parallel_tool_calls: bool,
    strip: &[&str],
) -> Option<Value> {
    if tools.is_empty() {
        return None;
    }
    let call_tags = tools
        .iter()
        .map(|tool| tool_call_tag(&tool["function"]))
        .collect();
    let tools_tag = tool_calls_tag(call_tags, parallel_tool_calls);
    // `_auto_suffix`: a retriggered second tools section would evade the single-call limit.
    let mut format = json!({
        "type": "triggered_tags",
        "triggers": [TOOLS_OPEN],
        "tags": [tools_tag],
        "at_least_one": false,
        "stop_after_first": !parallel_tool_calls,
        "excludes": [THINK_OPEN, THINK_CLOSE, CALL_OPEN],
    });
    strip_excludes(&mut format, strip);
    Some(json!({"type": "structural_tag", "format": format}))
}

fn strip_excludes(node: &mut Value, tokens: &[&str]) {
    match node {
        Value::Array(items) => items
            .iter_mut()
            .for_each(|item| strip_excludes(item, tokens)),
        Value::Object(map) => {
            if let Some(Value::Array(excludes)) = map.get_mut("excludes") {
                excludes.retain(|s| !s.as_str().is_some_and(|s| tokens.contains(&s)));
            }
            for key in ["format", "content", "tags", "elements"] {
                if let Some(child) = map.get_mut(key) {
                    strip_excludes(child, tokens);
                }
            }
        }
        _ => {}
    }
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

/// `_tool_call_tag` with `_auto_tool_arguments_format`.
fn tool_call_tag(function: &Value) -> Value {
    let name = escape_attr(function["name"].as_str().unwrap_or_default());
    json!({
        "type": "tag",
        "begin": format!("{CALL_OPEN} tool=\"{name}\" index=\""),
        "content": sequence(vec![
            json!({"type": "regex", "pattern": "[1-9][0-9]*"}),
            json!({"type": "const_string", "value": "\"<|sep|>"}),
            auto_arguments_format(&function["parameters"]),
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

/// `_auto_tool_arguments_format`: required arguments present and non-empty, then any text.
fn auto_arguments_format(parameters: &Value) -> Value {
    let loose = || any_text(&[]);
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
        let Some(xtml_type) = single_xtml_type(property, parameters) else {
            return loose();
        };
        elements.push(nonempty_argument_format(key, xtml_type));
    }
    if elements.is_empty() {
        let mut alternatives: Vec<Value> = properties
            .iter()
            .filter(|(_, s)| s.is_boolean() || s.is_object())
            .filter_map(|(key, s)| {
                Some(nonempty_argument_format(
                    key,
                    single_xtml_type(s, parameters)?,
                ))
            })
            .collect();
        match alternatives.len() {
            0 => return loose(),
            1 => elements.push(alternatives.remove(0)),
            _ => elements.push(json!({"type": "or", "elements": alternatives})),
        }
    }
    elements.push(any_text(&[CALL_OPEN, CALL_CLOSE]));
    sequence(elements)
}

/// `_single_xtml_type`.
fn single_xtml_type(schema: &Value, root: &Value) -> Option<&'static str> {
    match schema_types(schema, root, &mut Vec::new()).as_slice() {
        ["integer"] => Some("number"),
        [only] => Some(only),
        _ => None,
    }
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
fn schema_types(schema: &Value, root: &Value, seen: &mut Vec<String>) -> Vec<&'static str> {
    let schema = match schema {
        Value::Bool(false) => return Vec::new(),
        Value::Object(schema) => schema,
        _ => return JSON_TYPES.to_vec(),
    };
    if let Some(Value::String(reference)) = schema.get("$ref")
        && !seen.contains(reference)
        && let Some(target) = resolve_local_ref(reference, root)
    {
        let mut seen = seen.clone();
        seen.push(reference.clone());
        return schema_types(target, root, &mut seen);
    }
    match schema.get("type") {
        Some(Value::String(t)) => {
            return match JSON_TYPES.iter().find(|j| *j == t) {
                Some(t) => vec![*t],
                None => JSON_TYPES.to_vec(),
            };
        }
        Some(Value::Array(types)) => return ordered(|t| types.iter().any(|v| v == t)),
        _ => {}
    }
    for keyword in ["anyOf", "oneOf"] {
        if let Some(Value::Array(options)) = schema.get(keyword) {
            let types: Vec<&str> = options
                .iter()
                .filter(|o| o.is_boolean() || o.is_object())
                .flat_map(|o| schema_types(o, root, seen))
                .collect();
            return ordered(|t| types.contains(&t));
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
        let constrained: Vec<Vec<&'static str>> = options
            .iter()
            .filter(|o| o.is_boolean() || o.is_object())
            .map(|o| schema_types(o, root, seen))
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
            return ordered(|t| result.contains(&t));
        }
    }
    if let Some(value) = schema.get("const") {
        return vec![json_type(value)];
    }
    if let Some(Value::Array(values)) = schema.get("enum") {
        let types: Vec<&str> = values.iter().map(json_type).collect();
        return ordered(|t| types.contains(&t));
    }
    let has = |keywords: &[&str]| keywords.iter().any(|k| schema.contains_key(*k));
    if has(&OBJECT_KEYWORDS) {
        vec!["object"]
    } else if has(&ARRAY_KEYWORDS) {
        vec!["array"]
    } else if has(&STRING_KEYWORDS) {
        vec!["string"]
    } else if has(&NUMBER_KEYWORDS) {
        vec!["number"]
    } else {
        JSON_TYPES.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::py::python_json;

    /// `tests/fixtures/kimi_k3_tool_tags.json`, from `tests/scripts/generate_kimi_k3_tool_tags.py`.
    #[test]
    fn auto_tags_match_sglang() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/kimi_k3_tool_tags.json");
        let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let strip: Vec<&str> = case["strip"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s.as_str().unwrap())
                .collect();
            let tag = auto_tool_call_tag(
                case["tools"].as_array().unwrap(),
                case["parallel_tool_calls"] == true,
                &strip,
            );
            let got = tag.map(|tag| python_json(&tag, true));
            assert_eq!(
                got.as_deref(),
                case["structural_tag"].as_str(),
                "{}",
                case["tools"]
            );
        }
    }
}
