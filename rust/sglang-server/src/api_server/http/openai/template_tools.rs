//! Python-compatible tool definitions for the Jinja prompt.

use serde_json::Value;

/// Python/Pydantic's non-strict bool accepts these wire values without trimming.
pub(crate) fn python_bool(value: &Value) -> Result<bool, String> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Number(value) if value.as_f64() == Some(0.0) => Ok(false),
        Value::Number(value) if value.as_f64() == Some(1.0) => Ok(true),
        Value::String(value) => match value.to_ascii_lowercase().as_str() {
            "0" | "off" | "f" | "false" | "n" | "no" => Ok(false),
            "1" | "on" | "t" | "true" | "y" | "yes" => Ok(true),
            _ => Err("expected a boolean".into()),
        },
        _ => Err("expected a boolean".into()),
    }
}

/// Match Python's Function/Tool model_dump for the ordinary Jinja path.
/// Normalize schema type aliases without backfilling schema members.
pub(crate) fn normalize_prompt_tools(raw: &Value) -> Result<Option<Value>, String> {
    if raw.is_null() {
        return Ok(None);
    }
    let raw_tools = raw.as_array().ok_or("tools must be an array")?;
    for tool in raw_tools {
        if !tool.is_object() || !tool.get("function").is_some_and(Value::is_object) {
            return Err("each tool and its function must be an object".into());
        }
    }
    let mut tools: Vec<PromptTool> = serde_json::from_value(raw.clone())
        .map_err(|error| format!("invalid tool definition: {error}"))?;
    for tool in &mut tools {
        if tool.function.defer_loading.is_none() {
            tool.function.defer_loading = tool.defer_loading;
        }
        if let Some(parameters) = &mut tool.function.parameters {
            if !parameters.is_object() && !parameters.is_boolean() {
                return Err("tool parameters must be a JSON Schema object or boolean".into());
            }
            normalize_schema_types(parameters);
        }
    }
    serde_json::to_value(tools)
        .map(Some)
        .map_err(|error| error.to_string())
}

/// Python's normalize_json_schema_types walks only schema-valued keywords.
/// Full Draft202012 validation is a separate request-validation contract.
fn normalize_schema_types(schema: &mut Value) {
    if let Value::Array(items) = schema {
        for item in items {
            normalize_schema_types(item);
        }
        return;
    }
    let Some(schema) = schema.as_object_mut() else {
        return;
    };
    if let Some(types) = schema.get_mut("type") {
        if let Value::Array(types) = types {
            let mut normalized = Vec::with_capacity(types.len());
            for mut value in std::mem::take(types) {
                normalize_single_type(&mut value);
                if !normalized.contains(&value) {
                    normalized.push(value);
                }
            }
            *types = normalized;
        } else {
            normalize_single_type(types);
        }
    }
    for key in [
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
    ] {
        if let Some(Value::Object(nested)) = schema.get_mut(key) {
            for value in nested.values_mut() {
                normalize_schema_types(value);
            }
        }
    }
    for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
        if let Some(Value::Array(nested)) = schema.get_mut(key) {
            for value in nested {
                normalize_schema_types(value);
            }
        }
    }
    for key in [
        "items",
        "additionalProperties",
        "not",
        "if",
        "then",
        "else",
        "contains",
        "propertyNames",
        "unevaluatedItems",
        "unevaluatedProperties",
    ] {
        if let Some(nested) = schema.get_mut(key) {
            normalize_schema_types(nested);
        }
    }
}

fn normalize_single_type(value: &mut Value) {
    let Value::String(raw) = value else {
        return;
    };
    let base = raw
        .split('(')
        .next()
        .unwrap_or_default()
        // Python str.strip also includes these four ASCII separators.
        .trim_matches(|ch: char| ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch))
        .to_lowercase();
    let normalized = match base.as_str() {
        "null" => Some("null"),
        "boolean" | "bool" => Some("boolean"),
        "object" | "map" => Some("object"),
        "array" | "arr" | "tuple" | "set" => Some("array"),
        "number" | "double" | "decimal" | "real" | "numeric" => Some("number"),
        "integer" | "bigint" | "smallint" | "tinyint" => Some("integer"),
        "string" | "str" | "text" | "varchar" | "char" | "enum" | "uuid" | "date" | "datetime"
        | "time" | "timestamp" | "binary" | "blob" | "bytea" | "bytes" | "varbinary" => {
            Some("string")
        }
        _ => [
            ("int", "integer"),
            ("uint", "integer"),
            ("long", "integer"),
            ("short", "integer"),
            ("unsigned", "integer"),
            ("num", "number"),
            ("float", "number"),
            ("list", "array"),
            ("dict", "object"),
        ]
        .into_iter()
        .find_map(|(prefix, target)| {
            base.strip_prefix(prefix)
                .filter(|suffix| {
                    suffix.is_empty()
                        || suffix
                            .starts_with(|ch: char| ch.is_ascii_digit() || "[<( \t".contains(ch))
                })
                .map(|_| target)
        }),
    };
    if let Some(normalized) = normalized {
        *raw = normalized.into();
    }
}

// Declaration order is Python's prompt serialization order, independent of wire order.
#[derive(serde::Deserialize, serde::Serialize)]
struct PromptTool {
    #[serde(default = "function_type")]
    r#type: String,
    function: PromptFunction,
    #[serde(default, deserialize_with = "optional_bool")]
    defer_loading: Option<bool>,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct PromptFunction {
    #[serde(default)]
    description: Option<String>,
    name: String,
    #[serde(default)]
    parameters: Option<Value>,
    #[serde(default, deserialize_with = "required_bool")]
    strict: bool,
    #[serde(
        default,
        deserialize_with = "optional_bool",
        skip_serializing_if = "Option::is_none"
    )]
    defer_loading: Option<bool>,
}

fn function_type() -> String {
    "function".into()
}

fn required_bool<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
    use serde::Deserialize;
    python_bool(&Value::deserialize(deserializer)?).map_err(serde::de::Error::custom)
}

fn optional_bool<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<bool>, D::Error> {
    use serde::Deserialize;
    Option::<Value>::deserialize(deserializer)?
        .as_ref()
        .map(python_bool)
        .transpose()
        .map_err(serde::de::Error::custom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parameters(schema: Value) -> Value {
        normalize_prompt_tools(&json!([{"function":{"name":"weather","parameters":schema}}]))
            .unwrap()
            .unwrap()[0]["function"]["parameters"]
            .clone()
    }

    #[test]
    fn schema_types_match_python_aliases_and_prefix_boundaries() {
        for (expected, aliases) in [
            (
                "string",
                "str text varchar char enum uuid date datetime time timestamp binary blob bytea bytes varbinary",
            ),
            ("boolean", "bool"),
            (
                "integer",
                "bigint smallint tinyint int uint32 long short unsigned",
            ),
            ("number", "double decimal real numeric num float64"),
            ("array", "arr tuple set list[str]"),
            ("object", "map dict[str,int]"),
        ] {
            for alias in aliases.split_whitespace() {
                assert_eq!(
                    parameters(json!({"type":alias}))["type"],
                    expected,
                    "{alias}"
                );
            }
        }
        for alias in [
            " INTEGER(4) ",
            "int (64)",
            "\u{1c}\u{1d} INT32 \u{1e}\u{1f}",
        ] {
            assert_eq!(parameters(json!({"type":alias}))["type"], "integer");
        }
        for unknown in [
            "internal",
            "list_price",
            "dictation",
            "integerish",
            "custom",
        ] {
            assert_eq!(parameters(json!({"type":unknown}))["type"], unknown);
        }
    }

    #[test]
    fn schema_unions_keep_first_normalized_type_and_object_order() {
        let schema = serde_json::from_str(r#"{"description":"Keep order","type":["int32","integer","null","INT64"],"properties":{"city":{"description":"city","type":"varchar"}}}"#).unwrap();
        assert_eq!(
            parameters(schema).to_string(),
            r#"{"description":"Keep order","type":["integer","null"],"properties":{"city":{"description":"city","type":"string"}}}"#
        );
    }

    #[test]
    fn schema_walker_uses_python_schema_locations() {
        for key in [
            "properties",
            "patternProperties",
            "$defs",
            "definitions",
            "dependentSchemas",
        ] {
            assert_eq!(
                parameters(json!({key:{"x":{"type":"varchar"}}}))[key]["x"]["type"],
                "string",
                "{key}"
            );
        }
        for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
            assert_eq!(
                parameters(json!({key:[{"type":"varchar"}]}))[key][0]["type"],
                "string",
                "{key}"
            );
        }
        for key in [
            "items",
            "additionalProperties",
            "not",
            "if",
            "then",
            "else",
            "contains",
            "propertyNames",
            "unevaluatedItems",
            "unevaluatedProperties",
        ] {
            assert_eq!(
                parameters(json!({key:{"type":"varchar"}}))[key]["type"],
                "string",
                "{key}"
            );
        }
        let opaque = json!({"default":{"type":"varchar"},"enum":[{"type":"str"}],"const":{"type":"int32"},"examples":[{"type":"str"}],"x-schema":{"type":"int32"},"contentSchema":{"type":"varchar"},"dependencies":{"x":{"type":"varchar"}}});
        assert_eq!(parameters(opaque.clone()), opaque);
        assert_eq!(parameters(json!(true)), true);
        assert_eq!(parameters(json!(false)), false);
    }
}
