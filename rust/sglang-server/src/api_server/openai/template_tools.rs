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
/// This preserves schema shape; it does not backfill schema members.
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
        if let Some(parameters) = &tool.function.parameters
            && !parameters.is_object()
            && !parameters.is_boolean()
        {
            return Err("tool parameters must be a JSON Schema object or boolean".into());
        }
    }
    serde_json::to_value(tools)
        .map(Some)
        .map_err(|error| error.to_string())
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
    use super::{normalize_prompt_tools, python_bool};
    use serde_json::{Value, json};

    fn render_value(tool: Value) -> Value {
        normalize_prompt_tools(&json!([tool])).unwrap().unwrap()
    }

    #[test]
    fn python_tool_defaults_and_canonical_field_order() {
        let tools = render_value(json!({"function": {"name": "weather"}}));
        assert_eq!(
            tools.to_string(),
            r#"[{"type":"function","function":{"description":null,"name":"weather","parameters":null,"strict":false},"defer_loading":null}]"#
        );
    }

    #[test]
    fn python_tool_explicit_values_and_partial_schema_are_preserved() {
        let tool: Value = serde_json::from_str(
            r#"{"function":{"parameters":{"properties":{"z":{"type":"string"},"a":{"enum":["x","y"]}}},"strict":true,"name":"weather","description":""},"type":"function","defer_loading":false}"#,
        )
        .unwrap();
        assert_eq!(
            render_value(tool).to_string(),
            r#"[{"type":"function","function":{"description":"","name":"weather","parameters":{"properties":{"z":{"type":"string"},"a":{"enum":["x","y"]}}},"strict":true,"defer_loading":false},"defer_loading":false}]"#
        );
    }

    #[test]
    fn python_tool_null_empty_and_missing_fields_remain_distinct() {
        for parameters in [json!(null), json!({}), json!({"type": "object"})] {
            for description in [json!(null), json!(""), json!("description")] {
                let tools = render_value(json!({"type": "function", "function": {
                    "name": "weather", "parameters": parameters, "description": description
                }}));
                assert_eq!(tools[0]["function"]["parameters"], parameters);
                assert_eq!(tools[0]["function"]["description"], description);
            }
        }
    }

    #[test]
    fn python_tool_defer_loading_propagates_with_inner_precedence() {
        for outer in [
            None,
            Some(json!(null)),
            Some(json!(false)),
            Some(json!(true)),
        ] {
            for inner in [
                None,
                Some(json!(null)),
                Some(json!(false)),
                Some(json!(true)),
            ] {
                let mut tool = json!({"type":"function", "function":{"name":"weather"}});
                if let Some(value) = &outer {
                    tool["defer_loading"] = value.clone();
                }
                if let Some(value) = &inner {
                    tool["function"]["defer_loading"] = value.clone();
                }
                let tools = render_value(tool);
                let expected = inner
                    .as_ref()
                    .and_then(Value::as_bool)
                    .or_else(|| outer.as_ref().and_then(Value::as_bool));
                assert_eq!(
                    tools[0]["defer_loading"],
                    outer.clone().unwrap_or(Value::Null)
                );
                assert_eq!(
                    tools[0]["function"].get("defer_loading"),
                    expected.map(Value::Bool).as_ref()
                );
            }
        }
    }

    #[test]
    fn python_tool_bool_coercion_and_null_strict_rejection() {
        for (value, expected) in [
            (json!("TRUE"), true),
            (json!(1), true),
            (json!("off"), false),
            (json!(0.0), false),
        ] {
            let tools = render_value(json!({"type":"function", "defer_loading":value,
                "function":{"name":"weather", "strict":value}}));
            assert_eq!(tools[0]["function"]["strict"], expected);
            assert_eq!(tools[0]["function"]["defer_loading"], expected);
            assert_eq!(tools[0]["defer_loading"], expected);
        }
        for value in [json!(null), json!(2), json!(" true "), json!([]), json!({})] {
            assert!(
                normalize_prompt_tools(&json!([{"type":"function", "function":{
                    "name":"weather", "strict":value
                }}]))
                .is_err(),
                "strict={value}"
            );
        }
    }

    #[test]
    fn python_tool_model_rejects_invalid_string_fields_and_ignores_extras() {
        for tool in [
            json!({"type":null, "function":{"name":"weather"}}),
            json!({"type":"function", "function":{"name":4}}),
            json!({"type":"function", "function":{"name":"weather", "description":4}}),
            json!({"type":"function", "function":{"name":"weather", "defer_loading":"bad"}}),
            json!({"type":"function", "function":{"name":"weather"}, "defer_loading":2}),
        ] {
            assert!(normalize_prompt_tools(&json!([tool])).is_err());
        }
        let tools = render_value(json!({"type":"function", "ignored":1,
            "function":{"name":"weather", "ignored":2}}));
        assert!(tools[0].get("ignored").is_none());
        assert!(tools[0]["function"].get("ignored").is_none());
    }

    #[test]
    fn python_tool_empty_and_null_controls() {
        assert_eq!(normalize_prompt_tools(&Value::Null).unwrap(), None);
        assert_eq!(normalize_prompt_tools(&json!([])).unwrap(), Some(json!([])));
        for invalid in [json!({}), json!(true), json!("tools")] {
            assert!(normalize_prompt_tools(&invalid).is_err());
        }
    }

    #[test]
    fn python_tool_objects_reject_positional_arrays() {
        for tool in [
            json!(["function", {"name":"weather"}]),
            json!({"type":"function", "function":[null, "weather", null, false]}),
            json!({"type":"function", "function":[]}),
        ] {
            assert!(normalize_prompt_tools(&json!([tool])).is_err());
        }
    }

    #[test]
    fn python_tool_boolean_schemas_and_custom_type_are_preserved() {
        for parameters in [json!(true), json!(false)] {
            let tools = render_value(json!({"type":"custom", "function":{
                "name":"weather", "parameters":parameters
            }}));
            assert_eq!(tools[0]["type"], "custom");
            assert_eq!(tools[0]["function"]["parameters"], parameters);
        }
        for parameters in [json!([]), json!("schema"), json!(1)] {
            assert!(
                normalize_prompt_tools(&json!([{"function":{
                    "name":"weather", "parameters":parameters
                }}]))
                .is_err()
            );
        }
    }

    #[test]
    fn python_boolean_conversion_is_shared_with_request_flags() {
        for value in [
            json!(true),
            json!(1),
            json!(1.0),
            json!("TRUE"),
            json!("t"),
            json!("yes"),
            json!("y"),
            json!("on"),
            json!("1"),
        ] {
            assert!(python_bool(&value).unwrap());
        }
        for value in [
            json!(false),
            json!(0),
            json!(0.0),
            json!("FALSE"),
            json!("f"),
            json!("no"),
            json!("n"),
            json!("off"),
            json!("0"),
        ] {
            assert!(!python_bool(&value).unwrap());
        }
        for value in [
            json!(null),
            json!(2),
            json!(-1),
            json!(0.5),
            json!(" true "),
            json!("invalid"),
            json!([]),
            json!({}),
        ] {
            assert!(python_bool(&value).is_err());
        }
    }
}
