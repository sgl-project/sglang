//! MCP tool argument handling.
//!
//! Supports both JSON strings and parsed Maps with automatic type coercion.

use serde_json::Map;

/// Tool arguments input - supports both JSON strings and parsed Maps
pub enum ToolArgs {
    /// JSON string that needs parsing
    JsonString(String),
    /// Already parsed map
    Map(Option<Map<String, serde_json::Value>>),
}

impl ToolArgs {
    /// Convert to Map with type coercion based on tool schema
    pub(crate) fn into_map(
        self,
        tool_schema: Option<&serde_json::Value>,
    ) -> Result<Option<Map<String, serde_json::Value>>, String> {
        match self {
            ToolArgs::JsonString(s) => {
                if s.is_empty() || s == "{}" {
                    return Ok(None);
                }
                let mut value: serde_json::Value =
                    serde_json::from_str(&s).map_err(|e| format!("parse tool args: {}", e))?;
                Self::coerce_types(&mut value, tool_schema)?;
                let result = match value {
                    serde_json::Value::Object(m) => Some(m),
                    _ => None,
                };
                Ok(result)
            }
            ToolArgs::Map(map) => {
                if let Some(m) = map {
                    let mut value = serde_json::Value::Object(m);
                    Self::coerce_types(&mut value, tool_schema)?;
                    let result = match value {
                        serde_json::Value::Object(m) => Some(m),
                        _ => None,
                    };
                    Ok(result)
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// Coerce string numbers to actual numbers based on schema
    /// LLMs often output numbers as strings, so we need to convert them
    fn coerce_types(
        value: &mut serde_json::Value,
        tool_schema: Option<&serde_json::Value>,
    ) -> Result<(), String> {
        let Some(params) = tool_schema else {
            return Ok(());
        };
        let Some(props) = params.get("properties").and_then(|p| p.as_object()) else {
            return Ok(());
        };
        let Some(args) = value.as_object_mut() else {
            return Ok(());
        };

        for (key, val) in args.iter_mut() {
            let should_be_number = props
                .get(key)
                .and_then(|s| s.get("type"))
                .and_then(|t| t.as_str())
                .is_some_and(|t| matches!(t, "number" | "integer"));

            if should_be_number {
                if let Some(s) = val.as_str() {
                    if let Ok(num) = s.parse::<f64>() {
                        *val = serde_json::json!(num);
                    }
                }
            }
        }
        Ok(())
    }
}

// Implement From traits for convenient conversion
impl From<String> for ToolArgs {
    fn from(s: String) -> Self {
        ToolArgs::JsonString(s)
    }
}

impl From<&str> for ToolArgs {
    fn from(s: &str) -> Self {
        ToolArgs::JsonString(s.to_string())
    }
}

impl From<Option<Map<String, serde_json::Value>>> for ToolArgs {
    fn from(map: Option<Map<String, serde_json::Value>>) -> Self {
        ToolArgs::Map(map)
    }
}
