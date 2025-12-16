use std::collections::HashMap;

use serde_json::Value;

pub fn parse_tool_calls(raw: Option<String>) -> Result<Vec<Value>, String> {
    match raw {
        Some(s) if !s.is_empty() => serde_json::from_str(&s).map_err(|e| e.to_string()),
        _ => Ok(Vec::new()),
    }
}

pub fn parse_metadata(raw: Option<String>) -> Result<HashMap<String, Value>, String> {
    match raw {
        Some(s) if !s.is_empty() => serde_json::from_str(&s).map_err(|e| e.to_string()),
        _ => Ok(HashMap::new()),
    }
}

pub fn parse_raw_response(raw: Option<String>) -> Result<Value, String> {
    match raw {
        Some(s) if !s.is_empty() => serde_json::from_str(&s).map_err(|e| e.to_string()),
        _ => Ok(Value::Null),
    }
}

pub fn parse_json_value(raw: Option<String>) -> Result<Value, String> {
    match raw {
        Some(s) if !s.is_empty() => serde_json::from_str(&s).map_err(|e| e.to_string()),
        _ => Ok(Value::Array(vec![])),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parse_tool_calls_handles_empty_input() {
        assert!(parse_tool_calls(None).unwrap().is_empty());
        assert!(parse_tool_calls(Some(String::new())).unwrap().is_empty());
    }

    #[test]
    fn parse_tool_calls_round_trips() {
        let payload = json!([{ "type": "test", "value": 1 }]).to_string();
        let parsed = parse_tool_calls(Some(payload)).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["type"], "test");
        assert_eq!(parsed[0]["value"], 1);
    }

    #[test]
    fn parse_metadata_defaults_to_empty_map() {
        assert!(parse_metadata(None).unwrap().is_empty());
    }

    #[test]
    fn parse_metadata_round_trips() {
        let payload = json!({"key": "value", "nested": {"bool": true}}).to_string();
        let parsed = parse_metadata(Some(payload)).unwrap();
        assert_eq!(parsed.get("key").unwrap(), "value");
        assert_eq!(parsed["nested"]["bool"], true);
    }

    #[test]
    fn parse_raw_response_handles_null() {
        assert_eq!(parse_raw_response(None).unwrap(), Value::Null);
    }

    #[test]
    fn parse_raw_response_round_trips() {
        let payload = json!({"id": "abc"}).to_string();
        let parsed = parse_raw_response(Some(payload)).unwrap();
        assert_eq!(parsed["id"], "abc");
    }
}
