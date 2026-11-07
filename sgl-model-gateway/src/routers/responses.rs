use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{json, Map, Value};
use validator::Validate;

use crate::{
    core::UNKNOWN_MODEL_ID,
    protocols::{common::GenerationRequest, responses::ResponsesRequest, validated::Normalizable},
};

#[derive(Clone, Debug, Deserialize, Validate)]
#[serde(try_from = "Map<String, Value>")]
pub struct ResponsesRequestBody {
    raw: Value,
    routing: RoutingFields,
}

#[derive(Clone, Debug, Deserialize)]
struct RoutingFields {
    model: Option<String>,
    stream: Option<bool>,
    background: Option<bool>,
}

impl TryFrom<Map<String, Value>> for ResponsesRequestBody {
    type Error = serde_json::Error;

    fn try_from(fields: Map<String, Value>) -> Result<Self, Self::Error> {
        let raw = Value::Object(fields);
        let routing = RoutingFields::deserialize(&raw)?;
        Ok(Self { raw, routing })
    }
}

impl Serialize for ResponsesRequestBody {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.raw.serialize(serializer)
    }
}

impl Normalizable for ResponsesRequestBody {}

impl ResponsesRequestBody {
    pub fn background(&self) -> bool {
        self.routing.background.unwrap_or(false)
    }

    pub fn parse_typed(&self) -> Result<ResponsesRequest, Box<Response>> {
        let mut request: ResponsesRequest = serde_json::from_value(self.raw.clone())
            .map_err(|error| reject(error.to_string(), json!("json_parse_error")))?;
        request.normalize();
        request
            .validate()
            .map_err(|error| reject(error.to_string(), json!(400)))?;
        Ok(request)
    }
}

impl GenerationRequest for ResponsesRequestBody {
    fn is_stream(&self) -> bool {
        self.routing.stream.unwrap_or(false)
    }

    fn get_model(&self) -> Option<&str> {
        Some(self.routing.model.as_deref().unwrap_or(UNKNOWN_MODEL_ID))
    }

    fn extract_text_for_routing(&self) -> String {
        let mut texts = Vec::new();
        collect_text(&self.raw["input"], &mut texts);
        texts.join(" ")
    }
}

fn collect_text<'a>(value: &'a Value, texts: &mut Vec<&'a str>) {
    match value {
        Value::String(text) => texts.push(text),
        Value::Array(items) => items.iter().for_each(|item| collect_text(item, texts)),
        Value::Object(fields) => {
            for key in ["content", "text", "arguments", "output"] {
                if let Some(value) = fields.get(key) {
                    collect_text(value, texts);
                }
            }
        }
        _ => {}
    }
}

fn reject(message: String, code: Value) -> Box<Response> {
    Box::new(
        (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {"message": message, "type": "invalid_request_error", "code": code}
            })),
        )
            .into_response(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_routes_still_validate_responses() {
        for body in [
            json!({"input": "hello", "max_output_tokens": 0}),
            json!({"input": [{"type": "additional_tools", "tools": []}]}),
        ] {
            let body: ResponsesRequestBody = serde_json::from_value(body).unwrap();
            assert_eq!(
                body.parse_typed().unwrap_err().status(),
                StatusCode::BAD_REQUEST
            );
        }
        let body: ResponsesRequestBody = serde_json::from_value(json!({"input": "hello"})).unwrap();
        assert!(body.parse_typed().is_ok());
    }

    #[test]
    fn routing_text_includes_tool_replay_and_all_turns() {
        let body: ResponsesRequestBody = serde_json::from_value(json!({"input": [
            {"role": "user", "content": "Weather?"},
            {"type": "function_call", "arguments": "{\"city\":\"Paris\"}"},
            {"type": "function_call_output", "output": [{"type": "input_text", "text": "Sunny"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "Tomorrow?"}]}
        ]}))
        .unwrap();
        assert_eq!(
            body.extract_text_for_routing(),
            "Weather? {\"city\":\"Paris\"} Sunny Tomorrow?"
        );
    }
}
