use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::protocols::common::GenerationRequest;

/// Anthropic Messages request with minimal routing metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub messages: Vec<Value>,
    #[serde(default, flatten)]
    pub extra: Map<String, Value>,
}

/// Anthropic count_tokens request with minimal routing metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicCountTokensRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<Value>,
    #[serde(default, flatten)]
    pub extra: Map<String, Value>,
}

impl GenerationRequest for AnthropicMessagesRequest {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_model(&self) -> Option<&str> {
        Some(self.model.as_str())
    }

    fn extract_text_for_routing(&self) -> String {
        extract_text_from_messages(&self.messages)
    }
}

impl GenerationRequest for AnthropicCountTokensRequest {
    fn is_stream(&self) -> bool {
        false
    }

    fn get_model(&self) -> Option<&str> {
        Some(self.model.as_str())
    }

    fn extract_text_for_routing(&self) -> String {
        extract_text_from_messages(&self.messages)
    }
}

fn extract_text_from_messages(messages: &[Value]) -> String {
    let mut parts = Vec::new();

    for msg in messages {
        let Some(content) = msg.get("content") else { continue };

        match content {
            Value::String(text) => {
                if !text.is_empty() {
                    parts.push(text.clone());
                }
            }
            Value::Array(items) => {
                for item in items {
                    let Some(kind) = item.get("type").and_then(Value::as_str) else {
                        continue;
                    };
                    if kind == "text" {
                        if let Some(text) = item.get("text").and_then(Value::as_str) {
                            if !text.is_empty() {
                                parts.push(text.to_string());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    parts.join(" ")
}
