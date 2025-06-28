// Request adapter to bridge OpenAI API types with PD routing requirements

use crate::openai_api_types::{
    ChatCompletionRequest, CompletionRequest, GenerateRequest, GenerationRequest, StringOrArray,
};
use crate::pd_types::{Bootstrap, ChatReqInput, GenerateReqInput, SingleOrBatch};
use serde_json::Value;

/// Adapter trait to convert OpenAI requests to PD-compatible requests
pub trait ToPdRequest {
    type Output: Bootstrap;
    fn to_pd_request(self) -> Self::Output;
}

// Helper macro to insert optional fields into a map
macro_rules! insert_if_some {
    ($map:expr, $($field:expr => $key:expr),* $(,)?) => {
        $(
            if let Some(value) = $field {
                $map.insert($key.to_string(), serde_json::to_value(value).unwrap_or(Value::Null));
            }
        )*
    };
}

// Helper macro for simple value insertions
macro_rules! insert_value {
    ($map:expr, $($field:expr => $key:expr),* $(,)?) => {
        $(
            $map.insert($key.to_string(), $field.into());
        )*
    };
}

// ============= Generate Request Adapter =============

impl ToPdRequest for GenerateRequest {
    type Output = GenerateReqInput;

    fn to_pd_request(self) -> Self::Output {
        // Build the other fields first
        let mut other = serde_json::Map::new();

        // Handle text input - check in priority order: text (SGLang), prompt (OpenAI)
        let (text, input_ids) = if let Some(text_str) = self.text {
            // SGLang native format
            (Some(SingleOrBatch::Single(text_str)), None)
        } else if let Some(prompt) = self.prompt {
            // OpenAI style prompt
            let text = match prompt {
                StringOrArray::String(s) => Some(SingleOrBatch::Single(s)),
                StringOrArray::Array(v) => Some(SingleOrBatch::Batch(v)),
            };
            (text, None)
        } else if let Some(ids) = self.input_ids {
            // Input IDs case
            let input_ids = match ids {
                crate::openai_api_types::InputIds::Single(ids) => Some(SingleOrBatch::Single(ids)),
                crate::openai_api_types::InputIds::Batch(ids) => Some(SingleOrBatch::Batch(ids)),
            };
            (None, input_ids)
        } else {
            // No input provided
            (None, None)
        };

        // Add parameters to other - handle both old and new style
        if let Some(params) = self.parameters {
            // For generate endpoint, extract max_new_tokens to top level if present
            let mut params_value = serde_json::to_value(&params).unwrap_or(Value::Null);
            if let Value::Object(ref mut params_map) = params_value {
                // Move max_new_tokens to top level if it exists
                if let Some(max_new_tokens) = params_map.remove("max_new_tokens") {
                    other.insert("max_new_tokens".to_string(), max_new_tokens);
                }
                // Move temperature to top level if it exists
                if let Some(temperature) = params_map.remove("temperature") {
                    other.insert("temperature".to_string(), temperature);
                }
            }
            // Only add parameters if there are remaining fields
            if !params_value.is_null() && params_value.as_object().map_or(false, |m| !m.is_empty())
            {
                other.insert("parameters".to_string(), params_value);
            }
        }

        // Add sampling_params if present
        if let Some(sampling_params) = self.sampling_params {
            let params_value = serde_json::to_value(&sampling_params).unwrap_or(Value::Null);
            if !params_value.is_null() {
                // Extract commonly used fields to top level
                if let Value::Object(ref params_map) = params_value {
                    if let Some(max_new_tokens) = params_map.get("max_new_tokens") {
                        other.insert("max_new_tokens".to_string(), max_new_tokens.clone());
                    }
                    if let Some(temperature) = params_map.get("temperature") {
                        other.insert("temperature".to_string(), temperature.clone());
                    }
                }
                other.insert("sampling_params".to_string(), params_value);
            }
        }

        // Add other fields
        insert_value!(other,
            self.stream => "stream",
            self.return_logprob => "return_logprob"
        );

        GenerateReqInput {
            text,
            input_ids,
            stream: self.stream,
            bootstrap_host: None,
            bootstrap_port: None,
            bootstrap_room: None,
            other: Value::Object(other),
        }
    }
}

// ============= Completion Request Adapter =============

impl ToPdRequest for CompletionRequest {
    type Output = GenerateReqInput;

    fn to_pd_request(self) -> Self::Output {
        // Convert CompletionRequest to GenerateReqInput
        let text = match self.prompt {
            StringOrArray::String(s) => Some(SingleOrBatch::Single(s)),
            StringOrArray::Array(v) => Some(SingleOrBatch::Batch(v)),
        };

        // Map OpenAI parameters to generate parameters
        let mut other = serde_json::Map::new();

        // Create parameters object
        let mut params = serde_json::Map::new();

        // Map OpenAI fields to internal parameter names
        insert_if_some!(params,
            self.max_tokens => "max_new_tokens",
            self.temperature => "temperature",
            self.top_p => "top_p",
            self.n => "best_of",
            self.logprobs => "top_n_tokens",
            self.seed => "seed"
        );

        // Special handling for fields that need transformation
        if let Some(presence_penalty) = self.presence_penalty {
            params.insert(
                "repetition_penalty".to_string(),
                (1.0 + presence_penalty).into(),
            );
        }

        if let Some(stop) = self.stop {
            let stop_sequences = match stop {
                StringOrArray::String(s) => vec![s],
                StringOrArray::Array(v) => v,
            };
            params.insert("stop".to_string(), stop_sequences.into());
        }

        if self.echo {
            params.insert("return_full_text".to_string(), true.into());
        }

        other.insert("parameters".to_string(), Value::Object(params));

        // Store original model and stream flag
        insert_value!(other,
            self.model => "model",
            self.stream => "stream"
        );

        GenerateReqInput {
            text,
            input_ids: None,
            stream: self.stream,
            bootstrap_host: None,
            bootstrap_port: None,
            bootstrap_room: None,
            other: Value::Object(other),
        }
    }
}

// ============= Chat Completion Request Adapter =============

impl ToPdRequest for ChatCompletionRequest {
    type Output = ChatReqInput;

    fn to_pd_request(self) -> Self::Output {
        let mut other = serde_json::Map::new();

        // Add required fields
        insert_if_some!(other,
            Some(&self.messages) => "messages"
        );

        insert_value!(other,
            self.model => "model",
            self.stream => "stream"
        );

        // Add all optional fields
        insert_if_some!(other,
            self.temperature => "temperature",
            self.top_p => "top_p",
            self.n => "n",
            self.stop => "stop",
            self.max_tokens => "max_tokens",
            self.max_completion_tokens => "max_completion_tokens",
            self.presence_penalty => "presence_penalty",
            self.frequency_penalty => "frequency_penalty",
            self.logit_bias => "logit_bias",
            self.user => "user",
            self.seed => "seed",
            self.top_logprobs => "top_logprobs",
            self.response_format => "response_format",
            self.tools => "tools",
            self.tool_choice => "tool_choice",
            self.parallel_tool_calls => "parallel_tool_calls",
            self.functions => "functions",
            self.function_call => "function_call"
        );

        // Handle boolean logprobs flag
        if self.logprobs {
            other.insert("logprobs".to_string(), true.into());
        }

        ChatReqInput {
            stream: self.stream,
            bootstrap_host: None,
            bootstrap_port: None,
            bootstrap_room: None,
            other: Value::Object(other),
        }
    }
}

// ============= Direct routing support for regular router =============

/// Extension trait for routing without PD conversion
pub trait RouteableRequest: GenerationRequest + serde::Serialize + Clone {
    /// Convert to JSON for sending to backend
    fn to_json(&self) -> Result<Value, serde_json::Error> {
        serde_json::to_value(self)
    }

    /// Convert to bytes for legacy routing
    fn to_bytes(&self) -> Result<bytes::Bytes, serde_json::Error> {
        let json = serde_json::to_vec(self)?;
        Ok(bytes::Bytes::from(json))
    }
}

impl RouteableRequest for GenerateRequest {}
impl RouteableRequest for CompletionRequest {}
impl RouteableRequest for ChatCompletionRequest {}
