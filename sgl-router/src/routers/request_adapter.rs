// Request adapter to bridge OpenAI API types with PD routing requirements

use super::pd_types::{Bootstrap, ChatReqInput, GenerateReqInput, SingleOrBatch};
use crate::openai_api_types::{
    ChatCompletionRequest, CompletionRequest, GenerateRequest, GenerationRequest, StringOrArray,
};
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

        // Add SGLang extension fields
        insert_if_some!(other,
            // SGLang Extensions - Priority 1
            self.top_k => "top_k",
            self.min_p => "min_p",
            self.min_tokens => "min_tokens",
            self.repetition_penalty => "repetition_penalty",
            self.regex => "regex",
            self.ebnf => "ebnf",
            self.stop_token_ids => "stop_token_ids",
            // SGLang Extensions - Priority 2
            self.lora_path => "lora_path",
            self.session_params => "session_params"
        );

        // SGLang boolean extensions (CompletionRequest has these as bool, not Option<bool>)
        other.insert("no_stop_trim".to_string(), self.no_stop_trim.into());
        other.insert("ignore_eos".to_string(), self.ignore_eos.into());
        other.insert(
            "skip_special_tokens".to_string(),
            self.skip_special_tokens.into(),
        );
        other.insert(
            "return_hidden_states".to_string(),
            self.return_hidden_states.into(),
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
            self.stream_options => "stream_options",
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
            self.function_call => "function_call",
            // SGLang Extensions - Priority 1
            self.top_k => "top_k",
            self.min_p => "min_p",
            self.min_tokens => "min_tokens",
            self.repetition_penalty => "repetition_penalty",
            self.regex => "regex",
            self.ebnf => "ebnf",
            self.stop_token_ids => "stop_token_ids",
            // SGLang Extensions - Priority 2
            self.lora_path => "lora_path",
            self.session_params => "session_params"
        );

        // Handle boolean flags
        if self.logprobs {
            other.insert("logprobs".to_string(), true.into());
        }

        // SGLang boolean extensions (ChatCompletionRequest has these as bool, not Option<bool>)
        other.insert("no_stop_trim".to_string(), self.no_stop_trim.into());
        other.insert("ignore_eos".to_string(), self.ignore_eos.into());
        other.insert(
            "continue_final_message".to_string(),
            self.continue_final_message.into(),
        );
        other.insert(
            "skip_special_tokens".to_string(),
            self.skip_special_tokens.into(),
        );
        other.insert(
            "separate_reasoning".to_string(),
            self.separate_reasoning.into(),
        );
        other.insert("stream_reasoning".to_string(), self.stream_reasoning.into());
        other.insert(
            "return_hidden_states".to_string(),
            self.return_hidden_states.into(),
        );

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai_api_types::*;
    use serde_json::json;
    use std::collections::HashMap;

    // ============= Test Helper Functions =============
    //
    // These helper functions create default request instances with all required SGLang extension fields
    // properly initialized. Use the struct spread operator `..default_*_request()` to override only
    // the fields you need for specific tests, avoiding repetitive boilerplate code.
    //
    // Example usage:
    //   let req = GenerateRequest {
    //       text: Some("Custom text".to_string()),
    //       stream: true,
    //       ..default_generate_request()
    //   };

    /// Create a default GenerateRequest with minimal fields set
    fn default_generate_request() -> GenerateRequest {
        GenerateRequest {
            text: None,
            prompt: None,
            input_ids: None,
            stream: false,
            parameters: None,
            sampling_params: None,
            return_logprob: false,
            // SGLang Extensions
            lora_path: None,
            session_params: None,
            return_hidden_states: false,
            rid: None,
        }
    }

    /// Create a default CompletionRequest with minimal fields set
    fn default_completion_request() -> CompletionRequest {
        CompletionRequest {
            model: "test-model".to_string(),
            prompt: StringOrArray::String("test prompt".to_string()),
            max_tokens: None,
            temperature: None,
            top_p: None,
            n: None,
            stream: false,
            stream_options: None,
            logprobs: None,
            echo: false,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            best_of: None,
            logit_bias: None,
            user: None,
            seed: None,
            suffix: None,
            // SGLang Extensions
            top_k: None,
            min_p: None,
            min_tokens: None,
            repetition_penalty: None,
            regex: None,
            ebnf: None,
            json_schema: None,
            stop_token_ids: None,
            no_stop_trim: false,
            ignore_eos: false,
            skip_special_tokens: true,
            // SGLang Extensions
            lora_path: None,
            session_params: None,
            return_hidden_states: false,
            other: serde_json::Map::new(),
        }
    }

    /// Create a default ChatCompletionRequest with minimal fields set
    fn default_chat_completion_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![ChatMessage::User {
                role: "user".to_string(),
                content: UserMessageContent::Text("test message".to_string()),
                name: None,
            }],
            temperature: None,
            top_p: None,
            n: None,
            stream: false,
            stream_options: None,
            stop: None,
            max_tokens: None,
            max_completion_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            logprobs: false,
            top_logprobs: None,
            user: None,
            seed: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            functions: None,
            function_call: None,
            // SGLang Extensions
            top_k: None,
            min_p: None,
            min_tokens: None,
            repetition_penalty: None,
            regex: None,
            ebnf: None,
            stop_token_ids: None,
            no_stop_trim: false,
            ignore_eos: false,
            continue_final_message: false,
            skip_special_tokens: true,
            // SGLang Extensions
            lora_path: None,
            session_params: None,
            separate_reasoning: true,
            stream_reasoning: true,
            return_hidden_states: false,
        }
    }

    // ============= GenerateRequest to_pd_request Tests =============

    #[test]
    fn test_generate_to_pd_request_with_text_only() {
        let req = GenerateRequest {
            text: Some("Hello world".to_string()),
            ..default_generate_request()
        };

        let pd_req = req.to_pd_request();

        // Check text field conversion
        assert!(matches!(pd_req.text, Some(SingleOrBatch::Single(ref s)) if s == "Hello world"));
        assert!(pd_req.input_ids.is_none());

        // Check bootstrap fields are None
        assert!(pd_req.bootstrap_host.is_none());
        assert!(pd_req.bootstrap_port.is_none());
        assert!(pd_req.bootstrap_room.is_none());

        // Check stream flag
        assert_eq!(pd_req.stream, false);

        // Check other fields
        let other = pd_req.other.as_object().unwrap();
        assert_eq!(other.get("stream"), Some(&json!(false)));
        assert_eq!(other.get("return_logprob"), Some(&json!(false)));
    }

    #[test]
    fn test_generate_to_pd_request_with_prompt_string() {
        let req = GenerateRequest {
            prompt: Some(StringOrArray::String("Test prompt".to_string())),
            stream: true,
            return_logprob: true,
            ..default_generate_request()
        };

        let pd_req = req.to_pd_request();

        assert!(matches!(pd_req.text, Some(SingleOrBatch::Single(ref s)) if s == "Test prompt"));
        assert!(pd_req.input_ids.is_none());
        assert_eq!(pd_req.stream, true);

        let other = pd_req.other.as_object().unwrap();
        assert_eq!(other.get("stream"), Some(&json!(true)));
        assert_eq!(other.get("return_logprob"), Some(&json!(true)));
    }

    #[test]
    fn test_generate_to_pd_request_with_prompt_array() {
        let req = GenerateRequest {
            text: None,
            prompt: Some(StringOrArray::Array(vec![
                "Prompt 1".to_string(),
                "Prompt 2".to_string(),
                "Prompt 3".to_string(),
            ])),
            input_ids: None,
            stream: false,
            parameters: None,
            sampling_params: None,
            return_logprob: false,
            ..default_generate_request()
        };

        let pd_req = req.to_pd_request();

        match pd_req.text {
            Some(SingleOrBatch::Batch(ref batch)) => {
                assert_eq!(batch.len(), 3);
                assert_eq!(batch[0], "Prompt 1");
                assert_eq!(batch[1], "Prompt 2");
                assert_eq!(batch[2], "Prompt 3");
            }
            _ => panic!("Expected batch text"),
        }
    }

    #[test]
    fn test_generate_to_pd_request_with_single_input_ids() {
        let req = GenerateRequest {
            input_ids: Some(InputIds::Single(vec![100, 200, 300, 400])),
            ..default_generate_request()
        };

        let pd_req = req.to_pd_request();

        assert!(pd_req.text.is_none());
        assert!(matches!(
            pd_req.input_ids,
            Some(SingleOrBatch::Single(ref ids)) if ids == &vec![100, 200, 300, 400]
        ));
    }

    #[test]
    fn test_generate_to_pd_request_with_batch_input_ids() {
        let req = GenerateRequest {
            input_ids: Some(InputIds::Batch(vec![
                vec![1, 2, 3],
                vec![4, 5, 6, 7],
                vec![8, 9],
            ])),
            ..default_generate_request()
        };

        let pd_req = req.to_pd_request();

        match pd_req.input_ids {
            Some(SingleOrBatch::Batch(ref batch)) => {
                assert_eq!(batch.len(), 3);
                assert_eq!(batch[0], vec![1, 2, 3]);
                assert_eq!(batch[1], vec![4, 5, 6, 7]);
                assert_eq!(batch[2], vec![8, 9]);
            }
            _ => panic!("Expected batch input_ids"),
        }
    }

    #[test]
    fn test_generate_to_pd_request_priority_text_over_prompt() {
        let req = GenerateRequest {
            text: Some("SGLang text".to_string()),
            prompt: Some(StringOrArray::String("OpenAI prompt".to_string())),
            input_ids: Some(InputIds::Single(vec![1, 2, 3])),
            ..default_generate_request()
        };

        let pd_req = req.to_pd_request();

        // text should take priority
        assert!(matches!(pd_req.text, Some(SingleOrBatch::Single(ref s)) if s == "SGLang text"));
        assert!(pd_req.input_ids.is_none());
    }

    #[test]
    fn test_generate_to_pd_request_priority_prompt_over_input_ids() {
        let req = GenerateRequest {
            prompt: Some(StringOrArray::String("OpenAI prompt".to_string())),
            input_ids: Some(InputIds::Single(vec![1, 2, 3])),
            ..default_generate_request()
        };

        let pd_req = req.to_pd_request();

        // prompt should take priority over input_ids
        assert!(matches!(pd_req.text, Some(SingleOrBatch::Single(ref s)) if s == "OpenAI prompt"));
        assert!(pd_req.input_ids.is_none());
    }

    #[test]
    fn test_generate_to_pd_request_with_parameters() {
        let params = GenerateParameters {
            max_new_tokens: Some(100),
            temperature: Some(0.8),
            top_p: Some(0.95),
            seed: Some(12345),
            stop: Some(vec!["END".to_string(), "STOP".to_string()]),
            repetition_penalty: Some(1.1),
            ..Default::default()
        };

        let req = GenerateRequest {
            text: Some("test".to_string()),
            parameters: Some(params),
            ..default_generate_request()
        };

        let pd_req = req.to_pd_request();
        let other = pd_req.other.as_object().unwrap();

        // Check that max_new_tokens and temperature were extracted to top level
        assert_eq!(other.get("max_new_tokens"), Some(&json!(100)));
        assert!(other.get("temperature").unwrap().as_f64().unwrap() - 0.8 < 0.0001);

        // Check that other parameters remain under "parameters"
        let params = other.get("parameters").unwrap().as_object().unwrap();
        assert!(params.get("top_p").unwrap().as_f64().unwrap() - 0.95 < 0.0001);
        assert_eq!(params.get("seed"), Some(&json!(12345)));
        assert_eq!(params.get("stop"), Some(&json!(vec!["END", "STOP"])));
        assert!(params.get("repetition_penalty").unwrap().as_f64().unwrap() - 1.1 < 0.0001);
    }

    #[test]
    fn test_generate_to_pd_request_with_sampling_params() {
        let sampling = SamplingParams {
            max_new_tokens: Some(200),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(50),
            frequency_penalty: Some(0.1),
            presence_penalty: Some(0.2),
            repetition_penalty: Some(1.05),
            ..Default::default()
        };

        let req = GenerateRequest {
            text: Some("test".to_string()),
            sampling_params: Some(sampling),
            ..default_generate_request()
        };

        let pd_req = req.to_pd_request();
        let other = pd_req.other.as_object().unwrap();

        // Check extracted top-level fields
        assert_eq!(other.get("max_new_tokens"), Some(&json!(200)));
        assert!(other.get("temperature").unwrap().as_f64().unwrap() - 0.7 < 0.0001);

        // Check full sampling_params is preserved
        let sampling = other.get("sampling_params").unwrap().as_object().unwrap();
        assert_eq!(sampling.get("max_new_tokens"), Some(&json!(200)));
        assert!(sampling.get("temperature").unwrap().as_f64().unwrap() - 0.7 < 0.0001);
        assert!(sampling.get("top_p").unwrap().as_f64().unwrap() - 0.9 < 0.0001);
        assert_eq!(sampling.get("top_k"), Some(&json!(50)));
        assert!(sampling.get("frequency_penalty").unwrap().as_f64().unwrap() - 0.1 < 0.0001);
        assert!(sampling.get("presence_penalty").unwrap().as_f64().unwrap() - 0.2 < 0.0001);
    }

    #[test]
    fn test_generate_to_pd_request_sampling_params_override_parameters() {
        // When both parameters and sampling_params have max_new_tokens/temperature,
        // sampling_params should take precedence (processed last)
        let params = GenerateParameters {
            max_new_tokens: Some(100),
            temperature: Some(0.5),
            ..Default::default()
        };

        let sampling = SamplingParams {
            max_new_tokens: Some(200),
            temperature: Some(0.9),
            ..Default::default()
        };

        let req = GenerateRequest {
            text: Some("test".to_string()),
            prompt: None,
            input_ids: None,
            stream: false,
            parameters: Some(params),
            sampling_params: Some(sampling),
            return_logprob: false,
            ..default_generate_request()
        };

        let pd_req = req.to_pd_request();
        let other = pd_req.other.as_object().unwrap();

        // Should use values from sampling_params since they're processed last
        assert_eq!(other.get("max_new_tokens"), Some(&json!(200)));
        assert!(other.get("temperature").unwrap().as_f64().unwrap() - 0.9 < 0.0001);
    }

    #[test]
    fn test_generate_to_pd_request_empty_parameters() {
        let params = GenerateParameters::default();

        let req = GenerateRequest {
            text: Some("test".to_string()),
            prompt: None,
            input_ids: None,
            stream: false,
            parameters: Some(params),
            sampling_params: None,
            return_logprob: false,
            ..default_generate_request()
        };

        let pd_req = req.to_pd_request();
        let other = pd_req.other.as_object().unwrap();

        // Should not have parameters field if all values are None/default
        assert!(!other.contains_key("parameters"));
        assert!(!other.contains_key("max_new_tokens"));
        assert!(!other.contains_key("temperature"));
    }

    #[test]
    fn test_generate_to_pd_request_all_fields() {
        let params = GenerateParameters {
            max_new_tokens: Some(150),
            temperature: Some(0.6),
            top_k: Some(40),
            ..Default::default()
        };

        let sampling = SamplingParams {
            max_new_tokens: Some(250), // Will override parameters
            temperature: Some(0.8),    // Will override parameters
            presence_penalty: Some(0.1),
            ..Default::default()
        };

        let req = GenerateRequest {
            text: Some("Complex test".to_string()),
            prompt: Some(StringOrArray::String("Ignored prompt".to_string())),
            input_ids: None,
            stream: true,
            parameters: Some(params),
            sampling_params: Some(sampling),
            return_logprob: true,
            ..default_generate_request()
        };

        let pd_req = req.to_pd_request();

        // Verify all fields
        assert!(matches!(pd_req.text, Some(SingleOrBatch::Single(ref s)) if s == "Complex test"));
        assert!(pd_req.input_ids.is_none());
        assert_eq!(pd_req.stream, true);
        assert!(pd_req.bootstrap_host.is_none());
        assert!(pd_req.bootstrap_port.is_none());
        assert!(pd_req.bootstrap_room.is_none());

        let other = pd_req.other.as_object().unwrap();
        assert_eq!(other.get("stream"), Some(&json!(true)));
        assert_eq!(other.get("return_logprob"), Some(&json!(true)));
        // Sampling params override parameters
        assert_eq!(other.get("max_new_tokens"), Some(&json!(250)));
        assert!(other.get("temperature").unwrap().as_f64().unwrap() - 0.8 < 0.0001);
        assert!(other.contains_key("parameters"));
        assert!(other.contains_key("sampling_params"));
    }

    // ============= CompletionRequest to_pd_request Tests =============

    #[test]
    fn test_completion_to_pd_request_basic() {
        let req = CompletionRequest {
            model: "gpt-3.5-turbo".to_string(),
            prompt: StringOrArray::String("Complete this sentence".to_string()),
            ..default_completion_request()
        };

        let pd_req = req.to_pd_request();

        assert!(
            matches!(pd_req.text, Some(SingleOrBatch::Single(ref s)) if s == "Complete this sentence")
        );
        assert!(pd_req.input_ids.is_none());
        assert_eq!(pd_req.stream, false);

        let other = pd_req.other.as_object().unwrap();
        assert_eq!(other.get("model"), Some(&json!("gpt-3.5-turbo")));
        assert_eq!(other.get("stream"), Some(&json!(false)));
    }

    #[test]
    fn test_completion_to_pd_request_array_prompt() {
        let req = CompletionRequest {
            model: "test".to_string(),
            prompt: StringOrArray::Array(vec![
                "First prompt".to_string(),
                "Second prompt".to_string(),
            ]),
            ..default_completion_request()
        };

        let pd_req = req.to_pd_request();

        match pd_req.text {
            Some(SingleOrBatch::Batch(ref batch)) => {
                assert_eq!(batch.len(), 2);
                assert_eq!(batch[0], "First prompt");
                assert_eq!(batch[1], "Second prompt");
            }
            _ => panic!("Expected batch text"),
        }
    }

    #[test]
    fn test_completion_to_pd_request_parameter_mapping() {
        let req = CompletionRequest {
            model: "test".to_string(),
            prompt: StringOrArray::String("test".to_string()),
            max_tokens: Some(150), // -> max_new_tokens
            temperature: Some(0.75),
            top_p: Some(0.92),
            n: Some(3), // -> best_of
            stream: true,
            stream_options: None,
            logprobs: Some(10), // -> top_n_tokens
            echo: true,         // -> return_full_text
            stop: Some(StringOrArray::Array(vec![
                "\\n".to_string(),
                "END".to_string(),
            ])),
            presence_penalty: Some(0.5), // -> repetition_penalty = 1.5
            frequency_penalty: Some(0.2),
            best_of: Some(5),
            logit_bias: None,
            user: Some("user123".to_string()),
            seed: Some(42),
            suffix: Some("...".to_string()),
            ..default_completion_request()
        };

        let pd_req = req.to_pd_request();
        let other = pd_req.other.as_object().unwrap();
        let params = other.get("parameters").unwrap().as_object().unwrap();

        // Check parameter mappings
        assert_eq!(params.get("max_new_tokens"), Some(&json!(150)));
        assert!(params.get("temperature").unwrap().as_f64().unwrap() - 0.75 < 0.0001);
        assert!(params.get("top_p").unwrap().as_f64().unwrap() - 0.92 < 0.0001);
        assert_eq!(params.get("best_of"), Some(&json!(3)));
        assert_eq!(params.get("top_n_tokens"), Some(&json!(10)));
        assert_eq!(params.get("return_full_text"), Some(&json!(true)));
        assert_eq!(params.get("stop"), Some(&json!(vec!["\\n", "END"])));
        assert!(params.get("repetition_penalty").unwrap().as_f64().unwrap() - 1.5 < 0.0001);
        assert_eq!(params.get("seed"), Some(&json!(42)));

        // Check other fields
        assert_eq!(other.get("model"), Some(&json!("test")));
        assert_eq!(other.get("stream"), Some(&json!(true)));
    }

    #[test]
    fn test_completion_to_pd_request_stop_string() {
        let req = CompletionRequest {
            model: "test".to_string(),
            prompt: StringOrArray::String("test".to_string()),
            stop: Some(StringOrArray::String("STOP".to_string())),
            max_tokens: None,
            temperature: None,
            top_p: None,
            n: None,
            stream: false,
            stream_options: None,
            logprobs: None,
            echo: false,
            presence_penalty: None,
            frequency_penalty: None,
            best_of: None,
            logit_bias: None,
            user: None,
            seed: None,
            suffix: None,
            ..default_completion_request()
        };

        let pd_req = req.to_pd_request();
        let other = pd_req.other.as_object().unwrap();
        let params = other.get("parameters").unwrap().as_object().unwrap();

        // Single string stop should be converted to array
        assert_eq!(params.get("stop"), Some(&json!(vec!["STOP"])));
    }

    #[test]
    fn test_completion_to_pd_request_no_presence_penalty() {
        let req = CompletionRequest {
            model: "test".to_string(),
            prompt: StringOrArray::String("test".to_string()),
            presence_penalty: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            n: None,
            stream: false,
            stream_options: None,
            logprobs: None,
            echo: false,
            stop: None,
            frequency_penalty: None,
            best_of: None,
            logit_bias: None,
            user: None,
            seed: None,
            suffix: None,
            ..default_completion_request()
        };

        let pd_req = req.to_pd_request();
        let other = pd_req.other.as_object().unwrap();
        let params = other.get("parameters").unwrap().as_object().unwrap();

        // Should not have repetition_penalty if presence_penalty is None
        assert!(!params.contains_key("repetition_penalty"));
    }

    // ============= ChatCompletionRequest to_pd_request Tests =============

    #[test]
    fn test_chat_to_pd_request_basic() {
        let messages = vec![
            ChatMessage::System {
                role: "system".to_string(),
                content: "You are a helpful assistant".to_string(),
                name: None,
            },
            ChatMessage::User {
                role: "user".to_string(),
                content: UserMessageContent::Text("Hello!".to_string()),
                name: None,
            },
        ];

        let req = ChatCompletionRequest {
            messages,
            model: "gpt-4".to_string(),
            ..default_chat_completion_request()
        };

        let pd_req = req.to_pd_request();

        assert_eq!(pd_req.stream, false);
        assert!(pd_req.bootstrap_host.is_none());
        assert!(pd_req.bootstrap_port.is_none());
        assert!(pd_req.bootstrap_room.is_none());

        let other = pd_req.other.as_object().unwrap();
        assert!(other.contains_key("messages"));
        assert_eq!(other.get("model"), Some(&json!("gpt-4")));
        assert_eq!(other.get("stream"), Some(&json!(false)));

        // Check messages are preserved
        let messages = other.get("messages").unwrap().as_array().unwrap();
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_chat_to_pd_request_with_all_optional_fields() {
        let messages = vec![ChatMessage::User {
            role: "user".to_string(),
            content: UserMessageContent::Text("Test".to_string()),
            name: Some("test_user".to_string()),
        }];

        let mut logit_bias = HashMap::new();
        logit_bias.insert("50256".to_string(), -100.0f32);

        let tool = Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather info".to_string()),
                parameters: json!({"type": "object"}),
            },
        };

        let req = ChatCompletionRequest {
            messages,
            model: "gpt-4".to_string(),
            temperature: Some(0.8),
            top_p: Some(0.95),
            n: Some(2),
            stream: true,
            stream_options: Some(StreamOptions {
                include_usage: Some(true),
            }),
            stop: Some(StringOrArray::String("\\n\\n".to_string())),
            max_tokens: Some(200),
            max_completion_tokens: Some(150),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            logit_bias: Some(logit_bias),
            logprobs: true,
            top_logprobs: Some(5),
            user: Some("user456".to_string()),
            seed: Some(12345),
            response_format: Some(ResponseFormat::JsonObject),
            tools: Some(vec![tool]),
            tool_choice: Some(ToolChoice::Auto),
            parallel_tool_calls: Some(false),
            functions: None,
            function_call: None,
            ..default_chat_completion_request()
        };

        let pd_req = req.to_pd_request();
        let other = pd_req.other.as_object().unwrap();

        // Check all fields are preserved
        assert!(other.get("temperature").unwrap().as_f64().unwrap() - 0.8 < 0.0001);
        assert!(other.get("top_p").unwrap().as_f64().unwrap() - 0.95 < 0.0001);
        assert_eq!(other.get("n"), Some(&json!(2)));
        assert_eq!(other.get("stream"), Some(&json!(true)));
        assert!(other.contains_key("stream_options"));
        assert!(other.contains_key("stop"));
        assert_eq!(other.get("max_tokens"), Some(&json!(200)));
        assert_eq!(other.get("max_completion_tokens"), Some(&json!(150)));
        assert!(other.get("presence_penalty").unwrap().as_f64().unwrap() - 0.1 < 0.0001);
        assert!(other.get("frequency_penalty").unwrap().as_f64().unwrap() - 0.2 < 0.0001);
        assert!(other.contains_key("logit_bias"));
        assert_eq!(other.get("logprobs"), Some(&json!(true)));
        assert_eq!(other.get("top_logprobs"), Some(&json!(5)));
        assert_eq!(other.get("user"), Some(&json!("user456")));
        assert_eq!(other.get("seed"), Some(&json!(12345)));
        assert!(other.contains_key("response_format"));
        assert!(other.contains_key("tools"));
        assert!(other.contains_key("tool_choice"));
        assert_eq!(other.get("parallel_tool_calls"), Some(&json!(false)));
    }

    #[test]
    fn test_chat_to_pd_request_multimodal_content() {
        let messages = vec![ChatMessage::User {
            role: "user".to_string(),
            content: UserMessageContent::Parts(vec![
                ContentPart::Text {
                    text: "What's in this image?".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.jpg".to_string(),
                        detail: Some("high".to_string()),
                    },
                },
            ]),
            name: None,
        }];

        let req = ChatCompletionRequest {
            messages,
            model: "gpt-4-vision".to_string(),
            ..default_chat_completion_request()
        };

        let pd_req = req.to_pd_request();
        let other = pd_req.other.as_object().unwrap();

        // Messages with multimodal content should be preserved
        assert!(other.contains_key("messages"));
        let messages = other.get("messages").unwrap().as_array().unwrap();
        assert_eq!(messages.len(), 1);

        // Verify the message structure is preserved
        let msg = &messages[0];
        assert_eq!(msg["role"], "user");
        assert!(msg["content"].is_array());
    }

    #[test]
    fn test_chat_to_pd_request_logprobs_boolean() {
        let messages = vec![ChatMessage::User {
            role: "user".to_string(),
            content: UserMessageContent::Text("Test".to_string()),
            name: None,
        }];

        let req = ChatCompletionRequest {
            messages,
            model: "test".to_string(),
            logprobs: true, // Boolean logprobs flag
            top_logprobs: Some(3),
            temperature: None,
            top_p: None,
            n: None,
            stream: false,
            stream_options: None,
            stop: None,
            max_tokens: None,
            max_completion_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            seed: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            functions: None,
            function_call: None,
            ..default_chat_completion_request()
        };

        let pd_req = req.to_pd_request();
        let other = pd_req.other.as_object().unwrap();

        assert_eq!(other.get("logprobs"), Some(&json!(true)));
        assert_eq!(other.get("top_logprobs"), Some(&json!(3)));
    }

    #[test]
    fn test_chat_to_pd_request_minimal_fields() {
        let messages = vec![ChatMessage::Assistant {
            role: "assistant".to_string(),
            content: Some("I can help with that.".to_string()),
            name: None,
            tool_calls: None,
            function_call: None,
            reasoning_content: None,
        }];

        let req = ChatCompletionRequest {
            messages,
            model: "gpt-3.5-turbo".to_string(),
            ..default_chat_completion_request()
        };

        let pd_req = req.to_pd_request();
        let other = pd_req.other.as_object().unwrap();

        // Should only have required fields
        assert!(other.contains_key("messages"));
        assert!(other.contains_key("model"));
        assert!(other.contains_key("stream"));

        // Optional fields should not be present
        assert!(!other.contains_key("temperature"));
        assert!(!other.contains_key("top_p"));
        assert!(!other.contains_key("max_tokens"));
        assert!(!other.contains_key("stop"));
    }

    #[test]
    fn test_routeable_request_to_json() {
        let req = GenerateRequest {
            text: Some("test".to_string()),
            ..default_generate_request()
        };

        let json = req.to_json().unwrap();
        assert_eq!(json["text"], "test");
        assert_eq!(json["stream"], false);
    }

    // ============= Macro Tests =============

    #[test]
    fn test_insert_if_some_macro() {
        let mut map = serde_json::Map::new();

        let some_value: Option<i32> = Some(42);
        let none_value: Option<i32> = None;

        insert_if_some!(map,
            some_value => "present",
            none_value => "absent"
        );

        assert_eq!(map.get("present"), Some(&json!(42)));
        assert!(!map.contains_key("absent"));
    }

    #[test]
    fn test_insert_value_macro() {
        let mut map = serde_json::Map::new();

        let value1 = "test";
        let value2 = 42;

        insert_value!(map,
            value1 => "string_field",
            value2 => "int_field"
        );

        assert_eq!(map.get("string_field"), Some(&json!("test")));
        assert_eq!(map.get("int_field"), Some(&json!(42)));
    }

    // ============= Edge Cases and Error Handling =============

    #[test]
    fn test_null_value_handling() {
        let params = GenerateParameters {
            max_new_tokens: None,
            temperature: None,
            ..Default::default()
        };

        let req = GenerateRequest {
            text: Some("test".to_string()),
            prompt: None,
            input_ids: None,
            stream: false,
            parameters: Some(params),
            sampling_params: None,
            return_logprob: false,
            ..default_generate_request()
        };

        let pd_req = req.to_pd_request();
        let other = pd_req.other.as_object().unwrap();

        // Should not have parameters field if all fields are None
        assert!(!other.contains_key("parameters"));
    }

    #[test]
    fn test_large_batch_conversion() {
        let large_batch: Vec<String> = (0..1000).map(|i| format!("item_{}", i)).collect();

        let req = GenerateRequest {
            text: None,
            prompt: Some(StringOrArray::Array(large_batch.clone())),
            input_ids: None,
            stream: false,
            parameters: None,
            sampling_params: None,
            return_logprob: false,
            ..default_generate_request()
        };

        let pd_req = req.to_pd_request();

        if let Some(SingleOrBatch::Batch(batch)) = pd_req.text {
            assert_eq!(batch.len(), 1000);
            assert_eq!(batch[0], "item_0");
            assert_eq!(batch[999], "item_999");
        } else {
            panic!("Expected batch text");
        }
    }

    #[test]
    fn test_unicode_string_handling() {
        let unicode_text = "Hello 世界 🌍 नमस्ते мир".to_string();

        let req = GenerateRequest {
            text: Some(unicode_text.clone()),
            ..default_generate_request()
        };

        let pd_req = req.to_pd_request();

        if let Some(SingleOrBatch::Single(text)) = pd_req.text {
            assert_eq!(text, unicode_text);
        } else {
            panic!("Expected single text");
        }
    }

    #[test]
    fn test_deeply_nested_parameters() {
        let mut nested_params = serde_json::Map::new();
        nested_params.insert(
            "nested".to_string(),
            json!({
                "level1": {
                    "level2": {
                        "level3": "value"
                    }
                }
            }),
        );

        let params = GenerateParameters {
            max_new_tokens: Some(100),
            ..Default::default()
        };

        let req = GenerateRequest {
            text: Some("test".to_string()),
            prompt: None,
            input_ids: None,
            stream: false,
            parameters: Some(params),
            sampling_params: None,
            return_logprob: false,
            ..default_generate_request()
        };

        let pd_req = req.to_pd_request();
        let other = pd_req.other.as_object().unwrap();

        // Parameters should be preserved even with nested structures
        assert!(other.contains_key("max_new_tokens"));
    }

    // ============= Bootstrap Field Tests =============

    #[test]
    fn test_bootstrap_fields_none() {
        let req = GenerateRequest {
            text: Some("test".to_string()),
            ..default_generate_request()
        };

        let pd_req = req.to_pd_request();

        assert_eq!(pd_req.bootstrap_host, None);
        assert_eq!(pd_req.bootstrap_port, None);
        assert_eq!(pd_req.bootstrap_room, None);
    }

    // ============= SGLang Extension Field Pass-Through Tests =============

    #[test]
    fn test_chat_completion_sglang_extensions_passed_through() {
        let messages = vec![ChatMessage::User {
            role: "user".to_string(),
            content: UserMessageContent::Text("Test".to_string()),
            name: None,
        }];

        let mut session_params = std::collections::HashMap::new();
        session_params.insert(
            "key".to_string(),
            serde_json::Value::String("value".to_string()),
        );

        let req = ChatCompletionRequest {
            messages,
            model: "test-model".to_string(),
            // SGLang Extensions - Priority 1
            top_k: Some(40),
            min_p: Some(0.05),
            min_tokens: Some(10),
            repetition_penalty: Some(1.1),
            regex: Some("test_regex".to_string()),
            ebnf: Some("test_ebnf".to_string()),
            stop_token_ids: Some(vec![1, 2, 3]),
            // SGLang Extensions - Priority 2
            lora_path: Some(LoRAPath::Single(Some("test_lora.bin".to_string()))),
            session_params: Some(session_params.clone()),
            // Boolean extensions (ChatCompletionRequest has these as bool, not Option<bool>)
            no_stop_trim: true,
            ignore_eos: false,
            continue_final_message: true,
            skip_special_tokens: false,
            separate_reasoning: true,
            stream_reasoning: false,
            return_hidden_states: true,
            ..default_chat_completion_request()
        };

        let pd_req = req.to_pd_request();
        let other = pd_req.other.as_object().unwrap();

        // Verify SGLang extensions are passed through
        assert_eq!(other.get("top_k"), Some(&json!(40)));
        assert!((other.get("min_p").unwrap().as_f64().unwrap() - 0.05).abs() < 0.0001);
        assert_eq!(other.get("min_tokens"), Some(&json!(10)));
        assert!((other.get("repetition_penalty").unwrap().as_f64().unwrap() - 1.1).abs() < 0.0001);
        assert_eq!(other.get("regex"), Some(&json!("test_regex")));
        assert_eq!(other.get("ebnf"), Some(&json!("test_ebnf")));
        assert_eq!(other.get("stop_token_ids"), Some(&json!(vec![1, 2, 3])));
        assert_eq!(other.get("lora_path"), Some(&json!("test_lora.bin")));
        assert_eq!(
            other.get("session_params"),
            Some(&serde_json::to_value(&session_params).unwrap())
        );

        // Verify boolean extensions
        assert_eq!(other.get("no_stop_trim"), Some(&json!(true)));
        assert_eq!(other.get("ignore_eos"), Some(&json!(false)));
        assert_eq!(other.get("continue_final_message"), Some(&json!(true)));
        assert_eq!(other.get("skip_special_tokens"), Some(&json!(false)));
        assert_eq!(other.get("separate_reasoning"), Some(&json!(true)));
        assert_eq!(other.get("stream_reasoning"), Some(&json!(false)));
        assert_eq!(other.get("return_hidden_states"), Some(&json!(true)));
    }

    #[test]
    fn test_completion_request_sglang_extensions_passed_through() {
        let mut session_params = std::collections::HashMap::new();
        session_params.insert(
            "key".to_string(),
            serde_json::Value::String("value".to_string()),
        );

        let req = CompletionRequest {
            prompt: StringOrArray::String("Test prompt".to_string()),
            model: "test-model".to_string(),
            // SGLang Extensions - Priority 1
            top_k: Some(40),
            min_p: Some(0.05),
            min_tokens: Some(10),
            repetition_penalty: Some(1.1),
            regex: Some("test_regex".to_string()),
            ebnf: Some("test_ebnf".to_string()),
            stop_token_ids: Some(vec![1, 2, 3]),
            // SGLang Extensions - Priority 2
            lora_path: Some(LoRAPath::Single(Some("test_lora.bin".to_string()))),
            session_params: Some(session_params.clone()),
            // Boolean extensions (CompletionRequest only has these 4 boolean fields)
            no_stop_trim: true,
            ignore_eos: false,
            skip_special_tokens: false,
            return_hidden_states: true,
            ..default_completion_request()
        };

        let pd_req = req.to_pd_request();
        let other = pd_req.other.as_object().unwrap();

        // Verify SGLang extensions are passed through
        assert_eq!(other.get("top_k"), Some(&json!(40)));
        assert!((other.get("min_p").unwrap().as_f64().unwrap() - 0.05).abs() < 0.0001);
        assert_eq!(other.get("min_tokens"), Some(&json!(10)));
        assert!((other.get("repetition_penalty").unwrap().as_f64().unwrap() - 1.1).abs() < 0.0001);
        assert_eq!(other.get("regex"), Some(&json!("test_regex")));
        assert_eq!(other.get("ebnf"), Some(&json!("test_ebnf")));
        assert_eq!(other.get("stop_token_ids"), Some(&json!(vec![1, 2, 3])));
        assert_eq!(other.get("lora_path"), Some(&json!("test_lora.bin")));
        assert_eq!(
            other.get("session_params"),
            Some(&serde_json::to_value(&session_params).unwrap())
        );

        // Verify boolean extensions (only the ones CompletionRequest has)
        assert_eq!(other.get("no_stop_trim"), Some(&json!(true)));
        assert_eq!(other.get("ignore_eos"), Some(&json!(false)));
        assert_eq!(other.get("skip_special_tokens"), Some(&json!(false)));
        assert_eq!(other.get("return_hidden_states"), Some(&json!(true)));
    }

    #[test]
    fn test_sglang_extensions_none_values_not_passed_through() {
        let messages = vec![ChatMessage::User {
            role: "user".to_string(),
            content: UserMessageContent::Text("Test".to_string()),
            name: None,
        }];

        let req = ChatCompletionRequest {
            messages,
            model: "test-model".to_string(),
            // All SGLang extensions as None/default - Optional fields won't appear, bools will use defaults
            top_k: None,
            min_p: None,
            min_tokens: None,
            repetition_penalty: None,
            regex: None,
            ebnf: None,
            stop_token_ids: None,
            lora_path: None,
            session_params: None,
            // Boolean fields use defaults (false for most, true for some with default_true)
            no_stop_trim: false,
            ignore_eos: false,
            continue_final_message: false,
            skip_special_tokens: true, // This has default_true
            separate_reasoning: true,  // This has default_true
            stream_reasoning: true,    // This has default_true
            return_hidden_states: false,
            ..default_chat_completion_request()
        };

        let pd_req = req.to_pd_request();
        let other = pd_req.other.as_object().unwrap();

        // Verify None values are not included
        assert!(!other.contains_key("top_k"));
        assert!(!other.contains_key("min_p"));
        assert!(!other.contains_key("min_tokens"));
        assert!(!other.contains_key("repetition_penalty"));
        assert!(!other.contains_key("regex"));
        assert!(!other.contains_key("ebnf"));
        assert!(!other.contains_key("stop_token_ids"));
        assert!(!other.contains_key("lora_path"));
        assert!(!other.contains_key("session_params"));

        // Boolean fields are always present with their values (can't be None)
        assert_eq!(other.get("no_stop_trim"), Some(&json!(false)));
        assert_eq!(other.get("ignore_eos"), Some(&json!(false)));
        assert_eq!(other.get("continue_final_message"), Some(&json!(false)));
        assert_eq!(other.get("skip_special_tokens"), Some(&json!(true))); // default_true
        assert_eq!(other.get("separate_reasoning"), Some(&json!(true))); // default_true
        assert_eq!(other.get("stream_reasoning"), Some(&json!(true))); // default_true
        assert_eq!(other.get("return_hidden_states"), Some(&json!(false)));
    }
}
